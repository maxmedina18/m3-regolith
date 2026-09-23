"""M3 prototype: read -> diagnose -> original-constant FeO experiments -> figures.

NOT a validated abundance product. Does not fit a revised calibration, infer Al,
apply extra radiometric corrections, or create geographic coordinates.
Run from project root: python3 src/inspect_cube.py --data-dir '/path/to/cube set'
All figures use native zero-based row/column positions. Raw inputs are read only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Patch

NOTICE = 'PROTOTYPE • NOT VALIDATED • DO NOT PUBLISH AS COMPOSITION'
BRANCHES = ('signed', 'wrap_after', 'wrap_before')
BRANCH_LABELS = {'signed':'Signed angle', 'wrap_after':'Wrap after minus', 'wrap_before':'Wrap before minus'}
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#827000']

def field(text, key, default=None):
    m = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*([^\r\n]+)', text, re.M)
    if m: return m.group(1).strip().strip('"')
    if default is not None: return default
    raise ValueError(f'Missing metadata field: {key}')

def vector(text, key):
    m = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*\{([^}]+)\}',text,re.M|re.S)
    if not m: raise ValueError(f'Missing vector: {key}')
    return np.array([float(t) for t in m.group(1).split(',') if t.strip()])

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_cube(base, product):
    """Explicitly supports little-endian float32 BIL; rejects other layouts."""
    hdr_path=base/f'{product}_RFL.HDR'; lbl_path=base/f'{product}_L2.LBL'
    hdr=hdr_path.read_text(); lbl=lbl_path.read_text()
    nline,nband,nsample=[int(field(hdr,k)) for k in ('lines','bands','samples')]
    require(min(nline,nband,nsample)>0,'Dimensions must be positive')
    require(field(hdr,'interleave').lower()=='bil','Only BIL is implemented')
    require(int(field(hdr,'data type'))==4 and int(field(hdr,'byte order'))==0,'Only little-endian float32 is implemented')
    m=re.search(r'OBJECT\s*=\s*RFL_IMAGE\b(.*?)END_OBJECT\s*=\s*RFL_IMAGE',lbl,re.S)
    require(m is not None,'No RFL_IMAGE object')
    obj=m.group(1)
    require([int(field(obj,k)) for k in ('LINES','BANDS','LINE_SAMPLES')]==[nline,nband,nsample],'Label/header dimension disagreement')
    require(field(obj,'SAMPLE_TYPE')=='PC_REAL' and int(field(obj,'SAMPLE_BITS'))==32,'Label numeric type disagreement')
    require(field(obj,'BAND_STORAGE_TYPE')=='LINE_INTERLEAVED','Label interleave disagreement')
    offset=int(field(hdr,'header offset','0'));require(offset>=0,'Negative offset')
    invalid=float(field(hdr,'data ignore value'))
    require(invalid==float(field(obj,'INVALID_CONSTANT')),'Missing-data value disagreement')
    wave=vector(hdr,'wavelength'); bbl=vector(hdr,'bbl')
    require(len(wave)==len(bbl)==nband,'Wavelength/bbl length disagreement')
    require(np.all(np.isfinite(wave)) and np.all(np.diff(wave)>0),'Wavelengths must be finite and increasing')
    require(np.all(np.isin(bbl,[0,1])),'Unexpected bad-band flags')
    # Wavelength numeric units confirmed against VRT descriptions, not guessed.
    vrt=ET.parse(base/f'{product}_RFL.IMG.vrt').getroot()
    descriptions=[b.findtext('Description','').strip() for b in vrt.findall('VRTRasterBand')]
    require(len(descriptions)==nband and all(d.endswith(' nm') for d in descriptions),'VRT must define all wavelengths in nm')
    vw=np.array([float(d.split()[0]) for d in descriptions])
    wave_diff=float(np.max(abs(wave-vw)))
    require(wave_diff<=0.01,'Header/VRT wavelength difference exceeds 0.01 nm rounding tolerance')
    image=base/f'{product}_RFL.IMG'; expected=offset+nline*nband*nsample*4
    require(image.stat().st_size==expected,'Image size does not match metadata')
    disk=np.memmap(image,mode='r',dtype='<f4',offset=offset,shape=(nline,nband,nsample))
    cube=disk.transpose(0,2,1)
    checks=[(0,0,0),(nline-1,nsample-1,nband-1),(nline//2,nsample//2,nband//2),
            (nline//3,nsample//4,nband//3),(2*nline//3,3*nsample//4,2*nband//3)]
    with image.open('rb') as f:
        for row,col,k in checks:
            f.seek(offset+4*((row*nband+k)*nsample+col)); v=struct.unpack('<f',f.read(4))[0]
            require(v==float(cube[row,col,k]) or (np.isnan(v) and np.isnan(cube[row,col,k])),'Binary spot check failed')
    meta={'shape':list(cube.shape),'dtype':'little-endian float32','interleave':'BIL','header_offset':offset,
          'header_offset_explicit':bool(re.search(r'^\s*header offset\s*=',hdr,re.M)),
          'expected_bytes':expected,'binary_spot_checks':checks,'independent_reader_validated':False,
          'header_vrt_max_difference_nm':wave_diff,'bad_bands_1based':(np.flatnonzero(bbl==0)+1).tolist(),
          'thermal_correction_flag':field(lbl,'CH1:THERMAL_CORRECTION_FLAG'),
          'photometry_correction_file':field(lbl,'CH1:PHOTOMETRY_CORR_FILE_NAME'),
          'ground_truth_correction_file':field(lbl,'CH1:GROUND_TRUTH_CORR_FILE_NAME')}
    for role,key in [('location','CH1:PIXEL_LOCATION_FILE_NAME'),('geometry','CH1:OBS_GEOMETRY_FILE_NAME')]:
        name=field(lbl,key);meta[role]={'referenced_name':name,'present_in_data_dir':(base/name).is_file(),'loaded':False}
    return cube,wave,bbl.astype(bool),invalid,meta

def feo_from_reflectance(r750,r950,config,base_valid=None):
    """Three explicit angle experiments; no clipping and no fitted constants.

    Positive finite reflectances are required by our prototype input policy.
    Values above one are retained. Excludes the exact translated origin.
    Flag counts are not a calibration-domain validation.
    """
    a,b=np.broadcast_arrays(np.asarray(r750,dtype=float),np.asarray(r950,dtype=float))
    valid=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
    if base_valid is not None: valid &= np.asarray(base_valid,dtype=bool)
    ratio=np.full(a.shape,np.nan)
    np.divide(b,a,out=ratio,where=valid)
    x=a-config['x_origin'];y=ratio-config['y_origin']
    radius=np.hypot(x,y)
    valid &= np.isfinite(ratio)&(radius>0)
    ratio=np.where(valid,ratio,np.nan)
    phi=np.where(valid,np.arctan2(y,x),np.nan)
    theta={'signed':-phi,'wrap_after':np.mod(-phi,2*np.pi),'wrap_before':-np.mod(phi,2*np.pi)}
    feo={name:config['A']*angle+config['B'] for name,angle in theta.items()}
    return {'ratio':ratio,'theta':theta,'feo':feo,'valid':valid,'origin_distance':np.where(valid,radius,np.nan)}

def stats(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    if not v.size: return {'finite_count':0}
    return {'finite_count':int(v.size),'min':float(v.min()),'p02':float(np.percentile(v,2)),
            'median':float(np.median(v)),'p98':float(np.percentile(v,98)),'max':float(v.max()),
            'below_zero':int((v<0).sum()),'zero_to_100':int(((v>=0)&(v<=100)).sum()),
            'above_100':int((v>100).sum())}

def clean_json(x):
    if isinstance(x,dict):return {k:clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,np.ndarray):return clean_json(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x

def save_json(path,obj):path.write_text(json.dumps(clean_json(obj),indent=2,allow_nan=False)+'\n')

def footer(fig, extra='Native pixels only • no latitude/longitude • no calibrated uncertainty'):
    fig.text(.5,.015,NOTICE+'\n'+extra,ha='center',va='bottom',fontsize=9,color='#8F261F')

def image_panel(fig,ax,data,title,cmap='viridis',vmin=None,vmax=None,label='',norm=None):
    cm=plt.get_cmap(cmap).copy() if isinstance(cmap,str) else cmap
    cm.set_bad('#D9D9D9')
    im=ax.imshow(data,cmap=cm,origin='upper',interpolation='nearest',vmin=vmin,vmax=vmax,norm=norm)
    ax.set(title=title,xlabel='Column',ylabel='Row')
    fig.colorbar(im,ax=ax,orientation='horizontal',pad=.09,fraction=.035,label=label)
    return im

def make_figures(out,cube,wave,good,invalid,base,result,sensitivity,sample_rows,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    r750,r950=base;F=result['feo']['signed'];nline,nsample=F.shape
    # 1. Four narrow swaths preserve the native row/column aspect ratio.
    fig,axs=plt.subplots(1,4,figsize=(12,10.5))
    fig.subplots_adjust(top=.88,bottom=.14,left=.06,right=.98,wspace=.52)
    fig.suptitle('From reflectance to a provisional FeO calculation',fontsize=19,y=.97)
    fig.text(.5,.925,'Same pixels, four views • original published constants • signed-angle experiment',ha='center',fontsize=11)
    lo,hi=np.nanpercentile(r750,[2,98])
    image_panel(fig,axs[0],r750,'Measured reflectance\n750.44 nm',cmap='gray',vmin=lo,vmax=hi,label='Radiance factor · 2–98% stretch')
    v=result['ratio'];lo,hi=np.nanpercentile(v,[2,98])
    image_panel(fig,axs[1],v,'Spectral contrast\nR950 / R750',vmin=lo,vmax=hi,label='Ratio · 2–98% display stretch')
    lim=max(abs(np.nanmin(F)),abs(np.nanmax(F)))
    image_panel(fig,axs[2],F,'Raw equation output\nNegative values retained',cmap='RdBu_r',vmin=-lim,vmax=lim,label='Provisional FeO (wt%) · full range')
    category=np.zeros(F.shape,dtype=int);category[np.isfinite(F)&(F<0)]=1
    category[np.isfinite(F)&(F>=0)&(F<=100)]=2;category[np.isfinite(F)&(F>100)]=3
    cm=ListedColormap(['#D9D9D9','#B53C36','#178B87','#8F5FA8'])
    axs[3].imshow(category,cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),origin='upper',interpolation='nearest')
    axs[3].set(title='Arithmetic screening\nNot a validity map',xlabel='Column',ylabel='Row')
    pos=axs[2].get_position(); old=axs[3].get_position()
    axs[3].set_position([old.x0,pos.y0,old.width,pos.height])
    pos=axs[2].get_position(); old=axs[3].get_position()
    axs[3].set_position([old.x0,pos.y0,old.width,pos.height])
    axs[3].legend(handles=[Patch(color=cm.colors[k],label=t) for k,t in enumerate(['Excluded input','Below 0 wt%','0–100: unvalidated','Above 100 wt%'])],loc='upper center',bbox_to_anchor=(.5,-.04),frameon=False,fontsize=8)
    for i,s in enumerate(sample_rows[:5]):axs[0].scatter(s['column'],s['row'],s=22,color=COLORS[i],edgecolor='white',lw=.6)
    footer(fig);fig.savefig(out/'01_provisional_feo.png',dpi=190);plt.close(fig)
    # 2. Full output distributions reveal the effect of branch choice.
    fig,axs=plt.subplots(1,2,figsize=(12,5.7));fig.subplots_adjust(top=.80,bottom=.24,wspace=.32)
    fig.suptitle('The angle convention changes the answer',fontsize=19)
    fig.text(.5,.89,'Same measurements and constants; only the angle operation changes',ha='center',fontsize=11)
    vals=[result['feo'][k][result['valid']] for k in BRANCHES]
    edges=np.linspace(min(v.min() for v in vals),max(v.max() for v in vals),170)
    for name,v,c in zip(BRANCHES,vals,COLORS):
        axs[0].hist(v,bins=edges,histtype='step',lw=1.8,label=BRANCH_LABELS[name],color=c)
    axs[0].set(xlabel='Provisional FeO (wt%) — full numerical range',ylabel='Pixel count');axs[0].legend(frameon=False,fontsize=9);axs[0].axvline(0,color='#777',lw=.7);axs[0].axvline(100,color='#777',lw=.7)
    barcols=['#B53C36','#178B87','#8F5FA8'];left=np.zeros(3)
    for key,label,c in zip(['below_zero','zero_to_100','above_100'],['Below 0','0–100 (not validated)','Above 100'],barcols):
        pct=np.array([100*summary['branches'][k][key]/summary['branches'][k]['finite_count'] for k in BRANCHES])
        axs[1].barh(range(3),pct,left=left,color=c,label=label)
        for j,v in enumerate(pct):
            if v>8:axs[1].text(left[j]+v/2,j,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=10)
        left+=pct
    axs[1].set(yticks=range(3),yticklabels=[BRANCH_LABELS[k] for k in BRANCHES],xlabel='Percent of calculable pixels',xlim=(0,100));axs[1].invert_yaxis()
    handles,labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False,fontsize=9)
    footer(fig,'An angle-branch comparison is a sensitivity experiment, not a confidence interval')
    fig.savefig(out/'02_angle_conventions.png',dpi=190);plt.close(fig)
    # 3. Trace measured values into the equation for one actual pixel.
    fig,axs=plt.subplots(1,2,figsize=(12,5.6));fig.subplots_adjust(top=.81,bottom=.24,wspace=.35)
    fig.suptitle('Follow one real pixel into the calculation',fontsize=19)
    p=sample_rows[0]; row,col=p['row'],p['column'];s=np.array(cube[row,col,:],dtype=float)
    s[(~good)|(~np.isfinite(s))|(s==invalid)]=np.nan
    axs[0].plot(wave/1000,s,'.-',color=COLORS[0],lw=1.2,ms=3)
    for k,label in [(summary['channels']['index750'],'750'),(summary['channels']['index950'],'950')]:
        axs[0].axvline(wave[k]/1000,color='#999',lw=.8,ls='--')
    axs[0].set(title=f'P1 · row {row}, column {col}',xlabel='Wavelength (µm)',ylabel='Radiance factor (dimensionless)')
    axs[1].axis('off')
    lines=[f'R750 = {p["R750"]:.6f}',f'R950 = {p["R950"]:.6f}',f'R950 / R750 = {p["ratio"]:.6f}',
           f'Signed θ = {p["theta_signed_rad"]:.6f} rad',f'Signed FeO output = {p["feo_signed"]:.3f} wt%',
           f'Wrap-after output = {p["feo_wrap_after"]:.3f} wt%',f'Wrap-before output = {p["feo_wrap_before"]:.3f} wt%']
    axs[1].text(0,.96,'Measured values → ratio → angle → equation',weight='bold',fontsize=12,va='top')
    axs[1].text(0,.82,'\n\n'.join(lines),fontsize=11,va='top')
    footer(fig,'This calculation demonstrates the implementation; it does not establish the pixel’s true chemistry')
    fig.savefig(out/'03_pixel_walkthrough.png',dpi=190);plt.close(fig)
    # 4. Diagnostic adjacent-channel experiment, common support only.
    fig=plt.figure(figsize=(11,8));gs=fig.add_gridspec(2,2,width_ratios=[1,2.3]);ax=fig.add_subplot(gs[:,0]);axh=fig.add_subplot(gs[0,1]);axc=fig.add_subplot(gs[1,1])
    fig.subplots_adjust(top=.86,bottom=.17,wspace=.42,hspace=.45)
    fig.suptitle('How much does choosing a neighboring channel matter?',fontsize=18,y=.97)
    fig.text(.5,.91,'Nine combinations: nominal 750/950 channels and one neighbor on each side',ha='center',fontsize=11)
    im=ax.imshow(sensitivity,origin='upper',cmap='magma',vmin=0,vmax=np.nanmax(sensitivity),interpolation='nearest')
    ax.set(title='Max − min signed output',xlabel='Column',ylabel='Row')
    fig.colorbar(im,ax=ax,orientation='horizontal',pad=.12,label='FeO percentage points · full range')
    sv=sensitivity[np.isfinite(sensitivity)];axh.hist(sv,bins=100,color='#0072B2');axh.set(xlabel='Nine-case output range (percentage points)',ylabel='Pixel count',title='Common valid inputs in all nine cases')
    axc.plot(np.nanmedian(F,axis=0),color='#0072B2',lw=1);axc.axhline(0,color='#999',lw=.8)
    axc.set(xlabel='Column index',ylabel='Median signed FeO output (wt%)',title='Column structure persists in the equation output')
    footer(fig,'Channel-choice sensitivity only • not measurement error, confidence, or a validated uncertainty estimate')
    fig.savefig(out/'04_channel_sensitivity.png',dpi=190);plt.close(fig)
    # 5. Explicit original vs proposed architecture.
    fig,ax=plt.subplots(figsize=(12,6.2));fig.subplots_adjust(left=.02,right=.98,top=.83,bottom=.08);ax.axis('off')
    fig.suptitle('Proposed research pipeline and agent role',fontsize=19,y=.96)
    fig.text(.5,.89,'Working now: local Python calculation • proposed: archive, geolocation, validation and agent service',ha='center',fontsize=11)
    nodes=[(.16,.69,'PDS discovery + downloads\nPROPOSED · manifest + receipts'),(.50,.69,'Python reader + FeO experiments\nIMPLEMENTED · tests + numerical outputs'),(.84,.69,'LOC / OBS + ground references\nPROPOSED · coordinates + validation'),
           (.16,.23,'Codex now / Agents SDK later\nOperator calls explicit tools'),(.50,.23,'Run records + diagnostic reports\nIMPLEMENTED · hashes + figures'),(.84,.23,'Georeferenced FeO product\nPROPOSED · mask + uncertainty')]
    for x,y,t in nodes:
        active='IMPLEMENTED' in t
        ax.text(x,y,t,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.8',fc='#E8F2F4' if active else '#F3F3F3',ec='#497A86' if active else '#888'))
    for a,b in [((.31,.69),(.35,.69)),((.66,.69),(.70,.69)),((.84,.55),(.84,.38)),((.50,.55),(.50,.38)),((.31,.23),(.35,.23)),((.66,.23),(.70,.23))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#555',lw=1.5))
    ax.text(.5,-.035,'Agent chooses approved tasks; Python computes values; researcher approves scientific method changes.',ha='center',fontsize=11)
    fig.savefig(out/'05_architecture.png',dpi=190);plt.close(fig)

def run(data_dir,output_dir,config_path,product):
    base=Path(data_dir).resolve();out=Path(output_dir).resolve()
    require(out!=base and base not in out.parents,'Output must be outside the raw-data directory')
    require(not out.exists() or not any(out.iterdir()),'Output directory is not empty; choose a new run directory')
    cfg=json.loads(Path(config_path).read_text())
    require(cfg['primary_branch']=='signed' and cfg['angle_unit']=='radians','Unsupported primary branch or unit')
    require(all(np.isfinite(cfg[k]) for k in ['A','B','x_origin','y_origin']),'Nonfinite calibration coefficients')
    cube,wave,good,invalid,meta=read_cube(base,product)
    k750,k950=[int(np.argmin(abs(wave-t))) for t in cfg['target_nm']]
    require(k750!=k950 and good[k750] and good[k950],'Selected channels must be distinct and header-good')
    require(0<k750<len(wave)-1 and 0<k950<len(wave)-1,'Neighbor-channel experiment needs adjacent bands')
    raw750=np.array(cube[:,:,k750],dtype=float);raw950=np.array(cube[:,:,k950],dtype=float)
    initial=np.isfinite(raw750)&np.isfinite(raw950)&(raw750!=invalid)&(raw950!=invalid)
    result=feo_from_reflectance(raw750,raw950,cfg,initial)
    nine=[];cases=[]
    for a in range(k750-1,k750+2):
        for b in range(k950-1,k950+2):
            u=np.array(cube[:,:,a],dtype=float);v=np.array(cube[:,:,b],dtype=float)
            mask=(u!=invalid)&(v!=invalid)&good[a]&good[b]
            z=feo_from_reflectance(u,v,cfg,mask)['feo']['signed'];nine.append(z)
            cases.append({'indices':[a,b],'wavelengths_nm':[float(wave[a]),float(wave[b])]})
    stack=np.stack(nine);common=np.all(np.isfinite(stack),axis=0)
    sens=np.full(raw750.shape,np.nan);sens[common]=np.max(stack[:,common],axis=0)-np.min(stack[:,common],axis=0)
    nline,nsample,_=cube.shape
    locs=[(f'P{i+1}',int(row),nsample//2,'original center-column selection') for i,row in enumerate(np.linspace(int(.1*nline),int(.9*nline),5).astype(int))]
    locs += [(f'X{i+1}',nline//2,int(frac*(nsample-1)),'fixed cross-column diagnostic selection') for i,frac in enumerate([.1,.3,.7,.9])]
    samples=[]
    for name,row,col,rule in locs:
        samples.append({'name':name,'row':row,'column':col,'selection_rule':rule,'R750':raw750[row,col],'R950':raw950[row,col],
                        'ratio':result['ratio'][row,col],'theta_signed_rad':result['theta']['signed'][row,col],
                        **{f'feo_{k}':result['feo'][k][row,col] for k in BRANCHES},
                        'input_eligible':result['valid'][row,col],'nine_case_range_pp':sens[row,col]})
    total=nline*nsample
    summary={'notice':NOTICE,'product':product,'metadata':meta,'calibration':cfg,
      'channels':{'index750':k750,'index950':k950,'actual_nm':[wave[k750],wave[k950]]},
      'total_pixels':total,'calculable_pixels':int(result['valid'].sum()),'excluded_pixels':int((~result['valid']).sum()),
      'input_flags':{'fill_or_nonfinite_either':int((~initial).sum()),'nonpositive_either_nonmissing':int((initial&((raw750<=0)|(raw950<=0))).sum()),
                    'above_one_either_nonmissing':int((initial&((raw750>1)|(raw950>1))).sum()),
                    'note':'above-one values retained; flags may overlap. Positive-input policy is a prototype choice.'},
      'branches':{k:stats(result['feo'][k]) for k in BRANCHES},'two_pi_output_shift_pp':cfg['A']*2*np.pi,
      'sensitivity':{'common_pixels':int(common.sum()),'cases':cases,'range_pp':stats(sens),'meaning':'adjacent-channel choice range, NOT confidence interval'},
      'selected_pixels':samples,'state':{'calculation_executed':True,'chemically_validated':False,'geolocated':False,'revised_fit_reproduced':False,'agent_service_deployed':False},
      'warnings':['Unresolved stripes and jagged spectra','Unresolved published angle convention','Original coefficients only; revised fit absent','No independent reader or reference-chemistry validation','No geographic coordinates loaded','No calibrated uncertainty','0–100 outputs are not certified plausible for this calibration']}
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'summary.json',summary)
    with (out/'selected_pixels.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(clean_json(samples))
    np.savez_compressed(out/'prototype_arrays.npz',r750=raw750,r950=raw950,ratio=result['ratio'],theta_signed=result['theta']['signed'],
                        **{f'feo_{k}':result['feo'][k] for k in BRANCHES},input_eligible=result['valid'],
                        origin_distance=result['origin_distance'],channel_sensitivity_range_pp=sens,
                        wavelength_nm=wave,band_good=good)
    # Pixel-specific comparison retains true numeric values, no averaging/smoothing.
    np.savez_compressed(out/'selected_spectra.npz',wavelength_nm=wave,
                        spectra=np.stack([np.where(good & np.isfinite(cube[s['row'],s['column'],:]) & (cube[s['row'],s['column'],:]!=invalid),cube[s['row'],s['column'],:],np.nan) for s in samples]))
    display750=np.where(initial,raw750,np.nan);display950=np.where(initial,raw950,np.nan)
    make_figures(out,cube,wave,good,invalid,(display750,display950),result,sens,samples,summary)
    inputs=[f for f in sorted(base.iterdir()) if f.is_file() and f.name.startswith(product)]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'python':sys.version,'platform':platform.platform(),
              'numpy':np.__version__,'matplotlib':matplotlib.__version__,'data_dir':str(base),
              'inputs':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f),'publisher_checksum_verified':False} for f in inputs},
              'source_sha256':sha(__file__),'config_sha256':sha(config_path),'calibration':cfg,
              'outputs':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(out.iterdir()) if f.is_file()},
              'orientation':'native rows increase down, columns right; no geotransform','uncertainty':'not calibrated',
              'formulae':{'signed':'-atan2(y,x)','wrap_after':'mod(-atan2(y,x),2*pi)','wrap_before':'-mod(atan2(y,x),2*pi)'},
              'masks':'fill/nonfinite; selected bad bands rejected; nonpositive R750/R950 excluded; exact translated origin excluded; >1 retained',
              'processing':'no smoothing, normalization, destriping, extra correction, refitting, clipping or projection'}
    save_json(out/'run_manifest.json',manifest)
    print(json.dumps(clean_json({k:summary[k] for k in ['calculable_pixels','excluded_pixels','branches','selected_pixels']}),indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'prototype_run')
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'original_calibration.json')
    parser.add_argument('--product',default='M3G20081118T222604_V01')
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,args.config,args.product)
