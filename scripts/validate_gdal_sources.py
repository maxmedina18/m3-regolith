"""Read-only original-source audit and sampled GDAL comparison; preserves prior outputs."""
import csv, datetime, importlib.util, json, os, platform, shlex, shutil, sys
from pathlib import Path
import numpy as np
from osgeo import gdal
from compare_original_extractors import COORDS, sha, metrics, require_snapshot_wavelength_unit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gdal_validation'
HIST = ROOT / 'results/extractor_comparison'

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def assess(a,b,nodata):
    va,vb=np.isfinite(a),np.isfinite(b)
    if nodata is not None:
        va &= a != nodata; vb &= b != nodata
    return {'raw':metrics(a,b),'valid_only':metrics(a,b,va&vb),
            'valid_paired_values':int((va&vb).sum()),'invalid_paired_values':int((~va&~vb).sum()),
            'invalid_mask_disagreements':int((va!=vb).sum()),
            'fully_valid_pixels':int((va&vb).all(axis=1).sum()),
            'partially_invalid_pixels':int(((va&vb).any(axis=1)&~(va&vb).all(axis=1)).sum()),
            'fully_invalid_pixels':int((~(va&vb).any(axis=1)).sum()),
            'per_band':[metrics(a[:,i],b[:,i]) for i in range(a.shape[1])],
            'per_band_valid_only':[metrics(a[:,i],b[:,i],va[:,i]&vb[:,i]) for i in range(a.shape[1])],
            'valid_mask':va&vb}

def main():
    config=json.loads((ROOT/'configs/snapshot.local.json').read_text())
    require_snapshot_wavelength_unit(config)
    gdal.UseExceptions()
    if (OUT/'provenance.json').exists():
        old=json.loads((OUT/'provenance.json').read_text())
        if old.get('status')!='blocked_missing_gdal':
            raise RuntimeError('Refusing to overwrite non-placeholder validation')
        archive=OUT/'previous_blocked_attempt';archive.mkdir(exist_ok=False)
        for p in list(OUT.iterdir()):
            if p.is_file(): shutil.move(str(p),archive/p.name)
    OUT.mkdir(exist_ok=True)
    coords=json.loads((OUT/'previous_blocked_attempt/coordinates.json').read_text())['coordinates_row_column']
    assert len(coords)==len({tuple(x) for x in coords})==111
    assert coords[:11]==[list(x) for x in COORDS]
    save('coordinates.json',{'coordinates_row_column_0based':coords,'bands_1based':list(range(1,86)),
                            'plan':'Unchanged prior 11 + 100 deterministic grid points, cell offset 0.37'})
    hist=json.loads((HIST/'provenance.json').read_text())
    hist_hash={Path(x['path']).name:x['sha256'] for x in hist['input_files']}
    sys.path.insert(0,str(HIST/'original_source'))
    from lunar_m3.data_loading.pds3_envi import read_envi_header,read_envi_image
    advisor_path=ROOT/'external/advisor/M3_stage1_extractor_v2.py'
    assert sha(advisor_path)==hist['source_files'][1]['sha256']
    spec=importlib.util.spec_from_file_location('advisor_audit',advisor_path)
    advisor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=advisor;spec.loader.exec_module(advisor)
    inputs=[]; products={}; arrays={}; validations={}; source_hashes={}
    for role in ('rfl','loc','obs'):
        hdr=Path(config[role+'_hdr']);img=hdr.with_suffix('.IMG')
        for p in (img,hdr):
            h=sha(p);source_hashes[str(p)]=h
            inputs.append({'path':str(p),'name':p.name,'bytes':p.stat().st_size,'sha256':h,
                           'historical_sha256_match':h==hist_hash.get(p.name)})
        original_header=read_envi_header(hdr)
        original=read_envi_image(img,original_header)
        expected_bands={'rfl':85,'loc':3,'obs':10}[role]
        assert original.shape==(11935,304,expected_bands)
        ac=advisor.EnviCube(img,hdr,role.upper())
        ds=gdal.OpenEx(str(img),gdal.OF_RASTER|gdal.OF_READONLY,allowed_drivers=['ENVI'])
        assert ds is not None and ds.GetDriver().ShortName=='ENVI'
        shape=[ds.RasterYSize,ds.RasterXSize,ds.RasterCount]
        assert shape==list(original.shape)
        a=np.array([original[r,c,:] for r,c in coords]); b=np.empty(a.shape,dtype=ac.dtype)
        bandmeta=[]
        for i in range(ds.RasterCount):
            band=ds.GetRasterBand(i+1)
            for k,(r,c) in enumerate(coords):
                b[k,i]=band.ReadAsArray(xoff=c,yoff=r,win_xsize=1,win_ysize=1)[0,0]
            bandmeta.append({'band_1based':i+1,'type':gdal.GetDataTypeName(band.DataType),
                             'nodata':band.GetNoDataValue(),'metadata':band.GetMetadata()})
        meta=ds.GetMetadata('ENVI')
        nodata=ac.nodata
        assert all(x['nodata']==nodata for x in bandmeta)
        checks={'dimensions_match':True,'interleave_match':meta.get('interleave')==original_header.interleave,
                'byte_order_match':int(meta['byte_order'])==original_header.byte_order,
                'header_offset_match':int(meta.get('header_offset','0'))==original_header.header_offset,
                'offset_note':'Absent offset uses zero default; metadata agreement, not a separate offset measurement.',
                'nodata_match':True,'dtype_match_source':all(x['type']==('Float64' if role=='loc' else 'Float32') for x in bandmeta),
                'indexing':'NumPy [row,column,band-1]; GDAL band=band, xoff=column,yoff=row, 1x1 window; no transformations'}
        if role=='rfl':
            native=np.array([float(x['metadata']['wavelength']) for x in bandmeta])
            checks['wavelength_um']=metrics(original_header.wavelengths_um,native/1000)
            gd_bbl=[int(x.strip()) for x in meta['bbl'].strip('{} ').split(',')]
            checks['bbl_match_source']=gd_bbl==advisor.envi_list(ac.header,'bbl',int)
            checks['original_metadata_drops_bbl_nodata']=True
            checks['wavelength_units']='Header has no explicit unit; nm-to-um per protocol. Not an independent unit validation.'
            validations['vrt']=advisor.validate_vrt(Path(config['rfl_vrt']),ac)
        products[role]={'source_header':ac.header,'gdal_envi_metadata':meta,'gdal_file_list':ds.GetFileList(),
                        'gdal_bands':bandmeta,'original_shape':list(original.shape),'gdal_shape':shape,
                        'original_output_dtype':str(a.dtype),'gdal_output_dtype':str(b.dtype),'checks':checks}
        result=assess(a,b,nodata);valid=result.pop('valid_mask')
        if role=='loc':
            result['original_equals_gdal_cast_float32']=bool(np.array_equal(a,b.astype(np.float32)))
            result.pop('raw');result.pop('valid_only') # no mixed-unit aggregates
            result['units_by_plane']=['degrees_east','degrees','metres']
        validations[role]=result;arrays[role]=(a,b,valid)
        ds=None;del original
    vrt=Path(config['rfl_vrt']);h=sha(vrt)
    inputs.append({'path':str(vrt),'name':vrt.name,'bytes':vrt.stat().st_size,'sha256':h,'historical_sha256_match':h==hist_hash[vrt.name]})
    colocated=len({str(Path(x['path']).parent) for x in inputs})==1
    source={'required_files':inputs,'all_seven_present':len(inputs)==7,'all_current_hashes_match_historical':all(x['historical_sha256_match'] for x in inputs),
            'single_input_directory':colocated,'products':products,'vrt_validation':validations.pop('vrt'),
            'warning':'Files remain split across V01 and V02 directories. Strict same-directory requirement is NOT met; no copies, symlinks, substitutions or downloads performed.',
            'identity_limit':'Historical manifest hashes bind recorded inputs; worker logs do not independently hash every open/read. Consumption reconstructed from unchanged worker source and recorded arguments.'}
    save('source_manifest.json',source);save('validation_metrics.json',validations)
    with (OUT/'validation_spectra.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_0based','column_0based','band_1based','original','gdal','difference','valid_pair'])
        a,b,valid=arrays['rfl']
        for k,(r,c) in enumerate(coords):
            for i in range(85):w.writerow([r,c,i+1,float(a[k,i]),float(b[k,i]),float(a[k,i])-float(b[k,i]),int(valid[k,i])])
    with (OUT/'ancillary_validation.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['role','row_0based','column_0based','plane_1based','original','gdal','difference','valid_pair'])
        for role in ('loc','obs'):
            a,b,valid=arrays[role]
            for k,(r,c) in enumerate(coords):
                for i in range(a.shape[1]):w.writerow([role,r,c,i+1,float(a[k,i]),float(b[k,i]),float(a[k,i])-float(b[k,i]),int(valid[k,i])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(13,12),sharex=True)
    a,b,valid=arrays['rfl'];wavelengths=np.array([float(x['metadata']['wavelength']) for x in products['rfl']['gdal_bands']])/1000
    for k,ax in enumerate(axes.flat):
        if k==11:ax.axis('off');continue
        mask=valid[k]&np.array(gd_bbl,dtype=bool)
        ax.plot(wavelengths,np.where(mask,a[k],np.nan),label='Original f176272')
        ax.plot(wavelengths,np.where(mask,b[k],np.nan),'--',label='GDAL 3.9.3')
        ax.set_title(str(coords[k]));ax.set_xlabel('Wavelength (µm)');ax.set_ylabel('Reflectance')
    axes.flat[0].legend();fig.suptitle('Original 11 pixels; all 111 tested numerically. Invalid/BBL-bad values hidden only in plot.')
    fig.tight_layout();fig.savefig(OUT/'spectra_overlay.png',dpi=150);plt.close(fig)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve())]
    (OUT/'validation_commands.sh').write_text('#!/bin/sh\n# Exact completed command; refuses to overwrite completed evidence.\n'+shlex.join(cmd)+'\n')
    save('provenance.json',{'status':'completed_sampled_validation_with_source_layout_warning','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'argv':cmd,'cwd':str(ROOT),'python':sys.version,'numpy':np.__version__,'gdal':gdal.VersionInfo('--version'),
         'matplotlib':matplotlib.__version__,'platform':platform.platform(),'ENVI_driver':True,
         'source_manifest':'source_manifest.json','script_sha256':sha(Path(__file__)),
         'advisor_sha256_before_after':sha(advisor_path),'original_commit':hist['baseline_commit'],
         'historical_comparison_script_hash_matches':sha(ROOT/'scripts/compare_original_extractors.py')==hist['source_files'][0]['sha256'],
         'frozen_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (HIST/'original_source').rglob('*.py')},
         'input_source_files_unmodified':'Read-only opens; no source writes performed',
         'reproduction':'Use recorded interpreter/command and config. Preserve current result directory elsewhere before rerunning; script refuses completed-output overwrite.'})
    r=validations['rfl'];loc=validations['loc']
    report=f'''# GDAL validation completed with source-layout caveat

GDAL {gdal.VersionInfo('--version')}; Python {platform.python_version()}; NumPy {np.__version__}.
Original frozen reader versus GDAL ENVI direct IMG reads, 111 unchanged planned
coordinates × 85 bands = 9435 raw comparisons. No preprocessing.

RFL: {json.dumps(r['raw'])}
Valid-only: {json.dumps(r['valid_only'])}
Valid pairs: {r['valid_paired_values']}; both invalid: {r['invalid_paired_values']};
mask disagreements: {r['invalid_mask_disagreements']}.
Fully valid spectra: {r['fully_valid_pixels']}; partially invalid: {r['partially_invalid_pixels']}; fully invalid: {r['fully_invalid_pixels']}.
OBS: {json.dumps(validations['obs']['raw'])}
LOC original equals GDAL cast to Float32: {loc['original_equals_gdal_cast_float32']}.
LOC per-plane errors (longitude degrees, latitude degrees, radius metres):
{json.dumps(loc['per_band'],indent=2)}

All seven current file hashes match historical provenance. However, no single
input directory contains all seven: V01 and V02 remain separate. This run
validates sampled decoding of those exact recorded bytes, NOT strict compliance
with the same-directory instruction. Historical source usage is supported by
manifest hashes and code/command inspection, not independent per-worker I/O traces.
Band order, raw dimensions, types, layout, NoData/BBL and wavelength metadata
checks are recorded in source_manifest.json. The original reader discards NoData
and BBL metadata and converts LOC Float64 to Float32. Invalid pairs are retained
in CSV and counted separately; matching -999 values are not valid reflectance.

The earlier missing-GDAL attempt is preserved byte-for-byte in
previous_blocked_attempt/. The plot displays the original 11 spectra; CSV and
metrics cover all 111. Input hashes and exact command are in source_manifest.json,
provenance.json and validation_commands.sh. No source/advisor changes or Git push.

Remaining: seven-file colocation, V01/V02 spatial alignment (RFL label refers to
V03 ancillary products), NASAView numerical checks, elevation/whole-scene export
validation. This does NOT establish NASAView agreement, whole-scene correctness,
independent spatial alignment, NASA reflectance calibration accuracy,
mineralogical interpretation or FeO accuracy. Shared header errors remain possible.
'''
    (OUT/'validation_summary.md').write_text(report)
    audit='''# Stage-1 source provenance audit

## Current source-file evidence

All seven required filenames and versions exist, with 11935×304 spatial dimensions;
RFL/LOC/OBS have 85/3/10 bands. Current SHA-256 hashes match the historical manifest
for all seven files. Absolute paths, sizes, hashes, ENVI metadata and validated VRT
mapping are in ../gdal_validation/source_manifest.json. Files are split between
V01 and V02 directories: the literal one-input-directory requirement is unmet.
No source was moved, renamed, regenerated, downloaded or substituted.

## Historical evidence and consumption inferred from code

The historical provenance records two successful independent worker processes
using the same input_config, with original f176272 source and hash-verified advisor
source. Original worker used load_m3_cube/read_envi_header/read_envi_image for RFL,
and read_envi_header/read_envi_image for LOC and OBS. Numerical inputs were the
three IMG files; interpretation used their matching HDRs. It did not use the VRT.
Advisor worker instantiated EnviCube on the same three IMG/HDR pairs and used
read_band_rows for values; validate_vrt checked the RFL VRT without reading pixels
through it. The historical invocation used advisor APIs, not its full CLI export.
Neither worker consumed cached NPY/CSV/TSV arrays as source data. Generated NPY
arrays were subsequently read by the comparison stage. These statements follow
from the preserved script, arguments and logs; no independent I/O trace exists.

The parent comparison process hashed all seven files plus detached L1B/L2 labels.
Labels supplied provenance only, not numerical extraction. Header text references
calibration/support filenames; these readers do not recursively open those files.
The VRT names its RFL source but the advisor only validates the mapping.
No evidence of alternate snapshot, product substitution or cached extraction input
was found. Recorded hashes plus matching current files support historical byte
identity; they do not prove each worker independently checked those hashes while
reading. Historical source absolute-path prefixes were publication-sanitized, with
filenames and hashes preserved. Current local configuration supplies the paths.

## Historical numerical evidence

The archived run compared 935 RFL, 33 LOC and 110 OBS values. RFL and OBS and all
85 converted wavelengths matched exactly; LOC differed because the original
reader downcast Float64 to Float32. Matching outputs alone is not source-identity
proof. The new GDAL run extends this to 111 coordinates, preserving the old plan.
Prior comparisons, hashes and logs are unchanged; this audit is an added document.

## Unverified scientific assumptions

Source identity against an external NASA archive checksum was not independently
established. No download or substitution was made. V01 RFL references V03 ancillary
products whereas the requested files are V02. Dimensions match but spatial
alignment remains unresolved. Seven-file colocation and full Stage-1/NASAView
validation are not complete. Wavelength units are implicit and the nm convention
comes from protocol rather than an explicit header field.
'''
    (HIST/'source_file_audit.md').write_text(audit)
    print(json.dumps({'rfl':r['raw'],'valid':r['valid_paired_values'],'invalid':r['invalid_paired_values'],'obs':validations['obs']['raw'],'loc_per_plane_max':[x['max_absolute_error'] for x in loc['per_band']],'seven_hashes_match':source['all_current_hashes_match_historical'],'colocated':colocated}))

if __name__=='__main__': main()
