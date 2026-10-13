"""Tiny deterministic ENVI fixtures. Values are synthetic, never NASA observations."""
from pathlib import Path
import struct
import numpy as np


def make_fixture(directory, interleave='bil', byte_order=0, rows=3, cols=4, bands=85):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    prefix='SYNTHETIC'; config={'prefix':prefix,'expect_lines':rows,'expect_samples':cols,'expect_rfl_bands':bands,'wavelength_unit':'nm'}
    expected={}
    wave=np.linspace(500,2500,bands)
    for role,n,kind,code,version in [('RFL',bands,'f',4,'V01'),('LOC',3,'d',5,'V02'),('OBS',10,'f',4,'V02')]:
        stem=f'{prefix}_{version}_{role}'
        values=np.empty((rows,cols,n),dtype='float64' if code==5 else 'float32')
        for r in range(rows):
            for c in range(cols):
                for b in range(n):
                    v=(r*cols+c)*0.001+(b+1)*0.01
                    if role=='LOC':v=1737400.01234567+r*.123456789+c*.00012345 if b==2 else 10.123456789+r*.001+c*.00001+b
                    if role=='OBS':v=r*10+c+b*.125
                    values[r,c,b]=v
        values[0,0,0]=-999
        if role=='LOC':values[-1,-1,2]=-999
        if role=='RFL':values[-1,0,-1]=np.nan
        expected[role]=values
        hdr=directory/(stem+'.HDR');img=directory/(stem+'.IMG')
        text=f'ENVI\nsamples = {cols}\nlines = {rows}\nbands = {n}\nheader offset = 16\ndata type = {code}\ninterleave = {interleave}\nbyte order = {byte_order}\ndata ignore value = -999\n'
        if role=='RFL':text+='wavelength units = Nanometers\nwavelength = {'+','.join(map(str,wave))+'}\nbbl = {'+','.join(['0']+['1']*(bands-1))+'}\n'
        hdr.write_text(text)
        indices={'bil':((r,c,b) for r in range(rows) for b in range(n) for c in range(cols)),
                 'bsq':((r,c,b) for b in range(n) for r in range(rows) for c in range(cols)),
                 'bip':((r,c,b) for r in range(rows) for c in range(cols) for b in range(n))}[interleave]
        with img.open('wb') as f:
            f.write(b'X'*16)
            for r,c,b in indices:f.write(struct.pack(('>' if byte_order else '<')+kind,float(values[r,c,b])))
        config[role.lower()+'_hdr']=str(hdr.resolve())
    vrt=directory/f'{prefix}_V01_RFL.IMG.vrt'
    parts=[f'<VRTDataset rasterXSize="{cols}" rasterYSize="{rows}">']
    for b,w in enumerate(wave,1):
        parts.append(f'<VRTRasterBand dataType="Float32" band="{b}"><Description>{w} nm</Description><SimpleSource><SourceFilename relativeToVRT="1">{prefix}_V01_RFL.IMG</SourceFilename><SourceBand>{b}</SourceBand><SrcRect xOff="0" yOff="0" xSize="{cols}" ySize="{rows}"/><DstRect xOff="0" yOff="0" xSize="{cols}" ySize="{rows}"/></SimpleSource></VRTRasterBand>')
    vrt.write_text(''.join(parts)+'</VRTDataset>');config['rfl_vrt']=str(vrt.resolve())
    return config,expected
