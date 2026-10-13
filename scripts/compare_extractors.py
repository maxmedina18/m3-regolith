"""Bounded paired comparisons, preserving the supplied advisor file unchanged.

Runs the advisor CLI on a synthetic 99-table fixture first. Real-data comparisons
read fixed source pixels and write a tiny ROI through both implementations.
Neither path is NASAView validation or a full-scene extraction.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np
from lunar_m3.stage1 import ProductSet,export,sha256,write_json,valid_values
from lunar_m3.dev.stage1_fixture import make_fixture

ADVISOR_SHA='d190296d0ab7c7d61754c32857354b5342e4622738077e5637738ab2215ba855'


def compare(config,advisor_path,out):
    advisor_path=Path(advisor_path).resolve();out=Path(out).resolve()
    if sha256(advisor_path)!=ADVISOR_SHA:
        raise ValueError('Advisor checksum differs from received ZIP; review the new source first')
    out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('advisor_baseline',advisor_path)
    advisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(advisor)
    # No source changes, monkeypatching, or imported implementation reuse.
    with tempfile.TemporaryDirectory(prefix='m3-comparison-') as t:
        t=Path(t);cfg,expected=make_fixture(t/'inputs')
        command=[sys.executable,'-B',str(advisor_path),'--input-dir',str(t/'inputs'),'--output-dir',str(t/'advisor'),
                 '--prefix','SYNTHETIC','--expect-lines','3','--expect-samples','4','--expect-rfl-bands','85','--chunk-lines','2']
        run=subprocess.run(command,capture_output=True,text=True)
        (out/'advisor_synthetic.log').write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError('Advisor synthetic CLI failed; see log')
        ours=export(ProductSet(cfg),t/'own',format='tsv')
        comparisons=[]
        for rec in ours['outputs']:
            a=np.loadtxt(t/'own'/rec['file'],ndmin=2);b=np.loadtxt(t/'advisor'/rec['file'],ndmin=2)
            comparisons.append(bool(np.array_equal(a,b,equal_nan=True)))
        synthetic={'tables':len(comparisons),'exact_numeric_table_agreements':sum(comparisons),'command':command}
    p=ProductSet(config)
    refs={role:advisor.EnviCube(img,hdr,role) for role,(hdr,img) in p.paths.items()}
    rows,cols=p.shape
    points=sorted({(r,c) for r in (0,rows//2,rows-1) for c in (0,cols//2,cols-1)} | {(rows//3,cols//4),(2*rows//3,3*cols//4)})
    records=[]
    for role in ('RFL','LOC','OBS'):
        h=p.headers[role];raw=refs[role]
        with p.paths[role][1].open('rb') as f:
            for band in range(1,h.bands+1):
                for row,col in points:
                    own=float(p.plane(role,band,(row,row+1,col,col+1))[0,0]);ref=float(raw.read_band_rows(band,row,row+1)[0,col])
                    k=band-1
                    index={'bil':(row*h.bands+k)*h.samples+col,'bsq':(k*h.lines+row)*h.samples+col,'bip':(row*h.samples+col)*h.bands+k}[h.interleave]
                    size=p.arrays[role].dtype.itemsize
                    f.seek(h.header_offset+index*size)
                    direct=struct.unpack(('<' if h.byte_order==0 else '>')+('d' if size==8 else 'f'),f.read(size))[0]
                    same=(own==ref==direct) or (np.isnan(own) and np.isnan(ref) and np.isnan(direct))
                    finite=np.isfinite(own) and np.isfinite(ref)
                    old=float(np.float32(direct)) # Exact pre-fix output-cast operation on sampled value only.
                    records.append(dict(role=role,band_1based=band,row_0based=row,col_0based=col,
                        own_value=own,advisor_value=ref,direct_byte_value=direct,exact_agreement=same,
                        abs_difference=abs(own-ref) if finite else '',relative_difference=abs(own-ref)/abs(ref) if finite and ref else '',
                        source_valid=bool(valid_values(np.array(ref),h.nodata)),band_good=bool(h.band_good[k]),
                        legacy_float32_cast_value=old,legacy_cast_abs_difference=abs(old-direct) if np.isfinite(direct) else ''))
    with (out/'pixel_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    roi=(rows//2,min(rows,rows//2+3),cols//2,min(cols,cols//2+4))
    own=export(p,out/'own_roi',roi=roi,format='tsv',chunk_lines=2)
    advisor_out=out/'advisor_roi';advisor_out.mkdir()
    table_results=[]
    for rec in own['outputs']:
        role=rec['role'];band=rec['band_1based'];derived=rec['derived'];cube=refs[role]
        def reader(y0,y1):
            block=cube.read_band_rows(band,roi[0]+y0,roi[0]+y1)[:,roi[2]:roi[3]]
            if derived:
                block=np.array(block,dtype=np.float64,copy=True);mask=advisor.valid_mask(block,cube.nodata);block[mask]-=1737400.0
            return block
        advisor.write_tsv_atomic(advisor_out/rec['file'],reader,roi[1]-roi[0],roi[3]-roi[2],
                                 '%.17g' if role=='LOC' else '%.9g',cube.nodata,2)
        a=np.loadtxt(out/'own_roi'/rec['file'],ndmin=2);b=np.loadtxt(advisor_out/rec['file'],ndmin=2)
        table_results.append({'file':rec['file'],'equal_numeric_values':bool(np.array_equal(a,b,equal_nan=True)),
                              'own_sha256':sha256(out/'own_roi'/rec['file']),'advisor_sha256':sha256(advisor_out/rec['file'])})
    summary={'advisor_sha256':sha256(advisor_path),'python':sys.version,'numpy':np.__version__,
             'prescribed_environment':sys.version_info[:3]==(3,12,14) and np.__version__=='2.1.3',
             'synthetic_cli':synthetic,'real_samples':len(records),'exact_three_way_agreements':sum(r['exact_agreement'] for r in records),
             'sampled_nodata_or_nonfinite':sum(not r['source_valid'] for r in records),
             'per_role':{role:{'samples':sum(r['role']==role for r in records),
                'legacy_float32_cast_differences':sum(r['role']==role and isinstance(r['legacy_cast_abs_difference'],float) and r['legacy_cast_abs_difference']>0 for r in records),
                'max_legacy_cast_abs_difference':max([r['legacy_cast_abs_difference'] for r in records if r['role']==role and isinstance(r['legacy_cast_abs_difference'],float)],default=0)} for role in refs},
             'roi':list(roi),'real_export_table_comparisons':table_results,
             'warnings':p.warnings,'nasaview':'not_performed','full_scene_export':'not_performed',
             'legacy_comparison':'Sampled reproduction of the old float32 cast, not an old full-scene run',
             'scope':'Advisor unchanged CLI on synthetic data; unchanged reader/writer functions on bounded real pixels/ROI; byte-offset check shares source metadata assumptions',
             'inputs':p.manifest()['inputs']}
    write_json(out/'comparison_summary.json',summary)
    if summary['exact_three_way_agreements']!=len(records) or not all(x['equal_numeric_values'] for x in table_results) or not all(comparisons):
        raise RuntimeError('Comparison differences found; inspect summary')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--advisor',type=Path,default=Path('external/advisor/M3_stage1_extractor_v2.py'))
    ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    s=compare(json.loads(a.config.read_text()),a.advisor,a.output_dir)
    print(json.dumps({k:s[k] for k in ['synthetic_cli','real_samples','exact_three_way_agreements','sampled_nodata_or_nonfinite','per_role','prescribed_environment']},indent=2))
