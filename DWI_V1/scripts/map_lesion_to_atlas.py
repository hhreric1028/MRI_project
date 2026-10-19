"""Map a reviewed standard-space mask to an FSL maximum-probability atlas."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import nibabel as nib
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mask','atlas','labels','qc','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case-type',default='unspecified')
    a=p.parse_args(); qc=json.loads(a.qc.read_text())
    if qc.get('status')!='PASS': raise SystemExit('STOP: registration QC must be reviewed and PASS before atlas overlap.')
    m=nib.load(a.mask); atlas=nib.load(a.atlas)
    if m.shape!=atlas.shape or not np.allclose(m.affine,atlas.affine,rtol=1e-5,atol=1e-4):
        raise SystemExit('STOP: atlas grid differs; explicitly resample atlas with nearest-neighbor to the reference first.')
    if not np.allclose(m.header.get_zooms()[:3],atlas.header.get_zooms()[:3]): raise SystemExit('STOP: spacing mismatch')
    x=np.asanyarray(m.dataobj); y=np.asanyarray(atlas.dataobj)
    if not np.isfinite(x).all() or not np.isin(x,[0,1]).all(): raise SystemExit('STOP: invalid mask')
    if not np.isfinite(y).all() or not np.equal(y,np.floor(y)).all(): raise SystemExit('STOP: invalid atlas labels')
    if m.header.get_xyzt_units()[0]!='mm': raise SystemExit('STOP: expected mm units')
    xml=ET.parse(a.labels).getroot()
    # FSL probability-volume indices are zero-based; maxprob map IDs are index+1.
    names={int(t.attrib['index'])+1:t.text.strip() for t in xml.findall('./data/label')}
    if not set(np.unique(y)).issubset({0,*names}): raise SystemExit('STOP: atlas/XML labels do not agree')
    ids,counts=np.unique(y[x>0],return_counts=True); total=int(np.count_nonzero(x)); vv=float(np.prod(m.header.get_zooms()[:3],dtype=np.float64))
    regions=[{'label_id':int(i),'region_name':names[int(i)],'overlap_voxels':int(n),'overlap_volume_mm3':int(n)*vv,'overlap_volume_ml':int(n)*vv/1000,'lesion_fraction_percent':100*int(n)/total} for i,n in zip(ids,counts) if i>0]
    regions.sort(key=lambda r:(-r['overlap_voxels'],r['label_id']))
    for rank,r in enumerate(regions,1): r['rank']=rank
    assigned=sum(r['overlap_voxels'] for r in regions)
    result={'case_id':a.mask.name.removesuffix('.nii.gz'),'case_type':a.case_type,'registration':qc,'atlas':{'name':xml.findtext('./header/name'),'xml_schema_version':xml.attrib.get('version'),'release_version':'not specified by upstream XML','path':str(a.atlas.resolve()),'label_count':len(names),'spacing':list(map(float,atlas.header.get_zooms()[:3]))},'lesion':{'standard_space_volume_ml':total*vv/1000,'native_volume_ml':qc.get('native_volume_ml'),'assigned_voxels':assigned,'unassigned_voxels':total-assigned,'atlas_coverage_percent':assigned/total*100 if total else None},'regions':regions,'primary_anatomical_region':regions[0]['region_name'] if regions else None}
    a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
