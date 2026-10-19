"""Collect registration experiment artifacts and fixed-grid diagnostic checks."""
import argparse,json,hashlib
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('experiments','reference','modalities','prediction'): p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); ref=nib.load(a.reference); summary={}
    paths=list(a.modalities.rglob('*.nii.gz')); dwi=nib.load(next(x for x in paths if x.name.endswith('_dwi.nii.gz')))
    modalities={x.name:{'path':str(x.resolve()),'shape':list(nib.load(x).shape),'spacing':list(map(float,nib.load(x).header.get_zooms()[:3])),'affine':nib.load(x).affine.tolist(),'orientation':nib.aff2axcodes(nib.load(x).affine),'same_grid_as_dwi':nib.load(x).shape==dwi.shape and bool(np.allclose(nib.load(x).affine,dwi.affine))} for x in paths}
    for folder in a.experiments.iterdir():
        q=folder/'registration_qc.json'
        if not q.exists(): continue
        r=json.loads(q.read_text()); img=nib.load(folder/'registered_prediction.nii.gz'); data=np.asanyarray(img.dataobj)
        assert img.shape==ref.shape and np.allclose(img.affine,ref.affine) and np.isin(data,[0,1]).all()
        r['geometry_check']='PASS'; r['native_lesion_voxels']=int(np.count_nonzero(np.asanyarray(nib.load(a.prediction).dataobj)))
        r['registered_voxel_count_diagnostic_only']=int(np.count_nonzero(data))
        transform=s.ReadTransform(str(folder/'registration_transform.tfm')); inverse=transform.GetInverse()
        pts=[(0.,0.,0.),(20.,-30.,40.),(-40.,20.,-10.)]
        r['inverse_roundtrip_max_mm']=max(float(np.linalg.norm(np.array(inverse.TransformPoint(transform.TransformPoint(t)))-t)) for t in pts)
        r['native_prediction_sha256']=hashlib.sha256(a.prediction.read_bytes()).hexdigest()
        r['visual_qc']={k:'REVIEW' for k in ['brain_boundary','ventricles','brainstem','cerebellum','inferior_boundary']}
        r['visual_qc']['midline']='REVIEW'; r['visual_qc']['overall']='NEEDS REVIEW'
        r['visual_qc']['note']='This automated audit cannot approve anatomical landmarks. See the separate human-readable visual review.'
        if folder.name=='masked_refine': r['parameters'].update({'start_affine':'Block 4A baseline','learning_rate':.1,'maximum_step_mm':1.0})
        if folder.name=='flair': r['parameters'].update({'structural':'FLAIR','learning_rate':.5,'maximum_step_mm':1.0})
        if folder.name=='geometry': r['parameters']['maximum_step_mm']=0.0
        r['parameter_note']='Baseline and geometry used original default maximum step; masked_refine and FLAIR used explicit 1 mm cap. See experiment descriptions; these are not identical optimization runs.'
        (folder/'metrics.json').write_text(json.dumps(r,indent=2)); (folder/'log.txt').write_text(json.dumps(r,indent=2)); summary[folder.name]=r
    (a.experiments/'modalities.json').write_text(json.dumps(modalities,indent=2))
    pred=nib.load(a.prediction)
    native_ml=float(np.count_nonzero(np.asanyarray(pred.dataobj))*np.prod(pred.header.get_zooms()[:3],dtype=np.float64)/1000)
    (a.experiments/'comparison.json').write_text(json.dumps({'status':'NEEDS REVIEW','experiments':summary,'approved_best_transform':None,'candidate_for_review':'flair','native_prediction_volume_ml':native_ml},indent=2))
    print({k:v['registered_voxels'] for k,v in summary.items()})
    # FLAIR chain QC: DWI and structural already resampled to the same template grid.
    folder=a.experiments/'flair'
    if folder.exists():
        f=np.asanyarray(nib.load(folder/'registered_structural.nii.gz').dataobj); d=np.asanyarray(nib.load(folder/'registered_dwi.nii.gz').dataobj)
        fig,axs=plt.subplots(3,3,figsize=(12,12))
        for row,axis in enumerate((2,1,0)):
            x,y=[np.take(v,v.shape[axis]//2,axis=axis).T for v in (f,d)]
            for col,v in enumerate((x,y,x)):
                axs[row,col].imshow(v,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(v[v>0],99)); axs[row,col].axis('off')
                if col==2: axs[row,col].contour(y,levels=[np.max(y)*.12],colors='cyan',linewidths=.6)
                axs[row,col].set_title(['Registered FLAIR','Registered DWI','FLAIR + DWI contour'][col])
        fig.tight_layout(); fig.savefig(folder/'structural_chain_qc.png',dpi=100); plt.close(fig)

if __name__=='__main__': main()
