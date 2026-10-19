"""CPU native DWI/ADC to FLAIR rigid experiments and review-only QC artifacts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as s
from scipy.ndimage import label, sobel
from matplotlib.transforms import Bbox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metadata(path):
    n=nib.load(path); im=s.ReadImage(str(path)); data=np.asanyarray(n.dataobj)
    corners=np.array(list(itertools.product(*[(-.5,z-.5) for z in n.shape])))
    world=nib.affines.apply_affine(n.affine,corners)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(n.shape),'spacing_mm':list(map(float,n.header.get_zooms()[:3])),'affine_RAS':n.affine.tolist(),'orientation':list(nib.aff2axcodes(n.affine)),'origin_LPS':im.GetOrigin(),'direction_LPS':im.GetDirection(),'physical_extent_RAS_mm':{'min':world.min(axis=0).tolist(),'max':world.max(axis=0).tolist()},'finite':bool(np.isfinite(data).all()),'dtype':str(data.dtype)}

def norm(a):
    nonzero=a[a>0]
    top=np.percentile(nonzero,99) if nonzero.size else 1
    return np.clip(a/max(top,1e-12),0,1)

def qc(fixed,moving,pred,out):
    arrays=[s.GetArrayFromImage(v) for v in (fixed,moving,pred)]
    support=np.argwhere(arrays[0]>0)
    low,high=support.min(axis=0),support.max(axis=0)
    # SITK arrays Z,Y,X, displayed identically in all experiments.
    specs=[(axis,fr) for axis in (0,1,2) for fr in (.3,.5,.7)]
    for mode in ('multiplanar','edges'):
        fig,axs=plt.subplots(9,3,figsize=(12,30))
        for row,(axis,fr) in enumerate(specs):
            index=int(low[axis]+(high[axis]-low[axis])*fr)
            f,m,l=[np.take(v,index,axis=axis) for v in arrays]; f,m=norm(f),norm(m)
            for col,im in enumerate((f,m,f)):
                ax=axs[row,col]; ax.imshow(im,cmap='gray',origin='lower',vmin=0,vmax=1)
                if col==2:
                    if mode=='multiplanar': ax.imshow(m,cmap='magma',origin='lower',alpha=.35,vmin=0,vmax=1)
                    else:
                        edge=np.hypot(sobel(m,axis=0),sobel(m,axis=1))
                        if np.any(edge>0):
                            threshold=np.percentile(edge[edge>0],85)
                            ax.contour(edge,levels=[threshold],colors='cyan',linewidths=.35)
                    if l.any(): ax.contour(l,levels=[.5],colors='red',linewidths=.7)
                retained=[z for z in range(3) if z!=axis]; sp=fixed.GetSpacing()[::-1]
                ax.set_aspect(sp[retained[0]]/sp[retained[1]])
                ax.set_title(f'{["Axial","Coronal","Sagittal"][axis]} {fr:.1f} | {col+1}'); ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'qc_{mode}.png',dpi=110); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dwi','adc','flair','prediction','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dwi-transform',type=Path)
    p.add_argument('--reuse-existing',action='store_true',help='Regenerate QC using saved experiment transforms')
    p.add_argument('--verify-only',action='store_true',help='Verify existing results and compare transforms without registration')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); s.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    if a.verify_only:
        prior=json.loads((a.output/'metadata.json').read_text())
        unchanged=all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in prior['inputs'].values())
        if not unchanged: raise ValueError('An input file changed since analysis')
        pred=s.ReadImage(str(a.prediction)); fixed=nib.load(a.flair)
        transforms=[s.ReadTransform(str(a.output/f'{v}_to_flair'/'transform.tfm')) for v in ('dwi','adc')]
        forwards=[v.GetInverse() for v in transforms]
        differences=[]
        for idx in np.argwhere(s.GetArrayFromImage(pred)>0):
            point=pred.TransformIndexToPhysicalPoint(tuple(map(int,idx[::-1])))
            differences.append(float(np.linalg.norm(np.array(forwards[0].TransformPoint(point))-forwards[1].TransformPoint(point))))
        checks={}
        for role,t in zip(('dwi','adc'),transforms):
            folder=a.output/f'{role}_to_flair'; im=nib.load(folder/'registered_prediction.nii.gz'); data=np.asanyarray(im.dataobj)
            metrics=json.loads((folder/'metrics.json').read_text())
            assert im.shape==fixed.shape and np.allclose(im.affine,fixed.affine,atol=1e-4,rtol=1e-5)
            assert np.isfinite(data).all() and np.isin(data,[0,1]).all()
            assert sum(c['transformed_voxels_diagnostic'] for c in metrics['components'])==int(np.count_nonzero(data))
            checks[role]={'output_grid_match':True,'labels':np.unique(data).tolist(),'transformed_voxels_diagnostic':int(np.count_nonzero(data)),'retained_native_components':sum(c['transformed_voxels_diagnostic']>0 for c in metrics['components']),'outside_flair_nonzero_support_voxels':sum(c['outside_flair_nonzero_support_voxels'] for c in metrics['components']),'rigid_matrix_determinant':float(np.linalg.det(np.array(t.GetMatrix()).reshape(3,3)))}
        result={'inputs_unchanged':unchanged,'experiments':checks,'transform_disagreement_at_native_lesion_voxel_centers_mm':{'min':min(differences),'median':float(np.median(differences)),'max':max(differences)},'note':'Agreement between two transforms is not independent proof of anatomical accuracy.'}
        (a.output/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return
    meta={key:metadata(getattr(a,key)) for key in ('dwi','adc','flair','prediction')}
    d,c=meta['dwi'],meta['adc']; checks={'shape':d['shape']==c['shape'],'spacing':bool(np.allclose(d['spacing_mm'],c['spacing_mm'],atol=1e-4,rtol=1e-5)),'affine':bool(np.allclose(d['affine_RAS'],c['affine_RAS'],atol=1e-4,rtol=1e-5)),'orientation':d['orientation']==c['orientation']}
    if not all(checks.values()): raise ValueError('DWI/ADC grids differ; cannot use ADC transform on DWI mask')
    if not all(x['finite'] for x in meta.values()): raise ValueError('Nonfinite input')
    fixed=s.ReadImage(str(a.flair),s.sitkFloat32); pred=s.ReadImage(str(a.prediction))
    assert np.allclose(meta['prediction']['affine_RAS'],d['affine_RAS']) and meta['prediction']['shape']==d['shape']
    native=s.GetArrayFromImage(pred); assert np.isin(native,[0,1]).all()
    xyz_components,num=label(np.asanyarray(nib.load(a.prediction).dataobj)>0,np.ones((3,3,3),bool))
    components=xyz_components.transpose(2,1,0)
    lc=s.GetImageFromArray(components.astype(np.uint16)); lc.CopyInformation(pred)
    fm=s.GetArrayFromImage(fixed)
    (a.output/'metadata.json').write_text(json.dumps({'inputs':meta,'dwi_adc_checks':checks,'tolerance':{'rtol':1e-5,'atol':1e-4}},indent=2))
    summary={}
    for source in ('dwi','adc'):
        out=a.output/f'{source}_to_flair'; out.mkdir(exist_ok=True)
        moving=s.ReadImage(str(getattr(a,source)),s.sitkFloat32)
        if a.reuse_existing and (out/'transform.tfm').exists():
            transform=s.ReadTransform(str(out/'transform.tfm')); history=json.loads((out/'metrics.json').read_text())['history']
        elif source=='dwi' and a.dwi_transform:
            transform=s.ReadTransform(str(a.dwi_transform)); history={'reused':str(a.dwi_transform.resolve())}
        else:
            transform=s.CenteredTransformInitializer(fixed,moving,s.Euler3DTransform(),s.CenteredTransformInitializerFilter.MOMENTS)
            r=s.ImageRegistrationMethod(); r.SetMetricAsMattesMutualInformation(50); r.SetMetricSamplingStrategy(r.RANDOM); r.SetMetricSamplingPercentage(.3,2022)
            r.SetInterpolator(s.sitkLinear); r.SetOptimizerAsGradientDescent(.5,200,1e-6,15,maximumStepSizeInPhysicalUnits=1.)
            r.SetOptimizerScalesFromPhysicalShift(); r.SetShrinkFactorsPerLevel([4,2,1]); r.SetSmoothingSigmasPerLevel([2,1,0]); r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            r.SetInitialTransform(transform,inPlace=True); r.Execute(fixed,moving)
            history={'metric':r.GetMetricValue(),'stop':r.GetOptimizerStopConditionDescription()}
        s.WriteTransform(transform,str(out/'transform.tfm')); s.WriteTransform(transform.GetInverse(),str(out/'native_to_flair_forward.tfm'))
        registered=s.Resample(moving,fixed,transform,s.sitkLinear,0,s.sitkFloat32)
        rp=s.Resample(pred,fixed,transform,s.sitkNearestNeighbor,0,s.sitkUInt8)
        rc=s.GetArrayFromImage(s.Resample(lc,fixed,transform,s.sitkNearestNeighbor,0,s.sitkUInt16))
        s.WriteImage(registered,str(out/'registered_image.nii.gz')); s.WriteImage(rp,str(out/'registered_prediction.nii.gz'))
        print(source,'registration and resampling complete',flush=True)
        qc(fixed,registered,rp,out)
        rows=[]; fig,axs=plt.subplots(num,3,figsize=(12,4*num),squeeze=False)
        for i in range(1,num+1):
            vox=np.argwhere(rc==i); row={'native_label_id':i,'native_voxels':int(np.count_nonzero(components==i)),'transformed_voxels_diagnostic':len(vox)}
            row['outside_flair_nonzero_support_voxels']=int(np.count_nonzero((rc==i)&(fm<=0)))
            rows.append(row)
            if not len(vox):
                for ax in axs[i-1]: ax.text(.1,.5,'Component absent after resampling'); ax.axis('off')
                continue
            center=np.rint(vox.mean(axis=0)).astype(int)
            for col,axis in enumerate((0,1,2)):
                im=np.take(fm,center[axis],axis=axis); seg=np.take(rc==i,center[axis],axis=axis)
                axs[i-1,col].imshow(norm(im),cmap='gray',origin='lower'); axs[i-1,col].contour(seg,levels=[.5],colors='red',linewidths=.8)
                retained=[z for z in range(3) if z!=axis]; sp=fixed.GetSpacing()[::-1]
                axs[i-1,col].set_aspect(sp[retained[0]]/sp[retained[1]])
                # Local 40-mm context around each native component, in FLAIR grid.
                cy,cx=center[retained[0]],center[retained[1]]
                dy,dx=20/sp[retained[0]],20/sp[retained[1]]
                axs[i-1,col].set_xlim(max(0,cx-dx),min(im.shape[1]-1,cx+dx))
                axs[i-1,col].set_ylim(max(0,cy-dy),min(im.shape[0]-1,cy+dy))
                axs[i-1,col].set_title(f'Native component {i} | {"ACS"[col]}'); axs[i-1,col].axis('off')
        fig.tight_layout(); fig.savefig(out/'qc_lesions.png',dpi=85)
        fig.canvas.draw()
        for start in range(0,num,4):
            boxes=[ax.get_tightbbox(fig.canvas.get_renderer()) for row in axs[start:start+4] for ax in row]
            bounds=Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(out/f'qc_lesions_page_{start//4+1}.png',dpi=100,bbox_inches=bounds)
        plt.close(fig)
        result={'source':source,'status':'NEEDS REVIEW','method':'rigid','transform_direction':'FLAIR LPS to native LPS, resampling pull','parameters':{'initialization':'MOMENTS','metric':'Mattes MI','bins':50,'sampling':'RANDOM','fraction':.3,'seed':2022,'shrink':[4,2,1],'smoothing_mm':[2,1,0],'iterations':200,'learning_rate':.5,'max_step_mm':1},'history':history,'native_lesion_voxels':int(np.count_nonzero(native)),'native_clinical_volume_reference':'Block 2: 2.440 mL; not recomputed','transformed_labels':np.unique(s.GetArrayFromImage(rp)).tolist(),'transformed_voxels_diagnostic':int(np.count_nonzero(s.GetArrayFromImage(rp))),'components':rows,'note':'Nonzero FLAIR support is not a validated brain/CSF mask; visual lesion QC remains required.'}
        assert np.isin(s.GetArrayFromImage(rp),[0,1]).all()
        for name in ('metrics.json','log.txt'): (out/name).write_text(json.dumps(result,indent=2))
        summary[source]=result
    (a.output/'summary.json').write_text(json.dumps({'status':'NEEDS REVIEW','selected_source':'NONE','experiments':summary},indent=2))

if __name__=='__main__': main()
