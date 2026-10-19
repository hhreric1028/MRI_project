"""CPU affine registration; transform maps reference LPS points to native LPS points."""
import argparse
import json
from pathlib import Path
import SimpleITK as s
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dwi','mask','reference','output'): p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--initialization',choices=['moments','geometry'],default='moments')
    p.add_argument('--brain-mask',action='store_true')
    p.add_argument('--reuse-transform',type=Path)
    p.add_argument('--start-affine',type=Path)
    p.add_argument('--structural',type=Path,help='Optional structural bridge; fit DWI to structural then structural to template')
    p.add_argument('--learning-rate',type=float,default=.5)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    s.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    f=s.ReadImage(str(a.reference),s.sitkFloat32); m=s.ReadImage(str(a.dwi),s.sitkFloat32)
    mask=s.ReadImage(str(a.mask),s.sitkUInt8)
    ni=nib.load(a.dwi); nm=nib.load(a.mask)
    assert ni.shape==nm.shape and np.allclose(ni.affine,nm.affine,rtol=1e-5,atol=1e-4)
    assert np.isin(np.asanyarray(nm.dataobj),[0,1]).all()
    centering=s.CenteredTransformInitializerFilter.MOMENTS if a.initialization=='moments' else s.CenteredTransformInitializerFilter.GEOMETRY
    initial=s.CenteredTransformInitializer(f,m,s.Euler3DTransform(),centering)
    history=[]
    def optimize(t):
        r=s.ImageRegistrationMethod(); r.SetMetricAsMattesMutualInformation(50)
        if a.brain_mask:
            r.SetMetricFixedMask(s.Cast(f>0,s.sitkUInt8)); r.SetMetricMovingMask(s.Cast(m>0.001,s.sitkUInt8))
        r.SetMetricSamplingStrategy(r.RANDOM); r.SetMetricSamplingPercentage(0.3,2022)
        r.SetInterpolator(s.sitkLinear)
        r.SetOptimizerAsGradientDescent(learningRate=a.learning_rate,numberOfIterations=200,convergenceMinimumValue=1e-6,convergenceWindowSize=15,maximumStepSizeInPhysicalUnits=1.0)
        r.SetOptimizerScalesFromPhysicalShift(); r.SetShrinkFactorsPerLevel([4,2,1])
        r.SetSmoothingSigmasPerLevel([2,1,0]); r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        r.SetInitialTransform(t,inPlace=True); r.Execute(f,m)
        history.append({'metric':r.GetMetricValue(),'stop':r.GetOptimizerStopConditionDescription()})
        return t
    bridge=None; native_m=m
    if a.structural:
        standard=f
        f=s.ReadImage(str(a.structural),s.sitkFloat32)
        bridge_init=s.CenteredTransformInitializer(f,m,s.Euler3DTransform(),centering)
        bridge=optimize(bridge_init)
        s.WriteTransform(bridge,str(a.output/'structural_to_dwi_pull.tfm'))
        m=f; f=standard
        initial=s.CenteredTransformInitializer(f,m,s.Euler3DTransform(),centering)
    if a.reuse_transform:
        affine=s.ReadTransform(str(a.reuse_transform))
    elif a.start_affine:
        affine=optimize(s.AffineTransform(s.ReadTransform(str(a.start_affine))))
    else:
        rigid=optimize(initial)
        affine=s.AffineTransform(3); affine.SetCenter(rigid.GetCenter()); affine.SetMatrix(rigid.GetMatrix()); affine.SetTranslation(rigid.GetTranslation())
        affine=optimize(affine)
    affine_determinant=float(np.linalg.det(np.array(affine.GetMatrix()).reshape(3,3)))
    if bridge is not None:
        s.WriteTransform(affine,str(a.output/'standard_to_structural_pull.tfm'))
        s.WriteImage(s.Resample(m,f,affine,s.sitkLinear,0,s.sitkFloat32),str(a.output/'registered_structural.nii.gz'))
        chain=s.CompositeTransform(3); chain.AddTransform(bridge); chain.AddTransform(affine)
        affine=chain; m=native_m
    s.WriteTransform(affine,str(a.output/'registration_transform.tfm'))
    s.WriteTransform(affine.GetInverse(),str(a.output/'native_to_standard.tfm'))
    rd=s.Resample(m,f,affine,s.sitkLinear,0,s.sitkFloat32)
    rp=s.Resample(mask,f,affine,s.sitkNearestNeighbor,0,s.sitkUInt8)
    s.WriteImage(rd,str(a.output/'registered_dwi.nii.gz')); s.WriteImage(rp,str(a.output/'registered_prediction.nii.gz'))
    arrays=[s.GetArrayFromImage(x) for x in (f,rd,rp)]
    fig,axs=plt.subplots(3,3,figsize=(12,12))
    for row,axis in enumerate((0,1,2)):
        index=arrays[0].shape[axis]//2
        x,y,z=[np.take(v,index,axis=axis) for v in arrays]
        for col,img in enumerate((x,y,x)):
            ax=axs[row,col]; ax.imshow(img,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(img[img>0],99) if np.any(img>0) else 1)
            if col==2:
                ax.contour(y,levels=[float(np.max(y)*.12)],colors='cyan',linewidths=.6)
                if z.any(): ax.contour(z,levels=[.5],colors='red',linewidths=1)
            ax.set_title(['Template','Registered DWI','Template + DWI contour + lesion'][col]); ax.axis('off')
    fig.tight_layout(); fig.savefig(a.output/'registration_qc.png',dpi=140); plt.close(fig)
    # Nine matched physical-grid planes: each triplet is reference, moving, checkerboard.
    fig,axs=plt.subplots(9,3,figsize=(12,30))
    for row,(axis,fraction) in enumerate((ax,fr) for ax in (0,1,2) for fr in (.35,.5,.65)):
        index=int(arrays[0].shape[axis]*fraction)
        x,y,z=[np.take(v,index,axis=axis) for v in arrays]
        def norm(v): return np.clip(v/(np.percentile(v[v>0],99) if np.any(v>0) else 1),0,1)
        xx,yy=norm(x),norm(y); grid=np.indices(x.shape); checker=((grid[0]//6+grid[1]//6)%2)==0
        for col,img in enumerate((xx,yy,np.where(checker,xx,yy))):
            axs[row,col].imshow(img,origin='lower',cmap='gray',vmin=0,vmax=1)
            if col==2 and z.any(): axs[row,col].contour(z,levels=[.5],colors='red',linewidths=.6)
            axs[row,col].set_title(f'{["Axial","Coronal","Sagittal"][axis]} {fraction:.2f} | {col+1}'); axs[row,col].axis('off')
    fig.tight_layout(); fig.savefig(a.output/'qc_multiplanar.png',dpi=100); plt.close(fig)
    result={'status':'NEEDS REVIEW','tool':s.Version_VersionString(),'method':'DWI rigid to structural, structural rigid then affine to template' if bridge else 'rigid then affine, Mattes MI','reference':str(a.reference.resolve()),'transform_direction':'reference LPS to native LPS (resampling pull transform)','history':history,'affine_determinant':affine_determinant,'native_orientation':nib.aff2axcodes(ni.affine),'native_volume_ml':float(np.count_nonzero(np.asanyarray(nm.dataobj))*np.prod(nm.header.get_zooms()[:3])/1000),'registered_voxels':int(np.count_nonzero(arrays[2])),'registered_labels':np.unique(arrays[2]).tolist(),'note':'Visual QC required before atlas mapping. Standard-space volume differs from native volume after affine resampling.'}
    result['parameters']={'initialization':a.initialization,'brain_mask':a.brain_mask,'metric':'Mattes MI','bins':50,'sampling':'random','sampling_percentage':.3,'seed':2022,'shrink_factors':[4,2,1],'smoothing_sigmas_mm':[2,1,0],'maximum_iterations_per_level':200}
    result['resampling_diagnostic_only']=True
    result['parameters'].update({'learning_rate':a.learning_rate,'maximum_step_mm':1.0,'start_affine':str(a.start_affine) if a.start_affine else None,'structural':str(a.structural) if a.structural else None,'reuse_transform':str(a.reuse_transform) if a.reuse_transform else None})
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2))
    (a.output/'log.txt').write_text(json.dumps(result,indent=2))
    (a.output/'registration_qc.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
