"""One prospective 1000-patch audit per ratio; never regenerates existing audit."""
import argparse
import hashlib
import json
import os
import random
import traceback
import numpy as np
import nibabel as nib
import torch
from recall_common import PP, SUITE, save, protect_baseline, verify_baseline, provenance, now
from recall_analysis import group
from run_recall_experiment import make_trainer, seed_all

def geometric_coverage(coords,bbox_low,initial_shape,patch,params):
    """Fraction of source GT voxel centers inside the augmented patch support.

    Uses the inverse actual affine, and intersects with the initial crop support.
    Not post-augmentation voxel count divided by native volume (scaling biases that).
    Excludes interpolation/threshold effects; denominator is full preprocessed GT.
    """
    if not len(coords):return None
    local=coords-np.asarray(bbox_low)
    valid=np.all((local>=0)&(local<np.asarray(initial_shape)),axis=1)
    center=np.asarray(params['center_location_in_pixels'])
    if params['affine'] is None:
        low=np.floor(center)-np.asarray(patch)//2
        inside=np.all((local>=low)&(local<low+np.asarray(patch)),axis=1)
    else:
        assert params['elastic_offsets'] is None
        displacement=local+.5-center
        output=displacement@np.linalg.inv(params['affine']).T
        inside=np.all((output>=-np.asarray(patch)/2)&(output<np.asarray(patch)/2),axis=1)
    return float(np.mean(valid&inside))

def summarize(rows):
    def avg(k):
        x=[r[k] for r in rows if r[k] is not None]
        return float(np.mean(x)) if x else None
    return dict(patches=len(rows),distinct_cases=len({r['case'] for r in rows}),
        forced_foreground_rate=avg('force_fg'),lesion_containing_patch_rate=avg('contains_lesion'),
        foreground_voxel_fraction=avg('lesion_voxel_fraction'),lesion_voxel_count=avg('lesion_voxel_count'),
        lesion_coverage=avg('lesion_coverage'),small_lesion_containing_patches=sum(r['contains_lesion'] and r['gt_volume_group']=='1–5 mL' for r in rows))

def run_group(ratio,out):
    # Synchronous diagnostic uses the real loader/augmentation, but no nondeterministic worker scheduling.
    seed_all();trainer=make_trainer('B0-R2022','cpu');trainer.train_foreground=ratio
    loader,_=trainer.raw_loaders()
    spatial=loader.transforms.transforms[0]
    assert spatial.__class__.__name__=='SpatialTransform' and spatial.p_elastic_deform==0
    coords_cache={};volume_cache={};state={};records=[]
    original_indices=loader.get_indices;original_bbox=loader.get_bbox;original_params=spatial.get_parameters
    def indices():
        keys=original_indices();state.update(keys=keys,index=0);return keys
    def bbox(shape,force_fg,class_locations,*args,**kwargs):
        case=state['keys'][state['index']];state['index']+=1
        low,high=original_bbox(shape,force_fg,class_locations,*args,**kwargs)
        state.update(case=case,force_fg=bool(force_fg),low=low,shape=shape)
        return low,high
    def params(**data):
        value=original_params(**data);case=state['case']
        if case not in coords_cache:
            _,seg,_,_=loader._data.load_case(case)
            coords_cache[case]=np.argwhere(np.asarray(seg[:])[0]==1)
            gt=nib.load(PP/'gt_segmentations'/f'{case}.nii.gz')
            volume_cache[case]=float(np.count_nonzero(np.asarray(gt.dataobj)==1)*abs(np.linalg.det(gt.affine[:3,:3]))/1000)
        coverage=geometric_coverage(coords_cache[case],state['low'],data['image'].shape[1:],spatial.patch_size,value)
        records.append(dict(case=case,force_fg=state['force_fg'],gt_ml=volume_cache[case],gt_volume_group=group(volume_cache[case]),
            preprocessed_gt_voxels=len(coords_cache[case]),lesion_coverage=coverage,bbox_low=list(map(int,state['low']))))
        return value
    loader.get_indices=indices;loader.get_bbox=bbox;spatial.get_parameters=params
    # Reset after constructor inspection, so both groups start with identical RNG state.
    seed_all()
    with (out/f'patches_{ratio:.2f}.jsonl').open('x',encoding='utf-8') as f:
        for batch_idx in range(125):
            batch=next(loader);target=batch['target'][0]
            for j,r in enumerate(records[-8:]):
                mask=(target[j]==1);count=int(mask.sum())
                r.update(patch_index=batch_idx*8+j,ratio=ratio,contains_lesion=count>0,
                    lesion_voxel_count=count,lesion_voxel_fraction=count/mask.numel(),
                    segmentation_sha256=hashlib.sha256(target[j].numpy().tobytes()).hexdigest())
                f.write(json.dumps(r,allow_nan=False)+'\n')
            f.flush()
            if (batch_idx+1)%5==0:print(f'Sampling {ratio:.2f}: {(batch_idx+1)*8}/1000 post-augmentation patches',flush=True)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(SUITE/'sampling_audit'));a=p.parse_args()
    from pathlib import Path
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);protect_baseline()
    try:
        verify_baseline(out/'baseline_before.json');os.environ['nnUNet_results']=str(out/'loader_metadata')
        torch.set_num_threads(1);save(out/'provenance.json',provenance())
        groups={str(r):run_group(r,out) for r in (.33,.50)}
        result={}
        for key,rows in groups.items():
            result[key]={'overall':summarize(rows),'subgroups':{g:summarize([r for r in rows if r['gt_volume_group']==g]) for g in ['<1 mL','1–5 mL','5–20 mL','20–50 mL','≥50 mL']},
                'cases':{case:summarize([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})}}
        delta={}
        for scope in ['overall','<1 mL','1–5 mL','5–20 mL','20–50 mL','≥50 mL']:
            x=result['0.33']['overall'] if scope=='overall' else result['0.33']['subgroups'][scope]
            y=result['0.5']['overall'] if scope=='overall' else result['0.5']['subgroups'][scope]
            delta[scope]={k:y[k]-x[k] if x[k] is not None and y[k] is not None else None for k in x}
        save(out/'summary.json',dict(results=result,delta_050_minus_033=delta,
            coverage_definition=geometric_coverage.__doc__,seed=2022,patches_per_group=1000,validation_sampling=.33,
            missing_fields=[],decision='review_required: assess substantive lesion exposure increase, especially 1–5 mL; equality alone is NOT a decision criterion',
            caveat='Same initial seed does not pair subsequent cases when different cropping branches consume different random draws. Diagnostic is synchronous; full training uses seeded nondeterministic workers. Native GT defines subgroups; coverage denominator is preprocessed GT.'))
        verify_baseline(out/'baseline_after.json');save(out/'completion.json',dict(status='audit_complete_review_required',time=now()))
    except BaseException as e:
        save(out/'failure.json',dict(time=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
