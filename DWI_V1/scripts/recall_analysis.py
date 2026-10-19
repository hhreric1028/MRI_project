"""Read-only paired analysis, case bootstrap, and fixed-slice failure panels."""
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from recall_common import ROOT, PP, save

B0=ROOT/'experiments/B0_baseline/evaluation_v2'
P1=ROOT/'experiments/P1_spacing_1p5mm/run_seed2022_train_r2'
B0_PRED=ROOT/'outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation'
P1_PRED=P1/'results/Dataset011_ISLES2022_DWI/nnUNetTrainer_SeededPreserve__nnUNetPlans__3d_fullres/fold_0/validation'
LIMITATIONS='Fold 0 only; single seed screening. Controlled same-seed comparison is not fully deterministic. Actual sampling exposure change must be established by the audit. Case bootstrap CIs describe uncertainty within this validation cohort, not generalization across folds, centers or training seeds.'

def group(v):
    return '<1 mL' if v<1 else '1–5 mL' if v<5 else '5–20 mL' if v<20 else '20–50 mL' if v<50 else '≥50 mL'

def load_cases(folder):
    rows=json.loads((Path(folder)/'cases.json').read_text())
    ids=[r['case'] for r in rows]
    expected=json.loads((PP/'splits_final.json').read_text())[0]['val']
    assert len(ids)==len(set(ids))==len(expected)==50 and set(ids)==set(expected)
    return {r['case']:r for r in rows}

def stat(rows):
    def mean(key):
        x=[r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(x)) if x else np.nan
    def rate(predicate): return float(np.mean([predicate(r) for r in rows]))
    tp=sum(r['lesion_tp'] for r in rows);fp=sum(r['lesion_fp'] for r in rows);fn=sum(r['lesion_fn'] for r in rows)
    severe=sum(r['pred_gt_ratio'] is not None and r['pred_gt_ratio']<.5 for r in rows)
    return dict(mean_dice=mean('dice'),median_dice=float(np.median([r['dice'] for r in rows if r['dice'] is not None])),
        lesion_recall=tp/(tp+fn) if tp+fn else np.nan,lesion_precision=tp/(tp+fp) if tp+fp else np.nan,
        volume_mae_ml=mean('absolute_volume_error_ml'),signed_volume_error_ml=mean('signed_volume_error_ml'),
        severe_under_count=severe,severe_under_rate=severe/len(rows),dice_below_05_rate=rate(lambda r:r['dice'] is not None and r['dice']<.5),
        catastrophic_miss_rate=rate(lambda r:r['lesion_recall']==0 and r['gt_ml']>0),
        fp_lesions_per_case=fp/len(rows),hd95_finite_mean_mm=mean('hd95_mm'),
        hd95_infinite_rate=rate(lambda r:r['hd95_status']=='infinite_single_empty'))

def bootstrap(a,b,ids,n=10000):
    base=stat([a[i] for i in ids]); candidate=stat([b[i] for i in ids])
    rng=np.random.default_rng(2022); values={k:[] for k in base}
    for _ in range(n):
        sample=rng.choice(ids,len(ids),replace=True)
        x=stat([a[i] for i in sample]); y=stat([b[i] for i in sample])
        for k in values: values[k].append(y[k]-x[k])
    result={}
    for k,v in values.items():
        finite=np.asarray(v)[np.isfinite(v)]
        point=candidate[k]-base[k]
        result[k]=dict(baseline=base[k] if np.isfinite(base[k]) else None,
            candidate=candidate[k] if np.isfinite(candidate[k]) else None,
            delta=point if np.isfinite(point) else None,
            ci95=np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,valid_replicates=len(finite))
    return result

def compare(baseline,candidate,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    a=load_cases(baseline);b=load_cases(candidate);ids=sorted(a)
    paired=[]
    for case in ids:
        assert np.isclose(a[case]['gt_ml'],b[case]['gt_ml'])
        row={'case':case,'gt_volume_group':group(a[case]['gt_ml']),'baseline':a[case],'candidate':b[case]}
        row['delta']={k:b[case][k]-a[case][k] for k in a[case] if isinstance(a[case][k],(int,float)) and isinstance(b[case].get(k),(int,float))}
        paired.append(row)
    save(out/'paired_cases.json',paired)
    print('Paired case bootstrap: 10,000 resamples; seed 2022; candidate minus baseline.',flush=True)
    ci=bootstrap(a,b,ids)
    subgroups={}
    for g in ['<1 mL','1–5 mL','5–20 mL','20–50 mL','≥50 mL']:
        sub=[i for i in ids if group(a[i]['gt_ml'])==g]
        subgroups[g]=dict(cases=len(sub),bootstrap=bootstrap(a,b,sub) if sub else None)
    save(out/'paired_bootstrap.json',dict(iterations=10000,seed=2022,resampling_unit='case ID; identical IDs for both models; pooled lesion TP/FP/FN recomputed',overall=ci,subgroups=subgroups,limitations=LIMITATIONS))
    lines=['# Paired comparison',f'Baseline: {baseline}',f'Candidate: {candidate}',
        'Differences are candidate − baseline. Percentile 95% CI, paired case bootstrap, 10,000 iterations, seed 2022.',
        '', '| Metric | Baseline | Candidate | Delta | 95% CI |','|---|---:|---:|---:|---|']
    def fmt(v): return f'{v:.3f}' if v is not None else 'not provided'
    for k,r in ci.items():
        lines.append(f"| {k} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | {fmt(r['delta'])} | {r['ci95']} |")
    lines+=['','## Decision guard','Prioritize catastrophic miss and severe under-segmentation; then 1–5 mL recall/Dice; then overall Dice. A recall increase with increased FP is a tradeoff, not an automatic winner. No automatic combinations, additional seeds or P2.', '',LIMITATIONS]
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return paired

def failure(out):
    out=Path(out); paired=compare(B0,P1/'metrics',out)
    ordered=sorted(paired,key=lambda r:r['delta'].get('dice',0))
    reasons={}
    for key,cases in [('new_severe',['case_0118','case_0135','case_0160']),('top5_decline',[r['case'] for r in ordered[:5]]),
                      ('top5_improve',[r['case'] for r in ordered[-5:]]),('known_low',['case_0068','case_0182','case_0181','case_0140','case_0212'])]:
        for case in cases: reasons.setdefault(case,[]).append(key)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    details=[]; figdir=out/'figures';figdir.mkdir()
    for case,why in sorted(reasons.items()):
        paths=[ROOT/'datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI/imagesTr'/f'{case}_0000.nii.gz',
               PP/'gt_segmentations'/f'{case}.nii.gz',B0_PRED/f'{case}.nii.gz',P1_PRED/f'{case}.nii.gz']
        imgs=[nib.load(p) for p in paths]
        assert all(i.shape==imgs[0].shape and np.allclose(i.affine,imgs[0].affine) for i in imgs)
        # Canonical orientation is used consistently for all four arrays, without interpolation.
        arrays=[nib.as_closest_canonical(i).get_fdata(dtype=np.float32) for i in imgs]
        dwi=arrays[0];gt,b0,p1=[x>.5 for x in arrays[1:]]
        zs=[int(np.argmax(gt.sum((0,1)))),int(np.argmax((b0!=p1).sum((0,1))))]
        gl,n=label(gt,np.ones((3,3,3)));sizes=np.bincount(gl.ravel())[1:]
        info={'case':case,'selection_reasons':why,'axial_canonical_z':zs,'models':{}}
        for name,pred in [('B0',b0),('P1',p1)]:
            overlap=np.bincount(gl[pred].ravel(),minlength=n+1)[1:]
            pl,pn=label(pred,np.ones((3,3,3)))
            hit=np.unique(pl[gt]);extra=sum(j not in hit for j in range(1,pn+1))
            info['models'][name]=dict(whole_gt_components_no_overlap=int((overlap==0).sum()),
                detected_gt_components_with_incomplete_coverage=int(((overlap>0)&(overlap<sizes)).sum()),
                extra_prediction_components_no_gt_overlap=extra,gt_component_voxel_coverage=(overlap/sizes).tolist(),
                voxel_recall=float((gt&pred).sum()/gt.sum()) if gt.any() else None)
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        finite=dwi[np.isfinite(dwi)];lo,hi=np.percentile(finite,[1,99])
        for row,z in enumerate(zs):
            for col,(name,mask) in enumerate([('DWI',None),('GT',gt),('B0',b0),('P1',p1)]):
                ax=axes[row,col];ax.imshow(dwi[:,:,z].T,origin='lower',cmap='gray',vmin=lo,vmax=hi)
                if mask is not None:
                    overlay=np.ma.masked_where(~mask[:,:,z].T,mask[:,:,z].T)
                    ax.imshow(overlay,origin='lower',cmap='autumn',vmin=0,vmax=1,alpha=.55)
                    if gt[:,:,z].any():ax.contour(gt[:,:,z].T,levels=[.5],colors=['cyan'],linewidths=.6)
                ax.set_title(f'{name} | z={z}');ax.axis('off')
            axes[row,0].set_ylabel(['GT max area','B0/P1 max disagreement'][row])
        fig.suptitle(case+' | same DWI scale/slice; cyan = GT boundary; yellow = mask')
        fig.tight_layout();fig.savefig(figdir/f'{case}.png',dpi=120);plt.close(fig)
        details.append(info);print('Failure panel '+case,flush=True)
    save(out/'failure_patterns.json',details)
    lines=['# P1 failure pattern review','Panels: first row = maximal GT area; second row = maximal B0/P1 disagreement. All are the same native DWI slice; common intensity scaling.',
           'Component coverage is overlap-based, distinct from one-to-one evaluation lesion FN (merges can cause unmatched lesions). These are descriptive mask findings, not evidence that FOV or heterogeneity caused failure.','',
           '| Case | Model | Whole lesion components missed | Detected but incomplete | Extra FP components | Voxel recall |','|---|---|---:|---:|---:|---:|']
    for r in details:
        for name,m in r['models'].items():lines.append(f"| {r['case']} | {name} | {m['whole_gt_components_no_overlap']} | {m['detected_gt_components_with_incomplete_coverage']} | {m['extra_prediction_components_no_gt_overlap']} | {m['voxel_recall']:.3f} |")
    (out/'failure_review.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--failure',action='store_true');p.add_argument('--baseline');p.add_argument('--candidate');p.add_argument('--output',required=True);a=p.parse_args()
    if a.failure:failure(a.output)
    else:compare(a.baseline,a.candidate,a.output)
