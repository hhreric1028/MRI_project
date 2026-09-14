"""Native-grid evaluation; exclusive output directory, never rewrites predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_erosion, generate_binary_structure
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

def metrics(gt, pred, affine, voxel_ml):
    ng, npred = int(gt.sum()), int(pred.sum())
    inter = int((gt & pred).sum())
    gl, gn = label(gt, np.ones((3,3,3)))
    pl, pn = label(pred, np.ones((3,3,3)))
    gs, ps = np.bincount(gl.ravel())[1:], np.bincount(pl.ravel())[1:]
    matches = 0
    if gn and pn:
        overlap = np.zeros((gn,pn), dtype=float)
        both = (gl>0) & (pl>0)
        np.add.at(overlap, (gl[both]-1, pl[both]-1), 1)
        iou = overlap / (gs[:,None]+ps[None,:]-overlap)
        score = (overlap>0)*(min(gn,pn)+1) + iou
        a,b = linear_sum_assignment(score, maximize=True)
        matches = int((overlap[a,b]>0).sum())
    hd = None
    hd_status = 'both_empty'
    if bool(ng) != bool(npred):
        hd_status = 'infinite_single_empty'
    elif ng:
        surfaces = []
        for mask in (gt,pred):
            surface = mask & ~binary_erosion(mask, structure=generate_binary_structure(3,1), border_value=0)
            surfaces.append(nib.affines.apply_affine(affine, np.argwhere(surface)))
        hd = float(max(np.percentile(cKDTree(surfaces[0]).query(surfaces[1])[0],95), np.percentile(cKDTree(surfaces[1]).query(surfaces[0])[0],95)))
        hd_status = 'finite'
    signed = (npred-ng)*voxel_ml
    return dict(dice=2*inter/(ng+npred) if ng+npred else None,
                voxel_tp=inter, voxel_fn=ng-inter, voxel_fp=npred-inter,
                voxel_recall=inter/ng if ng else None, voxel_precision=inter/npred if npred else None,
                gt_ml=ng*voxel_ml, pred_ml=npred*voxel_ml,
                signed_volume_error_ml=signed, absolute_volume_error_ml=abs(signed),
                signed_percentage_error=100*(npred-ng)/ng if ng else None,
                absolute_percentage_error=100*abs(npred-ng)/ng if ng else None,
                pred_gt_ratio=npred/ng if ng else None,
                false_positive_ml=(npred-inter)*voxel_ml,
                hd95_mm=hd, hd95_status=hd_status,
                lesion_tp=matches, lesion_fn=gn-matches, lesion_fp=pn-matches,
                lesion_recall=matches/gn if gn else None,
                lesion_precision=matches/pn if pn else None,
                lesion_f1=2*matches/(gn+pn) if gn+pn else None,
                gt_component_ml=(gs*voxel_ml).tolist(), pred_component_ml=(ps*voxel_ml).tolist())

def distribution(values):
    x=np.asarray([v for v in values if v is not None],float)
    if not len(x): return {'n':0}
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std()),
                minimum=float(x.min()),maximum=float(x.max()),
                **{k:float(np.percentile(x,p)) for k,p in [('q1',25),('q3',75),('p10',10),('p90',90)]})

def summarize(rows):
    tp=sum(r['lesion_tp'] for r in rows); fn=sum(r['lesion_fn'] for r in rows); fp=sum(r['lesion_fp'] for r in rows)
    d=[r['dice'] for r in rows if r['dice'] is not None]
    return dict(cases=len(rows),dice=distribution(d),
                voxel_recall=distribution([r.get('voxel_recall') for r in rows]),
                voxel_precision=distribution([r.get('voxel_precision') for r in rows]),
                fp_lesions_per_case=fp/len(rows) if rows else None,
                distributions={k:distribution([r[k] for r in rows]) for k in ['absolute_volume_error_ml','signed_volume_error_ml','signed_percentage_error','absolute_percentage_error','hd95_mm']},
                hd95_single_empty=sum(r['hd95_status']=='infinite_single_empty' for r in rows),
                hd95_both_empty=sum(r['hd95_status']=='both_empty' for r in rows),
                dice_ge_08=sum(v>=.8 for v in d),dice_ge_07=sum(v>=.7 for v in d),
                dice_lt_05=sum(v<.5 for v in d),dice_lt_03=sum(v<.3 for v in d),
                severe_under_count=sum(r['pred_gt_ratio'] is not None and r['pred_gt_ratio']<.5 for r in rows),
                catastrophic_miss_count=sum(r['lesion_recall']==0 for r in rows),
                lesion_tp=tp,lesion_fn=fn,lesion_fp=fp,lesion_recall=tp/(tp+fn) if tp+fn else None,
                lesion_precision=tp/(tp+fp) if tp+fp else None,lesion_f1=2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise SystemExit('Existing experiment output detected. No files were modified.')
    pp=ROOT/'datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI'
    ids=json.loads((pp/'splits_final.json').read_text())[0]['val']
    if {p.name[:-7] for p in args.predictions.glob('*.nii.gz')}!=set(ids): raise ValueError('Prediction case IDs differ from fixed fold0')
    rows=[]
    for i,case in enumerate(ids):
        g=nib.load(pp/'gt_segmentations'/f'{case}.nii.gz'); p=nib.load(args.predictions/f'{case}.nii.gz')
        if g.shape!=p.shape or not np.allclose(g.affine,p.affine,atol=1e-3,rtol=0): raise ValueError(case+' geometry mismatch')
        if g.header.get_xyzt_units()[0]!='mm': raise ValueError('Expected mm')
        gd=g.get_fdata(); pd=p.get_fdata()
        if not np.isfinite(gd).all() or not np.isfinite(pd).all(): raise ValueError('Nonfinite mask')
        if not np.allclose(gd,np.round(gd),atol=1e-8,rtol=0) or not set(np.unique(np.round(gd))).issubset({0,1}) or not set(np.unique(pd)).issubset({0,1}): raise ValueError('Nonbinary mask')
        rows.append(dict(case=case,**metrics(gd>.5,pd>.5,g.affine,float(np.prod(g.header.get_zooms(),dtype=np.float64))/1000)))
        print(f'{i+1}/50 {case} Dice={rows[-1]["dice"]} lesions FN={rows[-1]["lesion_fn"]}',flush=True)
    summary=summarize(rows)
    summary['subgroups']={name:summarize([r for r in rows if low<=r['gt_ml']<high]) for name,low,high in [('<1',0,1),('1-5',1,5),('5-20',5,20),('20-50',20,50),('>=50',50,float('inf'))]}
    summary['definitions']=dict(connectivity=26,matching='maximum-cardinality then maximum-IoU one-to-one overlap',hd95='max of directed surface P95; infinite values represented by status and count',sd_ddof=0,checkpoint='checkpoint_final.pth',fold=0)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in [('cases.json',rows),('summary.json',summary)]:
        with (args.output/name).open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,allow_nan=False)
    fields=[k for k in rows[0] if not k.endswith('component_ml')]
    with (args.output/'cases.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
