"""Evaluate final-checkpoint validation only; do not launch any training."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import csv,json,random
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=ROOT/'outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0'
    state=json.loads((ROOT/'outputs/metrics/training_fold0/fold0_status.json').read_text())
    if state['status']!='training_and_validation_complete':
        raise RuntimeError('Training/final validation has not completed successfully')
    # Only load this locally produced trusted checkpoint.
    checkpoint=torch.load(base/'checkpoint_final.pth',map_location='cpu',weights_only=False)
    if checkpoint['current_epoch']!=1000:
        raise RuntimeError(f"Unexpected final epoch: {checkpoint['current_epoch']}")
    del checkpoint
    pp=ROOT/'datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI'
    ids=json.loads((pp/'splits_final.json').read_text())[0]['val']
    val=base/'validation'
    assert (val/'summary.json').exists()
    assert {p.name[:-7] for p in val.glob('*.nii.gz')}==set(ids)
    raw=ROOT/'datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI'
    rows=[]
    for case in ids:
        gtimg=nib.load(raw/'labelsTr'/f'{case}.nii.gz');predimg=nib.load(val/f'{case}.nii.gz')
        assert gtimg.shape==predimg.shape and np.allclose(gtimg.affine,predimg.affine,atol=1e-3,rtol=0),case
        assert nib.aff2axcodes(gtimg.affine)==nib.aff2axcodes(predimg.affine),case
        assert np.allclose(gtimg.header.get_zooms(),predimg.header.get_zooms(),atol=1e-5,rtol=0),case
        gt=gtimg.get_fdata(dtype=np.float32)>0.5;prediction=predimg.get_fdata(dtype=np.float32)
        assert np.isfinite(prediction).all() and set(np.unique(prediction)).issubset({0,1}),case
        with np.load(val/f'{case}.npz') as archive:
            probabilities=archive['probabilities']
            assert probabilities.shape==(2, *gtimg.shape[::-1]) and np.isfinite(probabilities).all(),case
            assert probabilities.min()>=-1e-6 and probabilities.max()<=1+1e-6,case
            assert np.allclose(probabilities.sum(axis=0),1,atol=2e-3),case
        prediction=prediction==1
        ng,npred=int(gt.sum()),int(prediction.sum());intersection=int((gt&prediction).sum())
        assert gtimg.header.get_xyzt_units()[0]=='mm',case
        voxel_ml=float(np.prod(gtimg.header.get_zooms()[:3],dtype=np.float64))/1000
        rows.append(dict(case=case,dice=2*intersection/(ng+npred) if ng+npred else None,
                         gt_voxels=ng,pred_voxels=npred,gt_ml=ng*voxel_ml,pred_ml=npred*voxel_ml,
                         absolute_volume_error_ml=abs(ng-npred)*voxel_ml,empty_gt=ng==0,
                         empty_prediction=npred==0,false_positive_ml=int((prediction&~gt).sum())*voxel_ml))
    metrics=ROOT/'outputs/metrics/training_fold0'
    with (metrics/'fold0_validation_cases.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    valid=[r['dice'] for r in rows if r['dice'] is not None]
    random_ids=sorted(random.Random(2022).sample(sorted(ids),5))
    worst=[r['case'] for r in sorted((r for r in rows if r['dice'] is not None),key=lambda r:(r['dice'],r['case']))[:5]]
    qa_ids=sorted(set(random_ids+worst));out=ROOT/'outputs/figures/training_fold0';out.mkdir(parents=True,exist_ok=True)
    for case in qa_ids:
        img=nib.load(raw/'imagesTr'/f'{case}_0000.nii.gz');data=img.get_fdata(dtype=np.float32)
        gt=nib.load(raw/'labelsTr'/f'{case}.nii.gz').get_fdata(dtype=np.float32)>0.5
        pred=nib.load(val/f'{case}.nii.gz').get_fdata(dtype=np.float32)==1
        support=gt if gt.any() else pred
        z=int(np.argmax(support.sum(axis=(0,1)))) if support.any() else data.shape[2]//2
        fig,axes=plt.subplots(1,3,figsize=(12,5))
        for ax,k in zip(axes,[max(0,z-1),z,min(data.shape[2]-1,z+1)]):
            plane=data[:,:,k].T;v=plane[plane!=0];lo,hi=np.percentile(v,[1,99]) if v.size else (0,1)
            ax.imshow(plane,origin='lower',cmap='gray',vmin=lo,vmax=hi,aspect=img.header.get_zooms()[1]/img.header.get_zooms()[0])
            for mask,color in ((gt,'lime'),(pred,'red')):
                if mask[:,:,k].any():ax.contour(mask[:,:,k].T,levels=[.5],colors=[color],linewidths=.8)
            ax.set_title(f'z={k}');ax.axis('off')
        fig.suptitle(f'{case} | GT green / prediction red | native voxel axes')
        fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/f'{case}.png',dpi=130);plt.close(fig)
    summary=dict(cases=len(rows),dice_valid_cases=len(valid),dice_mean=float(np.mean(valid)),dice_median=float(np.median(valid)),
                 mean_absolute_volume_error_ml=float(np.mean([r['absolute_volume_error_ml'] for r in rows])),
                 empty_gt_cases=[r for r in rows if r['empty_gt']],random_qa=random_ids,worst_qa=worst,
                 qa_cases=qa_ids,visual_review_status='pending human/agent image inspection',checkpoint='checkpoint_final.pth',epoch=1000)
    (metrics/'fold0_validation_summary.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'docs/training_report.md').write_text('# Fold 0 training report\n\nTraining and final validation completed. Visual review pending.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nFull-volume Dice above is distinct from training patch pseudo-Dice. Both-empty Dice is N/A and excluded with denominator reported. This is single-fold internal validation, not external testing.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
