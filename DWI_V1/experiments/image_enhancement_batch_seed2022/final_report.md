# DWI image enhancement controlled ablation — final report

All runs stopped at or before 300 epochs. No 1000-epoch training, enhancement combination, sampling/loss/spacing/modality/architecture change was started.

## Final comparison

| Model | Enhancement | 100 Dice | 100 Recall | 100 Precision | 100 FP/case | 100 Severe Under | 300 Dice | 300 Recall | 300 Precision | 300 FP/case | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | Original DWI + gamma-enhanced DWI | 0.784 | 0.634 | 0.678 | 2.720 | 5 | — | — | — | — | RECALL-PRECISION TRADE-OFF |
| E2 | Original DWI + axial slice-wise CLAHE DWI | 0.750 | 0.621 | 0.631 | 3.280 | 10 | — | — | — | — | RECALL-PRECISION TRADE-OFF |
| E3 | Original DWI + mildly scaled foreground-z-score DWI | 0.779 | 0.605 | 0.713 | 2.200 | 6 | 0.774 | 0.654 | 0.672 | 2.880 | RECALL-PRECISION TRADE-OFF |

## 1–5 mL subgroup

| Model | 100 Mean Dice | 100 Median Dice | 100 Recall | 100 Precision | 100 FN | 100 FP | 100 Volume MAE | 300 Mean Dice | 300 Recall | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | 0.657 | 0.779 | 0.800 | 0.774 | 12 | 14 | 0.932 | — | — | RECALL-PRECISION TRADE-OFF |
| E2 | 0.610 | 0.746 | 0.800 | 0.842 | 12 | 9 | 1.021 | — | — | RECALL-PRECISION TRADE-OFF |
| E3 | 0.662 | 0.760 | 0.783 | 0.783 | 13 | 13 | 0.954 | 0.693 | 0.833 | RECALL-PRECISION TRADE-OFF |

## Research questions

1. **Does enhancement improve lesion detectability?** Best observed candidate: E3; lesion recall delta versus B0-100 = +0.060. Interpret with its paired bootstrap interval.
2. **Recurring severe under-segmentation?** Best-candidate count changed from 8 to 5; fixed-case details are in each case-analysis table and panel set.
3. **1–5 mL lesions?** Mean Dice delta = +0.067; recall delta = +0.100.
4. **Only recall through more FP?** The best candidate did not demonstrate a guard-passing balanced sensitivity gain.
5. **More balanced than sampling/loss interventions?** For the selected best representation, S1a-100: no under the joint recall/precision/FP definition; L2a-100: no under the joint recall/precision/FP definition. This definition requires at least as much recall gain, no worse precision change, and no greater FP/case change versus B0-100.
6. **Improved failure phenotypes?** Fixed failures with Dice or lesion-FN improvement: ['case_0068', 'case_0181', 'case_0182', 'case_0186']. The largest Dice/FN improvement cohorts and images are in `E3/case_analysis/promote300`.
7. **Unresolved cases?** Fixed failures without Dice or FN improvement: ['case_0140']. Top-five Dice declines are retained in the same tracked-case table.
8. **Enough evidence for 1000 epochs?** Final labels: {'E1_gamma': 'RECALL-PRECISION TRADE-OFF', 'E2_clahe': 'RECALL-PRECISION TRADE-OFF', 'E3_intensity_scale': 'RECALL-PRECISION TRADE-OFF'}. This workflow does not start 1000 epochs under any label.

## QC parameter freeze

Gamma=1.2; CLAHE clip=0.02, kernel=[16, 16], axial slice-wise; intensity scale a=0.76, b=0.0.

## Context: prior interventions

- S1a-100: Dice 0.780, recall 0.605, precision 0.732, FP/case 2.000, severe-under 6.
- L2a-100: Dice 0.752, recall 0.670, precision 0.667, FP/case 3.020, severe-under 5.
- P3a-100: Dice 0.784, recall 0.639, precision 0.689, FP/case 2.600, severe-under 5.
- P3a-300: Dice 0.781, recall 0.623, precision 0.694, FP/case 2.480, severe-under 5.

## Limitations

Fold 0 and one seed only. Same-seed GPU training is not bitwise deterministic. Paired case bootstrap quantifies this validation cohort, not cross-fold, center or seed generalization. B0-R2022-300 is unavailable, so 300-epoch comparisons to B0-100 are contextual rather than horizon matched.
