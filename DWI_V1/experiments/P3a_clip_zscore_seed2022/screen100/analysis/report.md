# P3a-100 result

Primary comparison: matched B0-R2022-100 vs P3a-100; candidate minus baseline.

| Metric | B0-100 | P3a-100 | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.784 | 0.019 |
| median_dice | 0.824 | 0.832 | 0.009 |
| lesion_recall | 0.594 | 0.639 | 0.044 |
| lesion_precision | 0.715 | 0.689 | -0.026 |
| fp_lesions_per_case | 2.140 | 2.600 | 0.460 |
| severe_under_count | 8.000 | 5.000 | -3.000 |
| volume_mae_ml | 4.521 | 4.386 | -0.135 |
| small_mean_dice | 0.626 | 0.662 | 0.035 |
| small_recall | 0.733 | 0.717 | -0.017 |

## Promotion decision: PROMOTE TO 300

Core signals: 3/4. Guards: {'precision_drop_within_0.05': True, 'fp_per_case_increase_within_0.5': True, 'volume_mae_increase_within_0.5_ml': True, 'training_and_numerics_normal': True}

Fold 0 only; single-seed screening; same seed is not fully deterministic. Paired bootstrap CIs describe this validation cohort, not cross-fold generalization.
