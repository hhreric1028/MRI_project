# E3-100 result

Comparison is candidate minus B0-R2022-100.

| Metric | B0-100 | Candidate | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.779 | +0.015 |
| median_dice | 0.824 | 0.832 | +0.008 |
| lesion_recall | 0.594 | 0.605 | +0.011 |
| lesion_precision | 0.715 | 0.713 | -0.002 |
| fp_lesions_per_case | 2.140 | 2.200 | +0.060 |
| severe_under_count | 8.000 | 6.000 | -2.000 |
| volume_mae_ml | 4.521 | 4.465 | -0.056 |
| signed_volume_error_ml | -1.929 | -1.711 | +0.218 |
| small_mean_dice | 0.626 | 0.662 | +0.036 |
| small_recall | 0.733 | 0.783 | +0.050 |

Decision: **PROMOTE**

Core directions improved: 5/5. Safety guards: {'precision_drop_le_0.05': True, 'fp_per_case_increase_le_0.5': True, 'volume_mae_increase_le_0.5_ml': True, 'absolute_signed_bias_worsening_le_1_ml': True, 'training_numerically_stable': True}
