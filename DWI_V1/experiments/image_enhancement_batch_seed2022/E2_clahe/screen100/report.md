# E2-100 result

Comparison is candidate minus B0-R2022-100.

| Metric | B0-100 | Candidate | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.750 | -0.015 |
| median_dice | 0.824 | 0.802 | -0.022 |
| lesion_recall | 0.594 | 0.621 | +0.027 |
| lesion_precision | 0.715 | 0.631 | -0.084 |
| fp_lesions_per_case | 2.140 | 3.280 | +1.140 |
| severe_under_count | 8.000 | 10.000 | +2.000 |
| volume_mae_ml | 4.521 | 5.765 | +1.243 |
| signed_volume_error_ml | -1.929 | -4.264 | -2.334 |
| small_mean_dice | 0.626 | 0.610 | -0.016 |
| small_recall | 0.733 | 0.800 | +0.067 |

Decision: **RECALL-PRECISION TRADE-OFF**

Core directions improved: 2/5. Safety guards: {'precision_drop_le_0.05': False, 'fp_per_case_increase_le_0.5': False, 'volume_mae_increase_le_0.5_ml': False, 'absolute_signed_bias_worsening_le_1_ml': False, 'training_numerically_stable': True}
