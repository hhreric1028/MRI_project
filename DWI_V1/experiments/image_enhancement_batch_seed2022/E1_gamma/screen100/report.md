# E1-100 result

Comparison is candidate minus B0-R2022-100.

| Metric | B0-100 | Candidate | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.784 | +0.019 |
| median_dice | 0.824 | 0.833 | +0.009 |
| lesion_recall | 0.594 | 0.634 | +0.040 |
| lesion_precision | 0.715 | 0.678 | -0.037 |
| fp_lesions_per_case | 2.140 | 2.720 | +0.580 |
| severe_under_count | 8.000 | 5.000 | -3.000 |
| volume_mae_ml | 4.521 | 4.858 | +0.336 |
| signed_volume_error_ml | -1.929 | -2.767 | -0.837 |
| small_mean_dice | 0.626 | 0.657 | +0.031 |
| small_recall | 0.733 | 0.800 | +0.067 |

Decision: **RECALL-PRECISION TRADE-OFF**

Core directions improved: 5/5. Safety guards: {'precision_drop_le_0.05': True, 'fp_per_case_increase_le_0.5': False, 'volume_mae_increase_le_0.5_ml': True, 'absolute_signed_bias_worsening_le_1_ml': True, 'training_numerically_stable': True}
