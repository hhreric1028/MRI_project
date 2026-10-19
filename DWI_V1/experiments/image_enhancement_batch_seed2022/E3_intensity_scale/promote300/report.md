# E3-300 result

Comparison is candidate minus B0-R2022-100. B0-300 is unavailable, so this is contextual rather than horizon matched.

| Metric | B0-100 | Candidate | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.774 | +0.010 |
| median_dice | 0.824 | 0.830 | +0.006 |
| lesion_recall | 0.594 | 0.654 | +0.060 |
| lesion_precision | 0.715 | 0.672 | -0.043 |
| fp_lesions_per_case | 2.140 | 2.880 | +0.740 |
| severe_under_count | 8.000 | 5.000 | -3.000 |
| volume_mae_ml | 4.521 | 4.327 | -0.194 |
| signed_volume_error_ml | -1.929 | -3.092 | -1.163 |
| small_mean_dice | 0.626 | 0.693 | +0.067 |
| small_recall | 0.733 | 0.833 | +0.100 |

Decision: **RECALL-PRECISION TRADE-OFF**

Core directions improved: 5/5. Safety guards: {'precision_drop_le_0.05': True, 'fp_per_case_increase_le_0.5': False, 'volume_mae_increase_le_0.5_ml': True, 'absolute_signed_bias_worsening_le_1_ml': False, 'training_numerically_stable': True}
