# S1a 100→300 epoch trend analysis

Decision: **MIXED OR NO CONTINUED IMPROVEMENT**

Primary comparison is S1a-300 minus S1a-100 using paired case bootstrap, 10,000 iterations, seed 2022.

| Metric | S1a-100 | S1a-300 | Delta | 95% CI |
|---|---:|---:|---:|---|
| mean_dice | 0.780 | 0.759 | -0.020 | [-0.05845247087833296, 0.004301717193489814] |
| median_dice | 0.832 | 0.829 | -0.003 | [-0.026058410149844002, 0.015670067037092728] |
| lesion_recall | 0.605 | 0.659 | 0.053 | [0.026505625149605083, 0.07851239669421484] |
| lesion_precision | 0.732 | 0.681 | -0.051 | [-0.08395110328561772, -0.015174385167407222] |
| volume_mae_ml | 4.807 | 4.845 | 0.038 | [-0.4639032904277732, 0.4470261838637634] |
| signed_volume_error_ml | -2.525 | -2.745 | -0.220 | [-0.7642804164739415, 0.24252052234446048] |
| severe_under_rate | 0.120 | 0.120 | 0.000 | [-0.06, 0.06] |
| dice_below_05_rate | 0.100 | 0.120 | 0.020 | [-0.04000000000000001, 0.08000000000000002] |
| fp_lesions_per_case | 2.000 | 2.780 | 0.780 | [0.3400000000000001, 1.2600000000000002] |

## Signals

`{"mean_dice_increased": false, "lesion_recall_increased": true, "severe_under_rate_reduced": false, "volume_mae_reduced": false, "precision_drop_within_0.05": false, "fp_lesions_per_case_increase_within_0.5": false}`

Fold 0 only; single seed; controlled checkpoint continuation is not bitwise deterministic. The B0-100 comparison is contextual because there is no same-horizon B0-300 control.