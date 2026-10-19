# ISLES2022 DWI-only Fold 0：100-epoch screening

本報告為 screening experiment，不代表 final performance。三個模型皆只執行原 1000-epoch PolyLR 軌跡的前 100 epochs。

| Metric | B0-100 | S1a-100 | Δ | L2a-100 | Δ |
|---|---:|---:|---:|---:|---:|
| Mean Dice | 0.765 | 0.780 | 0.015 | 0.752 | -0.013 |
| Median Dice | 0.824 | 0.832 | 0.008 | 0.825 | 0.001 |
| Lesion recall | 0.594 | 0.605 | 0.011 | 0.670 | 0.075 |
| Lesion precision | 0.715 | 0.732 | 0.017 | 0.667 | -0.048 |
| Severe under | 8 | 6 | -2 | 5 | -3 |
| Dice <0.5 | 4 | 5 | 1 | 7 | 3 |
| 1–5 mL Dice | 0.626 | 0.647 | 0.021 | 0.630 | 0.004 |
| 1–5 mL recall | 0.733 | 0.700 | -0.033 | 0.883 | 0.150 |
| Volume MAE (mL) | 4.52 | 4.81 | 0.29 | 5.42 | 0.90 |
| Signed volume error (mL) | -1.93 | -2.53 | -0.60 | -0.19 | 1.74 |

## Paired bootstrap 重點

### S1a_screen100_seed2022

- Δ Mean Dice：0.015，95% CI [0.00264160753868411, 0.028764407564859262]
- Δ Lesion recall：0.011，95% CI [-0.02117647058823524, 0.04239443244567155]
- Δ Severe-under rate：-0.040，95% CI [-0.12, 0.039999999999999994]
- 1–5 mL Δ Mean Dice：0.021，95% CI [-0.012015610746310303, 0.06273752281725543]
- 1–5 mL Δ Lesion recall：-0.033，95% CI [-0.10465116279069775, 0.04999999999999993]
- Decision：**PROMOTE TO 300 EPOCHS**
- Core signals：{'lesion_recall_improved': True, 'severe_under_count_reduced': True, 'small_1_5_dice_recall_consistent_improvement': False}
- Safety/trade-off guards：{'precision_drop_within_0.05': True, 'fp_lesions_per_case_increase_within_0.5': True, 'training_curve_and_numerics_normal': True}

### L2a_screen100_seed2022

- Δ Mean Dice：-0.013，95% CI [-0.0537656013678488, 0.019327873271756696]
- Δ Lesion recall：0.075，95% CI [0.03218857561721836, 0.1228580034423407]
- Δ Severe-under rate：-0.060，95% CI [-0.14, 0.020000000000000004]
- 1–5 mL Δ Mean Dice：0.004，95% CI [-0.052414518444509096, 0.08057810754230785]
- 1–5 mL Δ Lesion recall：0.150，95% CI [0.014925373134328401, 0.2962962962962963]
- Decision：**STOP**
- Core signals：{'lesion_recall_improved': True, 'severe_under_count_reduced': True, 'small_1_5_dice_recall_consistent_improvement': True}
- Safety/trade-off guards：{'precision_drop_within_0.05': True, 'fp_lesions_per_case_increase_within_0.5': False, 'training_curve_and_numerics_normal': True}

## Sampling exposure audit

- Overall 0.50−0.33：{'patches': 0, 'distinct_cases': 0, 'forced_foreground_rate': 0.125, 'lesion_containing_patch_rate': -0.0020000000000000018, 'foreground_voxel_fraction': 0.0010343831380208324, 'lesion_voxel_count': 635.5250000000001, 'lesion_coverage': -0.002382068799295256, 'small_lesion_containing_patches': -27}
- 1–5 mL 0.50−0.33：{'patches': -27, 'distinct_cases': 0, 'forced_foreground_rate': 0.17685377685377685, 'lesion_containing_patch_rate': 0.0, 'foreground_voxel_fraction': -3.7800345417533495e-06, 'lesion_voxel_count': -2.322453222453248, 'lesion_coverage': -0.0002829945226100472, 'small_lesion_containing_patches': -27}

## Decision policy

至少兩項核心方向一致改善，且 lesion precision 降幅不超過 0.05、FP lesions/case 增幅不超過 0.5、訓練與數值正常，才可進入 300-epoch 候選。
L2a 若 recall／severe-under／small-lesion 有改善但 overall Dice 尚未追上，標記為 PROMISING BUT NOT CONVERGED。
本流程不會自動啟動 300 或 1000 epochs。

## Limitations

Fold 0 only；single seed screening；same-seed 不等於 fully deterministic。100 epochs 為早期篩選狀態；bootstrap CI 僅反映目前 validation cohort 的 case-level uncertainty。
