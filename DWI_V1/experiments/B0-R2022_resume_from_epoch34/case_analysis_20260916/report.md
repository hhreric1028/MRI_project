# B0-R2022 Fold 0 新訓練結果與 Dice 偏差病例分析

## 1. 訓練完成狀態

- 完成 epochs：1000（log 中 final epoch index = 999）
- Final train loss：-0.892
- Final validation loss：-0.770
- Final patch-based pseudo-Dice：0.862
- Final learning rate：0.00002
- 評估 checkpoint：checkpoint_final.pth
- 注意：patch-based pseudo-Dice 與完整 3D volume Dice 定義不同，不直接互相比較。

## 2. Overall validation result

- Validation cases：50；有效 Dice：49。
- Mean Dice：0.776；median：0.835；SD：0.191。
- Dice ≥0.8：32/49；Dice ≥0.7：40/49。
- Dice <0.5：5/49；Dice <0.3：3/49。
- Lesion recall：0.645；lesion precision：0.717；FP lesions/case：2.300。
- Volume MAE：4.78 mL；median absolute error：1.44 mL。
- Mean signed volume error：-2.23 mL，整體呈輕度低估方向。
- HD95：mean 9.72 mm；median 4.47 mm。
- case_0151 為 empty GT + empty prediction，保留於病例數及 volume 統計，但排除 Dice mean。

## 3. Dice 最低 10 例

| Case | Dice | Δ vs B0 | Δ vs P1 | GT mL | Pred mL | Pred/GT | Lesion FN | Category |
|---|---|---|---|---|---|---|---|---|
| case_0182 | 0.108 | -0.097 | -0.423 | 1.63 | 0.09 | 0.057 | 2 | absolute_low_dice_outlier;severe_under_segmentation;detected_but_coverage_insufficient;small_lesion_instability;large_decline_vs_historical_B0;large_decline_vs_P1 |
| case_0068 | 0.171 | 0.009 | 0.045 | 3.69 | 0.34 | 0.093 | 1 | absolute_low_dice_outlier;severe_under_segmentation;detected_but_coverage_insufficient;small_lesion_instability |
| case_0181 | 0.274 | -0.021 | -0.065 | 16.21 | 2.87 | 0.177 | 2 | absolute_low_dice_outlier;severe_under_segmentation;detected_but_coverage_insufficient;large_decline_vs_P1 |
| case_0212 | 0.385 | -0.123 | -0.030 | 1.10 | 0.35 | 0.319 | 2 | absolute_low_dice_outlier;severe_under_segmentation;detected_but_coverage_insufficient;small_lesion_instability;large_decline_vs_historical_B0 |
| case_0140 | 0.487 | 0.016 | 0.031 | 141.37 | 53.58 | 0.379 | 0 | absolute_low_dice_outlier;severe_under_segmentation;detected_but_coverage_insufficient;large_lesion_coverage_failure |
| case_0160 | 0.524 | -0.043 | 0.043 | 4.43 | 2.17 | 0.489 | 1 | absolute_low_dice_outlier;detected_but_coverage_insufficient;small_lesion_instability |
| case_0186 | 0.619 | 0.009 | -0.018 | 11.13 | 5.21 | 0.468 | 31 | detected_but_coverage_insufficient |
| case_0072 | 0.682 | -0.006 | 0.003 | 8.13 | 12.10 | 1.489 | 2 | volume_over_segmentation |
| case_0137 | 0.691 | 0.006 | 0.016 | 2.26 | 1.42 | 0.626 | 0 | no_flag |
| case_0146 | 0.710 | 0.013 | 0.039 | 21.02 | 11.86 | 0.565 | 2 | no_flag |

Tukey lower fence = 0.591；低於此門檻者視為本 cohort 的低 Dice 離群病例。

## 4. 模型間 Dice 變化

差值均為 B0-R2022 − comparator。歷史 B0 seed 未確認；P1 同時更改 spacing，因此僅作描述性病例比較。

### 相較歷史 B0

| 最大下降 | ΔDice | 最大改善 | ΔDice |
|---|---|---|---|
| case_0212 | -0.123 | case_0135 | 0.098 |
| case_0182 | -0.097 | case_0051 | 0.033 |
| case_0160 | -0.043 | case_0197 | 0.031 |
| case_0069 | -0.042 | case_0196 | 0.030 |
| case_0093 | -0.035 | case_0048 | 0.026 |

### 相較 P1

| 最大下降 | ΔDice | 最大改善 | ΔDice |
|---|---|---|---|
| case_0182 | -0.423 | case_0135 | 0.264 |
| case_0181 | -0.065 | case_0118 | 0.129 |
| case_0053 | -0.050 | case_0010 | 0.120 |
| case_0093 | -0.044 | case_0069 | 0.093 |
| case_0188 | -0.031 | case_0034 | 0.082 |

## 5. Paired case bootstrap

10,000 iterations，seed=2022；相同 sampled case IDs 套用兩模型，lesion TP/FP/FN 於每次抽樣後重新聚合。

| Metric | Δ vs B0 | 95% CI | Δ vs P1 | 95% CI |
|---|---|---|---|---|
| Δ Mean Dice | -0.003 | [-0.011, 0.006] | 0.011 | [-0.014, 0.032] |
| Δ Median Dice | 0.006 | [-0.012, 0.021] | 0.005 | [-0.014, 0.047] |
| Δ Lesion recall | -0.002 | [-0.026, 0.025] | 0.047 | [0.008, 0.084] |
| Δ Lesion precision | -0.004 | [-0.032, 0.023] | -0.035 | [-0.069, -0.001] |
| Δ Volume MAE (mL) | 0.27 | [-0.17, 0.85] | -0.07 | [-0.47, 0.33] |
| Δ Signed volume error (mL) | 0.39 | [-0.05, 0.97] | 0.57 | [0.23, 0.96] |
| Δ Severe-under rate | 0.020 | [0.000, 0.060] | -0.040 | [-0.100, 0.000] |
| Δ Dice <0.5 rate | 0.020 | [0.000, 0.060] | 0.000 | [-0.060, 0.060] |

## 6. Lesion-size subgroup

| GT group | Cases | Mean Dice | Median Dice | Dice SD | Volume MAE | Lesion recall | Dice <0.5 |
|---|---|---|---|---|---|---|---|
| <1 mL | 4 | 0.807 | 0.816 | 0.015 | 0.09 | 1.000 | 0 |
| 1–5 mL | 12 | 0.644 | 0.773 | 0.269 | 0.97 | 0.800 | 3 |
| 5–20 mL | 21 | 0.794 | 0.827 | 0.138 | 2.52 | 0.637 | 1 |
| 20–50 mL | 5 | 0.872 | 0.884 | 0.087 | 3.86 | 0.811 | 0 |
| ≥50 mL | 8 | 0.852 | 0.895 | 0.142 | 19.37 | 0.528 | 1 |

## 7. Failure pattern 分類討論

- Severe under-segmentation：case_0068, case_0140, case_0181, case_0182, case_0212。這些病例 Dice <0.5 且 prediction volume 不到 GT 的一半，主要是病灶覆蓋不足，而非單純假陽性造成低 Dice。
- 找到病灶但 coverage 不足：case_0068, case_0140, case_0160, case_0181, case_0182, case_0186, case_0212。其中 case_0160、case_0186 的 Dice 雖高於 0.5，體積比例仍低於 0.5。
- Small-lesion instability：case_0010, case_0034, case_0068, case_0094, case_0135, case_0160, case_0182, case_0212。1–5 mL 組 mean Dice 0.644、SD 0.269，呈現明顯病例間差異；但 case_0135 有明顯改善，因此不能推論所有小病灶都必然失敗。
- Large-lesion coverage failure：case_0140。case_0140 的大型病灶仍有廣泛漏覆蓋，顯示問題不限於小病灶。
- Volume over-segmentation：case_0053, case_0072, case_0093, case_0105, case_0201。這類病例需與 lesion FP 及 voxel precision 一併判讀；部分病例即使 Dice 尚可，仍可能有顯著額外體積。
- case_0182 是最需優先檢查的病例：不僅為最低 Dice，且相較 B0 與 P1 均下降，Pred/GT 僅約 0.057。
- case_0135 與 case_0118 顯示 P1 的部分低分並非固定不可恢復；但由於比較同時包含 spacing 等差異，不能將改善歸因於單一因素。

所有視覺圖均使用相同 DWI slice，第一列為 GT 最大面積切片，第二列為三模型差異最大的切片；cyan 為 GT boundary。component coverage 為描述性 overlap 分析，不等同 evaluation v2 的 one-to-one lesion matching。

## 8. 簡報版重點

- Mean Dice：0.776
- Median Dice：0.835
- Dice ≥0.8：32/49
- Lesion recall／precision：0.645／0.717
- Volume MAE：4.78 mL
- Lowest Dice：case_0182，0.108

B0-R2022 在 fold 0 的整體 3D Dice 表現尚可，但病例間差異明顯；主要失敗型態為少數病例的嚴重低估與病灶覆蓋不足，且同時出現在小型及大型病灶。後續應優先以 lesion recall、Pred/GT ratio 與 failure-case 影像共同驗證改善策略，避免只依 overall mean Dice 判定模型優劣。

## 9. Limitations

Fold 0 only; single-seed screening. The resumed run is a training-state continuation, not a bitwise deterministic continuation. Historical B0 has an unconfirmed seed, and P1 also changes spacing, so these comparisons are descriptive rather than single-variable causal comparisons. Paired bootstrap confidence intervals describe case-level uncertainty within this validation cohort and do not establish cross-fold, cross-center, or cross-seed generalization.

本次共輸出 22 個去重後的重點病例圖。
