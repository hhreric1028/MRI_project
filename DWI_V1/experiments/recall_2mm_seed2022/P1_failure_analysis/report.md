# Paired comparison
Baseline: C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\B0_baseline\evaluation_v2
Candidate: C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\P1_spacing_1p5mm\run_seed2022_train_r2\metrics
Differences are candidate − baseline. Percentile 95% CI, paired case bootstrap, 10,000 iterations, seed 2022.

| Metric | Baseline | Candidate | Delta | 95% CI |
|---|---:|---:|---:|---|
| mean_dice | 0.778 | 0.765 | -0.013 | [-0.031823451141108856, 0.007653801057151929] |
| median_dice | 0.829 | 0.831 | 0.002 | [-0.04156661441823739, 0.014752344287408903] |
| lesion_recall | 0.647 | 0.599 | -0.049 | [-0.0813403408306019, -0.015914066776135712] |
| lesion_precision | 0.721 | 0.752 | 0.031 | [0.001961237795325274, 0.058404967228301435] |
| volume_mae_ml | 4.509 | 4.853 | 0.344 | [-0.03437084715707792, 0.7757430268561536] |
| signed_volume_error_ml | -2.620 | -2.796 | -0.176 | [-0.5889546163147981, 0.24001936799382112] |
| severe_under_count | 6.000 | 9.000 | 3.000 | [0.0, 7.0] |
| severe_under_rate | 0.120 | 0.180 | 0.060 | [0.0, 0.13999999999999999] |
| dice_below_05_rate | 0.080 | 0.100 | 0.020 | [-0.04000000000000001, 0.09999999999999999] |
| catastrophic_miss_rate | 0.000 | 0.000 | 0.000 | [0.0, 0.0] |
| fp_lesions_per_case | 2.260 | 1.780 | -0.480 | [-0.7999999999999998, -0.17999999999999994] |
| hd95_finite_mean_mm | 9.150 | 9.954 | 0.805 | [-0.6674873334743253, 2.7361541957697453] |
| hd95_infinite_rate | 0.000 | 0.000 | 0.000 | [0.0, 0.0] |

## Decision guard
Prioritize catastrophic miss and severe under-segmentation; then 1–5 mL recall/Dice; then overall Dice. A recall increase with increased FP is a tradeoff, not an automatic winner. No automatic combinations, additional seeds or P2.

Fold 0 only; single seed screening. Controlled same-seed comparison is not fully deterministic. Actual sampling exposure change must be established by the audit. Case bootstrap CIs describe uncertainty within this validation cohort, not generalization across folds, centers or training seeds.