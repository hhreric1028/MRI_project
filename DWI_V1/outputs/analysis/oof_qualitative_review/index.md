# OOF qualitative review index

Reviewed 31 unique cases from the frozen 250-case OOF set. Probability backfill: 0 cases.

## Selection coverage

| Selection | Count | Cases |
|---|---:|---|
| A_good | 8 | case_0066, case_0176, case_0086, case_0061, case_0111, case_0067, case_0246, case_0237 |
| B_small_lesion_miss | 5 | case_0144, case_0174, case_0233, case_0139, case_0136 |
| C_multifocal_miss | 5 | case_0007, case_0229, case_0248, case_0230, case_0166 |
| D_large_partial_coverage | 6 | case_0023, case_0083, case_0092, case_0116, case_0146, case_0229 |
| E_severe_undersegmentation | 10 | case_0023, case_0083, case_0092, case_0116, case_0146, case_0229, case_0144, case_0007, case_0068, case_0027 |
| F_extreme_low_dice | 5 | case_0144, case_0150, case_0174, case_0233, case_0046 |
| G_fp_heavy | 5 | case_0116, case_0023, case_0140, case_0036, case_0246 |
| H_fold4_worst | 5 | case_0144, case_0139, case_0116, case_0130, case_0030 |

## Reviewed cases

| Case | Fold | Selection labels | Taxonomy | Montage |
|---|---:|---|---|---|
| case_0007 | 3 | C_multifocal_miss;E_severe_undersegmentation | multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0007_montage.png) |
| case_0023 | 3 | D_large_partial_coverage;E_severe_undersegmentation;G_fp_heavy | multifocal_component_miss;large_partial_coverage;boundary_shrinkage;low_confidence_lesion;fp_fragmentation | [open](case_montages/case_0023_montage.png) |
| case_0027 | 1 | E_severe_undersegmentation | multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0027_montage.png) |
| case_0030 | 4 | H_fold4_worst | small_lesion_miss;multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0030_montage.png) |
| case_0036 | 2 | G_fp_heavy | fp_fragmentation | [open](case_montages/case_0036_montage.png) |
| case_0046 | 2 | F_extreme_low_dice | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0046_montage.png) |
| case_0061 | 2 | A_good | other | [open](case_montages/case_0061_montage.png) |
| case_0066 | 3 | A_good | other | [open](case_montages/case_0066_montage.png) |
| case_0067 | 1 | A_good | other | [open](case_montages/case_0067_montage.png) |
| case_0068 | 0 | E_severe_undersegmentation | multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0068_montage.png) |
| case_0083 | 3 | D_large_partial_coverage;E_severe_undersegmentation | large_partial_coverage;low_confidence_lesion;fp_fragmentation | [open](case_montages/case_0083_montage.png) |
| case_0086 | 0 | A_good | other | [open](case_montages/case_0086_montage.png) |
| case_0092 | 4 | D_large_partial_coverage;E_severe_undersegmentation | large_partial_coverage;boundary_shrinkage;low_confidence_lesion | [open](case_montages/case_0092_montage.png) |
| case_0111 | 2 | A_good | other | [open](case_montages/case_0111_montage.png) |
| case_0116 | 4 | D_large_partial_coverage;E_severe_undersegmentation;G_fp_heavy;H_fold4_worst | multifocal_component_miss;large_partial_coverage;low_confidence_lesion | [open](case_montages/case_0116_montage.png) |
| case_0130 | 4 | H_fold4_worst | multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0130_montage.png) |
| case_0136 | 1 | B_small_lesion_miss | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0136_montage.png) |
| case_0139 | 4 | B_small_lesion_miss;H_fold4_worst | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0139_montage.png) |
| case_0140 | 0 | G_fp_heavy | boundary_shrinkage;low_confidence_lesion;fp_fragmentation | [open](case_montages/case_0140_montage.png) |
| case_0144 | 4 | B_small_lesion_miss;E_severe_undersegmentation;F_extreme_low_dice;H_fold4_worst | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0144_montage.png) |
| case_0146 | 0 | D_large_partial_coverage;E_severe_undersegmentation | multifocal_component_miss;large_partial_coverage;boundary_shrinkage;low_confidence_lesion;fp_fragmentation | [open](case_montages/case_0146_montage.png) |
| case_0150 | 3 | F_extreme_low_dice | other | [open](case_montages/case_0150_montage.png) |
| case_0166 | 1 | C_multifocal_miss | multifocal_component_miss | [open](case_montages/case_0166_montage.png) |
| case_0174 | 1 | B_small_lesion_miss;F_extreme_low_dice | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0174_montage.png) |
| case_0176 | 1 | A_good | multifocal_component_miss | [open](case_montages/case_0176_montage.png) |
| case_0229 | 1 | C_multifocal_miss;D_large_partial_coverage;E_severe_undersegmentation | multifocal_component_miss;large_partial_coverage;boundary_shrinkage;low_confidence_lesion | [open](case_montages/case_0229_montage.png) |
| case_0230 | 4 | C_multifocal_miss | multifocal_component_miss;boundary_shrinkage | [open](case_montages/case_0230_montage.png) |
| case_0233 | 2 | B_small_lesion_miss;F_extreme_low_dice | small_lesion_miss;low_confidence_lesion | [open](case_montages/case_0233_montage.png) |
| case_0237 | 3 | A_good | other | [open](case_montages/case_0237_montage.png) |
| case_0246 | 2 | A_good;G_fp_heavy | multifocal_component_miss;fp_fragmentation | [open](case_montages/case_0246_montage.png) |
| case_0248 | 4 | C_multifocal_miss | multifocal_component_miss;low_confidence_lesion | [open](case_montages/case_0248_montage.png) |

## Good-vs-bad comparisons

- large_lesion pair 1: case_0066 vs case_0083 — [open](comparison_montages/large_lesion_pair_1.png)
- large_lesion pair 2: case_0176 vs case_0116 — [open](comparison_montages/large_lesion_pair_2.png)
- large_lesion pair 3: case_0086 vs case_0092 — [open](comparison_montages/large_lesion_pair_3.png)
- small_lesion pair 1: case_0061 vs case_0144 — [open](comparison_montages/small_lesion_pair_1.png)
- small_lesion pair 2: case_0111 vs case_0174 — [open](comparison_montages/small_lesion_pair_2.png)
- small_lesion pair 3: case_0067 vs case_0233 — [open](comparison_montages/small_lesion_pair_3.png)
- multifocal_lesion pair 1: case_0176 vs case_0007 — [open](comparison_montages/multifocal_lesion_pair_1.png)
- multifocal_lesion pair 2: case_0246 vs case_0229 — [open](comparison_montages/multifocal_lesion_pair_2.png)
- multifocal_lesion pair 3: case_0237 vs case_0248 — [open](comparison_montages/multifocal_lesion_pair_3.png)

## Failure-category montages

- small_lesion_miss: [open](category_montages/small_lesion_miss.png)
- multifocal_component_miss: [open](category_montages/multifocal_component_miss.png)
- large_partial_coverage: [open](category_montages/large_partial_coverage.png)
- boundary_shrinkage: [open](category_montages/boundary_shrinkage.png)
- low_confidence_lesion: [open](category_montages/low_confidence_lesion.png)
- fp_fragmentation: [open](category_montages/fp_fragmentation.png)
- mixed: [open](category_montages/mixed.png)
- other: [open](category_montages/other.png)
