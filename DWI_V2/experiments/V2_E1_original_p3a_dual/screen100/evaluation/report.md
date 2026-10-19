# V2-E1 screen100 failure-priority evaluation

Final classification: **RECALL-PRECISION TRADE-OFF**

| Metric | B0-R2022-100 | Candidate | Delta |
|---|---:|---:|---:|
| mean_dice | 0.7645 | 0.7709 | +0.0064 |
| median_dice | 0.8239 | 0.8291 | +0.0053 |
| lesion_recall | 0.5942 | 0.6452 | +0.0510 |
| lesion_precision | 0.7147 | 0.6554 | -0.0593 |
| fp_lesions_per_case | 2.1400 | 3.0600 | +0.9200 |
| severe_under_count | 8.0000 | 6.0000 | -2.0000 |
| fixed_worst5_mean_dice | 0.2814 | 0.2986 | +0.0172 |
| fixed_worst5_mean_voxel_recall | 0.1875 | 0.2042 | +0.0168 |
| coverage_cohort_mean_dice | 0.3866 | 0.4102 | +0.0236 |
| coverage_cohort_mean_voxel_recall | 0.2729 | 0.3007 | +0.0278 |
| volume_mae_ml | 4.5212 | 5.2559 | +0.7347 |

case_0140 persistent FN recovered: 0.104 mL (0.1%).

## Promotion guards

- severe_under_not_increased: PASS
- fixed_worst5_improved: PASS
- anchor_or_cohort_coverage_improved: PASS
- overall_lesion_recall_not_decreased: PASS
- precision_drop_within_0p05: FAIL
- fp_per_case_increase_within_0p5: FAIL
- volume_mae_not_materially_worse: FAIL
- numerically_stable: FAIL

No 1000-epoch training is authorized by this evaluation.
