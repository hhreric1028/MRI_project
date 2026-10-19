# P3a DWI intensity preprocessing experiment

## P3a preprocessing verification

# P3a preprocessing verification

## Baseline normalization

- Dataset001 uses `ZScoreNormalization`.
- Statistics are calculated per individual case.
- `use_mask_for_norm=True`; the normalization mask is the filled nonzero foreground generated during cropping.
- Baseline DWI z-score normalization does not perform percentile clipping.
- The P0.5/P99.5 values in `dataset_fingerprint.json` are dataset-level descriptive properties and are not used by `ZScoreNormalization`.
- Operation order is transpose → nonzero crop → normalization → resampling.

## P3a controlled change

Dataset013 preserves case IDs, labels, geometry and the baseline nnUNet configuration. Before the same foreground z-score normalization, each individual DWI volume is clipped to its own foreground P0.5–P99.5 range. Background remains zero.

The generated plans match Dataset001 for the controlled fields:

- spacing: 2×2×2 mm
- patch: 80×96×80
- batch: 8
- architecture: PlainConvUNet
- normalization: ZScoreNormalization with foreground mask

Fold 0 `splits_final.json` is byte-identical to Dataset001.

## QC

- 250/250 cases were generated.
- 250/250 labels are byte-identical to Dataset001.
- All normalized foreground arrays are finite with mean approximately 0 and standard deviation approximately 1 before resampling.
- Thirteen fixed QC cases include five general cases, three 1–5 mL cases and five known failure cases.
- Sampled images preserve brain structure, lesion location and background masking.

## Recorded risk

DWI lesions are often among the brightest foreground voxels. Some lesions therefore have a substantial fraction at the P99.5 ceiling: the maximum lesion high-saturation fraction across the dataset is approximately 80.9%, and case_0128 is approximately 47.7%. The sampled images do not show gross anatomical distortion, but clipping may compress within-lesion contrast. This is treated as an explicit P3a risk to be tested by segmentation performance, not assumed to be beneficial.

## Matched baseline availability

- B0-R2022-100: available and used as the primary 100-epoch reference.
- B0-R2022-300: currently unavailable.
- Existing B0 checkpoints at epochs 894, 950 and 1000 are not valid 300-epoch matched references.

If P3a reaches 300 epochs, the absence of B0-300 must remain a limitation; B0-1000 must not be presented as a matched comparison.


## P3a-100 result and promotion decision

# P3a-100 result

Primary comparison: matched B0-R2022-100 vs P3a-100; candidate minus baseline.

| Metric | B0-100 | P3a-100 | Delta |
|---|---:|---:|---:|
| mean_dice | 0.765 | 0.784 | 0.019 |
| median_dice | 0.824 | 0.832 | 0.009 |
| lesion_recall | 0.594 | 0.639 | 0.044 |
| lesion_precision | 0.715 | 0.689 | -0.026 |
| fp_lesions_per_case | 2.140 | 2.600 | 0.460 |
| severe_under_count | 8.000 | 5.000 | -3.000 |
| volume_mae_ml | 4.521 | 4.386 | -0.135 |
| small_mean_dice | 0.626 | 0.662 | 0.035 |
| small_recall | 0.733 | 0.717 | -0.017 |

## Promotion decision: PROMOTE TO 300

Core signals: 3/4. Guards: {'precision_drop_within_0.05': True, 'fp_per_case_increase_within_0.5': True, 'volume_mae_increase_within_0.5_ml': True, 'training_and_numerics_normal': True}

Fold 0 only; single-seed screening; same seed is not fully deterministic. Paired bootstrap CIs describe this validation cohort, not cross-fold generalization.


## P3a-300 result and 100-to-300 trend

# P3a-300 result

Primary trend comparison: P3a-100 vs P3a-300.
B0-R2022-300 is unavailable; B0-100 is contextual only, not a horizon-matched primary comparison.

| Metric | P3a-100 | P3a-300 | Delta |
|---|---:|---:|---:|
| mean_dice | 0.784 | 0.781 | -0.003 |
| median_dice | 0.832 | 0.833 | 0.000 |
| lesion_recall | 0.639 | 0.623 | -0.016 |
| lesion_precision | 0.689 | 0.694 | 0.005 |
| fp_lesions_per_case | 2.600 | 2.480 | -0.120 |
| severe_under_count | 5.000 | 5.000 | 0.000 |
| volume_mae_ml | 4.386 | 4.457 | 0.070 |
| small_mean_dice | 0.662 | 0.667 | 0.005 |
| small_recall | 0.717 | 0.733 | 0.017 |

## Final decision: STOP

No 1000-epoch run was started.


## Failure-case analysis

Per-case tables and fixed-slice Original DWI / P3a DWI / GT / B0 / P3a panels are saved under each completed stage's analysis/case_analysis directory.

## Final decision

STOP

No 1000-epoch run was started.

Limitations: Fold 0 only; single-seed screening; same seed is not fully deterministic. Paired case-bootstrap intervals reflect uncertainty in the current validation cohort, not cross-fold, center, or seed generalization. No matched B0-R2022 epoch-300 checkpoint is available, so any 300-epoch contextual comparison to B0-100 is not horizon matched.
