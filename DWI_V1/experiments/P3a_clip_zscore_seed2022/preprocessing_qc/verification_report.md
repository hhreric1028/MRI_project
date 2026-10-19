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
