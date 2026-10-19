# ISLES2022 DWI-only Preprocessing Report

Completed 2026-09-08. Official planning/preprocessing exit code: 0. No training executed.

## Environment

```json
{
  "python": "3.11.16",
  "pytorch": "2.8.0+cu128",
  "cuda_runtime": "12.8",
  "nnunetv2": "2.8.1",
  "gpu": "NVIDIA GeForce RTX 5080",
  "conda_environment": "MRI_DWI"
}
```

## Dataset

Dataset ID: 1. Dataset name: `Dataset001_ISLES2022_DWI`. Expected training cases: 250.

One input channel: channel 0 = DWI. Two output classes: background 0 and ischemic lesion 1. ADC/FLAIR were not inputs.

## nnUNet Paths

- nnUNet_raw: `DWI_V1/datasets/ISLES2022/nnUNet_raw`
- nnUNet_preprocessed: `DWI_V1/datasets/ISLES2022/processed`
- nnUNet_results: `DWI_V1/outputs/models`

All paths were checked before execution. Exact official command (default parameters):

```powershell
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

## Dataset Fingerprint

Fingerprint: `DWI_V1/datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/dataset_fingerprint.json`

Contains 250 spacing records and 250 cropped-shape records.

Median relative size after cropping: 0.41914278819002393.

## Original Image Statistics

- True uncropped original NIfTI median shape (X,Y,Z): `[112.0, 112.0, 72.0]`.
- True uncropped original median shape in reader array order (Z,Y,X): `[72.0, 112.0, 112.0]`.
- Original median spacing (X,Y,Z), mm: `[2.0, 2.0, 2.0]`.
- Plans original_median_shape_after_transp: `[65, 84, 70]` (Z,Y,X).
- Plans original_median_spacing_after_transp: `[2.0, 2.0, 2.0]` mm.

**The plans field named original_median_shape_after_transp is computed from fingerprint shapes_after_crop. It is a cropped shape, not the original full NIfTI shape.**

SimpleITK returns Z,Y,X arrays. transpose_forward = `[0, 1, 2]`; transpose_backward = `[0, 1, 2]`. NIfTI X,Y,Z and array Z,Y,X must not be compared as though they were the same ordering.

## Target Spacing

3d_fullres: **[2, 2, 2] mm** in Z,Y,X order. 2d: in-plane [2, 2] mm, preserving each case’s original slice spacing.

## Planning Result

Plans: `DWI_V1/datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/nnUNetPlans.json`

Dataset metadata: `DWI_V1/datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/dataset.json`

Official ExperimentPlanner and DefaultPreprocessor, without source, architecture or normalization overrides.

Network: dynamic_network_architectures.architectures.unet.PlainConvUNet. Five stages; two convolutions per encoder/decoder stage.

3D features per stage: [32, 64, 128, 256, 320]; Conv3d, InstanceNorm3d, LeakyReLU. 2D features: [32, 64, 128, 256, 512].

Data resampling uses order 3; segmentation uses the official label-aware is_seg=True, order 1 path; order_z=0 and force_separate_z=None. The same target shape and spacing are passed for image and segmentation.

## Selected Configuration

Generated and preprocessed: **2d, 3d_fullres**. Neither 3d_lowres nor 3d_cascade_fullres was generated. The CLI’s lowres skip message is expected.

## Patch Size

- 3d_fullres: **[80, 96, 80]** (Z,Y,X).
- 2d: [96, 80] (Y,X).

## Batch Size

- 3d_fullres: **8**.
- 2d: 434.

These are the unchanged official plan values. GPU memory target was not overridden; training memory use has not been tested.

## Normalization

Channel 0 DWI: **ZScoreNormalization**, use_mask_for_norm=True in both configurations.

Mean and standard deviation are computed per image over the nonzero-derived valid image region (seg >= 0 after cropping), before resampling. This includes background tissue and lesion; it is not normalization over the lesion alone. The zero-valued exterior is excluded. No custom normalization was applied.

## Preprocessed Case Count

| Configuration | Expected | Successfully checked | Missing | Failed | Median actual spatial shape (Z,Y,X) |
|---|---:|---:|---:|---:|---|
| 2d | 250 | 250 | 0 | 0 | [65.0, 83.0, 68.0] |
| 3d_fullres | 250 | 250 | 0 | 0 | [66.0, 83.0, 68.0] |

Every case’s .b2nd image, _seg.b2nd segmentation and .pkl metadata was read using the official dataset loader. Checks covered finite values, matching image/segmentation shape, expected shape from crop/spacing metadata, and actual label values.

## Label Values

Both configurations: **[-1, 0, 1]**. The official crop_to_nonzero function assigns -1 outside the nonzero image region. This internal marker is not an additional output class. No unexpected values were found or changed.

The three original empty-lesion masks remain empty and are retained: case_0150, case_0151, case_0170. Native source scaling artifacts were interpreted by the official reader; no source mask was rewritten.

## Visual QA

Seed 2022 selected the same five subjects for both configurations. Ten figures were visually inspected; each shows the maximum-lesion-area slice plus its preceding and following slice, with the nearest original slice identified using the crop and shape metadata.

| Case | Original DWI and GT shape (X,Y,Z) | Original spacing (X,Y,Z), mm | 3d_fullres shape (Z,Y,X) | Target spacing (Z,Y,X), mm | Result |
|---|---|---|---|---|---|
| case_0074 | [112, 112, 73] | [2.0, 2.0, 2.0] | [62, 76, 62] | [2.0, 2.0, 2.0] | PASS |
| case_0080 | [112, 112, 73] | [2.0, 2.0, 2.0] | [68, 85, 67] | [2.0, 2.0, 2.0] | PASS |
| case_0114 | [112, 112, 73] | [2.0, 2.0, 2.0] | [64, 77, 67] | [2.0, 2.0, 2.0] | PASS |
| case_0137 | [170, 170, 33] | [1.352941, 1.352941, 4.4] | [64, 78, 67] | [2.0, 2.0, 2.0] | PASS |
| case_0140 | [112, 112, 73] | [2.0, 2.0, 2.0] | [65, 84, 74] | [2.0, 2.0, 2.0] | PASS |

The original GT shape and spacing equal the original DWI for all five. For case_0137, 2d preserves the 4.4 mm slice spacing and yields [29,78,67]; 3d_fullres resamples to 2 mm and yields [64,78,67]. The other four samples have identical 2d/3d_fullres spatial shapes.

For all ten sample/configuration pairs, rerunning the official preprocessor in memory reproduced the saved image exactly (maximum absolute difference 0) and segmentation voxel-for-voxel. The QA replay did not save preprocessing outputs.

- case_0074: multiple contours remain on the corresponding DWI foci across all three slices.
- case_0080: small posterior lesion remains aligned across adjacent slices.
- case_0114: compact lesion contour remains aligned with the bright focus.
- case_0137: lesion remains aligned after anisotropic-to-isotropic resampling. Two adjacent resampled slices may map to the same nearest original slice; this follows the 4.4-to-2 mm spacing change.
- case_0140: extensive/multifocal contours retain their relationship to the DWI abnormalities.

No obvious image-mask displacement, new left/right or up/down flip, or gross resampling distortion was observed in these inspected slices. Crop magnification, voxel interpolation and changed grayscale contrast are expected. Zero-valued normalized exterior can appear gray under the display window. The plots use native array coordinates, not a standardized radiological display.

Visual coverage is five cases and three slices per configuration, not an exhaustive anatomical validation of every voxel in 250 volumes.

QA script: `src/preprocessing/check_preprocessed_cases.py`. Detailed measurements: `outputs/metrics/preprocessing/preprocessing_qa.json`.

Figures:

- [case_0074 3d_fullres](../outputs/figures/preprocessing/case_0074_3d_fullres.png) / [2d](../outputs/figures/preprocessing/case_0074_2d.png)
- [case_0080 3d_fullres](../outputs/figures/preprocessing/case_0080_3d_fullres.png) / [2d](../outputs/figures/preprocessing/case_0080_2d.png)
- [case_0114 3d_fullres](../outputs/figures/preprocessing/case_0114_3d_fullres.png) / [2d](../outputs/figures/preprocessing/case_0114_2d.png)
- [case_0137 3d_fullres](../outputs/figures/preprocessing/case_0137_3d_fullres.png) / [2d](../outputs/figures/preprocessing/case_0137_2d.png)
- [case_0140 3d_fullres](../outputs/figures/preprocessing/case_0140_3d_fullres.png) / [2d](../outputs/figures/preprocessing/case_0140_2d.png)

## Problems / Warnings

No failed/missing cases or unexpected labels. The lowres configuration was absent and skipped normally. Three empty lesion masks were retained. The -1 crop marker and plan-versus-uncropped shape distinction are documented above.

Raw source and nnUNet input preservation verification is saved in outputs/metrics/preprocessing/preprocessing-input-preservation.json.

Execution log: `outputs/metrics/preprocessing/preprocessing.log`. QA log: `outputs/metrics/preprocessing/preprocessing_qa.log`.

## Conclusion

Official preprocessing completed successfully for all 250 cases in both generated configurations. Counts, shapes, labels, sample transformation replay and sampled visual alignment passed. The dataset is prepared for the next training decision; no training command, fold, pretrained model, optimizer or loss modification was executed.
