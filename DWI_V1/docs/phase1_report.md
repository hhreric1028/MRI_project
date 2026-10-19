# Phase 1 completion report

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

## Source and files

[Official ISLES2022 v2.3.1 release](https://zenodo.org/records/7960856)

Raw: `DWI_V1/datasets/ISLES2022/raw`

Official archive MD5 and extracted file hashes are recorded in documentation/.

## Measured completeness

```json
{
  "total_cases": 250,
  "cases_with_dwi": 250,
  "cases_with_adc": 250,
  "cases_with_flair": 250,
  "cases_with_msk": 250,
  "incomplete_cases": []
}
```

Observed GT values: `[0, 0.9999999997671694, 1]`.

Official SimpleITKIO values: `[0.0, 1.0]`. No mask rewriting or label remapping.

Empty masks: 3.

Empty-mask case IDs (all retained): sub-strokecase0150_ses-0001, sub-strokecase0151_ses-0001, sub-strokecase0170_ses-0001

NIfTI scaling artifact case IDs (native float64 foreground is near 1; official reader gives exactly 1; all files copied unchanged): sub-strokecase0046_ses-0001, sub-strokecase0073_ses-0001, sub-strokecase0101_ses-0001, sub-strokecase0137_ses-0001, sub-strokecase0153_ses-0001, sub-strokecase0154_ses-0001, sub-strokecase0155_ses-0001, sub-strokecase0159_ses-0001, sub-strokecase0161_ses-0001, sub-strokecase0162_ses-0001, sub-strokecase0164_ses-0001, sub-strokecase0165_ses-0001, sub-strokecase0175_ses-0001, sub-strokecase0176_ses-0001, sub-strokecase0182_ses-0001, sub-strokecase0184_ses-0001, sub-strokecase0186_ses-0001, sub-strokecase0187_ses-0001, sub-strokecase0188_ses-0001, sub-strokecase0189_ses-0001, sub-strokecase0191_ses-0001, sub-strokecase0192_ses-0001, sub-strokecase0193_ses-0001, sub-strokecase0195_ses-0001, sub-strokecase0198_ses-0001, sub-strokecase0201_ses-0001, sub-strokecase0202_ses-0001, sub-strokecase0206_ses-0001, sub-strokecase0207_ses-0001, sub-strokecase0208_ses-0001, sub-strokecase0210_ses-0001, sub-strokecase0215_ses-0001, sub-strokecase0219_ses-0001, sub-strokecase0226_ses-0001, sub-strokecase0227_ses-0001, sub-strokecase0228_ses-0001, sub-strokecase0229_ses-0001, sub-strokecase0230_ses-0001, sub-strokecase0232_ses-0001, sub-strokecase0236_ses-0001, sub-strokecase0238_ses-0001, sub-strokecase0239_ses-0001, sub-strokecase0241_ses-0001, sub-strokecase0243_ses-0001, sub-strokecase0245_ses-0001, sub-strokecase0249_ses-0001, sub-strokecase0250_ses-0001

## Five random samples (seed 2022)

| Case | Modality | Shape | Spacing (mm) | Orientation |
|---|---|---|---|---|
| sub-strokecase0074_ses-0001 | MSK | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0074_ses-0001 | FLAIR | [281, 352, 352] | [0.711538, 0.710227, 0.710227] | RAS |
| sub-strokecase0074_ses-0001 | ADC | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0074_ses-0001 | DWI | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0080_ses-0001 | MSK | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0080_ses-0001 | FLAIR | [528, 528, 29] | [0.436603, 0.436603, 5.0] | LAS |
| sub-strokecase0080_ses-0001 | ADC | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0080_ses-0001 | DWI | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0114_ses-0001 | MSK | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0114_ses-0001 | FLAIR | [281, 352, 352] | [0.711539, 0.710227, 0.710227] | RAS |
| sub-strokecase0114_ses-0001 | ADC | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0114_ses-0001 | DWI | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0137_ses-0001 | MSK | [170, 170, 33] | [1.352941, 1.352941, 4.4] | LAS |
| sub-strokecase0137_ses-0001 | FLAIR | [240, 320, 33] | [0.71875, 0.71875, 4.4] | LAS |
| sub-strokecase0137_ses-0001 | ADC | [170, 170, 33] | [1.352941, 1.352941, 4.4] | LAS |
| sub-strokecase0137_ses-0001 | DWI | [170, 170, 33] | [1.352941, 1.352941, 4.4] | LAS |
| sub-strokecase0140_ses-0001 | MSK | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0140_ses-0001 | FLAIR | [320, 320, 29] | [0.71875, 0.71875, 5.0] | LAS |
| sub-strokecase0140_ses-0001 | ADC | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |
| sub-strokecase0140_ses-0001 | DWI | [112, 112, 73] | [2.0, 2.0, 2.0] | LAS |

## Geometry differences

Tolerances: spacing absolute 1e-5; affine absolute 1e-3 (no relative tolerance).

Differences by comparison modality: {'flair': 250}.

Full case-by-case matrices, datatype, labels and counts are in outputs/metrics/data/isles2022_inspection.json and .csv.

| Case | DWI compared with | Differences |
|---|---|---|
| sub-strokecase0001_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0002_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0003_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0004_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0005_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0006_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0007_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0008_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0009_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0010_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0011_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0012_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0013_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0014_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0015_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0016_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0017_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0018_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0019_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0020_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0021_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0022_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0023_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0024_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0025_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0026_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0027_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0028_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0029_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0030_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0031_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0032_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0033_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0034_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0035_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0036_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0037_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0038_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0039_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0040_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0041_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0042_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0043_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0044_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0045_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0046_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0047_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0048_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0049_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0050_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0051_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0052_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0053_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0054_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0055_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0056_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0057_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0058_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0059_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0060_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0061_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0062_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0063_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0064_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0065_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0066_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0067_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0068_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0069_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0070_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0071_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0072_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0073_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0074_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0075_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0076_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0077_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0078_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0079_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0080_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0081_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0082_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0083_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0084_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0085_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0086_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0087_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0088_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0089_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0090_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0091_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0092_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0093_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0094_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0095_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0096_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0097_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0098_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0099_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0100_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0101_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0102_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0103_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0104_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0105_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0106_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0107_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0108_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0109_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0110_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0111_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0112_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0113_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0114_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0115_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0116_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0117_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0118_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0119_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0120_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0121_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0122_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0123_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0124_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0125_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0126_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0127_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0128_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0129_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0130_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0131_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0132_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0133_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0134_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0135_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0136_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0137_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0138_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0139_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0140_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0141_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0142_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0143_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0144_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0145_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0146_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0147_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0148_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0149_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0150_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0151_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0152_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0153_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0154_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0155_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0156_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0157_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0158_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0159_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0160_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0161_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0162_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0163_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0164_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0165_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0166_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0167_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0168_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0169_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0170_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0171_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0172_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0173_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0174_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0175_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0176_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0177_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0178_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0179_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0180_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0181_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0182_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0183_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0184_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0185_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0186_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0187_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0188_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0189_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0190_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0191_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0192_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0193_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0194_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0195_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0196_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0197_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0198_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0199_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0200_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0201_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0202_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0203_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0204_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0205_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0206_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0207_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0208_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0209_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0210_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0211_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0212_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0213_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0214_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0215_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0216_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0217_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0218_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0219_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0220_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0221_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0222_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0223_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0224_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0225_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0226_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0227_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0228_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0229_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0230_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0231_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0232_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0233_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0234_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0235_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0236_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0237_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0238_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0239_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0240_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0241_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0242_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0243_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0244_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0245_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0246_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0247_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible, orientation_equal |
| sub-strokecase0248_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0249_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |
| sub-strokecase0250_ses-0001 | flair | shape_equal, spacing_equal, affine_compatible |

Grid agreement does not by itself prove anatomical alignment. See visual-review.md for the five inspected QC images. No resampling was performed.

## nnUNet dataset

`DWI_V1/datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI`

```json
{
  "channel_names": {
    "0": "DWI"
  },
  "labels": {
    "background": 0,
    "lesion": 1
  },
  "numTraining": 250,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO"
}
```

## Official integrity check

```text
Official dataset ID resolution: 1 -> Dataset001_ISLES2022_DWI
Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> reader/writer

####################
verify_dataset_integrity Done. 
If you didn't see any error messages then your dataset is most likely OK!
####################

PASS: official integrity verification only; no fingerprint, planning, preprocessing or training.

```

The official planning CLI help was checked, and the same official integrity routine was run directly. No fingerprint extraction, planning, preprocessing or training was started.

## Directory tree

Full tree: [historical directory-tree.txt](reference/directory-tree.txt). Raw and converted files are all included.
