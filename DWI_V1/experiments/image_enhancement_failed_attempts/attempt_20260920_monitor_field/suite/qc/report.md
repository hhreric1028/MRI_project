# Image enhancement QC and parameter freeze

Parameter selection used fold-0 training cases only. Fixed validation failure cases were visualized only after parameters were frozen; no validation segmentation score selected any parameter.

Local surrounding tissue: GT-external, brain-foreground ring at Euclidean distance >0 to <= 6 mm (three voxels at 2 mm).

## E1 Gamma

Candidates: [0.7, 0.8, 1.2]; selected: 1.2. Transform is anchored at per-volume foreground P99.5 without clipping.

## E2 CLAHE

Slice-wise native axial CLAHE using scikit-image; clip candidates [0.005, 0.01, 0.02]; selected 0.02; kernel (16, 16) voxels (nominal 32×32 mm). Background is excluded and reset to zero.

## E3 Mild intensity scale

Baseline per-volume foreground z-score followed by I_enhanced = 0.76 × I + 0.0. Channel 1 uses NoNormalization so nnU-Net does not cancel the intended scale. This is a non-destructive affine representation control; its CNR is expected to remain invariant.

## Selection and evidence

General training cases: ['case_0023', 'case_0183', 'case_0218', 'case_0221', 'case_0236']
1–5 mL training cases: ['case_0098', 'case_0099', 'case_0142', 'case_0158', 'case_0169']
Multifocal training cases: ['case_0005', 'case_0143', 'case_0179']
Validation failure visualization only: ['case_0068', 'case_0181', 'case_0182', 'case_0140', 'case_0186']

Candidate-level intensity, CNR, local contrast, saturation, noise amplification and structure-correlation statistics are in `parameter_selection.json`; per-case evidence is in CSV/JSON and fixed-slice figures.
