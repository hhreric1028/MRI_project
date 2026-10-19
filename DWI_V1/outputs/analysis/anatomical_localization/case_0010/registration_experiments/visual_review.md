# Block 4B Registration Improvement — NEEDS REVIEW

Case: case_0010, validation. Native prediction volume remains 2.440 mL.

## Modalities

DWI and ADC: 112 x 112 x 72; 2 mm isotropic; LAS; matching affine.
FLAIR: 352 x 352 x 230; 0.681818 x 0.681818 x 1 mm; RAS; different grid/affine. No T1/T1w or T2/T2w found in the case directory.
Different grid alone does not establish misregistration. A rigid DWI/FLAIR transform was estimated and assessed.

## Controlled comparisons

- Baseline: preserved Block 4A transform, regenerated QC only; MOMENTS, Mattes MI 50 bins, random 30%, seed 2022, [4,2,1] shrink and [2,1,0] mm smoothing, 200 maximum iterations per level.
- Geometry: changed initialization from MOMENTS to GEOMETRY. No reliable improvement in ventricular/inferior correspondence.
- Masked: same MOMENTS start with positive-support metric masks; failed with all samples outside moving buffer. No usable transform or mask from this failed attempt.
- Masked refinement: restarted from baseline affine, positive-support masks, learning rate 0.1 and 1 mm maximum physical step. This changes multiple optimization settings and is a recovery experiment, not a pure masking ablation. Inferior alignment remains uncertain.
- FLAIR: DWI rigid to FLAIR, then FLAIR rigid/affine to MNI T1; MI settings unchanged, 1 mm maximum physical step. Compose reference-to-FLAIR then FLAIR-to-DWI for nearest-neighbor resampling. DWI and prediction are resampled only once to the final template grid.

## Visual landmark assessment

PASS below means no gross issue on inspected views; it is not clinical validation. REVIEW is insufficient confidence for small-lesion atlas assignment.

| Experiment | Brain Boundary | Ventricles | Midline | Brainstem | Cerebellum | Inferior Boundary | Overall |
|---|---|---|---|---|---|---|---|
| Baseline | REVIEW | REVIEW | PASS | REVIEW | REVIEW | REVIEW | NEEDS REVIEW |
| Geometry | REVIEW | REVIEW | PASS | REVIEW | REVIEW | REVIEW | NEEDS REVIEW |
| Masked refinement | REVIEW | REVIEW | PASS | REVIEW | REVIEW | REVIEW | NEEDS REVIEW |
| FLAIR chain | REVIEW | REVIEW | PASS | REVIEW | REVIEW | REVIEW | NEEDS REVIEW |
| Masked initial attempt | N/A | N/A | N/A | N/A | N/A | N/A | FAIL |

The template and subject ventricular shapes differ across the sampled planes. Inferior cortical and posterior fossa outlines also differ. The FLAIR/DWI bridge has reasonable broad alignment but residual cortical/inferior differences, visible in structural_chain_qc.png. These cannot be resolved into patient anatomical variation versus registration error from this limited QC alone.

No approved best transform is selected. The FLAIR chain is retained as a review candidate, not demonstrated superior and not approved for atlas mapping. Its registered mask has 479 voxels, labels [0,1]; baseline 449, geometry 455, masked refinement 331. All are resampling diagnostics. Native volume is the sole formal volume result.

No nonlinear experiment was promoted: the multimodal bridge/landmark correspondence still needs review before using deformation to alter ventricular or inferior anatomy. Next: inspect native DWI/FLAIR landmarks in a multiplanar viewer, quantify corresponding landmark residuals, then evaluate a separate regularized nonlinear structural-to-template experiment if warranted.

Harvard–Oxford Subcortical: available locally. Harvard–Oxford Cortical: not present locally in this run. Subcortical includes broad cerebral cortex/white matter labels as well as subcortical structures; it does not supply detailed cortical gyral or cerebellar parcellation. XML schema version 1.0 is not an atlas release number.

No atlas overlap performed. Original images and masks retained unchanged.
