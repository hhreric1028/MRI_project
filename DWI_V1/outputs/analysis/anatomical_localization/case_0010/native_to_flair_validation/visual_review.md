# Block 4B-1 Native-to-FLAIR Validation

Case: case_0010. Status: NEEDS REVIEW. Selected source: NONE.

## Inputs and method

DWI and ADC: LAS, shape 112 x 112 x 72, spacing 2 x 2 x 2 mm. Shape, spacing, affine and orientation match. ADC can therefore serve as an alternative intensity reference for the same native mask grid. FLAIR: RAS, shape 352 x 352 x 230, spacing approximately 0.681818 x 0.681818 x 1 mm; different grid. Full affines, LPS origins/directions and RAS voxel-boundary physical extents are saved in metadata.json.

Both experiments use rigid registration with FLAIR fixed, MOMENTS initialization, Mattes Mutual Information (50 bins), random 30% sampling with seed 2022, shrink [4,2,1], physical smoothing [2,1,0] mm, maximum 200 iterations per level, learning rate 0.5 and maximum physical step 1 mm. DWI reuses the separately saved native-to-FLAIR stage from Block 4B; ADC is independently fitted. Neither experiment uses an MNI transform.

transform.tfm is the FLAIR-LPS-to-native-LPS pull transform used for resampling. native_to_flair_forward.tfm is its inverse. Intensity interpolation is linear; mask interpolation is nearest neighbor. The rigid rotations preserve handedness (determinants +1); output grids match the original FLAIR. This excludes a transform-induced reflection, not errors in the original acquisition metadata.

## Visual comparison

Reviewed alpha and edge overlays across axial, coronal and sagittal planes, plus three local planes for each of the 13 native components in both experiments. QC plots use the FLAIR voxel spacing for physical aspect ratio. PASS means acceptable on sampled views, not a measured landmark-distance guarantee.

| QC item | DWI to FLAIR | ADC to FLAIR |
|---|---|---|
| Orientation / no transform reflection | PASS | PASS |
| Midline | PASS | PASS |
| Lateral ventricles, broad correspondence | PASS | PASS |
| Third ventricle, precise boundary | REVIEW | REVIEW |
| Cortical contour | REVIEW | REVIEW |
| Corpus callosum region, broad correspondence | PASS | PASS |
| Brainstem, broad correspondence | PASS | PASS |
| Cerebellum / fine boundary correspondence | REVIEW | REVIEW |
| Inferior boundary | REVIEW | REVIEW |
| Anterior/posterior extent | REVIEW | REVIEW |
| Lesion spatial plausibility | REVIEW | REVIEW |
| Overall | NEEDS REVIEW | NEEDS REVIEW |

## Lesion review

All 13 native components remain represented in both transformed masks. Neither result places lesion voxels outside the FLAIR nonzero support. This support is not a validated tissue mask and cannot establish absence of CSF overlap.

Components 1-10 and 12-13 show broadly plausible tissue placement on the sampled local views. Component 11 straddles or closely approaches a low-signal boundary in some planes; tissue-versus-CSF placement is not sufficiently resolved here. This is a review target, not proof that the registration failed or that the source prediction is incorrect. Native component IDs follow the prior XYZ-order connected-component convention; they are not size ranks.

DWI and ADC results are similar; no clear anatomical advantage supports selecting ADC as a surrogate source. Agreement between two related diffusion contrasts is not independent validation. Neither is promoted to a best transform.

## Mask and volume

Native prediction: 305 voxels. Existing formal native volume: 2.440 mL, retained from Block 2 and not recalculated.
Transformed mask counts (diagnostic only): DWI 5279; ADC 5239. Labels are [0,1]. No small components were removed, and no morphology cleanup was performed.

## Disposition

Best transform: NONE approved. Candidate transforms are retained in each experiment folder. The best folder contains only a decision record to prevent accidental use of an unapproved transform.

Next review should target corresponding native DWI/ADC/FLAIR landmarks and the component-11 boundary across adjacent slices; distinguish input prediction boundary uncertainty from modality-specific distortion before accepting the bridge. This result does not attribute the complete chain error exclusively to FLAIR-to-MNI.

No MNI nonlinear registration, atlas overlap or anatomical label assignment was performed.
