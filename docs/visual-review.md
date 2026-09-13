# Five-case native-grid visual review

Reviewed the generated QC PNGs for random seed 2022. Each shows the DWI slice with the
largest annotated lesion area, the same ADC/GT slice, and an independent central native FLAIR slice.
The figure coordinates are native voxel axes. They are not standardized radiological left/right displays.

| Case | DWI/GT inspected slice | Observation |
|---|---:|---|
| sub-strokecase0074 | 17 | Three visible GT contours overlap localized DWI bright regions; corresponding ADC regions are dark. No obvious gross offset on this slice. |
| sub-strokecase0080 | 34 | Small posterior GT contour overlaps a DWI bright focus and corresponding ADC dark focus. No obvious gross offset on this slice. |
| sub-strokecase0114 | 38 | Compact GT region follows a DWI bright focus and ADC dark region. No obvious gross offset on this slice. |
| sub-strokecase0137 | 22 | Small GT region overlaps a DWI bright focus and ADC dark region. No obvious gross offset on this slice. |
| sub-strokecase0140 | 35 | Extensive and multifocal GT contours broadly overlap visible DWI signal abnormalities and ADC low-signal regions. No obvious gross offset on this slice. |

All five DWI/ADC/GT grids agree in shape, spacing, affine and orientation, as do all 250 cases in the numerical audit.
All 250 FLAIR grids differ from DWI. FLAIR was viewed in its own native grid without GT overlay;
its slice is not necessarily the same anatomical level. FLAIR/GT anatomical registration has not been established.
No image was resampled, registered or reoriented. This sample review does not establish annotation accuracy
throughout every 3D volume or all 250 cases.
