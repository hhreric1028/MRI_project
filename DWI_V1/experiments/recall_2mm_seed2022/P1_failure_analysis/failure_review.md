# P1 failure pattern review
Panels: first row = maximal GT area; second row = maximal B0/P1 disagreement. All are the same native DWI slice; common intensity scaling.
Component coverage is overlap-based, distinct from one-to-one evaluation lesion FN (merges can cause unmatched lesions). These are descriptive mask findings, not evidence that FOV or heterogeneity caused failure.

| Case | Model | Whole lesion components missed | Detected but incomplete | Extra FP components | Voxel recall |
|---|---|---:|---:|---:|---:|
| case_0010 | B0 | 1 | 13 | 0 | 0.689 |
| case_0010 | P1 | 2 | 12 | 0 | 0.511 |
| case_0051 | B0 | 0 | 6 | 0 | 0.610 |
| case_0051 | P1 | 0 | 8 | 0 | 0.738 |
| case_0053 | B0 | 0 | 2 | 0 | 0.911 |
| case_0053 | P1 | 0 | 2 | 0 | 0.878 |
| case_0068 | B0 | 2 | 1 | 0 | 0.089 |
| case_0068 | P1 | 2 | 1 | 0 | 0.067 |
| case_0069 | B0 | 0 | 1 | 0 | 0.706 |
| case_0069 | P1 | 0 | 1 | 0 | 0.529 |
| case_0118 | B0 | 2 | 2 | 0 | 0.672 |
| case_0118 | P1 | 3 | 3 | 1 | 0.486 |
| case_0135 | B0 | 0 | 3 | 0 | 0.632 |
| case_0135 | P1 | 0 | 3 | 0 | 0.434 |
| case_0140 | B0 | 0 | 3 | 6 | 0.325 |
| case_0140 | P1 | 0 | 3 | 0 | 0.305 |
| case_0160 | B0 | 1 | 5 | 1 | 0.457 |
| case_0160 | P1 | 2 | 4 | 1 | 0.347 |
| case_0181 | B0 | 2 | 2 | 3 | 0.176 |
| case_0181 | P1 | 1 | 3 | 2 | 0.208 |
| case_0182 | B0 | 1 | 2 | 0 | 0.114 |
| case_0182 | P1 | 1 | 2 | 0 | 0.362 |
| case_0188 | B0 | 1 | 2 | 0 | 0.638 |
| case_0188 | P1 | 1 | 2 | 0 | 0.684 |
| case_0212 | B0 | 2 | 2 | 2 | 0.362 |
| case_0212 | P1 | 1 | 2 | 1 | 0.283 |