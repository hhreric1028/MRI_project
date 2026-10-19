# case_0140 failure analysis

> Phase-1 descriptive analysis only. case_0140 was not used to tune a threshold or model, and no training was started.

## One-page evidence summary

1. **Why did it fail?** B0-R2022 found the lesion but covered only 33.6% of GT; Pred/GT was 0.379. This is a large partial-coverage failure, not a whole-lesion detection miss.
2. **Main evidence.** GT volume was 141.368 mL across 3 components; the largest component held 99.8% of GT. B0-R2022 FN volume was 93.896 mL.
3. **Intensity/contrast?** FN median DWI was 0.838× TP median and FN local CNR was 1.032 versus TP 1.869. This supports an intensity/representation contribution only to the degree stated in `intensity_statistics.csv`; it is not by itself causal.
4. **Morphology/coverage?** Yes: `dominant_large_component_with_satellites` with AP extent 100.0 mm and IS extent 90.0 mm, while every model detected at least part of the dominant lesion but left substantial GT uncovered.
5. **Threshold?** FN voxels near 0.5 (0.40–0.50) were 0.9%; FN voxels below 0.10 were 94.5%. Evidence class: `representation_or_feature_learning_dominant`. Threshold sweeps are diagnostic only, not a tuning recommendation.
6. **Best existing interventions.** Highest Dice was P3a-100 (0.540); highest voxel recall was P3a-100 (0.427). Full precision/FP trade-offs are in `model_comparison.csv`.
7. **Persistent missed region.** The intersection of FN across all 12 available models was 72.896 mL (51.6% of GT).
8. **Similar Fold 0 cases?** Strict large-coverage matches: none. Conclusion: `isolated_large_coverage_failure_in_fold0`. Broader multifeature/coverage neighbors are retained without using Dice for selection.
9. **Next interventions.** See `../../../docs/experiment_matrix.md`; proposals are capped at three and are explicitly linked to observed mechanisms.
10. **Not supported now.** No evidence justifies blind loss, spacing, sampling, architecture, or multimodal changes without the proposal-specific validation gates.

## Baseline coverage

| Metric | Value |
|---|---:|
| dice | 0.487032 |
| gt_ml | 141.368011 |
| pred_ml | 53.576004 |
| pred_gt_ratio | 0.378983 |
| tp_ml | 47.472004 |
| fn_ml | 93.896007 |
| fp_ml | 6.104000 |
| voxel_recall | 0.335804 |
| voxel_precision | 0.886068 |
| lesion_recall | 1.000000 |
| lesion_precision | 0.214286 |

## Morphology

- Pattern: `dominant_large_component_with_satellites`.
- Components: 3; largest component 141.128 mL (99.8% of GT).
- Extent: RL 108.0 mm; AP 100.0 mm; IS 90.0 mm.

## Preprocessing check

- Native/preprocessed spacing change: False; resampling shape change: False.
- Preprocessed GT exactly matches native GT after crop restoration: True.
- Raw/preprocessed intensity correlation inside the brain: 1.000000.
- These observations do not establish or exclude a preprocessing causal mechanism.

## V1 model comparison

| Model | Dice | Pred/GT | Voxel recall | Voxel precision | FN mL | FP mL |
|---|---:|---:|---:|---:|---:|---:|
| P3a-100 | 0.540 | 0.581 | 0.427 | 0.735 | 80.98 | 21.82 |
| P3a-300 | 0.535 | 0.514 | 0.405 | 0.789 | 84.06 | 15.35 |
| E3-Scale-300 | 0.513 | 0.425 | 0.365 | 0.861 | 89.70 | 8.35 |
| B0-R2022 | 0.487 | 0.379 | 0.336 | 0.886 | 93.90 | 6.10 |
| E3-Scale-100 | 0.484 | 0.387 | 0.336 | 0.868 | 93.88 | 7.19 |
| S1a-300 | 0.482 | 0.358 | 0.327 | 0.914 | 95.07 | 4.34 |
| Historical_B0 | 0.471 | 0.379 | 0.325 | 0.857 | 95.42 | 7.68 |
| S1a-100 | 0.468 | 0.353 | 0.317 | 0.898 | 96.59 | 5.09 |
| E1-Gamma-100 | 0.462 | 0.353 | 0.312 | 0.885 | 97.22 | 5.76 |
| P1 | 0.456 | 0.339 | 0.305 | 0.900 | 98.21 | 4.78 |
| L2a-100 | 0.448 | 0.333 | 0.299 | 0.897 | 99.14 | 4.86 |
| E2-CLAHE-100 | 0.412 | 0.301 | 0.268 | 0.889 | 103.53 | 4.70 |

## FN overlap

`persistent_FN_all_models.nii.gz` is the GT region missed by every available model. Pairwise overlap and each model's rescue of B0-R2022 FN are in `fn_overlap.csv`.

## Similar-case cohort

Cohort ranking uses GT volume, component count, largest-component ratio, AP/IS extent, Pred/GT, and baseline voxel recall. Dice is deliberately excluded.

| Case | Selection | Distance | GT mL | Components | Pred/GT | Recall |
|---|---|---:|---:|---:|---:|---:|
| case_0146 | coverage_failure_neighbor | 2.168 | 21.02 | 11 | 0.565 | 0.556 |
| case_0181 | coverage_failure_neighbor | 2.280 | 16.21 | 4 | 0.177 | 0.161 |
| case_0017 | nearest_multifeature_neighbor | 2.648 | 15.43 | 8 | 0.748 | 0.710 |
| case_0205 | nearest_multifeature_neighbor | 2.728 | 36.86 | 6 | 0.835 | 0.798 |
| case_0188 | nearest_multifeature_neighbor | 2.831 | 10.06 | 3 | 0.679 | 0.653 |
| case_0160 | coverage_failure_neighbor | 2.872 | 4.43 | 6 | 0.489 | 0.390 |
| case_0068 | coverage_failure_neighbor | 3.276 | 3.69 | 3 | 0.093 | 0.093 |
| case_0212 | coverage_failure_neighbor | 3.359 | 1.10 | 4 | 0.319 | 0.254 |
| case_0186 | coverage_failure_neighbor | 3.423 | 11.13 | 50 | 0.468 | 0.454 |
| case_0182 | coverage_failure_neighbor | 3.697 | 1.63 | 3 | 0.057 | 0.057 |

## Evidence limitations

- Fold 0 only; most intervention comparisons are single-seed screens and some have different epoch horizons.
- Historical B0 seed is unconfirmed; P1 also changes spatial resolution.
- Voxel intensity observations are spatially correlated and are reported descriptively.
- No single anchor case can establish an intervention effect.
