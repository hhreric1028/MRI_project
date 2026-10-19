# DWI_V2 failure-driven research plan

## Research question

Why do a small number of DWI-only nnUNet cases show severe lesion-coverage failure,
and which intervention is mechanistically justified by the failure evidence?

`case_0140` is the phase-1 anchor because it combines a very large GT lesion
(141.368 mL) with severe undercoverage. It is used to discover and test hypotheses,
not to choose hyperparameters or checkpoints.

## Locked boundaries

- `../DWI_V1` is a read-only evidence source. The source paths are recorded in
  `../configs/paths.yaml` and the resolved manifest is written by the analysis.
- Every new metric, map, figure, comparison, configuration, and future checkpoint
  must be written under `DWI_V2`.
- Phase 1 cannot start training or change loss, sampling, spacing, preprocessing,
  architecture, threshold, or input modalities.
- Historical B0, B0-R2022, P1, S1a, L2a, P3a, and E1–E3 comparisons are
  descriptive. Seeds, spatial resolution, input representation, and training
  horizons are not all matched.

## Phase 1 workflow

1. Confirm source geometry and record a read-only manifest.
2. Quantify GT morphology, components, physical extent, and axial distribution.
3. Quantify B0-R2022 voxel coverage, lesion detection, component fragmentation,
   and slice-level failure.
4. Produce fixed-slice panels using a common native orientation and intensity
   window.
5. Compare TP, FN, and a 10 mm local tissue ring on native DWI.
6. Inspect saved B0-R2022 probability maps and run diagnostic-only threshold
   sweeps.
7. Compare native and nnUNet-preprocessed representations without changing either.
8. Compare all available V1 predictions on the same voxels and compute persistent
   FN regions.
9. Build a Fold 0 cohort using morphology and coverage features; Dice is excluded
   from neighbor selection.
10. Assign evidence-backed phenotypes and propose at most three interventions.

## Similar-case design

Two cohort levels are retained:

- **Strict anchor-like phenotype:** GT ≥50 mL, Pred/GT ≤0.6, and baseline voxel
  recall ≤0.6. No other Fold 0 case met all three criteria.
- **Broader coverage-failure cohort:** cases with Pred/GT ≤0.6 and voxel recall
  ≤0.6, ranked together with multifeature neighbors by GT volume, component count,
  largest-component ratio, AP/IS extent, Pred/GT, and voxel recall.

The multifeature distance is Fold 0 median/IQR scaled and does not include Dice.
The retained broader coverage-failure cases are `case_0146`, `case_0181`,
`case_0160`, `case_0068`, `case_0212`, `case_0186`, and `case_0182`. Exact ranks
and features are in `../outputs/comparisons/case_0140_like_cohort.csv`.

Because the strict large-lesion phenotype is isolated in Fold 0, no future method
may be promoted on `case_0140` alone. Promotion requires:

- `case_0140` as a descriptive anchor;
- the broader coverage-failure cohort;
- the complete fixed Fold 0 validation set; and
- at least one additional large-lesion severe-coverage case from a prespecified
  non-tuning cohort or another fold before claiming phenotype generalization.

## Evaluation contract for a future intervention

Report at minimum:

- Dice, GT/prediction volume, Pred/GT, TP/FN/FP volume;
- voxel recall and precision;
- one-to-one component-matched lesion recall and precision;
- case-level and slice-level coverage;
- baseline-FN rescue and baseline-TP newly missed;
- persistent-FN change;
- full Fold 0 mean/median Dice, lesion recall/precision, FP lesions per case, and
  severe-undercoverage count;
- paired case bootstrap intervals against the matched baseline.

Any improvement claim must state the precision/FP trade-off. A higher recall on the
anchor with worse cohort-wide behavior is not success.

## Phase-1 result and stop condition

The completed evidence points to a low-salience, representation/feature-learning
coverage failure rather than a threshold or preprocessing-geometry problem. Two
mechanism-linked proposals are documented in `experiment_matrix.md`. They remain
**proposed, not started**. Phase 1 stops here with `training_started: false` in the
analysis summary.

