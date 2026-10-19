# Fold 0 Replication FP Audit

## Executive decision

- **Mismatch classification: D. MINOR STOCHASTIC VARIATION** — with clear sensitivity of lesion-level metrics to small connected components and checkpoint trajectory.
- **Gate decision: keep STOP.** The original, predeclared final-vs-final gate remains failed: lesion precision changes by -0.0359 (tolerance 0.03) and FP/case changes by +0.34 (tolerance 0.30). This audit does not relax or redefine the gate.
- No training was started, Fold 1–4 were not started, and no existing B0 or replication output was overwritten. The only added model runs were inference-only runs from already existing best checkpoints.
- No inference or evaluation mismatch was found. The two final predictions have high overlap (mean prediction Dice 0.9347; median 0.9553), and the extra FP count is mainly driven by tiny components rather than a global increase in predicted foreground volume.
- This result should therefore be treated as a **small but gate-material stochastic/topological divergence**, not as evidence that the replication pipeline, inference path, or evaluator is different. Human review is required before any decision about Fold 1–4.

## 1. Source audit

Both compared gate predictions are the existing **epoch-100 final checkpoint** outputs.

| Item | B0-100 | Kamel replication Fold 0-100 |
|---|---|---|
| Prediction folder | `experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/train/results/Dataset001_ISLES2022_DWI/nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0/validation` | `experiments/kamel_replication_seed2022/fold_0/results/Dataset001_ISLES2022_DWI/nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres/fold_0/validation` |
| Checkpoint | `checkpoint_final.pth`, epoch 100 | `checkpoint_final.pth`, epoch 100 |
| Checkpoint SHA-256 | `cfca704ff0c337887cc091e83da983de45bbfa8c402d436a21ae31ec17b58b3d` | `c3591dc0f15e9c9232ea280ce48614607adbf726322b8ccf61cc4fc032d54f70` |
| Post-processing | none | none |
| Binary decision | foreground argmax (two-class equivalent threshold 0.5) | same |
| Components | 3D, 26-connectivity | same |
| Metric code | `scripts/evaluate_experiment.py` | imports the same `metrics()` and `summarize()` implementation |

Semantic plans and dataset JSONs match; split SHA-256, package manifest, and consequential evaluation source hash also match. Full provenance, runner commands, and paths are recorded in [source_audit.md](source_audit.md).

## 2. Paired 50-case comparison

The complete paired table is [fold0_paired_case_comparison.csv](fold0_paired_case_comparison.csv). It contains all requested B0, replication, and delta fields for the same 50 validation cases.

| Aggregate metric | B0 final | Replication final | Replication - B0 |
|---|---:|---:|---:|
| Mean Dice | 0.7645 | 0.7714 | +0.0069 |
| Median Dice | 0.8239 | 0.8332 | +0.0093 |
| Lesion recall | 0.5942 | 0.5809 | -0.0133 |
| Lesion precision | 0.7147 | 0.6788 | **-0.0359** |
| FP/case | 2.14 | 2.48 | **+0.34** |
| Total FP components | 107 | 124 | +17 |
| Severe-under cases | 8 | 8 | 0 |

The already reported paired mean absolute Dice delta is 0.0221, which is small relative to the component-level discrepancy.

## 3. Distribution of the FP increase

- 15/50 cases increased in FP count, 27/50 were unchanged, and 8/50 decreased.
- Gross increases totaled +30 FP components; decreases offset 13 of them, leaving a net +17.
- The three largest increasing cases account for 40.0% of the gross increase; the top five account for 56.7%.
- The largest increases were `case_0140` (+5), `case_0048` (+4), `case_0181` (+3), and `case_0201` (+3).

Conclusion: the increase is **moderately concentrated**, but not caused by a single outlier or only a couple of cases. Fifteen cases contribute positive changes, while a small leading group explains slightly more than half of the gross increase.

Top-10 outputs: [top_fp_increase.csv](top_fp_increase.csv), [top_precision_drop.csv](top_precision_drop.csv), [top_dice_delta.csv](top_dice_delta.csv), [top_pred_component_increase.csv](top_pred_component_increase.csv), and [top_volume_overseg_increase.csv](top_volume_overseg_increase.csv).

## 4. 3D component and FP-size audit

All component calculations use the existing 3D 26-connectivity rule and the same maximum-cardinality, then maximum-IoU, one-to-one matching rule as the gate evaluator. No minimum component-size filter was added.

| FP volume bin | B0 count | Replication count | Delta | B0 FP volume (mL) | Replication FP volume (mL) |
|---|---:|---:|---:|---:|---:|
| <0.01 mL | 18 | 31 | **+13** | 0.144 | 0.247 |
| 0.01–0.05 mL | 55 | 55 | 0 | 1.367 | 1.446 |
| 0.05–0.1 mL | 17 | 22 | **+5** | 1.218 | 1.551 |
| 0.1–0.5 mL | 13 | 9 | -4 | 3.182 | 1.544 |
| 0.5–1 mL | 1 | 2 | +1 | 0.656 | 1.208 |
| >=1 mL | 3 | 5 | +2 | 5.696 | 22.648 |

Count-wise, the net increase is dominated by very small components: the <0.01 mL bin alone adds 13, and all bins below 0.1 mL add 18 before offsets elsewhere. However, it is **not exclusively clinically trivial noise**: replication also has two additional FP components at least 1 mL, and that bin carries a meaningful FP-volume increase.

Detailed records are in [prediction_components.csv](prediction_components.csv), [fp_components.csv](fp_components.csv), and [fp_volume_distribution.csv](fp_volume_distribution.csv). Physical centroids and nearest-GT distances are reported in millimetres using the NIfTI affine. `boundary_adjacent_26` means a non-overlapping FP touches the one-voxel 26-neighbour dilation of GT; `completely_isolated` means neither overlap nor such adjacency.

## 5. Tiny-component and fragmentation analysis

| Quantity | B0 | Replication | Delta |
|---|---:|---:|---:|
| Total predicted foreground volume across 50 cases | 1097.331 mL | 1080.374 mL | **-16.957 mL (-1.55%)** |
| Predicted component count | 375 | 386 | +11 |
| Mean component volume | 2.926 mL | 2.799 mL | -0.127 mL |
| Median component volume | 0.072 mL | 0.072 mL | 0 |
| All components <0.05 mL | 155 | 163 | +8 |
| All components <0.1 mL | 222 | 231 | +9 |
| FP components <0.05 mL | 73 | 86 | +13 |
| FP components <0.1 mL | 90 | 108 | +18 |

Replication does **not** show global volume inflation. It predicts 1.55% less total foreground while producing 11 more connected components and 17 more unmatched FP components. Ten cases increased component count while keeping their total predicted volume within ±5% of B0, which is direct evidence for topology/fragmentation sensitivity.

Fragmentation alone is not a complete explanation. The spatial change table contains 31 replication-only new-FP events, 12 same-location fragmentation-increase events, and some larger new false lesions. Event counts from [fp_change_classification.csv](fp_change_classification.csv) are:

| Change class | Events |
|---|---:|
| A. Both models already had FP | 54 |
| B. Replication-only new FP | 31 |
| C. B0-only FP | 35 |
| D. Same location, replication larger | 17 |
| E. Same location, fragmentation increased | 12 |

These are spatial component-transition events, not case counts. The per-case tiny-component table is [tiny_component_analysis.csv](tiny_component_analysis.csv).

## 6. Prediction-vs-prediction overlap

- Mean B0-vs-replication prediction Dice: **0.9347**.
- Median prediction Dice: **0.9553**.
- Replication-only voxels: 4,616; 2,576 are inside GT (**55.81%**) and 2,040 are outside GT (44.19%).
- B0-only voxels: 6,585; 2,864 are inside GT (43.49%) and 3,721 are outside GT (56.51%).

Thus the replication did not simply add background. More than half of its added voxels are true lesion voxels, while it also removed more voxels than it added. The FP/case penalty arises because lesion metrics count unmatched connected components, so a few extra or split components can materially affect FP/case despite high voxel-wise agreement and slightly better mean Dice.

Full per-case overlap data are in [prediction_overlap.csv](prediction_overlap.csv).

## 7. Existing-checkpoint comparison

All four rows below use the same inference and evaluation path. The two best-checkpoint rows were generated inference-only from checkpoints that already existed.

| Variant | Epoch | Mean Dice | Median Dice | Lesion recall | Lesion precision | FP/case |
|---|---:|---:|---:|---:|---:|---:|
| B0 best | 94 | 0.7731 | 0.8242 | 0.6120 | 0.6987 | 2.38 |
| B0 final | 100 | 0.7645 | 0.8239 | 0.5942 | 0.7147 | 2.14 |
| Replication best | 68 | 0.7841 | 0.8211 | 0.6053 | 0.7261 | 2.06 |
| Replication final | 100 | 0.7714 | 0.8332 | 0.5809 | 0.6788 | 2.48 |

Checkpoint selection strongly changes lesion-level metrics, but it does **not** explain the original mismatch: the original gate compared final with final, so selection was matched. Replication best is much better than replication final for precision and FP/case, showing different optimization trajectories. Best-vs-best is close overall, but its absolute FP/case difference is 0.32, still narrowly outside the predeclared 0.30 tolerance. No checkpoint substitution or gate reinterpretation is performed here.

See [checkpoint_inference_comparison.csv](checkpoint_inference_comparison.csv), [checkpoint_pairwise_comparison.csv](checkpoint_pairwise_comparison.csv), and [checkpoint_inference_manifest.json](checkpoint_inference_manifest.json).

## 8. Inference settings audit

All checked settings are **SAME**: sliding-window step 0.5, Gaussian weighting enabled, mirroring/TTA on axes (0,1,2), CUDA device, probability export enabled, foreground argmax, native-space resampling, interpolation order, crop/transpose restoration, no post-processing, and label order. No `DIFFERENT` or consequential `UNKNOWN` item was found.

Details: [inference_setting_diff.md](inference_setting_diff.md).

## 9. Evaluation metric audit

The two paths use the same 26-connectivity, no minimum-size filter, lesion definitions, any-overlap edge definition, maximum-cardinality then maximum-IoU one-to-one matching, empty/double-empty handling, aggregate lesion-precision denominator, and FP/case denominator. No component-matching mismatch explains the result.

An important implication of the shared one-to-one rule is that a prediction component may overlap GT yet still be an FP when it is not selected in the one-to-one assignment. This is why fragmentation can increase FP count without a comparably large voxel-wise error.

Details: [evaluation_metric_audit.md](evaluation_metric_audit.md).

## 10. Visual review of highest-difference cases

The union of FP-increase Top 3 and precision-drop Top 3 contains six unique cases: `case_0140`, `case_0048`, `case_0181`, `case_0182`, `case_0086`, and `case_0105`. The montage for each case includes DWI, GT, B0 prediction, replication prediction, a B0/replication difference map, and the unmatched-FP overlay.

Visual review is consistent with the quantitative audit:

- `case_0140` shows the largest FP-count increase (+5) within a broadly similar large-lesion prediction; topology and small internal/disconnected regions matter more than global volume.
- `case_0048` adds four FPs while replication predicted volume is actually 2.144 mL lower, arguing against simple over-segmentation.
- `case_0181` improves Dice substantially (+0.1445) while FP count increases by three, demonstrating that better voxel overlap and worse lesion precision can coexist.
- `case_0182` and `case_0086` each add one component and therefore show large case-level precision drops despite minimal mask change.
- `case_0105` combines two extra FPs with recall loss, a less benign case than pure tiny-fragment noise.

Images are under [figures](figures), with slice selection recorded in [visualized_cases.csv](visualized_cases.csv). Each montage is a representative axial slice; all reported component counts and matching decisions are based on the complete 3D volumes.

## 11. Answers to the nine audit questions

1. **Is the FP increase mainly due to a few outliers?** Partly, but not exclusively. Fifteen cases increase; the top three explain 40.0% and top five 56.7% of the gross increase.
2. **Is it mainly from tiny components?** Yes for count: +13 components below 0.01 mL and +18 below 0.1 mL before offsets. No for all clinical burden: the >=1 mL FP bin also increases by two components.
3. **Did total predicted volume materially increase?** No. It decreases by 16.957 mL across 50 cases, or 1.55%.
4. **Is it only increased component fragmentation?** No. Fragmentation is a material mechanism, but replication-only new FPs and several larger false lesions are also present.
5. **What fraction of replication-only voxels is inside true GT lesion?** 55.81% (2,576/4,616).
6. **Can checkpoint selection explain the mismatch?** It strongly modifies the metrics but cannot explain the original gate failure because both gate predictions use final checkpoints. Best-vs-best remains narrowly outside the FP/case tolerance.
7. **Are inference settings consistent?** Yes; every checked consequential setting is the same.
8. **Is evaluation/component matching consistent?** Yes; both use the same implementation and matching rule.
9. **Is there evidence for seed/reproducibility stochasticity?** Yes. Source data, split, packages, architecture/plans, inference, and evaluator match, while masks remain highly overlapping and changes cluster in component topology. Best epochs also diverge (94 versus 68). The existing augmenter records fixed worker seeds but does not guarantee deterministic worker arrival order, so bitwise reproduction was never guaranteed. This is evidence consistent with stochastic training variation; it does not identify a single exclusive random mechanism.

## 12. Final mismatch classification and recommendation

### Classification: D. MINOR STOCHASTIC VARIATION

This is the best-supported classification because:

- voxel-wise predictions are highly overlapping;
- mean/median Dice remain close and replication mean Dice is slightly higher;
- no source, split, inference, post-processing, or evaluation mismatch was found;
- total prediction volume decreases rather than inflates;
- most of the extra FP count lies in tiny components;
- component fragmentation and one-to-one matching amplify small topological changes;
- existing best checkpoints reveal different stochastic optimization trajectories.

The classification is qualified as **gate-material**: the predeclared precision and FP/case tolerances are still exceeded, and there are a few larger new FPs. Therefore the operational recommendation is:

1. **Keep STOP.** Do not start Fold 1–4 under the current gate state.
2. **Do not change inference or evaluation and re-gate**, because no mismatch was found there.
3. Treat any future decision to accept stochastic equivalence, switch checkpoint-selection policy, or proceed to Fold 1–4 as a separate human-approved protocol decision. This audit does not make that change.

## Output inventory and integrity

The audit directory contains the requested full paired table, five Top-10 tables, complete prediction- and FP-component tables, spatial FP-change classification, tiny-component table, prediction-overlap table, FP-size distribution, checkpoint comparisons, source/inference/evaluation audits, six visual montages, evidence JSON, and manifests. The primary paired and per-case CSVs contain 50 data rows; all artifacts were written only beneath this audit directory.
