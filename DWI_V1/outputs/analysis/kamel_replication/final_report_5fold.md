# Kamel-style replication: five-fold final report

Fold 0 strict replication gate originally failed on lesion precision and FP/case.
A dedicated FP audit subsequently found no source, split, inference, post-processing,
or evaluation mismatch. Prediction masks remained highly overlapping, and most of the
FP-count divergence was attributable to stochastic connected-component topology
differences. After human review, the replication was accepted as stochastic-equivalent,
and Fold 1–4 were approved without changing model or evaluation protocol。

## Protocol completion

All five fixed folds completed 100 epochs with unchanged DWI-only B0 configuration. Each of the 250 cases appears exactly once in validation and contributes one checkpoint-final prediction from a model that did not train on that case. No ensemble inference was performed.

The original Fold 0 gate remains recorded as failed; continuation was a documented human decision after the dedicated FP audit and did not redefine the gate.

## Five-fold results

| Fold | Mean Dice | Median Dice | Recall | Precision | FP/case | FP <0.1/case | FP >=1/case | Severe under | Large partial |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7714 | 0.8332 | 0.5809 | 0.6788 | 2.480 | 2.160 | 0.100 | 8 | 1 |
| 1 | 0.7758 | 0.8393 | 0.5877 | 0.7515 | 1.640 | 1.340 | 0.020 | 5 | 1 |
| 2 | 0.7654 | 0.8452 | 0.5595 | 0.7538 | 2.240 | 1.880 | 0.020 | 3 | 0 |
| 3 | 0.8027 | 0.8700 | 0.6729 | 0.7107 | 2.060 | 1.860 | 0.000 | 3 | 2 |
| 4 | 0.7394 | 0.8229 | 0.4230 | 0.6519 | 2.200 | 1.960 | 0.040 | 9 | 2 |

Fold-level Mean Dice: **0.7709 ± 0.0203** (population SD across five fold means). Best fold: 3; worst fold: 4.

## 250-case OOF aggregate

- Mean / median / SD Dice: 0.7711 / 0.8491 / 0.2087
- Lesion recall / precision / F1: 0.5585 / 0.7119 / 0.6260
- FP / FN lesions per case: 2.124 / 4.148
- FP volume/case: 0.2828 mL; FP <0.1 mL/case: 1.840; FP >=1 mL/case: 0.036
- Volume MAE: 7.497 mL; mean / median Pred/GT: 7.8256 / 0.9167
- Severe under / large partial coverage / Dice<0.5 / Dice<0.3: 28 / 6 / 23 / 15

## Fold-to-fold variance

Recall SD 0.0807; precision SD 0.0400; FP/case SD 0.2772; tiny-FP/case SD 0.2716; >=1 mL FP/case SD 0.0344.
Severe-under counts by fold: [8, 5, 3, 3, 9]. Large partial-coverage counts by fold: [1, 1, 0, 2, 2].
1–5 mL Dice mean ± fold SD: 0.7439 ± 0.0649; recall mean ± fold SD: 0.6746 ± 0.0481.

## Reassessment of Fold 0

Fold 0 Mean Dice is 0.7714, versus the five-fold mean of fold means 0.7709. Its recall/precision are 0.5809/0.6788; comparison with the other folds is shown above. `case_0140` is not classified by the fixed >20 mL and Pred/GT <0.5 rule. Fold 0 tiny-FP and >=1 mL FP rates are 2.160 and 0.100 per case.

## Literature context

The OOF mean Dice is 0.7711, versus the supplied Kamel et al. DWI-only reference of approximately 0.810 ± 0.155; the mean difference is -0.0389. This is a method-level contextual comparison only: our result is five-fold OOF and the paper protocol/test set is not established as identical, so this is not a head-to-head superiority or inferiority claim.

## Answers to the requested final questions

1. **Fold 1–4 是否全部成功完成？** 是；status、checkpoint、100-epoch training summary 與 final-checkpoint validation 均完成。
2. **每 fold Mean Dice / Recall / Precision？** 見 Five-fold results 表。
3. **5-fold Mean ± Std？** 0.7709 ± 0.0203。
4. **250-case OOF performance？** Mean Dice 0.7711、lesion recall 0.5585、precision 0.7119、F1 0.6260。
5. **Fold 0 是否偏難？** Fold 0 Mean Dice 與五-fold mean 的差為 +0.0005；其在五 fold 中排名 3/5。
6. **Small lesion performance 是否穩定？** 1–5 mL Dice/recall 的 fold SD 為 0.0649/0.0481；各 fold n 明列於 subgroup outputs，小 n 不過度解讀。
7. **Severe under-segmentation 是否普遍？** OOF 共 28 例；各 fold為 [8, 5, 3, 3, 9]。
8. **0140-type large partial failure 有幾例？** 依固定 GT >20 mL 且 Pred/GT <0.5 規則共 6 例。
9. **FP increase 是否主要為 tiny components？** OOF FP 中 <0.1 mL 共 460 個（1.840/case）；總 FP 為 531 個。
10. **>=1 mL FP 是否常見？** 共 9 個（0.036/case）。
11. **Fold-to-fold stochastic / split variance？** Mean Dice fold SD 0.0203；recall/precision/FP-case SD 分別為 0.0807/0.0400/0.2772。
12. **與 literature reference 的距離？** Mean Dice 相差 -0.0389；Fold variance may explain part, but not all, of the literature-reference gap.
13. **下一步？** 先由人工依 OOF failure composition 與 literature protocol 差異決定：若重點是公平外部評估，優先建立 independent hold-out；若重點是提升現有系統，優先回到 failure-specific improvement。是否繼續模仿 paper protocol需先確認 protocol 可比性。本輪不自動啟動任何一項。

## Output index

- `five_fold_summary.csv`: fold metrics, fold mean, and fold SD.
- `OOF_all_cases.csv`, `OOF_summary.json`: exact 250-case final-checkpoint OOF results.
- `OOF_prediction_manifest.csv`, experiment `OOF_predictions/`: one correct-fold mask and probability file per case.
- `OOF_failure_mode_counts.csv`, `OOF_large_partial_coverage_failures.csv`: failure analysis.
- `fold_variance_analysis.json`: Dice, recall, precision, FP, tiny-FP, large-FP, and failure variance.
- `fold_0` through `fold_4`: per-fold validation, topology, subgroup, severe-under, and Top-5 outputs.
- `fold0_replication_check.json`: preserved strict gate failure; `human_approval_after_fold0_audit.json`: approval provenance.
