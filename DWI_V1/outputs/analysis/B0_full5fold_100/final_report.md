# Original B0-100 full five-fold vs KamelRep-100 final report

## Bottom line

**Verdict: C. STOCHASTICALLY EQUIVALENT.** The exact audit found no effective technical configuration difference. This is a comparison of two independent five-fold trainings with the same configuration, so observed deltas quantify stochastic variability rather than a Kamel-method effect.

## Fold completion and B0 results

Original B0 Fold 0 passed the reuse gate and was not retrained. Folds 1–4 each completed exactly 100 epochs, checkpoint-final inference, and the fixed evaluator.

| Fold | B0 Mean Dice | B0 Recall | B0 Precision | B0 FP/case | B0 Severe under | Kamel Mean Dice | Δ Dice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7645 | 0.5942 | 0.7147 | 2.140 | 8 | 0.7714 | +0.0069 |
| 1 | 0.7855 | 0.6066 | 0.7191 | 2.000 | 2 | 0.7758 | -0.0097 |
| 2 | 0.7633 | 0.5710 | 0.8009 | 1.740 | 3 | 0.7654 | +0.0020 |
| 3 | 0.8051 | 0.6330 | 0.7628 | 1.480 | 3 | 0.8027 | -0.0024 |
| 4 | 0.7341 | 0.3573 | 0.7803 | 0.980 | 9 | 0.7394 | +0.0054 |

B0 fold-mean Dice: **0.7705 ± 0.0238** (population SD across five fold means).

## 250-case OOF aggregate

| Metric | B0-100 | KamelRep-100 | Kamel − B0 |
|---|---:|---:|---:|
| Mean Dice | 0.7707 | 0.7711 | +0.0004 |
| Median Dice | 0.8439 | 0.8491 | +0.0052 |
| Dice SD | 0.2157 | 0.2087 | -0.0070 |
| Recall | 0.5475 | 0.5585 | +0.0111 |
| Precision | 0.7551 | 0.7119 | -0.0433 |
| F1 | 0.6347 | 0.6260 | -0.0088 |
| FP/case | 1.6680 | 2.1240 | +0.4560 |
| FN/case | 4.2520 | 4.1480 | -0.1040 |
| FP volume/case | 0.2229 | 0.2828 | +0.0599 |
| Volume MAE | 6.1770 | 7.4971 | +1.3200 |
| Median Pred/GT | 0.9456 | 0.9167 | -0.0289 |
| Severe under | 25.0000 | 28.0000 | +3.0000 |
| Large partial coverage | 4.0000 | 6.0000 | +2.0000 |
| Dice<0.5 | 23.0000 | 23.0000 | +0.0000 |
| Dice<0.3 | 15.0000 | 15.0000 | +0.0000 |
| 1-5 mL Dice | 0.7375 | 0.7451 | +0.0076 |
| 1-5 mL Recall | 0.6683 | 0.6659 | -0.0024 |

## Paired case result

Across 250 paired cases (248 with defined Dice; two empty-GT cases are not assigned a Dice delta), mean/median ΔDice (Kamel − B0) was +0.0004/+0.0017; 25 cases improved by >0.03, 29 regressed by >0.03, and 194 were stable within ±0.03.

Strongest apparent beneficiaries:

- `case_0129` (Fold 2, GT 1.080 mL): ΔDice +0.5376
- `case_0037` (Fold 4, GT 0.488 mL): ΔDice +0.3724
- `case_0008` (Fold 4, GT 0.752 mL): ΔDice +0.2894
- `case_0181` (Fold 0, GT 16.208 mL): ΔDice +0.1445
- `case_0103` (Fold 2, GT 0.192 mL): ΔDice +0.1061
- `case_0093` (Fold 0, GT 0.336 mL): ΔDice +0.0973
- `case_0141` (Fold 1, GT 0.120 mL): ΔDice +0.0906
- `case_0085` (Fold 2, GT 29.048 mL): ΔDice +0.0798
- `case_0143` (Fold 4, GT 2.640 mL): ΔDice +0.0762
- `case_0056` (Fold 1, GT 0.672 mL): ΔDice +0.0709

Strongest regressions:

- `case_0045` (Fold 2, GT 0.768 mL): ΔDice -0.3434
- `case_0130` (Fold 4, GT 4.392 mL): ΔDice -0.3297
- `case_0202` (Fold 4, GT 7.367 mL): ΔDice -0.1954
- `case_0064` (Fold 3, GT 0.336 mL): ΔDice -0.1867
- `case_0076` (Fold 4, GT 6.056 mL): ΔDice -0.1800
- `case_0016` (Fold 1, GT 0.208 mL): ΔDice -0.1448
- `case_0173` (Fold 1, GT 34.528 mL): ΔDice -0.1431
- `case_0138` (Fold 2, GT 0.200 mL): ΔDice -0.1326
- `case_0172` (Fold 2, GT 5.824 mL): ΔDice -0.1184
- `case_0220` (Fold 0, GT 5.440 mL): ΔDice -0.0830

## Failure transitions

| Failure type | Rescued | Newly introduced | Both fail | Net rescued |
|---|---:|---:|---:|---:|
| severe_under | 3 | 6 | 22 | -3 |
| large_partial | 0 | 2 | 4 | -2 |
| small_lesion_miss | 3 | 2 | 25 | +1 |
| multifocal_miss | 7 | 4 | 154 | +3 |
| dice_lt_0p3 | 3 | 3 | 12 | +0 |

Summed across the five explicitly requested failure flags, Kamel rescued 16 flag instances and introduced 17. The largest net improvement was `multifocal_miss` (+3); the weakest/worst was `severe_under` (-3). Counts can overlap because one case may have multiple failure labels.

## Direct answers

1. **B0 Fold 1–4 是否全部完成？** 是；四個 fold 均完成 100 epochs、final checkpoint inference 與固定評估。Fold 0 通過 reuse gate，未重跑。
2. **B0 5-fold Mean ± SD？** 0.7705 ± 0.0238（五個 fold mean 的 population SD）。
3. **B0 250-case OOF performance？** Mean Dice 0.7707、Recall 0.5475、Precision 0.7551、F1 0.6347。
4. **KamelRep OOF performance？** Mean Dice 0.7711、Recall 0.5585、Precision 0.7119、F1 0.6260。
5. **Mean Dice 相差多少？** Kamel − B0 = +0.0004。
6. **Recall 相差多少？** +0.0111。
7. **Precision / F1 相差多少？** -0.0433 / -0.0088。
8. **Severe under 相差多少？** B0 25、Kamel 28，差 +3。
9. **Large partial 相差多少？** B0 4、Kamel 6，差 +2。
10. **Small lesion performance 差多少？** 1–5 mL Dice +0.0076；Recall -0.0024。
11. **哪些 cases 明顯受益？** 見上方 strongest beneficiaries，完整清單在 paired CSV。
12. **哪些 cases 明顯退步？** 見上方 strongest regressions，完整清單在 paired CSV。
13. **Fold-to-fold improvement 是否一致？** Kamel 的 Mean Dice 在 3/5 folds 較高；fold deltas 為 [0.0069, -0.0097, 0.002, -0.0024, 0.0054]。
14. **Technical configuration 有哪些實質差異？** 有效設定差異數為 0；只有 trainer wrapper/class 名稱不同，目的僅是讓原 Fold-0-only split selector 可選 Fold 1–4。
15. **Kamel-style replication 是否造成 gain？** 不可歸因為方法 gain；兩組設定相同。本次證據分類為 **C. STOCHASTICALLY EQUIVALENT**，代表觀察到的是獨立訓練軌跡差異。
16. **下一步？** 保留 OOF 較佳的 baseline 作為工程基準，但研究結論應先聚焦 paired/failure stability；若要宣稱改良，下一輪必須引入單一、可稽核的 configuration change。這一輪不自動啟動後續實驗。

## Output index

- `B0_100_OOF_all_cases.csv` / `B0_100_OOF_summary.json`: B0 exact 250-case OOF.
- `B0_100_vs_KamelRep_100_summary.csv`: requested aggregate comparison.
- `paired_B0_vs_Kamel.csv`: requested 250-case paired table and Dice-change class.
- `failure_transition_B0_vs_Kamel.csv` / `failure_transition_summary.csv`: case-level and aggregate failure transitions.
- `foldwise_B0_vs_Kamel.csv`: requested fold consistency comparison.
- `B0_vs_Kamel_configuration_audit.md`: exact pre-training configuration audit.
