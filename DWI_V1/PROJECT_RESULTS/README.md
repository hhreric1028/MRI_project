# MRI 專案結果整理入口

這個資料夾將專案結果依用途與**實際完成的 epoch 數**分開。建議先從：

1. `01_COMPLETE_COMPARISONS/01_PRIMARY_FULL_5FOLD/B0_100_vs_KamelRep_100_FINAL/final_report.md`
2. `02_100_EPOCH_ONLY/`
3. `03_1000_EPOCH/`

開始閱讀。

## 分類原則

| 資料夾 | 內容 |
|---|---|
| `01_COMPLETE_COMPARISONS` | 已完成的比較與正式分析報告；主要五折比較和其他較小範圍分析分開。 |
| `02_100_EPOCH_ONLY` | checkpoint 實際停在 epoch 100 的 runs。即使 LR scheduler 保留 1000-epoch denominator，仍歸在這裡。 |
| `03_1000_EPOCH` | checkpoint 的 `current_epoch=1000`，包含 Original B0、B0-R2022 resume 與 P1 spacing。 |
| `04_300_EPOCH` | 實際完成到 epoch 300 的 promotion runs。 |
| `05_INCOMPLETE_FAILED_ARCHIVE` | 首次失敗、未完成或只作保護用途的舊路徑；不應當成正式結果。 |
| `06_AUXILIARY_ANALYSES` | failure、volume、connected-component、位置與 qualitative review 等輔助分析。 |

## 重要說明

- 此處是**整理視圖**，使用 Windows directory junction 指向原始資料；checkpoint、prediction 和報告沒有複製，也沒有搬動。
- 原始路徑保留，所以既有 scripts、JSON provenance 與 hard-coded paths 不會失效。
- 透過 junction 修改或刪除內容會同時影響原始資料；本資料夾不是獨立備份。
- `01_COMPLETE_COMPARISONS` 是依「分析是否完成」分類；`02`、`03`、`04` 則依「實際訓練 epoch」分類，因此同一實驗可能同時出現在兩種視圖中，但不占用雙倍空間。

## 目前主要結論

- Original B0-100 與 KamelRep-100 的五折設定沒有實質技術差異。
- 250-case OOF Mean Dice：B0 `0.7707`、KamelRep `0.7711`，差 `+0.0004`。
- 最終判定：`C. STOCHASTICALLY EQUIVALENT`。
- 1000-epoch results 是 Fold 0 或單一 run，不能和 100-epoch 五折 OOF 當作 matched comparison。

完整清單與原始目標位置見 [inventory.csv](inventory.csv)。
