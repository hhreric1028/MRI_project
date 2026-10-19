# B0-R2022 AMP-safe resume 執行狀態

更新：2026-09-15 12:15（Asia/Taipei）。正式續訓已完成第35–37個epochs，正在進入第38個epoch。原先34個epochs完整保留；1000epochs尚未完成。

## 驗證與啟動

- Source branch/commit：`fix/amp-safe-resume` / `54f514a495bda207d35785082f4ac12930eb0ac9`。
- 新trainer：`nnUNetTrainer_B0R2022_Resume`，原trainer／官方nnUNet程式未修改。
- Smoke：20warm-up＋100measured通過；第74iteration實際overflow，scale1,048,576→524,288，下一iteration恢復。Model／optimizer全程finite。0.19891秒／iteration，allocated5.26GiB，reserved6.18GiB。
- 正式續訓重新載入原checkpoint，model／optimizer／GradScaler逐項核對一致，optimizer tensors位於CUDA，scale恢復為原1,048,576。
- LR：checkpoint0.009702503915514164 → resume expected/actual0.009693473196581535。
- 正式訓練自2026-09-15 12:12的第35epoch開始，Terminal保持可見，stdout／stderr保存於新目錄。

## 前3epochs實測

| Epoch（人類編號） | Train loss | Validation loss | Patch pseudo-Dice | LR | Scale | Skipped steps | Nonfinite gradients | Mean/max finite unscaled norm |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 35 | -0.7606 | -0.7193 | 0.825 | 0.009693473197 | 524288 | 1 | 1 | 0.3408 / 1.9866 |
| 36 | -0.7627 | -0.7144 | 0.821 | 0.009684441543 | 262144 | 1 | 1 | 0.3121 / 1.3776 |
| 37 | -0.7515 | -0.7223 | 0.827 | 0.009675408953 | 262144 | 0 | 0 | 0.3765 / 1.8548 |

三個epochs均無nonfinite loss／model parameter／optimizer state事件。兩次overflow皆為單次skip後恢復，第37epoch沒有overflow。這些pseudo-Dice為training中的patch指標，並非完整3D validation Dice。早期結果支持續訓正常運作，不代表最終模型表現已確定。

## 保存與後續

原來源checkpoint SHA256：`f67ae113f4a68bad8e4ed6da31191f3497d7588bfa2c7a1c291e75dff1f865bd`。Smoke前後原run所有檔案hash一致，baseline1923檔案hash一致。正式的新checkpoint已生成於新run的results目錄，與原checkpoint完全分開。

實際狀態以 `experiments/B0-R2022_resume_from_epoch34/train/` 的 `early_epochs.jsonl`、`amp_events.jsonl`、`training_completed.json`、`completion.json`／`failure.json`，以及新root的 `logs/train.log` 為準。舊 `recall_2mm_seed2022/baseline_failure.json` 保留為歷史失敗紀錄，不代表新的resume已失敗。

正式run繼續至第1000epoch，再執行final validation。第35–37的額外摘要已完成，後續正常迭代維持一般logging，AMP overflow／nonfinite事件仍逐次記錄。完成前不把模型列為完整實驗結果，也不啟動S1a／L2a／P2等其他訓練。

This run is a training-state continuation, not a bitwise deterministic continuation. 原checkpoint缺少RNG／worker／iterator進度，新的worker streams維持原seed policy，不能聲稱重現原本不中斷時的batch sequence。

完整checkpoint schema、state可恢復表與續訓前14項回報見 `experiments/B0-R2022_resume_from_epoch34/pre_resume_report.md`。
