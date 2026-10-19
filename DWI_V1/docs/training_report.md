# ISLES2022 DWI-only fold 0 訓練報告

完成日期：2026-09-09（Asia/Taipei）。1,000 epochs、50 例完整影像驗證與指定 QA 已完成，官方程序 exit code 0。僅執行 fold 0，未啟動其他 folds。

## 設定與執行

沿用 docs/training_plan.md：MRI_DWI、nnUNetv2 2.8.1、PyTorch 2.8.0+cu128、RTX 5080。官方 nnUNetTrainer / nnUNetPlans / 3d_fullres / PlainConvUNet，DWI 單通道，spacing 2×2×2 mm、patch 80×96×80、batch 8、DA workers 4。使用官方預設 loss、optimizer、augmentation，無 pretrained weights 或 early stopping。

官方 seed 12345 五折分割，fold 0 訓練 200 例、驗證 50 例，原始 subject ID 無交集。三例空白 GT 均保留，case_0151 在本次驗證集。版本、設定及輸入校驗保存於 outputs/metrics/training_fold0/training_preflight.json；分折保存在 processed dataset 的 splits_final.json。

2026-09-08 16:09:30 啟動，2026-09-09 05:08:44 完成官方訓練與驗證，約 12 小時 59 分鐘。末期每 epoch 約 46 秒。

## 最終完整影像結果

固定採用 checkpoint_final.pth，已成功載入，current_epoch=1000，未按結果改選 best checkpoint。

- 完整影像驗證：50/50。
- Dice 平均：0.7781594639；中位數：0.8291330645；有效病例數：49。
- 病灶體積平均絕對誤差：4.509421468 mL（50 例）。
- case_0151 GT 與預測皆空，Dice=N/A，從 Dice 平均與中位數排除；假陽性體積 0 mL。

逐例結果：outputs/metrics/training_fold0/fold0_validation_cases.csv；摘要：outputs/metrics/training_fold0/fold0_validation_summary.json。獨立計算的平均 Dice 與官方 validation/summary.json 一致。

體積依原始 GT NIfTI header.get_zooms()[:3] 計算：voxel count × spacing_x × spacing_y × spacing_z / 1000，確認單位 mm，未假設原始 spacing 相同。評估器最初採 affine determinant；結案時改為明確的 header spacing 乘積並重新核對既有輸出，平均誤差差異約 0.00000115 mL，Dice 不變，未重新推論。

## 訓練曲線

最後一個 epoch training loss=-0.8897、validation loss=-0.777、patch-based pseudo-Dice=0.857。此 pseudo-Dice 不能當作 50 例完整影像 Dice。官方複合 loss 可為負值，全部 1,000 epochs loss 有限。

fold_0/progress.png 保存 loss、pseudo-Dice 及移動平均、epoch 耗時、learning rate 曲線；training log、debug.json、best/final checkpoints 保留。

## 視覺 QA

seed 2022 抽樣：0093、0094、0140、0168、0186。最低 Dice 五例：0068、0182、0181、0140、0212。去重共 9 例，每例 GT 最大面積切片及前後各一張，共 27 張。圖檔位於 outputs/figures/training_fold0，GT 綠色、prediction 紅色，使用 native voxel axes，未加解剖左右標籤。

| Case | Dice | GT mL | Prediction mL | 絕對誤差 mL | 所見 |
|---|---:|---:|---:|---:|---|
| case_0068 | 0.162055 | 3.688000 | 0.360000 | 3.328000 | 主要 GT 區域幾乎未覆蓋，明顯漏判。 |
| case_0182 | 0.205128 | 1.627295 | 0.185977 | 1.441318 | 所示三張切片未見紅色覆蓋；完整影像有少量預測。 |
| case_0181 | 0.295068 | 16.208000 | 3.096000 | 13.112000 | 僅涵蓋 GT 小部分，明顯低估。 |
| case_0140 | 0.471240 | 141.368000 | 53.624000 | 87.744000 | 廣泛 GT 中大片漏判，局部邊界誤判。 |
| case_0212 | 0.507614 | 1.104000 | 0.472000 | 0.632000 | 小病灶覆蓋不足，相鄰切片漏判。 |

其餘抽樣：0093 小病灶大致重合，有局部外擴；0094 主病灶重合但邊界與小區域有誤判；0168 主病灶覆蓋良好，周邊小區域同時有漏判與額外預測；0186 所示低位切片漏判明顯，部分區域在下一張才有覆蓋。

未見所示切片整體翻轉或平移錯位。這是有限切片的 agent 視覺檢查，不代表逐切片臨床審閱。低分如實保留，未設定任意 Dice 門檻宣稱有效。0140 大體積低估對平均體積誤差影響明顯；另 case_0201 高估約 29.071 mL，僅據數值記錄，未列入指定視覺樣本。

## 驗收與限制

outputs/metrics/training_fold0/fold0_final_audit.json 記錄：原始 1,688 檔 SHA256 未變、轉換後 500 檔未變、訓練前 753 個 preprocessed/config/split hashes 全部一致。50 個 segmentation 與 50 個 probability 檔齊全；預測 shape、spacing、affine、方向與原始 GT 相容，標籤僅 0/1。機率為兩通道、空間 shape 符合 native ZYX、全部有限且位於 [0,1]，通道和在 0.002 容差內為 1。final checkpoint 可載入，紀錄恰為 1,000 epochs。

這是單一 fold 內部驗證 baseline，非完整五折或外部測試。DWI-only 對若干病灶明顯漏判；有限 QA 無法涵蓋全部失敗模式。CUDA 不保證逐位元組可重現。後續 folds 或 DWI+ADC 需另行決定；本次完成後停止追蹤。
