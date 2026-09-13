# ISLES2022 DWI-only：Fold 0 訓練計畫

已核准執行：fold 0 訓練、50 例完整影像驗證與結果分析；不啟動其他 folds。

## 固定設定
MRI_DWI；官方 nnUNetv2 2.8.1；Dataset001_ISLES2022_DWI；3d_fullres；nnUNetTrainer；nnUNetPlans。
DWI 單通道，背景/病灶兩類；PlainConvUNet；spacing 2×2×2 mm；patch 80×96×80；batch 8。
ZScoreNormalization/use_mask_for_norm=True；1000 epochs；每 epoch 訓練250/驗證50迭代。
官方 Dice+CE/deep supervision；SGD lr=.01 momentum=.99 Nesterov weight_decay=3e-5；PolyLR；CUDA mixed precision。
RTX 5080 單GPU；nnUNet_n_proc_DA=4；Windows 官方停用 compile。不修改架構、augmentation、loss、plans；不使用 pretrained weights 或 early stopping。

## 分折與啟動
官方 generate_crossval_split seed=12345、5 folds，僅執行 fold 0：200 train/50 val。
保存並核對 splits_final.json，確認 case 與原始 subject 無洩漏；保留3例空白GT。
核對路徑、250組preprocessed檔案、CUDA、版本與輸入hash。既有不一致splits或training輸出不得覆寫。

```powershell
conda activate MRI_DWI
cd C:\Users\User\Desktop\MRI_project_v1
. .\scripts\set_paths.ps1
$env:nnUNet_n_proc_DA = "4"
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans -device cuda --npz
```

結果：outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/。
前10 epochs觀察有限loss、GPU使用與worker健康；正常接續1000 epochs，以穩定epoch耗時估算剩餘時間。
保存官方log、progress、debug、best/final與預設50epoch checkpoint。
中斷僅在checkpoint存在且設定一致時以--c續訓；不存在時明確記錄須重啟。
OOM、NaN/Inf、資料讀取或worker錯誤立即停止保存錯誤，不自動縮batch、改plans、刪病例或更換模型。

## 評估與驗收
以checkpoint_final.pth完成50例官方完整驗證，保存segmentation、npz機率與summary.json。
分開報告patch pseudo-Dice及完整影像Dice；逐例Dice、平均、中位數與有效數量；體積絕對誤差使用原始spacing換算mL。
空白GT另外列出假陽性體積，雙空Dice=N/A，不補成1。
seed=2022抽5個validation病例，加Dice最低5例去重，GT/prediction不同色疊圖，檢查漏判、誤判與對齊。
驗收：1000epochs、final可載入、50例預測與機率完整有限、幾何符合GT、labels為0/1。
保存metrics、QA與docs/training_report.md，不設任意Dice門檻；完成即停止，不訓練fold1。

## 限制
單fold內部驗證，不是完整五折或外部測試；CUDA不保證逐位元組可重現。
