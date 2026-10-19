# B0-R2022 正式續訓前驗證報告

2026-09-15 12:10（Asia/Taipei）。Resume verification 已通過，可以依授權進入正式續訓。本報告於正式續訓啟動前建立。

## 1. Checkpoint schema 與可恢復狀態

原檔實際 keys：`network_weights`、`optimizer_state`、`grad_scaler_state`、`logging`、`_best_ema`、`current_epoch`、`init_args`、`trainer_name`、`inference_allowed_mirroring_axes`。

| State | Present | Restorable | Notes |
|---|---|---|---|
| Model | 是 | 是，已驗證 | 240 tensors；載入後逐 tensor 與原檔完全相同且 finite |
| Optimizer | 是 | 是，已驗證 | 86 state tensors；SGD momentum 與 param groups 完整恢復、值相同，tensors 位於 CUDA device |
| GradScaler | 是 | 是，已驗證 | 完整 state 相同：scale=1,048,576、growth_factor=2、backoff_factor=0.5、growth_interval=2000、growth_tracker=500 |
| Epoch | 是 | 是 | current_epoch=34；歷史 train／validation logging 各34筆 |
| LR scheduler | 無獨立 state_dict | 可重建，已驗證 | 原 PolyLR：initial_lr=0.01、max_steps=1000、exponent=0.9；由絕對 epoch 位置恢复 |
| Trainer-specific state | 是 | 是 | logging、best EMA、init_args、mirroring axes 已載入 |
| RNG | 否 | 否 | 原 checkpoint 沒有完整 Python／NumPy／Torch／CUDA RNG states |
| DataLoader worker state | 否 | 否 | 沒有 worker RNG 或 iterator／prefetch 進度 |

## 2. Epoch 語意與起點

nnUNet 在 epoch 結束保存 `current_epoch+1`。來源 checkpoint 的34表示已完成內部 indices0–33（人類編號第1–34個epochs）；正式續訓從內部 index34，也就是第35個epoch開始，直到 index999／第1000個epoch。前34epochs不重跑。

## 3. LR continuity

```text
LR before interruption: 0.009702503915514164
Expected LR at resume:  0.009693473196581535
Actual LR after resume: 0.009693473196581535
```

前值是 checkpoint 保存的第34個epoch LR（step33）；後值是第35個epoch LR（step34）。兩者符合原始1000epoch PolyLR公式。原失敗 log 在進入 index34時也印出0.00969。没有把续训起点视为新的step0。

## 4. Gradient check 修改

原檢查已位於 `unscale_()` 之後；問題是 `error_if_nonfinite=True` 提早終止，阻斷 scaler.step/update。

新 subclass 維持一次 unscale，再檢查 unscaled gradients。正常 gradients 使用原 norm12 clipping。非有限 gradients 保留給 GradScaler 跳過更新及降倍率，並以 optimizer post-step hook 確認實際有無更新。每次更新後檢查 model／optimizer finite。原 trainer、官方 source、loss、sampling、SGD、LR schedule、augmentation、preprocessing 均未修改。

單次 overflow 不終止；連續16次 skip 且仍未恢復、或 loss／model／optimizer nonfinite 等情況才停止。16次為明確記錄的持續異常監控門檻，不是新增模型超參數。

## 5. 實際 resume smoke test

- 完成20次warm-up＋100次實測，共120次。
- 每步 loss finite；119步 unscaled gradients finite，1步出現可恢復 overflow。
- 第74次iteration：loss=−0.677904，gradient norm=Inf；實際 optimizer step skipped=true，scale 1,048,576→524,288。
- 第75次iteration：gradients恢復 finite、正常更新；gradient norm約0.24。後續完成所有剩餘步驟。
- Model／optimizer state 全程 finite；結束時沒有未恢復的連續overflow。
- Validation step loss=−0.583117，finite。
- 實測100次耗時19.891秒；每iteration0.19891秒。
- 峰值 allocated5.26GiB、reserved6.18GiB。
- 結束scale=524,288；正式續訓將重新讀取原checkpoint，恢復原scale=1,048,576，並非接續smoke模型。

這次實際觀察到正常的AMP skip／降倍率／恢復，支持新的處理流程可運作；原中斷batch未保存，仍不宣稱已精確重現原先錯誤batch。

## 6. RNG／worker限制

`This run is a training-state continuation, not a bitwise deterministic continuation.`

維持seed2022與既有worker規則：train2022–2025、validation3022–3023。第35epoch會建立新的worker streams；無法重建不中斷時的原batch sequence，也不會用重新設定seed聲稱恢復了遺失的state。後續controlled same-seed comparison須標註本run曾中斷／續訓。

## 7. 來源保全與版本

- 原checkpoint：`C:/Users/User/Desktop/MRI_project_v1/experiments/recall_2mm_seed2022/B0-R2022/train/results/Dataset001_ISLES2022_DWI/nnUNetTrainer_B0R2022__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth`
- SHA256：`f67ae113f4a68bad8e4ed6da31191f3497d7588bfa2c7a1c291e75dff1f865bd`
- 原run所有檔案經前後hash核對一致；baseline1923檔案也完全一致、無新增檔案。
- 新正式輸出：`C:/Users/User/Desktop/MRI_project_v1/experiments/B0-R2022_resume_from_epoch34/train`
- 新smoke輸出：`C:/Users/User/Desktop/MRI_project_v1/experiments/B0-R2022_resume_from_epoch34/smoke`
- Branch：`fix/amp-safe-resume`
- Commit：`54f514a495bda207d35785082f4ac12930eb0ac9`
- Trainer：`nnUNetTrainer_B0R2022_Resume`；來源hash與完整環境保存在各stage的resume_config.json。

## 8. 正式續訓判定

必要state恢復、LR、實際AMP恢復、數值穩定性、速度／VRAM及檔案保全檢查均通過，可啟動正式續訓。正式階段重新載入原checkpoint，獨立驗證state與LR；第35–37epochs額外保存loss、validation loss、pseudo-Dice、LR、scale、skipped steps、nonfinite events與mean/max unscaled gradient norm。所有新增checkpoint位於新run目錄。
