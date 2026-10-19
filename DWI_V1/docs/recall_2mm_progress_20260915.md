# 2 mm recall 執行進度 — 2026-09-15

- 已保留所有既有 training／evaluation／sampling 結果、checkpoint 及 Terminal。接續前唯讀程序檢查未見正在執行的 Python 實驗。
- 追加 protocol 已實作，固定 trainer／loss／seed／前處理未因追加內容變更。
- 6 項 protocol tests 與 3 項 evaluation-v2 tests 通過。涵蓋 Tversky 五情境、FP/FN 方向、四尺度穩定性、同種子設定、實際 loader 分離、幾何 coverage 與 paired bootstrap identity。
- 前後兩次 baseline audit：1,923 檔案全部一致，無新增受保護檔案。
- 環境與歷史 P1 的 PyTorch／nnUNet 版本相同；baseline augmentation 表示完全相同。歷史 B0 seed、當時來源 hash 及缺少的 deterministic flags 仍列為未提供，不以目前環境補填。
- 初次環境 provenance 因 namespace package 沒有 __file__ 而失敗，已保存 preflight_failure.json；只補做尚未完成的環境核對與 failure analysis，沒有重跑已完成的測試、training 或 sampling。
- P1／B0 的 50 例配對結果、10,000 次 case bootstrap、五組分層、13 例 failure panels 及實際視覺檢查完成。
- 下一步依序：B0-R2022 GPU feasibility20+100 → 通過才正式1000epochs → final validation/evaluation → sampling audit → 實質 exposure 審核 → S1a 決策 → L2a → 配對比較。
- Sampling audit 不再用兩組完全相同與否決定長訓練；實際 exposure 結果需審核，尚未核准的實質增幅門檻不會自行捏造。
- 所有正式執行以 experiments/recall_2mm_seed2022 下的 started/completed/failure 和 training log 為即時依據；本文件不是完整訓練完成宣告。

## 01:02 更新

B0-R2022 GPU feasibility 已通過：20 次 warm-up、100 次實測；每迭代 0.1925 秒，峰值 allocated 5.26 GiB／reserved 6.12 GiB，loss／gradients finite，試跑後 1,923 個 baseline hashes 全部一致。

正式 B0-R2022 已於 01:02 進入 epoch 0（nnUNet 零起算），使用獨立 train 目錄；smoke 結果完整保留，並非接續試跑權重。正在執行的 run 不會因後續補充而修改或重啟。可見 Terminal 已開啟，stdout／stderr 分階段保存。

GPU run 實際啟動紀錄：Python／NumPy／Torch／CUDA seeds=2022；train workers2022–2025、validation workers3022–3023；cuDNN deterministic=False、benchmark=False、deterministic algorithms=False。這是 controlled same-seed comparison，不是 fully deterministic。

目前只有 B0-R2022 正式訓練。正常完成後同一流程才做 final validation／evaluation 與 sampling audit；S1a、L2a 尚未啟動。Sampling 結束後需依實際 exposure 結果作決策，之後才進入候選訓練階段。P2／P3／focal／multimodal／組合／additional seeds 均不啟動。
