# 專案資料夾指南

所有下列路徑皆相對於 MRI_project_v1。2026-09-09 整理完成。

```text
MRI_project_v1/
├── README.md                   最新狀態與結果入口
├── environment.yml             Conda 環境
├── requirements.txt            Python 套件版本
├── configs/                    預留設定
├── notebooks/                  預留探索筆記
├── datasets/ISLES2022/
│   ├── raw/                    原始資料、GT、壓縮檔
│   ├── nnUNet_raw/             nnUNet 格式的 DWI／GT
│   ├── processed/              官方 plans、splits、預處理資料
│   └── documentation/          來源、授權、病例對照及原始 hashes
├── docs/
│   ├── training_report.md      最終訓練報告
│   ├── training_plan.md        已執行計畫
│   ├── preprocessing_report.md 預處理歷史報告
│   ├── phase1_report.md        資料準備歷史報告
│   ├── visual-review.md        原始資料 QA
│   ├── environment/            安裝說明、版本、環境決策
│   ├── logs/                   安裝、下載、解壓紀錄
│   └── reference/              官方程式碼快照與歷史目錄樹
├── scripts/                    操作入口（見下表）
├── src/
│   ├── data/                   資料檢查、轉換、原始 QA
│   ├── preprocessing/          預處理 QA
│   ├── analysis/               單一 GT 病灶體積測試
│   └── training,inference,utils/ 預留
└── outputs/
    ├── metrics/
    │   ├── data/               資料清單與檢查紀錄
    │   ├── preprocessing/      預處理結果、QA 與 log
    │   └── training_fold0/     最終數據、執行狀態、校驗及 log
    ├── figures/
    │   ├── data/               原始影像 QA
    │   ├── preprocessing/      預處理 QA
    │   └── training_fold0/     GT／prediction 疊圖
    ├── models/                 官方 nnUNet 實驗目錄
    ├── predictions/            預留；本次預測在 models/.../fold_0/validation
    └── .matplotlib/            繪圖快取
```

| 用途 | 操作入口 |
|---|---|
| 設定 nnUNet 路徑 | `scripts/set_paths.ps1` |
| 下載／解壓原始資料 | `scripts/download_isles2022.py`、`scripts/extract_isles2022.py` |
| 原始資料檢查／轉換／疊圖 | `src/data/inspect_isles2022.py`、`convert_isles2022.py`、`visualize_samples.py` |
| 官方資料完整性檢查 | `scripts/verify_dataset.py` |
| 預處理 QA | `src/preprocessing/check_preprocessed_cases.py` |
| 歷史報告產生器 | `scripts/report_phase1.py`、`scripts/report_preprocessing.py` |
| Fold 0 訓練／監控／評估 | `scripts/start_fold0.py`、`scripts/watch_fold0.py`、`scripts/evaluate_fold0.py` |
| 單一 GT 體積測試 | `src/analysis/test_single_lesion_volume.py` |
| 資料檢查測試 | `scripts/test_data_checks.py` |

目前訓練已完成，查看結果不需重新執行訓練或評估。評估程式會重新產生數據與報告草稿，覆寫人工補充的報告；僅需要重新計算時使用。

整理保留原始資料、預處理資料與官方模型目錄原位，操作腳本的入口也保留。移動的 40 個檔案均在搬移時驗證 SHA256 相同；舊新路徑對照保存在 `docs/reorganization_manifest.json`。之後更新過的說明文件可能與搬移時 hash 不同。

歷史 log、JSON 稽核證據與 `docs/reference/directory-tree.txt` 保留當時內容，其中舊路徑請用搬移對照查找。最新結構以本文件為準。`__pycache__` 與 `.matplotlib` 是程式自動產生的快取，不是實驗結果。
