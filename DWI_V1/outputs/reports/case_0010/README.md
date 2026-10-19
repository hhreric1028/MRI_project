# case_0010 Report Files

這個資料夾分成「主要閱讀檔案」與「技術支援檔案」。

## 主要閱讀檔案

- `clinical_report.txt`：Clinical-style Findings / Impression。日常查看報告時先開這份。
- `model_evaluation.txt`：Validation case 的 GT、體積誤差與 component 數；不是 clinical report。

## 技術支援檔案

- `data/structured_input.json`：報告產生器實際使用的結構化輸入。
- `validation/report_validation.json`：逐句來源、unsupported claims 與一致性檢查。
- `logs/report_generation.log`：最近一次報告產生紀錄。

## Archive

- `archive/20260916_pre_organization/`：整理前的 structured input、validation 與 log 備份。僅供追溯，不是目前版本。

目前有效版本以根目錄兩份報告，以及 `data/`、`validation/`、`logs/` 中的檔案為準。
