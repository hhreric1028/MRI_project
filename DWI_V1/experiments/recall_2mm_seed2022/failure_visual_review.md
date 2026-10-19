# P1 固定切片視覺檢查 — 2026-09-15

已逐一檢視去重後 13 例、每例兩個切片的 DWI／GT／B0／P1 panel。第一列為 GT 最大面積切片；第二列為 B0／P1 分歧最大切片。切片來自同一 native DWI，強度範圍一致；青色輪廓為 GT，黃色為該欄 mask。兩列選到同一切片時仍保留，以符合兩項選片規則。

本檢查是 assistant 的固定切片描述性檢視，不是醫師盲評，也不宣稱檢視完整 volume。Panel 第二列部分標題間距偏窄；確切 z index 保存在 failure_patterns.json，不影響 mask 判讀。所有原始 panel 與計算輸出保留不覆寫。

| Case | 固定切片觀察 |
|---|---|
| case_0010 | 多處小 GT 區域；P1 在部分切片比 B0 更少覆蓋。全 volume 統計中完全零重疊 GT components 由 1 增至 2。 |
| case_0051 | 分歧切片的 GT 區域，P1 覆蓋比 B0 完整；不能把單例改善推广為全部小病灶改善。 |
| case_0053 | 最大 GT 切片兩者皆定位；分歧切片可見 B0 在 GT 以外的黃色區域较多，P1 減少。Dice 改善不代表 voxel recall 同時提高。 |
| case_0068 | 最大 GT 區域在兩模型幾乎未被覆蓋；其他小區域仍有預測。整例 voxel recall B0 0.089、P1 0.067；屬嚴重漏覆蓋，非整例 prediction 完全空白。 |
| case_0069 | 小 GT 區域皆有定位，P1 的黃色覆蓋更小；主要是找到但覆蓋不足。 |
| case_0118 | 較大片 GT 區域 P1 出現更多內部缺口及邊界漏覆蓋；voxel recall 0.672→0.486。零重疊 GT components 2→3，另有額外 FP component。 |
| case_0135 | 最大面積切片仍有定位，但分歧切片的 GT 在 P1 幾乎未覆蓋；全 volume 三個 GT components 均仍有部分重疊，不能把切片漏失誤寫成整個 3D 病灶全漏。Voxel recall 0.632→0.434。 |
| case_0140 | 大範圍 GT 在兩模型均有明顯漏覆蓋，P1 分歧切片覆蓋更少；額外零重疊 FP components 6→0，但 voxel recall 0.325→0.305，呈現 FP 減少／coverage 未改善的取捨。 |
| case_0160 | 最大 GT 切片仍可定位；分歧切片同時有不足覆蓋與 GT 外預測。全 volume voxel recall 0.457→0.347，零重疊 GT components 1→2。 |
| case_0181 | 大部分 GT 仍缺乏覆蓋，但分歧切片 P1 補回部分 GT；voxel recall 0.176→0.208，改善後仍嚴重不足。 |
| case_0182 | B0 幾乎未覆蓋的 GT 區域，P1 補回其中一部分；voxel recall 0.114→0.362，仍不完整。 |
| case_0188 | 大片 GT 的部分區域 P1 覆蓋增加，但仍有漏覆蓋；voxel recall 0.638→0.684。 |
| case_0212 | 主 GT 區域兩者均覆蓋不足；P1 某切片覆蓋更小。零重疊 components 2→1，但 voxel recall 0.362→0.283，說明病灶定位數與覆蓋量可有不同方向。 |

## 與全 cohort 數據的整合

50 例 paired case bootstrap（10,000 次，seed=2022），P1−歷史 B0：

- Mean Dice Δ −0.013，95% CI [−0.032, 0.008]。
- Lesion recall Δ −0.049，95% CI [−0.081, −0.016]。
- Lesion precision Δ +0.031，95% CI [0.002, 0.058]。
- FP lesions/case Δ −0.480，95% CI [−0.800, −0.180]。
- Severe under-segmentation（Pred/GT<0.5）6→9 例。

本次 evaluation-v2 資料中，整例 overlap-based lesion recall=0 的病例數兩組皆為 0；不能因 case_0068 極低分，就稱它為「整例完全漏判」。整個 GT component 漏失、整例完全漏判，以及找到但嚴重覆蓋不足需分開使用。

P1 的較少 FP 伴隨 recall 下降；嚴重漏覆蓋包含小病灶以及 case_0140 這類大片 GT，不能只以 small-lesion problem 概括。這些觀察支持依原計畫回到 2 mm、建立同種子對照，再檢查 sampling exposure 與非對稱 FN penalty。圖與體積差異不證明 FOV 或影像異質性是原因。

歷史 B0 seed 未確認；B0／P1 本身不可稱為 controlled same-seed comparison。該名稱僅用於本輪 B0-R2022／S1a／L2a。所有結果為 fold0、單次篩選；bootstrap CI 僅反映目前 validation cohort 的病例不確定性。

前置檢查與視覺檢查完成，可依原計畫進入 B0-R2022 GPU feasibility；這不是 S1a 的 exposure 審核或任何模型優勝判定。
