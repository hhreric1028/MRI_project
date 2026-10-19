# A1-100 — DWI Intensity / Contrast Augmentation

## Executive conclusion

**Decision: STOP.** A1-100 does not support the targeted hypothesis strongly enough to promote this augmentation recipe. Mean Dice was essentially unchanged (0.7645 → 0.7633) and lesion recall improved slightly (0.5942 → 0.6031), but the prespecified target `case_0140` became materially worse: Dice 0.5372 → 0.4420, Pred/GT 0.5306 → 0.3514, voxel recall 0.4111 → 0.2987, and largest-component coverage 0.4104 → 0.2977. Volume MAE also worsened (4.521 → 5.835 mL).

The primary comparison is B0-100 versus A1-100. B0-1000 mean Dice 0.7782 is provided only as a non-fair duration reference and is not used for the decision.

## Controlled change

nnUNet v2.8.1 B0 defaults were: Gaussian noise variance 0–0.1 at p=0.10; Gaussian blur sigma 0.5–1.0 at p=0.20 (p/channel=0.50); multiplicative brightness 0.75–1.25 at p=0.15; contrast 0.75–1.25 at p=0.15; inverted gamma 0.7–1.5 at p=0.10; non-inverted gamma 0.7–1.5 at p=0.30, both retaining statistics.

A1 used one instance of each transform: noise variance 0–0.1 at p=0.15; brightness narrowed to 0.85–1.15 at p=0.25; contrast retained at 0.75–1.25 with p=0.30; non-inverted gamma retained at 0.7–1.5 with p=0.50 and retained statistics. The inverted-gamma branch was disabled. Gaussian blur and all spatial/low-resolution/mirror transforms were unchanged.

All other controls remained B0-identical: DWI-only Fold 0, 2 mm spacing, PlainConvUNet 32/64/128/256/320, patch 80×96×80, batch 8, foreground sampling 0.33, Dice+CE, SGD, and the original 1000-epoch PolyLR trajectory sampled for exactly 100 actual epochs.

## Augmentation sanity check

The preview used `case_0023` (large lesion), `case_0225` (2.640 mL), and `case_0248` (multifocal), with eight changed intensity draws per case. All 24 outputs were finite, all intensity arrays changed, and segmentation-mask hashes were byte-identical. Visual review found the lesion visible in every panel without inversion, geometric shift, or implausible corruption. Files are in `outputs\analysis\A1_augmentation_preview`.

## Fair B0-100 versus A1-100

| Metric | B0-100 | A1-100 | Delta |
|---|---:|---:|---:|
| Mean Dice | 0.7645 | 0.7633 | -0.0012 |
| Median Dice | 0.8239 | 0.8205 | -0.0034 |
| Lesion recall | 0.5942 | 0.6031 | +0.0089 |
| Lesion precision | 0.7147 | 0.7083 | -0.0063 |
| FP lesions/case | 2.140 | 2.240 | +0.100 |
| FN lesions/case | 3.660 | 3.580 | -0.080 |
| 1–5 mL mean Dice | 0.6262 | 0.6291 | +0.0029 |
| Severe under | 8 | 8 | +0 |
| Dice <0.5 | 4 | 6 | +2 |
| Dice <0.3 | 3 | 3 | +0 |
| Volume MAE (mL) | 4.521 | 5.835 | +1.314 |
| Mean Pred/GT | 0.8159 | 0.8083 | -0.0076 |
| Median Pred/GT | 0.8471 | 0.8346 | -0.0124 |

## Target case analyses

### case_0140

GT volume is fixed at 141.368 mL. B0-100 → A1-100: prediction volume 75.016 → 49.672 mL; Pred/GT 0.531 → 0.351; Dice 0.537 → 0.442; largest-component coverage 0.410 → 0.298. This is a clear regression in partial coverage. The B0-1000 values in `case_0140_comparison.csv` are context only.

Missed-region probability median changed 0.000144824 → 1.78481e-05; P<0.2 changed 93.1% → 96.4%. Even if a distribution statistic shifts upward, the hard segmentation and coverage became worse, so this does not count as target success.

### case_0201

Dice changed 0.850 → 0.797; Pred/GT 1.286 → 1.387; largest-component coverage 0.974 → 0.950; lesion recall 0.364 → 0.182. A1 therefore did not selectively rescue 0140 and also degraded the previously strong comparison case.

## Small-lesion and failure-mode findings

The 1–5 mL subgroup mean Dice changed 0.6262 → 0.6291, lesion recall 0.7333 → 0.7667, lesion precision 0.8980 → 0.8214, FN lesions 16 → 14, and severe-under cases 5 → 4. This is a mixed small-lesion result: detection/recall improved slightly, but subgroup precision fell by 0.0765.

Failure-mode labels changed as follows: small-lesion miss 5 → 5, large partial under-segmentation 3 → 4, multifocal miss 35 → 34, boundary under-segmentation 9 → 9, low-confidence 3 → 0, and FP-dominant 0 → 0; primary mixed cases changed 15 → 13. The low-confidence reduction is not evidence of rescue here: in `case_0140`, missed-region probabilities moved downward into a more background-confident pattern while hard coverage worsened.

## Decision rule and trade-off

The “mean Dice not obviously lower” guard passed, as did the precision and FP guards. Two screening positives were present: lesion recall increased and 1–5 mL Dice did not decrease. However, severe-under count did not fall, volume MAE worsened, case_0140 target coverage/Pred/GT/Dice worsened, and case_0201 also regressed. Because the experiment was specifically designed to improve the 0140-type low-salience partial-coverage failure, these results support **STOP**, not promotion to A1-300.

Training stopped at exactly 100 epochs. There was no 300/1000-epoch continuation and no subsequent experiment. Eight isolated AMP overflow steps were safely skipped among 25,000 training iterations (0.032%); there were no unrecovered consecutive overflows.

## Answers to the 11 requested questions

1. **A1 實際增加了哪些 augmentation？** 提高 non-inverting gamma、contrast、brightness 與 mild noise 的套用機率；brightness 幅度縮至 0.85–1.15；gamma 0.7–1.5 並保留 statistics；關閉 inverted-gamma branch。沒有重複套用相同 transform，也沒有強化 blur。
2. **Augmentation 是否保持 DWI lesion morphology 合理？** 是。三種病例共 24 張 preview 皆保留病灶可見性，mask hash 完全一致，未見幾何改變或不合理影像。
3. **A1 是否改善 overall segmentation？** 否。Mean Dice 0.7645 → 0.7633，基本持平但略降；Volume MAE 明顯變差。
4. **A1 是否改善 lesion recall？** 小幅改善：0.5942 → 0.6031（+0.0089）。
5. **A1 是否改善 small lesion？** 結果混合：1–5 mL mean Dice 小幅上升 0.6262 → 0.6291，lesion recall 上升、FN 減少，但 lesion precision 由 0.8980 降至 0.8214。不是無代價的改善。
6. **A1 是否改善 severe under-segmentation？** 否；病例數 8 → 8，沒有下降。
7. **case_0140 是否改善？** 否，Dice、Pred/GT、voxel recall、prediction volume 與 largest-component coverage 全部惡化。
8. **0140 missed-region probability 是否上升？** 否。中位數 0.000144824 → 1.78481e-05，下降約 8 倍；P<0.2 由 93.1% 升至 96.4%。soft probability 與 hard coverage 同方向惡化。
9. **是否產生 precision / FP trade-off？** 有輕微 trade-off：lesion precision 0.7147 → 0.7083，FP/case 2.14 → 2.24；不是崩潰，但方向不利。
10. **最終判定：GO / HOLD / STOP？** **STOP**。A1 沒有改善主要 target failure，且 0140、0201 與 volume MAE 都惡化，不應延長到 300 epochs。
11. **下一步最多一個建議。** 保留 B0 為主線，先由人工審閱本次 0140/0201 probability 與 coverage 結果後，再決定一個不同機制的單變因；不要延長目前 A1 recipe。

## Output index

- `comparison.csv`: fair B0-100 versus A1-100 metrics.
- `case_0140_comparison.csv`, `case_0201_comparison.csv`: target/control diagnostics.
- `small_lesion_subgroup_comparison.csv`: 1–5 mL subgroup.
- `failure_mode_comparison.csv`, `failure_mode_cases.csv`: V2 classifications.
- `probability_analysis.csv`, `probability_analysis.json`: missed-region probability statistics and bins.
- `validation_metrics.json`, `decision.json`, `paired_cases.csv`: complete fixed-analysis support.
