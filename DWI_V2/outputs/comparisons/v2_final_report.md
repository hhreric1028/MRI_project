# DWI_V2 controlled-ablation final report

Final classification: **STOP**

No 1000-epoch run, combined E1+S1 run, threshold tuning, or additional intervention was started.

Numerical note: E1 had 8 isolated AMP skipped steps and 0 unrecovered consecutive overflows. The process and checkpoint completed, but the prespecified zero-event numerical-stability guard failed.

## Candidate comparison

| Run | Classification | Mean Dice | Lesion recall | Lesion precision | FP/case | Severe-under | Fixed Worst-5 Dice | Coverage-cohort recall | Persistent FN recovered (mL) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-E1-100 | RECALL-PRECISION TRADE-OFF | 0.7709 | 0.6452 | 0.6554 | 3.06 | 6 | 0.2986 | 0.3007 | 0.104 |

## Ten required conclusions

1. **V2-E1 是否恢復 persistent FN？** 恢復 0.104 mL persistent FN；分類為 RECALL-PRECISION TRADE-OFF。
2. **V2-S1 是否提高 low-salience exposure？** 未通過：audit 標記 `SAMPLING INTERVENTION INEFFECTIVE`，因此依方案未啟動 V2-S1 長訓練。
3. **case_0140 是否泛化？** case_0140 recall 0.4111→0.3335，未改善；broader cohort recall +0.0278，但 95% CI 跨 0，因此只有弱的 cohort signal。
4. **Fixed Worst-5 是否改善？** 有方向性改善：Mean Dice 0.2814→0.2986（Δ+0.0172），recall Δ+0.0168；兩者 bootstrap 95% CI 均跨 0。
5. **Severe under 是否下降？** 有下降：8→6 cases；rate Δ-0.040，bootstrap 95% CI [-0.120, +0.040]。
6. **是否只是 recall↑ / FP↑？** 是。Lesion recall Δ+0.0510（95% CI +0.0225 至 +0.0800），但 precision Δ-0.0593（95% CI -0.1115 至 -0.0136）、FP/case Δ+0.92（95% CI +0.36 至 +1.66），且 volume MAE +0.735 mL。
7. **Original+P3a 是否比 P3a-only 平衡？** 相對 P3a-only-100：Mean Dice -0.0130、lesion recall +0.0067、precision -0.0336、FP/case +0.46。
8. **Hard-region sampling 是否優於一般 oversampling？** V2-S1 未進入正式訓練，故不能聲稱優於 S1a。
9. **最值得保留的 intervention？** 無 intervention 通過全部 promotion guards。
10. **是否足以進入 full training？** 最終分類為 STOP；本輪不授權也未執行 1000 epochs。

## Stop boundary

Workflow stopped at 100 epochs as required.
