# DWI_V2 controlled-ablation final report

Final classification: **STOP**

No 1000-epoch run, combined E1+S1 run, threshold tuning, or additional intervention was started.

## Candidate comparison

| Run | Classification | Mean Dice | Lesion recall | Lesion precision | FP/case | Severe-under | Fixed Worst-5 Dice | Coverage-cohort recall | Persistent FN recovered (mL) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-E1-100 | RECALL-PRECISION TRADE-OFF | 0.7709 | 0.6452 | 0.6554 | 3.06 | 6 | 0.2986 | 0.3007 | 0.104 |

## Ten required conclusions

1. **V2-E1 是否恢復 persistent FN？** 恢復 0.104 mL persistent FN；分類為 RECALL-PRECISION TRADE-OFF。
2. **V2-S1 是否提高 low-salience exposure？** 未通過：audit 標記 `SAMPLING INTERVENTION INEFFECTIVE`，因此依方案未啟動 V2-S1 長訓練。
3. **case_0140 是否泛化？** 無候選通過 promotion guards。
4. **Fixed Worst-5 是否改善？** 沒有通過規定的綜合改善門檻。
5. **Severe under 是否下降？** 沒有可 promotion 的下降訊號。
6. **是否只是 recall↑ / FP↑？** 以 precision drop≤0.05 與 FP/case increase≤0.5 的固定 safety guards 判定；只有全部通過者可升級。
7. **Original+P3a 是否比 P3a-only 平衡？** 相對 P3a-only-100：Mean Dice -0.0130、lesion recall +0.0067、precision -0.0336、FP/case +0.46。
8. **Hard-region sampling 是否優於一般 oversampling？** V2-S1 未進入正式訓練，故不能聲稱優於 S1a。
9. **最值得保留的 intervention？** 無 intervention 通過全部 promotion guards。
10. **是否足以進入 full training？** 最終分類為 STOP；本輪不授權也未執行 1000 epochs。

## Stop boundary

Workflow stopped at 100 epochs as required.
