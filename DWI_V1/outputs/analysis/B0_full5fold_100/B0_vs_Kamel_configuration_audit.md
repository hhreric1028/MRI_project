# B0-100 vs KamelRep-100 exact configuration audit

## Conclusion

**No effective technical configuration difference was found.** The only named difference is the trainer wrapper/class name. Both five-fold wrappers inherit the same `Screen100Mixin` and original `nnUNetTrainer_B0R2022`; the wrapper only permits selecting the already frozen Fold 0–4 entries from the same `splits_final.json`.

Therefore this comparison must be interpreted as **two independent five-fold trainings using the same configuration**, designed to estimate stochastic training variability. It must not be reported as a Kamel-method gain.

## Exact comparison

| Item | B0-100 | KamelRep-100 | Status | Note |
|---|---|---|---|---|
| trainer class | nnUNetTrainer_B0R2022_Screen100 (Fold 0); nnUNetTrainer_B0Full5fold_Screen100 (Fold 1-4 selector wrapper) | nnUNetTrainer_KamelRep_Screen100 | **DIFFERENT** | Class names differ. Both wrappers inherit Screen100Mixin + nnUNetTrainer_B0R2022; Fold 1-4 wrapper changes only do_split to select the audited fixed fold. |
| seed policy | Python/NumPy/Torch/CUDA=2022; train workers 2022-2025; val workers 3022-3023; non-deterministic arrival order | Python/NumPy/Torch/CUDA=2022; train workers 2022-2025; val workers 3022-3023; non-deterministic arrival order | **SAME** |  |
| split file/hash | splits_final.json / 09ed64d65dc3b5ece2656639a5abfc1a4f76b6e33121512a4931f6b6d6814d31 | splits_final.json / 09ed64d65dc3b5ece2656639a5abfc1a4f76b6e33121512a4931f6b6d6814d31 | **SAME** |  |
| nnUNet plans/hash | nnUNetPlans / 09a3894923f0437d714fd14e44c3728785b59d2c28d01d12d8f7681b38d8340e | nnUNetPlans / 09a3894923f0437d714fd14e44c3728785b59d2c28d01d12d8f7681b38d8340e | **SAME** |  |
| target spacing | [2.0, 2.0, 2.0] | [2.0, 2.0, 2.0] | **SAME** |  |
| patch size | [80, 96, 80] | [80, 96, 80] | **SAME** |  |
| batch size | 8 | 8 | **SAME** |  |
| architecture | dynamic_network_architectures.architectures.unet.PlainConvUNet | dynamic_network_architectures.architectures.unet.PlainConvUNet | **SAME** |  |
| feature channels | [32, 64, 128, 256, 320] | [32, 64, 128, 256, 320] | **SAME** |  |
| normalization | ["ZScoreNormalization"] | ["ZScoreNormalization"] | **SAME** |  |
| normalization mask | [true] | [true] | **SAME** |  |
| foreground sampling | train 0.33 / validation 0.33 | train 0.33 / validation 0.33 | **SAME** |  |
| loss | B0 Dice+CE, equal weights, batch Dice false | B0 Dice+CE, equal weights, batch Dice false | **SAME** |  |
| optimizer | SGD momentum 0.99, Nesterov, weight decay 3e-5 | SGD momentum 0.99, Nesterov, weight decay 3e-5 | **SAME** |  |
| initial LR | 0.01 | 0.01 | **SAME** |  |
| LR scheduler | PolyLR exponent 0.9; 1000-epoch denominator | PolyLR exponent 0.9; 1000-epoch denominator | **SAME** |  |
| augmentation pipeline | nnUNet v2.8.1 default: rotation/scaling, noise, blur, brightness, contrast, low-resolution, two gamma transforms, mirroring axes (0,1,2); no override | nnUNet v2.8.1 default: rotation/scaling, noise, blur, brightness, contrast, low-resolution, two gamma transforms, mirroring axes (0,1,2); no override | **SAME** |  |
| number of epochs | 100 | 100 | **SAME** |  |
| checkpoint policy | checkpoint_best retained; primary checkpoint_final at epoch 100 | checkpoint_best retained; primary checkpoint_final at epoch 100 | **SAME** |  |
| inference configuration | nnUNet saved-checkpoint sliding-window validation; mirroring axes (0,1,2); threshold-equivalent argmax 0.5 | nnUNet saved-checkpoint sliding-window validation; mirroring axes (0,1,2); threshold-equivalent argmax 0.5 | **SAME** |  |
| post-processing | none | none | **SAME** |  |
| evaluation code | scripts/evaluate_cv_fold.py -> evaluate_experiment.metrics/summarize | scripts/evaluate_cv_fold.py -> evaluate_experiment.metrics/summarize | **SAME** |  |
| connected-component definition | 26-connectivity; max-cardinality then max-IoU one-to-one overlap | 26-connectivity; max-cardinality then max-IoU one-to-one overlap | **SAME** |  |

## Existing Original B0 Fold 0 reuse gate

**PASS.** Correct trainer: `nnUNetTrainer_B0R2022_Screen100`; 100 epoch rows; checkpoint epoch 100; checkpoint hash matches completion metadata; 50 masks and 50 probability files exactly match fixed Fold 0 validation IDs. The unified evaluator can therefore recompute Fold 0 without retraining.

## Fixed split audit

Five folds cover all 250 cases exactly once in validation; each fold is 200 train / 50 validation; no train-validation overlap was found. Split SHA-256: `09ed64d65dc3b5ece2656639a5abfc1a4f76b6e33121512a4931f6b6d6814d31`.
