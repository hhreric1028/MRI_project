# Kamel-style replication configuration audit

## Audit result

**PASS.** The current B0 3d_fullres configuration is semantically identical to the configuration saved in the B0-100 result. The plans identify nnUNet `ExperimentPlanner`; no separate manual plan divergence was found. This experiment reuses the existing preprocessing and does not re-plan or modify it.

Important protocol note: actual training stops at 100 epochs, while PolyLR keeps the native/B0 reference horizon of 1000 epochs. This reproduces the B0-100 learning-rate trajectory and does not compress PolyLR into 100 epochs.

## Fixed configuration

| Item | Planner/current B0 value | Consistency |
|---|---|---|
| Target spacing | `[2.0, 2.0, 2.0]` mm | Match |
| Patch size | `[80, 96, 80]` (Z,Y,X) | Match |
| Batch size | `8` | Match |
| Architecture | `dynamic_network_architectures.architectures.unet.PlainConvUNet` | Match |
| Stages/features | `5` / `[32, 64, 128, 256, 320]` | Match |
| Normalization | `['ZScoreNormalization']`, mask `[True]` | Match |
| Loss | Dice + CE, equal weights, batch Dice false | Match |
| Optimizer | SGD, momentum 0.99, Nesterov, weight decay 3e-5 | Match |
| Initial LR | 0.01 | Match |
| LR scheduler | PolyLR exponent 0.9, 1000-epoch denominator | Match |
| Foreground oversampling | 0.33 | Match |
| Inference | Sliding window, mirroring axes (0,1,2), softmax/argmax | Match |
| Post-processing | None | Match |

## Default augmentation retained

- **spatial:** rotation p=0.2; scaling p=0.2 range 0.7-1.4; elastic disabled
- **gaussian_noise:** variance 0-0.1; p=0.10
- **gaussian_blur:** sigma 0.5-1.0; p=0.20; p/channel=0.50
- **multiplicative_brightness:** 0.75-1.25; p=0.15
- **contrast:** 0.75-1.25; p=0.15; preserve range
- **low_resolution:** scale 0.5-1.0; p=0.25
- **gamma_inverted:** 0.7-1.5; p=0.10; retain statistics
- **gamma_non_inverted:** 0.7-1.5; p=0.30; retain statistics
- **mirroring:** axes (0,1,2)

A1 augmentation is not imported or used. There is no custom loss, sampling, threshold, architecture, spacing, patch, or batch change.

## Fixed split audit

- Five fixed folds found in `splits_final.json`; each fold has 200 training and 50 validation cases.
- Validation union: 250 unique cases.
- Every case appears as validation exactly once: `True`.
- Train/validation leakage: `False`.
- The split file is read-only and will not be regenerated.

## Source hashes

- plans: `09a3894923f0437d714fd14e44c3728785b59d2c28d01d12d8f7681b38d8340e`
- fingerprint: `ba4c5d098dd03427d2160b2906c2fdaa5c457dd16f8d0935f27a6f088fddcce1`
- splits: `09ed64d65dc3b5ece2656639a5abfc1a4f76b6e33121512a4931f6b6d6814d31`
