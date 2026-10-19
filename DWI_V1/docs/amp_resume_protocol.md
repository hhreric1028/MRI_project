# B0-R2022 AMP-safe resume from epoch 34

User-authorized continuation, 2026-09-15. Original experiment classes and installed nnUNet code remain unchanged. New class: `nnUNetTrainer_B0R2022_Resume`. New root: `experiments/B0-R2022_resume_from_epoch34/{smoke,train}`. Source checkpoint is the retained `checkpoint_best.pth` from `experiments/recall_2mm_seed2022/B0-R2022/train`, SHA256 `f67ae113f4a68bad8e4ed6da31191f3497d7588bfa2c7a1c291e75dff1f865bd`.

## State and epoch semantics

nnUNet saves `current_epoch + 1` at the end of an epoch. The source contains `current_epoch=34` and 34 complete logging entries (indices0–33). Restore index34, print human epoch35; train through index999/human epoch1000. No completed epoch is repeated.

Restore network weights, SGD momentum/state, GradScaler state, logging/EMA, and mirroring axes. Check exact tensor values and optimizer tensor devices before any update. The initial objects are constructed only to load saved state; smoke weights never serve as formal continuation. The formal stage reloads the original checkpoint independently after the verification report is available.

No scheduler state dictionary exists in the source checkpoint. Reconstruct the original PolyLR from unchanged configuration: initial LR0.01, max_steps1000, exponent0.9. The saved optimizer LR must equal the step33 value; call scheduler.step(34) and assert its value against the original formula. Validate the schedule again at each epoch.

## AMP changes

The old check already ran after unscale; its failure was that raising immediately prevented scaler.step/update. The new sequence is autocast forward → finite loss check → scaled backward → exactly one unscale → unscaled norm/entry inspection → clip finite gradients at the unchanged norm12 → scaler.step → scaler.update. Nonfinite gradients skip clipping and remain registered in GradScaler. An optimizer post-step hook independently verifies whether an actual optimizer update occurred. A skipped step must lower a positive finite scale. Model and optimizer state are checked for finite values after each attempted update.

Transient Inf/NaN gradients with finite loss/state can recover by skipped steps. Event-only logs include epoch/iteration/case IDs/loss/LR/scale before-after/unscaled norm/finite flag/skip flag. Sixteen consecutive skips without an intervening successful update trigger a persistent-overflow failure; this is an explicitly recorded monitoring threshold (up to65536-fold scale reduction), not a change to model/optimizer hyperparameters. Nonfinite loss, model/optimizer state, unrecoverable AMP behavior, LR mismatch, missing restore state or output collisions also stop. On failure, the actual augmented batch is saved exclusively for later diagnosis.

Normal finite updates retain the original clipping and optimizer behavior. Tests verify equality with the original AMP finite update, injected transient Inf and NaN skip/recovery with unchanged momentum/parameters during the skip, non-AMP failure handling, and PolyLR positioning. Real MRI verification:20warm-up+100measured iterations, finite-state checks, LR/device/state verification, timing/VRAM, overflow logging and recovery check. Final validation-step loss must be finite.

First three completed continuation epochs (human35–37) additionally report train/validation loss, pseudo-Dice, LR, scaler scale, skip/nonfinite counts and mean/max finite unscaled gradient norm. ETA is based only on epochs actually completed during continuation, excluding the historical34epochs.

## Preservation and limitations

The source run is added to the read-only path guard; its files are hashed before and after. Baseline1923-file checks remain enabled. Independent output directories and exclusive logs prevent collisions. Source/dependency hashes, environment, commit, restore metadata and checkpoint hashes are saved.

This run is a training-state continuation, not a bitwise deterministic continuation. Full RNG/worker/iterator state is absent in the source. Keep seed2022 and existing worker rules (train2022–2025; validation3022–3023), explicitly establishing new worker streams at epoch35. Do not claim those reproduce the missing pre-interruption iterator position. The resumed run must be labeled as interrupted/resumed in later same-seed comparisons.

The original root-level baseline failure marker is preserved as historical evidence; current resume status lives only in the new root. P2/P3/focal/multimodal/combinations/additional seeds are not launched by this resume dispatcher.
