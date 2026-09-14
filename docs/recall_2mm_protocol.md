# 2 mm recall protocol — 2026-09-15 addendum

This protocol supersedes the previous P2/P3/loss-candidate execution order. No running training/evaluation/sampling job may be interrupted, restarted or modified to retrofit this addendum. Missing historical fields are reported as unavailable or future protocol improvements, not reconstructed by retraining.

Order: P1 failure analysis → B0-R2022 → sampling mechanism audit → S1a decision → L2a → paired comparison. P2 preprocessing is retained read-only; P2 training, P3, focal variants, multimodal, combinations and seeds 2023/2024 are not queued.

| Run | Change | Training FG | Validation FG |
|---|---|---:|---:|
| B0-R2022 | Same-seed control, original Dice+CE | 0.33 | 0.33 |
| S1a | Training FG sampling only | 0.50 | 0.33 |
| L2a | Foreground per-sample Tversky + CE | 0.33 | 0.33 |

All use Dataset001 read-only, fold0 fixed 200/50 cases, DWI, 2 mm, patch80×96×80, batch8, original augmentation, SGD/PolyLR, 1000epochs, 250train/50validation iterations per epoch, final checkpoint, no pretrained weights/early stopping. Python, NumPy, Torch and CUDA seeds2022; worker seeds train2022–2025 and validation3022–3023. NonDetMultiThreadedAugmenter preserves the existing worker rule; this is a **controlled same-seed comparison**, not fully deterministic. cuDNN deterministic/benchmark and deterministic-algorithm flags are recorded without changing defaults.

L2a: TI=(TP+1e-5)/(TP+0.3FP+0.7FN+1e-5); loss=mean(1−TI)+CE. Foreground class1, background excluded from TI but included in CE; no ignore label; float32 softmax and accumulation; unit component weights; DS=4/7,2/7,1/7,0. Tests independently evaluate every head, including the zero-weight head, for finite loss/gradients. Empty/perfect/all-background/all-foreground/extreme-logit cases and increasing FP/FN are covered. The difference from negative-Dice loss includes a constant offset; absolute train losses across loss families must not be interpreted as performance differences.

## Preservation and gates

New classes and exclusive directories under experiments/recall_2mm_seed2022. No unpack. Python write-audit guard protects baseline preprocessing/models/metrics; Blosc2 arrays are opened read-only. Full historical hashes plus unexpected-file checks run at phase boundaries. Existing evidence is retained. Commands, combined stdout/stderr, source/environment hashes, commit, settings, checkpoint hashes, completion/failure records are saved. Smoke is20warm-up+100measured GPU iterations, followed by a fresh-process formal training only if passed. Any output conflict or nonfinite loss/gradient stops that run; no automatic retry, checkpoint deletion, seed/LR/batch/loss adjustment.

The visible pipeline has three phases: preflight, baseline, candidates. Preflight must finish and actual failure-panel review be recorded before baseline starts. Baseline phase performs B0-R2022 smoke/full/final evaluation then the sampling audit. The candidates phase requires an explicit evidence-based sampling_decision.json referencing the audit hash; it does not guess a scientific materiality threshold. This deliberate review gate implements the addendum's replacement of the old equality-only rule.

## Sampling mechanism audit

Exactly1000post-augmentation training patches per0.33/0.50, same initial seed, no validation sampling. Record force_fg, lesion presence/count/fraction, case ID, native GT volume subgroup and source-GT coverage; summarize all five groups, prioritize1–5mL. With batch8 the nominal0.33 setting forces3/8 (37.5%) patches and0.50 forces4/8 (50%); these are not necessarily lesion-containing patch rates. Original cropping branches can consume different random draws; same initial seed does not guarantee matched subsequent case IDs. Actual records identify that limitation.

Coverage is the fraction of full preprocessed GT voxel centers lying in both the initial crop and the actual augmented patch support, computed by inverse spatial affine. It is not augmented foreground voxels/native GT voxels, which would confound scaling/interpolation. Native GT defines subgroups. Foreground fraction is measured on the final augmented full-resolution target. Report coverage's geometric/interpolation limitation. No redraw if any prior audit exists. A meaningful exposure increase, especially for1–5mL cases, is reviewed using actual effect sizes, not bitwise equality; no arbitrary unapproved cutoff is silently substituted.

## Analysis

Existing evaluation-v2 lesion matching/HD95/empty conventions remain unchanged; additive voxel counts/recall/precision are now recorded for future outputs. All native50case predictions and five volume groups required. P1 panels use identical native DWI and GT/B0/P1 slices at GT maximum and prediction maximum disagreement. Distinguish zero-overlap whole-lesion miss, incomplete detected-lesion coverage and extra FP. One-to-one matched FN can also reflect merging and differs from zero-overlap misses. Volume differences do not establish FOV or heterogeneity as causes.

Paired case-level bootstrap:10000resamples,seed2022, identical sampled IDs for baseline/candidate, difference=candidate−baseline. Reaggregate lesion TP/FP/FN after each resample for recall/precision; lesions are not independent resampling units. Mean/median Dice, lesion recall/precision, volume MAE/signed error, severe-under count/rate, Dice<0.5 rate, catastrophic miss, FP/case and finite-HD95 plus infinity rate are included. Severe under means nonemptyGT ratio<0.5. Prioritize severe/catastrophic failures, then1–5mL performance, then overall Dice. More recall with more FP is a tradeoff, not an automatic win.

## Limitations

Fold0only; single-seed screening. Same seed is not fully deterministic. Actual sampling exposure change requires audit evidence. Bootstrap CIs reflect this validation cohort's case-level uncertainty, not training-seed uncertainty or a generalization conclusion across folds/centers. Only after improvement signals should paired seeds2023/2024 be planned; never automatically launched.
