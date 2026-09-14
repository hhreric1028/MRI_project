# Improvement execution status

Started 2026-09-14 (Asia/Taipei). Baseline source commit 0aac38b remains unchanged.

- Evaluation v2 implemented and run on all 50 B0 volumes in a separate directory. Lesion micro recall 0.647, precision 0.721, F1 0.682; severe under-segmentation 6; catastrophic miss 0 under overlap-based one-to-one component matching. Original metrics untouched.
- Tests passed: both/single-empty masks, known physical distance, merge/split components, micro aggregation.
- Dataset011 (1.5mm) and Dataset012 (1mm) copied into independent raw/preprocessed directories and officially preprocessed, 250 cases each. Exact baseline split copied. Only 3d_fullres was preprocessed; no baseline arrays rewritten.
- P1 first smoke failed before training with missing continue_training constructor argument. Failed directory retained. r2 corrects constructor compatibility and environment path handling.
- P1 r2 GPU feasibility and training use nnUNetTrainer_SeededPreserve: same architecture/loss/augmentation, explicit worker seeds, finite-loss guard, ETA printing and preserved latest checkpoint. External module is scripts/train_spacing_experiment.py; reconstruct with this committed module for checkpoint loading, not an unregistered stock CLI class lookup.
- Seeded worker streams do not guarantee deterministic interleaving. Historical B0 training seed is not established; controlled seed replicates remain required for strong comparative claims.
- Visible PowerShell windows and transcript logs under outputs/ are retained. Each run's experiment_config, completion/failure and results live under experiments/P1_spacing_1p5mm/.
- Remaining: P1 full 1000 epochs and final evaluation, P2 GPU trial/full run, P3/S1/L1/L2 implementation and trials, paired comparison, then evidence-supported combinations. Secondary candidates and multimodal stages have not started.

Ground truth for live progress: each run's completion.json/failure.json and official training log. This document is a milestone record, not a claim that remaining stages have completed.
