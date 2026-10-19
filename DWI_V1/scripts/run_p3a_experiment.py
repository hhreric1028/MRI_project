"""Run P3a smoke, 100-epoch screen, or checkpoint continuation to epoch 300."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from amp_resume_trainer import all_finite, tensors_in
from p3a_trainers import P3_PP, nnUNetTrainer_P3a_Promote300, nnUNetTrainer_P3a_Screen100
from recall_common import PROTECTED, ROOT, digest, now, protect_baseline, provenance, save, verify_baseline
from recall_trainers import DS_WEIGHTS


SUITE = ROOT / "experiments/P3a_clip_zscore_seed2022"
P3_RAW = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset013_ISLES2022_DWI_ClipZScore"
SOURCE_100 = SUITE / (
    "screen100/train/results/Dataset013_ISLES2022_DWI_ClipZScore/"
    "nnUNetTrainer_P3a_Screen100__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
PROTECTED.extend([P3_RAW, P3_PP, ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"])


def seed_all() -> None:
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)


def make_trainer(stage: str, device: str = "cuda"):
    plans = json.loads((P3_PP / "nnUNetPlans.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]
    if cfg["patch_size"] != [80, 96, 80] or cfg["batch_size"] != 8 or cfg["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("P3a plans differ from controlled baseline geometry")
    if cfg["normalization_schemes"] != ["ZScoreNormalization"] or cfg["use_mask_for_norm"] != [True]:
        raise RuntimeError("P3a z-score normalization differs from baseline")
    plans["continue_training"] = False
    cls = nnUNetTrainer_P3a_Screen100 if stage == "screen100" else nnUNetTrainer_P3a_Promote300
    trainer = cls(
        plans,
        "3d_fullres",
        0,
        json.loads((P3_PP / "dataset.json").read_text()),
        device=torch.device(device),
    )
    trainer.execution_started = time.monotonic()
    if trainer.train_foreground != 0.33 or trainer.lr_schedule_max_epochs != 1000:
        raise RuntimeError("P3a sampling or LR reference horizon changed")
    return trainer


def source_checkpoint_state() -> dict:
    completion = json.loads((SUITE / "screen100/train/completion.json").read_text())
    if completion["status"] != "training_and_validation_complete" or completion["actual_total_epochs"] != 100:
        raise RuntimeError("P3a-100 completion is invalid")
    if Path(completion["checkpoint"]).resolve() != SOURCE_100.resolve():
        raise RuntimeError("P3a-100 checkpoint path mismatch")
    if digest(SOURCE_100) != completion["checkpoint_sha256"]:
        raise RuntimeError("P3a-100 checkpoint hash mismatch")
    checkpoint = torch.load(SOURCE_100, map_location="cpu", weights_only=False)
    if checkpoint.get("current_epoch") != 100:
        raise RuntimeError("P3a-100 checkpoint epoch mismatch")
    return {
        "checkpoint": str(SOURCE_100),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "current_epoch": checkpoint["current_epoch"],
        "trainer_name": checkpoint.get("trainer_name"),
        "rng_limitation": "Model, optimizer and GradScaler are restored; complete dataloader RNG state is not checkpointed.",
    }


def add_sources(meta: dict) -> None:
    for name in [
        "prepare_p3a_dataset.py", "p3a_trainers.py", "run_p3a_experiment.py", "p3a_pipeline.py",
        "p3a_analysis.py", "test_p3a_protocol.py", "screen100_trainers.py", "recall_trainers.py",
        "amp_resume_trainer.py", "evaluate_experiment.py",
    ]:
        path = ROOT / "scripts" / name
        meta["source_sha256"][str(path)] = digest(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen100", "promote300"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    args = parser.parse_args()
    run = SUITE / args.stage / args.mode
    run.mkdir(parents=True, exist_ok=False)
    protect_baseline()
    trainer = None
    try:
        verify_baseline(run / "baseline_before.json")
        os.environ["nnUNet_raw"] = str(P3_RAW.parent)
        os.environ["nnUNet_preprocessed"] = str(P3_PP.parent)
        os.environ["nnUNet_results"] = str(run / "results")
        seed_all()
        torch.set_num_threads(8)
        meta = provenance()
        add_sources(meta)
        source = source_checkpoint_state() if args.stage == "promote300" else None
        meta.update(
            experiment="P3a_clip_zscore_seed2022",
            stage=args.stage,
            mode=args.mode,
            controlled_change="per-volume foreground P0.5-P99.5 clipping before baseline-equivalent foreground z-score",
            dataset_id=13,
            dataset_name="Dataset013_ISLES2022_DWI_ClipZScore",
            source_checkpoint=source,
            seed_python=2022,
            seed_numpy=2022,
            seed_torch=2022,
            seed_cuda=2022,
            worker_seeds_train=[2022, 2023, 2024, 2025],
            worker_seeds_val=[3022, 3023],
            worker_order="NonDetMultiThreadedAugmenter; arrival order not deterministic",
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            train_foreground=0.33,
            validation_foreground=0.33,
            target_total_epoch=100 if args.stage == "screen100" else 300,
            lr_schedule="PolyLR exponent 0.9; reference horizon 1000 epochs",
            loss_definition=(
                "DC_and_CE_loss: MemoryEfficientSoftDiceLoss (negative Dice) + RobustCrossEntropyLoss; "
                "weights 1/1; per-sample; no background Dice; smooth 1e-5; ignore=None"
            ),
            deep_supervision_weights=DS_WEIGHTS,
            early_stopping=False,
            evaluation_checkpoint="checkpoint_final.pth",
            prohibited_auto_followup="No automatic 1000-epoch run or non-P3a experiment",
        )
        save(run / "experiment_config.json", meta)
        trainer = make_trainer(args.stage)
        trainer.configure_screen_monitoring(run, args.mode)
        if args.stage == "promote300":
            trainer.load_checkpoint(str(SOURCE_100))
            if trainer.current_epoch != 100:
                raise RuntimeError("P3a continuation did not restore epoch 100")
            if not all_finite(trainer.network.parameters()) or not all_finite(tensors_in(trainer.optimizer.state)):
                raise FloatingPointError("Nonfinite restored P3a model or optimizer state")

        if args.mode == "smoke":
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            expected_index = 0 if args.stage == "screen100" else 100
            if trainer.current_epoch != expected_index:
                raise RuntimeError("Smoke epoch state mismatch")
            expected_lr = trainer.initial_lr * (1 - expected_index / 1000) ** 0.9
            actual_lr = float(trainer.optimizer.param_groups[0]["lr"])
            if not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=1e-14):
                raise RuntimeError("P3a LR continuity mismatch")
            events = []
            for index in range(120):
                if index == 20:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                step = trainer.train_step(next(trainer.dataloader_train))
                events.append(dict(trainer.last_amp_event))
                if (index + 1) % 10 == 0:
                    print(f"P3a {args.stage} feasibility {index + 1}/120 | loss={float(step['loss']):.5f} | skipped={trainer.total_skipped}", flush=True)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            trainer.network.eval()
            with torch.no_grad():
                validation = trainer.validation_step(next(trainer.dataloader_val))
            with (run / "verification_iterations.json").open("x", encoding="utf-8") as handle:
                json.dump(events, handle, indent=2, allow_nan=False)
            result = {
                "status": "passed",
                "stage": args.stage,
                "restored_epoch": trainer.current_epoch,
                "current_lr": actual_lr,
                "warmup": 20,
                "measured": 100,
                "seconds_per_iteration": seconds / 100,
                "skipped_steps": trainer.total_skipped,
                "nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "validation_loss": float(validation["loss"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            trainer.close_loaders()
        else:
            smoke = json.loads((SUITE / args.stage / "smoke/completion.json").read_text())
            if smoke["status"] != "passed":
                raise RuntimeError("P3a smoke test did not pass")
            trainer.run_training()
            checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
            trainer.perform_actual_validation(save_probabilities=True)
            result = {
                "status": "training_and_validation_complete",
                "stage": args.stage,
                "resume_epoch": 0 if args.stage == "screen100" else 100,
                "actual_total_epochs": trainer.current_epoch,
                "lr_schedule_reference_epochs": 1000,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "predictions": str(checkpoint.parent / "validation"),
                "total_amp_skipped_steps": trainer.total_skipped,
                "total_nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "unrecovered_consecutive_overflows": trainer.consecutive_overflows,
            }
        verify_baseline(run / "baseline_after.json")
        result["finished_at"] = now()
        save(run / "completion.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        save(run / "failure.json", {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        if trainer:
            trainer.close_loaders()
        if not (run / "baseline_after.json").exists():
            verify_baseline(run / "baseline_after.json")
        raise


if __name__ == "__main__":
    main()

