"""Validate and continue S1a from epoch 100 to total epoch 300."""

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
from recall_common import PP, PROTECTED, ROOT, digest, now, protect_baseline, provenance, save, verify_baseline
from recall_trainers import DS_WEIGHTS
from s1a300_trainer import nnUNetTrainer_S1a_Promote300


SUITE = ROOT / "experiments/S1a_promote300_seed2022_from100"
SOURCE_SUITE = ROOT / "experiments/recall_screen100_seed2022_r2"
SOURCE_RUN = SOURCE_SUITE / "S1a_screen100_seed2022"
SOURCE_CHECKPOINT = SOURCE_RUN / (
    "train/results/Dataset001_ISLES2022_DWI/"
    "nnUNetTrainer_S1a_Screen100__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
PROTECTED.extend([SOURCE_SUITE])


def seed_all() -> None:
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)


def source_state() -> dict:
    completion = json.loads((SOURCE_RUN / "train/completion.json").read_text())
    comparison = json.loads((SOURCE_SUITE / "screening_comparison/screening_summary.json").read_text())
    decision = comparison["comparisons"]["S1a_screen100_seed2022"]["decision"]["decision"]
    if completion["status"] != "screening_training_and_validation_complete":
        raise RuntimeError("S1a-100 source run is incomplete")
    if decision != "PROMOTE TO 300 EPOCHS":
        raise RuntimeError(f"S1a-100 promotion decision is {decision!r}")
    if Path(completion["checkpoint"]).resolve() != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("S1a-100 checkpoint path differs from completion record")
    actual_hash = digest(SOURCE_CHECKPOINT)
    if actual_hash != completion["checkpoint_sha256"]:
        raise RuntimeError("S1a-100 checkpoint hash mismatch")
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("current_epoch") != 100:
        raise RuntimeError("S1a-100 checkpoint does not represent 100 completed epochs")
    required = {"network_weights", "optimizer_state", "grad_scaler_state", "logging", "current_epoch"}
    if not required.issubset(checkpoint):
        raise RuntimeError("S1a-100 checkpoint is missing continuation state")
    return {
        "checkpoint": str(SOURCE_CHECKPOINT),
        "checkpoint_sha256": actual_hash,
        "checkpoint_epoch": checkpoint["current_epoch"],
        "checkpoint_trainer": checkpoint.get("trainer_name"),
        "source_completion": str(SOURCE_RUN / "train/completion.json"),
        "source_comparison": str(SOURCE_SUITE / "screening_comparison/screening_summary.json"),
        "promotion_decision": decision,
        "rng_limitation": (
            "Checkpoint restores model, optimizer and AMP scaler but not complete Python/NumPy/"
            "PyTorch/dataloader RNG states; controlled checkpoint continuation is not bitwise deterministic."
        ),
    }


def make_trainer(device: str = "cuda"):
    plans = json.loads((PP / "nnUNetPlans.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]
    if cfg["patch_size"] != [80, 96, 80] or cfg["batch_size"] != 8 or cfg["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("Baseline plans changed")
    plans["continue_training"] = False
    trainer = nnUNetTrainer_S1a_Promote300(
        plans,
        "3d_fullres",
        0,
        json.loads((PP / "dataset.json").read_text()),
        device=torch.device(device),
    )
    trainer.execution_started = time.monotonic()
    if trainer.num_epochs != 300 or trainer.lr_schedule_max_epochs != 1000:
        raise RuntimeError("Promotion epoch count or LR reference horizon changed")
    if trainer.train_foreground != 0.50:
        raise RuntimeError("S1a training sampling configuration changed")
    return trainer


def add_sources(meta: dict) -> None:
    for name in [
        "s1a300_trainer.py",
        "run_s1a300_experiment.py",
        "s1a300_pipeline.py",
        "s1a300_analysis.py",
        "test_s1a300_protocol.py",
        "screen100_trainers.py",
        "recall_trainers.py",
        "amp_resume_trainer.py",
    ]:
        path = ROOT / "scripts" / name
        meta["source_sha256"][str(path)] = digest(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    args = parser.parse_args()
    run = SUITE / args.mode
    run.mkdir(parents=True, exist_ok=False)
    protect_baseline()
    trainer = None
    try:
        verify_baseline(run / "baseline_before.json")
        source = source_state()
        os.environ["nnUNet_results"] = str(run / "results")
        seed_all()
        torch.set_num_threads(8)
        meta = provenance()
        add_sources(meta)
        meta.update(
            mode=args.mode,
            experiment="S1a promotion 100 to 300",
            label="300-epoch promoted screening; checkpoint continuation from S1a-100",
            source=source,
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
            train_foreground=0.50,
            validation_foreground=0.33,
            resume_epoch=100,
            target_total_epoch=300,
            continuation_epochs=200,
            lr_schedule="PolyLR exponent 0.9; reference horizon 1000 epochs",
            pretrained_weights=None,
            early_stopping=False,
            evaluation_checkpoint="checkpoint_final.pth",
            loss_definition=(
                "DC_and_CE_loss: MemoryEfficientSoftDiceLoss (negative Dice) + RobustCrossEntropyLoss; "
                "weights 1/1; per-sample; no background Dice; smooth 1e-5; ignore=None"
            ),
            deep_supervision_weights=DS_WEIGHTS,
            prohibited_auto_followup=(
                "No automatic 1000-epoch run, extra seed, P2/P3, focal, multimodal, combination or ensemble"
            ),
        )
        save(run / "experiment_config.json", meta)
        trainer = make_trainer()
        trainer.configure_screen_monitoring(run, args.mode)
        trainer.load_checkpoint(str(SOURCE_CHECKPOINT))
        if trainer.current_epoch != 100:
            raise RuntimeError("Trainer did not restore epoch 100")
        if not all_finite(trainer.network.parameters()) or not all_finite(tensors_in(trainer.optimizer.state)):
            raise FloatingPointError("Nonfinite restored model or optimizer state")

        if args.mode == "smoke":
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            expected_lr = trainer.initial_lr * (1 - 100 / 1000) ** 0.9
            actual_lr = float(trainer.optimizer.param_groups[0]["lr"])
            if not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=1e-14):
                raise RuntimeError("Epoch-101 LR continuity check failed")
            events = []
            for index in range(120):
                if index == 20:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                result = trainer.train_step(next(trainer.dataloader_train))
                events.append(dict(trainer.last_amp_event))
                if (index + 1) % 10 == 0:
                    print(
                        f"S1a-300 continuation feasibility {index + 1}/120 | "
                        f"loss={float(result['loss']):.5f} | skipped={trainer.total_skipped}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            trainer.network.eval()
            with torch.no_grad():
                validation = trainer.validation_step(next(trainer.dataloader_val))
            with (run / "verification_iterations.json").open("x", encoding="utf-8") as handle:
                json.dump(events, handle, indent=2, allow_nan=False)
            result = {
                "status": "passed",
                "restored_epoch": trainer.current_epoch,
                "epoch_101_lr": actual_lr,
                "warmup": 20,
                "measured": 100,
                "seconds": seconds,
                "seconds_per_iteration": seconds / 100,
                "skipped_steps": trainer.total_skipped,
                "nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "validation_loss": float(validation["loss"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            trainer.close_loaders()
        else:
            smoke = json.loads((SUITE / "smoke/completion.json").read_text())
            if smoke["status"] != "passed":
                raise RuntimeError("Continuation smoke test did not pass")
            trainer.run_training()
            checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
            trainer.perform_actual_validation(save_probabilities=True)
            result = {
                "status": "training_and_validation_complete",
                "resume_epoch": 100,
                "actual_total_epochs": trainer.current_epoch,
                "continuation_epochs": trainer.current_epoch - 100,
                "lr_schedule_reference_epochs": 1000,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "predictions": str(checkpoint.parent / "validation"),
                "total_amp_skipped_steps_during_continuation": trainer.total_skipped,
                "total_nonfinite_gradient_events_during_continuation": trainer.total_nonfinite_gradient_events,
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
