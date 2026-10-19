"""Run one isolated 100-epoch screening experiment with exclusive outputs."""

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

from recall_common import (
    PP,
    PROTECTED,
    ROOT,
    digest,
    now,
    protect_baseline,
    provenance,
    save,
    verify_baseline,
)
from recall_trainers import DS_WEIGHTS
from screen100_trainers import SCREEN_TRAINERS


SUITE = ROOT / "experiments/recall_screen100_seed2022_r2"
EXISTING_EXPERIMENTS = [
    ROOT / "experiments/B0-R2022_resume_from_epoch34",
    ROOT / "experiments/P1_spacing_1p5mm",
    ROOT / "experiments/recall_2mm_seed2022",
]


def seed_all() -> None:
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)


def make_trainer(experiment: str, device: str = "cuda"):
    plans = json.loads((PP / "nnUNetPlans.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]
    if cfg["patch_size"] != [80, 96, 80] or cfg["batch_size"] != 8 or cfg["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("Baseline plans changed")
    plans["continue_training"] = False
    trainer = SCREEN_TRAINERS[experiment](
        plans,
        "3d_fullres",
        0,
        json.loads((PP / "dataset.json").read_text()),
        device=torch.device(device),
    )
    if trainer.num_epochs != 100 or trainer.num_iterations_per_epoch != 250 or trainer.num_val_iterations_per_epoch != 50:
        raise RuntimeError("Screening run-length or iteration counts changed")
    # Required by nnUNetTrainer_SeededPreserve.on_epoch_end for progress/ETA logging.
    trainer.execution_started = time.monotonic()
    return trainer


def add_screen_sources(meta: dict) -> None:
    for name in [
        "screen100_trainers.py",
        "run_screen100_experiment.py",
        "screen100_pipeline.py",
        "screen100_analysis.py",
        "test_screen100_protocol.py",
        "amp_resume_trainer.py",
    ]:
        path = ROOT / "scripts" / name
        if path.exists():
            meta["source_sha256"][str(path)] = digest(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=SCREEN_TRAINERS, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    args = parser.parse_args()

    run = SUITE / args.experiment / args.mode
    run.mkdir(parents=True, exist_ok=False)
    PROTECTED.extend(EXISTING_EXPERIMENTS)
    protect_baseline()
    trainer = None
    try:
        verify_baseline(run / "baseline_before.json")
        os.environ["nnUNet_results"] = str(run / "results")
        seed_all()
        torch.set_num_threads(8)
        meta = provenance()
        add_screen_sources(meta)
        if args.mode == "train":
            smoke = run.parent / "smoke"
            status = json.loads((smoke / "completion.json").read_text())
            prior = json.loads((smoke / "experiment_config.json").read_text())
            if status["status"] != "passed":
                raise RuntimeError("Smoke test did not pass")
            if meta["packages"] != prior["packages"] or meta["plans_sha256"] != prior["plans_sha256"]:
                raise RuntimeError("Environment or plans changed after smoke test")
            current_sources = dict(meta["source_sha256"])
            if current_sources != prior["source_sha256"]:
                raise RuntimeError("Screening source changed after smoke test")

        trainer = make_trainer(args.experiment)
        trainer.configure_screen_monitoring(run, args.mode)
        expected_epoch100_lr = trainer.initial_lr * (1 - 99 / 1000) ** 0.9
        is_l2a = args.experiment.startswith("L2a")
        meta.update(
            experiment=args.experiment,
            mode=args.mode,
            label="screening experiment; not final performance",
            comparison="controlled same-seed screening comparison",
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
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            train_foreground=trainer.train_foreground,
            validation_foreground=0.33,
            configuration=trainer.configuration_manager.configuration,
            actual_run_epochs=100,
            lr_schedule_type="PolyLR exponent 0.9",
            lr_schedule_reference_epochs=1000,
            lr_epoch_index_0=trainer.initial_lr,
            lr_epoch_index_99=expected_epoch100_lr,
            pretrained_weights=None,
            early_stopping=False,
            evaluation_checkpoint="checkpoint_final.pth",
            fresh_initialization=True,
            deep_supervision_weights=DS_WEIGHTS,
            trainer_class=trainer.__class__.__name__,
            loss_definition=(
                "mean(1-TI)+CE; foreground per-sample; FP=.3 FN=.7 smooth=1e-5 float32; no focal exponent"
                if is_l2a
                else "DC_and_CE_loss: MemoryEfficientSoftDiceLoss (negative Dice) + "
                "RobustCrossEntropyLoss; weights1/1; per-sample; no background Dice; smooth1e-5; ignore=None"
            ),
            baseline_preprocessing="read-only Blosc2; unpack disabled; Python write audit guard",
            gradient_policy="AMP-safe unscale; finite gradients clip12; transient overflow skips update and lowers scale",
            prohibited_auto_followup="No automatic 300/1000-epoch run, extra seed, P2/P3, focal, multimodal, combination, or ensemble",
        )
        save(run / "experiment_config.json", meta)

        if args.mode == "smoke":
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            events = []
            for i in range(120):
                if i == 20:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                result = trainer.train_step(next(trainer.dataloader_train))
                events.append(dict(trainer.last_amp_event))
                if (i + 1) % 10 == 0:
                    print(
                        f"{args.experiment} screening feasibility {i + 1}/120 | "
                        f"loss={float(result['loss']):.5f} | skipped={trainer.total_skipped}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            trainer.network.eval()
            with torch.no_grad():
                val = trainer.validation_step(next(trainer.dataloader_val))
            if trainer.consecutive_overflows:
                raise RuntimeError("Smoke ends in unrecovered AMP overflow")
            with (run / "verification_iterations.json").open("x", encoding="utf-8") as f:
                json.dump(events, f, indent=2, allow_nan=False)
            result = {
                "status": "passed",
                "warmup": 20,
                "measured": 100,
                "seconds": seconds,
                "seconds_per_iteration": seconds / 100,
                "skipped_steps": trainer.total_skipped,
                "nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "validation_loss": float(val["loss"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            trainer.close_loaders()
        else:
            trainer.run_training()
            checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
            trainer.perform_actual_validation(save_probabilities=True)
            result = {
                "status": "screening_training_and_validation_complete",
                "label": "screening experiment; not final performance",
                "actual_epochs": trainer.current_epoch,
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
