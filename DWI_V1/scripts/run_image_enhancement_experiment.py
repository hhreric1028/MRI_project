"""Run one enhancement smoke/train stage, with AMP-safe 100-to-300 continuation."""

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
from image_enhancement_common import BASE_RAW, EXPERIMENTS, ROOT, SEED, SUITE, experiment_dir
from image_enhancement_trainers import TRAINERS
from recall_common import PROTECTED, digest, now, protect_baseline, provenance, save, verify_baseline
from recall_trainers import DS_WEIGHTS


PROTECTED.append(BASE_RAW)


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def source_checkpoint(experiment: str) -> tuple[Path, dict]:
    completion_path = experiment_dir(experiment) / "screen100/train/completion.json"
    completion = json.loads(completion_path.read_text())
    path = Path(completion["checkpoint"])
    if completion["actual_total_epochs"] != 100 or digest(path) != completion["checkpoint_sha256"]:
        raise RuntimeError("Invalid frozen 100-epoch source checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("current_epoch") != 100:
        raise RuntimeError("Source checkpoint is not at epoch 100")
    return path, {
        "checkpoint": str(path),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "current_epoch": 100,
        "trainer_name": checkpoint.get("trainer_name"),
        "rng_limitation": "Model, optimizer and GradScaler are restored; full dataloader iterator state is unavailable.",
    }


def make_trainer(experiment: str, stage: str, result_root: Path, device: str = "cuda"):
    spec = EXPERIMENTS[experiment]
    pp = ROOT / "datasets/ISLES2022/processed" / spec["dataset_name"]
    plans = json.loads((pp / "nnUNetPlans.json").read_text())
    cfg = plans["configurations"]["3d_fullres"]
    expected = (spec["normalization_schemes"], spec["use_mask_for_norm"])
    actual = (cfg["normalization_schemes"], cfg["use_mask_for_norm"])
    if cfg["patch_size"] != [80, 96, 80] or cfg["batch_size"] != 8 or cfg["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("Enhancement plans differ from controlled baseline geometry")
    if actual != expected:
        raise RuntimeError(f"Unexpected normalization plan: {actual} != {expected}")
    plans["continue_training"] = False
    os.environ["nnUNet_results"] = str(result_root)
    trainer = TRAINERS[experiment][stage](
        plans,
        "3d_fullres",
        0,
        json.loads((pp / "dataset.json").read_text()),
        device=torch.device(device),
    )
    trainer.execution_started = time.monotonic()
    if trainer.train_foreground != 0.33 or trainer.lr_schedule_max_epochs != 1000:
        raise RuntimeError("Sampling or PolyLR reference horizon changed")
    return trainer


def add_sources(meta: dict) -> None:
    names = [
        "image_enhancement_common.py", "prepare_image_enhancement_datasets.py",
        "image_enhancement_trainers.py", "run_image_enhancement_experiment.py",
        "image_enhancement_analysis.py", "image_enhancement_pipeline.py",
        "test_image_enhancement_protocol.py", "screen100_trainers.py",
        "recall_trainers.py", "amp_resume_trainer.py", "evaluate_experiment.py",
    ]
    for name in names:
        path = ROOT / "scripts" / name
        meta["source_sha256"][str(path)] = digest(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=["screen100", "promote300"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    args = parser.parse_args()
    exp_root = experiment_dir(args.experiment)
    run = exp_root / args.stage / args.mode
    run.mkdir(parents=True, exist_ok=False)
    result_root = exp_root / "predictions" / args.stage / args.mode / "results"
    if result_root.exists():
        raise RuntimeError(f"Result collision: {result_root}")
    protect_baseline()
    trainer = None
    try:
        verify_baseline(run / "baseline_before.json")
        spec = EXPERIMENTS[args.experiment]
        raw = ROOT / "datasets/ISLES2022/nnUNet_raw" / spec["dataset_name"]
        pp = ROOT / "datasets/ISLES2022/processed" / spec["dataset_name"]
        os.environ["nnUNet_raw"] = str(raw.parent)
        os.environ["nnUNet_preprocessed"] = str(pp.parent)
        seed_all()
        torch.set_num_threads(8)
        meta = provenance()
        add_sources(meta)
        checkpoint_path, source = source_checkpoint(args.experiment) if args.stage == "promote300" else (None, None)
        params = json.loads((SUITE / "qc/parameter_selection.json").read_text())
        parameter_key = {"E1_gamma": "gamma", "E2_clahe": "clahe", "E3_intensity_scale": "intensity_scale"}[args.experiment]
        meta.update(
            experiment_id=spec["id"],
            experiment_name=args.experiment,
            dataset_id=spec["dataset_id"],
            dataset_name=spec["dataset_name"],
            enhancement_method=spec["method"],
            enhancement_parameters=params[parameter_key],
            source_checkpoint=source,
            stage=args.stage,
            mode=args.mode,
            trainer_class=TRAINERS[args.experiment][args.stage].__name__,
            seed=SEED,
            worker_seeds_train=[2022, 2023, 2024, 2025],
            worker_seeds_val=[3022, 3023],
            plans={"spacing": [2, 2, 2], "patch_size": [80, 96, 80], "batch_size": 8, "architecture": "PlainConvUNet"},
            loss="Dice + cross-entropy (unchanged B0-R2022)",
            foreground_sampling=0.33,
            lr_schedule="SGD + PolyLR exponent 0.9; fixed reference horizon 1000 epochs",
            target_total_epoch=100 if args.stage == "screen100" else 300,
            source_code_hashes={name: value for name, value in meta["source_sha256"].items() if "image_enhancement" in name},
            deep_supervision_weights=DS_WEIGHTS,
            prohibited_followups=["automatic 1000 epochs", "enhancement combinations", "sampling/loss/spacing/modality/architecture changes"],
        )
        save(run / "experiment_config.json", meta)
        trainer = make_trainer(args.experiment, args.stage, result_root)
        trainer.configure_screen_monitoring(run, args.mode)
        if checkpoint_path is not None:
            trainer.load_checkpoint(str(checkpoint_path))
            if trainer.current_epoch != 100:
                raise RuntimeError("Continuation did not restore epoch 100")
            if not all_finite(trainer.network.parameters()) or not all_finite(tensors_in(trainer.optimizer.state)):
                raise FloatingPointError("Nonfinite restored model or optimizer state")

        if args.mode == "smoke":
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            expected_index = 0 if args.stage == "screen100" else 100
            expected_lr = trainer.initial_lr * (1 - expected_index / 1000) ** 0.9
            actual_lr = float(trainer.optimizer.param_groups[0]["lr"])
            if trainer.current_epoch != expected_index or not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=1e-14):
                raise RuntimeError("Smoke restore/LR trajectory mismatch")
            events = []
            for index in range(120):
                if index == 20:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                step = trainer.train_step(next(trainer.dataloader_train))
                events.append(dict(trainer.last_amp_event))
                if (index + 1) % 20 == 0:
                    print(f"{args.experiment} {args.stage} feasibility {index + 1}/120 | loss={float(step['loss']):.5f} | skipped={trainer.total_skipped}", flush=True)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            trainer.network.eval()
            with torch.no_grad():
                validation = trainer.validation_step(next(trainer.dataloader_val))
            with (run / "verification_iterations.json").open("x", encoding="utf-8") as handle:
                json.dump(events, handle, indent=2, allow_nan=False)
            result = {
                "status": "passed", "experiment": args.experiment, "stage": args.stage,
                "restored_epoch": trainer.current_epoch, "current_lr": actual_lr,
                "warmup": 20, "measured": 100, "seconds_per_iteration": seconds / 100,
                "skipped_steps": trainer.total_skipped,
                "nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "validation_loss": float(validation["loss"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            trainer.close_loaders()
        else:
            smoke = json.loads((exp_root / args.stage / "smoke/completion.json").read_text())
            if smoke["status"] != "passed":
                raise RuntimeError("Smoke test did not pass")
            trainer.run_training()
            checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
            trainer.perform_actual_validation(save_probabilities=True)
            result = {
                "status": "training_and_validation_complete",
                "experiment": args.experiment,
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
