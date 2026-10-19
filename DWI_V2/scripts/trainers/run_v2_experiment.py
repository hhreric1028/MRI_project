"""Run one isolated V2 smoke, 100-epoch screen, or 100-to-300 continuation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

V2_SCRIPTS = Path(__file__).resolve().parents[1]
V1_SCRIPTS = V2_SCRIPTS.parent.parent / "DWI_V1" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))
sys.path.insert(0, str(V1_SCRIPTS))

from amp_resume_trainer import all_finite, tensors_in  # noqa: E402
from recall_trainers import DS_WEIGHTS  # noqa: E402
from v2_common import (  # noqa: E402
    E1_EXPERIMENT,
    E1_PP,
    E1_RAW,
    S1_EXPERIMENT,
    V1_PP,
    V1_RAW,
    assert_hashes,
    digest,
    git_commit,
    now,
    save_json,
    source_hashes,
    v1_guard_manifest,
)
from trainers.v2_trainers import (  # noqa: E402
    nnUNetTrainer_V2E1_Promote300,
    nnUNetTrainer_V2E1_Screen100,
    nnUNetTrainer_V2S1_Promote300,
    nnUNetTrainer_V2S1_Screen100,
)


EXPERIMENTS = {
    "E1": {
        "root": E1_EXPERIMENT,
        "raw": E1_RAW,
        "pp": E1_PP,
        "dataset_id": 101,
        "dataset_name": E1_RAW.name,
        "screen_cls": nnUNetTrainer_V2E1_Screen100,
        "promote_cls": nnUNetTrainer_V2E1_Promote300,
        "controlled_change": "second input channel is exact P3a P0.5-P99.5 clipped representation; original DWI retained as channel 0",
    },
    "S1": {
        "root": S1_EXPERIMENT,
        "raw": V1_RAW,
        "pp": V1_PP,
        "dataset_id": 1,
        "dataset_name": V1_RAW.name,
        "screen_cls": nnUNetTrainer_V2S1_Screen100,
        "promote_cls": nnUNetTrainer_V2S1_Promote300,
        "controlled_change": "training-only low-salience hard-region preference within the unchanged 0.33 forced-foreground framework",
    },
}


def seed_all() -> None:
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)


def make_trainer(experiment: str, stage: str, device: str = "cuda"):
    spec = EXPERIMENTS[experiment]
    plans = json.loads((spec["pp"] / "nnUNetPlans.json").read_text(encoding="utf-8"))
    config = plans["configurations"]["3d_fullres"]
    if config["patch_size"] != [80, 96, 80] or config["batch_size"] != 8 or config["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("Controlled baseline geometry changed")
    expected_channels = 2 if experiment == "E1" else 1
    if config["normalization_schemes"] != ["ZScoreNormalization"] * expected_channels:
        raise RuntimeError("Normalization differs from controlled specification")
    if config["use_mask_for_norm"] != [True] * expected_channels:
        raise RuntimeError("Foreground normalization mask differs")
    plans["continue_training"] = False
    cls = spec["screen_cls"] if stage == "screen100" else spec["promote_cls"]
    trainer = cls(
        plans,
        "3d_fullres",
        0,
        json.loads((spec["pp"] / "dataset.json").read_text(encoding="utf-8")),
        device=torch.device(device),
    )
    trainer.execution_started = time.monotonic()
    if trainer.train_foreground != 0.33 or trainer.lr_schedule_max_epochs != 1000:
        raise RuntimeError("Foreground framework or PolyLR reference horizon changed")
    return trainer


def screen_completion(experiment: str) -> dict:
    path = EXPERIMENTS[experiment]["root"] / "screen100/train/completion.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(completion["checkpoint"])
    if completion["status"] != "training_and_validation_complete" or completion["actual_total_epochs"] != 100:
        raise RuntimeError("Screen100 completion is invalid")
    if digest(checkpoint) != completion["checkpoint_sha256"]:
        raise RuntimeError("Screen100 checkpoint hash mismatch")
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=["screen100", "promote300"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    args = parser.parse_args()
    spec = EXPERIMENTS[args.experiment]
    run = spec["root"] / args.stage / args.mode
    if run.exists():
        raise RuntimeError(f"Run directory already exists; refusing overwrite: {run}")
    run.mkdir(parents=True)
    trainer = None
    v1_before = v1_guard_manifest()
    sources = source_hashes(
        [
            Path(__file__),
            Path(__file__).with_name("v2_trainers.py"),
            V1_SCRIPTS / "recall_trainers.py",
            V1_SCRIPTS / "screen100_trainers.py",
            V1_SCRIPTS / "amp_resume_trainer.py",
            spec["pp"] / "nnUNetPlans.json",
            spec["pp"] / "splits_final.json",
        ]
    )
    try:
        os.environ["nnUNet_raw"] = str(spec["raw"].parent)
        os.environ["nnUNet_preprocessed"] = str(spec["pp"].parent)
        os.environ["nnUNet_results"] = str(run / "results")
        os.environ["nnUNet_n_proc_DA"] = "4"
        os.environ["nnUNet_compile"] = "false"
        os.environ["MPLCONFIGDIR"] = str(run / ".matplotlib")
        seed_all()
        torch.set_num_threads(8)
        source = screen_completion(args.experiment) if args.stage == "promote300" else None
        config = {
            "experiment": args.experiment,
            "stage": args.stage,
            "mode": args.mode,
            "controlled_change": spec["controlled_change"],
            "dataset_id": spec["dataset_id"],
            "dataset_name": spec["dataset_name"],
            "fold": 0,
            "seed": 2022,
            "worker_seeds_train": [2022, 2023, 2024, 2025],
            "worker_seeds_validation": [3022, 3023],
            "target_total_epoch": 100 if args.stage == "screen100" else 300,
            "lr_schedule": "PolyLR exponent 0.9; reference horizon 1000 epochs",
            "patch_size": [80, 96, 80],
            "batch_size": 8,
            "spacing": [2.0, 2.0, 2.0],
            "loss": "baseline Dice + CE",
            "optimizer": "baseline SGD",
            "augmentation": "baseline unchanged",
            "foreground_framework": 0.33,
            "deep_supervision_weights": DS_WEIGHTS,
            "source_checkpoint": None if source is None else source["checkpoint"],
            "source_checkpoint_sha256": None if source is None else source["checkpoint_sha256"],
            "source_hashes": sources,
            "v1_guard_before": v1_before,
            "git_commit": git_commit(),
            "started_at": now(),
            "prohibited": ["1000 epochs", "E1+S1 combination", "ADC/FLAIR", "threshold tuning", "loss/spacing/architecture change"],
        }
        save_json(run / "experiment_config.json", config, exclusive=True)
        trainer = make_trainer(args.experiment, args.stage)
        trainer.configure_screen_monitoring(run, args.mode)
        if args.stage == "promote300":
            trainer.load_checkpoint(source["checkpoint"])
            if trainer.current_epoch != 100:
                raise RuntimeError("Continuation did not restore epoch 100")
            if not all_finite(trainer.network.parameters()) or not all_finite(tensors_in(trainer.optimizer.state)):
                raise FloatingPointError("Nonfinite restored model or optimizer state")

        if args.mode == "smoke":
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            expected_epoch = 0 if args.stage == "screen100" else 100
            if trainer.current_epoch != expected_epoch:
                raise RuntimeError("Smoke epoch state mismatch")
            expected_lr = trainer.initial_lr * (1 - expected_epoch / 1000) ** 0.9
            actual_lr = float(trainer.optimizer.param_groups[0]["lr"])
            if not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=1e-14):
                raise RuntimeError("PolyLR continuity mismatch")
            events = []
            for index in range(40):
                if index == 10:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    measured_at = time.monotonic()
                step = trainer.train_step(next(trainer.dataloader_train))
                events.append(dict(trainer.last_amp_event))
            torch.cuda.synchronize()
            seconds = time.monotonic() - measured_at
            trainer.network.eval()
            with torch.no_grad():
                validation = trainer.validation_step(next(trainer.dataloader_val))
            save_json(run / "verification_iterations.json", events, exclusive=True)
            result = {
                "status": "passed",
                "stage": args.stage,
                "restored_epoch": trainer.current_epoch,
                "current_lr": actual_lr,
                "warmup_iterations": 10,
                "measured_iterations": 30,
                "seconds_per_iteration": seconds / 30,
                "skipped_optimizer_steps": trainer.total_skipped,
                "nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
                "validation_loss": float(validation["loss"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            trainer.close_loaders()
        else:
            smoke = json.loads((spec["root"] / args.stage / "smoke/completion.json").read_text(encoding="utf-8"))
            if smoke.get("status") != "passed":
                raise RuntimeError("Smoke test did not pass")
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
        assert_hashes(v1_before)
        result["v1_guard_after"] = v1_guard_manifest()
        result["finished_at"] = now()
        save_json(run / "completion.json", result, exclusive=True)
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        failure = {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()}
        save_json(run / "failure.json", failure)
        if trainer is not None and hasattr(trainer, "dataloader_train"):
            trainer.close_loaders()
        raise


if __name__ == "__main__":
    main()
