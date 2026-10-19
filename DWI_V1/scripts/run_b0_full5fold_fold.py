"""Train and infer one new Original B0-100 fold for the full five-fold family."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from b0_full5fold_trainers import B0_FULL5FOLD_TRAINER
from recall_common import PP, PROTECTED, ROOT, digest, now, protect_baseline, provenance, save, verify_baseline
from recall_trainers import DS_WEIGHTS


SUITE = ROOT / "experiments/B0_full5fold_100"
ANALYSIS = ROOT / "outputs/analysis/B0_full5fold_100"
STATUS = ANALYSIS / "status.json"
EXISTING_EXPERIMENTS = [
    ROOT / "experiments/B0-R2022_resume_from_epoch34",
    ROOT / "experiments/P1_spacing_1p5mm",
    ROOT / "experiments/recall_2mm_seed2022",
    ROOT / "experiments/recall_screen100_seed2022_r2",
    ROOT / "experiments/A1_DWIIntensityAug_seed2022",
    ROOT / "experiments/kamel_replication_seed2022",
    ROOT / "outputs/analysis/kamel_replication",
    ROOT / "outputs/models/Dataset001_ISLES2022_DWI",
]
TRAINING_SOURCE_NAMES = [
    "b0_full5fold_trainers.py",
    "run_b0_full5fold_fold.py",
    "screen100_trainers.py",
    "recall_trainers.py",
    "amp_resume_trainer.py",
    "train_spacing_experiment.py",
]


def seed_all() -> None:
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)


def update_status(fold: int, **updates) -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    status[f"fold_{fold}"].update(updates)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def make_trainer(fold: int):
    plans = json.loads((PP / "nnUNetPlans.json").read_text(encoding="utf-8"))
    cfg = plans["configurations"]["3d_fullres"]
    expected = {"patch_size": [80, 96, 80], "batch_size": 8, "spacing": [2.0, 2.0, 2.0]}
    if any(cfg[key] != value for key, value in expected.items()):
        raise RuntimeError("Audited B0 plans changed")
    plans["continue_training"] = False
    trainer = B0_FULL5FOLD_TRAINER(
        plans,
        "3d_fullres",
        fold,
        json.loads((PP / "dataset.json").read_text(encoding="utf-8")),
        device=torch.device("cuda"),
    )
    if trainer.num_epochs != 100 or trainer.num_iterations_per_epoch != 250 or trainer.num_val_iterations_per_epoch != 50:
        raise RuntimeError("B0 run length or iteration count changed")
    trainer.execution_started = time.monotonic()
    return trainer


def training_source_hashes() -> dict[str, str]:
    return {str(ROOT / "scripts" / name): digest(ROOT / "scripts" / name) for name in TRAINING_SOURCE_NAMES}


def summarize_training(run: Path, fold: int, trainer, runtime_seconds: float) -> dict:
    rows = [json.loads(line) for line in (run / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 100 or rows[-1]["epoch"] != 100:
        raise RuntimeError(f"Fold {fold} did not produce exactly 100 epoch rows")
    with (run / "training_log.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_best = Path(trainer.output_folder) / "checkpoint_best.pth"
    best_state = torch.load(checkpoint_best, map_location="cpu", weights_only=False)
    summary = {
        "fold": fold,
        "best_epoch": int(best_state["current_epoch"]),
        "best_validation_metric": float(best_state["_best_ema"]),
        "best_validation_metric_definition": "EMA of patch pseudo-Dice used by nnUNet checkpoint_best",
        "final_epoch": int(rows[-1]["epoch"]),
        "final_train_loss": float(rows[-1]["train_loss"]),
        "final_val_loss": float(rows[-1]["validation_loss"]),
        "final_pseudo_dice": float(rows[-1]["patch_pseudo_dice"]),
        "amp_overflow_count": int(trainer.total_skipped),
        "unrecovered_consecutive_overflows": int(trainer.consecutive_overflows),
        "runtime_seconds": runtime_seconds,
        "checkpoint_best": str(checkpoint_best),
        "checkpoint_final": str(Path(trainer.output_folder) / "checkpoint_final.pth"),
    }
    save(run / f"fold_{fold}_training_summary.json", summary)
    save(ANALYSIS / f"fold_{fold}_training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 5), required=True)
    args = parser.parse_args()
    fold = args.fold
    if not STATUS.exists():
        raise RuntimeError("Run B0/Kamel configuration audit before training")
    status = json.loads(STATUS.read_text(encoding="utf-8"))[f"fold_{fold}"]
    if status["status"] != "pending":
        raise RuntimeError(f"Fold {fold} status is not pending: {status['status']}")
    run = SUITE / f"fold_{fold}"
    run.mkdir(parents=True, exist_ok=False)
    PROTECTED.extend(EXISTING_EXPERIMENTS)
    protect_baseline()
    trainer = None
    started_wall = datetime.now().astimezone()
    started = time.monotonic()
    update_status(fold, status="running", start_time=started_wall.isoformat(), validation_status="pending")
    try:
        verify_baseline(run / "baseline_before.json")
        os.environ["nnUNet_results"] = str(run / "results")
        seed_all()
        torch.set_num_threads(8)
        meta = provenance()
        meta["b0_training_source_sha256"] = training_source_hashes()
        if fold > 1:
            reference = json.loads((SUITE / "fold_1/experiment_config.json").read_text(encoding="utf-8"))
            for key in ["packages", "plans_sha256", "split_sha256", "b0_training_source_sha256"]:
                if meta[key] != reference[key]:
                    raise RuntimeError(f"Environment/source drift before fold {fold}: {key}")
        trainer = make_trainer(fold)
        trainer.configure_screen_monitoring(run, "train")
        expected_epoch100_lr = trainer.initial_lr * (1 - 99 / 1000) ** 0.9
        meta.update(
            experiment="Original B0-100 full five-fold family",
            fold=fold,
            actual_run_epochs=100,
            seed_python=2022,
            seed_numpy=2022,
            seed_torch=2022,
            seed_cuda=2022,
            worker_seeds_train=[2022, 2023, 2024, 2025],
            worker_seeds_val=[3022, 3023],
            worker_order="NonDetMultiThreadedAugmenter; arrival order not deterministic",
            trainer_class=trainer.__class__.__name__,
            trainer_semantics="Original nnUNetTrainer_B0R2022_Screen100; do_split generalized only to fixed folds 0-4",
            configuration=trainer.configuration_manager.configuration,
            train_foreground=trainer.train_foreground,
            validation_foreground=0.33,
            loss_definition="unchanged B0 Dice+CE",
            optimizer="unchanged B0 SGD momentum 0.99 Nesterov; weight decay 3e-5",
            lr_schedule="PolyLR exponent 0.9; 1000-epoch reference horizon",
            lr_epoch_index_0=trainer.initial_lr,
            lr_epoch_index_99=expected_epoch100_lr,
            deep_supervision_weights=DS_WEIGHTS,
            augmentation="nnUNet v2.8.1 default; no override",
            inference="default saved-checkpoint sliding-window validation; probabilities retained",
            postprocessing="none",
            pretrained_weights=None,
            fresh_initialization=True,
            prohibited="No A1/S1a/L2a/P3a/P1/ResEnc/custom sampling/custom threshold/ensemble/follow-up experiment",
        )
        save(run / "experiment_config.json", meta)
        trainer.run_training()
        checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
        if trainer.current_epoch != 100 or not checkpoint.exists():
            raise RuntimeError(f"Fold {fold} training did not stop cleanly at 100 epochs")
        runtime_seconds = time.monotonic() - started
        update_status(
            fold,
            status="trained",
            runtime_seconds=runtime_seconds,
            checkpoint_path=str(checkpoint),
            validation_status="running",
        )
        trainer.perform_actual_validation(save_probabilities=True)
        predictions = checkpoint.parent / "validation"
        if len(list(predictions.glob("*.nii.gz"))) != 50 or len(list(predictions.glob("*.npz"))) != 50:
            raise RuntimeError(f"Fold {fold} validation output is incomplete")
        training_summary = summarize_training(run, fold, trainer, runtime_seconds)
        result = {
            "status": "training_and_prediction_complete",
            "fold": fold,
            "actual_epochs": trainer.current_epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "predictions": str(predictions),
            "total_amp_skipped_steps": trainer.total_skipped,
            "total_nonfinite_gradient_events": trainer.total_nonfinite_gradient_events,
            "unrecovered_consecutive_overflows": trainer.consecutive_overflows,
            "training_summary": training_summary,
            "finished_at": now(),
        }
        verify_baseline(run / "baseline_after.json")
        save(run / "completion.json", result)
        update_status(
            fold,
            status="predicted",
            end_time=result["finished_at"],
            runtime_seconds=time.monotonic() - started,
            checkpoint_path=str(checkpoint),
            validation_status="prediction_complete_evaluation_pending",
        )
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        failure = {"time": now(), "fold": fold, "error": repr(exc), "traceback": traceback.format_exc()}
        save(run / "failure.json", failure)
        update_status(
            fold,
            status="failed",
            end_time=failure["time"],
            runtime_seconds=time.monotonic() - started,
            validation_status="failed",
        )
        if trainer:
            trainer.close_loaders()
        if not (run / "baseline_after.json").exists():
            verify_baseline(run / "baseline_after.json")
        raise


if __name__ == "__main__":
    main()
