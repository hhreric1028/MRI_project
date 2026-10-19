"""Sequential fail-fast Original B0 Fold 1-4 training, evaluation, and paired analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SUITE = ROOT / "experiments/B0_full5fold_100"
OUT = ROOT / "outputs/analysis/B0_full5fold_100"
STATUS = OUT / "status.json"
B0_FOLD0_PREDICTIONS = (
    ROOT
    / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0/validation"
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(*arguments: str | Path) -> None:
    command = [sys.executable, *map(str, arguments)]
    print(f"[{now()}] RUN {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prediction_path(fold: int) -> Path:
    if fold == 0:
        return B0_FOLD0_PREDICTIONS
    return (
        SUITE
        / f"fold_{fold}/results/Dataset001_ISLES2022_DWI"
        / f"nnUNetTrainer_B0Full5fold_Screen100__nnUNetPlans__3d_fullres/fold_{fold}/validation"
    )


def evaluate_fold(fold: int) -> None:
    metrics = SUITE / f"fold_{fold}/metrics"
    predictions = prediction_path(fold)
    run(
        SCRIPTS / "evaluate_cv_fold.py",
        "--fold", str(fold),
        "--predictions", predictions,
        "--output", metrics,
        "--status", STATUS,
    )
    run(
        SCRIPTS / "kamel_fold_analysis.py",
        "--fold", str(fold),
        "--metrics", metrics,
        "--predictions", predictions,
        "--analysis", OUT,
    )
    run(
        SCRIPTS / "kamel_fp_topology.py",
        "--fold", str(fold),
        "--metrics", metrics,
        "--predictions", predictions,
        "--analysis", OUT,
    )


def main() -> None:
    if not STATUS.exists():
        raise RuntimeError("Configuration audit must complete before batch launch")
    started = now()
    save(OUT / "batch_started.json", {"started_at": started, "python": sys.executable, "order": [0, 1, 2, 3, 4]})
    try:
        run(SCRIPTS / "test_b0_full5fold_protocol.py")
        state = json.loads(STATUS.read_text(encoding="utf-8"))
        if state["fold_0"]["status"] != "reused_verified":
            raise RuntimeError("Fold 0 reuse status changed before unified evaluation")
        evaluate_fold(0)
        for fold in range(1, 5):
            state = json.loads(STATUS.read_text(encoding="utf-8"))
            if state[f"fold_{fold}"]["status"] != "pending":
                raise RuntimeError(f"Fold {fold} is not pending; fail-fast rather than skip")
            run(SCRIPTS / "run_b0_full5fold_fold.py", "--fold", str(fold))
            evaluate_fold(fold)
        run(SCRIPTS / "b0_full5fold_final_analysis.py")
        completion = {"status": "complete", "started_at": started, "finished_at": now()}
        save(OUT / "batch_completion.json", completion)
        print("B0-100 FULL 5-FOLD COMPARISON COMPLETE", flush=True)
        print("WAITING FOR HUMAN REVIEW", flush=True)
    except BaseException as exc:
        failure = {
            "status": "failed_stop_batch",
            "started_at": started,
            "failed_at": now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "policy": "Batch stopped immediately; no later fold was skipped or launched.",
        }
        save(OUT / "batch_failure.json", failure)
        print("B0-100 BATCH STOPPED ON FAILURE", flush=True)
        raise


if __name__ == "__main__":
    main()
