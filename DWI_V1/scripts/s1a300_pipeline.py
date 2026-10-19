"""Fail-stop S1a epoch-100 to epoch-300 promotion pipeline."""

import json
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path

from recall_common import ROOT, digest, now, save


SUITE = ROOT / "experiments/S1a_promote300_seed2022_from100"
SOURCE_FILES = [
    ROOT / "scripts/s1a300_trainer.py",
    ROOT / "scripts/run_s1a300_experiment.py",
    ROOT / "scripts/s1a300_pipeline.py",
    ROOT / "scripts/s1a300_analysis.py",
    ROOT / "scripts/test_s1a300_protocol.py",
    ROOT / "scripts/screen100_trainers.py",
    ROOT / "scripts/recall_trainers.py",
    ROOT / "scripts/amp_resume_trainer.py",
    ROOT / "scripts/recall_common.py",
    ROOT / "scripts/recall_analysis.py",
    ROOT / "scripts/evaluate_experiment.py",
]


def stage(name: str, script: str, *args) -> None:
    logs = SUITE / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)]
    save(logs / f"{name}.command.json", {"command": command, "time": now()})
    with (logs / f"{name}.log").open("x", encoding="utf-8") as log:
        print(f"\nSTAGE {name}", flush=True)
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    save(logs / f"{name}.exit.json", {"exit_code": code, "time": now()})
    if code:
        raise RuntimeError(f"{name} failed ({code}); outputs retained; no automatic retry")


def manifest() -> dict:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing S1a-300 sources: {missing}")
    return {str(path): digest(path) for path in SOURCE_FILES}


def check_sources(expected: dict) -> None:
    changed = [path for path, value in expected.items() if not Path(path).is_file() or digest(path) != value]
    if changed:
        raise RuntimeError(f"Frozen S1a-300 sources changed: {changed}")


def main() -> None:
    SUITE.mkdir(parents=True, exist_ok=False)
    with (SUITE / "pipeline.lock").open("a+b") as lock:
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        save(SUITE / "started.json", {"time": now(), "pid": os.getpid()})
        try:
            expected = manifest()
            save(SUITE / "source_manifest.json", expected)
            stage("protocol_tests", "test_s1a300_protocol.py")
            check_sources(expected)
            stage("continuation_smoke", "run_s1a300_experiment.py", "--mode", "smoke")
            check_sources(expected)
            stage("continuation_train", "run_s1a300_experiment.py", "--mode", "train")
            completion = json.loads((SUITE / "train/completion.json").read_text())
            check_sources(expected)
            stage(
                "evaluation",
                "evaluate_experiment.py",
                "--predictions",
                completion["predictions"],
                "--output",
                SUITE / "metrics",
            )
            check_sources(expected)
            stage("trend_analysis", "s1a300_analysis.py")
            save(SUITE / "completed.json", {
                "time": now(),
                "status": "S1a_300_training_evaluation_and_trend_analysis_complete",
                "automatic_followup_started": False,
                "next_step": "Review S1a 100-to-300 paired trend before any longer run",
            })
            print("S1a-300 promotion complete. No longer follow-up was started.", flush=True)
        except BaseException as exc:
            save(SUITE / "failure.json", {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            print("STOP: S1a-300 outputs preserved; no automatic retry or longer training.", flush=True)
            raise


if __name__ == "__main__":
    main()

