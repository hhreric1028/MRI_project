"""Serial fail-stop pipeline for B0/S1a/L2a 100-epoch screening."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path

from recall_common import ROOT, digest, now, save, verify_baseline


SUITE = ROOT / "experiments/recall_screen100_seed2022_r2"
REUSED_SAMPLING_AUDIT = ROOT / "experiments/recall_screen100_seed2022/sampling_audit"
EXPERIMENTS = ["B0_R2022_screen100", "S1a_screen100_seed2022", "L2a_screen100_seed2022"]
SOURCE_FILES = [
    ROOT / "scripts/screen100_trainers.py",
    ROOT / "scripts/run_screen100_experiment.py",
    ROOT / "scripts/screen100_pipeline.py",
    ROOT / "scripts/screen100_analysis.py",
    ROOT / "scripts/test_screen100_protocol.py",
    ROOT / "scripts/amp_resume_trainer.py",
    ROOT / "scripts/recall_trainers.py",
    ROOT / "scripts/recall_common.py",
    ROOT / "scripts/recall_analysis.py",
    ROOT / "scripts/recall_sampling.py",
    ROOT / "scripts/evaluate_experiment.py",
    ROOT / "scripts/train_spacing_experiment.py",
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


def source_manifest() -> dict:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing screening sources: {missing}")
    return {str(path): digest(path) for path in SOURCE_FILES}


def check_sources(expected: dict) -> None:
    changed = [path for path, value in expected.items() if not Path(path).is_file() or digest(path) != value]
    if changed:
        raise RuntimeError(f"Frozen screening sources changed: {changed}")


def train_and_evaluate(name: str, expected_sources: dict) -> None:
    check_sources(expected_sources)
    stage(f"{name}_smoke", "run_screen100_experiment.py", "--experiment", name, "--mode", "smoke")
    check_sources(expected_sources)
    stage(f"{name}_train", "run_screen100_experiment.py", "--experiment", name, "--mode", "train")
    completion = json.loads((SUITE / name / "train/completion.json").read_text())
    check_sources(expected_sources)
    stage(
        f"{name}_evaluation",
        "evaluate_experiment.py",
        "--predictions",
        completion["predictions"],
        "--output",
        SUITE / name / "metrics",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all"], default="all")
    args = parser.parse_args()
    del args
    SUITE.mkdir(parents=True, exist_ok=False)
    with (SUITE / "pipeline.lock").open("a+b") as lock:
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        save(SUITE / "screening_started.json", {"time": now(), "pid": os.getpid()})
        try:
            expected_sources = source_manifest()
            save(SUITE / "source_manifest.json", expected_sources)
            verify_baseline(SUITE / "baseline_before.json")
            stage("screen100_protocol_tests", "test_screen100_protocol.py")
            stage("amp_safety_tests", "test_amp_resume.py")
            check_sources(expected_sources)
            audit_summary = REUSED_SAMPLING_AUDIT / "summary.json"
            audit_completion = REUSED_SAMPLING_AUDIT / "completion.json"
            if not audit_summary.is_file() or not audit_completion.is_file():
                raise RuntimeError("Completed sampling audit is unavailable for read-only reuse")
            completion = json.loads(audit_completion.read_text())
            if completion.get("status") != "audit_complete_review_required":
                raise RuntimeError("Sampling audit completion status is invalid")
            save(
                SUITE / "sampling_audit_reference.json",
                {
                    "mode": "read-only reuse; no resampling",
                    "source": str(REUSED_SAMPLING_AUDIT),
                    "summary_sha256": digest(audit_summary),
                    "completion_sha256": digest(audit_completion),
                },
            )
            print(f"Reusing completed sampling audit: {audit_summary}", flush=True)
            for name in EXPERIMENTS:
                train_and_evaluate(name, expected_sources)
            check_sources(expected_sources)
            stage("screening_comparison", "screen100_analysis.py")
            verify_baseline(SUITE / "baseline_after.json")
            save(
                SUITE / "screening_completed.json",
                {
                    "time": now(),
                    "status": "screening_complete",
                    "automatic_followup_started": False,
                    "next_step": "Review STOP/PROMISING/PROMOTE decisions before any 300-epoch run",
                },
            )
            print("100-epoch screening complete. No 300/1000-epoch follow-up was started.", flush=True)
        except BaseException as exc:
            save(
                SUITE / "screening_failure.json",
                {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()},
            )
            print("STOP: screening outputs preserved; no automatic retry or longer training.", flush=True)
            raise


if __name__ == "__main__":
    main()
