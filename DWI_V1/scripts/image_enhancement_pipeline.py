"""Fail-stop QC -> 100 -> promotion (max two) -> 300 enhancement workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from image_enhancement_common import EXPERIMENTS, ROOT, SUITE, experiment_dir
from recall_common import digest, now, save, verify_baseline


SOURCE_FILES = [ROOT / "scripts" / name for name in [
    "image_enhancement_common.py", "prepare_image_enhancement_datasets.py",
    "image_enhancement_trainers.py", "run_image_enhancement_experiment.py",
    "image_enhancement_analysis.py", "image_enhancement_pipeline.py",
    "test_image_enhancement_protocol.py", "screen100_trainers.py",
    "recall_trainers.py", "amp_resume_trainer.py", "recall_analysis.py",
    "recall_common.py", "evaluate_experiment.py",
]]


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise RuntimeError(f"Log collision: {log_path}")
    print("RUN " + " ".join(command), flush=True)
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
    lines = []
    for line in child.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = child.wait()
    with log_path.open("x", encoding="utf-8") as handle:
        handle.writelines(lines)
    save(log_path.with_suffix(".command.json"), {"command": command, "time": now()})
    save(log_path.with_suffix(".exit.json"), {"exit_code": code, "time": now()})
    if code:
        raise RuntimeError(f"Stage failed with exit code {code}: {command}")


def run_preparation() -> None:
    command = [sys.executable, "-u", str(ROOT / "scripts/prepare_image_enhancement_datasets.py")]
    child = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )
    lines = []
    for line in child.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = child.wait()
    if SUITE.exists():
        log = SUITE / "qc/preparation.log"
        with log.open("x", encoding="utf-8") as handle:
            handle.writelines(lines)
        save(SUITE / "qc/preparation.command.json", {"command": command, "time": now()})
        save(SUITE / "qc/preparation.exit.json", {"exit_code": code, "time": now()})
    if code:
        raise RuntimeError(f"Preparation failed with exit code {code}")


def manifest() -> dict:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing enhancement sources: {missing}")
    return {str(path): digest(path) for path in SOURCE_FILES}


def check_sources(expected: dict) -> None:
    changed = [path for path, expected_hash in expected.items() if not Path(path).is_file() or digest(path) != expected_hash]
    if changed:
        raise RuntimeError(f"Frozen enhancement sources changed during workflow: {changed}")


def stage(experiment: str, label: str, script: str, *args: str) -> None:
    log = experiment_dir(experiment) / "logs" / f"{label}.log"
    run_command([sys.executable, "-u", str(ROOT / "scripts" / script), *args], log)


def train_evaluate_analyse(experiment: str, stage_name: str, expected: dict) -> None:
    exp = experiment_dir(experiment)
    check_sources(expected)
    stage(experiment, f"{stage_name}_smoke", "run_image_enhancement_experiment.py", "--experiment", experiment, "--stage", stage_name, "--mode", "smoke")
    check_sources(expected)
    stage(experiment, f"{stage_name}_train", "run_image_enhancement_experiment.py", "--experiment", experiment, "--stage", stage_name, "--mode", "train")
    completion = json.loads((exp / stage_name / "train/completion.json").read_text())
    check_sources(expected)
    stage(
        experiment,
        f"{stage_name}_evaluation",
        "evaluate_experiment.py",
        "--predictions", completion["predictions"],
        "--output", str(exp / "metrics" / stage_name),
    )
    check_sources(expected)
    stage(experiment, f"{stage_name}_analysis", "image_enhancement_analysis.py", "--experiment", experiment, "--stage", stage_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-after-preparation", action="store_true")
    parser.add_argument("--resume-from-e1-analysis", action="store_true")
    args = parser.parse_args()
    if args.resume_after_preparation and args.resume_from_e1_analysis:
        raise RuntimeError("Choose only one resume point")
    if args.resume_from_e1_analysis:
        required = [
            SUITE / "qc/completion.json",
            experiment_dir("E1_gamma") / "screen100/train/completion.json",
            experiment_dir("E1_gamma") / "metrics/screen100/summary.json",
        ]
        if not all(path.is_file() for path in required) or (SUITE / "completed.json").exists():
            raise RuntimeError("E1-analysis resume prerequisites are incomplete")
    elif args.resume_after_preparation:
        if not SUITE.is_dir() or (SUITE / "completed.json").exists():
            raise RuntimeError("Resume requires an existing incomplete workflow")
        if not (SUITE / "qc/completion.json").is_file():
            run_command(
                [sys.executable, "-u", str(ROOT / "scripts/prepare_image_enhancement_datasets.py"), "--resume-preprocessing"],
                SUITE / "qc/preparation_resume.log",
            )
    else:
        if SUITE.exists():
            raise RuntimeError("Enhancement suite already exists; refusing duplicate workflow")
        run_preparation()
    if not (SUITE / "qc/completion.json").is_file():
        raise RuntimeError("Enhancement QC/preprocessing did not complete")
    try:
        expected = manifest()
        manifest_name = "source_manifest_resume_e1_analysis.json" if args.resume_from_e1_analysis else "source_manifest.json"
        save(SUITE / manifest_name, expected)
        baseline_before_name = "baseline_before_training_resume_e1_analysis.json" if args.resume_from_e1_analysis else "baseline_before_training.json"
        verify_baseline(SUITE / baseline_before_name)
        test_log = SUITE / "qc" / ("protocol_tests_resume_e1_analysis.log" if args.resume_from_e1_analysis else "protocol_tests.log")
        run_command([sys.executable, "-u", str(ROOT / "scripts/test_image_enhancement_protocol.py")], test_log)
        if args.resume_from_e1_analysis:
            check_sources(expected)
            stage("E1_gamma", "screen100_analysis", "image_enhancement_analysis.py", "--experiment", "E1_gamma", "--stage", "screen100")
            remaining = [name for name in EXPERIMENTS if name != "E1_gamma"]
        else:
            remaining = list(EXPERIMENTS)
        for experiment in remaining:
            train_evaluate_analyse(experiment, "screen100", expected)
        run_command(
            [sys.executable, "-u", str(ROOT / "scripts/image_enhancement_analysis.py"), "--select-promotions"],
            SUITE / "promotion_selection.log",
        )
        promoted = json.loads((SUITE / "promotion_selection.json").read_text())["selected_maximum_two"]
        if len(promoted) > 2:
            raise RuntimeError("Promotion cap violated")
        for experiment in promoted:
            train_evaluate_analyse(experiment, "promote300", expected)
        run_command(
            [sys.executable, "-u", str(ROOT / "scripts/image_enhancement_analysis.py"), "--final-report"],
            SUITE / "final_report_generation.log",
        )
        verify_baseline(SUITE / "baseline_after_training.json")
        final = json.loads((SUITE / "final_decision.json").read_text())
        save(SUITE / "completed.json", {
            "time": now(),
            "status": "complete",
            "promoted_to_300": promoted,
            "final_decisions": final["final_decisions"],
            "automatic_1000_started": False,
        })
        print("Enhancement workflow complete at the 300-epoch hard stop. No 1000-epoch run started.", flush=True)
    except BaseException as exc:
        if SUITE.exists() and not (SUITE / "failure.json").exists():
            save(SUITE / "failure.json", {"time": now(), "error": repr(exc), "traceback": traceback.format_exc(), "automatic_1000_started": False})
        print("STOP: outputs retained; no automatic retry and no 1000-epoch run.", flush=True)
        raise


if __name__ == "__main__":
    main()
