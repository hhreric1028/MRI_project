"""Fail-stop P3a-100 workflow with conditional checkpoint continuation to epoch 300."""

import json
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path

from recall_common import ROOT, digest, now, save, verify_baseline


SUITE = ROOT / "experiments/P3a_clip_zscore_seed2022"
P3_PP = ROOT / "datasets/ISLES2022/processed/Dataset013_ISLES2022_DWI_ClipZScore"
SOURCE_FILES = [ROOT / "scripts" / name for name in [
    "prepare_p3a_dataset.py", "p3a_trainers.py", "run_p3a_experiment.py", "p3a_pipeline.py",
    "p3a_analysis.py", "test_p3a_protocol.py", "screen100_trainers.py", "recall_trainers.py",
    "amp_resume_trainer.py", "recall_common.py", "recall_analysis.py", "evaluate_experiment.py",
]]


def stage(name: str, script: str, *args) -> None:
    logs = SUITE / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)]
    save(logs / f"{name}.command.json", {"command": command, "time": now()})
    with (logs / f"{name}.log").open("x", encoding="utf-8") as log:
        print(f"\nSTAGE {name}", flush=True)
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace",
                                 env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    save(logs / f"{name}.exit.json", {"exit_code": code, "time": now()})
    if code:
        raise RuntimeError(f"{name} failed ({code}); outputs retained; no automatic retry")


def manifest() -> dict:
    missing = [str(p) for p in SOURCE_FILES if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing P3a sources: {missing}")
    return {str(p): digest(p) for p in SOURCE_FILES}


def check_sources(expected: dict) -> None:
    changed = [p for p, h in expected.items() if not Path(p).is_file() or digest(p) != h]
    if changed:
        raise RuntimeError(f"Frozen P3a sources changed: {changed}")


def train_evaluate(stage_name: str, expected: dict) -> None:
    check_sources(expected)
    stage(f"{stage_name}_smoke", "run_p3a_experiment.py", "--stage", stage_name, "--mode", "smoke")
    check_sources(expected)
    stage(f"{stage_name}_train", "run_p3a_experiment.py", "--stage", stage_name, "--mode", "train")
    completion = json.loads((SUITE / stage_name / "train/completion.json").read_text())
    check_sources(expected)
    stage(f"{stage_name}_evaluation", "evaluate_experiment.py", "--predictions", completion["predictions"],
          "--output", SUITE / stage_name / "metrics")
    check_sources(expected)
    stage(f"{stage_name}_analysis", "p3a_analysis.py", "--stage", stage_name)


def write_final_report(decision: str, final: str) -> None:
    sections = [
        "# P3a DWI intensity preprocessing experiment",
        "",
        "## P3a preprocessing verification",
        "",
        (SUITE / "preprocessing_qc/verification_report.md").read_text(encoding="utf-8"),
        "",
        "## P3a-100 result and promotion decision",
        "",
        (SUITE / "screen100/analysis/report.md").read_text(encoding="utf-8"),
    ]
    if decision == "PROMOTE TO 300":
        sections += ["", "## P3a-300 result and 100-to-300 trend", "",
                     (SUITE / "promote300/analysis/report.md").read_text(encoding="utf-8")]
    else:
        sections += ["", "## P3a-300 result", "", "Not executed because the prespecified P3a-100 promotion rule was not met."]
    sections += [
        "", "## Failure-case analysis", "",
        "Per-case tables and fixed-slice Original DWI / P3a DWI / GT / B0 / P3a panels are saved under each completed stage's analysis/case_analysis directory.",
        "", "## Final decision", "", final, "",
        "No 1000-epoch run was started.", "",
        "Limitations: Fold 0 only; single-seed screening; same seed is not fully deterministic. "
        "Paired case-bootstrap intervals reflect uncertainty in the current validation cohort, not cross-fold, center, or seed generalization. "
        "No matched B0-R2022 epoch-300 checkpoint is available, so any 300-epoch contextual comparison to B0-100 is not horizon matched.",
    ]
    with (SUITE / "final_report.md").open("x", encoding="utf-8", errors="strict") as handle:
        handle.write("\n".join(sections) + "\n")


def main() -> None:
    if not SUITE.is_dir() or not (SUITE / "preprocessing_qc/completion.json").is_file():
        raise RuntimeError("Completed P3a preprocessing QC is unavailable")
    if not (P3_PP / "nnUNetPlans.json").is_file() or not (P3_PP / "splits_final.json").is_file():
        raise RuntimeError("Dataset013 preprocessing is incomplete")
    lock_path = SUITE / "pipeline.lock"
    with lock_path.open("x+b") as lock:
        lock.write(b"0"); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        save(SUITE / "started.json", {"time": now(), "pid": os.getpid()})
        try:
            expected = manifest()
            save(SUITE / "source_manifest.json", expected)
            verify_baseline(SUITE / "baseline_before.json")
            stage("protocol_tests", "test_p3a_protocol.py")
            train_evaluate("screen100", expected)
            decision = json.loads((SUITE / "screen100/analysis/decision.json").read_text())["decision"]
            if decision == "PROMOTE TO 300":
                train_evaluate("promote300", expected)
                final = json.loads((SUITE / "promote300/analysis/trend_and_final_decision.json").read_text())["final_decision"]
            else:
                final = decision
            write_final_report(decision, final)
            verify_baseline(SUITE / "baseline_after.json")
            save(SUITE / "completed.json", {"time": now(), "status": "P3a_workflow_complete",
                 "screen100_decision": decision, "promote300_executed": decision == "PROMOTE TO 300",
                 "final_decision": final, "automatic_1000_started": False})
            print(f"P3a complete: {final}. No 1000-epoch run was started.", flush=True)
        except BaseException as exc:
            save(SUITE / "failure.json", {"time": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            print("STOP: P3a outputs preserved; no automatic retry or 1000-epoch run.", flush=True)
            raise


if __name__ == "__main__":
    main()
