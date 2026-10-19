"""Sequential, event-driven DWI_V2 workflow supervisor.

Child stdout/stderr is written continuously to local stage logs. The supervisor emits
only stage start/exit events and never reads training logs during a healthy run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (  # noqa: E402
    E1_EXPERIMENT,
    PYTHON,
    S1_EXPERIMENT,
    V1_ROOT,
    V2_PP_ROOT,
    V2_RAW_ROOT,
    V2_RESULTS_ROOT,
    V2_ROOT,
    now,
    save_json,
)


LOG_ROOT = V2_ROOT / "outputs/logs/v2_workflow"
STATE_PATH = LOG_ROOT / "workflow_state.json"
FAILURE_PATH = LOG_ROOT / "workflow_failure.json"
COMPLETION_PATH = LOG_ROOT / "workflow_completion.json"
NNUNET_SCRIPTS = PYTHON.parent / "Scripts"


def load_state() -> dict:
    if not STATE_PATH.is_file():
        return {"status": "running", "started_at": now(), "completed_stages": [], "skipped_stages": []}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def persist(state: dict) -> None:
    state["updated_at"] = now()
    save_json(STATE_PATH, state)


def stage(name: str, command: list[str], state: dict, env: dict[str, str] | None = None) -> None:
    if name in state["completed_stages"]:
        print(f"STAGE_SKIP {name} already_complete", flush=True)
        return
    reviewed_retry = state.get("reviewed_failure_stage") == name
    if reviewed_retry:
        retry = 1
        while (LOG_ROOT / f"{name}.resume{retry}.log").exists():
            retry += 1
        log_path = LOG_ROOT / f"{name}.resume{retry}.log"
    else:
        log_path = LOG_ROOT / f"{name}.log"
    if log_path.exists():
        raise RuntimeError(f"Stage log exists without completed state; manual review required: {log_path}")
    print(f"STAGE_START {name}", flush=True)
    with log_path.open("x", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=V1_ROOT,
            env=os.environ.copy() if env is None else env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = process.wait()
    print(f"STAGE_EXIT {name} code={code}", flush=True)
    if code != 0:
        failure = {
            "status": "failed",
            "time": now(),
            "stage": name,
            "exit_code": code,
            "log": str(log_path),
            "results_preserved": True,
            "automatic_retry": False,
        }
        save_json(FAILURE_PATH, failure)
        state["status"] = "failed"
        state["failure"] = failure
        persist(state)
        raise RuntimeError(f"Stage failed: {name} (exit {code})")
    state["completed_stages"].append(name)
    if reviewed_retry:
        state.pop("reviewed_failure_stage", None)
    persist(state)


def skip(name: str, reason: str, state: dict) -> None:
    if name not in state["skipped_stages"]:
        state["skipped_stages"].append(name)
        state.setdefault("skip_reasons", {})[name] = reason
        persist(state)
    print(f"STAGE_SKIP {name} {reason}", flush=True)


def nnunet_env() -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(V2_RAW_ROOT)
    env["nnUNet_preprocessed"] = str(V2_PP_ROOT)
    env["nnUNet_results"] = str(V2_RESULTS_ROOT)
    env["nnUNet_compile"] = "false"
    env["MPLCONFIGDIR"] = str(LOG_ROOT / ".matplotlib")
    return env


def py(script: str, *args: str) -> list[str]:
    return [str(PYTHON), str(V2_ROOT / "scripts" / script), *args]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-after-review",
        action="store_true",
        help="Resume only after a human has reviewed a prior failed stage and its preserved outputs.",
    )
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if state.get("status") == "complete":
        print("WORKFLOW_EXIT complete", flush=True)
        return
    if state.get("status") == "failed" and not args.resume_after_review:
        raise RuntimeError(f"Previous failure requires manual review before resume: {FAILURE_PATH}")
    if state.get("status") == "failed":
        failed_stage = state.get("failure", {}).get("stage")
        if failed_stage:
            state["reviewed_failure_stage"] = failed_stage
        state["status"] = "running"
        state.pop("failure", None)
        persist(state)

    base_env = nnunet_env()
    try:
        stage("00_protocol_tests", [str(PYTHON), "-m", "unittest", "discover", "-s", str(V2_ROOT / "tests"), "-p", "test_*.py"], state)
        stage("01_prepare_e1_raw", py("preprocessing/prepare_e1_dataset.py"), state)
        stage(
            "02_e1_fingerprint",
            [str(NNUNET_SCRIPTS / "nnUNetv2_extract_fingerprint.exe"), "-d", "101", "-np", "4", "--verify_dataset_integrity", "--no_pbar"],
            state,
            base_env,
        )
        stage("03_freeze_e1_plan", py("preprocessing/freeze_e1_plan.py"), state)
        stage("03b_verify_e1_plan_channels", py("preprocessing/verify_e1_plan_channel_properties.py"), state)
        stage(
            "04_e1_preprocess",
            [str(NNUNET_SCRIPTS / "nnUNetv2_preprocess.exe"), "-d", "101", "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "4", "--no_pbar"],
            state,
            base_env,
        )
        stage("05_verify_e1_preprocessed", py("preprocessing/verify_e1_preprocessed.py"), state)
        stage("06_s1_sampling_audit", py("preprocessing/analyze_s1_sampling.py"), state)

        stage("07_e1_smoke100", py("trainers/run_v2_experiment.py", "--experiment", "E1", "--stage", "screen100", "--mode", "smoke"), state)
        stage("08_e1_train100", py("trainers/run_v2_experiment.py", "--experiment", "E1", "--stage", "screen100", "--mode", "train"), state)
        stage("09_e1_evaluate100", py("evaluation/evaluate_v2_candidate.py", "--experiment", "E1", "--stage", "screen100"), state)

        audit = json.loads((S1_EXPERIMENT / "sampling_audit/completion.json").read_text(encoding="utf-8"))
        if audit.get("training_allowed"):
            stage("10_s1_smoke100", py("trainers/run_v2_experiment.py", "--experiment", "S1", "--stage", "screen100", "--mode", "smoke"), state)
            stage("11_s1_train100", py("trainers/run_v2_experiment.py", "--experiment", "S1", "--stage", "screen100", "--mode", "train"), state)
            stage("12_s1_evaluate100", py("evaluation/evaluate_v2_candidate.py", "--experiment", "S1", "--stage", "screen100"), state)
        else:
            for name in ["10_s1_smoke100", "11_s1_train100", "12_s1_evaluate100"]:
                skip(name, "sampling_intervention_ineffective", state)

        stage("13_select_promotion", py("evaluation/finalize_v2_workflow.py", "--mode", "select"), state)
        selection = json.loads((V2_ROOT / "outputs/comparisons/v2_promotion_selection.json").read_text(encoding="utf-8"))
        selected = selection.get("selected_experiment")
        if selected:
            stage("14_promoted_smoke300", py("trainers/run_v2_experiment.py", "--experiment", selected, "--stage", "promote300", "--mode", "smoke"), state)
            stage("15_promoted_train300", py("trainers/run_v2_experiment.py", "--experiment", selected, "--stage", "promote300", "--mode", "train"), state)
            stage("16_promoted_evaluate300", py("evaluation/evaluate_v2_candidate.py", "--experiment", selected, "--stage", "promote300"), state)
        else:
            for name in ["14_promoted_smoke300", "15_promoted_train300", "16_promoted_evaluate300"]:
                skip(name, "no_failure_robustness_signal", state)
        stage("17_final_report", py("evaluation/finalize_v2_workflow.py", "--mode", "final"), state)
        state["status"] = "complete"
        state["finished_at"] = now()
        state["full_1000_epoch_training"] = False
        persist(state)
        save_json(COMPLETION_PATH, state, exclusive=True)
        print("WORKFLOW_EXIT complete", flush=True)
    except BaseException as exc:
        if not FAILURE_PATH.exists():
            failure = {
                "status": "failed",
                "time": now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "results_preserved": True,
                "automatic_retry": False,
            }
            save_json(FAILURE_PATH, failure)
            state["status"] = "failed"
            state["failure"] = failure
            persist(state)
        raise


if __name__ == "__main__":
    main()
