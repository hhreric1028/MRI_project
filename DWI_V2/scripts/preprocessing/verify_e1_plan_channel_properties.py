"""Verify/repair Dataset101 plan intensity metadata after a copied-plan freeze."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import E1_EXPERIMENT, E1_PP, digest, now, save_json  # noqa: E402


def main() -> None:
    completion_path = E1_EXPERIMENT / "preprocessing_qc/plan_channel_properties_completion.json"
    plan_path = E1_PP / "nnUNetPlans.json"
    fingerprint_path = E1_PP / "dataset_fingerprint.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    expected = fingerprint["foreground_intensity_properties_per_channel"]
    if set(expected) != {"0", "1"}:
        raise RuntimeError(f"Fingerprint does not define exactly two channels: {expected.keys()}")
    action = "verified"
    backup = E1_PP / "nnUNetPlans.pre_channel_properties_repair.json"
    if plan.get("foreground_intensity_properties_per_channel") != expected:
        if not backup.exists():
            shutil.copy2(plan_path, backup)
        plan["foreground_intensity_properties_per_channel"] = copy.deepcopy(expected)
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(plan_path)
        action = "repaired"
    verified = json.loads(plan_path.read_text(encoding="utf-8"))["foreground_intensity_properties_per_channel"]
    if verified != expected:
        raise RuntimeError("Plan channel intensity properties still differ from Dataset101 fingerprint")
    completion = {
        "status": "complete",
        "time": now(),
        "action": action,
        "channels": sorted(verified),
        "plan_sha256": digest(plan_path),
        "fingerprint_sha256": digest(fingerprint_path),
        "preserved_pre_repair_plan": str(backup) if backup.exists() else None,
    }
    if completion_path.exists():
        previous = json.loads(completion_path.read_text(encoding="utf-8"))
        if previous.get("plan_sha256") != completion["plan_sha256"]:
            raise RuntimeError("Existing plan verification marker does not match current plan")
    else:
        save_json(completion_path, completion, exclusive=True)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
