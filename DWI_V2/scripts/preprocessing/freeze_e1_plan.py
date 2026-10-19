"""Freeze Dataset101 to the Dataset001 2 mm plan, extended to two Z-score channels."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import E1_EXPERIMENT, E1_PP, E1_RAW, V1_PP, digest, now, save_json  # noqa: E402


def main() -> None:
    completion_path = E1_EXPERIMENT / "preprocessing_qc/frozen_plan_completion.json"
    if completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"))
        return
    fingerprint = E1_PP / "dataset_fingerprint.json"
    dataset_json = E1_PP / "dataset.json"
    if not fingerprint.is_file():
        raise RuntimeError("Dataset101 fingerprint extraction has not completed")
    if not dataset_json.exists():
        shutil.copy2(E1_RAW / "dataset.json", dataset_json)
    if digest(dataset_json) != digest(E1_RAW / "dataset.json"):
        raise RuntimeError("Dataset101 preprocessed dataset.json differs from isolated raw dataset.json")
    fp = json.loads(fingerprint.read_text(encoding="utf-8"))
    channel_props = fp.get("foreground_intensity_properties_per_channel", {})
    if set(map(str, channel_props)) != {"0", "1"}:
        raise RuntimeError(f"Dataset101 fingerprint does not contain two channels: {channel_props.keys()}")

    source_plan_path = V1_PP / "nnUNetPlans.json"
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    plan = copy.deepcopy(source_plan)
    plan["dataset_name"] = E1_PP.name
    plan["plans_name"] = "nnUNetPlans"
    plan["foreground_intensity_properties_per_channel"] = copy.deepcopy(channel_props)
    for config_name, config in plan["configurations"].items():
        config["normalization_schemes"] = ["ZScoreNormalization", "ZScoreNormalization"]
        config["use_mask_for_norm"] = [True, True]
    config = plan["configurations"]["3d_fullres"]
    invariants = {
        "patch_size": config["patch_size"],
        "batch_size": config["batch_size"],
        "spacing": config["spacing"],
        "architecture": config["architecture"]["network_class_name"],
        "normalization_schemes": config["normalization_schemes"],
        "use_mask_for_norm": config["use_mask_for_norm"],
    }
    expected = {
        "patch_size": [80, 96, 80],
        "batch_size": 8,
        "spacing": [2.0, 2.0, 2.0],
        "architecture": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "normalization_schemes": ["ZScoreNormalization", "ZScoreNormalization"],
        "use_mask_for_norm": [True, True],
    }
    if invariants != expected:
        raise RuntimeError(f"Frozen E1 plan invariants differ: {invariants}")
    target_plan = E1_PP / "nnUNetPlans.json"
    save_json(target_plan, plan, exclusive=True)
    shutil.copy2(V1_PP / "splits_final.json", E1_PP / "splits_final.json")
    split_equal = digest(V1_PP / "splits_final.json") == digest(E1_PP / "splits_final.json")
    completion = {
        "status": "complete",
        "time": now(),
        "method": "Dataset001 nnUNetPlans copied; dataset name changed; normalization/mask arrays extended from one to two channels; geometry/batch/architecture unchanged",
        "source_plan": str(source_plan_path),
        "source_plan_sha256": digest(source_plan_path),
        "target_plan_sha256": digest(target_plan),
        "invariants": invariants,
        "fold0_split_byte_identical": split_equal,
    }
    save_json(completion_path, completion, exclusive=True)
    if not split_equal:
        raise RuntimeError("Fold 0 split changed")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
