"""Verify the isolated two-channel preprocessed Dataset101 before training."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import blosc2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import E1_EXPERIMENT, E1_PP, V1_PP, digest, now, save_json  # noqa: E402


def main() -> None:
    completion_path = E1_EXPERIMENT / "preprocessing_qc/preprocessed_completion.json"
    if completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"))
        return
    plan = json.loads((E1_PP / "nnUNetPlans.json").read_text(encoding="utf-8"))
    config = plan["configurations"]["3d_fullres"]
    if config["patch_size"] != [80, 96, 80] or config["batch_size"] != 8 or config["spacing"] != [2.0, 2.0, 2.0]:
        raise RuntimeError("Dataset101 controlled plan geometry changed")
    if config["normalization_schemes"] != ["ZScoreNormalization", "ZScoreNormalization"] or config["use_mask_for_norm"] != [True, True]:
        raise RuntimeError("Dataset101 two-channel normalization changed")
    folder = E1_PP / "nnUNetPlans_3d_fullres"
    cases = sorted(path.stem for path in folder.glob("case_*.pkl"))
    if len(cases) != 250:
        raise RuntimeError(f"Expected 250 preprocessed cases, found {len(cases)}")
    rows = []
    for case in cases:
        data = np.asarray(blosc2.open(urlpath=str(folder / f"{case}.b2nd"), mode="r")[:])
        seg = np.asarray(blosc2.open(urlpath=str(folder / f"{case}_seg.b2nd"), mode="r")[:])
        with (folder / f"{case}.pkl").open("rb") as handle:
            properties = pickle.load(handle)
        if data.shape[0] != 2 or seg.shape[0] != 1 or data.shape[1:] != seg.shape[1:]:
            raise RuntimeError(f"Preprocessed channel/seg shape mismatch: {case} {data.shape} {seg.shape}")
        if not np.isfinite(data).all():
            raise RuntimeError(f"Nonfinite preprocessed data: {case}")
        fg = seg[0] >= 0
        rows.append(
            {
                "case": case,
                "shape": "x".join(map(str, data.shape)),
                "finite": True,
                "channel0_mean_foreground": float(data[0][fg].mean()),
                "channel0_sd_foreground": float(data[0][fg].std()),
                "channel1_mean_foreground": float(data[1][fg].mean()),
                "channel1_sd_foreground": float(data[1][fg].std()),
                "class_location_keys": ";".join(map(str, properties["class_locations"].keys())),
            }
        )
    csv_path = E1_EXPERIMENT / "preprocessing_qc/preprocessed_per_case.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_equal = digest(V1_PP / "splits_final.json") == digest(E1_PP / "splits_final.json")
    completion = {
        "status": "complete",
        "time": now(),
        "cases": len(rows),
        "channels": 2,
        "all_finite": True,
        "all_shapes_aligned": True,
        "fold0_split_byte_identical": split_equal,
        "plan_invariants": {"patch_size": config["patch_size"], "batch_size": config["batch_size"], "spacing": config["spacing"]},
    }
    save_json(completion_path, completion, exclusive=True)
    if not split_equal:
        raise RuntimeError("Fold 0 split differs after preprocessing")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()

