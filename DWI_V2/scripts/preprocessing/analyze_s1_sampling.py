"""Training-only low-salience definition and paired 1,000-patch sampling audit."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import blosc2
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import S1_EXPERIMENT, S1_HARD, V1_PP, git_commit, now, save_json  # noqa: E402


PERCENTILE_CANDIDATES = [20, 30, 40]
PREFERENCE_CANDIDATES = [1 / 3, 1 / 2, 2 / 3]
AUDIT_PATCHES = 1000
INITIAL_PATCH_SIZE = np.array([137, 144, 128], dtype=int)
FINAL_PATCH_SIZE = np.array([80, 96, 80], dtype=int)
STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def case_seed(case: str, salt: str) -> int:
    value = hashlib.sha256(f"{case}|{salt}|2022".encode()).digest()
    return int.from_bytes(value[:4], "little")


def exact_low_rank_mask(data: np.ndarray, seg: np.ndarray, percentile: int) -> np.ndarray:
    components, count = label(seg, STRUCTURE)
    result = np.zeros_like(seg, dtype=bool)
    for component in range(1, count + 1):
        coords = np.argwhere(components == component)
        values = data[components == component]
        count_hard = max(1, int(np.ceil(percentile / 100 * len(values))))
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], values))
        chosen = coords[order[:count_hard]]
        result[tuple(chosen.T)] = True
    return result


def summarize_candidate(case_rows: list[dict], percentile: int) -> dict:
    selected = [row for row in case_rows if row["percentile"] == percentile]
    return {
        "percentile": percentile,
        "lesion_positive_training_cases": len(selected),
        "mean_hard_fraction": float(np.mean([row["hard_fraction"] for row in selected])),
        "median_hard_fraction": float(np.median([row["hard_fraction"] for row in selected])),
        "p10_hard_voxels": float(np.percentile([row["hard_voxels"] for row in selected], 10)),
        "cases_with_at_least_20_hard_voxels": int(sum(row["hard_voxels"] >= 20 for row in selected)),
        "mean_standardized_easy_minus_hard": float(np.mean([row["standardized_easy_minus_hard"] for row in selected])),
        "all_cases_have_hard_voxels": all(row["hard_voxels"] > 0 for row in selected),
    }


def select_percentile(summaries: list[dict]) -> tuple[int, str]:
    for summary in summaries:
        if (
            summary["all_cases_have_hard_voxels"]
            and summary["median_hard_fraction"] >= 0.18
            and summary["p10_hard_voxels"] >= 10
            and summary["mean_standardized_easy_minus_hard"] >= 0.5
        ):
            return int(summary["percentile"]), (
                "smallest audited percentile with hard voxels in every lesion-positive training case, "
                "median hard fraction >=18%, 10th-percentile hard-voxel count >=10, "
                "and mean within-lesion standardized easy-minus-hard separation >=0.5"
            )
    raise RuntimeError("No simple training-only percentile definition met the prespecified adequacy rule")


def patch_metrics(seg: np.ndarray, hard: np.ndarray, bbox_lbs: list[int], bbox_ubs: list[int]) -> dict:
    bbox = [[int(a), int(b)] for a, b in zip(bbox_lbs, bbox_ubs)]
    seg_crop = crop_and_pad_nd(seg[None].astype(np.int16), bbox, -1)[0] > 0
    hard_crop = crop_and_pad_nd(hard[None].astype(np.uint8), bbox, 0)[0] > 0
    components, _ = label(seg, STRUCTURE)
    comp_crop = crop_and_pad_nd(components[None].astype(np.int16), bbox, 0)[0]
    covered_components = len(set(np.unique(comp_crop[seg_crop]).tolist()) - {0})
    patch_voxels = int(np.prod(seg_crop.shape))
    return {
        "lesion_containing_patch": bool(seg_crop.any()),
        "low_salience_containing_patch": bool(hard_crop.any()),
        "lesion_voxel_fraction": float(seg_crop.sum() / patch_voxels),
        "hard_region_voxel_fraction": float(hard_crop.sum() / patch_voxels),
        "connected_components_covered": covered_components,
        "total_lesion_coverage": float(seg_crop.sum() / max(seg.sum(), 1)),
        "low_salience_lesion_coverage": float(hard_crop.sum() / max(hard.sum(), 1)),
    }


def audit_scenario(
    name: str,
    preference: float,
    case_sequence: list[str],
    dataset: nnUNetDatasetBlosc2,
    loader: nnUNetDataLoader,
    hard_masks: dict[str, np.ndarray],
    hard_locations: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for index, case in enumerate(case_sequence):
        data, seg_arr, _, properties = dataset.load_case(case)
        shape = np.asarray(data.shape[1:], dtype=int)
        seg = np.asarray(seg_arr[0]) > 0
        hard = hard_masks[case]
        force_fg = (index % 8) >= round(8 * (1 - 0.33))
        hard_centered = False
        classes = dict(properties["class_locations"])
        classes[2] = hard_locations[case]
        np.random.seed(2022 + index)
        if force_fg:
            decision_rng = np.random.default_rng(9022 + index)
            hard_centered = bool(name != "B0" and decision_rng.random() < preference)
            overwrite = 2 if hard_centered else 1
            lbs, ubs = loader.get_bbox(shape, True, classes, overwrite_class=overwrite)
        else:
            lbs, ubs = loader.get_bbox(shape, False, classes)
        rows.append(
            {
                "scenario": name,
                "patch_index": index,
                "case": case,
                "forced_foreground": force_fg,
                "hard_centered": hard_centered,
                **patch_metrics(seg, hard, lbs, ubs),
            }
        )
    return rows


def summarize_sampling(rows: list[dict], name: str, preference: float) -> dict:
    forced = [row for row in rows if row["forced_foreground"]]
    return {
        "scenario": name,
        "hard_preference_within_forced_foreground": preference,
        "patches": len(rows),
        "forced_foreground_patches": len(forced),
        "lesion_containing_patch_rate": float(np.mean([row["lesion_containing_patch"] for row in rows])),
        "low_salience_containing_patch_rate": float(np.mean([row["low_salience_containing_patch"] for row in rows])),
        "forced_low_salience_containing_patch_rate": float(np.mean([row["low_salience_containing_patch"] for row in forced])),
        "mean_lesion_voxel_fraction": float(np.mean([row["lesion_voxel_fraction"] for row in rows])),
        "mean_hard_region_voxel_fraction": float(np.mean([row["hard_region_voxel_fraction"] for row in rows])),
        "forced_mean_total_lesion_coverage": float(np.mean([row["total_lesion_coverage"] for row in forced])),
        "forced_mean_low_salience_coverage": float(np.mean([row["low_salience_lesion_coverage"] for row in forced])),
        "forced_mean_components_covered": float(np.mean([row["connected_components_covered"] for row in forced])),
    }


def main() -> None:
    completion_path = S1_EXPERIMENT / "sampling_audit/completion.json"
    if completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"))
        return
    folder = V1_PP / "nnUNetPlans_3d_fullres"
    split = json.loads((V1_PP / "splits_final.json").read_text(encoding="utf-8"))[0]
    train_ids = sorted(split["train"])
    val_ids = set(split["val"])
    if len(train_ids) != 200 or len(val_ids) != 50:
        raise RuntimeError("Fold 0 split is not 200/50")

    S1_HARD.mkdir(parents=True, exist_ok=True)
    existing_hard_files = list(S1_HARD.glob("*.npz"))
    if existing_hard_files:
        raise RuntimeError(f"Partial hard-region files require review before rerun: {len(existing_hard_files)} files")
    case_rows: list[dict] = []
    empty_training_cases: list[str] = []
    for index, case in enumerate(train_ids, 1):
        data = np.asarray(blosc2.open(urlpath=str(folder / f"{case}.b2nd"), mode="r")[:], dtype=np.float32)[0]
        seg = np.asarray(blosc2.open(urlpath=str(folder / f"{case}_seg.b2nd"), mode="r")[:])[0] > 0
        if not np.isfinite(data).all():
            raise RuntimeError(f"Nonfinite training data for {case}")
        if not seg.any():
            empty_training_cases.append(case)
            continue
        for percentile in PERCENTILE_CANDIDATES:
            hard = exact_low_rank_mask(data, seg, percentile)
            easy = seg & ~hard
            pooled_sd = np.sqrt((data[hard].var() + data[easy].var()) / 2)
            case_rows.append(
                {
                    "case": case,
                    "percentile": percentile,
                    "lesion_voxels": int(seg.sum()),
                    "hard_voxels": int(hard.sum()),
                    "hard_fraction": float(hard.sum() / seg.sum()),
                    "hard_mean_zscore": float(data[hard].mean()),
                    "easy_mean_zscore": float(data[easy].mean()),
                    "standardized_easy_minus_hard": float((data[easy].mean() - data[hard].mean()) / max(pooled_sd, 1e-8)),
                }
            )
        if index % 25 == 0:
            print(f"Training-only salience audit {index}/200", flush=True)
    summaries = [summarize_candidate(case_rows, percentile) for percentile in PERCENTILE_CANDIDATES]
    selected_percentile, selection_rule = select_percentile(summaries)

    selected_masks: dict[str, np.ndarray] = {}
    hard_locations: dict[str, np.ndarray] = {}
    for case in train_ids:
        data = np.asarray(blosc2.open(urlpath=str(folder / f"{case}.b2nd"), mode="r")[:], dtype=np.float32)[0]
        seg = np.asarray(blosc2.open(urlpath=str(folder / f"{case}_seg.b2nd"), mode="r")[:])[0] > 0
        selected_masks[case] = exact_low_rank_mask(data, seg, selected_percentile) if seg.any() else np.zeros_like(seg, dtype=bool)
        coords = np.argwhere(selected_masks[case])
        rng = np.random.default_rng(case_seed(case, f"P{selected_percentile}"))
        if len(coords) > 10000:
            coords = coords[rng.choice(len(coords), 10000, replace=False)]
        locations = np.column_stack([np.zeros(len(coords), dtype=int), coords]).astype(np.int32)
        hard_locations[case] = locations
        np.savez_compressed(S1_HARD / f"{case}.npz", mask=selected_masks[case].astype(np.uint8), locations=locations)
    if any((S1_HARD / f"{case}.npz").exists() for case in val_ids):
        raise RuntimeError("Validation leakage: hard-region artifact exists for a validation case")

    dataset = nnUNetDatasetBlosc2(str(folder), identifiers=train_ids)
    plans_manager = PlansManager(V1_PP / "nnUNetPlans.json")
    dataset_json = json.loads((V1_PP / "dataset.json").read_text(encoding="utf-8"))
    label_manager = plans_manager.get_label_manager(dataset_json)
    loader = nnUNetDataLoader(
        dataset,
        8,
        INITIAL_PATCH_SIZE,
        FINAL_PATCH_SIZE,
        label_manager,
        oversample_foreground_percent=0.33,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=None,
        probabilistic_oversampling=False,
    )
    case_rng = np.random.default_rng(2022)
    case_sequence = [train_ids[i] for i in case_rng.integers(0, len(train_ids), size=AUDIT_PATCHES)]
    all_patch_rows: list[dict] = []
    all_summaries: list[dict] = []
    baseline_rows = audit_scenario("B0", 0.0, case_sequence, dataset, loader, selected_masks, hard_locations)
    all_patch_rows.extend(baseline_rows)
    baseline_summary = summarize_sampling(baseline_rows, "B0", 0.0)
    all_summaries.append(baseline_summary)
    passing = []
    for preference in PREFERENCE_CANDIDATES:
        name = f"V2-S1-{preference:.3f}"
        rows = audit_scenario(name, preference, case_sequence, dataset, loader, selected_masks, hard_locations)
        summary = summarize_sampling(rows, name, preference)
        summary["absolute_forced_hard_coverage_gain"] = summary["forced_mean_low_salience_coverage"] - baseline_summary["forced_mean_low_salience_coverage"]
        summary["absolute_forced_hard_presence_gain"] = summary["forced_low_salience_containing_patch_rate"] - baseline_summary["forced_low_salience_containing_patch_rate"]
        summary["absolute_lesion_patch_rate_change"] = summary["lesion_containing_patch_rate"] - baseline_summary["lesion_containing_patch_rate"]
        summary["effective"] = bool(
            summary["absolute_forced_hard_coverage_gain"] >= 0.10
            and summary["absolute_forced_hard_presence_gain"] >= 0.05
            and summary["absolute_lesion_patch_rate_change"] >= -0.01
        )
        if summary["effective"]:
            passing.append(summary)
        all_patch_rows.extend(rows)
        all_summaries.append(summary)

    audit_dir = S1_EXPERIMENT / "sampling_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "training_salience_candidates.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    with (audit_dir / "patches.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_patch_rows[0]))
        writer.writeheader()
        writer.writerows(all_patch_rows)
    with (audit_dir / "summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        fields = list(dict.fromkeys(key for row in all_summaries for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summaries)

    selected_sampling = min(passing, key=lambda row: row["hard_preference_within_forced_foreground"]) if passing else None
    status = "AUDIT PASSED" if selected_sampling else "SAMPLING INTERVENTION INEFFECTIVE"
    definition = {
        "data_scope": "Fold 0 training 200 only; validation IDs excluded",
        "method": "within each 26-connected lesion component, select the exact lowest intensity rank fraction on baseline nnUNet-preprocessed DWI",
        "percentile_candidates": PERCENTILE_CANDIDATES,
        "selected_percentile": selected_percentile,
        "selection_rule": selection_rule,
        "model_errors_used": False,
        "validation_data_used": False,
        "hard_location_cap_per_case": 10000,
    }
    save_json(audit_dir / "hard_region_definition.json", definition, exclusive=True)
    completion = {
        "status": status,
        "time": now(),
        "training_allowed": bool(selected_sampling),
        "training_cases": len(train_ids),
        "lesion_positive_training_cases": len(train_ids) - len(empty_training_cases),
        "empty_gt_training_cases": empty_training_cases,
        "validation_cases_used": 0,
        "audited_patches_per_scenario": AUDIT_PATCHES,
        "initial_patch_size": INITIAL_PATCH_SIZE.tolist(),
        "final_patch_size": FINAL_PATCH_SIZE.tolist(),
        "baseline_foreground_sampling": 0.33,
        "hard_region_definition": definition,
        "candidate_summaries": all_summaries,
        "selected_sampling": selected_sampling,
        "effectiveness_rule": "forced hard coverage gain >=0.10, forced hard-presence gain >=0.05, lesion-patch rate change >=-0.01",
        "git_commit": git_commit(),
    }
    save_json(completion_path, completion, exclusive=True)
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
