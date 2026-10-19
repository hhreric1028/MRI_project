"""Add fixed 26-connectivity FP topology metrics to one completed fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
DEFAULT_ANALYSIS = ROOT / "outputs/analysis/kamel_replication"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        path.write_text("case\n", encoding="utf-8-sig")
        return
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matched_prediction_ids(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, set[int]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_labels, gt_n = label(gt, structure)
    pred_labels, pred_n = label(pred, structure)
    gt_sizes = np.bincount(gt_labels.ravel())[1:]
    pred_sizes = np.bincount(pred_labels.ravel())[1:]
    matched: set[int] = set()
    if gt_n and pred_n:
        overlap = np.zeros((gt_n, pred_n), dtype=np.float64)
        both = (gt_labels > 0) & (pred_labels > 0)
        np.add.at(overlap, (gt_labels[both] - 1, pred_labels[both] - 1), 1)
        iou = overlap / (gt_sizes[:, None] + pred_sizes[None, :] - overlap)
        score = (overlap > 0) * (min(gt_n, pred_n) + 1) + iou
        gt_ids, pred_ids = linear_sum_assignment(score, maximize=True)
        matched = {int(p + 1) for g, p in zip(gt_ids, pred_ids) if overlap[g, p] > 0}
    return gt_labels, pred_labels, matched


def case_topology(gt: np.ndarray, pred: np.ndarray, voxel_ml: float) -> dict[str, Any]:
    gt_labels, pred_labels, matched = matched_prediction_ids(gt, pred)
    pred_sizes = np.bincount(pred_labels.ravel())[1:]
    fp_volumes = [float(size * voxel_ml) for index, size in enumerate(pred_sizes, start=1) if index not in matched]
    ge1 = [volume for volume in fp_volumes if volume >= 1.0]
    return {
        "predicted_component_count": int(len(pred_sizes)),
        "total_predicted_foreground_volume_ml": float(pred.sum() * voxel_ml),
        "total_fp_components": int(len(fp_volumes)),
        "fp_total_volume_ml": float(sum(fp_volumes)),
        "largest_fp_volume_ml": float(max(fp_volumes, default=0.0)),
        "fp_components_lt_0p01_ml": int(sum(volume < 0.01 for volume in fp_volumes)),
        "fp_components_lt_0p05_ml": int(sum(volume < 0.05 for volume in fp_volumes)),
        "fp_components_lt_0p1_ml": int(sum(volume < 0.1 for volume in fp_volumes)),
        "fp_components_ge_1_ml": int(len(ge1)),
        "fp_volume_ge_1_ml": float(sum(ge1)),
        "fp_component_volumes_ml": fp_volumes,
    }


def safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def safe_median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.median(clean)) if clean else None


def aggregate_subgroup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["lesion_tp"]) for row in rows)
    fn = sum(int(row["lesion_fn"]) for row in rows)
    fp = sum(int(row["lesion_fp"]) for row in rows)
    return {
        "n": len(rows),
        "mean_dice": safe_mean([row["dice"] for row in rows]),
        "median_dice": safe_median([row["dice"] for row in rows]),
        "lesion_recall": tp / (tp + fn) if tp + fn else None,
        "lesion_precision": tp / (tp + fp) if tp + fp else None,
        "mean_pred_gt": safe_mean([row["pred_gt_ratio"] for row in rows]),
        "median_pred_gt": safe_median([row["pred_gt_ratio"] for row in rows]),
    }


def build_fold_outputs(
    fold: int, metrics_dir: Path, predictions: Path, analysis: Path = DEFAULT_ANALYSIS
) -> dict[str, Any]:
    output = analysis / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((metrics_dir / "cases.json").read_text(encoding="utf-8"))
    expected = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))[fold]["val"]
    if [row["case"] for row in rows] != expected:
        raise RuntimeError(f"Fold {fold} case order differs from the fixed split")
    topology_rows: list[dict[str, Any]] = []
    fp_volumes_all: list[float] = []
    for row in rows:
        case = row["case"]
        gt_img = nib.load(PP / "gt_segmentations" / f"{case}.nii.gz")
        pred_img = nib.load(predictions / f"{case}.nii.gz")
        if gt_img.shape != pred_img.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3, rtol=0):
            raise RuntimeError(f"{case} geometry mismatch")
        gt_raw = gt_img.get_fdata(dtype=np.float32)
        pred_raw = pred_img.get_fdata(dtype=np.float32)
        if not np.isfinite(gt_raw).all() or not np.isfinite(pred_raw).all():
            raise RuntimeError(f"{case} nonfinite mask")
        gt = np.round(gt_raw).astype(np.uint8) > 0
        pred = pred_raw > 0.5
        voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3], dtype=np.float64) / 1000)
        topology = case_topology(gt, pred, voxel_ml)
        fp_volumes_all.extend(topology.pop("fp_component_volumes_ml"))
        topology_rows.append({"case": case, "fold": fold, **topology})

    total_fp = sum(row["total_fp_components"] for row in topology_rows)
    summary = {
        "fold": fold,
        "cases": len(rows),
        "connectivity": 26,
        "matching": "maximum-cardinality then maximum-IoU one-to-one overlap",
        "total_fp_components": int(total_fp),
        "fp_components_per_case": float(total_fp / len(rows)),
        "fp_volume_per_case_ml": float(sum(row["fp_total_volume_ml"] for row in topology_rows) / len(rows)),
        "median_fp_component_volume_ml": float(np.median(fp_volumes_all)) if fp_volumes_all else 0.0,
        "fp_components_lt_0p01_ml": int(sum(row["fp_components_lt_0p01_ml"] for row in topology_rows)),
        "fp_components_lt_0p05_ml": int(sum(row["fp_components_lt_0p05_ml"] for row in topology_rows)),
        "fp_components_lt_0p1_ml": int(sum(row["fp_components_lt_0p1_ml"] for row in topology_rows)),
        "fp_components_ge_1_ml": int(sum(row["fp_components_ge_1_ml"] for row in topology_rows)),
        "predicted_component_count": int(sum(row["predicted_component_count"] for row in topology_rows)),
        "total_predicted_foreground_volume_ml": float(
            sum(row["total_predicted_foreground_volume_ml"] for row in topology_rows)
        ),
    }
    (output / "fp_topology_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(output / "fp_topology_cases.csv", topology_rows)

    by_case = {row["case"]: row for row in topology_rows}
    enriched = [{**row, **by_case[row["case"]]} for row in rows]
    write_csv(
        output / "top5_highest_fp_lesion_count.csv",
        sorted(enriched, key=lambda row: (-row["lesion_fp"], row["case"]))[:5],
    )
    write_csv(
        output / "top5_largest_ge1ml_fp_volume.csv",
        sorted(enriched, key=lambda row: (-row["fp_volume_ge_1_ml"], row["case"]))[:5],
    )

    severe = []
    for row in enriched:
        if row["gt_ml"] <= 0 or row["pred_gt_ratio"] >= 0.5:
            continue
        gt_components = [float(value) for value in row["gt_component_ml"]]
        pred_img = nib.load(predictions / f"{row['case']}.nii.gz")
        gt_img = nib.load(PP / "gt_segmentations" / f"{row['case']}.nii.gz")
        gt = np.round(gt_img.get_fdata(dtype=np.float32)).astype(np.uint8) > 0
        pred = pred_img.get_fdata(dtype=np.float32) > 0.5
        gt_labels, gt_n = label(gt, np.ones((3, 3, 3), dtype=np.uint8))
        largest_coverage = 0.0
        if gt_n:
            sizes = np.bincount(gt_labels.ravel())[1:]
            largest_id = int(np.argmax(sizes) + 1)
            largest_coverage = float(pred[gt_labels == largest_id].mean())
        severe.append(
            {
                "case": row["case"],
                "fold": fold,
                "gt_volume_ml": row["gt_ml"],
                "pred_volume_ml": row["pred_ml"],
                "pred_gt_ratio": row["pred_gt_ratio"],
                "dice": row["dice"],
                "voxel_recall": row["voxel_recall"],
                "gt_lesion_count": len(gt_components),
                "pred_lesion_count": len(row["pred_component_ml"]),
                "largest_gt_component_volume_ml": max(gt_components, default=0.0),
                "largest_component_coverage": largest_coverage,
                "large_partial_coverage_failure": bool(row["gt_ml"] > 20 and row["pred_gt_ratio"] < 0.5),
            }
        )
    write_csv(output / "severe_under_cases_extended.csv", severe)
    write_csv(
        output / "large_partial_coverage_failures.csv",
        [row for row in severe if row["large_partial_coverage_failure"]],
    )

    subgroup_rows = []
    for name, low, high in [("<1", 0, 1), ("1-5", 1, 5), ("5-20", 5, 20), (">20", 20, float("inf"))]:
        group = [row for row in rows if low <= row["gt_ml"] < high]
        subgroup_rows.append({"fold": fold, "subgroup": name, **aggregate_subgroup(group)})
    write_csv(output / "subgroup_metrics_extended.csv", subgroup_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    summary = build_fold_outputs(args.fold, args.metrics, args.predictions, args.analysis)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
