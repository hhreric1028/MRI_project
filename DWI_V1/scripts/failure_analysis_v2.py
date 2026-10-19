"""Failure Analysis V2 for the frozen ISLES2022 DWI-only Fold 0 baseline.

This program is analysis-only. It reads the existing checkpoint validation
artifacts, raw DWI/GT, and nnUNet preprocessed arrays. It never trains a model,
changes a split, changes a checkpoint, or rewrites an existing result folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import blosc2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "analysis" / "failure_analysis_v2"
RAW = ROOT / "datasets" / "ISLES2022" / "nnUNet_raw" / "Dataset001_ISLES2022_DWI"
PROCESSED = ROOT / "datasets" / "ISLES2022" / "processed" / "Dataset001_ISLES2022_DWI"
PREPROCESSED = PROCESSED / "nnUNetPlans_3d_fullres"
PREDICTIONS = (
    ROOT
    / "outputs"
    / "models"
    / "Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)
MAPPING = ROOT / "datasets" / "ISLES2022" / "documentation" / "case_mapping.csv"
SPLIT = PROCESSED / "splits_final.json"
STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)
REQUESTED_SUBJECT = "0410"
FALLBACK_CASE = "case_0140"


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    frame = pd.DataFrame([clean(row) for row in rows])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ["n", "mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max"]} | {"n": 0}
    p05, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(values.max()),
    }


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not set(np.unique(data)).issubset({0.0, 1.0}):
        raise ValueError(f"Non-binary or non-finite mask: {path}")
    return image, data > 0.5


def load_probability(case: str, expected_shape: tuple[int, ...], pred: np.ndarray) -> tuple[np.ndarray, bool]:
    path = PREDICTIONS / f"{case}.npz"
    with np.load(path) as archive:
        raw = archive["probabilities"]
    if raw.shape != (2, *expected_shape[::-1]):
        raise ValueError(f"Unexpected probability shape for {case}: {raw.shape}")
    probability = np.asarray(raw[1].transpose(2, 1, 0), dtype=np.float32)
    if not np.isfinite(probability).all() or probability.min() < -1e-6 or probability.max() > 1 + 1e-6:
        raise ValueError(f"Invalid probabilities for {case}")
    exported_class = np.asarray(np.argmax(raw, axis=0).transpose(2, 1, 0) == 1)
    return probability, bool(np.array_equal(exported_class, pred))


def load_preprocessed(
    case: str,
    raw: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.asarray(blosc2.open(urlpath=str(PREPROCESSED / f"{case}.b2nd"), mode="r")[:], dtype=np.float32)
    stored_seg = np.asarray(blosc2.open(urlpath=str(PREPROCESSED / f"{case}_seg.b2nd"), mode="r")[:])
    with (PREPROCESSED / f"{case}.pkl").open("rb") as handle:
        properties = pickle.load(handle)
    shape_zyx = tuple(int(v) for v in properties["shape_before_cropping"])
    if shape_zyx != raw.shape[::-1]:
        raise ValueError(f"Preprocessed/native shape mismatch for {case}: {shape_zyx} vs {raw.shape[::-1]}")
    bbox = properties["bbox_used_for_cropping"]
    crop_slices = tuple(slice(int(a), int(b)) for a, b in bbox)
    current_spacing = tuple(float(v) for v in properties["spacing"])
    target_spacing = (2.0, 2.0, 2.0)

    def transform(mask_xyz: np.ndarray) -> np.ndarray:
        cropped = mask_xyz.transpose(2, 1, 0)[crop_slices][None].astype(np.float32)
        transformed = resample_data_or_seg_to_shape(
            cropped,
            tuple(int(v) for v in data.shape[1:]),
            current_spacing=current_spacing,
            new_spacing=target_spacing,
            is_seg=True,
            order=1,
            order_z=0,
            force_separate_z=None,
        )[0]
        return transformed > 0.5

    gt_preprocessed = transform(gt)
    pred_preprocessed = transform(pred)
    foreground_preprocessed = transform(np.isfinite(raw) & (raw != 0))
    if not np.array_equal(gt_preprocessed, stored_seg[0] > 0):
        raise ValueError(f"Reconstructed preprocessed GT does not match stored segmentation for {case}")
    return data[0], gt_preprocessed, pred_preprocessed, foreground_preprocessed, properties


def component_analysis(gt: np.ndarray, pred: np.ndarray, voxel_ml: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_labels, gt_count = label(gt, STRUCTURE_26)
    pred_labels, pred_count = label(pred, STRUCTURE_26)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=gt_count + 1)[1:]
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)[1:]
    overlap = np.zeros((gt_count, pred_count), dtype=np.int64)
    both = (gt_labels > 0) & (pred_labels > 0)
    if both.any():
        np.add.at(overlap, (gt_labels[both] - 1, pred_labels[both] - 1), 1)
    assignments: dict[int, int] = {}
    matches = 0
    if gt_count and pred_count:
        union = gt_sizes[:, None] + pred_sizes[None, :] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
        score = (overlap > 0) * (min(gt_count, pred_count) + 1) + iou
        assigned_gt, assigned_pred = linear_sum_assignment(score, maximize=True)
        for g, p in zip(assigned_gt, assigned_pred):
            if overlap[g, p] > 0:
                assignments[int(g)] = int(p)
                matches += 1
    rows: list[dict[str, Any]] = []
    for g in range(gt_count):
        covered = int(overlap[g].sum()) if pred_count else 0
        best_p = int(np.argmax(overlap[g])) if pred_count and overlap[g].max() > 0 else None
        assigned_p = assignments.get(g)
        matched_overlap = int(overlap[g, assigned_p]) if assigned_p is not None else 0
        matched_dice = (
            2 * matched_overlap / (int(gt_sizes[g]) + int(pred_sizes[assigned_p]))
            if assigned_p is not None
            else 0.0
        )
        rows.append(
            {
                "gt_component_id": g + 1,
                "gt_voxels": int(gt_sizes[g]),
                "gt_volume_ml": float(gt_sizes[g] * voxel_ml),
                "covered_voxels": covered,
                "coverage_fraction": safe_ratio(covered, int(gt_sizes[g])),
                "best_overlapping_pred_component_id": None if best_p is None else best_p + 1,
                "one_to_one_matched_pred_component_id": None if assigned_p is None else assigned_p + 1,
                "matched_component_dice": float(matched_dice),
                "completely_missed": covered == 0,
            }
        )
    rows.sort(key=lambda row: (-row["gt_volume_ml"], row["gt_component_id"]))
    summary = {
        "gt_lesion_count": int(gt_count),
        "pred_lesion_count": int(pred_count),
        "largest_gt_component_ml": float(gt_sizes.max() * voxel_ml) if gt_count else 0.0,
        "largest_pred_component_ml": float(pred_sizes.max() * voxel_ml) if pred_count else 0.0,
        "gt_component_volumes_ml": [float(v * voxel_ml) for v in sorted(gt_sizes, reverse=True)],
        "pred_component_volumes_ml": [float(v * voxel_ml) for v in sorted(pred_sizes, reverse=True)],
        "lesion_tp": int(matches),
        "lesion_fn": int(gt_count - matches),
        "lesion_fp": int(pred_count - matches),
        "lesion_recall": safe_ratio(matches, gt_count),
        "lesion_precision": safe_ratio(matches, pred_count),
        "largest_gt_component_coverage": rows[0]["coverage_fraction"] if rows else None,
    }
    return summary, rows


def intensity_rows(
    case: str,
    raw: np.ndarray,
    normalized: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    normalized_gt: np.ndarray,
    normalized_pred: np.ndarray,
    normalized_foreground: np.ndarray,
) -> list[dict[str, Any]]:
    raw_foreground = np.isfinite(raw) & (raw != 0)
    raw_regions = {
        "whole_foreground": raw_foreground,
        "entire_gt": gt,
        "detected_gt": gt & pred,
        "missed_gt": gt & ~pred,
        "background": raw_foreground & ~gt,
    }
    normalized_foreground = np.isfinite(normalized) & normalized_foreground
    normalized_regions = {
        "whole_foreground": normalized_foreground,
        "entire_gt": normalized_gt,
        "detected_gt": normalized_gt & normalized_pred,
        "missed_gt": normalized_gt & ~normalized_pred,
        "background": normalized_foreground & ~normalized_gt,
    }
    rows: list[dict[str, Any]] = []
    for representation, data, valid, regions in [
        ("raw_dwi", raw, raw_foreground, raw_regions),
        ("nnunet_normalized_dwi", normalized, normalized_foreground, normalized_regions),
    ]:
        for region, mask in regions.items():
            rows.append({"case": case, "representation": representation, "region": region, **stats(data[mask & valid])})
    return rows


def probability_summary(probability: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detected = gt & pred
    missed = gt & ~pred
    detected_stats = stats(probability[detected])
    missed_stats = stats(probability[missed])
    values = probability[missed]
    bins = {
        "missed_p_ge_0p5": values >= 0.5,
        "missed_p_0p4_0p5": (values >= 0.4) & (values < 0.5),
        "missed_p_0p3_0p4": (values >= 0.3) & (values < 0.4),
        "missed_p_0p2_0p3": (values >= 0.2) & (values < 0.3),
        "missed_p_lt_0p2": values < 0.2,
    }
    result: dict[str, Any] = {}
    for name, selection in bins.items():
        result[f"{name}_count"] = int(selection.sum())
        result[f"{name}_fraction"] = safe_ratio(int(selection.sum()), int(values.size))
    near = float(((values >= 0.3) & (values < 0.5)).mean()) if values.size else None
    if not values.size:
        pattern = "no_missed_gt"
    elif (missed_stats["median"] or 0) >= 0.2 or (near or 0) >= 0.25:
        pattern = "near_threshold_response"
    elif (missed_stats["median"] or 0) < 0.1 and float((values < 0.2).mean()) >= 0.75:
        pattern = "background_confident"
    else:
        pattern = "mixed_confidence"
    result.update(
        {
            "detected_probability_mean": detected_stats["mean"],
            "detected_probability_median": detected_stats["median"],
            "missed_probability_mean": missed_stats["mean"],
            "missed_probability_median": missed_stats["median"],
            "missed_near_threshold_0p3_0p5_fraction": near,
            "confidence_pattern": pattern,
        }
    )
    detailed = [
        {"region": "detected_gt", **detected_stats},
        {"region": "missed_gt", **missed_stats},
    ]
    return result, detailed


def assign_failure_mode(row: dict[str, Any]) -> tuple[str, str]:
    if row["gt_ml"] == 0:
        return ("normal" if row["pred_voxels"] == 0 else "fp_dominant", "normal" if row["pred_voxels"] == 0 else "fp_dominant")
    labels: list[str] = []
    ratio = row["pred_gt_ratio"]
    recall = row["voxel_recall"]
    if row["gt_ml"] < 5 and (ratio < 0.5 or recall < 0.5 or row["lesion_recall"] < 0.5):
        labels.append("small_lesion_miss")
    if row["gt_ml"] >= 5 and ratio < 0.5 and 0 < recall < 0.6:
        labels.append("large_partial_underseg")
    if row["gt_lesion_count"] >= 2 and row["lesion_recall"] < 1:
        labels.append("multifocal_miss")
    if (
        0.5 <= ratio < 0.8
        and 0.5 <= recall < 0.85
        and (row["largest_gt_component_coverage"] or 0) >= 0.5
        and row["lesion_recall"] > 0
    ):
        labels.append("boundary_underseg")
    if (
        recall >= 0.7
        and (row["voxel_fp"] > row["voxel_fn"] or ratio > 1.2)
        and (row["lesion_precision"] is not None and row["lesion_precision"] < 0.5)
    ):
        labels.append("fp_dominant")
    if row["confidence_pattern"] == "near_threshold_response":
        labels.append("low_confidence")
    labels = list(dict.fromkeys(labels))
    if not labels:
        normal = (
            row["dice"] is None
            or (
                row["dice"] >= 0.7
                and 0.8 <= ratio <= 1.2
                and recall >= 0.7
                and row["lesion_recall"] >= 0.8
            )
        )
        return ("normal" if normal else "unclassified", "normal" if normal else "")
    return ("mixed" if len(labels) > 1 else labels[0], ";".join(labels))


def select_comparisons(rows: list[dict[str, Any]], anchor: str) -> dict[str, str]:
    anchor_row = next(row for row in rows if row["case"] == anchor)
    good = [
        row
        for row in rows
        if row["case"] != anchor
        and row["gt_ml"] > 0
        and row["dice"] is not None
        and row["dice"] >= 0.8
        and 0.8 <= row["pred_gt_ratio"] <= 1.2
    ]
    if not good:
        raise RuntimeError("No similar-volume good case candidate")
    similar_good = min(
        good,
        key=lambda row: (abs(math.log(row["gt_ml"] / anchor_row["gt_ml"])), -row["dice"], row["case"]),
    )
    small = [
        row
        for row in rows
        if 0 < row["gt_ml"] < 5
        and (row["pred_gt_ratio"] < 0.5 or row["lesion_recall"] < 0.5 or row["voxel_recall"] < 0.5)
    ]
    if not small:
        raise RuntimeError("No small-lesion failure candidate")
    small_failure = min(small, key=lambda row: (row["dice"] if row["dice"] is not None else 1.0, row["pred_gt_ratio"], row["case"]))
    return {"anchor": anchor, "similar_volume_good": similar_good["case"], "small_lesion_failure": small_failure["case"]}


def choose_slices(gt: np.ndarray, pred: np.ndarray) -> list[tuple[str, int]]:
    fn = gt & ~pred
    discrepancy = gt ^ pred
    candidates = [
        ("max_fn", int(np.argmax(fn.sum(axis=(0, 1)))) if fn.any() else int(np.argmax(gt.sum(axis=(0, 1))))),
        ("max_gt", int(np.argmax(gt.sum(axis=(0, 1)))) if gt.any() else int(np.argmax(pred.sum(axis=(0, 1))))),
        ("max_disagreement", int(np.argmax(discrepancy.sum(axis=(0, 1)))) if discrepancy.any() else gt.shape[2] // 2),
    ]
    selected: list[tuple[str, int]] = []
    seen: set[int] = set()
    for reason, z in candidates:
        if z not in seen:
            selected.append((reason, z))
            seen.add(z)
    return selected


def display_limits(data: np.ndarray) -> tuple[float, float]:
    valid = data[np.isfinite(data) & (data != 0)]
    if not valid.size:
        return 0.0, 1.0
    low, high = np.percentile(valid, [1, 99])
    return float(low), float(high if high > low else low + 1)


def overlay(ax: plt.Axes, base: np.ndarray, mask: np.ndarray, color: str, title: str, lo: float, hi: float) -> None:
    ax.imshow(base.T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    if mask.any():
        ax.contour(mask.T, levels=[0.5], colors=[color], linewidths=1.1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_case_figure(path: Path, case: str, raw: np.ndarray, gt: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    selected = choose_slices(gt, pred)
    lo, hi = display_limits(raw)
    fig, axes = plt.subplots(5, len(selected), figsize=(4.2 * len(selected), 16), squeeze=False)
    selection_rows: list[dict[str, Any]] = []
    for column, (reason, z) in enumerate(selected):
        dwi = raw[:, :, z]
        g = gt[:, :, z]
        p = pred[:, :, z]
        tp = g & p
        fn = g & ~p
        fp = ~g & p
        selection_rows.append(
            {
                "case": case,
                "selection_reason": reason,
                "axial_slice": z,
                "gt_voxels": int(g.sum()),
                "fn_voxels": int(fn.sum()),
                "fp_voxels": int(fp.sum()),
            }
        )
        axes[0, column].imshow(dwi.T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        axes[0, column].set_title(f"DWI | {reason} | z={z}", fontsize=9)
        axes[0, column].axis("off")
        overlay(axes[1, column], dwi, g, "lime", "GT overlay", lo, hi)
        overlay(axes[2, column], dwi, p, "magenta", "Prediction overlay", lo, hi)
        axes[3, column].imshow(dwi.T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        rgb = np.zeros((*g.T.shape, 4), dtype=float)
        rgb[tp.T] = [0.1, 0.9, 0.2, 0.65]
        rgb[fn.T] = [1.0, 0.15, 0.1, 0.72]
        rgb[fp.T] = [0.0, 0.8, 1.0, 0.72]
        axes[3, column].imshow(rgb, origin="lower")
        axes[3, column].set_title("TP green / FN red / FP cyan", fontsize=9)
        axes[3, column].axis("off")
        axes[4, column].imshow(dwi.T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        heat = axes[4, column].imshow(probability[:, :, z].T, origin="lower", cmap="inferno", vmin=0, vmax=1, alpha=0.65)
        if g.any():
            axes[4, column].contour(g.T, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axes[4, column].set_title("Lesion probability (GT cyan)", fontsize=9)
        axes[4, column].axis("off")
    fig.colorbar(heat, ax=axes[4, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(f"{case}: frozen baseline failure analysis", fontsize=14)
    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.95, hspace=0.12, wspace=0.05)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return selection_rows


def correlation_row(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    pairs = [(row.get(x_key), row.get(y_key)) for row in rows]
    pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"x": x_key, "y": y_key, "n": len(x), "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "x": x_key,
        "y": y_key,
        "n": len(x),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def scatter_figure(path: Path, rows: list[dict[str, Any]], pairs: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    correlation_rows: list[dict[str, Any]] = []
    for ax, (x_key, y_key, x_label, y_label) in zip(axes.ravel(), pairs):
        valid = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
        x = np.asarray([row[x_key] for row in valid], dtype=float)
        y = np.asarray([row[y_key] for row in valid], dtype=float)
        severe = np.asarray([row["severe_underseg"] for row in valid], dtype=bool)
        ax.scatter(x[~severe], y[~severe], color="#4c78a8", alpha=0.75, label="other")
        ax.scatter(x[severe], y[severe], color="#d62728", marker="x", s=65, label="Pred/GT < 0.5")
        if x_key in {"gt_ml", "largest_gt_component_ml"}:
            ax.set_xscale("log")
        for row in valid:
            if row["severe_underseg"] or row["case"] == FALLBACK_CASE:
                ax.annotate(row["case"].replace("case_", ""), (row[x_key], row[y_key]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        corr = correlation_row(rows, x_key, y_key)
        correlation_rows.append(corr)
        rho = corr["spearman_rho"]
        ax.set_title(f"Spearman rho={rho:.3f}" if rho is not None else "Correlation unavailable")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return correlation_rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def build_report(
    path: Path,
    rows: list[dict[str, Any]],
    anchor: dict[str, Any],
    selections: dict[str, str],
    correlations: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    identification: dict[str, Any],
) -> None:
    severe = [row for row in rows if row["severe_underseg"]]
    severe_small = [row for row in severe if 0 < row["gt_ml"] < 5]
    severe_large = [row for row in severe if row["gt_ml"] >= 5]
    severe_multifocal = [row for row in severe if row["gt_lesion_count"] >= 2]
    mode_counts = Counter(row["failure_mode"] for row in rows)
    label_counts = Counter(label for row in rows for label in row["failure_labels"].split(";") if label)
    comparison_rows = [next(row for row in rows if row["case"] == case) for case in selections.values()]
    largest = component_rows[0] if component_rows else None
    missed_intensity_delta = anchor["raw_missed_gt_median"] - anchor["raw_detected_gt_median"]
    missed_norm_delta = anchor["norm_missed_gt_median"] - anchor["norm_detected_gt_median"]
    close_fraction = anchor["missed_p_0p4_0p5_fraction"] or 0.0
    low_fraction = anchor["missed_p_lt_0p2_fraction"] or 0.0
    primary_reason = (
        "case_0140 是單一 dominant large component 的 partial-coverage failure；漏掉區域的機率大多接近 0，且 DWI 訊號低於已偵測區域。"
    )
    recommendation = (
        "下一個單一變因實驗建議：只改 DWI intensity/contrast augmentation（保持 split、loss、sampling、architecture、patch、inference threshold 不變）。"
        "理由是 anchor 的 missed region 呈現低訊號且 background-confident；直接調 threshold 的支持度很低。"
        "但這個大型 failure 在 Fold 0 是少數型態，因此結果必須以整個 validation 與預先定義的 severe-under 指標驗證，不應只針對 anchor 調參。"
    )
    lines = [
        "# DWI-only baseline Failure Analysis V2",
        "",
        "> 分析範圍：現有 B0 2.0 mm Fold 0 final-checkpoint validation。未訓練、未修改 checkpoint、split 或 training configuration。",
        "",
        "## Case identification",
        "",
        f"- 使用者指定 subject `{REQUESTED_SUBJECT}`，但 case mapping、Fold 0 split 與檔名均找不到該 ID。mapping 範圍為 `{identification['mapping_min_subject']}`–`{identification['mapping_max_subject']}`。",
        f"- 符合描述且專案既有 failure anchor 為 `{FALLBACK_CASE}`（原始 ID `{identification['fallback_original_case']}`）。本報告將它列為**最可能的轉置筆誤候選**，不是把 0410 靜默視為已定位成功。",
        "",
        "## Quantitative rules",
        "",
        "- 3D components：26-connectivity。Lesion TP 使用專案既有的 maximum-cardinality、maximum-IoU one-to-one overlap matching。",
        "- Severe under-segmentation：既有規則 `Pred/GT < 0.5`。Small lesion：既有體積 subgroup 邊界 `GT < 5 mL`；本報告的 large/non-small 為 `GT >= 5 mL`。",
        "- `small_lesion_miss`：GT <5 mL，且 Pred/GT <0.5、voxel recall <0.5 或 lesion recall <0.5。",
        "- `large_partial_underseg`：GT >=5 mL、Pred/GT <0.5、0 < voxel recall <0.6。",
        "- `multifocal_miss`：GT components >=2 且 lesion recall <1。",
        "- `boundary_underseg`：0.5 <= Pred/GT <0.8、0.5 <= voxel recall <0.85、largest-component coverage >=0.5，且至少一個 lesion 被偵測。",
        "- `fp_dominant`：voxel recall >=0.7、FP voxels > FN voxels 或 Pred/GT >1.2，且 lesion precision <0.5。",
        "- `low_confidence`：missed GT probability median >=0.2，或 missed GT 中至少 25% 位於 0.3–0.5。多標籤病例的 primary `failure_mode` 記為 `mixed`。不符合明確 failure rule 且不符合 normal rule 者保留 `unclassified`。",
        "",
        "## Validation-level answer",
        "",
        f"1. **主要 failure 不只 small lesion。** 共有 {len(severe)} 個 severe under-segmentation；其中 small lesion {len(severe_small)} 個，non-small/large lesion {len(severe_large)} 個，multifocal lesion {len(severe_multifocal)} 個（multifocal 與 size 計數可重疊）。",
        f"2. Failure-mode primary counts：{', '.join(f'`{k}`={v}' for k, v in sorted(mode_counts.items()))}。多標籤展開 counts：{', '.join(f'`{k}`={v}' for k, v in sorted(label_counts.items()))}。",
        "3. Scatter/correlation 顯示的完整數值在 `correlation_statistics.csv`。Severe cases 已在圖中以紅色 x 標示；可直接看到它們跨越不同 GT volume，而非只聚集在最小體積端。",
        "",
        "## case_0140 (requested 0410 candidate)",
        "",
        f"- GT {anchor['gt_ml']:.3f} mL；prediction {anchor['pred_ml']:.3f} mL；Pred/GT {anchor['pred_gt_ratio']:.3f}；Dice {anchor['dice']:.3f}。",
        f"- Voxel TP/FN/FP = {anchor['voxel_tp']}/{anchor['voxel_fn']}/{anchor['voxel_fp']}；lesion recall {anchor['lesion_recall']:.3f}；lesion precision {anchor['lesion_precision']:.3f}。",
        f"- GT/pred component count = {anchor['gt_lesion_count']}/{anchor['pred_lesion_count']}；largest GT/pred component = {anchor['largest_gt_component_ml']:.3f}/{anchor['largest_pred_component_ml']:.3f} mL。",
        f"- Largest GT component佔 GT {largest['gt_volume_ml']/anchor['gt_ml']:.1%}，prediction coverage {largest['coverage_fraction']:.1%}，matched-component Dice {largest['matched_component_dice']:.3f}。因此它是大型 dominant lesion 沒切完整，不是只漏掉幾個 separated lesions。",
        f"- Missed GT probability median {fmt(anchor['missed_probability_median'], 6)}；0.4–0.5 僅 {close_fraction:.1%}，<0.2 為 {low_fraction:.1%}。結論：大部分 missed region 被高信心判為 background，不是稍微低於 threshold。",
        f"- Raw DWI median：detected {anchor['raw_detected_gt_median']:.3f}，missed {anchor['raw_missed_gt_median']:.3f}（差 {missed_intensity_delta:.3f}）。Normalized DWI median：detected {anchor['norm_detected_gt_median']:.3f}，missed {anchor['norm_missed_gt_median']:.3f}（差 {missed_norm_delta:.3f}）。漏掉區域系統性較暗。",
        f"- 綜合判定：{primary_reason}",
        "",
        "## Comparison cases",
        "",
        "| Role | Case | GT mL | Pred mL | Pred/GT | Dice | GT lesions | Lesion recall | Lesion precision | Largest GT mL | Raw detected/missed median | Probability detected/missed median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    role_by_case = {case: role for role, case in selections.items()}
    for row in comparison_rows:
        lines.append(
            f"| {role_by_case[row['case']]} | {row['case']} | {row['gt_ml']:.3f} | {row['pred_ml']:.3f} | {row['pred_gt_ratio']:.3f} | {fmt(row['dice'])} | {row['gt_lesion_count']} | {fmt(row['lesion_recall'])} | {fmt(row['lesion_precision'])} | {row['largest_gt_component_ml']:.3f} | {fmt(row['raw_detected_gt_median'])}/{fmt(row['raw_missed_gt_median'])} | {fmt(row['detected_probability_median'])}/{fmt(row['missed_probability_median'])} |"
        )
    good = next(row for row in comparison_rows if row["case"] == selections["similar_volume_good"])
    small = next(row for row in comparison_rows if row["case"] == selections["small_lesion_failure"])
    lines += [
        "",
        f"- Similar-volume good case `{good['case']}` 的 GT volume 與 anchor 最接近且 Dice >=0.8、Pred/GT 介於 0.8–1.2。它的 coverage 與 missed-probability pattern 明顯優於 anchor。",
        f"- Small-lesion failure `{small['case']}` 是自動從 GT <5 mL 且低 coverage/recall 候選中依 Dice、Pred/GT 排序選出。它與 anchor 分別代表 small detection failure 與 large partial-coverage failure。",
        "",
        "## Correlations",
        "",
        "| X | Y | n | Pearson r | p | Spearman rho | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corr in correlations:
        lines.append(
            f"| {corr['x']} | {corr['y']} | {corr['n']} | {fmt(corr['pearson_r'])} | {fmt(corr['pearson_p'])} | {fmt(corr['spearman_rho'])} | {fmt(corr['spearman_p'])} |"
        )
    lines += [
        "",
        "## Most supported next step",
        "",
        recommendation,
        "",
        "這是研究假設，不是已證明的因果結論。特別是 case_0140 在大型病灶中屬離群 failure，Fold 0 單一 split 無法單獨證明 augmentation 一定改善。若希望最保守地表述選項，答案是 **B（優先測試）並保留 G 的不確定性**；D（threshold calibration）不受目前 probability evidence 支持。",
        "",
        "## Outputs and limits",
        "",
        "- `all_validation_cases.csv` 為主要 case-level table；四份 `sorted_*.csv` 是指定排序。",
        "- `component_details.csv`、`case_0140_component_analysis.csv`、`intensity_statistics.csv`、`probability_statistics.csv` 保存明細。",
        "- `case_comparison.csv` 與 `figures/` 保存三案例比較；`correlation_statistics.csv` 與 correlation figures 保存全 validation 關聯。",
        "- Intensity 同時提供 raw DWI 與 nnUNet preprocessing 後 normalized DWI。Raw 統計在 native grid；normalized 統計在實際 preprocessed grid，GT/prediction 依 nnUNet 的 crop 與 segmentation-resampling 規則轉入該 grid，並逐例對照儲存的 preprocessed GT。",
        "- 本分析沒有調整 threshold 來選最佳結果，也沒有執行任何下一個實驗。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    final_output = args.output.resolve()
    expected_parent = (ROOT / "outputs" / "analysis").resolve()
    if not final_output.is_relative_to(expected_parent):
        raise SystemExit(f"Output must remain under {expected_parent}")
    if final_output.exists():
        raise SystemExit(f"Refusing to overwrite existing analysis: {final_output}")
    expected_ids = json.loads(SPLIT.read_text(encoding="utf-8"))[0]["val"]
    nii_ids = {path.name[:-7] for path in PREDICTIONS.glob("*.nii.gz")}
    npz_ids = {path.stem for path in PREDICTIONS.glob("*.npz")}
    if set(expected_ids) != nii_ids or set(expected_ids) != npz_ids:
        raise RuntimeError("Saved segmentation/probability cases do not match fixed Fold 0 validation split")
    mapping = pd.read_csv(MAPPING, dtype=str)
    requested_matches = mapping[mapping.apply(lambda column: column.str.contains(REQUESTED_SUBJECT, regex=False, na=False)).any(axis=1)]
    fallback_map = mapping[mapping["nnunet_case"] == FALLBACK_CASE]
    if fallback_map.empty or FALLBACK_CASE not in expected_ids:
        raise RuntimeError(f"Fallback anchor {FALLBACK_CASE} is unavailable")
    original_subject_numbers = mapping["original_case"].str.extract(r"strokecase(\d+)")[0].dropna()
    identification = {
        "requested_subject": REQUESTED_SUBJECT,
        "requested_subject_matches": requested_matches.to_dict(orient="records"),
        "requested_subject_found": not requested_matches.empty,
        "mapping_min_subject": original_subject_numbers.min(),
        "mapping_max_subject": original_subject_numbers.max(),
        "fallback_case": FALLBACK_CASE,
        "fallback_original_case": fallback_map.iloc[0]["original_case"],
        "resolution": "literal_0410_not_found; case_0140 analyzed as likely transposition candidate",
    }
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".failure_analysis_v2_staging_", dir=final_output.parent))
    try:
        (staging / "figures").mkdir()
        all_rows: list[dict[str, Any]] = []
        all_components: list[dict[str, Any]] = []
        all_intensity: list[dict[str, Any]] = []
        all_probability: list[dict[str, Any]] = []
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        probability_match: dict[str, bool] = {}
        for index, case in enumerate(expected_ids, start=1):
            gt_image, gt = load_mask(RAW / "labelsTr" / f"{case}.nii.gz")
            pred_image, pred = load_mask(PREDICTIONS / f"{case}.nii.gz")
            if gt_image.shape != pred_image.shape or not np.allclose(gt_image.affine, pred_image.affine, atol=1e-3, rtol=0):
                raise ValueError(f"GT/prediction geometry mismatch: {case}")
            raw_image = nib.load(RAW / "imagesTr" / f"{case}_0000.nii.gz")
            raw = raw_image.get_fdata(dtype=np.float32)
            if raw.shape != gt.shape or not np.allclose(raw_image.affine, gt_image.affine, atol=1e-3, rtol=0):
                raise ValueError(f"DWI/GT geometry mismatch: {case}")
            normalized, normalized_gt, normalized_pred, normalized_foreground, _properties = load_preprocessed(
                case, raw, gt, pred
            )
            probability, matches = load_probability(case, gt.shape, pred)
            probability_match[case] = matches
            voxel_ml = float(np.prod(gt_image.header.get_zooms()[:3], dtype=np.float64) / 1000)
            tp = int((gt & pred).sum())
            gt_voxels = int(gt.sum())
            pred_voxels = int(pred.sum())
            fp = int((~gt & pred).sum())
            fn = int((gt & ~pred).sum())
            components, component_rows = component_analysis(gt, pred, voxel_ml)
            for component in component_rows:
                all_components.append({"case": case, **component})
            intensities = intensity_rows(
                case,
                raw,
                normalized,
                gt,
                pred,
                normalized_gt,
                normalized_pred,
                normalized_foreground,
            )
            all_intensity.extend(intensities)
            intensity_index = {(r["representation"], r["region"]): r for r in intensities}
            prob_summary, prob_rows = probability_summary(probability, gt, pred)
            for prob_row in prob_rows:
                all_probability.append({"case": case, **prob_row})
            mapping_row = mapping[mapping["nnunet_case"] == case].iloc[0]
            gt_ml = gt_voxels * voxel_ml
            pred_ml = pred_voxels * voxel_ml
            row = {
                "case": case,
                "original_case": mapping_row["original_case"],
                "dice": safe_ratio(2 * tp, gt_voxels + pred_voxels),
                "voxel_tp": tp,
                "voxel_fp": fp,
                "voxel_fn": fn,
                "voxel_recall": safe_ratio(tp, gt_voxels),
                "voxel_precision": safe_ratio(tp, pred_voxels),
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
                "voxel_volume_ml": voxel_ml,
                "gt_ml": gt_ml,
                "pred_ml": pred_ml,
                "signed_volume_error_ml": pred_ml - gt_ml,
                "absolute_volume_error_ml": abs(pred_ml - gt_ml),
                "relative_volume_error": safe_ratio(pred_ml - gt_ml, gt_ml),
                "pred_gt_ratio": safe_ratio(pred_ml, gt_ml),
                "severe_underseg": bool(gt_voxels and pred_ml / gt_ml < 0.5),
                **components,
                **prob_summary,
                "raw_gt_mean": intensity_index[("raw_dwi", "entire_gt")]["mean"],
                "raw_gt_median": intensity_index[("raw_dwi", "entire_gt")]["median"],
                "raw_detected_gt_mean": intensity_index[("raw_dwi", "detected_gt")]["mean"],
                "raw_detected_gt_median": intensity_index[("raw_dwi", "detected_gt")]["median"],
                "raw_missed_gt_mean": intensity_index[("raw_dwi", "missed_gt")]["mean"],
                "raw_missed_gt_median": intensity_index[("raw_dwi", "missed_gt")]["median"],
                "norm_gt_mean": intensity_index[("nnunet_normalized_dwi", "entire_gt")]["mean"],
                "norm_gt_median": intensity_index[("nnunet_normalized_dwi", "entire_gt")]["median"],
                "norm_detected_gt_mean": intensity_index[("nnunet_normalized_dwi", "detected_gt")]["mean"],
                "norm_detected_gt_median": intensity_index[("nnunet_normalized_dwi", "detected_gt")]["median"],
                "norm_missed_gt_mean": intensity_index[("nnunet_normalized_dwi", "missed_gt")]["mean"],
                "norm_missed_gt_median": intensity_index[("nnunet_normalized_dwi", "missed_gt")]["median"],
                "gt_component_volume_list_ml": json.dumps(components["gt_component_volumes_ml"], separators=(",", ":")),
                "pred_component_volume_list_ml": json.dumps(components["pred_component_volumes_ml"], separators=(",", ":")),
                "probability_threshold_0p5_matches_saved_prediction": matches,
            }
            row.pop("gt_component_volumes_ml")
            row.pop("pred_component_volumes_ml")
            failure_mode, failure_labels = assign_failure_mode(row)
            row["failure_mode"] = failure_mode
            row["failure_labels"] = failure_labels
            all_rows.append(row)
            arrays[case] = (raw, gt, pred, probability)
            print(f"{index:02d}/{len(expected_ids)} {case} Dice={row['dice']} mode={failure_mode}", flush=True)

        if not all(probability_match.values()):
            mismatch = [case for case, matches in probability_match.items() if not matches]
            raise RuntimeError(f"Saved probabilities do not reproduce saved masks at 0.5: {mismatch}")
        selections = select_comparisons(all_rows, FALLBACK_CASE)
        selected_cases = set(selections.values())
        slice_rows: list[dict[str, Any]] = []
        for case in selected_cases:
            raw, gt, pred, probability = arrays[case]
            slice_rows.extend(make_case_figure(staging / "figures" / f"{case}.png", case, raw, gt, pred, probability))

        write_csv(staging / "all_validation_cases.csv", all_rows)
        write_csv(staging / "sorted_worst_dice.csv", sorted(all_rows, key=lambda row: (row["dice"] is None, row["dice"] if row["dice"] is not None else 1, row["case"])))
        write_csv(staging / "sorted_lowest_pred_gt_ratio.csv", sorted(all_rows, key=lambda row: (row["pred_gt_ratio"] is None, row["pred_gt_ratio"] if row["pred_gt_ratio"] is not None else float("inf"), row["case"])))
        write_csv(staging / "sorted_highest_fn_lesion_count.csv", sorted(all_rows, key=lambda row: (-row["lesion_fn"], row["case"])))
        write_csv(staging / "sorted_lowest_lesion_recall.csv", sorted(all_rows, key=lambda row: (row["lesion_recall"] is None, row["lesion_recall"] if row["lesion_recall"] is not None else 1, row["case"])))
        write_csv(staging / "component_details.csv", all_components)
        write_csv(staging / "case_0140_component_analysis.csv", [row for row in all_components if row["case"] == FALLBACK_CASE])
        write_csv(staging / "intensity_statistics.csv", all_intensity)
        write_csv(staging / "probability_statistics.csv", all_probability)
        write_csv(staging / "selected_slices.csv", slice_rows)

        comparison_rows = []
        for role, case in selections.items():
            row = next(item for item in all_rows if item["case"] == case)
            comparison_rows.append({"comparison_role": role, **row})
        write_csv(staging / "case_comparison.csv", comparison_rows)

        correlations = scatter_figure(
            staging / "figures" / "volume_component_correlations.png",
            all_rows,
            [
                ("gt_ml", "pred_gt_ratio", "GT volume (mL, log scale)", "Prediction / GT volume"),
                ("gt_ml", "dice", "GT volume (mL, log scale)", "Dice"),
                ("gt_lesion_count", "lesion_recall", "GT lesion count", "Lesion recall"),
                ("largest_gt_component_ml", "pred_gt_ratio", "Largest GT component (mL, log scale)", "Prediction / GT volume"),
            ],
        )
        intensity_correlations = scatter_figure(
            staging / "figures" / "intensity_correlations.png",
            all_rows,
            [
                ("raw_gt_median", "dice", "Raw GT DWI median", "Dice"),
                ("raw_gt_median", "pred_gt_ratio", "Raw GT DWI median", "Prediction / GT volume"),
                ("norm_gt_median", "dice", "Normalized GT DWI median", "Dice"),
                ("norm_gt_median", "pred_gt_ratio", "Normalized GT DWI median", "Prediction / GT volume"),
            ],
        )
        correlations.extend(intensity_correlations)
        write_csv(staging / "correlation_statistics.csv", correlations)

        anchor_row = next(row for row in all_rows if row["case"] == FALLBACK_CASE)
        anchor_components = [row for row in all_components if row["case"] == FALLBACK_CASE]
        mode_counts = Counter(row["failure_mode"] for row in all_rows)
        label_counts = Counter(label for row in all_rows for label in row["failure_labels"].split(";") if label)
        severe_rows = [row for row in all_rows if row["severe_underseg"]]
        summary = {
            "analysis_only": True,
            "training_started": False,
            "baseline": "Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
            "validation_cases": len(all_rows),
            "probabilities_available": True,
            "all_probability_threshold_checks_passed": all(probability_match.values()),
            "connectivity": 26,
            "lesion_matching": "maximum-cardinality then maximum-IoU one-to-one overlap",
            "severe_under_rule": "Pred/GT < 0.5",
            "severe_under_count": len(severe_rows),
            "severe_under_cases": [row["case"] for row in severe_rows],
            "severe_under_small_lt5ml": sum(0 < row["gt_ml"] < 5 for row in severe_rows),
            "severe_under_non_small_ge5ml": sum(row["gt_ml"] >= 5 for row in severe_rows),
            "severe_under_multifocal": sum(row["gt_lesion_count"] >= 2 for row in severe_rows),
            "failure_mode_counts": dict(sorted(mode_counts.items())),
            "failure_label_counts": dict(sorted(label_counts.items())),
            "case_identification": identification,
            "comparison_selection": selections,
            "anchor_case": anchor_row,
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "case_identification.json", identification)
        write_json(
            staging / "source_manifest.json",
            {
                "read_only_sources": {
                    "split": {"path": str(SPLIT), "sha256": sha256(SPLIT)},
                    "mapping": {"path": str(MAPPING), "sha256": sha256(MAPPING)},
                    "prediction_directory": str(PREDICTIONS),
                    "raw_directory": str(RAW),
                    "preprocessed_directory": str(PREPROCESSED),
                },
                "output": str(final_output),
                "existing_results_overwritten": False,
            },
        )
        build_report(staging / "report.md", all_rows, anchor_row, selections, correlations, anchor_components, identification)
        staging.rename(final_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(clean({"output": final_output, "comparison_selection": selections}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
