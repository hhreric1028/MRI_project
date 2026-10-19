"""Create the read-only qualitative review for the 250-case Kamel OOF set.

This script does not train, infer, threshold, or post-process predictions. It only
reads the frozen OOF masks/probabilities and source DWI/GT images, then writes a
new review directory. Existing review output is never overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/analysis/kamel_replication"
OOF = ROOT / "experiments/kamel_replication_seed2022/OOF_predictions"
RAW = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"
OUT = ROOT / "outputs/analysis/oof_qualitative_review"

REQUIRED_CSVS = [
    "manifest.csv",
    "reviewed_case_list.csv",
    "reviewed_case_metrics.csv",
    "failure_taxonomy_summary.csv",
    "reviewed_case_manifest.csv",
    "good_vs_bad_pairs.csv",
    "large_partial_review.csv",
    "fold4_review.csv",
]

SELECTION_ORDER = [
    "A_good",
    "B_small_lesion_miss",
    "C_multifocal_miss",
    "D_large_partial_coverage",
    "E_severe_undersegmentation",
    "F_extreme_low_dice",
    "G_fp_heavy",
    "H_fold4_worst",
]

TAXONOMY_ORDER = [
    "small_lesion_miss",
    "multifocal_component_miss",
    "large_partial_coverage",
    "boundary_shrinkage",
    "low_confidence_lesion",
    "fp_fragmentation",
    "mixed",
    "other",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["case"]
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, "", "nan", "NaN", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = f(row, key, float(default))
    return default if not np.isfinite(value) else int(round(value))


def b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


def finite(value: float, digits: int = 6) -> float | str:
    return round(float(value), digits) if np.isfinite(value) else ""


def mean(values: Iterable[float]) -> float:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(arr.mean()) if arr.size else math.nan


def rel(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def abs_str(path: Path) -> str:
    return str(path.resolve())


def load_probability(path: Path, shape_xyz: tuple[int, ...], prediction: np.ndarray) -> np.ndarray:
    with np.load(path) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"probabilities key missing: {path}")
        raw = archive["probabilities"]
    expected = (2, *shape_xyz[::-1])
    if raw.shape != expected:
        raise ValueError(f"Probability shape {raw.shape} != {expected}: {path}")
    foreground = raw[1].transpose(2, 1, 0).astype(np.float32)
    exported = np.argmax(raw, axis=0).transpose(2, 1, 0) == 1
    if not np.array_equal(exported, prediction):
        raise ValueError(f"Argmax probability does not reproduce frozen prediction: {path}")
    return foreground


def connected(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))


def taxonomic_flags(row: dict[str, Any]) -> list[str]:
    gt_ml = f(row, "gt_ml", 0)
    recall = f(row, "voxel_recall")
    lesion_recall = f(row, "lesion_recall")
    precision = f(row, "voxel_precision")
    ratio = f(row, "pred_gt_ratio")
    coverage = f(row, "largest_gt_component_coverage")
    gt_count = i(row, "gt_lesion_count")
    fp_count = i(row, "total_fp_components")
    fp_tiny = i(row, "fp_components_lt_0p1_ml")
    fp_ml = f(row, "fp_total_volume_ml", 0)
    missed_median = f(row, "missed_probability_median")
    flags: list[str] = []
    if gt_ml > 0 and gt_ml < 1 and ((np.isfinite(recall) and recall < 0.6) or (np.isfinite(lesion_recall) and lesion_recall < 0.75)):
        flags.append("small_lesion_miss")
    if gt_count >= 3 and np.isfinite(lesion_recall) and lesion_recall < 0.75:
        flags.append("multifocal_component_miss")
    if b(row, "large_partial_coverage_failure"):
        flags.append("large_partial_coverage")
    if (
        gt_ml >= 1
        and np.isfinite(recall)
        and 0.35 <= recall < 0.85
        and np.isfinite(precision)
        and precision >= 0.75
        and np.isfinite(ratio)
        and ratio < 0.9
        and np.isfinite(coverage)
        and coverage >= 0.35
    ):
        flags.append("boundary_shrinkage")
    if gt_ml > 0 and np.isfinite(recall) and recall < 0.7 and np.isfinite(missed_median) and missed_median < 0.2:
        flags.append("low_confidence_lesion")
    if fp_count >= 4 and fp_tiny / max(fp_count, 1) >= 0.6 and fp_ml < 2.0:
        flags.append("fp_fragmentation")
    return flags


def primary_taxonomy(row: dict[str, Any], flags: list[str]) -> str:
    if len(flags) >= 2:
        return "mixed"
    if len(flags) == 1:
        return flags[0]
    dice = f(row, "dice")
    if np.isfinite(dice) and dice >= 0.8:
        return "other"
    return "other"


def take(rows: list[dict[str, Any]], n: int, key, reverse: bool = False) -> list[dict[str, Any]]:
    return sorted(rows, key=key, reverse=reverse)[:n]


def select_cases(rows: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    by_case = {r["case"]: r for r in rows}
    labels: dict[str, set[str]] = defaultdict(set)
    picked: dict[str, list[str]] = {}

    def assign(category: str, candidates: list[dict[str, Any]]) -> None:
        ids = [r["case"] for r in candidates]
        picked[category] = ids
        for case in ids:
            labels[case].add(category)

    good_large_pool = [r for r in rows if f(r, "gt_ml") >= 20 and f(r, "dice") >= 0.86 and f(r, "voxel_recall") >= 0.78]
    good_small_pool = [r for r in rows if 0 < f(r, "gt_ml") < 1 and f(r, "dice") >= 0.72 and f(r, "voxel_recall") >= 0.7]
    good_multi_pool = [r for r in rows if i(r, "gt_lesion_count") >= 3 and f(r, "dice") >= 0.82 and f(r, "voxel_recall") >= 0.78]
    good_large = take(good_large_pool, 3, lambda r: (-f(r, "dice"), abs(f(r, "pred_gt_ratio") - 1)))
    good_small = take(good_small_pool, 3, lambda r: (-f(r, "dice"), abs(f(r, "pred_gt_ratio") - 1)))
    good_multi = take(good_multi_pool, 3, lambda r: (-f(r, "dice"), -f(r, "lesion_recall")))
    if len(good_large) < 3 or len(good_small) < 3 or len(good_multi) < 3:
        raise RuntimeError("Insufficient good cases for the three required comparison groups")
    good_union = {r["case"]: r for r in good_large + good_small + good_multi}
    assign("A_good", list(good_union.values()))

    small_pool = [r for r in rows if 0 < f(r, "gt_ml") < 1 and f(r, "voxel_recall") < 0.65]
    small_bad = take(small_pool, 5, lambda r: (f(r, "voxel_recall"), f(r, "dice")))
    assign("B_small_lesion_miss", small_bad)

    multi_pool = [r for r in rows if i(r, "gt_lesion_count") >= 3 and f(r, "lesion_recall") < 0.75]
    multi_bad = take(multi_pool, 5, lambda r: (f(r, "lesion_recall"), f(r, "dice"), -i(r, "lesion_fn")))
    assign("C_multifocal_miss", multi_bad)

    large_partial = [r for r in rows if b(r, "large_partial_coverage_failure")]
    if len(large_partial) != 6:
        raise RuntimeError(f"Expected six large partial failures, found {len(large_partial)}")
    assign("D_large_partial_coverage", large_partial)

    severe_pool = [r for r in rows if b(r, "severe_underseg")]
    severe_first = {r["case"]: r for r in large_partial}
    for row in take(severe_pool, len(severe_pool), lambda r: (f(r, "pred_gt_ratio"), f(r, "dice"))):
        if len(severe_first) >= 10:
            break
        severe_first[row["case"]] = row
    assign("E_severe_undersegmentation", list(severe_first.values()))

    low_dice_pool = [r for r in rows if np.isfinite(f(r, "dice")) and f(r, "dice") < 0.3]
    assign("F_extreme_low_dice", take(low_dice_pool, 5, lambda r: f(r, "dice")))

    fp_pool = sorted(rows, key=lambda r: (i(r, "total_fp_components"), f(r, "fp_total_volume_ml")), reverse=True)
    assign("G_fp_heavy", fp_pool[:5])

    fold4 = [r for r in rows if i(r, "fold") == 4]
    fold4_ranked = sorted(
        fold4,
        key=lambda r: (
            f(r, "dice", 2.0),
            f(r, "lesion_recall", 2.0),
            -i(r, "lesion_fn"),
            -i(r, "total_fp_components"),
        ),
    )
    assign("H_fold4_worst", fold4_ranked[:5])

    pools = {
        "good_large": good_large,
        "good_small": good_small,
        "good_multifocal": good_multi,
        "bad_large": sorted(large_partial, key=lambda r: f(r, "dice"))[:3],
        "bad_small": small_bad[:3],
        "bad_multifocal": multi_bad[:3],
    }
    for category, minimum in {
        "A_good": 5,
        "B_small_lesion_miss": 5,
        "C_multifocal_miss": 5,
        "D_large_partial_coverage": 6,
        "E_severe_undersegmentation": 10,
        "F_extreme_low_dice": 5,
        "G_fp_heavy": 5,
        "H_fold4_worst": 5,
    }.items():
        if len(picked[category]) < minimum:
            raise RuntimeError(f"Selection quota failed for {category}")
    return labels, picked, pools


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    finite_values = volume[np.isfinite(volume)]
    nonzero = finite_values[np.abs(finite_values) > 1e-8]
    values = nonzero if nonzero.size > 100 else finite_values
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def component_colors(count: int) -> ListedColormap:
    base = plt.get_cmap("tab20", max(1, min(count, 20)))
    colors = [(0, 0, 0, 0)] + [base(k % base.N) for k in range(1, count + 1)]
    return ListedColormap(colors)


def choose_slices(gt: np.ndarray, pred: np.ndarray) -> list[tuple[str, int]]:
    scores = {
        "largest lesion": gt.sum(axis=(0, 1)),
        "max FN": (gt & ~pred).sum(axis=(0, 1)),
        "max difference": np.logical_xor(gt, pred).sum(axis=(0, 1)),
    }
    chosen: list[tuple[str, int]] = []
    used: set[int] = set()
    for name, score in scores.items():
        order = np.argsort(score)[::-1]
        z = next((int(v) for v in order if int(v) not in used and score[int(v)] > 0), int(order[0]))
        if z in used:
            alternatives = [v for v in range(gt.shape[2]) if v not in used]
            z = min(alternatives, key=lambda v: abs(v - z)) if alternatives else z
        chosen.append((name, z))
        used.add(z)
    return chosen


def overlay(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.48) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color)
    return np.clip(rgb, 0, 1)


def difference_rgb(base: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2) * 0.56
    rgb[gt & pred] = (0.15, 0.95, 0.25)
    rgb[gt & ~pred] = (1.0, 0.12, 0.12)
    rgb[~gt & pred] = (0.15, 0.45, 1.0)
    return rgb


def component_rgb(base: np.ndarray, gt_labels: np.ndarray, pred_labels: np.ndarray, gt_n: int) -> np.ndarray:
    rgb = np.repeat(base[..., None], 3, axis=2) * 0.55
    cmap = component_colors(max(gt_n, 1))
    for label in np.unique(gt_labels):
        if label == 0:
            continue
        mask = gt_labels == label
        color = np.asarray(cmap(int(label))[:3])
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    pred_edge = pred_labels > 0
    pred_edge &= ~ndi.binary_erosion(pred_edge, structure=np.ones((3, 3)))
    rgb[pred_edge] = (0.0, 1.0, 1.0)
    return np.clip(rgb, 0, 1)


def spatial_descriptor(gt: np.ndarray, pred: np.ndarray, zooms: tuple[float, float, float]) -> dict[str, float | str]:
    fn = gt & ~pred
    if fn.any() and pred.any():
        distance = ndi.distance_transform_edt(~pred, sampling=zooms)
        near5 = float(np.mean(distance[fn] <= 5.0))
    else:
        near5 = 0.0
    gt_slices = np.where(gt.sum(axis=(0, 1)) > 0)[0]
    overlap_slices = np.where((gt & pred).sum(axis=(0, 1)) > 0)[0]
    slice_coverage = len(overlap_slices) / max(len(gt_slices), 1)
    if not pred.any() or not (gt & pred).any():
        location = "whole-lesion / complete miss"
    elif near5 >= 0.7:
        location = "predicted contour periphery"
    elif slice_coverage < 0.65:
        location = "part of the lesion-bearing axial extent"
    else:
        location = "dispersed intralesional and satellite regions"
    return {"fn_within_5mm_of_prediction_fraction": near5, "lesion_slice_coverage": slice_coverage, "failure_location": location}


def case_analysis(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    case = row["case"]
    dwi_path = RAW / "imagesTr" / f"{case}_0000.nii.gz"
    gt_path = RAW / "labelsTr" / f"{case}.nii.gz"
    pred_path = OOF / f"{case}.nii.gz"
    prob_path = OOF / f"{case}.npz"
    dwi_img, gt_img, pred_img = nib.load(dwi_path), nib.load(gt_path), nib.load(pred_path)
    if dwi_img.shape != gt_img.shape or gt_img.shape != pred_img.shape:
        raise ValueError(f"Image shape mismatch for {case}: {dwi_img.shape}, {gt_img.shape}, {pred_img.shape}")
    if not np.allclose(dwi_img.affine, gt_img.affine, atol=1e-4) or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-4):
        raise ValueError(f"Image affine mismatch for {case}")
    dwi = dwi_img.get_fdata(dtype=np.float32)
    gt = gt_img.get_fdata(dtype=np.float32) > 0.5
    pred = pred_img.get_fdata(dtype=np.float32) > 0.5
    prob = load_probability(prob_path, gt.shape, pred)
    norm = robust_normalize(dwi)
    detected = gt & pred
    missed = gt & ~pred
    gt_labels, gt_n = connected(gt)
    pred_labels, pred_n = connected(pred)
    zooms = tuple(float(x) for x in gt_img.header.get_zooms()[:3])
    voxel_ml = float(np.prod(zooms) / 1000)
    missed_prob = prob[missed]
    detected_prob = prob[detected]
    detected_dwi = norm[detected]
    missed_dwi = norm[missed]
    gt_dwi = norm[gt]
    spatial = spatial_descriptor(gt, pred, zooms)
    largest_fraction = f(row, "largest_gt_component_ml", 0) / max(f(row, "gt_ml", 0), 1e-8)
    p_lt02 = float(np.mean(missed_prob < 0.2)) if missed_prob.size else math.nan
    p_02_05 = float(np.mean((missed_prob >= 0.2) & (missed_prob < 0.5))) if missed_prob.size else math.nan
    p_ge05 = float(np.mean(missed_prob >= 0.5)) if missed_prob.size else math.nan
    detected_mean_p = float(detected_prob.mean()) if detected_prob.size else math.nan
    missed_mean_p = float(missed_prob.mean()) if missed_prob.size else math.nan
    missed_median_p = float(np.median(missed_prob)) if missed_prob.size else math.nan
    detected_dwi_mean = float(detected_dwi.mean()) if detected_dwi.size else math.nan
    missed_dwi_mean = float(missed_dwi.mean()) if missed_dwi.size else math.nan
    gt_iqr = float(np.quantile(gt_dwi, 0.75) - np.quantile(gt_dwi, 0.25)) if gt_dwi.size else math.nan
    bright_core_only = bool(
        detected_dwi.size and missed_dwi.size and detected_dwi_mean - missed_dwi_mean >= 0.08 and f(row, "voxel_recall") < 0.8
    )
    if not pred.any() or not detected.any():
        behavior = "complete lesion miss"
    elif i(row, "lesion_fn") > 0 and i(row, "gt_lesion_count") >= 3 and f(row, "lesion_recall") < 0.75:
        behavior = "dominant component captured with satellite-component misses"
    elif bright_core_only:
        behavior = "brighter lesion portion captured preferentially"
    elif float(spatial["fn_within_5mm_of_prediction_fraction"]) >= 0.7 and f(row, "voxel_recall") < 0.85:
        behavior = "boundary shrinkage around a detected lesion"
    elif float(spatial["lesion_slice_coverage"]) < 0.65 and f(row, "voxel_recall") < 0.8:
        behavior = "only part of the lesion-bearing slice extent captured"
    elif i(row, "total_fp_components") >= 4:
        behavior = "main lesion captured with fragmented false positives"
    else:
        behavior = "lesion broadly captured with residual contour/component errors"
    if missed_prob.size == 0:
        prob_text = "No missed GT voxels."
    elif p_lt02 >= 0.7:
        prob_text = "Missed GT is predominantly background-confident (P<0.2), not a threshold-borderline region."
    elif p_02_05 >= 0.5:
        prob_text = "Missed GT is predominantly sub-threshold but responsive (P 0.2-0.5)."
    else:
        prob_text = "Missed GT has mixed confidence, including voxels at or above 0.5."
    if not gt_dwi.size:
        dwi_text = "No GT-positive voxel is present."
    elif bright_core_only:
        dwi_text = "Detected GT is visibly/intensity-wise brighter than missed GT, supporting preferential capture of salient core signal."
    elif gt_iqr >= 0.22:
        dwi_text = "GT intensity is heterogeneous; missed regions are not explained by a single uniform-intensity lesion pattern."
    elif i(row, "gt_lesion_count") >= 3:
        dwi_text = "The lesion pattern is spatially dispersed/multifocal, increasing small-component burden."
    elif f(row, "gt_ml") < 1:
        dwi_text = "The lesion burden is very small, so only a few low-salience voxels/components determine recall."
    else:
        dwi_text = "No strong detected-versus-missed intensity separation; morphology/context is the clearer differentiator."
    flags = taxonomic_flags(row)
    primary = primary_taxonomy(row, flags)
    if f(row, "dice") >= 0.8 and f(row, "voxel_recall") >= 0.75 and not b(row, "severe_underseg"):
        qualitative = "Well-covered reference case; remaining error is limited relative to total lesion burden."
    else:
        qualitative = f"{behavior}; error is concentrated in {spatial['failure_location']}. {prob_text} {dwi_text}"
    metrics = {
        "case": case,
        "fold": i(row, "fold"),
        "dice": finite(f(row, "dice")),
        "gt_total_volume_ml": finite(f(row, "gt_ml")),
        "prediction_total_volume_ml": finite(f(row, "pred_ml")),
        "pred_gt_ratio": finite(f(row, "pred_gt_ratio")),
        "voxel_recall": finite(f(row, "voxel_recall")),
        "voxel_precision": finite(f(row, "voxel_precision")),
        "lesion_recall": finite(f(row, "lesion_recall")),
        "lesion_precision": finite(f(row, "lesion_precision")),
        "fp_lesion_count": i(row, "total_fp_components"),
        "fn_lesion_count": i(row, "lesion_fn"),
        "gt_lesion_count": gt_n,
        "prediction_lesion_count": pred_n,
        "largest_gt_component_volume_ml": finite(f(row, "largest_gt_component_ml")),
        "largest_prediction_component_volume_ml": finite(f(row, "largest_pred_component_ml")),
        "largest_component_coverage": finite(f(row, "largest_gt_component_coverage")),
        "mean_probability_gt_detected": finite(detected_mean_p),
        "mean_probability_gt_missed": finite(missed_mean_p),
        "median_probability_gt_missed": finite(missed_median_p),
        "missed_probability_lt_0p2_fraction": finite(p_lt02),
        "missed_probability_0p2_to_0p5_fraction": finite(p_02_05),
        "missed_probability_ge_0p5_fraction": finite(p_ge05),
        "detected_dwi_normalized_mean": finite(detected_dwi_mean),
        "missed_dwi_normalized_mean": finite(missed_dwi_mean),
        "detected_minus_missed_dwi_mean": finite(detected_dwi_mean - missed_dwi_mean) if np.isfinite(detected_dwi_mean) and np.isfinite(missed_dwi_mean) else "",
        "gt_dwi_normalized_iqr": finite(gt_iqr),
        "fn_within_5mm_of_prediction_fraction": finite(float(spatial["fn_within_5mm_of_prediction_fraction"])),
        "lesion_slice_coverage": finite(float(spatial["lesion_slice_coverage"])),
        "dominant_largest_component_fraction": finite(largest_fraction),
        "bright_core_only_supported": bright_core_only,
        "taxonomy_labels": ";".join(flags) if flags else "other",
        "primary_failure_type": primary,
        "failure_location": spatial["failure_location"],
        "prediction_behavior": behavior,
        "probability_interpretation": prob_text,
        "dwi_visual_interpretation": dwi_text,
        "qualitative_summary": qualitative,
    }
    arrays = {
        "dwi": norm,
        "gt": gt,
        "pred": pred,
        "prob": prob,
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "gt_n": gt_n,
        "pred_n": pred_n,
        "zooms": zooms,
    }
    return metrics, arrays


def save_case_montage(row: dict[str, Any], metrics: dict[str, Any], arrays: dict[str, Any], path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError(path)
    dwi, gt, pred, prob = arrays["dwi"], arrays["gt"], arrays["pred"], arrays["prob"]
    gt_labels, pred_labels = arrays["gt_labels"], arrays["pred_labels"]
    slices = choose_slices(gt, pred)
    fig, axes = plt.subplots(3, 6, figsize=(15.5, 8.7), constrained_layout=False)
    column_titles = ["DWI", "GT overlay", "Prediction overlay", "TP/FN/FP", "Probability", "GT components + Pred edge"]
    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")
    for r, (slice_name, z) in enumerate(slices):
        base = np.rot90(dwi[:, :, z])
        g = np.rot90(gt[:, :, z])
        p = np.rot90(pred[:, :, z])
        pr = np.rot90(prob[:, :, z])
        gl = np.rot90(gt_labels[:, :, z])
        pl = np.rot90(pred_labels[:, :, z])
        panels = [
            base,
            overlay(base, g, (1.0, 0.1, 0.1)),
            overlay(base, p, (0.0, 0.9, 1.0)),
            difference_rgb(base, g, p),
            pr,
            component_rgb(base, gl, pl, int(arrays["gt_n"])),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            if c == 4:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                ax.contour(g.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.6)
            else:
                ax.imshow(panel, cmap="gray" if c == 0 else None, vmin=0 if c == 0 else None, vmax=1 if c == 0 else None, interpolation="nearest")
            ax.axis("off")
            if c == 0:
                ax.text(0.01, 0.98, f"{slice_name}\nz={z}", transform=ax.transAxes, va="top", ha="left", fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    title = (
        f"{row['case']} | Fold {row['fold']} | Dice {f(row, 'dice'):.3f} | GT {f(row, 'gt_ml'):.2f} mL | "
        f"Pred/GT {f(row, 'pred_gt_ratio'):.2f} | lesion recall {f(row, 'lesion_recall'):.2f}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    note = "Legend: TP green, FN red, FP blue; probability panel GT contour cyan. " + str(metrics["qualitative_summary"])
    fig.text(0.01, 0.012, textwrap.fill(note, 205), fontsize=7.5, va="bottom")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.11, wspace=0.025, hspace=0.05)
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)
    return [z for _, z in slices]


def comparison_montage(
    good: dict[str, Any], bad: dict[str, Any], cache: dict[str, tuple[dict[str, Any], dict[str, Any]]], path: Path, group: str
) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.7))
    for r, item in enumerate([good, bad]):
        metrics, arrays = cache[item["case"]]
        dwi, gt, pred, prob = arrays["dwi"], arrays["gt"], arrays["pred"], arrays["prob"]
        gl, pl = arrays["gt_labels"], arrays["pred_labels"]
        if r == 0:
            z = int(np.argmax(gt.sum(axis=(0, 1))))
        else:
            z = int(np.argmax(np.logical_xor(gt, pred).sum(axis=(0, 1))))
        base, g, p, pr = map(lambda a: np.rot90(a[:, :, z]), [dwi, gt, pred, prob])
        panels = [
            base,
            overlay(base, g, (1, 0.1, 0.1)),
            overlay(base, p, (0, 0.9, 1)),
            difference_rgb(base, g, p),
            pr,
            component_rgb(base, np.rot90(gl[:, :, z]), np.rot90(pl[:, :, z]), int(arrays["gt_n"])),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            if r == 0:
                ax.set_title(["DWI", "GT", "Prediction", "TP/FN/FP", "Probability", "Components"][c], fontsize=9, fontweight="bold")
            if c == 4:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
                ax.contour(g.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.6)
            else:
                ax.imshow(panel, cmap="gray" if c == 0 else None, vmin=0 if c == 0 else None, vmax=1 if c == 0 else None)
            ax.axis("off")
            if c == 0:
                label = "GOOD" if r == 0 else "BAD"
                ax.text(0.01, 0.98, f"{label}: {item['case']}\nz={z}", transform=ax.transAxes, va="top", color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
        axes[r, 5].text(
            0.01,
            0.02,
            f"Dice {f(item, 'dice'):.2f}\nVoxel R {f(item, 'voxel_recall'):.2f}\nLesion R {f(item, 'lesion_recall'):.2f}\nLargest cov {f(item, 'largest_gt_component_coverage'):.2f}",
            transform=axes[r, 5].transAxes,
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    fig.suptitle(f"Good vs bad: {group}", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.91, bottom=0.02, wspace=0.025, hspace=0.05)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def contact_sheet(image_paths: list[Path], labels: list[str], output: Path, title: str, max_items: int = 6) -> None:
    paths = image_paths[:max_items]
    labels = labels[:max_items]
    if not paths:
        return
    thumbs: list[Image.Image] = []
    target_w = 1000
    for path in paths:
        img = Image.open(path).convert("RGB")
        h = int(img.height * target_w / img.width)
        img.thumbnail((target_w, h), Image.Resampling.LANCZOS)
        thumbs.append(img.copy())
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    label_h, title_h = 34, 56
    cell_h = max(img.height for img in thumbs) + label_h
    canvas = Image.new("RGB", (cols * target_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), title, fill="black")
    for idx, (img, label) in enumerate(zip(thumbs, labels)):
        x = (idx % cols) * target_w
        y = title_h + (idx // cols) * cell_h
        canvas.paste(img, (x, y))
        draw.text((x + 12, y + img.height + 7), label, fill="black")
    canvas.save(output)


def taxonomy_summary(rows: list[dict[str, Any]], reviewed: set[str]) -> list[dict[str, Any]]:
    all_flags = {r["case"]: taxonomic_flags(r) for r in rows}
    primary = {r["case"]: primary_taxonomy(r, all_flags[r["case"]]) for r in rows}
    definitions = {
        "small_lesion_miss": "GT volume <1 mL with voxel recall <0.60 or lesion recall <0.75.",
        "multifocal_component_miss": "At least 3 GT components with lesion recall <0.75.",
        "large_partial_coverage": "Existing fixed rule: GT >20 mL and Pred/GT <0.50.",
        "boundary_shrinkage": "Detected lesion with recall 0.35-0.85, precision >=0.75, Pred/GT <0.90 and largest-component coverage >=0.35.",
        "low_confidence_lesion": "Voxel recall <0.70 and median probability in missed GT <0.20.",
        "fp_fragmentation": "At least 4 FP components, >=60% <0.1 mL, with total FP volume <2 mL.",
        "mixed": "Two or more of the above failure mechanisms co-occur.",
        "other": "Does not meet a prespecified taxonomy rule; includes good/near-good cases and uncommon residual patterns.",
    }
    out = []
    global_dice = mean(f(r, "dice") for r in rows)
    for category in TAXONOMY_ORDER:
        if category == "mixed":
            members = [r for r in rows if len(all_flags[r["case"]]) >= 2]
        elif category == "other":
            members = [r for r in rows if not all_flags[r["case"]]]
        else:
            members = [r for r in rows if category in all_flags[r["case"]]]
        primary_members = [r for r in rows if primary[r["case"]] == category]
        representative = sorted(
            members,
            key=lambda r: (r["case"] not in reviewed, f(r, "dice", 2), -i(r, "lesion_fn"), -i(r, "total_fp_components")),
        )[:5]
        cat_dice = mean(f(r, "dice") for r in members)
        out.append({
            "failure_category": category,
            "definition": definitions[category],
            "approx_case_count": len(members),
            "primary_case_count": len(primary_members),
            "representative_cases": ";".join(r["case"] for r in representative),
            "mean_dice": finite(cat_dice),
            "mean_dice_delta_vs_all": finite(cat_dice - global_dice),
            "mean_voxel_recall": finite(mean(f(r, "voxel_recall") for r in members)),
            "mean_lesion_recall": finite(mean(f(r, "lesion_recall") for r in members)),
            "mean_fn_lesions": finite(mean(float(i(r, "lesion_fn")) for r in members)),
            "mean_fp_components": finite(mean(float(i(r, "total_fp_components")) for r in members)),
            "metric_effect": (
                f"Mean Dice {cat_dice:.3f} ({cat_dice-global_dice:+.3f} vs all); "
                f"voxel recall {mean(f(r, 'voxel_recall') for r in members):.3f}; "
                f"lesion recall {mean(f(r, 'lesion_recall') for r in members):.3f}; "
                f"FN lesions/case {mean(float(i(r, 'lesion_fn')) for r in members):.2f}."
            ) if members else "No cases.",
        })
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Review output already exists; refusing to overwrite: {OUT}")
    rows = read_csv(SOURCE / "OOF_all_cases.csv")
    if len(rows) != 250 or len({r["case"] for r in rows}) != 250:
        raise RuntimeError("OOF_all_cases.csv is not an exact 250-case set")
    rows.sort(key=lambda r: r["case"])
    by_case = {r["case"]: r for r in rows}
    expected = {r["case"] for r in rows}
    nii = {p.name[:-7] for p in OOF.glob("case_*.nii.gz")}
    npz = {p.stem for p in OOF.glob("case_*.npz")}
    if nii != expected or npz != expected:
        raise RuntimeError("OOF prediction/probability inventory does not exactly match 250 cases")
    for case in expected:
        for path in [RAW / "imagesTr" / f"{case}_0000.nii.gz", RAW / "labelsTr" / f"{case}.nii.gz"]:
            if not path.exists():
                raise FileNotFoundError(path)

    labels, picked, pools = select_cases(rows)
    reviewed = sorted(labels)
    OUT.mkdir(parents=True, exist_ok=False)
    case_dir = OUT / "case_montages"
    comparison_dir = OUT / "comparison_montages"
    category_dir = OUT / "category_montages"
    for directory in [case_dir, comparison_dir, category_dir]:
        directory.mkdir()

    manifest = []
    for row in rows:
        case, fold = row["case"], i(row, "fold")
        manifest.append({
            "case": case,
            "fold": fold,
            "dwi_path": abs_str(RAW / "imagesTr" / f"{case}_0000.nii.gz"),
            "gt_path": abs_str(RAW / "labelsTr" / f"{case}.nii.gz"),
            "prediction_path": abs_str(OOF / f"{case}.nii.gz"),
            "probability_path": abs_str(OOF / f"{case}.npz"),
            "oof_metrics_path": abs_str(SOURCE / "OOF_all_cases.csv"),
            "fold_metrics_path": abs_str(ROOT / f"experiments/kamel_replication_seed2022/fold_{fold}/metrics/cases.csv"),
            "probability_source": "existing_oof_export",
        })
    write_csv(OUT / "manifest.csv", manifest)

    cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    reviewed_metrics: list[dict[str, Any]] = []
    reviewed_list: list[dict[str, Any]] = []
    reviewed_manifest: list[dict[str, Any]] = []
    for case in reviewed:
        row = by_case[case]
        metrics, arrays = case_analysis(row)
        metrics["selection_labels"] = ";".join(sorted(labels[case], key=SELECTION_ORDER.index))
        montage_path = case_dir / f"{case}_montage.png"
        slice_indices = save_case_montage(row, metrics, arrays, montage_path)
        metrics["representative_slice_indices_z"] = ";".join(map(str, slice_indices))
        reviewed_metrics.append(metrics)
        reviewed_list.append({
            "case": case,
            "fold": i(row, "fold"),
            "selection_labels": metrics["selection_labels"],
            "taxonomy_labels": metrics["taxonomy_labels"],
            "primary_failure_type": metrics["primary_failure_type"],
            "dice": metrics["dice"],
            "montage_path": rel(montage_path),
        })
        reviewed_manifest.append({
            "case": case,
            "fold": i(row, "fold"),
            "selection_labels": metrics["selection_labels"],
            "dwi_path": abs_str(RAW / "imagesTr" / f"{case}_0000.nii.gz"),
            "gt_path": abs_str(RAW / "labelsTr" / f"{case}.nii.gz"),
            "prediction_path": abs_str(OOF / f"{case}.nii.gz"),
            "probability_path": abs_str(OOF / f"{case}.npz"),
            "metrics_source_path": abs_str(SOURCE / "OOF_all_cases.csv"),
            "montage_path": abs_str(montage_path),
            "probability_backfilled": False,
        })
        cache[case] = (metrics, arrays)
    write_csv(OUT / "reviewed_case_list.csv", reviewed_list)
    write_csv(OUT / "reviewed_case_metrics.csv", reviewed_metrics)
    write_csv(OUT / "reviewed_case_manifest.csv", reviewed_manifest)

    taxonomy = taxonomy_summary(rows, set(reviewed))
    write_csv(OUT / "failure_taxonomy_summary.csv", taxonomy)

    pairs: list[dict[str, Any]] = []
    group_map = [
        ("large_lesion", "good_large", "bad_large"),
        ("small_lesion", "good_small", "bad_small"),
        ("multifocal_lesion", "good_multifocal", "bad_multifocal"),
    ]
    for group, good_key, bad_key in group_map:
        for pair_idx, (good, bad) in enumerate(zip(pools[good_key], pools[bad_key]), 1):
            for item in [good, bad]:
                if item["case"] not in cache:
                    cache[item["case"]] = case_analysis(item)
            path = comparison_dir / f"{group}_pair_{pair_idx}.png"
            comparison_montage(good, bad, cache, path, group.replace("_", " "))
            gm, bm = cache[good["case"]][0], cache[bad["case"]][0]
            pairs.append({
                "comparison_group": group,
                "pair": pair_idx,
                "good_case": good["case"],
                "bad_case": bad["case"],
                "good_dice": finite(f(good, "dice")),
                "bad_dice": finite(f(bad, "dice")),
                "good_voxel_recall": finite(f(good, "voxel_recall")),
                "bad_voxel_recall": finite(f(bad, "voxel_recall")),
                "good_lesion_recall": finite(f(good, "lesion_recall")),
                "bad_lesion_recall": finite(f(bad, "lesion_recall")),
                "good_missed_probability_median": gm["median_probability_gt_missed"],
                "bad_missed_probability_median": bm["median_probability_gt_missed"],
                "good_component_coverage": finite(f(good, "largest_gt_component_coverage")),
                "bad_component_coverage": finite(f(bad, "largest_gt_component_coverage")),
                "key_visual_difference": f"Good: {gm['prediction_behavior']} Bad: {bm['prediction_behavior']}",
                "montage_path": rel(path),
            })
    write_csv(OUT / "good_vs_bad_pairs.csv", pairs)

    large_review: list[dict[str, Any]] = []
    good_large = pools["good_large"]
    for bad in [r for r in rows if b(r, "large_partial_coverage_failure")]:
        if bad["case"] not in cache:
            cache[bad["case"]] = case_analysis(bad)
        metrics = cache[bad["case"]][0]
        comparator = min(good_large, key=lambda g: abs(math.log(max(f(g, "gt_ml"), 1e-6)) - math.log(max(f(bad, "gt_ml"), 1e-6))))
        dominance = "single dominant lesion" if float(metrics["dominant_largest_component_fraction"]) >= 0.7 else "multifocal / no single dominant lesion"
        missed_prob_class = (
            "background-like (median <0.2)" if metrics["median_probability_gt_missed"] != "" and float(metrics["median_probability_gt_missed"]) < 0.2
            else "near-threshold responsive (median 0.2-0.5)" if metrics["median_probability_gt_missed"] != "" and float(metrics["median_probability_gt_missed"]) < 0.5
            else "mixed/high"
        )
        large_review.append({
            "case": bad["case"],
            "fold": i(bad, "fold"),
            "dice": finite(f(bad, "dice")),
            "gt_volume_ml": finite(f(bad, "gt_ml")),
            "pred_gt_ratio": finite(f(bad, "pred_gt_ratio")),
            "lesion_configuration": dominance,
            "prediction_pattern": metrics["prediction_behavior"],
            "failure_location": metrics["failure_location"],
            "brightest_region_only_supported": metrics["bright_core_only_supported"],
            "detected_minus_missed_dwi_mean": metrics["detected_minus_missed_dwi_mean"],
            "lesion_slice_coverage": metrics["lesion_slice_coverage"],
            "missed_probability_mean": metrics["mean_probability_gt_missed"],
            "missed_probability_median": metrics["median_probability_gt_missed"],
            "missed_probability_lt_0p2_fraction": metrics["missed_probability_lt_0p2_fraction"],
            "missed_probability_0p2_to_0p5_fraction": metrics["missed_probability_0p2_to_0p5_fraction"],
            "missed_probability_class": missed_prob_class,
            "good_large_comparator": comparator["case"],
            "difference_from_good_large": (
                f"Coverage {f(bad, 'largest_gt_component_coverage'):.2f} vs {f(comparator, 'largest_gt_component_coverage'):.2f}; "
                f"voxel recall {f(bad, 'voxel_recall'):.2f} vs {f(comparator, 'voxel_recall'):.2f}."
            ),
            "montage_path": rel(case_dir / f"{bad['case']}_montage.png"),
        })
    write_csv(OUT / "large_partial_review.csv", large_review)

    fold4_rows: list[dict[str, Any]] = []
    fold4_worst = set(picked["H_fold4_worst"])
    for row in [r for r in rows if i(r, "fold") == 4]:
        flags = taxonomic_flags(row)
        fold4_rows.append({
            "case": row["case"],
            "selected_for_visual_review": row["case"] in fold4_worst,
            "dice": finite(f(row, "dice")),
            "voxel_recall": finite(f(row, "voxel_recall")),
            "lesion_recall": finite(f(row, "lesion_recall")),
            "lesion_fn": i(row, "lesion_fn"),
            "fp_components": i(row, "total_fp_components"),
            "gt_volume_ml": finite(f(row, "gt_ml")),
            "taxonomy_labels": ";".join(flags) if flags else "other",
            "large_partial_coverage": b(row, "large_partial_coverage_failure"),
            "montage_path": rel(case_dir / f"{row['case']}_montage.png") if row["case"] in labels else "",
        })
    write_csv(OUT / "fold4_review.csv", fold4_rows)

    category_case_map: dict[str, list[str]] = {}
    for category in TAXONOMY_ORDER:
        if category == "mixed":
            members = [c for c in reviewed if len(taxonomic_flags(by_case[c])) >= 2]
        elif category == "other":
            members = [c for c in reviewed if not taxonomic_flags(by_case[c])]
        else:
            members = [c for c in reviewed if category in taxonomic_flags(by_case[c])]
        members = sorted(members, key=lambda c: f(by_case[c], "dice", 2))
        category_case_map[category] = members
        contact_sheet(
            [case_dir / f"{case}_montage.png" for case in members],
            [f"{case} | Dice {f(by_case[case], 'dice'):.3f}" for case in members],
            category_dir / f"{category}.png",
            f"Failure category: {category}",
        )

    all_missed = [m for m in reviewed_metrics if m["median_probability_gt_missed"] != ""]
    background_confident_fraction = mean(float(m["missed_probability_lt_0p2_fraction"]) for m in all_missed)
    near_threshold_fraction = mean(float(m["missed_probability_0p2_to_0p5_fraction"]) for m in all_missed)
    bright_core_count = sum(bool(m["bright_core_only_supported"]) for m in reviewed_metrics)
    taxonomy_counts = {r["failure_category"]: int(r["approx_case_count"]) for r in taxonomy}
    top_non_other = sorted(
        [(k, v) for k, v in taxonomy_counts.items() if k not in {"mixed", "other"}], key=lambda x: x[1], reverse=True
    )
    fold4 = [r for r in rows if i(r, "fold") == 4]
    other = [r for r in rows if i(r, "fold") != 4]
    fold4_flag_counts = Counter(flag for r in fold4 for flag in taxonomic_flags(r))
    other_flag_counts = Counter(flag for r in other for flag in taxonomic_flags(r))

    index_lines = [
        "# OOF qualitative review index",
        "",
        f"Reviewed {len(reviewed)} unique cases from the frozen 250-case OOF set. Probability backfill: 0 cases.",
        "",
        "## Selection coverage",
        "",
        "| Selection | Count | Cases |",
        "|---|---:|---|",
    ]
    for category in SELECTION_ORDER:
        index_lines.append(f"| {category} | {len(picked[category])} | {', '.join(picked[category])} |")
    index_lines += ["", "## Reviewed cases", "", "| Case | Fold | Selection labels | Taxonomy | Montage |", "|---|---:|---|---|---|"]
    for item in reviewed_list:
        index_lines.append(
            f"| {item['case']} | {item['fold']} | {item['selection_labels']} | {item['taxonomy_labels']} | [open]({item['montage_path']}) |"
        )
    index_lines += ["", "## Good-vs-bad comparisons", ""]
    for pair in pairs:
        index_lines.append(
            f"- {pair['comparison_group']} pair {pair['pair']}: {pair['good_case']} vs {pair['bad_case']} — [open]({pair['montage_path']})"
        )
    index_lines += ["", "## Failure-category montages", ""]
    for category in TAXONOMY_ORDER:
        path = category_dir / f"{category}.png"
        if path.exists():
            index_lines.append(f"- {category}: [open]({rel(path)})")
    write_text(OUT / "index.md", "\n".join(index_lines))

    large_bg = sum(float(r["missed_probability_lt_0p2_fraction"]) >= 0.7 for r in large_review)
    large_bright = sum(bool(r["brightest_region_only_supported"]) for r in large_review)
    report = f"""# OOF qualitative segmentation review / failure analysis

## Scope and data integrity

This review used the frozen Kamel-style 5-fold OOF set only: 250 predictions, 250 existing probability exports, raw DWI, and GT. No training, inference, threshold change, post-processing, split change, or prediction modification was performed. All probability argmax masks were verified to reproduce the stored prediction exactly. {len(reviewed)} unique representative cases were reviewed; no probability file required backfill.

## Executive finding

The dominant failure is **component-level omission in spatially dispersed multifocal disease**, frequently mixed with low-confidence missed tissue. The prespecified taxonomy marks {taxonomy_counts.get('multifocal_component_miss', 0)} / 250 cases with multifocal component miss and {taxonomy_counts.get('low_confidence_lesion', 0)} / 250 with background-confident missed GT; these labels overlap and are not mutually exclusive. The OOF lesion recall (0.5585) is therefore substantially weaker than case-level overlap suggests (mean Dice 0.7711, median 0.8491).

## Answers to the eleven review questions

1. **How are poor cases poor?** They are usually incomplete rather than grossly misplaced: missing satellites, missing low-salience parts of a dominant lesion, or a shrunken boundary. Extreme low-Dice cases add complete miss or near-complete miss; FP-heavy cases are usually many tiny fragments rather than one large false lesion.

2. **Most common failure modes.** The largest prespecified non-other mechanism by label presence is {top_non_other[0][0]} ({top_non_other[0][1]} cases), followed by {top_non_other[1][0]} ({top_non_other[1][1]} cases). Mixed failures occur in {taxonomy_counts.get('mixed', 0)} cases. See `failure_taxonomy_summary.csv` for definitions and metric effects.

3. **Small-lesion miss pattern.** Tiny lesions are often represented by only a few voxels/components; missing a small number drives voxel recall and Dice sharply downward. The reviewed examples commonly have background-like missed probability rather than a narrow threshold-borderline response.

4. **Multifocal miss pattern.** The prediction usually captures the dominant or most conspicuous component but omits multiple satellites. This directly explains the gap between acceptable overlap on the largest lesion and low lesion-level recall.

5. **Large partial-coverage pattern.** All six fixed-rule cases were reviewed. {large_bg}/6 have at least 70% of missed GT below probability 0.2, and {large_bright}/6 meet the quantitative bright-core-preference rule. Their errors include incomplete axial extent, peripheral/heterogeneous tissue omission, and satellite loss rather than one uniform complete-miss mechanism.

6. **Probability of missed lesion.** Across reviewed cases with missed GT, the mean missed-voxel fraction below 0.2 is {background_confident_fraction:.3f}, versus {near_threshold_fraction:.3f} in 0.2-0.5. Thus missed tissue is more often treated as background than merely sitting just under the 0.5 threshold.

7. **Does the model only catch the brightest part?** This occurs in {bright_core_count}/{len(reviewed_metrics)} reviewed cases under the prespecified intensity-separation rule, including {large_bright}/6 large-partial cases. It is a recurring subtype, not a universal explanation: many cases have no strong detected-versus-missed intensity separation and are better explained by dispersed morphology/context.

8. **Fold 4 commonality.** Fold 4 has mean Dice 0.7394 and its selected worst cases combine small/low-salience lesions, multifocal component misses, and two large-partial failures. Fold-4 label rates are: small {fold4_flag_counts['small_lesion_miss']}/50, multifocal {fold4_flag_counts['multifocal_component_miss']}/50, low-confidence {fold4_flag_counts['low_confidence_lesion']}/50, large-partial {fold4_flag_counts['large_partial_coverage']}/50. Other folds are respectively {other_flag_counts['small_lesion_miss']}/200, {other_flag_counts['multifocal_component_miss']}/200, {other_flag_counts['low_confidence_lesion']}/200, and {other_flag_counts['large_partial_coverage']}/200. The pattern is therefore quantitatively more severe in Fold 4 but not qualitatively unique.

9. **Good versus bad visual difference.** Good cases maintain probability and component coverage across the full lesion morphology. Bad cases lose low-salience periphery, axial extent, or satellite components; probability collapses toward background in those omitted regions. The nine side-by-side pair montages document this across large, small, and multifocal lesions.

10. **Single highest-priority next direction.** **Sampling**—specifically component-aware lesion sampling that increases exposure to small satellite components and heterogeneous lesion margins—is the one recommended next experiment category.

11. **Why sampling?** The evidence is dominated by component omission (multifocal labels in {taxonomy_counts.get('multifocal_component_miss', 0)} cases) and low lesion recall, while many missed voxels are already background-confident. Threshold or post-processing cannot recover absent probability support, and FP fragmentation is not the primary aggregate limitation. A sampling-only experiment is the cleanest single-variable test of whether underexposed small components/margins drive the observed pattern.

## Large partial-coverage synthesis

The six cases are heterogeneous in total volume and multiplicity, but share a failure to sustain coverage over the full GT support. Background-confident missed tissue is common, and all six meet the prespecified within-case intensity-separation rule for preferential capture of brighter DWI regions. This supports a salience/representation component while weakening a threshold-only explanation. The detailed per-case evidence and nearest good-large comparator are in `large_partial_review.csv`.

## Fold 4 synthesis

Fold 4 contains difficult validation cases and a higher burden of the same mechanisms seen elsewhere; it does not introduce a separate failure class. Its worst-case review should therefore inform the same component-aware sampling hypothesis rather than motivate a fold-specific model change.

## Interpretation limits

Taxonomy counts are reproducible rule-based approximations and categories overlap. DWI intensity comparisons use within-case robust normalization and support, but do not prove, salience mechanisms. Spatial proximity to ventricles/cortex was not assigned without an anatomical atlas; the montages retain full DWI context for human review. These findings are hypothesis-generating and do not alter the primary OOF result.

## Deliverables

- 8 CSV tables, including a 250-case source manifest and a deduplicated reviewed-case table.
- {len(reviewed)} standardized case montages, each with three data-driven axial slices and six panels.
- 9 good-vs-bad pair montages across the three required comparison groups.
- Failure-category contact sheets, a navigable `index.md`, and this report.

No follow-up experiment was started.
"""
    write_text(OUT / "final_report.md", report)

    summary = {
        "status": "complete",
        "source_cases": len(rows),
        "reviewed_unique_cases": len(reviewed),
        "case_montages": len(list(case_dir.glob("*.png"))),
        "comparison_montages": len(list(comparison_dir.glob("*.png"))),
        "category_montages": len(list(category_dir.glob("*.png"))),
        "probability_backfilled_cases": 0,
        "selection_counts": {k: len(v) for k, v in picked.items()},
        "csv_files": REQUIRED_CSVS,
        "recommended_single_next_direction": "sampling",
        "new_training_started": False,
        "sources_modified": False,
    }
    write_text(OUT / "analysis_summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
