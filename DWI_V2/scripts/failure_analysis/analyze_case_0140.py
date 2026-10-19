"""Phase-1, read-only failure analysis for the case_0140 anchor.

The script reads V1 NIfTI/NPZ/B2ND artifacts and writes only below DWI_V2.
It deliberately contains no training entry point.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import blosc2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from scipy.ndimage import center_of_mass, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment


SCRIPT = Path(__file__).resolve()
V2_ROOT = SCRIPT.parents[2]
CONFIG_PATH = V2_ROOT / "configs" / "paths.yaml"
STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, value: Any) -> None:
    assert path.resolve().is_relative_to(V2_ROOT.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert path.resolve().is_relative_to(V2_ROOT.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_ready(row))


def load_config() -> tuple[dict[str, Any], Path, Path]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    v1 = Path(config["v1_root"]).resolve()
    v2 = Path(config["v2_root"]).resolve()
    if v2 != V2_ROOT.resolve():
        raise RuntimeError(f"Configured V2 root mismatch: {v2} != {V2_ROOT.resolve()}")
    if v1 == v2 or v1.is_relative_to(v2):
        raise RuntimeError("V1 and V2 roots are not safely separated")
    return config, v1, v2


def resolve_source(v1: Path, relative: str) -> Path:
    path = (v1 / relative).resolve()
    if not path.is_relative_to(v1):
        raise RuntimeError(f"Source escapes V1 root: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    return image, image.get_fdata(dtype=np.float32) > 0.5


def validate_geometry(reference: nib.Nifti1Image, candidate: nib.Nifti1Image, name: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference.affine, candidate.affine, atol=1e-3, rtol=0):
        raise RuntimeError(f"Geometry mismatch for {name}: {candidate.shape} vs {reference.shape}")


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def mask_metrics(gt: np.ndarray, pred: np.ndarray, voxel_ml: float) -> dict[str, Any]:
    tp = int(np.count_nonzero(gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    fp = int(np.count_nonzero(~gt & pred))
    gt_n = tp + fn
    pred_n = tp + fp
    dice = safe_ratio(2 * tp, gt_n + pred_n)
    gl, gn = label(gt, STRUCTURE_26)
    pl, pn = label(pred, STRUCTURE_26)
    gt_sizes = np.bincount(gl.ravel(), minlength=gn + 1)[1:]
    pred_sizes = np.bincount(pl.ravel(), minlength=pn + 1)[1:]
    matches = 0
    if gn and pn:
        overlap = np.zeros((gn, pn), dtype=float)
        both = (gl > 0) & (pl > 0)
        np.add.at(overlap, (gl[both] - 1, pl[both] - 1), 1)
        iou = overlap / (gt_sizes[:, None] + pred_sizes[None, :] - overlap)
        score = (overlap > 0) * (min(gn, pn) + 1) + iou
        assigned_gt, assigned_pred = linear_sum_assignment(score, maximize=True)
        matches = int((overlap[assigned_gt, assigned_pred] > 0).sum())
    return {
        "dice": dice,
        "gt_ml": gt_n * voxel_ml,
        "pred_ml": pred_n * voxel_ml,
        "pred_gt_ratio": safe_ratio(pred_n, gt_n),
        "tp_ml": tp * voxel_ml,
        "fn_ml": fn * voxel_ml,
        "fp_ml": fp * voxel_ml,
        "voxel_recall": safe_ratio(tp, gt_n),
        "voxel_precision": safe_ratio(tp, pred_n),
        "gt_components": int(gn),
        "pred_components": int(pn),
        "lesion_tp": matches,
        "lesion_fn": int(gn - matches),
        "lesion_fp": int(pn - matches),
        "lesion_recall": safe_ratio(matches, gn),
        "lesion_precision": safe_ratio(matches, pn),
    }


def axis_extents(mask: np.ndarray, image: nib.Nifti1Image) -> dict[str, Any]:
    coords = np.argwhere(mask)
    if not len(coords):
        return {"bbox_voxel": None, "rl_extent_mm": 0.0, "ap_extent_mm": 0.0, "is_extent_mm": 0.0}
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    raw_extents = (upper - lower + 1) * zooms
    axcodes = nib.aff2axcodes(image.affine)
    semantic: dict[str, float] = {}
    for index, code in enumerate(axcodes):
        if code in {"L", "R"}:
            semantic["rl_extent_mm"] = float(raw_extents[index])
        elif code in {"A", "P"}:
            semantic["ap_extent_mm"] = float(raw_extents[index])
        elif code in {"I", "S"}:
            semantic["is_extent_mm"] = float(raw_extents[index])
    return {
        "bbox_voxel": {"min": lower.tolist(), "max": upper.tolist()},
        "axis_codes": list(axcodes),
        **semantic,
    }


def gt_components(gt: np.ndarray, image: nib.Nifti1Image, voxel_ml: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, count = label(gt, STRUCTURE_26)
    rows: list[dict[str, Any]] = []
    for component in range(1, count + 1):
        mask = labels == component
        size = int(mask.sum())
        centroid = np.asarray(center_of_mass(mask), dtype=float)
        world = nib.affines.apply_affine(image.affine, centroid)
        extents = axis_extents(mask, image)
        rows.append(
            {
                "component": component,
                "voxels": size,
                "volume_ml": size * voxel_ml,
                "fraction_of_gt": size / int(gt.sum()),
                "centroid_voxel": ";".join(f"{v:.2f}" for v in centroid),
                "centroid_world_mm": ";".join(f"{v:.2f}" for v in world),
                "bbox_min": ";".join(str(v) for v in extents["bbox_voxel"]["min"]),
                "bbox_max": ";".join(str(v) for v in extents["bbox_voxel"]["max"]),
            }
        )
    rows.sort(key=lambda row: row["voxels"], reverse=True)
    largest_ratio = rows[0]["fraction_of_gt"] if rows else 0.0
    if count == 1:
        phenotype = "single_large_lesion"
    elif largest_ratio >= 0.9:
        phenotype = "dominant_large_component_with_satellites"
    elif largest_ratio >= 0.6:
        phenotype = "multifocal_with_dominant_component"
    else:
        phenotype = "multifocal_or_diffuse_distribution"
    summary = {
        "component_count": int(count),
        "largest_component_ml": rows[0]["volume_ml"] if rows else 0.0,
        "largest_component_ratio": largest_ratio,
        "morphology_pattern": phenotype,
        **axis_extents(gt, image),
    }
    return rows, summary


def region_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"n": 0}
    percentiles = np.percentile(finite, [5, 10, 25, 50, 75, 90, 95])
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p05": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
    }


def intensity_analysis(
    dwi: np.ndarray,
    gt: np.ndarray,
    baseline: np.ndarray,
    zooms: tuple[float, float, float],
    ring_mm: float,
    out_csv: Path,
    out_hist_csv: Path,
    out_figure: Path,
) -> dict[str, Any]:
    tp = gt & baseline
    fn = gt & ~baseline
    valid = np.isfinite(dwi) & (dwi != 0)
    distances = distance_transform_edt(~gt, sampling=zooms)
    ring = (~gt) & valid & (distances > 0) & (distances <= ring_mm)
    regions = {"TP": tp & valid, "FN": fn & valid, f"local_ring_{ring_mm:g}mm": ring}
    stats = {name: region_stats(dwi[mask]) for name, mask in regions.items()}
    ring_stats = stats[f"local_ring_{ring_mm:g}mm"]
    for name in ["TP", "FN"]:
        stats[name]["local_cnr_vs_ring"] = safe_ratio(
            stats[name]["mean"] - ring_stats["mean"], ring_stats["sd"]
        )
    pooled_sd = np.sqrt((stats["TP"]["sd"] ** 2 + stats["FN"]["sd"] ** 2) / 2)
    comparison = {
        "fn_minus_tp_mean": stats["FN"]["mean"] - stats["TP"]["mean"],
        "fn_to_tp_median_ratio": safe_ratio(stats["FN"]["median"], stats["TP"]["median"]),
        "standardized_mean_difference_fn_minus_tp": safe_ratio(
            stats["FN"]["mean"] - stats["TP"]["mean"], pooled_sd
        ),
        "interpretation_rule": "descriptive only; spatially correlated voxels are not treated as independent samples",
    }
    write_csv(out_csv, [{"region": name, **values} for name, values in stats.items()])

    sample = dwi[valid]
    low, high = np.percentile(sample, [0.5, 99.5])
    bins = np.linspace(low, high, 61)
    hist_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"TP": "#23a55a", "FN": "#d62728", f"local_ring_{ring_mm:g}mm": "#5875a4"}
    for name, mask in regions.items():
        counts, edges = np.histogram(dwi[mask], bins=bins)
        density = counts / max(counts.sum(), 1)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, density, label=name, color=colors[name], linewidth=2)
        for left, right, count, fraction in zip(edges[:-1], edges[1:], counts, density):
            hist_rows.append({"region": name, "bin_left": left, "bin_right": right, "count": int(count), "fraction": fraction})
    ax.set_title("case_0140 native DWI intensity distributions")
    ax.set_xlabel("DWI intensity (native units)")
    ax.set_ylabel("Fraction of region voxels")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=170)
    plt.close(fig)
    write_csv(out_hist_csv, hist_rows)
    return {"regions": stats, "comparison": comparison, "ring_definition_mm": ring_mm}


def load_probability(npz_path: Path, expected_shape: tuple[int, ...], pred: np.ndarray) -> tuple[np.ndarray, bool]:
    archive = np.load(npz_path)
    if "probabilities" not in archive:
        raise KeyError(f"No probabilities array in {npz_path}")
    raw = archive["probabilities"]
    if raw.ndim != 4 or raw.shape[0] < 2:
        raise ValueError(f"Unexpected probability shape: {raw.shape}")
    probability = raw[1].transpose(2, 1, 0)
    if probability.shape != expected_shape:
        raise ValueError(f"Probability geometry mismatch: {probability.shape} vs {expected_shape}")
    return probability, bool(np.array_equal(probability >= 0.5, pred))


def probability_analysis(
    probability: np.ndarray,
    gt: np.ndarray,
    baseline: np.ndarray,
    out_csv: Path,
    sweep_csv: Path,
    out_figure: Path,
) -> dict[str, Any]:
    tp = gt & baseline
    fn = gt & ~baseline
    rows = [{"region": "TP", **region_stats(probability[tp])}, {"region": "FN", **region_stats(probability[fn])}]
    write_csv(out_csv, rows)
    fn_values = probability[fn]
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, mask, color in [("TP", tp, "#23a55a"), ("FN", fn, "#d62728")]:
        ax.hist(probability[mask], bins=bins, density=True, histtype="step", linewidth=2.2, label=name, color=color)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.4, label="0.5 decision threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Lesion probability")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("B0-R2022 probability inside case_0140 GT")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=170)
    plt.close(fig)

    sweep_rows = []
    voxel_ml = 1.0
    for threshold in [0.5, 0.45, 0.4, 0.3, 0.2, 0.1]:
        metrics = mask_metrics(gt, probability >= threshold, voxel_ml)
        sweep_rows.append({"threshold": threshold, **{k: metrics[k] for k in ["dice", "pred_gt_ratio", "voxel_recall", "voxel_precision"]}})
    write_csv(sweep_csv, sweep_rows)
    near = float(np.mean((fn_values >= 0.4) & (fn_values < 0.5))) if fn_values.size else 0.0
    very_low = float(np.mean(fn_values < 0.1)) if fn_values.size else 0.0
    low = float(np.mean(fn_values < 0.25)) if fn_values.size else 0.0
    if near >= 0.5:
        interpretation = "threshold_or_calibration_dominant"
    elif very_low >= 0.5:
        interpretation = "representation_or_feature_learning_dominant"
    else:
        interpretation = "mixed_confidence_failure"
    return {
        "tp_probability": rows[0],
        "fn_probability": rows[1],
        "fn_fraction_0p4_to_0p5": near,
        "fn_fraction_below_0p25": low,
        "fn_fraction_below_0p1": very_low,
        "evidence_classification": interpretation,
        "threshold_sweep_is_diagnostic_only": True,
    }


def slice_analysis(gt: np.ndarray, pred: np.ndarray, voxel_ml: float) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    for z in range(gt.shape[2]):
        g = gt[:, :, z]
        p = pred[:, :, z]
        tp = int(np.count_nonzero(g & p))
        fn = int(np.count_nonzero(g & ~p))
        fp = int(np.count_nonzero(~g & p))
        denom = int(g.sum() + p.sum())
        rows.append(
            {
                "axial_slice": z,
                "gt_voxels": int(g.sum()),
                "pred_voxels": int(p.sum()),
                "tp_voxels": tp,
                "fn_voxels": fn,
                "fp_voxels": fp,
                "gt_ml": float(g.sum() * voxel_ml),
                "fn_ml": fn * voxel_ml,
                "slice_dice": None if denom == 0 else 2 * tp / denom,
            }
        )
    lesion_rows = [row for row in rows if row["gt_voxels"] > 0]
    gt_z = int(max(lesion_rows, key=lambda row: row["gt_voxels"])["axial_slice"])
    fn_z = int(max(lesion_rows, key=lambda row: row["fn_voxels"])["axial_slice"])
    worst_z = int(min(lesion_rows, key=lambda row: row["slice_dice"])["axial_slice"])
    occupied = np.array([row["axial_slice"] for row in lesion_rows], dtype=int)
    representatives = [
        ("inferior", int(occupied[0])),
        ("middle", int(occupied[len(occupied) // 2])),
        ("superior", int(occupied[-1])),
    ]
    candidates = [("GT max area", gt_z), ("FN max", fn_z), ("worst Dice", worst_z), *representatives]
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for reason, z in candidates:
        if z in used:
            selected = [(f"{old_reason}; {reason}" if old_z == z else old_reason, old_z) for old_reason, old_z in selected]
        else:
            selected.append((reason, z))
            used.add(z)
    if len(selected) < 5:
        order = sorted(lesion_rows, key=lambda row: (row["fn_voxels"], row["gt_voxels"]), reverse=True)
        for row in order:
            z = int(row["axial_slice"])
            if z not in used:
                selected.append(("additional high-FN", z))
                used.add(z)
            if len(selected) >= 5:
                break
    return rows, selected


def show_base(ax: plt.Axes, dwi_slice: np.ndarray, window: tuple[float, float], title: str) -> None:
    ax.imshow(dwi_slice.T, origin="lower", cmap="gray", vmin=window[0], vmax=window[1])
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def add_mask(ax: plt.Axes, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.55) -> None:
    rgba = np.zeros((*mask.T.shape, 4), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = mask.T.astype(float) * alpha
    ax.imshow(rgba, origin="lower")


def case_panel(
    dwi: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    selected: list[tuple[str, int]],
    path: Path,
) -> None:
    brain = dwi[np.isfinite(dwi) & (dwi != 0)]
    window = tuple(np.percentile(brain, [1, 99]).tolist())
    columns = ["Original DWI", "GT", "B0-R2022", "TP", "FN", "FP"]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(17, 3.1 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    for row, (reason, z) in enumerate(selected):
        masks = [None, gt[:, :, z], pred[:, :, z], (gt & pred)[:, :, z], (gt & ~pred)[:, :, z], (~gt & pred)[:, :, z]]
        colors = [None, (1.0, 0.75, 0.0), (0.0, 0.75, 1.0), (0.1, 0.8, 0.25), (1.0, 0.1, 0.1), (0.95, 0.1, 0.85)]
        for column, title in enumerate(columns):
            ax = axes[row, column]
            show_base(ax, dwi[:, :, z], window, f"{title} | z={z}")
            if masks[column] is not None:
                add_mask(ax, masks[column], colors[column])
            if column == 0:
                ax.text(-0.04, 0.5, reason, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("case_0140 fixed-slice failure panel (identical native orientation/window)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_panel(
    dwi: np.ndarray,
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: list[tuple[str, int]],
    path: Path,
) -> None:
    brain = dwi[np.isfinite(dwi) & (dwi != 0)]
    window = tuple(np.percentile(brain, [1, 99]).tolist())
    fig, axes = plt.subplots(len(predictions), len(selected), figsize=(3.1 * len(selected), 2.7 * len(predictions)))
    if len(predictions) == 1:
        axes = axes[None, :]
    for row, (name, pred) in enumerate(predictions.items()):
        for column, (reason, z) in enumerate(selected):
            ax = axes[row, column]
            show_base(ax, dwi[:, :, z], window, f"{reason} | z={z}" if row == 0 else "")
            add_mask(ax, pred[:, :, z], (1.0, 0.2, 0.15), alpha=0.48)
            if gt[:, :, z].any():
                ax.contour(gt[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.7)
            if column == 0:
                ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("All V1 models on fixed case_0140 slices (red=prediction, cyan=GT boundary)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def fn_overlap_panel(
    dwi: np.ndarray,
    gt: np.ndarray,
    miss_count: np.ndarray,
    persistent: np.ndarray,
    model_count: int,
    selected: list[tuple[str, int]],
    path: Path,
) -> None:
    brain = dwi[np.isfinite(dwi) & (dwi != 0)]
    window = tuple(np.percentile(brain, [1, 99]).tolist())
    fig, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6.2))
    for column, (reason, z) in enumerate(selected):
        for row in range(2):
            show_base(axes[row, column], dwi[:, :, z], window, f"{reason} | z={z}" if row == 0 else "")
        heat = np.ma.masked_where(~gt[:, :, z].T, miss_count[:, :, z].T)
        shown = axes[0, column].imshow(heat, origin="lower", cmap="magma", vmin=0, vmax=model_count, alpha=0.72)
        add_mask(axes[1, column], persistent[:, :, z], (1.0, 0.85, 0.0), alpha=0.75)
    axes[0, 0].text(-0.05, 0.5, "FN count across models", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.05, 0.5, "persistent FN", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.06, top=0.90, wspace=0.10, hspace=0.18)
    colorbar_axis = fig.add_axes([0.92, 0.56, 0.014, 0.28])
    fig.colorbar(shown, cax=colorbar_axis, label=f"Models missing voxel (0–{model_count})")
    fig.suptitle("case_0140 false-negative persistence across V1 interventions")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def preprocessing_analysis(
    dwi: np.ndarray,
    gt: np.ndarray,
    image: nib.Nifti1Image,
    data_path: Path,
    seg_path: Path,
    properties_path: Path,
    selected: list[tuple[str, int]],
    out_figure: Path,
) -> dict[str, Any]:
    data = np.asarray(blosc2.open(urlpath=str(data_path), mode="r")[:], dtype=np.float32)
    seg = np.asarray(blosc2.open(urlpath=str(seg_path), mode="r")[:])
    with properties_path.open("rb") as handle:
        properties = pickle.load(handle)
    shape_zyx = tuple(int(v) for v in properties["shape_before_cropping"])
    bbox = properties["bbox_used_for_cropping"]
    slices = tuple(slice(int(a), int(b)) for a, b in bbox)
    expected_crop = tuple(int(b - a) for a, b in bbox)
    full_data = np.full(shape_zyx, np.nan, dtype=np.float32)
    full_seg = np.zeros(shape_zyx, dtype=bool)
    full_data[slices] = data[0]
    full_seg[slices] = seg[0] > 0
    raw_zyx = dwi.transpose(2, 1, 0)
    gt_zyx = gt.transpose(2, 1, 0)
    if raw_zyx.shape != shape_zyx:
        raise RuntimeError(f"Raw/preprocessed shape mismatch: {raw_zyx.shape} vs {shape_zyx}")
    valid = np.isfinite(full_data) & (raw_zyx != 0)
    correlation = float(np.corrcoef(raw_zyx[valid], full_data[valid])[0, 1])
    label_equal = bool(np.array_equal(full_seg, gt_zyx))
    resampling_shape_change = tuple(data.shape[1:]) != expected_crop
    spacing_native_zyx = tuple(reversed(image.header.get_zooms()[:3]))
    spacing_preprocessed = tuple(float(v) for v in properties["spacing"])
    spacing_change = not np.allclose(spacing_native_zyx, spacing_preprocessed, atol=1e-5)

    prep_xyz = full_data.transpose(2, 1, 0)
    raw_brain = dwi[np.isfinite(dwi) & (dwi != 0)]
    raw_window = tuple(np.percentile(raw_brain, [1, 99]).tolist())
    prep_window = tuple(np.nanpercentile(prep_xyz, [1, 99]).tolist())
    fig, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6.2))
    for column, (reason, z) in enumerate(selected):
        show_base(axes[0, column], dwi[:, :, z], raw_window, f"{reason} | z={z}")
        add_mask(axes[0, column], gt[:, :, z], (1.0, 0.75, 0.0), 0.35)
        show_base(axes[1, column], prep_xyz[:, :, z], prep_window, "")
        add_mask(axes[1, column], full_seg.transpose(2, 1, 0)[:, :, z], (1.0, 0.75, 0.0), 0.35)
    axes[0, 0].text(-0.05, 0.5, "native DWI", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.05, 0.5, "nnUNet preprocessed", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("case_0140 native vs nnUNet-preprocessed DWI (separate fixed windows)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=160)
    plt.close(fig)
    return {
        "raw_shape_xyz": list(dwi.shape),
        "shape_before_cropping_zyx": list(shape_zyx),
        "bbox_used_for_cropping_zyx": bbox,
        "expected_crop_shape_zyx": list(expected_crop),
        "preprocessed_shape_czyx": list(data.shape),
        "native_spacing_zyx_mm": spacing_native_zyx,
        "preprocessed_spacing_zyx_mm": spacing_preprocessed,
        "resampling_shape_change": resampling_shape_change,
        "spacing_change": spacing_change,
        "raw_preprocessed_pearson_within_brain": correlation,
        "preprocessed_gt_matches_native_gt_exactly": label_equal,
        "causal_claim": "none; this is a descriptive artifact comparison",
    }


def save_mask(path: Path, mask: np.ndarray, reference: nib.Nifti1Image, dtype=np.uint8) -> None:
    assert path.resolve().is_relative_to(V2_ROOT.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(dtype), reference.affine, reference.header)
    image.set_data_dtype(dtype)
    nib.save(image, path)


def fold0_features(
    baseline_dir: Path,
    gt_dir: Path,
    anchor_case: str,
    anchor_image: nib.Nifti1Image,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pred_path in sorted(baseline_dir.glob("case_*.nii.gz")):
        case = pred_path.name.removesuffix(".nii.gz")
        gt_path = gt_dir / f"{case}.nii.gz"
        if not gt_path.is_file():
            continue
        gt_image, gt = load_mask(gt_path)
        pred_image, pred = load_mask(pred_path)
        validate_geometry(gt_image, pred_image, case)
        voxel_ml = abs(float(np.linalg.det(gt_image.affine[:3, :3]))) / 1000.0
        metrics = mask_metrics(gt, pred, voxel_ml)
        _, morphology = gt_components(gt, gt_image, voxel_ml)
        rows.append({"case": case, **metrics, **{k: morphology[k] for k in ["component_count", "largest_component_ratio", "rl_extent_mm", "ap_extent_mm", "is_extent_mm"]}})
    row_map = {row["case"]: row for row in rows}
    if anchor_case not in row_map:
        raise RuntimeError(f"Anchor {anchor_case} is absent from Fold 0")
    valid = [row for row in rows if row["gt_ml"] > 0]
    feature_names = ["log_gt_ml", "log_components", "largest_component_ratio", "ap_extent_mm", "is_extent_mm", "pred_gt_ratio", "voxel_recall"]
    matrix = []
    for row in valid:
        matrix.append([
            np.log1p(row["gt_ml"]),
            np.log1p(row["component_count"]),
            row["largest_component_ratio"],
            row["ap_extent_mm"],
            row["is_extent_mm"],
            min(row["pred_gt_ratio"] or 0.0, 2.0),
            row["voxel_recall"] or 0.0,
        ])
    matrix_np = np.asarray(matrix, dtype=float)
    median = np.median(matrix_np, axis=0)
    q25, q75 = np.percentile(matrix_np, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    anchor_index = next(i for i, row in enumerate(valid) if row["case"] == anchor_case)
    distances = np.sqrt(np.sum(((matrix_np - matrix_np[anchor_index]) / scale) ** 2, axis=1))
    for row, distance in zip(valid, distances):
        row["anchor_feature_distance"] = float(distance)
        row["strict_anchor_like"] = bool(
            row["case"] != anchor_case
            and row["gt_ml"] >= 50
            and (row["pred_gt_ratio"] or 0) <= 0.6
            and (row["voxel_recall"] or 0) <= 0.6
        )
        row["coverage_failure"] = bool(
            row["case"] != anchor_case
            and (row["pred_gt_ratio"] or 0) <= 0.6
            and (row["voxel_recall"] or 0) <= 0.6
        )
    neighbors = sorted([row for row in valid if row["case"] != anchor_case], key=lambda row: row["anchor_feature_distance"])
    selected_cases = {row["case"] for row in neighbors[:5]}
    selected_cases.update(row["case"] for row in valid if row.get("strict_anchor_like"))
    selected_cases.update(row["case"] for row in neighbors if row.get("coverage_failure") and row["anchor_feature_distance"] <= 5)
    cohort: list[dict[str, Any]] = []
    for rank, row in enumerate(neighbors, 1):
        if row["case"] in selected_cases:
            if row["strict_anchor_like"]:
                reason = "strict_large_coverage_failure_match"
            elif row["coverage_failure"]:
                reason = "coverage_failure_neighbor"
            else:
                reason = "nearest_multifeature_neighbor"
            cohort.append({"distance_rank": rank, "selection_reason": reason, **row})
    strict = [row["case"] for row in cohort if row["strict_anchor_like"]]
    summary = {
        "feature_names": feature_names,
        "distance_scaling": "Fold 0 median/IQR; Euclidean distance; Dice excluded",
        "strict_rule": "GT >=50 mL AND Pred/GT <=0.6 AND voxel recall <=0.6; anchor excluded",
        "strict_matches": strict,
        "strict_match_count": len(strict),
        "interpretation": "repeated_large_coverage_failure" if strict else "isolated_large_coverage_failure_in_fold0",
        "nearest_neighbor_count": 5,
    }
    return rows, cohort, summary


def build_report(
    path: Path,
    anchor: dict[str, Any],
    morphology: dict[str, Any],
    intensity: dict[str, Any],
    probability: dict[str, Any],
    preprocessing: dict[str, Any],
    model_rows: list[dict[str, Any]],
    fn_rows: list[dict[str, Any]],
    cohort_summary: dict[str, Any],
    cohort: list[dict[str, Any]],
    persistent_ml: float,
    persistent_fraction: float,
) -> None:
    best_dice = max(model_rows, key=lambda row: row["dice"])
    best_recall = max(model_rows, key=lambda row: row["voxel_recall"])
    baseline = next(row for row in model_rows if row["model"] == "B0-R2022")
    intensity_ratio = intensity["comparison"]["fn_to_tp_median_ratio"]
    strict = cohort_summary["strict_matches"]
    lines = [
        "# case_0140 failure analysis",
        "",
        "> Phase-1 descriptive analysis only. case_0140 was not used to tune a threshold or model, and no training was started.",
        "",
        "## One-page evidence summary",
        "",
        f"1. **Why did it fail?** B0-R2022 found the lesion but covered only {baseline['voxel_recall']:.1%} of GT; Pred/GT was {baseline['pred_gt_ratio']:.3f}. This is a large partial-coverage failure, not a whole-lesion detection miss.",
        f"2. **Main evidence.** GT volume was {anchor['gt_ml']:.3f} mL across {morphology['component_count']} components; the largest component held {morphology['largest_component_ratio']:.1%} of GT. B0-R2022 FN volume was {baseline['fn_ml']:.3f} mL.",
        f"3. **Intensity/contrast?** FN median DWI was {intensity_ratio:.3f}× TP median and FN local CNR was {intensity['regions']['FN']['local_cnr_vs_ring']:.3f} versus TP {intensity['regions']['TP']['local_cnr_vs_ring']:.3f}. This supports an intensity/representation contribution only to the degree stated in `intensity_statistics.csv`; it is not by itself causal.",
        f"4. **Morphology/coverage?** Yes: `{morphology['morphology_pattern']}` with AP extent {morphology['ap_extent_mm']:.1f} mm and IS extent {morphology['is_extent_mm']:.1f} mm, while every model detected at least part of the dominant lesion but left substantial GT uncovered.",
        f"5. **Threshold?** FN voxels near 0.5 (0.40–0.50) were {probability['fn_fraction_0p4_to_0p5']:.1%}; FN voxels below 0.10 were {probability['fn_fraction_below_0p1']:.1%}. Evidence class: `{probability['evidence_classification']}`. Threshold sweeps are diagnostic only, not a tuning recommendation.",
        f"6. **Best existing interventions.** Highest Dice was {best_dice['model']} ({best_dice['dice']:.3f}); highest voxel recall was {best_recall['model']} ({best_recall['voxel_recall']:.3f}). Full precision/FP trade-offs are in `model_comparison.csv`.",
        f"7. **Persistent missed region.** The intersection of FN across all {len(model_rows)} available models was {persistent_ml:.3f} mL ({persistent_fraction:.1%} of GT).",
        f"8. **Similar Fold 0 cases?** Strict large-coverage matches: {', '.join(strict) if strict else 'none'}. Conclusion: `{cohort_summary['interpretation']}`. Broader multifeature/coverage neighbors are retained without using Dice for selection.",
        "9. **Next interventions.** See `../../../docs/experiment_matrix.md`; proposals are capped at three and are explicitly linked to observed mechanisms.",
        "10. **Not supported now.** No evidence justifies blind loss, spacing, sampling, architecture, or multimodal changes without the proposal-specific validation gates.",
        "",
        "## Baseline coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["dice", "gt_ml", "pred_ml", "pred_gt_ratio", "tp_ml", "fn_ml", "fp_ml", "voxel_recall", "voxel_precision", "lesion_recall", "lesion_precision"]:
        value = baseline[key]
        lines.append(f"| {key} | {value:.6f} |")
    lines += [
        "",
        "## Morphology",
        "",
        f"- Pattern: `{morphology['morphology_pattern']}`.",
        f"- Components: {morphology['component_count']}; largest component {morphology['largest_component_ml']:.3f} mL ({morphology['largest_component_ratio']:.1%} of GT).",
        f"- Extent: RL {morphology['rl_extent_mm']:.1f} mm; AP {morphology['ap_extent_mm']:.1f} mm; IS {morphology['is_extent_mm']:.1f} mm.",
        "",
        "## Preprocessing check",
        "",
        f"- Native/preprocessed spacing change: {preprocessing['spacing_change']}; resampling shape change: {preprocessing['resampling_shape_change']}.",
        f"- Preprocessed GT exactly matches native GT after crop restoration: {preprocessing['preprocessed_gt_matches_native_gt_exactly']}.",
        f"- Raw/preprocessed intensity correlation inside the brain: {preprocessing['raw_preprocessed_pearson_within_brain']:.6f}.",
        "- These observations do not establish or exclude a preprocessing causal mechanism.",
        "",
        "## V1 model comparison",
        "",
        "| Model | Dice | Pred/GT | Voxel recall | Voxel precision | FN mL | FP mL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(model_rows, key=lambda item: item["dice"], reverse=True):
        lines.append(f"| {row['model']} | {row['dice']:.3f} | {row['pred_gt_ratio']:.3f} | {row['voxel_recall']:.3f} | {row['voxel_precision']:.3f} | {row['fn_ml']:.2f} | {row['fp_ml']:.2f} |")
    lines += [
        "",
        "## FN overlap",
        "",
        "`persistent_FN_all_models.nii.gz` is the GT region missed by every available model. Pairwise overlap and each model's rescue of B0-R2022 FN are in `fn_overlap.csv`.",
        "",
        "## Similar-case cohort",
        "",
        "Cohort ranking uses GT volume, component count, largest-component ratio, AP/IS extent, Pred/GT, and baseline voxel recall. Dice is deliberately excluded.",
        "",
        "| Case | Selection | Distance | GT mL | Components | Pred/GT | Recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cohort:
        lines.append(f"| {row['case']} | {row['selection_reason']} | {row['anchor_feature_distance']:.3f} | {row['gt_ml']:.2f} | {row['component_count']} | {row['pred_gt_ratio']:.3f} | {row['voxel_recall']:.3f} |")
    lines += [
        "",
        "## Evidence limitations",
        "",
        "- Fold 0 only; most intervention comparisons are single-seed screens and some have different epoch horizons.",
        "- Historical B0 seed is unconfirmed; P1 also changes spatial resolution.",
        "- Voxel intensity observations are spatially correlated and are reported descriptively.",
        "- No single anchor case can establish an intervention effect.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config, v1, _ = load_config()
    case = config["case_id"]
    sources = {name: resolve_source(v1, value) for name, value in config["sources"].items()}
    model_dirs = {name: resolve_source(v1, value) for name, value in config["models"].items()}
    analysis_dir = V2_ROOT / "analysis" / "failure_cases" / case
    metrics_dir = V2_ROOT / "outputs" / "metrics"
    figures_dir = V2_ROOT / "outputs" / "figures"
    maps_dir = V2_ROOT / "outputs" / "predictions" / case
    comparisons_dir = V2_ROOT / "outputs" / "comparisons"

    dwi_image = nib.load(sources["raw_dwi"])
    dwi = dwi_image.get_fdata(dtype=np.float32)
    gt_image, gt = load_mask(sources["raw_gt"])
    validate_geometry(dwi_image, gt_image, "raw GT")
    voxel_ml = abs(float(np.linalg.det(gt_image.affine[:3, :3]))) / 1000.0
    zooms = tuple(float(v) for v in gt_image.header.get_zooms()[:3])

    predictions: dict[str, np.ndarray] = {}
    model_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for name, directory in model_dirs.items():
        pred_path = directory / f"{case}.nii.gz"
        pred_image, pred = load_mask(pred_path)
        validate_geometry(gt_image, pred_image, name)
        predictions[name] = pred
        model_rows.append({"model": name, **mask_metrics(gt, pred, voxel_ml)})
        npz_path = directory / f"{case}.npz"
        source_manifest.append({"model": name, "prediction": str(pred_path), "probability": str(npz_path) if npz_path.is_file() else None})

    baseline_name = config["baseline_model"]
    baseline = predictions[baseline_name]
    baseline_metrics = next(row for row in model_rows if row["model"] == baseline_name)
    component_rows, morphology = gt_components(gt, gt_image, voxel_ml)
    morphology.update({"gt_ml": float(gt.sum() * voxel_ml)})
    slice_rows, selected_slices = slice_analysis(gt, baseline, voxel_ml)

    write_csv(analysis_dir / "gt_components.csv", component_rows)
    write_json(analysis_dir / "gt_morphology.json", morphology)
    write_csv(analysis_dir / "slice_metrics.csv", slice_rows)
    write_json(analysis_dir / "selected_slices.json", [{"reason": reason, "axial_slice": z} for reason, z in selected_slices])
    write_csv(comparisons_dir / "model_comparison.csv", model_rows)
    write_json(V2_ROOT / "configs" / "resolved_source_manifest.json", {"sources": {k: str(v) for k, v in sources.items()}, "models": source_manifest, "v1_read_only": True})

    intensity = intensity_analysis(
        dwi,
        gt,
        baseline,
        zooms,
        float(config["local_ring_mm"]),
        analysis_dir / "intensity_statistics.csv",
        analysis_dir / "intensity_histogram.csv",
        figures_dir / case / "intensity_histogram.png",
    )
    write_json(analysis_dir / "intensity_analysis.json", intensity)

    probability_path = model_dirs[baseline_name] / f"{case}.npz"
    probability, probability_matches_prediction = load_probability(probability_path, gt.shape, baseline)
    probability_result = probability_analysis(
        probability,
        gt,
        baseline,
        analysis_dir / "probability_statistics.csv",
        analysis_dir / "probability_threshold_diagnostic.csv",
        figures_dir / case / "probability_histogram.png",
    )
    probability_result["probability_map_available"] = True
    probability_result["threshold_0p5_exactly_matches_saved_segmentation"] = probability_matches_prediction
    write_json(analysis_dir / "probability_analysis.json", probability_result)

    preprocessing = preprocessing_analysis(
        dwi,
        gt,
        dwi_image,
        sources["preprocessed_data"],
        sources["preprocessed_seg"],
        sources["preprocessed_properties"],
        selected_slices,
        figures_dir / case / "native_vs_preprocessed.png",
    )
    write_json(analysis_dir / "preprocessing_comparison.json", preprocessing)

    baseline_fn = gt & ~baseline
    fn_masks = {name: gt & ~pred for name, pred in predictions.items()}
    persistent = np.logical_and.reduce(list(fn_masks.values()))
    miss_count = np.sum(np.stack(list(fn_masks.values()), axis=0), axis=0).astype(np.uint8)
    fn_rows: list[dict[str, Any]] = []
    for name, fn in fn_masks.items():
        intersection = baseline_fn & fn
        union = baseline_fn | fn
        rescued = baseline_fn & predictions[name]
        newly_missed = (gt & baseline) & ~predictions[name]
        fn_rows.append(
            {
                "model": name,
                "fn_ml": float(fn.sum() * voxel_ml),
                "b0_fn_intersection_ml": float(intersection.sum() * voxel_ml),
                "b0_fn_overlap_fraction": safe_ratio(int(intersection.sum()), int(baseline_fn.sum())),
                "b0_fn_jaccard": safe_ratio(int(intersection.sum()), int(union.sum())),
                "b0_fn_rescued_ml": float(rescued.sum() * voxel_ml),
                "b0_fn_rescued_fraction": safe_ratio(int(rescued.sum()), int(baseline_fn.sum())),
                "b0_tp_newly_missed_ml": float(newly_missed.sum() * voxel_ml),
            }
        )
    write_csv(comparisons_dir / "fn_overlap.csv", fn_rows)
    persistent_ml = float(persistent.sum() * voxel_ml)
    persistent_fraction = float(persistent.sum() / gt.sum())
    write_json(
        comparisons_dir / "persistent_fn_summary.json",
        {
            "models": list(predictions),
            "model_count": len(predictions),
            "persistent_fn_ml": persistent_ml,
            "persistent_fn_fraction_of_gt": persistent_fraction,
            "definition": "GT voxels missed by every available model listed in paths.yaml",
        },
    )

    save_mask(maps_dir / "B0-R2022_TP.nii.gz", gt & baseline, gt_image)
    save_mask(maps_dir / "B0-R2022_FN.nii.gz", baseline_fn, gt_image)
    save_mask(maps_dir / "B0-R2022_FP.nii.gz", ~gt & baseline, gt_image)
    save_mask(maps_dir / "persistent_FN_all_models.nii.gz", persistent, gt_image)
    save_mask(maps_dir / "FN_model_count.nii.gz", miss_count, gt_image, dtype=np.uint8)

    case_panel(dwi, gt, baseline, selected_slices, figures_dir / case / "fixed_case_panel.png")
    model_panel(dwi, gt, predictions, selected_slices, figures_dir / case / "all_models_fixed_slices.png")
    fn_overlap_panel(dwi, gt, miss_count, persistent, len(predictions), selected_slices, figures_dir / case / "fn_overlap_panel.png")

    fold_rows, cohort, cohort_summary = fold0_features(
        model_dirs[baseline_name], sources["fold0_gt_dir"], case, gt_image
    )
    write_csv(metrics_dir / "fold0_case_features.csv", fold_rows)
    write_csv(comparisons_dir / "case_0140_like_cohort.csv", cohort)
    write_json(comparisons_dir / "case_0140_like_cohort_summary.json", cohort_summary)

    summary = {
        "case": case,
        "baseline_model": baseline_name,
        "baseline": baseline_metrics,
        "morphology": morphology,
        "intensity": intensity,
        "probability": probability_result,
        "preprocessing": preprocessing,
        "persistent_fn": {"ml": persistent_ml, "fraction_of_gt": persistent_fraction},
        "similar_cohort": cohort_summary,
        "training_started": False,
    }
    write_json(analysis_dir / "analysis_summary.json", summary)
    build_report(
        analysis_dir / "report.md",
        baseline_metrics,
        morphology,
        intensity,
        probability_result,
        preprocessing,
        model_rows,
        fn_rows,
        cohort_summary,
        cohort,
        persistent_ml,
        persistent_fraction,
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
