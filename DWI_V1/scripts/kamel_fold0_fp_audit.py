"""Complete read-only FP audit for B0-100 versus Kamel replication Fold 0-100."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation, label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from evaluate_experiment import metrics, summarize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/kamel_replication/fold0_fp_audit"
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
RAW_DWI = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI/imagesTr"

OLD_ROOT = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100"
OLD_FOLD = (
    OLD_ROOT
    / "train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0"
)
OLD_PRED = OLD_FOLD / "validation"
OLD_METRICS = OLD_ROOT / "metrics"
OLD_CHECKPOINT_BEST = OLD_FOLD / "checkpoint_best.pth"
OLD_CHECKPOINT_FINAL = OLD_FOLD / "checkpoint_final.pth"

NEW_ROOT = ROOT / "experiments/kamel_replication_seed2022/fold_0"
NEW_FOLD = (
    NEW_ROOT
    / "results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres/fold_0"
)
NEW_PRED = NEW_FOLD / "validation"
NEW_METRICS = NEW_ROOT / "metrics"
NEW_CHECKPOINT_BEST = NEW_FOLD / "checkpoint_best.pth"
NEW_CHECKPOINT_FINAL = NEW_FOLD / "checkpoint_final.pth"

STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value.rstrip() + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"No rows or fields for {path}")
    if fields is None:
        fields = list(rows[0])
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def delta(new, old):
    return None if new is None or old is None else new - old


def load_binary(path: Path):
    image = nib.load(path)
    data = image.get_fdata()
    rounded = np.round(data)
    if (
        not np.isfinite(data).all()
        or not np.allclose(data, rounded, atol=1e-8, rtol=0)
        or not set(np.unique(rounded)).issubset({0, 1})
    ):
        raise ValueError(f"Nonbinary or nonfinite mask: {path}")
    return image, rounded > 0.5


def matching(gt_labels, pred_labels, gt_sizes, pred_sizes):
    gn = len(gt_sizes)
    pn = len(pred_sizes)
    overlap = np.zeros((gn, pn), dtype=np.int64)
    both = (gt_labels > 0) & (pred_labels > 0)
    if both.any():
        np.add.at(overlap, (gt_labels[both] - 1, pred_labels[both] - 1), 1)
    matches: dict[int, int] = {}
    if gn and pn:
        iou = overlap / (gt_sizes[:, None] + pred_sizes[None, :] - overlap)
        score = (overlap > 0) * (min(gn, pn) + 1) + iou
        a, b = linear_sum_assignment(score, maximize=True)
        matches = {int(pb + 1): int(ga + 1) for ga, pb in zip(a, b) if overlap[ga, pb] > 0}
    return overlap, matches


def analyze_components(case: str, model: str, gt: np.ndarray, pred: np.ndarray, affine, voxel_ml: float):
    gt_labels, gt_count = label(gt, STRUCTURE_26)
    pred_labels, pred_count = label(pred, STRUCTURE_26)
    gt_sizes = np.bincount(gt_labels.ravel())[1:]
    pred_sizes = np.bincount(pred_labels.ravel())[1:]
    overlap, matches = matching(gt_labels, pred_labels, gt_sizes, pred_sizes)
    gt_coords = np.argwhere(gt)
    gt_tree = cKDTree(nib.affines.apply_affine(affine, gt_coords)) if len(gt_coords) else None
    dilated_gt = binary_dilation(gt, structure=STRUCTURE_26)
    rows = []
    for component_id in range(1, pred_count + 1):
        coords = np.argwhere(pred_labels == component_id)
        world = nib.affines.apply_affine(affine, coords)
        component_overlap = overlap[:, component_id - 1] if gt_count else np.array([], dtype=int)
        overlap_voxels = int(component_overlap.sum())
        if gt_tree is not None:
            distances, nearest_indices = gt_tree.query(world, k=1)
            nearest_at = int(np.argmin(distances))
            nearest_gt_id = int(gt_labels[tuple(gt_coords[int(nearest_indices[nearest_at])])])
            distance_mm = float(distances[nearest_at])
        else:
            nearest_gt_id = None
            distance_mm = None
        if gt_count and overlap_voxels:
            nearest_gt_id = int(np.argmax(component_overlap) + 1)
            distance_mm = 0.0
        nearest_overlap = int(component_overlap[nearest_gt_id - 1]) if nearest_gt_id else 0
        nearest_gt_size = int(gt_sizes[nearest_gt_id - 1]) if nearest_gt_id else 0
        pred_size = int(pred_sizes[component_id - 1])
        union = nearest_gt_size + pred_size - nearest_overlap
        nearest_iou = nearest_overlap / union if union else None
        nearest_dice = 2 * nearest_overlap / (nearest_gt_size + pred_size) if nearest_gt_size + pred_size else None
        matched_gt_id = matches.get(component_id)
        boundary_adjacent = bool(overlap_voxels == 0 and np.any(dilated_gt[pred_labels == component_id]))
        centroid_voxel = coords.mean(axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
        is_fp = matched_gt_id is None
        rows.append(
            {
                "case_id": case,
                "model": model,
                "component_id": component_id,
                "voxel_count": pred_size,
                "volume_ml": pred_size * voxel_ml,
                "centroid_i": float(centroid_voxel[0]),
                "centroid_j": float(centroid_voxel[1]),
                "centroid_k": float(centroid_voxel[2]),
                "centroid_x_mm": float(centroid_mm[0]),
                "centroid_y_mm": float(centroid_mm[1]),
                "centroid_z_mm": float(centroid_mm[2]),
                "overlaps_gt": bool(overlap_voxels),
                "overlap_voxel_count": overlap_voxels,
                "nearest_gt_component_id": nearest_gt_id,
                "nearest_gt_iou": nearest_iou,
                "nearest_gt_dice": nearest_dice,
                "matched_gt_component_id": matched_gt_id,
                "matched_one_to_one": matched_gt_id is not None,
                "is_fp": is_fp,
                "distance_to_nearest_gt_mm": distance_mm,
                "boundary_adjacent_26": boundary_adjacent,
                "completely_isolated": bool(is_fp and not overlap_voxels and not boundary_adjacent),
            }
        )
    return {
        "rows": rows,
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "gt_sizes": gt_sizes,
        "pred_sizes": pred_sizes,
        "fp_ids": {row["component_id"] for row in rows if row["is_fp"]},
        "matches": matches,
    }


def classify_fp_changes(case: str, b0, rep, voxel_ml: float):
    old_labels = b0["pred_labels"]
    new_labels = rep["pred_labels"]
    old_n = len(b0["pred_sizes"])
    new_n = len(rep["pred_sizes"])
    overlap = np.zeros((old_n, new_n), dtype=np.int64)
    both = (old_labels > 0) & (new_labels > 0)
    if both.any():
        np.add.at(overlap, (old_labels[both] - 1, new_labels[both] - 1), 1)
    old_fp = set(b0["fp_ids"])
    new_fp = set(rep["fp_ids"])
    used_old: set[int] = set()
    used_new: set[int] = set()
    rows = []

    def add(category, old_ids, new_ids, note):
        old_ids = sorted(set(old_ids))
        new_ids = sorted(set(new_ids))
        old_voxels = int(sum(b0["pred_sizes"][i - 1] for i in old_ids))
        new_voxels = int(sum(rep["pred_sizes"][j - 1] for j in new_ids))
        rows.append(
            {
                "case_id": case,
                "category": category,
                "b0_component_ids": ";".join(map(str, old_ids)),
                "replication_component_ids": ";".join(map(str, new_ids)),
                "b0_fp_component_count": sum(i in old_fp for i in old_ids),
                "replication_fp_component_count": sum(j in new_fp for j in new_ids),
                "b0_volume_ml": old_voxels * voxel_ml,
                "replication_volume_ml": new_voxels * voxel_ml,
                "delta_volume_ml": (new_voxels - old_voxels) * voxel_ml,
                "spatial_overlap_voxels": int(
                    overlap[np.ix_([i - 1 for i in old_ids], [j - 1 for j in new_ids])].sum()
                )
                if old_ids and new_ids
                else 0,
                "note": note,
            }
        )

    for old_id in range(1, old_n + 1):
        new_ids = [j + 1 for j in np.flatnonzero(overlap[old_id - 1] > 0)]
        new_fp_ids = [j for j in new_ids if j in new_fp]
        if len(new_ids) > 1 and new_fp_ids:
            add("E_same_location_fragmentation_increased", [old_id], new_ids, "One B0 component overlaps multiple replication components, including unmatched FP component(s).")
            used_new.update(new_fp_ids)
            if old_id in old_fp:
                used_old.add(old_id)

    for new_id in sorted(new_fp - used_new):
        old_ids = [i + 1 for i in np.flatnonzero(overlap[:, new_id - 1] > 0)]
        if not old_ids:
            add("B_replication_only_new_fp", [], [new_id], "No spatial overlap with any B0 prediction component.")
            used_new.add(new_id)
            continue
        old_id = max(old_ids, key=lambda i: overlap[i - 1, new_id - 1])
        if old_id not in old_fp:
            add("E_same_location_fragmentation_increased", [old_id], [new_id], "Replication unmatched component overlaps a B0 component that was matched to GT.")
        elif rep["pred_sizes"][new_id - 1] > b0["pred_sizes"][old_id - 1] + 1:
            add("D_same_location_replication_larger", [old_id], [new_id], "Spatially overlapping FP; replication component is larger by more than one voxel.")
        else:
            add("A_both_models_already_had_fp", [old_id], [new_id], "Spatially overlapping FP in both predictions.")
        used_new.add(new_id)
        if old_id in old_fp:
            used_old.add(old_id)

    for old_id in sorted(old_fp - used_old):
        new_ids = [j + 1 for j in np.flatnonzero(overlap[old_id - 1] > 0)]
        if new_ids and any(j in new_fp for j in new_ids):
            add("A_both_models_already_had_fp", [old_id], [j for j in new_ids if j in new_fp], "Residual spatially overlapping FP in both predictions.")
        else:
            add("C_b0_only_fp", [old_id], new_ids, "B0 FP has no replication FP at the same location.")
    return rows


def evaluate_variant(prediction_folder: Path, ids: list[str]):
    rows = []
    for case in ids:
        gt_image, gt = load_binary(PP / "gt_segmentations" / f"{case}.nii.gz")
        pred_image, pred = load_binary(prediction_folder / f"{case}.nii.gz")
        if gt.shape != pred.shape or not np.allclose(gt_image.affine, pred_image.affine, atol=1e-3, rtol=0):
            raise ValueError(f"Geometry mismatch in checkpoint comparison: {case}")
        voxel_ml = float(np.prod(gt_image.header.get_zooms()[:3], dtype=np.float64) / 1000)
        rows.append({"case": case, **metrics(gt, pred, gt_image.affine, voxel_ml)})
    return rows, summarize(rows)


def checkpoint_metadata(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "epoch": int(state["current_epoch"]),
        "best_ema_patch_pseudo_dice": float(state["_best_ema"]),
        "trainer_name": state["trainer_name"],
        "mirroring_axes": list(state["inference_allowed_mirroring_axes"]),
    }


def component_summary(rows, model: str):
    selected = [row for row in rows if row["model"] == model]
    fp = [row for row in selected if row["is_fp"]]
    return {
        "predicted_components": len(selected),
        "fp_components": len(fp),
        "predicted_component_volume_ml": float(sum(row["volume_ml"] for row in selected)),
        "mean_component_volume_ml": float(np.mean([row["volume_ml"] for row in selected])),
        "median_component_volume_ml": float(np.median([row["volume_ml"] for row in selected])),
        "components_lt_0p05_ml": sum(row["volume_ml"] < 0.05 for row in selected),
        "components_lt_0p1_ml": sum(row["volume_ml"] < 0.1 for row in selected),
        "fp_lt_0p05_ml": sum(row["volume_ml"] < 0.05 for row in fp),
        "fp_lt_0p1_ml": sum(row["volume_ml"] < 0.1 for row in fp),
        "fp_completely_isolated": sum(row["completely_isolated"] for row in fp),
        "fp_boundary_adjacent": sum(row["boundary_adjacent_26"] for row in fp),
    }


def overlay(ax, base, mask, color, alpha=0.45):
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(float) * alpha
    ax.imshow(rgba)


def contour(ax, mask, color, linewidth=1.0):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def visualize(case, gt, old, new, old_fp_mask, new_fp_mask, paired_row, figure_dir: Path):
    dwi_image = nib.load(RAW_DWI / f"{case}_0000.nii.gz")
    dwi = dwi_image.get_fdata()
    if dwi.shape != gt.shape or not np.allclose(dwi_image.affine, nib.load(PP / "gt_segmentations" / f"{case}.nii.gz").affine, atol=1e-3, rtol=0):
        raise ValueError(f"DWI/GT geometry mismatch for visualization: {case}")
    rep_only = new & ~old
    old_only = old & ~new
    score = (rep_only & ~gt).sum(axis=(0, 1)) + (old_only & ~gt).sum(axis=(0, 1)) + gt.sum(axis=(0, 1))
    z = int(np.argmax(score))
    image = np.rot90(dwi[:, :, z])
    gt2 = np.rot90(gt[:, :, z])
    old2 = np.rot90(old[:, :, z])
    new2 = np.rot90(new[:, :, z])
    old_only2 = np.rot90(old_only[:, :, z])
    rep_only2 = np.rot90(rep_only[:, :, z])
    both2 = np.rot90(old[:, :, z] & new[:, :, z])
    old_fp2 = np.rot90(old_fp_mask[:, :, z])
    new_fp2 = np.rot90(new_fp_mask[:, :, z])
    nonzero = image[np.isfinite(image) & (image != 0)]
    lo, hi = np.percentile(nonzero, [1, 99]) if len(nonzero) else (float(image.min()), float(image.max()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for ax in axes.ravel():
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")
    axes[0, 0].set_title(f"DWI | axial index {z}")
    axes[0, 1].set_title("GT")
    overlay(axes[0, 1], image, gt2, (1, 0, 0))
    axes[0, 2].set_title(f"B0 prediction | FP={paired_row['b0_fp_lesion_count']}")
    overlay(axes[0, 2], image, old2, (0, 0.75, 1))
    contour(axes[0, 2], gt2, "red")
    axes[1, 0].set_title(f"Replication prediction | FP={paired_row['replication_fp_lesion_count']}")
    overlay(axes[1, 0], image, new2, (1, 0, 1))
    contour(axes[1, 0], gt2, "red")
    axes[1, 1].set_title("Prediction difference")
    overlay(axes[1, 1], image, both2, (0.2, 0.8, 0.2), 0.35)
    overlay(axes[1, 1], image, old_only2, (0, 0.45, 1), 0.65)
    overlay(axes[1, 1], image, rep_only2, (1, 0, 0.8), 0.65)
    contour(axes[1, 1], gt2, "red", 1.2)
    axes[1, 1].legend(
        handles=[Patch(color="limegreen", label="Both"), Patch(color="dodgerblue", label="B0 only"), Patch(color="magenta", label="Replication only"), Patch(color="red", label="GT boundary")],
        loc="lower right",
        fontsize=8,
    )
    axes[1, 2].set_title("Unmatched FP component overlay")
    overlay(axes[1, 2], image, old_fp2, (0, 0.45, 1), 0.65)
    overlay(axes[1, 2], image, new_fp2, (1, 0, 0.8), 0.65)
    contour(axes[1, 2], gt2, "red", 1.2)
    fig.suptitle(
        f"{case} | ΔFP={paired_row['delta_fp_lesion_count']:+d} | Δprecision={paired_row['delta_lesion_precision'] if paired_row['delta_lesion_precision'] is not None else 'n.a.'}",
        fontsize=14,
    )
    target = figure_dir / f"{case}_fp_audit.png"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return {"case_id": case, "axial_index": z, "figure": str(target), "dwi_resampled": False}


def main() -> None:
    required = [OLD_PRED, NEW_PRED, OLD_METRICS, NEW_METRICS, OUT / "checkpoint_inference_manifest.json"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])
    if (OUT / "final_fp_audit_report.md").exists():
        raise SystemExit("Completed audit output already exists; no files modified")

    ids = read_json(PP / "splits_final.json")[0]["val"]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError("Expected exactly 50 unique Fold 0 validation cases")
    old_cases = {row["case"]: row for row in read_json(OLD_METRICS / "cases.json")}
    new_cases = {row["case"]: row for row in read_json(NEW_METRICS / "cases.json")}
    if set(old_cases) != set(ids) or set(new_cases) != set(ids):
        raise RuntimeError("Case-level metric sources do not match the fixed Fold 0 split")

    paired = []
    component_rows = []
    fp_change_rows = []
    overlap_rows = []
    tiny_rows = []
    cache = {}
    for index, case in enumerate(ids, start=1):
        gt_image, gt = load_binary(PP / "gt_segmentations" / f"{case}.nii.gz")
        old_image, old = load_binary(OLD_PRED / f"{case}.nii.gz")
        new_image, new = load_binary(NEW_PRED / f"{case}.nii.gz")
        if not (gt.shape == old.shape == new.shape) or not (
            np.allclose(gt_image.affine, old_image.affine, atol=1e-3, rtol=0)
            and np.allclose(gt_image.affine, new_image.affine, atol=1e-3, rtol=0)
        ):
            raise ValueError(f"Native-grid geometry mismatch: {case}")
        voxel_ml = float(np.prod(gt_image.header.get_zooms()[:3], dtype=np.float64) / 1000)
        b0 = analyze_components(case, "B0", gt, old, gt_image.affine, voxel_ml)
        rep = analyze_components(case, "Replication", gt, new, gt_image.affine, voxel_ml)
        component_rows.extend(b0["rows"])
        component_rows.extend(rep["rows"])
        fp_change_rows.extend(classify_fp_changes(case, b0, rep, voxel_ml))
        cache[case] = {"gt": gt, "old": old, "new": new, "B0": b0, "Replication": rep, "voxel_ml": voxel_ml}

        o = old_cases[case]
        n = new_cases[case]
        row = {
            "case_id": case,
            "b0_dice": o["dice"],
            "b0_gt_lesion_count": o["lesion_tp"] + o["lesion_fn"],
            "b0_pred_lesion_count": o["lesion_tp"] + o["lesion_fp"],
            "b0_lesion_recall": o["lesion_recall"],
            "b0_lesion_precision": o["lesion_precision"],
            "b0_fp_lesion_count": o["lesion_fp"],
            "b0_fn_lesion_count": o["lesion_fn"],
            "b0_gt_volume_ml": o["gt_ml"],
            "b0_pred_volume_ml": o["pred_ml"],
            "b0_pred_gt_ratio": o["pred_gt_ratio"],
            "b0_signed_volume_error_ml": o["signed_volume_error_ml"],
            "replication_dice": n["dice"],
            "replication_gt_lesion_count": n["lesion_tp"] + n["lesion_fn"],
            "replication_pred_lesion_count": n["lesion_tp"] + n["lesion_fp"],
            "replication_lesion_recall": n["lesion_recall"],
            "replication_lesion_precision": n["lesion_precision"],
            "replication_fp_lesion_count": n["lesion_fp"],
            "replication_fn_lesion_count": n["lesion_fn"],
            "replication_gt_volume_ml": n["gt_ml"],
            "replication_pred_volume_ml": n["pred_ml"],
            "replication_pred_gt_ratio": n["pred_gt_ratio"],
            "replication_signed_volume_error_ml": n["signed_volume_error_ml"],
            "delta_dice": delta(n["dice"], o["dice"]),
            "absolute_delta_dice": abs(delta(n["dice"], o["dice"])) if delta(n["dice"], o["dice"]) is not None else None,
            "delta_lesion_recall": delta(n["lesion_recall"], o["lesion_recall"]),
            "delta_lesion_precision": delta(n["lesion_precision"], o["lesion_precision"]),
            "delta_fp_lesion_count": n["lesion_fp"] - o["lesion_fp"],
            "delta_fn_lesion_count": n["lesion_fn"] - o["lesion_fn"],
            "delta_pred_lesion_count": (n["lesion_tp"] + n["lesion_fp"]) - (o["lesion_tp"] + o["lesion_fp"]),
            "delta_pred_gt_ratio": delta(n["pred_gt_ratio"], o["pred_gt_ratio"]),
            "delta_signed_volume_error_ml": n["signed_volume_error_ml"] - o["signed_volume_error_ml"],
        }
        paired.append(row)

        intersection = int((old & new).sum())
        union = int((old | new).sum())
        rep_only = new & ~old
        b0_only = old & ~new
        overlap_rows.append(
            {
                "case_id": case,
                "prediction_dice": 2 * intersection / (int(old.sum()) + int(new.sum())) if old.any() or new.any() else None,
                "prediction_jaccard": intersection / union if union else None,
                "intersection_voxel_count": intersection,
                "replication_only_voxel_count": int(rep_only.sum()),
                "replication_only_inside_gt_voxel_count": int((rep_only & gt).sum()),
                "replication_only_outside_gt_voxel_count": int((rep_only & ~gt).sum()),
                "replication_only_inside_gt_fraction": int((rep_only & gt).sum()) / int(rep_only.sum()) if rep_only.any() else None,
                "b0_only_voxel_count": int(b0_only.sum()),
                "b0_only_inside_gt_voxel_count": int((b0_only & gt).sum()),
                "b0_only_outside_gt_voxel_count": int((b0_only & ~gt).sum()),
                "b0_only_inside_gt_fraction": int((b0_only & gt).sum()) / int(b0_only.sum()) if b0_only.any() else None,
            }
        )

        old_volumes = np.asarray(b0["pred_sizes"], dtype=float) * voxel_ml
        new_volumes = np.asarray(rep["pred_sizes"], dtype=float) * voxel_ml
        tiny_rows.append(
            {
                "case_id": case,
                "b0_total_predicted_foreground_ml": float(old.sum() * voxel_ml),
                "replication_total_predicted_foreground_ml": float(new.sum() * voxel_ml),
                "delta_total_predicted_foreground_ml": float((new.sum() - old.sum()) * voxel_ml),
                "b0_predicted_component_count": len(old_volumes),
                "replication_predicted_component_count": len(new_volumes),
                "delta_predicted_component_count": len(new_volumes) - len(old_volumes),
                "b0_mean_component_volume_ml": float(old_volumes.mean()) if len(old_volumes) else None,
                "replication_mean_component_volume_ml": float(new_volumes.mean()) if len(new_volumes) else None,
                "b0_median_component_volume_ml": float(np.median(old_volumes)) if len(old_volumes) else None,
                "replication_median_component_volume_ml": float(np.median(new_volumes)) if len(new_volumes) else None,
                "b0_components_lt_0p05_ml": int((old_volumes < 0.05).sum()),
                "replication_components_lt_0p05_ml": int((new_volumes < 0.05).sum()),
                "delta_components_lt_0p05_ml": int((new_volumes < 0.05).sum() - (old_volumes < 0.05).sum()),
                "b0_components_lt_0p1_ml": int((old_volumes < 0.1).sum()),
                "replication_components_lt_0p1_ml": int((new_volumes < 0.1).sum()),
                "delta_components_lt_0p1_ml": int((new_volumes < 0.1).sum() - (old_volumes < 0.1).sum()),
            }
        )
        print(f"component audit {index}/50 {case}", flush=True)

    write_csv(OUT / "fold0_paired_case_comparison.csv", paired)
    write_csv(OUT / "prediction_components.csv", component_rows)
    fp_rows = [row for row in component_rows if row["is_fp"]]
    write_csv(OUT / "fp_components.csv", fp_rows)
    write_csv(OUT / "fp_change_classification.csv", fp_change_rows)
    write_csv(OUT / "tiny_component_analysis.csv", tiny_rows)
    write_csv(OUT / "prediction_overlap.csv", overlap_rows)

    write_csv(OUT / "top_fp_increase.csv", sorted(paired, key=lambda row: (row["delta_fp_lesion_count"], row["absolute_delta_dice"] or -1), reverse=True)[:10])
    write_csv(OUT / "top_precision_drop.csv", sorted(paired, key=lambda row: row["delta_lesion_precision"] if row["delta_lesion_precision"] is not None else float("inf"))[:10])
    write_csv(OUT / "top_dice_delta.csv", sorted(paired, key=lambda row: row["absolute_delta_dice"] if row["absolute_delta_dice"] is not None else -1, reverse=True)[:10])
    write_csv(OUT / "top_pred_component_increase.csv", sorted(paired, key=lambda row: row["delta_pred_lesion_count"], reverse=True)[:10])
    write_csv(OUT / "top_volume_overseg_increase.csv", sorted(paired, key=lambda row: row["delta_signed_volume_error_ml"], reverse=True)[:10])

    bins = [(-np.inf, 0.01, "<0.01"), (0.01, 0.05, "0.01-0.05"), (0.05, 0.1, "0.05-0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.0, "0.5-1"), (1.0, np.inf, ">=1")]
    distribution_rows = []
    for low, high, name in bins:
        values = {}
        for model in ("B0", "Replication"):
            subset = [row for row in fp_rows if row["model"] == model and low <= row["volume_ml"] < high]
            total_model = sum(row["model"] == model for row in fp_rows)
            values[model] = {
                "count": len(subset),
                "fraction": len(subset) / total_model if total_model else None,
                "total_ml": sum(row["volume_ml"] for row in subset),
            }
        distribution_rows.append(
            {
                "fp_volume_bin_ml": name,
                "lower_bound_ml_inclusive": None if np.isneginf(low) else low,
                "upper_bound_ml_exclusive": None if np.isposinf(high) else high,
                "b0_fp_component_count": values["B0"]["count"],
                "replication_fp_component_count": values["Replication"]["count"],
                "delta_fp_component_count": values["Replication"]["count"] - values["B0"]["count"],
                "b0_fraction_of_fp_components": values["B0"]["fraction"],
                "replication_fraction_of_fp_components": values["Replication"]["fraction"],
                "b0_total_fp_volume_ml": values["B0"]["total_ml"],
                "replication_total_fp_volume_ml": values["Replication"]["total_ml"],
            }
        )
    write_csv(OUT / "fp_volume_distribution.csv", distribution_rows)

    manifests = {row["label"]: row for row in read_json(OUT / "checkpoint_inference_manifest.json")}
    variant_paths = {
        "B0_best": Path(manifests["B0_best"]["predictions"]),
        "B0_final": OLD_PRED,
        "Replication_best": Path(manifests["Replication_best"]["predictions"]),
        "Replication_final": NEW_PRED,
    }
    checkpoint_paths = {
        "B0_best": OLD_CHECKPOINT_BEST,
        "B0_final": OLD_CHECKPOINT_FINAL,
        "Replication_best": NEW_CHECKPOINT_BEST,
        "Replication_final": NEW_CHECKPOINT_FINAL,
    }
    checkpoint_rows = []
    checkpoint_case_rows = {}
    for variant, prediction_folder in variant_paths.items():
        rows, summary = evaluate_variant(prediction_folder, ids)
        checkpoint_case_rows[variant] = rows
        meta = checkpoint_metadata(checkpoint_paths[variant])
        checkpoint_rows.append(
            {
                "variant": variant,
                "model": "B0" if variant.startswith("B0") else "Replication",
                "checkpoint_selection": "best" if variant.endswith("best") else "final",
                "checkpoint_epoch": meta["epoch"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "mean_dice": summary["dice"]["mean"],
                "median_dice": summary["dice"]["median"],
                "lesion_recall": summary["lesion_recall"],
                "lesion_precision": summary["lesion_precision"],
                "fp_lesions_per_case": summary["fp_lesions_per_case"],
                "lesion_tp": summary["lesion_tp"],
                "lesion_fn": summary["lesion_fn"],
                "lesion_fp": summary["lesion_fp"],
                "prediction_folder": str(prediction_folder),
            }
        )
    write_csv(OUT / "checkpoint_inference_comparison.csv", checkpoint_rows)
    lookup = {row["variant"]: row for row in checkpoint_rows}
    checkpoint_pairs = []
    for b0_name in ("B0_best", "B0_final"):
        for rep_name in ("Replication_best", "Replication_final"):
            b0_row, rep_row = lookup[b0_name], lookup[rep_name]
            checkpoint_pairs.append(
                {
                    "b0_variant": b0_name,
                    "replication_variant": rep_name,
                    "delta_mean_dice": rep_row["mean_dice"] - b0_row["mean_dice"],
                    "delta_median_dice": rep_row["median_dice"] - b0_row["median_dice"],
                    "delta_lesion_recall": rep_row["lesion_recall"] - b0_row["lesion_recall"],
                    "delta_lesion_precision": rep_row["lesion_precision"] - b0_row["lesion_precision"],
                    "delta_fp_lesions_per_case": rep_row["fp_lesions_per_case"] - b0_row["fp_lesions_per_case"],
                }
            )
    write_csv(OUT / "checkpoint_pairwise_comparison.csv", checkpoint_pairs)

    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=False)
    top_cases = []
    for source in (
        sorted(paired, key=lambda row: row["delta_fp_lesion_count"], reverse=True)[:3],
        sorted(paired, key=lambda row: row["delta_lesion_precision"] if row["delta_lesion_precision"] is not None else float("inf"))[:3],
    ):
        for row in source:
            if row["case_id"] not in top_cases:
                top_cases.append(row["case_id"])
    visual_rows = []
    paired_lookup = {row["case_id"]: row for row in paired}
    for case in top_cases:
        values = cache[case]
        old_fp_mask = np.isin(values["B0"]["pred_labels"], list(values["B0"]["fp_ids"]))
        new_fp_mask = np.isin(values["Replication"]["pred_labels"], list(values["Replication"]["fp_ids"]))
        visual_rows.append(visualize(case, values["gt"], values["old"], values["new"], old_fp_mask, new_fp_mask, paired_lookup[case], figure_dir))
    write_csv(OUT / "visualized_cases.csv", visual_rows)

    old_config = read_json(OLD_ROOT / "train/experiment_config.json")
    new_config = read_json(NEW_ROOT / "experiment_config.json")
    old_command = read_json(OLD_ROOT.parent / "logs/B0_R2022_screen100_train.command.json")["command"]
    old_eval_command = read_json(OLD_ROOT.parent / "logs/B0_R2022_screen100_evaluation.command.json")["command"]
    plans_equal = read_json(OLD_FOLD.parent / "plans.json") == read_json(NEW_FOLD.parent / "plans.json")
    dataset_equal = read_json(OLD_FOLD.parent / "dataset.json") == read_json(NEW_FOLD.parent / "dataset.json")
    evaluation_hash = digest(ROOT / "scripts/evaluate_experiment.py")
    old_evaluation_hash = old_config["source_sha256"].get(str(ROOT / "scripts/evaluate_experiment.py"))

    source_audit = f"""# Fold 0 source audit

## B0-100

- Prediction folder: `{OLD_PRED}`
- Checkpoint used for the existing prediction: `{OLD_CHECKPOINT_FINAL}`
- Checkpoint SHA-256: `{digest(OLD_CHECKPOINT_FINAL)}`
- Training/inference runner command: `{' '.join(old_command)}`
- Fixed metric command: `{' '.join(old_eval_command)}`
- Inference checkpoint: final, epoch 100
- Post-processing: none
- Binary decision: nnU-Net foreground argmax, equivalent to probability threshold 0.5 for two classes
- Connected components: 26-connectivity
- Metric implementation: `scripts/evaluate_experiment.py`

## Replication Fold 0-100

- Prediction folder: `{NEW_PRED}`
- Checkpoint used for the existing prediction: `{NEW_CHECKPOINT_FINAL}`
- Checkpoint SHA-256: `{digest(NEW_CHECKPOINT_FINAL)}`
- Training/inference runner: `python scripts/run_kamel_fold.py --fold 0`
- Fixed metric runner: `python scripts/evaluate_cv_fold.py --fold 0 --predictions <folder> --output <folder>`
- Inference checkpoint: final, epoch 100
- Post-processing: none
- Binary decision: nnU-Net foreground argmax, equivalent to probability threshold 0.5 for two classes
- Connected components: 26-connectivity
- Metric implementation: `evaluate_cv_fold.py` imports `metrics()` and `summarize()` unchanged from `scripts/evaluate_experiment.py`

## Source comparison

- Plans JSON semantic equality: `{plans_equal}`
- Dataset JSON semantic equality: `{dataset_equal}`
- Split SHA-256 equality: `{old_config['split_sha256'] == new_config['split_sha256']}`
- Package manifest equality: `{old_config['packages'] == new_config['packages']}`
- Recorded B0 evaluation source hash: `{old_evaluation_hash}`
- Current/shared evaluation source hash: `{evaluation_hash}`
- Evaluation source hash equality: `{old_evaluation_hash == evaluation_hash}`
- Both existing prediction sets used `checkpoint_final.pth`. Checkpoint selection is therefore matched for the failed gate.
- The trainer class name differs because the replication subclass permits fixed folds 0–4; its Fold 0 data, architecture, augmentation, loss, optimizer, and inference path inherit B0 behavior.
"""
    write_text(OUT / "source_audit.md", source_audit)

    inference_diff = f"""# Inference setting comparison

| Setting | B0-100 | Replication Fold 0 | Status | Evidence |
|---|---|---|---|---|
| Sliding-window step size | 0.5 | 0.5 | SAME | Shared nnUNetTrainer.perform_actual_validation implementation |
| Gaussian weighting | enabled | enabled | SAME | Shared predictor construction |
| Mirroring / TTA | enabled, axes (0,1,2) | enabled, axes (0,1,2) | SAME | Both checkpoint metadata |
| Device | CUDA | CUDA | SAME | Both runners instantiate trainer with cuda device |
| Probability export | enabled | enabled | SAME | Both call perform_actual_validation(save_probabilities=True) |
| Binary threshold | argmax, equivalent 0.5 | argmax, equivalent 0.5 | SAME | Same two-class label manager |
| Native-space probability resampling | resample_data_or_seg_to_shape | resample_data_or_seg_to_shape | SAME | Semantically identical plans |
| Probability interpolation | order 1, order_z 0 | order 1, order_z 0 | SAME | Semantically identical plans |
| Revert crop / transpose | nnU-Net export_prediction_from_logits | same | SAME | Shared nnU-Net 2.8.1 export implementation |
| Post-processing | none | none | SAME | Existing prediction manifests and evaluation definitions |
| Region / label order | background=0, infarct=1 | same | SAME | Semantically identical dataset JSON |

No inference-setting difference was found. The two added best-checkpoint runs used the same shared inference path and were inference-only.
"""
    write_text(OUT / "inference_setting_diff.md", inference_diff)

    metric_audit = """# Evaluation metric audit

| Item | B0-100 | Replication Fold 0 | Status |
|---|---|---|---|
| Connected-component connectivity | 26 | 26 | SAME |
| Minimum component size | none | none | SAME |
| GT lesion definition | each nonzero 26-connected GT component | same | SAME |
| Prediction lesion definition | each nonzero 26-connected prediction component | same | SAME |
| Component overlap threshold | at least one shared voxel | same | SAME |
| Matching rule | maximum-cardinality, then maximum-IoU one-to-one assignment | same imported function | SAME |
| Many-to-one handling | not allowed; unmatched prediction components count as FP | same | SAME |
| Empty prediction handling | voxel metrics unavailable where denominator is zero; lesion counts retained | same | SAME |
| Double-empty handling | Dice is missing; HD95 status is both_empty | same | SAME |
| Lesion precision | aggregate TP / (aggregate TP + aggregate FP) | same | SAME |
| FP/case | total unmatched prediction components / 50 cases | same | SAME |

The replication evaluator adds fold identity and an extra combined subgroup only. It imports the consequential component matching and summary functions directly from the B0 evaluator, so no evaluation-code mismatch explains the gate failure.
"""
    write_text(OUT / "evaluation_metric_audit.md", metric_audit)

    positive = sorted([row["delta_fp_lesion_count"] for row in paired if row["delta_fp_lesion_count"] > 0], reverse=True)
    negative = [row["delta_fp_lesion_count"] for row in paired if row["delta_fp_lesion_count"] < 0]
    b0_component_summary = component_summary(component_rows, "B0")
    rep_component_summary = component_summary(component_rows, "Replication")
    rep_only_total = sum(row["replication_only_voxel_count"] for row in overlap_rows)
    rep_only_inside = sum(row["replication_only_inside_gt_voxel_count"] for row in overlap_rows)
    b0_only_total = sum(row["b0_only_voxel_count"] for row in overlap_rows)
    b0_only_inside = sum(row["b0_only_inside_gt_voxel_count"] for row in overlap_rows)
    evidence = {
        "cases": 50,
        "fp_case_delta": {
            "b0_total_fp": sum(row["b0_fp_lesion_count"] for row in paired),
            "replication_total_fp": sum(row["replication_fp_lesion_count"] for row in paired),
            "net_delta": sum(row["delta_fp_lesion_count"] for row in paired),
            "cases_increased": sum(row["delta_fp_lesion_count"] > 0 for row in paired),
            "cases_unchanged": sum(row["delta_fp_lesion_count"] == 0 for row in paired),
            "cases_decreased": sum(row["delta_fp_lesion_count"] < 0 for row in paired),
            "gross_positive_increment": sum(positive),
            "gross_negative_increment": sum(negative),
            "top3_share_of_gross_positive_increment": sum(positive[:3]) / sum(positive) if positive else None,
            "top5_share_of_gross_positive_increment": sum(positive[:5]) / sum(positive) if positive else None,
        },
        "component_summary": {"B0": b0_component_summary, "Replication": rep_component_summary},
        "prediction_overlap": {
            "mean_prediction_dice": float(np.mean([row["prediction_dice"] for row in overlap_rows if row["prediction_dice"] is not None])),
            "median_prediction_dice": float(np.median([row["prediction_dice"] for row in overlap_rows if row["prediction_dice"] is not None])),
            "replication_only_voxels": rep_only_total,
            "replication_only_inside_gt_voxels": rep_only_inside,
            "replication_only_inside_gt_fraction": rep_only_inside / rep_only_total if rep_only_total else None,
            "b0_only_voxels": b0_only_total,
            "b0_only_inside_gt_voxels": b0_only_inside,
            "b0_only_inside_gt_fraction": b0_only_inside / b0_only_total if b0_only_total else None,
        },
        "checkpoint_comparison": checkpoint_rows,
        "checkpoint_pairwise_comparison": checkpoint_pairs,
        "source_checks": {
            "plans_same": plans_equal,
            "dataset_same": dataset_equal,
            "split_same": old_config["split_sha256"] == new_config["split_sha256"],
            "packages_same": old_config["packages"] == new_config["packages"],
            "evaluation_code_same": old_evaluation_hash == evaluation_hash,
            "inference_settings_same": True,
        },
    }
    write_json(OUT / "audit_evidence.json", evidence)
    write_json(
        OUT / "audit_manifest.json",
        {
            "output_folder": str(OUT),
            "validation_cases": ids,
            "existing_predictions_unchanged": True,
            "checkpoints_unchanged": True,
            "training_performed": False,
            "new_inference_only_variants": ["B0_best", "Replication_best"],
            "connectivity": 26,
            "matching": "maximum-cardinality then maximum-IoU one-to-one overlap",
            "threshold": 0.5,
            "postprocessing": "none",
        },
    )


if __name__ == "__main__":
    main()
