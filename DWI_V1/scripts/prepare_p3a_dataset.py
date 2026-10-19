"""Create Dataset013 with per-volume foreground P0.5-P99.5 clipped DWI and QC artifacts."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from recall_common import ROOT, digest, now, save


SOURCE = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"
TARGET = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset013_ISLES2022_DWI_ClipZScore"
EXPERIMENT = ROOT / "experiments/P3a_clip_zscore_seed2022"
QC = EXPERIMENT / "preprocessing_qc"
FAILURES = ["case_0068", "case_0182", "case_0181", "case_0140", "case_0212"]


def foreground_mask(image: np.ndarray) -> np.ndarray:
    return binary_fill_holes(image != 0)


def quantiles(values: np.ndarray) -> dict:
    points = [0, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 100]
    return {str(point): float(value) for point, value in zip(points, np.percentile(values, points))}


def main() -> None:
    if TARGET.exists() or EXPERIMENT.exists():
        raise RuntimeError("P3a target or experiment directory already exists; refusing overwrite")
    images = sorted((SOURCE / "imagesTr").glob("case_*_0000.nii.gz"))
    labels = sorted((SOURCE / "labelsTr").glob("case_*.nii.gz"))
    if len(images) != 250 or len(labels) != 250:
        raise RuntimeError("Dataset001 case count differs from 250")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [*images, *labels, SOURCE / "dataset.json"]}

    (TARGET / "imagesTr").mkdir(parents=True)
    (TARGET / "labelsTr").mkdir()
    (QC / "figures").mkdir(parents=True)
    dataset_json = json.loads((SOURCE / "dataset.json").read_text())
    save(TARGET / "dataset.json", dataset_json)

    rows = []
    arrays = {}
    for index, image_path in enumerate(images, 1):
        case = image_path.name[:-12]
        label_path = SOURCE / "labelsTr" / f"{case}.nii.gz"
        image_obj = nib.load(image_path)
        label_obj = nib.load(label_path)
        image = image_obj.get_fdata(dtype=np.float32)
        label = label_obj.get_fdata(dtype=np.float32) > 0.5
        if image.shape != label.shape or not np.allclose(image_obj.affine, label_obj.affine):
            raise RuntimeError(f"Geometry mismatch: {case}")
        if not np.isfinite(image).all():
            raise RuntimeError(f"Nonfinite DWI: {case}")
        mask = foreground_mask(image)
        values = image[mask]
        if not values.size:
            raise RuntimeError(f"Empty foreground: {case}")
        lower, upper = np.percentile(values, [0.5, 99.5])
        clipped = image.copy()
        clipped[mask] = np.clip(clipped[mask], lower, upper)
        clipped_values = clipped[mask]
        mean, std = float(clipped_values.mean()), float(clipped_values.std())
        if not np.isfinite(std) or std <= 1e-8:
            raise RuntimeError(f"Invalid clipped foreground std: {case}")
        normalized = np.zeros_like(clipped, dtype=np.float32)
        normalized[mask] = (clipped_values - mean) / std
        lesion_values = image[label & mask]
        lesion_clipped = clipped[label & mask]
        voxel_ml = float(np.prod(label_obj.header.get_zooms()[:3], dtype=np.float64) / 1000)
        row = {
            "case": case,
            "gt_ml": float(label.sum() * voxel_ml),
            "foreground_voxels": int(mask.sum()),
            "foreground_min": float(values.min()),
            "foreground_max": float(values.max()),
            "p00_5": float(lower),
            "p99_5": float(upper),
            "foreground_mean_original": float(values.mean()),
            "foreground_std_original": float(values.std()),
            "foreground_mean_clipped": mean,
            "foreground_std_clipped": std,
            "foreground_mean_normalized": float(normalized[mask].mean()),
            "foreground_std_normalized": float(normalized[mask].std()),
            "saturation_low_percent": float(100 * np.mean(values <= lower)),
            "saturation_high_percent": float(100 * np.mean(values >= upper)),
            "lesion_voxels": int((label & mask).sum()),
            "lesion_low_saturation_percent": float(100 * np.mean(lesion_values <= lower)) if lesion_values.size else None,
            "lesion_high_saturation_percent": float(100 * np.mean(lesion_values >= upper)) if lesion_values.size else None,
            "lesion_mean_original": float(lesion_values.mean()) if lesion_values.size else None,
            "lesion_mean_clipped": float(lesion_clipped.mean()) if lesion_clipped.size else None,
            "histogram_original": quantiles(values),
            "histogram_clipped": quantiles(clipped_values),
            "histogram_normalized": quantiles(normalized[mask]),
        }
        rows.append(row)
        target_image = TARGET / "imagesTr" / image_path.name
        header = image_obj.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(clipped.astype(np.float32), image_obj.affine, header), target_image)
        shutil.copy2(label_path, TARGET / "labelsTr" / label_path.name)
        arrays[case] = (image, clipped, normalized, label, mask)
        if index % 25 == 0:
            print(f"Prepared {index}/250", flush=True)

    by_case = {row["case"]: row for row in rows}
    rng = random.Random(2022)
    small_pool = [r["case"] for r in rows if 1 <= r["gt_ml"] < 5 and r["case"] not in FAILURES]
    general_pool = [r["case"] for r in rows if r["gt_ml"] >= 5 and r["case"] not in FAILURES]
    small = sorted(rng.sample(small_pool, 3))
    general = sorted(rng.sample(general_pool, 5))
    selected = list(dict.fromkeys([*general, *small, *FAILURES]))
    selection_reason = {
        case: (["general"] if case in general else [])
        + (["1-5 mL"] if case in small else [])
        + (["known failure"] if case in FAILURES else [])
        for case in selected
    }

    for case in selected:
        original, clipped, normalized, label, mask = arrays[case]
        z = int(np.argmax(label.sum((0, 1)))) if label.any() else int(np.argmax(mask.sum((0, 1))))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        panels = [(original, "Original DWI"), (clipped, "P0.5-P99.5 clipped"), (normalized, "Clipped + foreground z-score")]
        for axis, (array, title) in zip(axes, panels):
            values = array[mask]
            low, high = np.percentile(values, [1, 99])
            axis.imshow(array[:, :, z].T, origin="lower", cmap="gray", vmin=low, vmax=high)
            if label[:, :, z].any():
                axis.contour(label[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.8)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{case} | z={z} | {', '.join(selection_reason[case])}")
        fig.tight_layout()
        fig.savefig(QC / "figures" / f"{case}.png", dpi=150)
        plt.close(fig)

    simple_fields = [key for key, value in rows[0].items() if not isinstance(value, dict)]
    with (QC / "per_case_stats.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=simple_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save(QC / "per_case_stats.json", rows)
    save(QC / "selection.json", {
        "seed": 2022,
        "general_cases": general,
        "small_1_5_ml_cases": small,
        "known_failure_cases": FAILURES,
        "selected_unique_cases": selected,
        "selection_reason": selection_reason,
    })
    target_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sorted(TARGET.rglob("*")) if path.is_file()}
    label_equal = all(
        digest(SOURCE / "labelsTr" / label.name) == digest(TARGET / "labelsTr" / label.name)
        for label in labels
    )
    normalized_ok = all(abs(row["foreground_mean_normalized"]) < 1e-5 and abs(row["foreground_std_normalized"] - 1) < 1e-5 for row in rows)
    completion = {
        "time": now(),
        "status": "raw_clip_dataset_and_qc_complete",
        "source_dataset": "Dataset001_ISLES2022_DWI",
        "target_dataset": "Dataset013_ISLES2022_DWI_ClipZScore",
        "cases": len(rows),
        "method": "per individual volume; binary-filled nonzero foreground; P0.5-P99.5 clip; standard foreground z-score follows in nnUNet preprocessing",
        "labels_byte_identical": label_equal,
        "normalized_qc_finite_unit_statistics": normalized_ok,
        "source_hashes": source_hashes,
        "target_hashes": target_hashes,
    }
    save(QC / "completion.json", completion)
    if not label_equal or not normalized_ok:
        raise RuntimeError("P3a QC invariant failed")
    print(json.dumps({key: completion[key] for key in ["status", "target_dataset", "cases", "labels_byte_identical", "normalized_qc_finite_unit_statistics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
