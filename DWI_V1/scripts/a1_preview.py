"""Create the mandatory pre-training visual QA for A1 intensity augmentation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import blosc2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import label

from a1_trainers import A1_INTENSITY_AUGMENTATION, B0_INTENSITY_AUGMENTATION, get_a1_intensity_preview_transform


ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
DATA = PP / "nnUNetPlans_3d_fullres"
OUTPUT = ROOT / "outputs/analysis/A1_augmentation_preview"
EXAMPLES_PER_CASE = 8


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def select_cases() -> list[dict]:
    training = json.loads((PP / "splits_final.json").read_text())[0]["train"]
    rows = []
    for case in training:
        image = nib.load(PP / "gt_segmentations" / f"{case}.nii.gz")
        mask = image.get_fdata() > 0.5
        voxel_ml = float(np.prod(image.header.get_zooms()[:3], dtype=np.float64) / 1000)
        _, count = label(mask, np.ones((3, 3, 3), dtype=np.uint8))
        rows.append({"case": case, "gt_ml": float(mask.sum() * voxel_ml), "gt_components": int(count)})
    large = max(rows, key=lambda row: row["gt_ml"])
    smalls = sorted((row for row in rows if 1 <= row["gt_ml"] < 5), key=lambda row: row["gt_ml"])
    small = smalls[len(smalls) // 2]
    multifocal = max(
        (row for row in rows if row["case"] not in {large["case"], small["case"]} and row["gt_ml"] >= 1),
        key=lambda row: (row["gt_components"], row["gt_ml"]),
    )
    return [
        {"selection_role": "large_lesion", **large},
        {"selection_role": "small_1_to_5_ml", **small},
        {"selection_role": "multifocal", **multifocal},
    ]


def apply_examples(data: np.ndarray, segmentation: np.ndarray, case_index: int):
    transform = get_a1_intensity_preview_transform()
    examples = []
    records = []
    for example_index in range(1, EXAMPLES_PER_CASE + 1):
        # Reject no-op draws so each displayed example actually demonstrates the A1 intensity pipeline.
        for attempt in range(1, 101):
            seed = 2022000 + case_index * 10000 + example_index * 100 + attempt
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            result = transform(
                image=torch.from_numpy(data.copy()), segmentation=torch.from_numpy(segmentation.copy())
            )
            augmented = result["image"].cpu().numpy()
            returned_seg = result["segmentation"].cpu().numpy()
            if not np.array_equal(augmented, data):
                break
        if np.array_equal(augmented, data):
            raise RuntimeError("Could not draw a non-identity A1 preview example")
        if not np.array_equal(returned_seg, segmentation):
            raise RuntimeError("A1 intensity transform changed mask geometry")
        if not np.isfinite(augmented).all():
            raise RuntimeError("A1 preview produced nonfinite intensities")
        mask = segmentation[0] > 0
        background = ~mask
        records.append(
            {
                "example": example_index,
                "rng_seed": seed,
                "draw_attempt": attempt,
                "image_changed": True,
                "mask_identical": True,
                "input_mean": float(data.mean()),
                "input_std": float(data.std()),
                "output_mean": float(augmented.mean()),
                "output_std": float(augmented.std()),
                "output_min": float(augmented.min()),
                "output_max": float(augmented.max()),
                "lesion_mean": float(augmented[0][mask].mean()),
                "background_mean": float(augmented[0][background].mean()),
                "lesion_background_difference": float(
                    augmented[0][mask].mean() - augmented[0][background].mean()
                ),
                "pearson_r_input_output": float(np.corrcoef(data.ravel(), augmented.ravel())[0, 1]),
            }
        )
        examples.append(augmented)
    return examples, records


def save_montage(case_row: dict, data: np.ndarray, segmentation: np.ndarray, examples: list[np.ndarray]) -> Path:
    mask = segmentation[0] > 0
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    original = data[0, z]
    values = original[np.isfinite(original)]
    vmin, vmax = np.percentile(values, [1, 99.5])
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    panels = [("Original normalized DWI", data), *[(f"A1 draw {i}", a) for i, a in enumerate(examples, 1)]]
    for axis, (title, volume) in zip(axes.ravel(), panels):
        axis.imshow(volume[0, z], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axis.contour(mask[z], levels=[0.5], colors=["#ff3b30"], linewidths=0.8, origin="lower")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        f"{case_row['selection_role']} | {case_row['case']} | GT {case_row['gt_ml']:.3f} mL | "
        f"components {case_row['gt_components']} | axial z={z}",
        fontsize=12,
    )
    path = OUTPUT / f"{case_row['selection_role']}__{case_row['case']}.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected = select_cases()
    all_stats = []
    visual_files = []
    geometry_checks = []
    for case_index, row in enumerate(selected):
        case = row["case"]
        data = np.asarray(blosc2.open(DATA / f"{case}.b2nd", mode="r")[:], dtype=np.float32)
        segmentation = np.asarray(blosc2.open(DATA / f"{case}_seg.b2nd", mode="r")[:])
        before = sha256(segmentation)
        examples, stats = apply_examples(data, segmentation, case_index)
        after = sha256(segmentation)
        geometry_checks.append({"case": case, "mask_sha256_before": before, "mask_sha256_after": after, "identical": before == after})
        for stat in stats:
            all_stats.append({"selection_role": row["selection_role"], "case": case, **stat})
        visual_files.append(str(save_montage(row, data, segmentation, examples)))

    with (OUTPUT / "augmentation_config.json").open("x", encoding="utf-8") as f:
        json.dump(
            {
                "purpose": "A1-100 pre-training visual QA; intensity transforms only",
                "b0_defaults": B0_INTENSITY_AUGMENTATION,
                "a1_settings": A1_INTENSITY_AUGMENTATION,
                "unchanged": [
                    "Gaussian blur sigma/probability",
                    "spatial transforms",
                    "low-resolution simulation",
                    "mirror transforms",
                    "mask/label/deep-supervision transforms",
                ],
                "preview_note": "Preview applies only the four changed A1 intensity branches; no spatial transform is used, so mask geometry must be byte-identical.",
                "examples_per_case": EXAMPLES_PER_CASE,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    for filename, rows in [("selected_cases.csv", selected), ("example_statistics.csv", all_stats)]:
        with (OUTPUT / filename).open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (OUTPUT / "sanity_check.json").open("x", encoding="utf-8") as f:
        json.dump(
            {
                "quantitative_status": "passed",
                "all_outputs_finite": True,
                "all_examples_changed_intensity": all(row["image_changed"] for row in all_stats),
                "all_masks_identical": all(row["identical"] for row in geometry_checks),
                "geometry_checks": geometry_checks,
                "visual_review_status": "pending_manual_review",
                "visual_files": visual_files,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps({"selected": selected, "visual_files": visual_files}, indent=2), flush=True)


if __name__ == "__main__":
    main()
