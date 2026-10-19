"""Create the isolated original-DWI + exact-P3a-clipped two-channel raw dataset."""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import (  # noqa: E402
    E1_EXPERIMENT,
    E1_RAW,
    V1_PP,
    V1_RAW,
    digest,
    git_commit,
    now,
    save_json,
    source_hashes,
)


P3A_SOURCE = Path(__file__).resolve().parents[3] / "DWI_V1" / "scripts" / "prepare_p3a_dataset.py"
FAILURE_CASES = ["case_0140", "case_0068", "case_0181", "case_0186"]


def p3a_clip(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    foreground = binary_fill_holes(image != 0)
    values = image[foreground]
    if not values.size:
        raise ValueError("Empty foreground")
    lower, upper = np.percentile(values, [0.5, 99.5])
    clipped = image.copy()
    clipped[foreground] = np.clip(clipped[foreground], lower, upper)
    return clipped.astype(np.float32), foreground, float(lower), float(upper)


def choose_qc_cases(rows: list[dict], train_ids: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    rng = random.Random(2022)
    fixed = list(FAILURE_CASES)
    general_pool = [row["case"] for row in rows if row["case"] in train_ids and row["gt_ml"] >= 20 and row["case"] not in fixed]
    small_pool = [row["case"] for row in rows if row["case"] in train_ids and 1 <= row["gt_ml"] < 5 and row["case"] not in fixed]
    general = sorted(rng.sample(general_pool, 3))
    small = sorted(rng.sample(small_pool, 3))
    selected = list(dict.fromkeys([*fixed, *general, *small]))
    reasons = {
        case: (["fixed failure anchor"] if case in fixed else [])
        + (["general training case"] if case in general else [])
        + (["1-5 mL training case"] if case in small else [])
        for case in selected
    }
    return selected, reasons


def main() -> None:
    completion_path = E1_EXPERIMENT / "preprocessing_qc/raw_dataset_completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            raise RuntimeError("Existing E1 raw dataset completion is invalid")
        print(json.dumps({"status": "already_complete", "dataset": str(E1_RAW)}, indent=2))
        return
    if E1_RAW.exists():
        raise RuntimeError(f"Partial E1 raw dataset exists without completion marker: {E1_RAW}")

    images = sorted((V1_RAW / "imagesTr").glob("case_*_0000.nii.gz"))
    labels = sorted((V1_RAW / "labelsTr").glob("case_*.nii.gz"))
    if len(images) != 250 or len(labels) != 250:
        raise RuntimeError("Dataset001 does not contain the expected 250 image/label pairs")
    split = json.loads((V1_PP / "splits_final.json").read_text(encoding="utf-8"))[0]
    train_ids = set(split["train"])
    if len(train_ids) != 200 or len(split["val"]) != 50:
        raise RuntimeError("Fold 0 split is not 200/50")

    (E1_RAW / "imagesTr").mkdir(parents=True)
    (E1_RAW / "labelsTr").mkdir()
    figure_dir = E1_EXPERIMENT / "preprocessing_qc/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset_json = json.loads((V1_RAW / "dataset.json").read_text(encoding="utf-8"))
    dataset_json["channel_names"] = {"0": "DWI_original", "1": "DWI_P3a_clip"}
    save_json(E1_RAW / "dataset.json", dataset_json, exclusive=True)

    rows: list[dict] = []
    source_label_hashes: dict[str, str] = {}
    target_label_hashes: dict[str, str] = {}
    for index, image_path in enumerate(images, 1):
        case = image_path.name.removesuffix("_0000.nii.gz")
        label_path = V1_RAW / "labelsTr" / f"{case}.nii.gz"
        image_obj = nib.load(image_path)
        label_obj = nib.load(label_path)
        image = image_obj.get_fdata(dtype=np.float32)
        lesion = label_obj.get_fdata(dtype=np.float32) > 0.5
        if image.shape != lesion.shape or not np.allclose(image_obj.affine, label_obj.affine, atol=1e-5, rtol=0):
            raise RuntimeError(f"Image/label geometry mismatch: {case}")
        if not np.isfinite(image).all():
            raise RuntimeError(f"Nonfinite original DWI: {case}")
        clipped, foreground, lower, upper = p3a_clip(image)
        if not np.isfinite(clipped).all():
            raise RuntimeError(f"Nonfinite P3a channel: {case}")

        channel0 = E1_RAW / "imagesTr" / f"{case}_0000.nii.gz"
        channel1 = E1_RAW / "imagesTr" / f"{case}_0001.nii.gz"
        target_label = E1_RAW / "labelsTr" / f"{case}.nii.gz"
        shutil.copy2(image_path, channel0)
        header = image_obj.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(clipped, image_obj.affine, header), channel1)
        shutil.copy2(label_path, target_label)
        for written in [channel0, channel1, target_label]:
            check = nib.load(written)
            if check.shape != image_obj.shape or not np.allclose(check.affine, image_obj.affine, atol=1e-5, rtol=0):
                raise RuntimeError(f"Written geometry mismatch: {written}")

        voxel_ml = float(abs(np.linalg.det(label_obj.affine[:3, :3])) / 1000)
        rows.append(
            {
                "case": case,
                "fold0_role": "train" if case in train_ids else "validation",
                "shape": "x".join(map(str, image.shape)),
                "finite_original": bool(np.isfinite(image).all()),
                "finite_p3a": bool(np.isfinite(clipped).all()),
                "foreground_voxels": int(foreground.sum()),
                "p00_5": lower,
                "p99_5": upper,
                "gt_ml": float(lesion.sum() * voxel_ml),
                "channel0_sha256": digest(channel0),
                "channel1_sha256": digest(channel1),
                "label_sha256": digest(target_label),
            }
        )
        source_label_hashes[case] = digest(label_path)
        target_label_hashes[case] = digest(target_label)
        if index % 25 == 0:
            print(f"Prepared {index}/250", flush=True)

    selected, reasons = choose_qc_cases(rows, train_ids)
    for case in selected:
        original = nib.load(E1_RAW / "imagesTr" / f"{case}_0000.nii.gz").get_fdata(dtype=np.float32)
        clipped = nib.load(E1_RAW / "imagesTr" / f"{case}_0001.nii.gz").get_fdata(dtype=np.float32)
        lesion = nib.load(E1_RAW / "labelsTr" / f"{case}.nii.gz").get_fdata(dtype=np.float32) > 0.5
        foreground = binary_fill_holes(original != 0)
        z = int(np.argmax(lesion.sum(axis=(0, 1)))) if lesion.any() else int(np.argmax(foreground.sum(axis=(0, 1))))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        for ax, array, title in zip(axes, [original, clipped, original], ["Original channel 0", "Exact P3a clip channel 1", "GT"]):
            values = array[foreground]
            low, high = np.percentile(values, [1, 99])
            ax.imshow(array[:, :, z].T, origin="lower", cmap="gray", vmin=low, vmax=high)
            if title == "GT":
                rgba = np.zeros((*lesion[:, :, z].T.shape, 4), dtype=float)
                rgba[..., :3] = (1.0, 0.7, 0.0)
                rgba[..., 3] = lesion[:, :, z].T.astype(float) * 0.55
                ax.imshow(rgba, origin="lower")
            elif lesion[:, :, z].any():
                ax.contour(lesion[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.7)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{case} | z={z} | {', '.join(reasons[case])}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{case}.png", dpi=150)
        plt.close(fig)

    csv_path = E1_EXPERIMENT / "preprocessing_qc/per_case_geometry_and_intensity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    label_identical = source_label_hashes == target_label_hashes
    completion = {
        "status": "complete",
        "time": now(),
        "dataset_id": 101,
        "dataset_name": E1_RAW.name,
        "cases": len(rows),
        "fold0_train": len(train_ids),
        "fold0_validation": len(split["val"]),
        "channel_0": "byte-identical copy of Dataset001 original DWI",
        "channel_1": "per-volume binary_fill_holes(image != 0) foreground; P0.5-P99.5 clip; background unchanged; float32; nnUNet foreground z-score occurs later",
        "p3a_source": str(P3A_SOURCE),
        "p3a_source_sha256": digest(P3A_SOURCE),
        "labels_byte_identical": label_identical,
        "all_geometry_aligned": True,
        "all_finite": True,
        "qc_cases": selected,
        "qc_reasons": reasons,
        "git_commit": git_commit(),
        "source_hashes": source_hashes([V1_RAW / "dataset.json", V1_PP / "splits_final.json", P3A_SOURCE]),
    }
    save_json(completion_path, completion, exclusive=True)
    if not label_identical:
        raise RuntimeError("Copied labels are not byte-identical to Dataset001")
    print(json.dumps(completion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
