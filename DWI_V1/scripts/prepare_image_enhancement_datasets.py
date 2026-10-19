"""Select enhancement parameters by training-only QC and build isolated dual-channel datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.stats import spearmanr

from image_enhancement_common import (
    BASE_PP,
    BASE_PREDICTIONS,
    BASE_RAW,
    EXPERIMENTS,
    KNOWN_FAILURES,
    RING_OUTER_MM,
    SEED,
    SUITE,
    clahe_enhance,
    foreground_mask,
    foreground_zscore,
    gamma_enhance,
    intensity_scaled,
    local_ring,
)
from recall_common import ROOT, digest, now, save, verify_baseline


GAMMA_CANDIDATES = [0.7, 0.8, 1.2]
CLAHE_CANDIDATES = [0.005, 0.01, 0.02]
CLAHE_KERNEL = (16, 16)


def quantiles(values: np.ndarray) -> dict:
    points = [0, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 100]
    return {str(p): float(v) for p, v in zip(points, np.percentile(values, points))}


def robust_unit(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    lo, hi = np.percentile(values, [0.5, 99.5])
    out = np.zeros_like(image, dtype=np.float32)
    out[mask] = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0, 1)
    return out


def representation_metrics(
    case: str,
    name: str,
    image: np.ndarray,
    network_input: np.ndarray,
    brain: np.ndarray,
    lesion: np.ndarray,
    ring: np.ndarray,
    original_network: np.ndarray,
) -> dict:
    foreground = network_input[brain]
    lesion_values = network_input[lesion & brain]
    ring_values = network_input[ring]
    if not lesion_values.size or not ring_values.size:
        raise RuntimeError(f"Empty lesion/ring QC values: {case}")
    unit = robust_unit(image, brain)
    lesion_unit, ring_unit = unit[lesion & brain], unit[ring]
    var_sum = float(lesion_values.var() + ring_values.var())
    cnr = abs(float(lesion_values.mean() - ring_values.mean())) / np.sqrt(max(var_sum, 1e-12))
    reference_ring_std = float(original_network[ring].std())
    sample_step = max(1, int(brain.sum() // 200_000))
    a = original_network[brain][::sample_step]
    b = network_input[brain][::sample_step]
    correlation = float(spearmanr(a, b).statistic)
    minimum, maximum = float(foreground.min()), float(foreground.max())
    tolerance = max(1e-7, (maximum - minimum) * 1e-7)
    saturation = 100 * float(np.mean((foreground <= minimum + tolerance) | (foreground >= maximum - tolerance)))
    return {
        "case": case,
        "representation": name,
        "foreground_mean": float(foreground.mean()),
        "foreground_std": float(foreground.std()),
        "foreground_min": minimum,
        "foreground_max": maximum,
        "percentiles": quantiles(foreground),
        "saturation_percent": saturation,
        "lesion_mean": float(lesion_values.mean()),
        "surrounding_mean": float(ring_values.mean()),
        "lesion_surrounding_ratio_nonnegative": float((lesion_unit.mean() + 1e-6) / (ring_unit.mean() + 1e-6)),
        "cnr": float(cnr),
        "surrounding_std": float(ring_values.std()),
        "noise_amplification_vs_original": float(ring_values.std() / max(reference_ring_std, 1e-8)),
        "foreground_spearman_vs_original": correlation,
        "local_ring_outer_mm": RING_OUTER_MM,
    }


def summarize(rows: list[dict]) -> dict:
    keys = [
        "cnr", "lesion_surrounding_ratio_nonnegative", "saturation_percent",
        "noise_amplification_vs_original", "foreground_spearman_vs_original",
    ]
    return {
        key: {
            "median": float(np.median([r[key] for r in rows])),
            "mean": float(np.mean([r[key] for r in rows])),
            "min": float(np.min([r[key] for r in rows])),
            "max": float(np.max([r[key] for r in rows])),
        }
        for key in keys
    }


def load_case(case: str):
    image_obj = nib.load(BASE_RAW / "imagesTr" / f"{case}_0000.nii.gz")
    label_obj = nib.load(BASE_RAW / "labelsTr" / f"{case}.nii.gz")
    image = image_obj.get_fdata(dtype=np.float32)
    lesion = label_obj.get_fdata(dtype=np.float32) > 0.5
    if image.shape != lesion.shape or not np.allclose(image_obj.affine, label_obj.affine):
        raise RuntimeError(f"Geometry mismatch: {case}")
    brain = foreground_mask(image)
    spacing = tuple(float(x) for x in image_obj.header.get_zooms()[:3])
    ring = local_ring(lesion, brain, spacing)
    return image_obj, image, lesion, brain, ring, spacing


def select_cases() -> dict:
    split = json.loads((BASE_PP / "splits_final.json").read_text())[0]
    training = set(split["train"])
    validation = set(split["val"])
    if set(KNOWN_FAILURES) - validation:
        raise RuntimeError("Known failure cases are expected to remain validation-only QC cases")
    rows = []
    for case in sorted(training):
        label_obj = nib.load(BASE_RAW / "labelsTr" / f"{case}.nii.gz")
        lesion = label_obj.get_fdata(dtype=np.float32) > 0.5
        voxel_ml = float(np.prod(label_obj.header.get_zooms()[:3], dtype=np.float64) / 1000)
        _, components = cc_label(lesion, np.ones((3, 3, 3), dtype=np.uint8))
        rows.append({"case": case, "gt_ml": float(lesion.sum() * voxel_ml), "components": int(components)})
    rng = random.Random(SEED)
    small_pool = [r["case"] for r in rows if 1 <= r["gt_ml"] < 5]
    general_pool = [r["case"] for r in rows if r["gt_ml"] >= 5]
    multifocal_pool = [r["case"] for r in rows if r["components"] >= 2]
    small = sorted(rng.sample(small_pool, 5))
    general = sorted(rng.sample(general_pool, 5))
    multifocal = sorted(rng.sample(multifocal_pool, min(3, len(multifocal_pool))))
    tuning = list(dict.fromkeys([*general, *small, *multifocal]))
    return {
        "seed": SEED,
        "parameter_selection_partition": "fold0 training only",
        "general_cases": general,
        "small_1_5_ml_cases": small,
        "multifocal_cases": multifocal,
        "parameter_selection_cases": tuning,
        "known_failure_cases_validation_visualization_only": KNOWN_FAILURES,
        "all_qc_cases": list(dict.fromkeys([*tuning, *KNOWN_FAILURES])),
    }


def choose_parameters(selection: dict) -> tuple[dict, list[dict]]:
    cases = {}
    baseline_ranges = []
    for case in selection["parameter_selection_cases"]:
        loaded = load_case(case)
        cases[case] = loaded
        _, image, _, brain, _, _ = loaded
        z = foreground_zscore(image, brain)
        baseline_ranges.append(np.percentile(z[brain], [0.5, 99.5]))
    upper = float(np.median([r[1] for r in baseline_ranges]))
    scale = float(np.clip(np.round(2.5 / max(upper, 1e-8), 2), 0.75, 1.0))
    shift = 0.0

    rows = []
    for case, (_, image, lesion, brain, ring, _) in cases.items():
        original_network = foreground_zscore(image, brain)
        rows.append(representation_metrics(case, "original", image, original_network, brain, lesion, ring, original_network))
        for gamma in GAMMA_CANDIDATES:
            enhanced = gamma_enhance(image, brain, gamma)
            network = foreground_zscore(enhanced, brain)
            rows.append(representation_metrics(case, f"gamma_{gamma:g}", enhanced, network, brain, lesion, ring, original_network))
        for clip in CLAHE_CANDIDATES:
            enhanced = clahe_enhance(image, brain, clip, CLAHE_KERNEL)
            network = foreground_zscore(enhanced, brain)
            rows.append(representation_metrics(case, f"clahe_{clip:g}", enhanced, network, brain, lesion, ring, original_network))
        scaled = intensity_scaled(image, brain, scale, shift)
        rows.append(representation_metrics(case, f"intensity_scale_{scale:g}_{shift:g}", scaled, scaled, brain, lesion, ring, original_network))

    by_representation = {}
    for name in sorted({r["representation"] for r in rows}):
        by_representation[name] = summarize([r for r in rows if r["representation"] == name])
    original_cnr = by_representation["original"]["cnr"]["median"]
    original_contrast = by_representation["original"]["lesion_surrounding_ratio_nonnegative"]["median"]

    def balanced_contrast_score(name: str) -> float:
        """Balance standardized CNR and nonnegative lesion/ring contrast."""
        summary = by_representation[name]
        return (
            summary["cnr"]["median"] / max(original_cnr, 1e-8)
            * summary["lesion_surrounding_ratio_nonnegative"]["median"] / max(original_contrast, 1e-8)
        )

    def gamma_safe(name: str) -> bool:
        s = by_representation[name]
        return (
            s["foreground_spearman_vs_original"]["min"] >= 0.98
            and s["noise_amplification_vs_original"]["max"] <= 1.3
            and s["saturation_percent"]["max"] <= 2.0
        )

    gamma_names = [f"gamma_{x:g}" for x in GAMMA_CANDIDATES]
    safe_gamma = [n for n in gamma_names if gamma_safe(n)]
    selected_gamma_name = max(safe_gamma, key=balanced_contrast_score) if safe_gamma else None

    def clahe_safe(name: str) -> bool:
        s = by_representation[name]
        return (
            s["foreground_spearman_vs_original"]["min"] >= 0.80
            and s["noise_amplification_vs_original"]["max"] <= 1.5
            and s["saturation_percent"]["max"] <= 5.0
        )

    clahe_names = [f"clahe_{x:g}" for x in CLAHE_CANDIDATES]
    safe_clahe = [n for n in clahe_names if clahe_safe(n)]
    selected_clahe_name = max(safe_clahe, key=balanced_contrast_score) if safe_clahe else None

    gamma_value = float(selected_gamma_name.split("_")[1]) if selected_gamma_name else None
    clahe_value = float(selected_clahe_name.split("_")[1]) if selected_clahe_name else None
    params = {
        "selection_partition": "fold0 training only; validation segmentation performance not used",
        "local_background": f"GT-external foreground ring, Euclidean distance >0 and <= {RING_OUTER_MM:g} mm",
        "gamma": {
            "candidates": GAMMA_CANDIDATES,
            "selected": gamma_value,
            "anchor": "per-volume foreground P99.5; no clipping",
            "qc_rejected": selected_gamma_name is None,
            "selection_score": "(median CNR / original median CNR) × (median lesion/surround ratio / original median ratio)",
            "reason": "No candidate passed structure/noise/saturation guards" if selected_gamma_name is None else "highest balanced CNR-and-local-contrast score among guard-passing candidates",
        },
        "clahe": {
            "implementation": "skimage.exposure.equalize_adapthist",
            "candidates_clip_limit": CLAHE_CANDIDATES,
            "selected_clip_limit": clahe_value,
            "kernel_size_voxels": list(CLAHE_KERNEL),
            "kernel_size_nominal_mm_at_2mm": [32, 32],
            "slice_wise": True,
            "anatomical_axis": "native axial axis 2",
            "background": "excluded from statistics and reset to zero; processing limited to per-slice brain bounding box",
            "input_scaling": "per-volume foreground P0.5-P99.5 to [0,1] with clipping",
            "qc_rejected": selected_clahe_name is None,
            "selection_score": "(median CNR / original median CNR) × (median lesion/surround ratio / original median ratio)",
            "reason": "No candidate passed worst-case structure/noise/saturation guards" if selected_clahe_name is None else "highest balanced CNR-and-local-contrast score among candidates passing worst-case structure/noise/saturation guards",
        },
        "intensity_scale": {
            "input": "baseline per-volume filled-foreground z-score",
            "scale": scale,
            "shift": shift,
            "selection_rule": "target training-case median foreground P99.5 near 2.5; rounded to 0.01 and clipped to [0.75,1.0]",
            "observed_training_median_p99_5_before_scaling": upper,
            "network_preprocessing": "NoNormalization on channel 1; channel 0 retains baseline masked ZScoreNormalization",
            "qc_rejected": False,
            "interpretation": "non-destructive mild affine representation control; CNR is expected to remain invariant",
        },
        "candidate_summary": by_representation,
        "original_median_cnr": original_cnr,
    }
    return params, rows


def transform_for_experiment(name: str, image: np.ndarray, brain: np.ndarray, params: dict) -> np.ndarray:
    if name == "E1_gamma":
        return gamma_enhance(image, brain, params["gamma"]["selected"])
    if name == "E2_clahe":
        return clahe_enhance(image, brain, params["clahe"]["selected_clip_limit"], CLAHE_KERNEL)
    if name == "E3_intensity_scale":
        return intensity_scaled(image, brain, params["intensity_scale"]["scale"], params["intensity_scale"]["shift"])
    raise KeyError(name)


def qc_rows_for_frozen_parameters(selection: dict, params: dict) -> list[dict]:
    rows = []
    for case in selection["all_qc_cases"]:
        _, image, lesion, brain, ring, _ = load_case(case)
        original_network = foreground_zscore(image, brain)
        rows.append(representation_metrics(case, "original", image, original_network, brain, lesion, ring, original_network))
        for name in EXPERIMENTS:
            method = name.split("_", 1)[0]
            enhanced = transform_for_experiment(name, image, brain, params)
            network = enhanced if name == "E3_intensity_scale" else foreground_zscore(enhanced, brain)
            rows.append(representation_metrics(case, method, enhanced, network, brain, lesion, ring, original_network))
    return rows


def slice_choices(case: str, lesion: np.ndarray) -> list[tuple[str, int]]:
    choices = [("GT max area", int(np.argmax(lesion.sum((0, 1)))))]
    if case in KNOWN_FAILURES:
        pred = nib.load(BASE_PREDICTIONS / f"{case}.nii.gz").get_fdata(dtype=np.float32) > 0.5
        missed = lesion & ~pred
        choices.append(("B0-100 missed-lesion max", int(np.argmax(missed.sum((0, 1)))) if missed.any() else choices[0][1]))
    components, count = cc_label(lesion, np.ones((3, 3, 3), dtype=np.uint8))
    if count >= 2:
        sizes = np.bincount(components.ravel())[1:]
        positive = np.where(sizes > 0)[0]
        component = int(positive[np.argmin(sizes[positive])] + 1)
        small = components == component
        choices.append(("smallest multifocal component", int(np.argmax(small.sum((0, 1))))))
    unique = []
    for label, z in choices:
        if z not in [item[1] for item in unique]:
            unique.append((label, z))
    return unique


def make_figures(selection: dict, params: dict) -> None:
    figure_root = SUITE / "qc/figures"
    figure_root.mkdir(parents=True, exist_ok=False)
    for case in selection["all_qc_cases"]:
        _, image, lesion, brain, _, _ = load_case(case)
        gamma = transform_for_experiment("E1_gamma", image, brain, params)
        clahe = transform_for_experiment("E2_clahe", image, brain, params)
        scaled = transform_for_experiment("E3_intensity_scale", image, brain, params)
        slices = slice_choices(case, lesion)
        fig, axes = plt.subplots(len(slices), 5, figsize=(18, 4.2 * len(slices)), squeeze=False)
        panels = [("Original DWI", image), ("Gamma", gamma), ("CLAHE", clahe), ("Intensity-scaled", scaled)]
        for row, (reason, z) in enumerate(slices):
            for col, (title, array) in enumerate(panels):
                values = array[brain]
                lo, hi = np.percentile(values, [1, 99])
                axes[row, col].imshow(array[:, :, z].T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
                if lesion[:, :, z].any():
                    axes[row, col].contour(lesion[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.7)
                axes[row, col].set_title(f"{title} | z={z}")
                axes[row, col].axis("off")
            ax = axes[row, 4]
            values = image[brain]
            lo, hi = np.percentile(values, [1, 99])
            ax.imshow(image[:, :, z].T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            overlay = np.ma.masked_where(~lesion[:, :, z].T, lesion[:, :, z].T)
            ax.imshow(overlay, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.55)
            ax.set_title(f"GT overlay | z={z}")
            ax.axis("off")
            axes[row, 0].set_ylabel(reason)
        fig.suptitle(f"{case} | identical native axial slices; cyan/overlay = GT")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}.png", dpi=140)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    flattened = []
    for row in rows:
        item = dict(row)
        item["percentiles"] = json.dumps(item["percentiles"], sort_keys=True)
        flattened.append(item)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def create_raw_datasets(params: dict) -> dict:
    images = sorted((BASE_RAW / "imagesTr").glob("case_*_0000.nii.gz"))
    labels = sorted((BASE_RAW / "labelsTr").glob("case_*.nii.gz"))
    if len(images) != 250 or len(labels) != 250:
        raise RuntimeError("Dataset001 must contain 250 images and labels")
    save(SUITE / "qc/baseline_raw_manifest.json", {
        "dataset": "Dataset001_ISLES2022_DWI",
        "dataset_json_sha256": digest(BASE_RAW / "dataset.json"),
        "images": {p.name: digest(p) for p in images},
        "labels": {p.name: digest(p) for p in labels},
    })
    created = {}
    for name, spec in EXPERIMENTS.items():
        target = BASE_RAW.parent / spec["dataset_name"]
        if target.exists() or (BASE_PP.parent / spec["dataset_name"]).exists():
            raise RuntimeError(f"Dataset target already exists: {target}")
        (target / "imagesTr").mkdir(parents=True)
        (target / "labelsTr").mkdir()
        dataset = json.loads((BASE_RAW / "dataset.json").read_text())
        dataset["channel_names"] = {"0": "DWI", "1": "noNorm" if name == "E3_intensity_scale" else spec["id"] + "_enhanced_DWI"}
        with (target / "dataset.json").open("x", encoding="utf-8") as handle:
            json.dump(dataset, handle, indent=2)
        for index, image_path in enumerate(images, 1):
            case = image_path.name[:-12]
            image_obj = nib.load(image_path)
            image = image_obj.get_fdata(dtype=np.float32)
            brain = foreground_mask(image)
            enhanced = transform_for_experiment(name, image, brain, params)
            if not np.isfinite(enhanced).all() or np.any(enhanced[~brain] != 0):
                raise RuntimeError(f"Invalid enhanced image: {name} {case}")
            shutil.copy2(image_path, target / "imagesTr" / image_path.name)
            header = image_obj.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(enhanced.astype(np.float32), image_obj.affine, header), target / "imagesTr" / f"{case}_0001.nii.gz")
            source_label = BASE_RAW / "labelsTr" / f"{case}.nii.gz"
            shutil.copy2(source_label, target / "labelsTr" / source_label.name)
            if index % 25 == 0:
                print(f"{name}: generated {index}/250", flush=True)
        labels_equal = all(digest(p) == digest(target / "labelsTr" / p.name) for p in labels)
        originals_equal = all(digest(p) == digest(target / "imagesTr" / p.name) for p in images)
        if not labels_equal or not originals_equal:
            raise RuntimeError(f"Source preservation failed for {name}")
        created[name] = {"raw": str(target), "labels_byte_identical": labels_equal, "original_channel_byte_identical": originals_equal}
    return created


def preprocess_datasets() -> dict:
    os.environ["nnUNet_raw"] = str(BASE_RAW.parent)
    os.environ["nnUNet_preprocessed"] = str(BASE_PP.parent)
    os.environ.setdefault("nnUNet_results", str(ROOT / "outputs/models"))
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset

    missing_fingerprints = [
        spec["dataset_id"] for spec in EXPERIMENTS.values()
        if not (BASE_PP.parent / spec["dataset_name"] / "dataset_fingerprint.json").is_file()
    ]
    if missing_fingerprints:
        extract_fingerprints(missing_fingerprints, num_processes=8, check_dataset_integrity=True, clean=True, verbose=False, show_progress_bar=True)
    base_plans = json.loads((BASE_PP / "nnUNetPlans.json").read_text())
    results = {}
    for name, spec in EXPERIMENTS.items():
        target = BASE_PP.parent / spec["dataset_name"]
        fingerprint = json.loads((target / "dataset_fingerprint.json").read_text())
        raw_dataset_json = BASE_RAW.parent / spec["dataset_name"] / "dataset.json"
        target_dataset_json = target / "dataset.json"
        if target_dataset_json.exists():
            if json.loads(target_dataset_json.read_text()) != json.loads(raw_dataset_json.read_text()):
                raise RuntimeError(f"Existing processed dataset.json mismatch for {name}")
        else:
            shutil.copy2(raw_dataset_json, target_dataset_json)
        plans = json.loads(json.dumps(base_plans))
        plans["dataset_name"] = spec["dataset_name"]
        plans["foreground_intensity_properties_per_channel"] = fingerprint["foreground_intensity_properties_per_channel"]
        for configuration in plans["configurations"].values():
            configuration["normalization_schemes"] = spec["normalization_schemes"]
            configuration["use_mask_for_norm"] = spec["use_mask_for_norm"]
        plans_path = target / "nnUNetPlans.json"
        if plans_path.exists():
            if json.loads(plans_path.read_text()) != plans:
                raise RuntimeError(f"Existing controlled plans mismatch for {name}")
        else:
            with plans_path.open("x", encoding="utf-8") as handle:
                json.dump(plans, handle, indent=2)
        split_path = target / "splits_final.json"
        if split_path.exists():
            if digest(split_path) != digest(BASE_PP / "splits_final.json"):
                raise RuntimeError(f"Existing split mismatch for {name}")
        else:
            shutil.copy2(BASE_PP / "splits_final.json", split_path)
        data_folder = target / "nnUNetPlans_3d_fullres"
        complete_cases = len([p for p in data_folder.glob("case_*.b2nd") if not p.stem.endswith("_seg")]) if data_folder.exists() else 0
        if complete_cases != 250:
            if complete_cases != 0:
                raise RuntimeError(f"Partial preprocessing detected for {name}: {complete_cases}/250; manual audit required")
            preprocess_dataset(spec["dataset_id"], configurations=("3d_fullres",), num_processes=(8,), verbose=False, show_progress_bar=True)
        cfg = plans["configurations"]["3d_fullres"]
        if cfg["spacing"] != [2.0, 2.0, 2.0] or cfg["patch_size"] != [80, 96, 80] or cfg["batch_size"] != 8:
            raise RuntimeError(f"Controlled plan invariant failed for {name}")
        if digest(target / "splits_final.json") != digest(BASE_PP / "splits_final.json"):
            raise RuntimeError(f"Split changed for {name}")
        results[name] = {
            "preprocessed": str(target),
            "plans_sha256": digest(target / "nnUNetPlans.json"),
            "split_sha256": digest(target / "splits_final.json"),
            "preprocessed_cases": len([p for p in (target / "nnUNetPlans_3d_fullres").glob("case_*.b2nd") if not p.stem.endswith("_seg")]),
        }
    return results


def write_report(selection: dict, params: dict) -> None:
    lines = [
        "# Image enhancement QC and parameter freeze",
        "",
        "Parameter selection used fold-0 training cases only. Fixed validation failure cases were visualized only after parameters were frozen; no validation segmentation score selected any parameter.",
        "",
        f"Local surrounding tissue: GT-external, brain-foreground ring at Euclidean distance >0 to <= {RING_OUTER_MM:g} mm (three voxels at 2 mm).",
        "",
        "## E1 Gamma",
        "",
        f"Candidates: {GAMMA_CANDIDATES}; selected: {params['gamma']['selected']}. Transform is anchored at per-volume foreground P99.5 without clipping.",
        "",
        "## E2 CLAHE",
        "",
        f"Slice-wise native axial CLAHE using scikit-image; clip candidates {CLAHE_CANDIDATES}; selected {params['clahe']['selected_clip_limit']}; kernel {CLAHE_KERNEL} voxels (nominal 32×32 mm). Background is excluded and reset to zero.",
        "",
        "## E3 Mild intensity scale",
        "",
        f"Baseline per-volume foreground z-score followed by I_enhanced = {params['intensity_scale']['scale']} × I + {params['intensity_scale']['shift']}. Channel 1 uses NoNormalization so nnU-Net does not cancel the intended scale. This is a non-destructive affine representation control; its CNR is expected to remain invariant.",
        "",
        "## Selection and evidence",
        "",
        f"General training cases: {selection['general_cases']}",
        f"1–5 mL training cases: {selection['small_1_5_ml_cases']}",
        f"Multifocal training cases: {selection['multifocal_cases']}",
        f"Validation failure visualization only: {KNOWN_FAILURES}",
        "",
        "Candidate-level intensity, CNR, local contrast, saturation, noise amplification and structure-correlation statistics are in `parameter_selection.json`; per-case evidence is in CSV/JSON and fixed-slice figures.",
    ]
    (SUITE / "qc/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_existing_raw_datasets() -> None:
    baseline_images = sorted((BASE_RAW / "imagesTr").glob("case_*_0000.nii.gz"))
    baseline_labels = sorted((BASE_RAW / "labelsTr").glob("case_*.nii.gz"))
    for name, spec in EXPERIMENTS.items():
        target = BASE_RAW.parent / spec["dataset_name"]
        originals = sorted((target / "imagesTr").glob("case_*_0000.nii.gz"))
        enhanced = sorted((target / "imagesTr").glob("case_*_0001.nii.gz"))
        labels = sorted((target / "labelsTr").glob("case_*.nii.gz"))
        if len(originals) != 250 or len(enhanced) != 250 or len(labels) != 250:
            raise RuntimeError(f"Incomplete raw dataset during resume: {name}")
        if any(digest(a) != digest(b) for a, b in zip(baseline_images, originals)):
            raise RuntimeError(f"Original channel mismatch during resume: {name}")
        if any(digest(a) != digest(b) for a, b in zip(baseline_labels, labels)):
            raise RuntimeError(f"Label mismatch during resume: {name}")


def finish_preprocessing(params: dict, created: dict | None = None) -> None:
    preprocessed = preprocess_datasets()
    completion = {
        "time": now(),
        "status": "enhancement_qc_and_preprocessing_complete",
        "dataset_id_audit": {name: spec["dataset_id"] for name, spec in EXPERIMENTS.items()},
        "parameters": params,
        "raw_datasets": created or {name: {"raw": str(BASE_RAW.parent / spec["dataset_name"]), "resume_verified": True} for name, spec in EXPERIMENTS.items()},
        "preprocessed_datasets": preprocessed,
        "baseline_dataset_untouched": True,
        "validation_performance_used_for_parameter_selection": False,
    }
    save(SUITE / "qc/completion.json", completion)
    verify_baseline(SUITE / "baseline_after_preparation.json")
    print(json.dumps({"status": completion["status"], "datasets": completion["dataset_id_audit"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-preprocessing", action="store_true")
    args = parser.parse_args()
    if args.resume_preprocessing:
        if not SUITE.is_dir() or (SUITE / "qc/completion.json").exists():
            raise RuntimeError("Resume requires an incomplete existing suite")
        verify_existing_raw_datasets()
        params = json.loads((SUITE / "qc/parameter_selection.json").read_text())
        finish_preprocessing(params)
        return
    if SUITE.exists():
        raise RuntimeError("Enhancement suite already exists; refusing overwrite")
    for spec in EXPERIMENTS.values():
        if (BASE_RAW.parent / spec["dataset_name"]).exists() or (BASE_PP.parent / spec["dataset_name"]).exists():
            raise RuntimeError(f"Dataset ID {spec['dataset_id']} is already in use")
    SUITE.mkdir(parents=True)
    (SUITE / "qc").mkdir()
    for name in EXPERIMENTS:
        for folder in ["qc", "screen100", "promote300", "metrics", "predictions", "bootstrap", "case_analysis", "logs"]:
            (SUITE / name / folder).mkdir(parents=True, exist_ok=False)
    verify_baseline(SUITE / "baseline_before_preparation.json")
    selection = select_cases()
    save(SUITE / "qc/selection.json", selection)
    params, tuning_rows = choose_parameters(selection)
    if params["gamma"]["qc_rejected"] or params["clahe"]["qc_rejected"]:
        save(SUITE / "qc/parameter_selection.json", params)
        raise RuntimeError("E1 or E2 failed prespecified QC guards; evidence retained and training not started")
    save(SUITE / "qc/parameter_selection.json", params)
    save(SUITE / "qc/parameter_selection_per_case.json", tuning_rows)
    write_csv(SUITE / "qc/parameter_selection_per_case.csv", tuning_rows)
    frozen_rows = qc_rows_for_frozen_parameters(selection, params)
    save(SUITE / "qc/frozen_representation_stats.json", frozen_rows)
    write_csv(SUITE / "qc/frozen_representation_stats.csv", frozen_rows)
    make_figures(selection, params)
    write_report(selection, params)
    created = create_raw_datasets(params)
    finish_preprocessing(params, created)


if __name__ == "__main__":
    main()
