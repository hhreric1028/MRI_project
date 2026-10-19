"""Shared definitions for the seed-2022 DWI enhancement controlled ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from recall_common import ROOT


SUITE = ROOT / "experiments/image_enhancement_batch_seed2022"
BASE_RAW = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"
BASE_PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
BASE_SCREEN = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100"
BASE_METRICS = BASE_SCREEN / "metrics"
BASE_PREDICTIONS = Path(json.loads((BASE_SCREEN / "train/completion.json").read_text())["predictions"])

KNOWN_FAILURES = ["case_0068", "case_0181", "case_0182", "case_0140", "case_0186"]
RING_OUTER_MM = 6.0
BOOTSTRAP_ITERATIONS = 10_000
SEED = 2022

EXPERIMENTS = {
    "E1_gamma": {
        "id": "E1",
        "dataset_id": 21,
        "dataset_name": "Dataset021_ISLES2022_DWI_GammaDual",
        "method": "Original DWI + gamma-enhanced DWI",
        "normalization_schemes": ["ZScoreNormalization", "ZScoreNormalization"],
        "use_mask_for_norm": [True, True],
    },
    "E2_clahe": {
        "id": "E2",
        "dataset_id": 22,
        "dataset_name": "Dataset022_ISLES2022_DWI_CLAHEDual",
        "method": "Original DWI + axial slice-wise CLAHE DWI",
        "normalization_schemes": ["ZScoreNormalization", "ZScoreNormalization"],
        "use_mask_for_norm": [True, True],
    },
    "E3_intensity_scale": {
        "id": "E3",
        "dataset_id": 23,
        "dataset_name": "Dataset023_ISLES2022_DWI_IntensityScaleDual",
        "method": "Original DWI + mildly scaled foreground-z-score DWI",
        "normalization_schemes": ["ZScoreNormalization", "NoNormalization"],
        "use_mask_for_norm": [True, False],
    },
}


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Baseline-compatible filled nonzero foreground."""
    return binary_fill_holes(np.asarray(image) != 0)


def foreground_zscore(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image, dtype=np.float32)
    values = np.asarray(image, dtype=np.float32)[mask]
    std = float(values.std())
    if not values.size or not np.isfinite(std) or std <= 1e-8:
        raise ValueError("Invalid foreground for z-score")
    result[mask] = (values - float(values.mean())) / std
    return result


def gamma_enhance(image: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    """Monotonic gamma transform anchored at each volume's foreground P99.5.

    Values are not clipped; P99.5 is merely the unit-scale anchor. This avoids
    introducing a ceiling while keeping the transform comparable across cases.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    result = np.zeros_like(image, dtype=np.float32)
    values = np.maximum(np.asarray(image, dtype=np.float32)[mask], 0)
    anchor = float(np.percentile(values, 99.5))
    if not np.isfinite(anchor) or anchor <= 1e-8:
        raise ValueError("Invalid gamma anchor")
    result[mask] = anchor * np.power(values / anchor, gamma)
    return result


def clahe_enhance(
    image: np.ndarray,
    mask: np.ndarray,
    clip_limit: float,
    kernel_size: tuple[int, int] = (16, 16),
    axis: int = 2,
) -> np.ndarray:
    """Apply 2D CLAHE slice-wise inside each slice's brain bounding box."""
    from skimage.exposure import equalize_adapthist

    if axis != 2:
        raise ValueError("This protocol freezes CLAHE to native axial axis 2")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    image = np.asarray(image, dtype=np.float32)
    result = np.zeros_like(image, dtype=np.float32)
    values = image[mask]
    lower, upper = np.percentile(values, [0.5, 99.5])
    if upper <= lower:
        raise ValueError("Invalid CLAHE robust range")
    scaled = np.clip((image - lower) / (upper - lower), 0, 1)
    for z in range(image.shape[2]):
        slice_mask = mask[:, :, z]
        if not slice_mask.any():
            continue
        coords = np.argwhere(slice_mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        crop = scaled[x0:x1, y0:y1, z]
        enhanced = equalize_adapthist(
            crop,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=256,
        ).astype(np.float32, copy=False)
        target = result[x0:x1, y0:y1, z]
        target_mask = slice_mask[x0:x1, y0:y1]
        target[target_mask] = enhanced[target_mask]
    return result


def intensity_scaled(image: np.ndarray, mask: np.ndarray, scale: float, shift: float = 0.0) -> np.ndarray:
    """Scale the baseline per-volume foreground z-score; keep background zero."""
    z = foreground_zscore(image, mask)
    result = np.zeros_like(z, dtype=np.float32)
    result[mask] = scale * z[mask] + shift
    return result


def local_ring(label: np.ndarray, brain: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """GT-external, brain-internal ring extending 0–6 mm from the lesion."""
    label = np.asarray(label, dtype=bool)
    distance = distance_transform_edt(~label, sampling=spacing)
    return (~label) & brain & (distance > 0) & (distance <= RING_OUTER_MM)


def volume_group(volume_ml: float) -> str:
    if volume_ml < 1:
        return "<1"
    if volume_ml < 5:
        return "1-5"
    if volume_ml < 20:
        return "5-20"
    if volume_ml < 50:
        return "20-50"
    return ">=50"


def experiment_dir(name: str) -> Path:
    if name not in EXPERIMENTS:
        raise KeyError(name)
    return SUITE / name

