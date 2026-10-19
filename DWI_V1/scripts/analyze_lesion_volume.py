"""Compare lesion volumes for one binary prediction and ground-truth NIfTI pair."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class AnalysisError(RuntimeError):
    """Raised when the inputs are unsafe for volume comparison."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Prediction mask (.nii or .nii.gz)")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth mask (.nii or .nii.gz)")
    parser.add_argument("--case-id", help="Case identifier; defaults to the prediction filename")
    parser.add_argument("--case-type", default="unspecified", help="For example: validation or test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder; defaults to outputs/analysis/volume/<case-id>",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative geometry tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute geometry tolerance")
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("lesion_volume")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_binary_mask(path: Path, role: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, list[int]]:
    if not path.is_file():
        raise AnalysisError(f"{role} file does not exist: {path}")
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise AnalysisError(f"Unable to read {role} NIfTI: {path}: {exc}") from exc

    if data.ndim != 3:
        raise AnalysisError(f"{role} must be 3D, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise AnalysisError(f"{role} contains NaN or infinite values")

    unique_values = np.unique(data)
    if not np.isin(unique_values, (0, 1)).all():
        raise AnalysisError(
            f"{role} contains non-binary labels {unique_values.tolist()}; no remapping was performed"
        )
    labels = [int(value) for value in unique_values.tolist()]
    return image, data, labels


def validate_spacing(image: nib.spatialimages.SpatialImage, role: str) -> np.ndarray:
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    units = image.header.get_xyzt_units()[0]
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise AnalysisError(f"{role} has invalid voxel spacing: {spacing.tolist()}")
    if units != "mm":
        raise AnalysisError(f"{role} spatial units must be mm, got {units!r}")
    if not np.isfinite(image.affine).all():
        raise AnalysisError(f"{role} affine contains NaN or infinite values")
    return spacing


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def analyze(args: argparse.Namespace, logger: logging.Logger, output_dir: Path, case_id: str) -> dict:
    pred_path = args.pred.resolve()
    gt_path = args.gt.resolve()
    pred_image, pred_data, pred_labels = load_binary_mask(pred_path, "Prediction")
    gt_image, gt_data, gt_labels = load_binary_mask(gt_path, "Ground truth")
    pred_spacing = validate_spacing(pred_image, "Prediction")
    gt_spacing = validate_spacing(gt_image, "Ground truth")

    shape_match = pred_image.shape == gt_image.shape
    spacing_match = bool(np.allclose(pred_spacing, gt_spacing, rtol=args.rtol, atol=args.atol))
    affine_match = bool(np.allclose(pred_image.affine, gt_image.affine, rtol=args.rtol, atol=args.atol))

    logger.info("Case: %s (%s)", case_id, args.case_type)
    logger.info("Prediction: %s", pred_path)
    logger.info("Ground truth: %s", gt_path)
    logger.info("Geometry tolerance: rtol=%g, atol=%g", args.rtol, args.atol)
    logger.info(
        "Prediction shape=%s spacing=%s dtype=%s labels=%s finite=True",
        pred_image.shape,
        pred_spacing.tolist(),
        pred_image.get_data_dtype(),
        pred_labels,
    )
    logger.info(
        "Ground truth shape=%s spacing=%s dtype=%s labels=%s finite=True",
        gt_image.shape,
        gt_spacing.tolist(),
        gt_image.get_data_dtype(),
        gt_labels,
    )
    logger.info("Prediction affine:\n%s", np.array2string(pred_image.affine, precision=8))
    logger.info("Ground-truth affine:\n%s", np.array2string(gt_image.affine, precision=8))
    logger.info(
        "Physical space: shape_match=%s spacing_match=%s affine_match=%s",
        shape_match,
        spacing_match,
        affine_match,
    )

    if not shape_match:
        raise AnalysisError(
            f"Physical-space check failed: shapes differ ({pred_image.shape} vs {gt_image.shape})"
        )
    if not spacing_match:
        raise AnalysisError(
            f"Physical-space check failed: spacing differs ({pred_spacing.tolist()} vs {gt_spacing.tolist()})"
        )
    if not affine_match:
        max_diff = float(np.max(np.abs(pred_image.affine - gt_image.affine)))
        raise AnalysisError(f"Physical-space check failed: affines differ (max absolute difference {max_diff})")

    pred_voxel_volume_mm3 = float(np.prod(pred_spacing, dtype=np.float64))
    gt_voxel_volume_mm3 = float(np.prod(gt_spacing, dtype=np.float64))
    pred_voxels = int(np.count_nonzero(pred_data > 0))
    gt_voxels = int(np.count_nonzero(gt_data > 0))
    pred_volume_mm3 = pred_voxels * pred_voxel_volume_mm3
    gt_volume_mm3 = gt_voxels * gt_voxel_volume_mm3
    pred_volume_ml = pred_volume_mm3 / 1000.0
    gt_volume_ml = gt_volume_mm3 / 1000.0
    signed_mm3 = pred_volume_mm3 - gt_volume_mm3
    signed_ml = signed_mm3 / 1000.0
    absolute_mm3 = abs(signed_mm3)
    absolute_ml = absolute_mm3 / 1000.0

    if gt_volume_mm3 == 0:
        relative_percent = None
        signed_percent = None
    else:
        relative_percent = absolute_mm3 / gt_volume_mm3 * 100.0
        signed_percent = signed_mm3 / gt_volume_mm3 * 100.0

    if np.isclose(signed_mm3, 0.0, rtol=1e-12, atol=1e-9):
        interpretation = "Equal"
    elif signed_mm3 > 0:
        interpretation = "Overestimated"
    else:
        interpretation = "Underestimated"

    max_affine_difference = float(np.max(np.abs(pred_image.affine - gt_image.affine)))
    metrics = {
        "case_id": case_id,
        "case_type": args.case_type,
        "prediction_path": str(pred_path),
        "ground_truth_path": str(gt_path),
        "physical_space": {
            "shape_match": shape_match,
            "spacing_match": spacing_match,
            "affine_match": affine_match,
            "max_affine_absolute_difference": max_affine_difference,
            "tolerance": {"rtol": args.rtol, "atol": args.atol},
        },
        "prediction": {
            "shape": list(pred_image.shape),
            "spacing_mm": pred_spacing.tolist(),
            "affine": pred_image.affine.tolist(),
            "datatype": str(pred_image.get_data_dtype()),
            "labels": pred_labels,
            "finite": True,
            "lesion_voxels": pred_voxels,
            "voxel_volume_mm3": rounded(pred_voxel_volume_mm3),
            "volume_mm3": rounded(pred_volume_mm3),
            "volume_ml": rounded(pred_volume_ml),
        },
        "ground_truth": {
            "shape": list(gt_image.shape),
            "spacing_mm": gt_spacing.tolist(),
            "affine": gt_image.affine.tolist(),
            "datatype": str(gt_image.get_data_dtype()),
            "labels": gt_labels,
            "finite": True,
            "lesion_voxels": gt_voxels,
            "voxel_volume_mm3": rounded(gt_voxel_volume_mm3),
            "volume_mm3": rounded(gt_volume_mm3),
            "volume_ml": rounded(gt_volume_ml),
        },
        "error": {
            "absolute_mm3": rounded(absolute_mm3),
            "absolute_ml": rounded(absolute_ml),
            "relative_percent": rounded(relative_percent),
            "signed_mm3": rounded(signed_mm3),
            "signed_ml": rounded(signed_ml),
            "signed_percent": rounded(signed_percent),
            "interpretation": interpretation,
        },
        "status": "PASS",
    }

    output_path = output_dir / "volume_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Prediction: voxels=%d voxel_volume=%.3f mm3 volume=%.3f mm3 (%.3f mL)",
        pred_voxels,
        pred_voxel_volume_mm3,
        pred_volume_mm3,
        pred_volume_ml,
    )
    logger.info(
        "Ground truth: voxels=%d voxel_volume=%.3f mm3 volume=%.3f mm3 (%.3f mL)",
        gt_voxels,
        gt_voxel_volume_mm3,
        gt_volume_mm3,
        gt_volume_ml,
    )
    logger.info(
        "Error: absolute=%.3f mm3 (%.3f mL), relative=%s%%, signed=%.3f mm3 (%.3f mL), signed_percent=%s%%",
        absolute_mm3,
        absolute_ml,
        "N/A" if relative_percent is None else f"{relative_percent:.3f}",
        signed_mm3,
        signed_ml,
        "N/A" if signed_percent is None else f"{signed_percent:.3f}",
    )
    logger.info("Interpretation: %s", interpretation)
    logger.info("JSON output: %s", output_path)
    logger.info("Block 2 Status: PASS")
    return metrics


def main() -> int:
    args = parse_args()
    case_id = args.case_id or nifti_stem(args.pred)
    output_dir = (args.output_dir or ROOT / "outputs" / "analysis" / "volume" / case_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "volume_analysis.log")
    try:
        analyze(args, logger, output_dir, case_id)
    except AnalysisError as exc:
        logger.error("%s", exc)
        logger.error("Block 2 Status: FAIL")
        return 1
    except Exception:
        logger.exception("Unexpected failure")
        logger.error("Block 2 Status: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
