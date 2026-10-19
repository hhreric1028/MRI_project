"""Analyze 3D connected lesion components in one prediction/ground-truth pair."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label


ROOT = Path(__file__).resolve().parents[1]
CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


class AnalysisError(RuntimeError):
    """Raised when inputs cannot be compared safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Prediction mask (.nii or .nii.gz)")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth mask (.nii or .nii.gz)")
    parser.add_argument("--connectivity", type=int, choices=sorted(CONNECTIVITY_RANK), default=26)
    parser.add_argument("--case-id", help="Case identifier; defaults to the prediction filename")
    parser.add_argument("--case-type", default="unspecified", help="For example: validation or test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder; defaults to outputs/analysis/connected_components/<case-id>",
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


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("connected_lesions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
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
    values = np.unique(data)
    if not np.isin(values, (0, 1)).all():
        raise AnalysisError(f"{role} contains non-binary labels {values.tolist()}; no remapping was performed")
    return image, data, [int(value) for value in values.tolist()]


def get_spacing(image: nib.spatialimages.SpatialImage, role: str) -> np.ndarray:
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise AnalysisError(f"{role} has invalid spacing: {spacing.tolist()}")
    units = image.header.get_xyzt_units()[0]
    if units != "mm":
        raise AnalysisError(f"{role} spatial units must be mm, got {units!r}")
    if not np.isfinite(image.affine).all():
        raise AnalysisError(f"{role} affine contains NaN or infinite values")
    return spacing


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def component_metrics(mask: np.ndarray, voxel_volume_mm3: float, structure: np.ndarray) -> dict:
    labeled, component_count = label(mask, structure=structure)
    counts = np.bincount(labeled.ravel(), minlength=component_count + 1)[1:]
    total_voxels = int(np.count_nonzero(mask))
    total_volume_mm3 = total_voxels * voxel_volume_mm3

    ordered = sorted(
        ((label_id, int(count)) for label_id, count in enumerate(counts, start=1)),
        key=lambda item: (-item[1], item[0]),
    )
    components = []
    for rank, (label_id, voxel_count) in enumerate(ordered, start=1):
        volume_mm3 = voxel_count * voxel_volume_mm3
        fraction_percent = volume_mm3 / total_volume_mm3 * 100.0 if total_volume_mm3 else None
        components.append(
            {
                "rank": rank,
                "label_id": label_id,
                "voxel_count": voxel_count,
                "volume_mm3": rounded(volume_mm3),
                "volume_ml": rounded(volume_mm3 / 1000.0),
                "fraction_percent": rounded(fraction_percent),
            }
        )

    component_voxel_sum = sum(component["voxel_count"] for component in components)
    component_volume_sum_mm3 = sum(component["voxel_count"] * voxel_volume_mm3 for component in components)
    voxel_sum_match = component_voxel_sum == total_voxels
    volume_sum_match = bool(
        np.isclose(component_volume_sum_mm3, total_volume_mm3, rtol=1e-12, atol=1e-9)
    )
    if not voxel_sum_match or not volume_sum_match:
        raise AnalysisError("Component sum validation failed")

    volumes_ml = [component["volume_ml"] for component in components]
    return {
        "component_count": int(component_count),
        "total_voxels": total_voxels,
        "voxel_volume_mm3": rounded(voxel_volume_mm3),
        "total_volume_mm3": rounded(total_volume_mm3),
        "total_volume_ml": rounded(total_volume_mm3 / 1000.0),
        "components": components,
        "size_distribution": {
            "largest_component_ml": volumes_ml[0] if volumes_ml else None,
            "second_largest_component_ml": volumes_ml[1] if len(volumes_ml) > 1 else None,
            "smallest_component_ml": volumes_ml[-1] if volumes_ml else None,
            "largest_component_fraction_percent": (
                components[0]["fraction_percent"] if components else None
            ),
        },
        "sum_check": {
            "component_voxel_sum": component_voxel_sum,
            "component_volume_sum_mm3": rounded(component_volume_sum_mm3),
            "voxel_sum_match": voxel_sum_match,
            "volume_sum_match": volume_sum_match,
            "status": "PASS",
        },
    }


def log_component_summary(logger: logging.Logger, role: str, result: dict) -> None:
    logger.info(
        "%s: total_voxels=%d total_volume=%.3f mL component_count=%d",
        role,
        result["total_voxels"],
        result["total_volume_ml"],
        result["component_count"],
    )
    for component in result["components"]:
        logger.info(
            "%s component rank=%d label_id=%d voxels=%d volume=%.3f mm3 (%.3f mL) fraction=%.3f%%",
            role,
            component["rank"],
            component["label_id"],
            component["voxel_count"],
            component["volume_mm3"],
            component["volume_ml"],
            component["fraction_percent"],
        )
    distribution = result["size_distribution"]
    logger.info(
        "%s distribution: largest=%s mL second_largest=%s mL smallest=%s mL largest_fraction=%s%%",
        role,
        distribution["largest_component_ml"],
        distribution["second_largest_component_ml"],
        distribution["smallest_component_ml"],
        distribution["largest_component_fraction_percent"],
    )
    logger.info("%s component sum check: %s", role, result["sum_check"]["status"])


def analyze(args: argparse.Namespace, logger: logging.Logger, output_dir: Path, case_id: str) -> dict:
    pred_path = args.pred.resolve()
    gt_path = args.gt.resolve()
    pred_image, pred_data, pred_labels = load_binary_mask(pred_path, "Prediction")
    gt_image, gt_data, gt_labels = load_binary_mask(gt_path, "Ground truth")
    pred_spacing = get_spacing(pred_image, "Prediction")
    gt_spacing = get_spacing(gt_image, "Ground truth")

    shape_match = pred_image.shape == gt_image.shape
    spacing_match = bool(np.allclose(pred_spacing, gt_spacing, rtol=args.rtol, atol=args.atol))
    affine_match = bool(np.allclose(pred_image.affine, gt_image.affine, rtol=args.rtol, atol=args.atol))
    logger.info("Case: %s (%s)", case_id, args.case_type)
    logger.info("Prediction: %s", pred_path)
    logger.info("Ground truth: %s", gt_path)
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
    logger.info("Geometry tolerance: rtol=%g, atol=%g", args.rtol, args.atol)
    logger.info(
        "Physical space: shape_match=%s spacing_match=%s affine_match=%s",
        shape_match,
        spacing_match,
        affine_match,
    )
    if not shape_match:
        raise AnalysisError(f"Physical-space check failed: shapes differ ({pred_image.shape} vs {gt_image.shape})")
    if not spacing_match:
        raise AnalysisError(
            f"Physical-space check failed: spacing differs ({pred_spacing.tolist()} vs {gt_spacing.tolist()})"
        )
    if not affine_match:
        difference = float(np.max(np.abs(pred_image.affine - gt_image.affine)))
        raise AnalysisError(f"Physical-space check failed: affines differ (max absolute difference {difference})")

    structure = generate_binary_structure(3, CONNECTIVITY_RANK[args.connectivity])
    neighbor_count = int(np.count_nonzero(structure) - 1)
    if neighbor_count != args.connectivity:
        raise AnalysisError(
            f"Connectivity structure mismatch: requested {args.connectivity}, generated {neighbor_count} neighbors"
        )
    logger.info(
        "Connectivity: 3D %d-connectivity (%d neighbors; faces/edges/corners included=%s)",
        args.connectivity,
        neighbor_count,
        args.connectivity == 26,
    )

    pred_voxel_volume_mm3 = float(np.prod(pred_spacing, dtype=np.float64))
    gt_voxel_volume_mm3 = float(np.prod(gt_spacing, dtype=np.float64))
    prediction = component_metrics(pred_data > 0, pred_voxel_volume_mm3, structure)
    ground_truth = component_metrics(gt_data > 0, gt_voxel_volume_mm3, structure)
    log_component_summary(logger, "Prediction", prediction)
    log_component_summary(logger, "Ground truth", ground_truth)

    metrics = {
        "case_id": case_id,
        "case_type": args.case_type,
        "connectivity": args.connectivity,
        "connectivity_dimension": "3D",
        "neighbor_count": neighbor_count,
        "prediction_path": str(pred_path),
        "ground_truth_path": str(gt_path),
        "physical_space": {
            "shape_match": shape_match,
            "spacing_match": spacing_match,
            "affine_match": affine_match,
            "max_affine_absolute_difference": float(
                np.max(np.abs(pred_image.affine - gt_image.affine))
            ),
            "tolerance": {"rtol": args.rtol, "atol": args.atol},
        },
        "input_validation": {
            "prediction_labels": pred_labels,
            "ground_truth_labels": gt_labels,
            "prediction_finite": True,
            "ground_truth_finite": True,
            "prediction_spacing_mm": pred_spacing.tolist(),
            "ground_truth_spacing_mm": gt_spacing.tolist(),
        },
        "prediction": prediction,
        "ground_truth": ground_truth,
        "status": "PASS",
    }
    output_path = output_dir / "connected_components.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("JSON output: %s", output_path)
    logger.info("Block 3 Status: PASS")
    return metrics


def main() -> int:
    args = parse_args()
    case_id = args.case_id or nifti_stem(args.pred)
    output_dir = (
        args.output_dir or ROOT / "outputs" / "analysis" / "connected_components" / case_id
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "connected_components.log")
    try:
        analyze(args, logger, output_dir, case_id)
    except AnalysisError as exc:
        logger.error("%s", exc)
        logger.error("Block 3 Status: FAIL")
        return 1
    except Exception:
        logger.exception("Unexpected failure")
        logger.error("Block 3 Status: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
