"""Describe coarse lesion location in native patient space without atlas mapping."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_26 = generate_binary_structure(3, 3)


class AnalysisError(RuntimeError):
    """Raised when the inputs cannot support a safe native-space analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Native DWI NIfTI")
    parser.add_argument("--mask", required=True, type=Path, help="Prediction mask NIfTI")
    parser.add_argument("--gt", type=Path, help="Optional ground-truth mask NIfTI")
    parser.add_argument("--case-id", help="Case identifier; defaults to prediction filename")
    parser.add_argument("--case-type", default="unspecified")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to outputs/analysis/native_location/<case-id>",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("native_lesion_location")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_image(path: Path, role: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    path = path.resolve()
    if not path.is_file():
        raise AnalysisError(f"{role} file does not exist: {path}")
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise AnalysisError(f"Cannot read {role}: {path}: {exc}") from exc
    if data.ndim != 3:
        raise AnalysisError(f"{role} must be 3D, got {data.shape}")
    if not np.isfinite(data).all():
        raise AnalysisError(f"{role} contains NaN or infinite values")
    if not np.isfinite(image.affine).all():
        raise AnalysisError(f"{role} affine contains NaN or infinite values")
    units = image.header.get_xyzt_units()[0]
    if units != "mm":
        raise AnalysisError(f"{role} spatial units must be mm, got {units!r}")
    return image, data


def load_binary_mask(path: Path, role: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, list[int]]:
    image, data = load_image(path, role)
    values = np.unique(data)
    if not np.isin(values, (0, 1)).all():
        raise AnalysisError(f"{role} has non-binary values {values.tolist()}")
    return image, data > 0, [int(v) for v in values.tolist()]


def assert_same_grid(
    reference: nib.spatialimages.SpatialImage,
    other: nib.spatialimages.SpatialImage,
    role: str,
    rtol: float,
    atol: float,
) -> dict:
    shape_match = reference.shape == other.shape
    affine_match = bool(np.allclose(reference.affine, other.affine, rtol=rtol, atol=atol))
    spacing_reference = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    spacing_other = np.asarray(other.header.get_zooms()[:3], dtype=float)
    spacing_match = bool(np.allclose(spacing_reference, spacing_other, rtol=rtol, atol=atol))
    result = {
        "shape_match": shape_match,
        "spacing_match": spacing_match,
        "affine_match": affine_match,
        "max_affine_absolute_difference": float(np.max(np.abs(reference.affine - other.affine))),
    }
    if not all((shape_match, spacing_match, affine_match)):
        raise AnalysisError(f"{role} does not share the native image grid: {result}")
    return result


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def orientation_metadata(image: nib.spatialimages.SpatialImage) -> dict:
    axcodes = tuple(str(v) for v in nib.aff2axcodes(image.affine))
    io = nib.orientations.io_orientation(image.affine)
    world_names = {0: "Left-Right", 1: "Posterior-Anterior", 2: "Inferior-Superior"}
    positive_codes = {0: "R", 1: "A", 2: "S"}
    negative_codes = {0: "L", 1: "P", 2: "I"}
    array_axes = []
    for array_axis, (world_axis, direction) in enumerate(io):
        world_axis = int(world_axis)
        direction = int(direction)
        array_axes.append(
            {
                "array_axis": array_axis,
                "dominant_world_axis": world_names[world_axis],
                "increasing_index_direction": positive_codes[world_axis]
                if direction > 0
                else negative_codes[world_axis],
                "orientation_transform": [world_axis, direction],
            }
        )
    return {
        "axis_codes": list(axcodes),
        "axis_codes_string": "".join(axcodes),
        "world_coordinate_convention": "NIfTI RAS+: +x Right, +y Anterior, +z Superior",
        "array_axes": array_axes,
        "affine": np.asarray(image.affine, dtype=float).tolist(),
    }


def estimate_brain_support(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    magnitude = np.abs(data.astype(np.float64, copy=False))
    scale = float(np.max(magnitude))
    if not np.isfinite(scale) or scale <= 0:
        raise AnalysisError("DWI has no usable non-background signal")
    threshold = max(scale * 1e-8, np.finfo(np.float64).eps * scale * 10)
    raw_support = magnitude > threshold
    labeled, count = label(raw_support, structure=STRUCTURE_26)
    component_counts = np.bincount(labeled.ravel(), minlength=count + 1)[1:]
    if component_counts.size == 0:
        raise AnalysisError("No DWI brain-like support was found")
    largest_label = int(np.argmax(component_counts) + 1)
    support = labeled == largest_label
    support_indices = np.argwhere(support)
    support_world = nib.affines.apply_affine(affine, support_indices)
    world_min = support_world.min(axis=0)
    world_max = support_world.max(axis=0)
    world_range = world_max - world_min
    index_min = support_indices.min(axis=0)
    index_max = support_indices.max(axis=0)
    index_span = index_max - index_min + 1
    shape = np.asarray(data.shape)
    touches_border = bool(
        np.any(index_min == 0) or np.any(index_max == shape - 1)
    )
    support_fraction = float(np.mean(support))
    raw_count = int(np.count_nonzero(raw_support))
    largest_fraction = float(np.count_nonzero(support) / raw_count)
    exact_zero_fraction = float(np.mean(data == 0))
    checks = {
        "has_background": exact_zero_fraction >= 0.01,
        "support_fraction_plausible": 0.05 <= support_fraction <= 0.80,
        "largest_component_dominant": largest_fraction >= 0.95,
        "does_not_touch_image_border": not touches_border,
        "spans_plausible_fraction_each_axis": bool(np.all(index_span / shape >= 0.30)),
        "physical_ranges_plausible": bool(np.all(world_range >= 50.0)),
    }
    status = "PASS" if all(checks.values()) else "NEEDS REVIEW"
    midline_x = float((world_min[0] + world_max[0]) / 2.0)
    lr_tolerance = float(max(2.0, world_range[0] * 0.02))
    metadata = {
        "status": status,
        "method": "largest 26-connected finite DWI support above scale-relative background threshold",
        "intensity_threshold_absolute": threshold,
        "intensity_scale_max_abs": scale,
        "exact_zero_fraction_percent": rounded(exact_zero_fraction * 100),
        "raw_support_voxels": raw_count,
        "largest_support_voxels": int(np.count_nonzero(support)),
        "largest_component_fraction_percent": rounded(largest_fraction * 100),
        "support_fraction_of_image_percent": rounded(support_fraction * 100),
        "support_component_count": int(count),
        "index_bounding_box_inclusive": {
            "minimum": index_min.astype(int).tolist(),
            "maximum": index_max.astype(int).tolist(),
            "span_voxels": index_span.astype(int).tolist(),
        },
        "touches_image_border": touches_border,
        "checks": checks,
        "limitations": (
            "Coarse DWI non-background support, not a validated anatomical brain or CSF segmentation."
        ),
    }
    extent = {
        "world_ras_voxel_center_min_mm": world_min.tolist(),
        "world_ras_voxel_center_max_mm": world_max.tolist(),
        "world_ras_range_mm": world_range.tolist(),
        "left_right": {"left_mm": float(world_min[0]), "right_mm": float(world_max[0])},
        "posterior_anterior": {
            "posterior_mm": float(world_min[1]),
            "anterior_mm": float(world_max[1]),
            "posterior_central_boundary_mm": float(world_min[1] + world_range[1] / 3.0),
            "central_anterior_boundary_mm": float(world_min[1] + 2.0 * world_range[1] / 3.0),
        },
        "inferior_superior": {
            "inferior_mm": float(world_min[2]),
            "superior_mm": float(world_max[2]),
            "inferior_middle_boundary_mm": float(world_min[2] + world_range[2] / 3.0),
            "middle_superior_boundary_mm": float(world_min[2] + 2.0 * world_range[2] / 3.0),
        },
        "estimated_midline_x_mm": midline_x,
        "midline_uncertainty_band_half_width_mm": lr_tolerance,
    }
    return support, metadata, extent


def side_metrics(world: np.ndarray, midline_x: float, tolerance: float) -> dict:
    x = world[:, 0]
    count = len(x)
    left = int(np.count_nonzero(x < midline_x))
    right = int(np.count_nonzero(x > midline_x))
    exact = count - left - right
    robust_left = int(np.count_nonzero(x < midline_x - tolerance))
    robust_right = int(np.count_nonzero(x > midline_x + tolerance))
    band = count - robust_left - robust_right
    left_pct = left / count * 100.0
    right_pct = right / count * 100.0
    robust_left_pct = robust_left / count * 100.0
    robust_right_pct = robust_right / count * 100.0
    robust_crossing = robust_left >= 2 and robust_right >= 2 and min(robust_left_pct, robust_right_pct) >= 5.0
    if robust_crossing:
        label_text = "Midline-crossing"
    elif left == count:
        label_text = "Left"
    elif right == count:
        label_text = "Right"
    elif left_pct >= 70.0:
        label_text = "Predominantly Left"
    elif right_pct >= 70.0:
        label_text = "Predominantly Right"
    else:
        label_text = "Bilateral"
    return {
        "laterality": label_text,
        "left_fraction_percent": rounded(left_pct),
        "right_fraction_percent": rounded(right_pct),
        "exact_midline_fraction_percent": rounded(exact / count * 100.0),
        "outside_uncertainty_band": {
            "left_fraction_percent": rounded(robust_left_pct),
            "right_fraction_percent": rounded(robust_right_pct),
            "midline_band_fraction_percent": rounded(band / count * 100.0),
        },
        "robust_midline_crossing": robust_crossing,
    }


def third_metrics(values: np.ndarray, lower: float, upper: float, labels: tuple[str, str, str]) -> dict:
    width = upper - lower
    if width <= 0:
        raise AnalysisError("Invalid patient-specific physical extent")
    normalized = (values - lower) / width
    groups = np.zeros(len(values), dtype=np.int8)
    groups[normalized >= 1.0 / 3.0] = 1
    groups[normalized >= 2.0 / 3.0] = 2
    counts = np.bincount(groups, minlength=3)
    fractions = counts / len(values) * 100.0
    dominant = int(np.argmax(counts))
    centroid = float(np.mean(normalized))
    centroid_group = 0 if centroid < 1.0 / 3.0 else 1 if centroid < 2.0 / 3.0 else 2
    boundary_distance = min(abs(centroid - 1.0 / 3.0), abs(centroid - 2.0 / 3.0))
    reliable = bool(fractions[dominant] >= 55.0 and boundary_distance >= 0.03)
    return {
        "region": labels[dominant],
        "region_by_centroid": labels[centroid_group],
        "region_fractions_percent": {
            labels[index]: rounded(fractions[index]) for index in range(3)
        },
        "centroid_relative_position": rounded(centroid, 6),
        "centroid_distance_to_nearest_third_boundary": rounded(boundary_distance, 6),
        "classification_status": "PASS" if reliable else "NEEDS REVIEW",
    }


def component_and_overall_metrics(
    mask: np.ndarray,
    affine: np.ndarray,
    extent: dict,
) -> tuple[dict, list[dict], np.ndarray]:
    labeled, component_count = label(mask, structure=STRUCTURE_26)
    counts = np.bincount(labeled.ravel(), minlength=component_count + 1)[1:]
    ordered = sorted(
        ((label_id, int(count)) for label_id, count in enumerate(counts, start=1)),
        key=lambda item: (-item[1], item[0]),
    )
    voxel_volume_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    midline = float(extent["estimated_midline_x_mm"])
    tolerance = float(extent["midline_uncertainty_band_half_width_mm"])
    y_lower = float(extent["posterior_anterior"]["posterior_mm"])
    y_upper = float(extent["posterior_anterior"]["anterior_mm"])
    z_lower = float(extent["inferior_superior"]["inferior_mm"])
    z_upper = float(extent["inferior_superior"]["superior_mm"])

    components = []
    for rank, (label_id, voxel_count) in enumerate(ordered, start=1):
        indices = np.argwhere(labeled == label_id)
        world = nib.affines.apply_affine(affine, indices)
        component = {
            "rank": rank,
            "label_id": label_id,
            "voxel_count": voxel_count,
            "volume_mm3": rounded(voxel_count * voxel_volume_mm3),
            "volume_ml": rounded(voxel_count * voxel_volume_mm3 / 1000.0),
            "centroid_world_ras_mm": [rounded(v, 6) for v in world.mean(axis=0)],
            "world_bounding_box_mm": {
                "minimum": [rounded(v, 6) for v in world.min(axis=0)],
                "maximum": [rounded(v, 6) for v in world.max(axis=0)],
            },
        }
        component.update(side_metrics(world, midline, tolerance))
        component["anterior_posterior"] = third_metrics(
            world[:, 1], y_lower, y_upper, ("Posterior", "Central", "Anterior")
        )
        component["superior_inferior"] = third_metrics(
            world[:, 2], z_lower, z_upper, ("Inferior", "Middle", "Superior")
        )
        component["location_status"] = (
            "PASS"
            if component["anterior_posterior"]["classification_status"] == "PASS"
            and component["superior_inferior"]["classification_status"] == "PASS"
            else "NEEDS REVIEW"
        )
        components.append(component)

    indices = np.argwhere(mask)
    world = nib.affines.apply_affine(affine, indices)
    side = side_metrics(world, midline, tolerance)
    crossing_ranks = [c["rank"] for c in components if c["robust_midline_crossing"]]
    if crossing_ranks:
        whole_laterality = "Midline-crossing"
    elif side["left_fraction_percent"] >= 10.0 and side["right_fraction_percent"] >= 10.0:
        whole_laterality = "Bilateral"
    elif side["left_fraction_percent"] > side["right_fraction_percent"]:
        whole_laterality = "Predominantly Left"
    else:
        whole_laterality = "Predominantly Right"
    overall = {
        "lesion_voxels": int(np.count_nonzero(mask)),
        "component_count": int(component_count),
        "distribution": "Multifocal" if component_count > 1 else "Focal",
        "laterality": whole_laterality,
        "left_fraction_percent": side["left_fraction_percent"],
        "right_fraction_percent": side["right_fraction_percent"],
        "exact_midline_fraction_percent": side["exact_midline_fraction_percent"],
        "outside_midline_uncertainty_band": side["outside_uncertainty_band"],
        "midline_crossing_component_ranks": crossing_ranks,
        "centroid_world_ras_mm": [rounded(v, 6) for v in world.mean(axis=0)],
        "anterior_posterior": third_metrics(
            world[:, 1], y_lower, y_upper, ("Posterior", "Central", "Anterior")
        ),
        "superior_inferior": third_metrics(
            world[:, 2], z_lower, z_upper, ("Inferior", "Middle", "Superior")
        ),
    }
    overall["location_status"] = (
        "PASS"
        if overall["anterior_posterior"]["classification_status"] == "PASS"
        and overall["superior_inferior"]["classification_status"] == "PASS"
        else "NEEDS REVIEW"
    )
    return overall, components, labeled


def contour_if_present(ax: plt.Axes, field: np.ndarray, level: float, **kwargs: object) -> None:
    finite = field[np.isfinite(field)]
    if finite.size and float(finite.min()) <= level <= float(finite.max()):
        ax.contour(field, levels=[level], **kwargs)


def save_qc(
    image_data: np.ndarray,
    lesion: np.ndarray,
    support: np.ndarray,
    affine: np.ndarray,
    extent: dict,
    output: Path,
    case_id: str,
) -> None:
    support_values = image_data[support]
    vmin, vmax = np.percentile(support_values, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(support_values)), float(np.max(support_values))

    panels = [
        (2, int(np.argmax(np.sum(lesion, axis=(0, 1)))), (0, 1), "Axial-like native plane"),
        (1, int(np.argmax(np.sum(lesion, axis=(0, 2)))), (0, 2), "Coronal-like native plane"),
        (0, int(np.argmax(np.sum(lesion, axis=(1, 2)))), (1, 2), "Sagittal-like native plane"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (fixed_axis, fixed_index, varying, title) in zip(axes, panels):
        slicer: list[object] = [slice(None), slice(None), slice(None)]
        slicer[fixed_axis] = fixed_index
        sl = tuple(slicer)
        image_slice = np.asarray(image_data[sl]).T
        lesion_slice = np.asarray(lesion[sl]).T
        support_slice = np.asarray(support[sl]).T
        ax.imshow(image_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        overlay = np.ma.masked_where(~lesion_slice, lesion_slice)
        ax.imshow(
            overlay,
            cmap=ListedColormap(["red"]),
            origin="lower",
            alpha=0.75,
            vmin=0,
            vmax=1,
        )
        if np.any(support_slice) and np.any(~support_slice):
            ax.contour(support_slice.astype(float), levels=[0.5], colors="white", linewidths=0.6, alpha=0.8)

        first_size = image_data.shape[varying[0]]
        second_size = image_data.shape[varying[1]]
        first, second = np.meshgrid(np.arange(first_size), np.arange(second_size), indexing="xy")
        voxel = np.zeros((second_size, first_size, 3), dtype=float)
        voxel[..., fixed_axis] = fixed_index
        voxel[..., varying[0]] = first
        voxel[..., varying[1]] = second
        world = nib.affines.apply_affine(affine, voxel)

        midline_x = float(extent["estimated_midline_x_mm"])
        ap1 = float(extent["posterior_anterior"]["posterior_central_boundary_mm"])
        ap2 = float(extent["posterior_anterior"]["central_anterior_boundary_mm"])
        si1 = float(extent["inferior_superior"]["inferior_middle_boundary_mm"])
        si2 = float(extent["inferior_superior"]["middle_superior_boundary_mm"])
        contour_if_present(ax, world[..., 0], midline_x, colors="yellow", linewidths=1.2)
        for boundary in (ap1, ap2):
            contour_if_present(ax, world[..., 1], boundary, colors="cyan", linewidths=1.0, linestyles="--")
        for boundary in (si1, si2):
            contour_if_present(ax, world[..., 2], boundary, colors="lime", linewidths=1.0, linestyles="--")
        ax.set_title(f"{title}\narray axis {fixed_axis} = {fixed_index}")
        ax.set_xlabel(f"array axis {varying[0]}")
        ax.set_ylabel(f"array axis {varying[1]}")

    fig.suptitle(
        f"{case_id} native DWI + prediction | red lesion; white support; "
        "yellow estimated midline; cyan AP thirds; green SI thirds\n"
        "QC guides only; not atlas or medical structure boundaries",
        fontsize=12,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace, output_dir: Path, logger: logging.Logger, case_id: str) -> dict:
    image_path = args.image.resolve()
    mask_path = args.mask.resolve()
    gt_path = args.gt.resolve() if args.gt else None
    image, image_data = load_image(image_path, "DWI")
    mask_image, mask, mask_labels = load_binary_mask(mask_path, "Prediction")
    prediction_grid = assert_same_grid(image, mask_image, "Prediction", args.rtol, args.atol)
    if not np.any(mask):
        raise AnalysisError("Prediction is empty")
    orientation = orientation_metadata(image)
    support, support_metadata, extent = estimate_brain_support(image_data, image.affine)
    pred_overall, pred_components, _ = component_and_overall_metrics(mask, image.affine, extent)

    logger.info("Case: %s (%s)", case_id, args.case_type)
    logger.info("DWI: %s", image_path)
    logger.info("Prediction: %s", mask_path)
    logger.info("Shape=%s spacing_mm=%s orientation=%s", image.shape, image.header.get_zooms()[:3], orientation["axis_codes_string"])
    logger.info("Array/world axis mapping: %s", orientation["array_axes"])
    logger.info("Brain extent estimation: %s; checks=%s", support_metadata["status"], support_metadata["checks"])
    logger.info("Estimated midline x=%.3f mm (uncertainty half-width %.3f mm)", extent["estimated_midline_x_mm"], extent["midline_uncertainty_band_half_width_mm"])
    logger.info("Prediction overall: %s", pred_overall)
    for component in pred_components:
        logger.info(
            "Prediction component rank=%d label_id=%d voxels=%d volume_ml=%.3f laterality=%s AP=%s SI=%s status=%s",
            component["rank"], component["label_id"], component["voxel_count"], component["volume_ml"],
            component["laterality"], component["anterior_posterior"]["region"],
            component["superior_inferior"]["region"], component["location_status"],
        )

    ground_truth = None
    comparison = None
    gt_grid = None
    gt_labels = None
    if gt_path:
        gt_image, gt_mask, gt_labels = load_binary_mask(gt_path, "Ground truth")
        gt_grid = assert_same_grid(image, gt_image, "Ground truth", args.rtol, args.atol)
        if not np.any(gt_mask):
            raise AnalysisError("Ground truth is empty")
        gt_overall, gt_components, _ = component_and_overall_metrics(gt_mask, image.affine, extent)
        ground_truth = {"overall": gt_overall, "components": gt_components}
        comparison = {
            "purpose": "coarse distribution comparison only; not a Dice or spatial-overlap substitute",
            "laterality_match": pred_overall["laterality"] == gt_overall["laterality"],
            "anterior_posterior_match": pred_overall["anterior_posterior"]["region"] == gt_overall["anterior_posterior"]["region"],
            "superior_inferior_match": pred_overall["superior_inferior"]["region"] == gt_overall["superior_inferior"]["region"],
            "prediction": {
                "laterality": pred_overall["laterality"],
                "anterior_posterior": pred_overall["anterior_posterior"]["region"],
                "superior_inferior": pred_overall["superior_inferior"]["region"],
                "component_count": pred_overall["component_count"],
            },
            "ground_truth": {
                "laterality": gt_overall["laterality"],
                "anterior_posterior": gt_overall["anterior_posterior"]["region"],
                "superior_inferior": gt_overall["superior_inferior"]["region"],
                "component_count": gt_overall["component_count"],
            },
        }
        logger.info("Ground truth overall: %s", gt_overall)
        logger.info("Prediction vs GT coarse comparison: %s", comparison)

    qc_path = output_dir / "native_location_qc.png"
    save_qc(image_data, mask, support, image.affine, extent, qc_path, case_id)
    block_status = (
        "PASS"
        if support_metadata["status"] == "PASS" and pred_overall["location_status"] == "PASS"
        else "NEEDS REVIEW"
    )
    result = {
        "case_id": case_id,
        "case_type": args.case_type,
        "space": "native",
        "analysis_scope": "coarse native-space location only; no MNI registration or atlas mapping",
        "paths": {
            "image": str(image_path),
            "prediction": str(mask_path),
            "ground_truth": str(gt_path) if gt_path else None,
        },
        "image_geometry": {
            "shape": list(image.shape),
            "spacing_mm": [float(v) for v in image.header.get_zooms()[:3]],
            "orientation": orientation,
            "prediction_grid": prediction_grid,
            "ground_truth_grid": gt_grid,
            "tolerance": {"rtol": args.rtol, "atol": args.atol},
        },
        "input_validation": {
            "prediction_labels": mask_labels,
            "ground_truth_labels": gt_labels,
            "all_inputs_finite": True,
        },
        "brain_extent_estimation": support_metadata,
        "brain_extent": extent,
        "overall": pred_overall,
        "components": pred_components,
        "ground_truth": ground_truth,
        "prediction_vs_ground_truth": comparison,
        "superficial_deep": {
            "classification": "Not determined",
            "reason": "The coarse DWI support is not a validated brain-surface segmentation.",
        },
        "qc": {
            "image": str(qc_path.resolve()),
            "status": "PASS" if support_metadata["status"] == "PASS" else "NEEDS REVIEW",
            "note": "Lines are calculation/QC guides, not medical structure boundaries.",
        },
        "status": block_status,
    }
    json_path = output_dir / "native_location.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("QC image: %s", qc_path)
    logger.info("JSON: %s", json_path)
    logger.info("Block 4C-lite Status: %s", block_status)
    return result


def main() -> int:
    args = parse_args()
    case_id = args.case_id or nifti_stem(args.mask)
    output_dir = (
        args.output_dir or ROOT / "outputs" / "analysis" / "native_location" / case_id
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "native_location.log")
    try:
        analyze(args, output_dir, logger, case_id)
    except AnalysisError as exc:
        logger.error("%s", exc)
        logger.error("Block 4C-lite Status: FAIL")
        return 1
    except Exception:
        logger.exception("Unexpected failure")
        logger.error("Block 4C-lite Status: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
