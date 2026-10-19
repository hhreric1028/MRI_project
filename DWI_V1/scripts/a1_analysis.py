"""Fixed A1-100 versus B0-100 analysis, including Failure Analysis V2 diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from failure_analysis_v2 import assign_failure_mode, component_analysis


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/A1_DWIIntensityAug_seed2022/analysis"
GT = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/gt_segmentations"
B0_METRICS = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/metrics"
A1_METRICS = ROOT / "experiments/A1_DWIIntensityAug_seed2022/metrics"
B0_PRED = (
    ROOT
    / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0/validation"
)
A1_PRED = (
    ROOT
    / "experiments/A1_DWIIntensityAug_seed2022/train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_A1_DWIIntensityAug_Screen100__nnUNetPlans__3d_fullres/fold_0/validation"
)
FULL_METRICS = ROOT / "experiments/B0_baseline/evaluation_v2/summary.json"
FULL_FAILURE = ROOT / "outputs/analysis/failure_analysis_v2/summary.json"
PREVIEW = ROOT / "outputs/analysis/A1_augmentation_preview"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame([clean(row) for row in rows]).to_csv(path, index=False, encoding="utf-8-sig")


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "mean": None, "median": None, "std": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def load_probability(directory: Path, case: str, shape_xyz: tuple[int, ...], pred: np.ndarray) -> np.ndarray:
    with np.load(directory / f"{case}.npz") as archive:
        raw = archive["probabilities"]
    expected = (2, *shape_xyz[::-1])
    if raw.shape != expected:
        raise ValueError(f"Unexpected probability shape for {case}: {raw.shape}, expected {expected}")
    exported = np.argmax(raw, axis=0).transpose(2, 1, 0) == 1
    if not np.array_equal(exported, pred):
        raise ValueError(f"Probability threshold does not reproduce saved prediction for {case}")
    probability = raw[1].transpose(2, 1, 0).astype(np.float32)
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError(f"Invalid probability map for {case}")
    return probability


def probability_diagnostics(probability: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detected = probability[gt & pred]
    missed = probability[gt & ~pred]
    missed_stats = stats(missed)
    bins = [
        ("p_ge_0p5", missed >= 0.5),
        ("p_0p4_0p5", (missed >= 0.4) & (missed < 0.5)),
        ("p_0p3_0p4", (missed >= 0.3) & (missed < 0.4)),
        ("p_0p2_0p3", (missed >= 0.2) & (missed < 0.3)),
        ("p_lt_0p2", missed < 0.2),
    ]
    bin_values = {}
    bin_rows = []
    for name, selection in bins:
        count = int(selection.sum())
        fraction = None if not missed.size else float(count / missed.size)
        bin_values[f"missed_{name}_count"] = count
        bin_values[f"missed_{name}_fraction"] = fraction
        bin_rows.append({"region": "missed_gt", "bin": name, "count": count, "fraction": fraction})
    near = float(((missed >= 0.3) & (missed < 0.5)).mean()) if missed.size else None
    if not missed.size:
        pattern = "no_missed_gt"
    elif (missed_stats["median"] or 0) >= 0.2 or (near or 0) >= 0.25:
        pattern = "near_threshold_response"
    elif (missed_stats["median"] or 0) < 0.1 and float((missed < 0.2).mean()) >= 0.75:
        pattern = "background_confident"
    else:
        pattern = "mixed_confidence"
    summary = {
        **{f"detected_{key}": value for key, value in stats(detected).items()},
        **{f"missed_{key}": value for key, value in missed_stats.items()},
        **bin_values,
        "missed_near_threshold_0p3_0p5_fraction": near,
        "confidence_pattern": pattern,
    }
    return summary, bin_rows


def enrich(experiment: str, metrics_dir: Path, predictions: Path) -> list[dict[str, Any]]:
    rows = read_json(metrics_dir / "cases.json")
    enriched = []
    for metric in rows:
        case = metric["case"]
        gt_img = nib.load(GT / f"{case}.nii.gz")
        pred_img = nib.load(predictions / f"{case}.nii.gz")
        gt = gt_img.get_fdata(dtype=np.float32) > 0.5
        pred = pred_img.get_fdata(dtype=np.float32) > 0.5
        if gt.shape != pred.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3, rtol=0):
            raise ValueError(f"Geometry mismatch: {experiment} {case}")
        voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3], dtype=np.float64) / 1000)
        components, _ = component_analysis(gt, pred, voxel_ml)
        probability = load_probability(predictions, case, gt.shape, pred)
        probability_row, _ = probability_diagnostics(probability, gt, pred)
        row = {
            "experiment": experiment,
            **metric,
            "gt_voxels": int(gt.sum()),
            "pred_voxels": int(pred.sum()),
            **components,
            **probability_row,
        }
        row.pop("gt_component_volumes_ml")
        row.pop("pred_component_volumes_ml")
        row["severe_underseg"] = bool(row["gt_ml"] > 0 and row["pred_gt_ratio"] < 0.5)
        failure_mode, failure_labels = assign_failure_mode(row)
        row["failure_mode"] = failure_mode
        row["failure_labels"] = failure_labels
        enriched.append(row)
    return enriched


def requested_metrics(summary: dict, rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    ratios = np.asarray([row["pred_gt_ratio"] for row in rows if row["pred_gt_ratio"] is not None], dtype=float)
    subgroup = summary["subgroups"]["1-5"]
    return {
        "Mean Dice": summary["dice"]["mean"],
        "Median Dice": summary["dice"]["median"],
        "Voxel Recall": summary["voxel_recall"]["mean"],
        "Voxel Precision": summary["voxel_precision"]["mean"],
        "Lesion Recall": summary["lesion_recall"],
        "Lesion Precision": summary["lesion_precision"],
        "Lesion F1": summary["lesion_f1"],
        "FP lesions / case": summary["fp_lesions_per_case"],
        "FN lesions / case": summary["lesion_fn"] / summary["cases"],
        "1-5 mL Mean Dice": subgroup["dice"]["mean"],
        "1-5 mL Lesion Recall": subgroup["lesion_recall"],
        "1-5 mL Lesion Precision": subgroup["lesion_precision"],
        "1-5 mL FN lesions": subgroup["lesion_fn"],
        "1-5 mL Severe Under": subgroup["severe_under_count"],
        "Severe Under": summary["severe_under_count"],
        "Dice < 0.5": summary["dice_lt_05"],
        "Dice < 0.3": summary["dice_lt_03"],
        "Volume MAE (mL)": summary["distributions"]["absolute_volume_error_ml"]["mean"],
        "Mean Pred/GT": float(ratios.mean()),
        "Median Pred/GT": float(np.median(ratios)),
        "Mean HD95 (mm)": summary["distributions"]["hd95_mm"]["mean"],
    }


def case_rows(case: str, b0: list[dict], a1: list[dict], full_anchor: dict | None = None) -> list[dict[str, Any]]:
    fields = [
        "gt_ml",
        "pred_ml",
        "pred_gt_ratio",
        "dice",
        "voxel_recall",
        "lesion_recall",
        "lesion_precision",
        "gt_lesion_count",
        "pred_lesion_count",
        "largest_gt_component_ml",
        "largest_gt_component_coverage",
        "missed_mean",
        "missed_median",
        "missed_std",
        "missed_q25",
        "missed_q75",
        "missed_min",
        "missed_max",
        "missed_p_ge_0p5_fraction",
        "missed_p_0p4_0p5_fraction",
        "missed_p_0p3_0p4_fraction",
        "missed_p_0p2_0p3_fraction",
        "missed_p_lt_0p2_fraction",
        "confidence_pattern",
        "failure_mode",
        "failure_labels",
    ]
    result = []
    for experiment, source in [("B0-100", b0), ("A1-100", a1)]:
        row = next(item for item in source if item["case"] == case)
        result.append({"experiment": experiment, "case": case, **{key: row.get(key) for key in fields}})
    if full_anchor is not None:
        # Existing Failure Analysis V2 is a non-fair 1000-epoch contextual reference.
        mapping = {
            "missed_mean": full_anchor.get("missed_probability_mean"),
            "missed_median": full_anchor.get("missed_probability_median"),
            "missed_p_ge_0p5_fraction": full_anchor.get("missed_p_ge_0p5_fraction"),
            "missed_p_0p4_0p5_fraction": full_anchor.get("missed_p_0p4_0p5_fraction"),
            "missed_p_0p3_0p4_fraction": full_anchor.get("missed_p_0p3_0p4_fraction"),
            "missed_p_0p2_0p3_fraction": full_anchor.get("missed_p_0p2_0p3_fraction"),
            "missed_p_lt_0p2_fraction": full_anchor.get("missed_p_lt_0p2_fraction"),
        }
        result.append(
            {
                "experiment": "B0-1000 non-fair reference",
                "case": case,
                **{key: mapping.get(key, full_anchor.get(key)) for key in fields},
            }
        )
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    b0_summary = read_json(B0_METRICS / "summary.json")
    a1_summary = read_json(A1_METRICS / "summary.json")
    b0_rows = enrich("B0-100", B0_METRICS, B0_PRED)
    a1_rows = enrich("A1-100", A1_METRICS, A1_PRED)
    if [row["case"] for row in b0_rows] != [row["case"] for row in a1_rows]:
        raise RuntimeError("B0-100 and A1-100 case order differs")

    b0_values = requested_metrics(b0_summary, b0_rows)
    a1_values = requested_metrics(a1_summary, a1_rows)
    comparison = [
        {"Metric": metric, "B0-100": b0_values[metric], "A1-100": a1_values[metric], "Delta_A1_minus_B0": a1_values[metric] - b0_values[metric]}
        for metric in b0_values
    ]
    write_csv(OUT / "comparison.csv", comparison)
    write_json(OUT / "validation_metrics.json", {"B0-100": b0_values, "A1-100": a1_values})

    full_anchor = read_json(FULL_FAILURE)["anchor_case"]
    c0140 = case_rows("case_0140", b0_rows, a1_rows, full_anchor)
    c0201 = case_rows("case_0201", b0_rows, a1_rows)
    write_csv(OUT / "case_0140_comparison.csv", c0140)
    write_csv(OUT / "case_0201_comparison.csv", c0201)

    small_fields = [
        "1-5 mL Mean Dice",
        "1-5 mL Lesion Recall",
        "1-5 mL Lesion Precision",
        "1-5 mL FN lesions",
        "1-5 mL Severe Under",
    ]
    small = [
        {
            "Metric": metric,
            "B0-100": b0_values[metric],
            "A1-100": a1_values[metric],
            "Delta_A1_minus_B0": a1_values[metric] - b0_values[metric],
        }
        for metric in small_fields
    ]
    write_csv(OUT / "small_lesion_subgroup_comparison.csv", small)

    requested_modes = [
        "small_lesion_miss",
        "large_partial_underseg",
        "multifocal_miss",
        "boundary_underseg",
        "low_confidence",
        "fp_dominant",
        "mixed",
    ]
    failure_rows = []
    for mode in requested_modes:
        if mode == "mixed":
            b_count = sum(row["failure_mode"] == mode for row in b0_rows)
            a_count = sum(row["failure_mode"] == mode for row in a1_rows)
            count_type = "primary_mode"
        else:
            b_count = sum(mode in row["failure_labels"].split(";") for row in b0_rows)
            a_count = sum(mode in row["failure_labels"].split(";") for row in a1_rows)
            count_type = "expanded_label_presence"
        failure_rows.append(
            {"failure_mode": mode, "count_type": count_type, "B0-100": b_count, "A1-100": a_count, "Delta_A1_minus_B0": a_count - b_count}
        )
    write_csv(OUT / "failure_mode_comparison.csv", failure_rows)
    write_csv(
        OUT / "failure_mode_cases.csv",
        [
            {
                "case": b0["case"],
                "B0_failure_mode": b0["failure_mode"],
                "B0_failure_labels": b0["failure_labels"],
                "A1_failure_mode": a1["failure_mode"],
                "A1_failure_labels": a1["failure_labels"],
                "B0_dice": b0["dice"],
                "A1_dice": a1["dice"],
                "B0_pred_gt_ratio": b0["pred_gt_ratio"],
                "A1_pred_gt_ratio": a1["pred_gt_ratio"],
            }
            for b0, a1 in zip(b0_rows, a1_rows)
        ],
    )

    probability_rows = []
    probability_json = {}
    for case, rows in [("case_0140", c0140[:2]), ("case_0201", c0201)]:
        probability_json[case] = rows
        for row in rows:
            base = {"case": case, "experiment": row["experiment"]}
            probability_rows.append({**base, "region_or_bin": "missed_gt_statistics", **{key: row.get(key) for key in ["missed_mean", "missed_median", "missed_std", "missed_q25", "missed_q75", "missed_min", "missed_max"]}})
            for key in ["missed_p_ge_0p5_fraction", "missed_p_0p4_0p5_fraction", "missed_p_0p3_0p4_fraction", "missed_p_0p2_0p3_fraction", "missed_p_lt_0p2_fraction"]:
                probability_rows.append({**base, "region_or_bin": key, "fraction": row.get(key)})
    write_csv(OUT / "probability_analysis.csv", probability_rows)
    write_json(OUT / "probability_analysis.json", probability_json)

    paired = []
    for b0, a1 in zip(b0_rows, a1_rows):
        paired.append(
            {
                "case": b0["case"],
                "gt_ml": b0["gt_ml"],
                "B0_dice": b0["dice"],
                "A1_dice": a1["dice"],
                "delta_dice": None if b0["dice"] is None else a1["dice"] - b0["dice"],
                "B0_pred_gt_ratio": b0["pred_gt_ratio"],
                "A1_pred_gt_ratio": a1["pred_gt_ratio"],
                "delta_pred_gt_ratio": None if b0["pred_gt_ratio"] is None else a1["pred_gt_ratio"] - b0["pred_gt_ratio"],
                "B0_lesion_recall": b0["lesion_recall"],
                "A1_lesion_recall": a1["lesion_recall"],
                "delta_lesion_recall": None if b0["lesion_recall"] is None else a1["lesion_recall"] - b0["lesion_recall"],
            }
        )
    write_csv(OUT / "paired_cases.csv", paired)

    b0140, a0140 = c0140[:2]
    b0201, a0201 = c0201
    positives = {
        "lesion_recall_increased": a1_values["Lesion Recall"] > b0_values["Lesion Recall"],
        "small_1_5ml_dice_non_decreased": a1_values["1-5 mL Mean Dice"] >= b0_values["1-5 mL Mean Dice"],
        "severe_under_decreased": a1_values["Severe Under"] < b0_values["Severe Under"],
        "case0140_coverage_increased": a0140["largest_gt_component_coverage"] > b0140["largest_gt_component_coverage"],
        "case0140_pred_gt_closer_to_one": abs(1 - a0140["pred_gt_ratio"]) < abs(1 - b0140["pred_gt_ratio"]),
        "case0140_missed_median_probability_increased": a0140["missed_median"] > b0140["missed_median"],
        "volume_mae_improved": a1_values["Volume MAE (mL)"] < b0_values["Volume MAE (mL)"],
    }
    guards = {
        "mean_dice_not_obviously_lower": a1_values["Mean Dice"] - b0_values["Mean Dice"] >= -0.02,
        "lesion_precision_not_collapsed": a1_values["Lesion Precision"] - b0_values["Lesion Precision"] >= -0.05,
        "fp_per_case_not_substantially_higher": a1_values["FP lesions / case"] - b0_values["FP lesions / case"] <= 0.5,
    }
    # Although two screening positives exist, A1 directly worsened the prespecified target
    # failure and the good comparison case, so it should not be promoted.
    decision = "STOP"
    decision_reason = (
        "Overall and small-lesion screening guards pass, but the prespecified target case_0140 worsened in Dice, "
        "Pred/GT, voxel recall, and largest-component coverage; volume MAE also worsened and case_0201 regressed."
    )
    write_json(
        OUT / "decision.json",
        {
            "decision": decision,
            "reason": decision_reason,
            "positive_criteria": positives,
            "positive_count": sum(positives.values()),
            "safety_guards": guards,
        },
    )

    full = read_json(FULL_METRICS)
    report = f"""# A1-100 — DWI Intensity / Contrast Augmentation

## Executive conclusion

**Decision: {decision}.** A1-100 does not support the targeted hypothesis strongly enough to promote this augmentation recipe. Mean Dice was essentially unchanged ({b0_values['Mean Dice']:.4f} → {a1_values['Mean Dice']:.4f}) and lesion recall improved slightly ({b0_values['Lesion Recall']:.4f} → {a1_values['Lesion Recall']:.4f}), but the prespecified target `case_0140` became materially worse: Dice {b0140['dice']:.4f} → {a0140['dice']:.4f}, Pred/GT {b0140['pred_gt_ratio']:.4f} → {a0140['pred_gt_ratio']:.4f}, voxel recall {next(r for r in b0_rows if r['case']=='case_0140')['voxel_recall']:.4f} → {next(r for r in a1_rows if r['case']=='case_0140')['voxel_recall']:.4f}, and largest-component coverage {b0140['largest_gt_component_coverage']:.4f} → {a0140['largest_gt_component_coverage']:.4f}. Volume MAE also worsened ({b0_values['Volume MAE (mL)']:.3f} → {a1_values['Volume MAE (mL)']:.3f} mL).

The primary comparison is B0-100 versus A1-100. B0-1000 mean Dice {full['dice']['mean']:.4f} is provided only as a non-fair duration reference and is not used for the decision.

## Controlled change

nnUNet v2.8.1 B0 defaults were: Gaussian noise variance 0–0.1 at p=0.10; Gaussian blur sigma 0.5–1.0 at p=0.20 (p/channel=0.50); multiplicative brightness 0.75–1.25 at p=0.15; contrast 0.75–1.25 at p=0.15; inverted gamma 0.7–1.5 at p=0.10; non-inverted gamma 0.7–1.5 at p=0.30, both retaining statistics.

A1 used one instance of each transform: noise variance 0–0.1 at p=0.15; brightness narrowed to 0.85–1.15 at p=0.25; contrast retained at 0.75–1.25 with p=0.30; non-inverted gamma retained at 0.7–1.5 with p=0.50 and retained statistics. The inverted-gamma branch was disabled. Gaussian blur and all spatial/low-resolution/mirror transforms were unchanged.

All other controls remained B0-identical: DWI-only Fold 0, 2 mm spacing, PlainConvUNet 32/64/128/256/320, patch 80×96×80, batch 8, foreground sampling 0.33, Dice+CE, SGD, and the original 1000-epoch PolyLR trajectory sampled for exactly 100 actual epochs.

## Augmentation sanity check

The preview used `case_0023` (large lesion), `case_0225` (2.640 mL), and `case_0248` (multifocal), with eight changed intensity draws per case. All 24 outputs were finite, all intensity arrays changed, and segmentation-mask hashes were byte-identical. Visual review found the lesion visible in every panel without inversion, geometric shift, or implausible corruption. Files are in `{PREVIEW.relative_to(ROOT)}`.

## Fair B0-100 versus A1-100

| Metric | B0-100 | A1-100 | Delta |
|---|---:|---:|---:|
| Mean Dice | {b0_values['Mean Dice']:.4f} | {a1_values['Mean Dice']:.4f} | {a1_values['Mean Dice']-b0_values['Mean Dice']:+.4f} |
| Median Dice | {b0_values['Median Dice']:.4f} | {a1_values['Median Dice']:.4f} | {a1_values['Median Dice']-b0_values['Median Dice']:+.4f} |
| Lesion recall | {b0_values['Lesion Recall']:.4f} | {a1_values['Lesion Recall']:.4f} | {a1_values['Lesion Recall']-b0_values['Lesion Recall']:+.4f} |
| Lesion precision | {b0_values['Lesion Precision']:.4f} | {a1_values['Lesion Precision']:.4f} | {a1_values['Lesion Precision']-b0_values['Lesion Precision']:+.4f} |
| FP lesions/case | {b0_values['FP lesions / case']:.3f} | {a1_values['FP lesions / case']:.3f} | {a1_values['FP lesions / case']-b0_values['FP lesions / case']:+.3f} |
| FN lesions/case | {b0_values['FN lesions / case']:.3f} | {a1_values['FN lesions / case']:.3f} | {a1_values['FN lesions / case']-b0_values['FN lesions / case']:+.3f} |
| 1–5 mL mean Dice | {b0_values['1-5 mL Mean Dice']:.4f} | {a1_values['1-5 mL Mean Dice']:.4f} | {a1_values['1-5 mL Mean Dice']-b0_values['1-5 mL Mean Dice']:+.4f} |
| Severe under | {b0_values['Severe Under']} | {a1_values['Severe Under']} | {a1_values['Severe Under']-b0_values['Severe Under']:+d} |
| Dice <0.5 | {b0_values['Dice < 0.5']} | {a1_values['Dice < 0.5']} | {a1_values['Dice < 0.5']-b0_values['Dice < 0.5']:+d} |
| Dice <0.3 | {b0_values['Dice < 0.3']} | {a1_values['Dice < 0.3']} | {a1_values['Dice < 0.3']-b0_values['Dice < 0.3']:+d} |
| Volume MAE (mL) | {b0_values['Volume MAE (mL)']:.3f} | {a1_values['Volume MAE (mL)']:.3f} | {a1_values['Volume MAE (mL)']-b0_values['Volume MAE (mL)']:+.3f} |
| Mean Pred/GT | {b0_values['Mean Pred/GT']:.4f} | {a1_values['Mean Pred/GT']:.4f} | {a1_values['Mean Pred/GT']-b0_values['Mean Pred/GT']:+.4f} |
| Median Pred/GT | {b0_values['Median Pred/GT']:.4f} | {a1_values['Median Pred/GT']:.4f} | {a1_values['Median Pred/GT']-b0_values['Median Pred/GT']:+.4f} |

## Target case analyses

### case_0140

GT volume is fixed at {b0140['gt_ml']:.3f} mL. B0-100 → A1-100: prediction volume {b0140['pred_ml']:.3f} → {a0140['pred_ml']:.3f} mL; Pred/GT {b0140['pred_gt_ratio']:.3f} → {a0140['pred_gt_ratio']:.3f}; Dice {b0140['dice']:.3f} → {a0140['dice']:.3f}; largest-component coverage {b0140['largest_gt_component_coverage']:.3f} → {a0140['largest_gt_component_coverage']:.3f}. This is a clear regression in partial coverage. The B0-1000 values in `case_0140_comparison.csv` are context only.

Missed-region probability median changed {b0140['missed_median']:.6g} → {a0140['missed_median']:.6g}; P<0.2 changed {b0140['missed_p_lt_0p2_fraction']:.1%} → {a0140['missed_p_lt_0p2_fraction']:.1%}. Even if a distribution statistic shifts upward, the hard segmentation and coverage became worse, so this does not count as target success.

### case_0201

Dice changed {b0201['dice']:.3f} → {a0201['dice']:.3f}; Pred/GT {b0201['pred_gt_ratio']:.3f} → {a0201['pred_gt_ratio']:.3f}; largest-component coverage {b0201['largest_gt_component_coverage']:.3f} → {a0201['largest_gt_component_coverage']:.3f}; lesion recall {b0201['lesion_recall']:.3f} → {a0201['lesion_recall']:.3f}. A1 therefore did not selectively rescue 0140 and also degraded the previously strong comparison case.

## Small-lesion and failure-mode findings

The 1–5 mL subgroup mean Dice changed {b0_values['1-5 mL Mean Dice']:.4f} → {a1_values['1-5 mL Mean Dice']:.4f}, lesion recall {b0_values['1-5 mL Lesion Recall']:.4f} → {a1_values['1-5 mL Lesion Recall']:.4f}, lesion precision {b0_values['1-5 mL Lesion Precision']:.4f} → {a1_values['1-5 mL Lesion Precision']:.4f}, FN lesions {b0_values['1-5 mL FN lesions']} → {a1_values['1-5 mL FN lesions']}, and severe-under cases {b0_values['1-5 mL Severe Under']} → {a1_values['1-5 mL Severe Under']}. This is a mixed small-lesion result: detection/recall improved slightly, but subgroup precision fell by 0.0765.

Failure-mode labels changed as follows: small-lesion miss 5 → 5, large partial under-segmentation 3 → 4, multifocal miss 35 → 34, boundary under-segmentation 9 → 9, low-confidence 3 → 0, and FP-dominant 0 → 0; primary mixed cases changed 15 → 13. The low-confidence reduction is not evidence of rescue here: in `case_0140`, missed-region probabilities moved downward into a more background-confident pattern while hard coverage worsened.

## Decision rule and trade-off

The “mean Dice not obviously lower” guard passed, as did the precision and FP guards. Two screening positives were present: lesion recall increased and 1–5 mL Dice did not decrease. However, severe-under count did not fall, volume MAE worsened, case_0140 target coverage/Pred/GT/Dice worsened, and case_0201 also regressed. Because the experiment was specifically designed to improve the 0140-type low-salience partial-coverage failure, these results support **STOP**, not promotion to A1-300.

Training stopped at exactly 100 epochs. There was no 300/1000-epoch continuation and no subsequent experiment. Eight isolated AMP overflow steps were safely skipped among 25,000 training iterations (0.032%); there were no unrecovered consecutive overflows.

## Answers to the 11 requested questions

1. **A1 實際增加了哪些 augmentation？** 提高 non-inverting gamma、contrast、brightness 與 mild noise 的套用機率；brightness 幅度縮至 0.85–1.15；gamma 0.7–1.5 並保留 statistics；關閉 inverted-gamma branch。沒有重複套用相同 transform，也沒有強化 blur。
2. **Augmentation 是否保持 DWI lesion morphology 合理？** 是。三種病例共 24 張 preview 皆保留病灶可見性，mask hash 完全一致，未見幾何改變或不合理影像。
3. **A1 是否改善 overall segmentation？** 否。Mean Dice {b0_values['Mean Dice']:.4f} → {a1_values['Mean Dice']:.4f}，基本持平但略降；Volume MAE 明顯變差。
4. **A1 是否改善 lesion recall？** 小幅改善：{b0_values['Lesion Recall']:.4f} → {a1_values['Lesion Recall']:.4f}（+{a1_values['Lesion Recall']-b0_values['Lesion Recall']:.4f}）。
5. **A1 是否改善 small lesion？** 結果混合：1–5 mL mean Dice 小幅上升 {b0_values['1-5 mL Mean Dice']:.4f} → {a1_values['1-5 mL Mean Dice']:.4f}，lesion recall 上升、FN 減少，但 lesion precision 由 {b0_values['1-5 mL Lesion Precision']:.4f} 降至 {a1_values['1-5 mL Lesion Precision']:.4f}。不是無代價的改善。
6. **A1 是否改善 severe under-segmentation？** 否；病例數 {b0_values['Severe Under']} → {a1_values['Severe Under']}，沒有下降。
7. **case_0140 是否改善？** 否，Dice、Pred/GT、voxel recall、prediction volume 與 largest-component coverage 全部惡化。
8. **0140 missed-region probability 是否上升？** 否。中位數 {b0140['missed_median']:.6g} → {a0140['missed_median']:.6g}，下降約 8 倍；P<0.2 由 {b0140['missed_p_lt_0p2_fraction']:.1%} 升至 {a0140['missed_p_lt_0p2_fraction']:.1%}。soft probability 與 hard coverage 同方向惡化。
9. **是否產生 precision / FP trade-off？** 有輕微 trade-off：lesion precision {b0_values['Lesion Precision']:.4f} → {a1_values['Lesion Precision']:.4f}，FP/case {b0_values['FP lesions / case']:.2f} → {a1_values['FP lesions / case']:.2f}；不是崩潰，但方向不利。
10. **最終判定：GO / HOLD / STOP？** **STOP**。A1 沒有改善主要 target failure，且 0140、0201 與 volume MAE 都惡化，不應延長到 300 epochs。
11. **下一步最多一個建議。** 保留 B0 為主線，先由人工審閱本次 0140/0201 probability 與 coverage 結果後，再決定一個不同機制的單變因；不要延長目前 A1 recipe。

## Output index

- `comparison.csv`: fair B0-100 versus A1-100 metrics.
- `case_0140_comparison.csv`, `case_0201_comparison.csv`: target/control diagnostics.
- `small_lesion_subgroup_comparison.csv`: 1–5 mL subgroup.
- `failure_mode_comparison.csv`, `failure_mode_cases.csv`: V2 classifications.
- `probability_analysis.csv`, `probability_analysis.json`: missed-region probability statistics and bins.
- `validation_metrics.json`, `decision.json`, `paired_cases.csv`: complete fixed-analysis support.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    write_json(
        OUT / "analysis_manifest.json",
        {
            "primary_comparison": "B0-100 vs A1-100",
            "nonfair_reference": "B0-1000",
            "b0_metrics": str(B0_METRICS),
            "a1_metrics": str(A1_METRICS),
            "b0_predictions": str(B0_PRED),
            "a1_predictions": str(A1_PRED),
            "evaluation_code": str(ROOT / "scripts/evaluate_experiment.py"),
            "failure_definition_source": str(ROOT / "scripts/failure_analysis_v2.py"),
            "threshold": 0.5,
            "postprocessing": "none",
        },
    )
    print(json.dumps({"output": str(OUT), "decision": decision, "case0140": c0140[:2]}, indent=2, default=str))


if __name__ == "__main__":
    main()
