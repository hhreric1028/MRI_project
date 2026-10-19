"""Build B0 five-fold OOF and the paired B0-100 versus KamelRep-100 report."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from evaluate_experiment import summarize
from failure_analysis_v2 import assign_failure_mode, component_analysis, probability_summary
from kamel_fp_topology import case_topology


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/B0_full5fold_100"
OUT = ROOT / "outputs/analysis/B0_full5fold_100"
KAMEL_OUT = ROOT / "outputs/analysis/kamel_replication"
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
B0_FOLD0 = (
    ROOT
    / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        path.write_text("case\n", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fold_paths(fold: int) -> tuple[Path, Path]:
    run = SUITE / f"fold_{fold}"
    if fold == 0:
        return run / "metrics", B0_FOLD0 / "validation"
    trainer = "nnUNetTrainer_B0Full5fold_Screen100__nnUNetPlans__3d_fullres"
    prediction = run / "results/Dataset001_ISLES2022_DWI" / trainer / f"fold_{fold}/validation"
    return run / "metrics", prediction


def load_probability(path: Path, shape_xyz: tuple[int, ...], pred: np.ndarray) -> np.ndarray:
    with np.load(path) as archive:
        raw = archive["probabilities"]
    if raw.shape != (2, *shape_xyz[::-1]):
        raise ValueError(f"Unexpected probability shape: {path}")
    exported = np.argmax(raw, axis=0).transpose(2, 1, 0) == 1
    if not np.array_equal(exported, pred):
        raise ValueError(f"Probability threshold mismatch: {path}")
    return raw[1].transpose(2, 1, 0).astype(np.float32)


def subgroup_map(fold: int) -> dict[str, dict[str, float | int | None]]:
    converted = {}
    for row in read_csv(OUT / f"fold_{fold}/subgroup_metrics_extended.csv"):
        values: dict[str, float | int | None] = {"n": int(row["n"])}
        for key in ["mean_dice", "median_dice", "lesion_recall", "lesion_precision", "mean_pred_gt", "median_pred_gt"]:
            values[key] = float(row[key]) if row[key] else None
        converted[row["subgroup"]] = values
    return converted


def scalar(value: str | None, kind=float):
    if value in (None, ""):
        return None
    return kind(value)


def boolean(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def has_label(row: dict[str, Any], label: str) -> bool:
    return label == row.get("failure_mode") or label in str(row.get("failure_labels", "")).split(";")


def transition(before: bool, after: bool) -> str:
    if before and not after:
        return "B0 failure -> Kamel normal"
    if not before and after:
        return "B0 normal -> Kamel failure"
    if before and after:
        return "Both failure"
    return "Both normal"


def classification(delta: float) -> str:
    if delta > 0.10:
        return "Strong improvement"
    if delta > 0.03:
        return "Moderate improvement"
    if delta >= -0.03:
        return "Stable"
    if delta >= -0.10:
        return "Moderate regression"
    return "Strong regression"


def safe_delta(after: float | int | None, before: float | int | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def is_dice_lt_0p3(row: dict[str, Any]) -> bool:
    value = row.get("dice")
    if value in (None, ""):
        return False
    return float(value) < 0.3


def aggregate_summary(rows: list[dict[str, Any]], fp_volumes: list[float]) -> dict[str, Any]:
    result = summarize(rows)
    result["subgroups"] = {
        name: summarize([row for row in rows if low <= row["gt_ml"] < high])
        for name, low, high in [("<1", 0, 1), ("1-5", 1, 5), ("5-20", 5, 20), (">20", 20, float("inf"))]
    }
    ratios = [row["pred_gt_ratio"] for row in rows if row["pred_gt_ratio"] is not None]
    result["mean_pred_gt_ratio"] = float(np.mean(ratios))
    result["median_pred_gt_ratio"] = float(np.median(ratios))
    result["fn_lesions_per_case"] = result["lesion_fn"] / result["cases"]
    result["fp_topology"] = {
        "total_fp_components": int(sum(row["total_fp_components"] for row in rows)),
        "fp_volume_per_case_ml": float(sum(row["fp_total_volume_ml"] for row in rows) / len(rows)),
        "median_fp_component_volume_ml": float(np.median(fp_volumes)) if fp_volumes else 0.0,
        "fp_components_lt_0p1_ml": int(sum(row["fp_components_lt_0p1_ml"] for row in rows)),
        "fp_components_lt_0p1_ml_per_case": float(sum(row["fp_components_lt_0p1_ml"] for row in rows) / len(rows)),
        "fp_components_ge_1_ml": int(sum(row["fp_components_ge_1_ml"] for row in rows)),
        "fp_components_ge_1_ml_per_case": float(sum(row["fp_components_ge_1_ml"] for row in rows) / len(rows)),
        "predicted_component_count": int(sum(row["predicted_component_count"] for row in rows)),
        "total_predicted_foreground_volume_ml": float(sum(row["total_predicted_foreground_volume_ml"] for row in rows)),
    }
    result["large_partial_coverage_failure_count"] = int(sum(row["large_partial_coverage_failure"] for row in rows))
    return result


def metric_values(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "Mean Dice": summary["dice"]["mean"],
        "Median Dice": summary["dice"]["median"],
        "Dice SD": summary["dice"]["sd"],
        "Recall": summary["lesion_recall"],
        "Precision": summary["lesion_precision"],
        "F1": summary["lesion_f1"],
        "FP/case": summary["fp_lesions_per_case"],
        "FN/case": summary["fn_lesions_per_case"],
        "FP volume/case": summary["fp_topology"]["fp_volume_per_case_ml"],
        "Tiny FP/case": summary["fp_topology"]["fp_components_lt_0p1_ml_per_case"],
        ">=1 mL FP/case": summary["fp_topology"]["fp_components_ge_1_ml_per_case"],
        "1-5 mL Dice": summary["subgroups"]["1-5"]["dice"]["mean"],
        "1-5 mL Recall": summary["subgroups"]["1-5"]["lesion_recall"],
        "Severe under": summary["severe_under_count"],
        "Large partial coverage": summary["large_partial_coverage_failure_count"],
        "Dice<0.5": summary["dice_lt_05"],
        "Dice<0.3": summary["dice_lt_03"],
        "Volume MAE": summary["distributions"]["absolute_volume_error_ml"]["mean"],
        "Median Pred/GT": summary["median_pred_gt_ratio"],
    }


def main() -> None:
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if any(status[f"fold_{fold}"]["status"] != "complete" for fold in range(5)):
        raise RuntimeError("All five B0 folds must be complete before final analysis")
    kamel_status = json.loads((KAMEL_OUT / "status.json").read_text(encoding="utf-8"))
    if kamel_status.get("overall", {}).get("status") != "complete":
        raise RuntimeError("Kamel five-fold OOF is not complete")

    flats = [json.loads((OUT / f"fold_{fold}/fold_metrics_flat.json").read_text(encoding="utf-8")) for fold in range(5)]
    topologies = [json.loads((OUT / f"fold_{fold}/fp_topology_summary.json").read_text(encoding="utf-8")) for fold in range(5)]
    subgroups = [subgroup_map(fold) for fold in range(5)]
    fold_records = []
    for fold in range(5):
        metrics_dir, _ = fold_paths(fold)
        rows = json.loads((metrics_dir / "cases.json").read_text(encoding="utf-8"))
        fold_records.append(
            {
                **flats[fold],
                "FP volume/case": topologies[fold]["fp_volume_per_case_ml"],
                "FP <0.1 mL / case": topologies[fold]["fp_components_lt_0p1_ml"] / 50,
                "FP >=1 mL / case": topologies[fold]["fp_components_ge_1_ml"] / 50,
                "Large partial coverage failure count": sum(
                    row["gt_ml"] > 20 and row["pred_gt_ratio"] < 0.5 for row in rows
                ),
            }
        )
    five_fold_metrics = [
        "Mean Dice", "Median Dice", "Lesion Recall", "Lesion Precision", "Lesion F1", "FP/case", "FN/case",
        "FP volume/case", "FP <0.1 mL / case", "FP >=1 mL / case", "1-5 mL Dice", "1-5 mL Recall",
        "Severe under count", "Large partial coverage failure count", "Dice<0.5", "Dice<0.3", "Volume MAE", "Mean Pred/GT",
    ]
    five_fold_rows = []
    for metric in five_fold_metrics:
        values = np.asarray([float(record[metric]) for record in fold_records])
        five_fold_rows.append(
            {"Metric": metric, **{f"Fold{fold}": float(values[fold]) for fold in range(5)}, "Mean": float(values.mean()), "Std": float(values.std(ddof=0))}
        )
    write_csv(OUT / "B0_100_five_fold_summary.csv", five_fold_rows)

    split = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))
    oof_dir = SUITE / "OOF_predictions"
    oof_dir.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    all_fp_volumes: list[float] = []
    seen: set[str] = set()
    for fold in range(5):
        metrics_dir, predictions = fold_paths(fold)
        rows = json.loads((metrics_dir / "cases.json").read_text(encoding="utf-8"))
        if [row["case"] for row in rows] != split[fold]["val"]:
            raise RuntimeError(f"Fold {fold} evaluated case order differs from fixed split")
        for row in rows:
            case = row["case"]
            if case in seen:
                raise RuntimeError(f"Duplicate OOF case: {case}")
            seen.add(case)
            gt_img = nib.load(PP / "gt_segmentations" / f"{case}.nii.gz")
            pred_img = nib.load(predictions / f"{case}.nii.gz")
            gt = np.round(gt_img.get_fdata(dtype=np.float32)).astype(np.uint8) > 0
            pred = pred_img.get_fdata(dtype=np.float32) > 0.5
            voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3], dtype=np.float64) / 1000)
            components, _ = component_analysis(gt, pred, voxel_ml)
            topology = case_topology(gt, pred, voxel_ml)
            all_fp_volumes.extend(topology.pop("fp_component_volumes_ml"))
            probability = load_probability(predictions / f"{case}.npz", gt.shape, pred)
            probability_values, _ = probability_summary(probability, gt, pred)
            large_partial = bool(row["gt_ml"] > 20 and row["pred_gt_ratio"] < 0.5)
            enriched = {
                **row,
                **components,
                **topology,
                **probability_values,
                "fold": fold,
                "gt_voxels": int(gt.sum()),
                "pred_voxels": int(pred.sum()),
                "severe_underseg": bool(row["gt_ml"] > 0 and row["pred_gt_ratio"] < 0.5),
                "large_partial_coverage_failure": large_partial,
            }
            enriched.pop("gt_component_volumes_ml")
            enriched.pop("pred_component_volumes_ml")
            mode, labels = assign_failure_mode(enriched)
            enriched["failure_mode"] = mode
            enriched["failure_labels"] = labels + (";large_partial_coverage_failure" if large_partial else "")
            enriched["gt_component_ml"] = json.dumps(row["gt_component_ml"], separators=(",", ":"))
            enriched["pred_component_ml"] = json.dumps(row["pred_component_ml"], separators=(",", ":"))
            all_rows.append(enriched)
            for extension in [".nii.gz", ".npz"]:
                source = predictions / f"{case}{extension}"
                destination = oof_dir / source.name
                os.link(source, destination)
                manifest.append(
                    {"case": case, "fold": fold, "file_type": extension, "source_path": str(source), "oof_path": str(destination)}
                )

    all_ids = sorted(path.name[:-7] for path in (PP / "gt_segmentations").glob("*.nii.gz"))
    if seen != set(all_ids) or len(all_rows) != 250:
        raise RuntimeError("B0 OOF set is not an exact one-per-case cover of all 250 cases")
    all_rows.sort(key=lambda row: row["case"])
    write_csv(OUT / "B0_100_OOF_all_cases.csv", all_rows)
    write_csv(OUT / "B0_100_OOF_prediction_manifest.csv", manifest)
    b0_summary = aggregate_summary(all_rows, all_fp_volumes)
    (OUT / "B0_100_OOF_summary.json").write_text(
        json.dumps(b0_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    modes = Counter(row["failure_mode"] for row in all_rows)
    labels = Counter(label for row in all_rows for label in row["failure_labels"].split(";") if label)
    requested_modes = [
        "small_lesion_miss", "multifocal_miss", "large_partial_underseg", "boundary_underseg", "fp_dominant",
        "low_confidence", "mixed", "normal", "unclassified", "large_partial_coverage_failure",
    ]
    write_csv(
        OUT / "B0_100_OOF_failure_mode_counts.csv",
        [{"failure_mode": mode, "primary_count": modes.get(mode, 0), "expanded_label_presence": labels.get(mode, 0)} for mode in requested_modes],
    )

    kamel_summary = json.loads((KAMEL_OUT / "OOF_summary.json").read_text(encoding="utf-8"))
    b0_metrics = metric_values(b0_summary)
    kamel_metrics = metric_values(kamel_summary)
    summary_rows = []
    for metric in b0_metrics:
        b0_value = float(b0_metrics[metric])
        kamel_value = float(kamel_metrics[metric])
        summary_rows.append(
            {
                "Metric": metric,
                "B0-100 OOF": b0_value,
                "KamelRep-100 OOF": kamel_value,
                "Delta": kamel_value - b0_value,
                "Relative Delta": (kamel_value - b0_value) / b0_value if b0_value else None,
            }
        )
    write_csv(OUT / "B0_100_vs_KamelRep_100_summary.csv", summary_rows)

    kamel_rows_raw = read_csv(KAMEL_OUT / "OOF_all_cases.csv")
    kamel_by_case = {row["case"]: row for row in kamel_rows_raw}
    if set(kamel_by_case) != seen:
        raise RuntimeError("Kamel and B0 OOF case sets differ")
    paired_rows: list[dict[str, Any]] = []
    for b0 in all_rows:
        kamel = kamel_by_case[b0["case"]]
        if int(kamel["fold"]) != b0["fold"]:
            raise RuntimeError(f"Fold mismatch for {b0['case']}")
        k = {
            key: scalar(kamel[key], int if key in {"lesion_fp", "lesion_fn", "gt_lesion_count"} else float)
            for key in ["dice", "voxel_recall", "voxel_precision", "lesion_fp", "lesion_fn", "pred_gt_ratio", "pred_ml", "gt_ml", "gt_lesion_count"]
        }
        delta_dice = safe_delta(k["dice"], b0["dice"])
        paired_rows.append(
            {
                "Case": b0["case"],
                "Fold": b0["fold"],
                "GT volume mL": b0["gt_ml"],
                "GT lesion count": b0["gt_lesion_count"],
                "B0 Dice": b0["dice"],
                "B0 Recall": b0["voxel_recall"],
                "B0 Precision": b0["voxel_precision"],
                "B0 FP": b0["lesion_fp"],
                "B0 FN": b0["lesion_fn"],
                "B0 Pred/GT": b0["pred_gt_ratio"],
                "B0 Pred volume mL": b0["pred_ml"],
                "Kamel Dice": k["dice"],
                "Kamel Recall": k["voxel_recall"],
                "Kamel Precision": k["voxel_precision"],
                "Kamel FP": k["lesion_fp"],
                "Kamel FN": k["lesion_fn"],
                "Kamel Pred/GT": k["pred_gt_ratio"],
                "Kamel Pred volume mL": k["pred_ml"],
                "Delta Dice": delta_dice,
                "Delta Recall": safe_delta(k["voxel_recall"], b0["voxel_recall"]),
                "Delta Precision": safe_delta(k["voxel_precision"], b0["voxel_precision"]),
                "Delta FP": safe_delta(k["lesion_fp"], b0["lesion_fp"]),
                "Delta FN": safe_delta(k["lesion_fn"], b0["lesion_fn"]),
                "Delta Pred/GT": safe_delta(k["pred_gt_ratio"], b0["pred_gt_ratio"]),
                "Dice change class": classification(delta_dice) if delta_dice is not None else "Not applicable (empty GT)",
                "B0 failure mode": b0["failure_mode"],
                "Kamel failure mode": kamel["failure_mode"],
            }
        )
    paired_rows.sort(key=lambda row: row["Case"])
    write_csv(OUT / "paired_B0_vs_Kamel.csv", paired_rows)

    category_rows = []
    for category in ["Strong improvement", "Moderate improvement", "Stable", "Moderate regression", "Strong regression", "Not applicable (empty GT)"]:
        group = [row for row in paired_rows if row["Dice change class"] == category]
        volumes = [row["GT volume mL"] for row in group]
        lesions = [row["GT lesion count"] for row in group]
        failure_counts = Counter(row["B0 failure mode"] for row in group)
        category_rows.append(
            {
                "Category": category,
                "Case count": len(group),
                "GT volume median mL": float(np.median(volumes)) if volumes else None,
                "GT volume mean mL": float(np.mean(volumes)) if volumes else None,
                "GT volume min mL": min(volumes) if volumes else None,
                "GT volume max mL": max(volumes) if volumes else None,
                "Lesion count median": float(np.median(lesions)) if lesions else None,
                "Lesion count mean": float(np.mean(lesions)) if lesions else None,
                "B0 failure type counts": json.dumps(failure_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    write_csv(OUT / "improvement_regression_categories.csv", category_rows)

    b0_by_case = {row["case"]: row for row in all_rows}
    transition_rows: list[dict[str, Any]] = []
    failure_definitions = {
        "severe_under": lambda row: boolean(row["severe_underseg"]),
        "large_partial": lambda row: boolean(row["large_partial_coverage_failure"]),
        "small_lesion_miss": lambda row: has_label(row, "small_lesion_miss"),
        "multifocal_miss": lambda row: has_label(row, "multifocal_miss"),
        "dice_lt_0p3": is_dice_lt_0p3,
    }
    transition_counts: dict[str, Counter] = {name: Counter() for name in failure_definitions}
    for case in sorted(seen):
        b0 = b0_by_case[case]
        kamel = kamel_by_case[case]
        row: dict[str, Any] = {
            "Case": case,
            "Fold": b0["fold"],
            "GT volume mL": b0["gt_ml"],
            "GT lesion count": b0["gt_lesion_count"],
            "B0 failure mode": b0["failure_mode"],
            "Kamel failure mode": kamel["failure_mode"],
        }
        for name, detector in failure_definitions.items():
            before = detector(b0)
            after = detector(kamel)
            label = transition(before, after)
            row[f"B0 {name}"] = before
            row[f"Kamel {name}"] = after
            row[f"{name} transition"] = label
            transition_counts[name][label] += 1
        transition_rows.append(row)
    write_csv(OUT / "failure_transition_B0_vs_Kamel.csv", transition_rows)
    transition_summary_rows = []
    for name, counts in transition_counts.items():
        transition_summary_rows.append(
            {
                "Failure type": name,
                "B0 failure -> Kamel normal": counts["B0 failure -> Kamel normal"],
                "B0 normal -> Kamel failure": counts["B0 normal -> Kamel failure"],
                "Both failure": counts["Both failure"],
                "Both normal": counts["Both normal"],
                "Net rescued": counts["B0 failure -> Kamel normal"] - counts["B0 normal -> Kamel failure"],
            }
        )
    write_csv(OUT / "failure_transition_summary.csv", transition_summary_rows)

    kamel_flats = [json.loads((KAMEL_OUT / f"fold_{fold}/fold_metrics_flat.json").read_text(encoding="utf-8")) for fold in range(5)]
    foldwise_rows = []
    for fold in range(5):
        row: dict[str, Any] = {"Fold": fold}
        for label, key in [
            ("Mean Dice", "Mean Dice"),
            ("Recall", "Lesion Recall"),
            ("Precision", "Lesion Precision"),
            ("FP/case", "FP/case"),
            ("Severe under", "Severe under count"),
        ]:
            b0_value = float(flats[fold][key])
            kamel_value = float(kamel_flats[fold][key])
            row[f"B0 {label}"] = b0_value
            row[f"Kamel {label}"] = kamel_value
            row[f"Delta {label}"] = kamel_value - b0_value
        foldwise_rows.append(row)
    write_csv(OUT / "foldwise_B0_vs_Kamel.csv", foldwise_rows)

    dice_deltas = np.asarray([row["Delta Dice"] for row in paired_rows if row["Delta Dice"] is not None], dtype=float)
    fold_dice_deltas = np.asarray([row["Delta Mean Dice"] for row in foldwise_rows], dtype=float)
    improvement_cases = sum(delta > 0.03 for delta in dice_deltas)
    regression_cases = sum(delta < -0.03 for delta in dice_deltas)
    positive_folds = int(sum(fold_dice_deltas > 0))
    mean_delta = kamel_metrics["Mean Dice"] - b0_metrics["Mean Dice"]
    recall_delta = kamel_metrics["Recall"] - b0_metrics["Recall"]
    precision_delta = kamel_metrics["Precision"] - b0_metrics["Precision"]
    failure_net = sum(row["Net rescued"] for row in transition_summary_rows)
    if mean_delta > 0.01 and positive_folds >= 4 and improvement_cases > regression_cases * 1.25 and failure_net > 0:
        verdict = "A. CONSISTENT IMPROVEMENT"
    elif mean_delta < -0.01 and positive_folds <= 1 and regression_cases > improvement_cases * 1.25 and failure_net < 0:
        verdict = "D. B0 BETTER"
    elif abs(mean_delta) < 0.01:
        verdict = "C. STOCHASTICALLY EQUIVALENT"
    else:
        verdict = "B. SMALL / MIXED IMPROVEMENT" if mean_delta >= 0 else "D. B0 BETTER"

    paired_summary = {
        "cases": len(paired_rows),
        "cases_with_defined_dice_delta": int(len(dice_deltas)),
        "mean_delta_dice": float(dice_deltas.mean()),
        "median_delta_dice": float(np.median(dice_deltas)),
        "sd_delta_dice": float(dice_deltas.std(ddof=0)),
        "positive_delta_cases": int(sum(dice_deltas > 0)),
        "negative_delta_cases": int(sum(dice_deltas < 0)),
        "stable_abs_le_0p03_cases": int(sum(np.abs(dice_deltas) <= 0.03)),
        "improvement_gt_0p03_cases": int(improvement_cases),
        "regression_lt_minus_0p03_cases": int(regression_cases),
        "positive_mean_dice_folds": positive_folds,
        "verdict": verdict,
        "interpretation_constraint": "Configurations are technically identical; deltas estimate stochastic training variability and cannot be attributed to a Kamel method change.",
    }
    (OUT / "paired_comparison_summary.json").write_text(
        json.dumps(paired_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    mean_dices = np.asarray([record["Mean Dice"] for record in fold_records])
    paired_with_dice = [row for row in paired_rows if row["Delta Dice"] is not None]
    top_improved = sorted(paired_with_dice, key=lambda row: (-row["Delta Dice"], row["Case"]))[:10]
    top_regressed = sorted(paired_with_dice, key=lambda row: (row["Delta Dice"], row["Case"]))[:10]
    audit = json.loads((OUT / "configuration_audit.json").read_text(encoding="utf-8"))
    rescued_total = sum(row["B0 failure -> Kamel normal"] for row in transition_summary_rows)
    new_total = sum(row["B0 normal -> Kamel failure"] for row in transition_summary_rows)
    best_rescue = max(transition_summary_rows, key=lambda row: row["Net rescued"])
    worst_rescue = min(transition_summary_rows, key=lambda row: row["Net rescued"])
    small_b0 = b0_metrics["1-5 mL Dice"]
    small_kamel = kamel_metrics["1-5 mL Dice"]

    lines = [
        "# Original B0-100 full five-fold vs KamelRep-100 final report",
        "",
        "## Bottom line",
        "",
        f"**Verdict: {verdict}.** The exact audit found no effective technical configuration difference. This is a comparison of two independent five-fold trainings with the same configuration, so observed deltas quantify stochastic variability rather than a Kamel-method effect.",
        "",
        "## Fold completion and B0 results",
        "",
        "Original B0 Fold 0 passed the reuse gate and was not retrained. Folds 1–4 each completed exactly 100 epochs, checkpoint-final inference, and the fixed evaluator.",
        "",
        "| Fold | B0 Mean Dice | B0 Recall | B0 Precision | B0 FP/case | B0 Severe under | Kamel Mean Dice | Δ Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, row in enumerate(foldwise_rows):
        lines.append(
            f"| {fold} | {row['B0 Mean Dice']:.4f} | {row['B0 Recall']:.4f} | {row['B0 Precision']:.4f} | {row['B0 FP/case']:.3f} | {int(row['B0 Severe under'])} | {row['Kamel Mean Dice']:.4f} | {row['Delta Mean Dice']:+.4f} |"
        )
    lines += [
        "",
        f"B0 fold-mean Dice: **{mean_dices.mean():.4f} ± {mean_dices.std(ddof=0):.4f}** (population SD across five fold means).",
        "",
        "## 250-case OOF aggregate",
        "",
        "| Metric | B0-100 | KamelRep-100 | Kamel − B0 |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["Mean Dice", "Median Dice", "Dice SD", "Recall", "Precision", "F1", "FP/case", "FN/case", "FP volume/case", "Volume MAE", "Median Pred/GT", "Severe under", "Large partial coverage", "Dice<0.5", "Dice<0.3", "1-5 mL Dice", "1-5 mL Recall"]:
        lines.append(f"| {metric} | {b0_metrics[metric]:.4f} | {kamel_metrics[metric]:.4f} | {kamel_metrics[metric] - b0_metrics[metric]:+.4f} |")
    lines += [
        "",
        "## Paired case result",
        "",
        f"Across 250 paired cases ({len(dice_deltas)} with defined Dice; two empty-GT cases are not assigned a Dice delta), mean/median ΔDice (Kamel − B0) was {dice_deltas.mean():+.4f}/{np.median(dice_deltas):+.4f}; {improvement_cases} cases improved by >0.03, {regression_cases} regressed by >0.03, and {sum(np.abs(dice_deltas) <= 0.03)} were stable within ±0.03.",
        "",
        "Strongest apparent beneficiaries:",
        "",
    ]
    lines.extend(f"- `{row['Case']}` (Fold {row['Fold']}, GT {row['GT volume mL']:.3f} mL): ΔDice {row['Delta Dice']:+.4f}" for row in top_improved)
    lines += ["", "Strongest regressions:", ""]
    lines.extend(f"- `{row['Case']}` (Fold {row['Fold']}, GT {row['GT volume mL']:.3f} mL): ΔDice {row['Delta Dice']:+.4f}" for row in top_regressed)
    lines += [
        "",
        "## Failure transitions",
        "",
        "| Failure type | Rescued | Newly introduced | Both fail | Net rescued |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in transition_summary_rows:
        lines.append(
            f"| {row['Failure type']} | {row['B0 failure -> Kamel normal']} | {row['B0 normal -> Kamel failure']} | {row['Both failure']} | {row['Net rescued']:+d} |"
        )
    lines += [
        "",
        f"Summed across the five explicitly requested failure flags, Kamel rescued {rescued_total} flag instances and introduced {new_total}. The largest net improvement was `{best_rescue['Failure type']}` ({best_rescue['Net rescued']:+d}); the weakest/worst was `{worst_rescue['Failure type']}` ({worst_rescue['Net rescued']:+d}). Counts can overlap because one case may have multiple failure labels.",
        "",
        "## Direct answers",
        "",
        "1. **B0 Fold 1–4 是否全部完成？** 是；四個 fold 均完成 100 epochs、final checkpoint inference 與固定評估。Fold 0 通過 reuse gate，未重跑。",
        f"2. **B0 5-fold Mean ± SD？** {mean_dices.mean():.4f} ± {mean_dices.std(ddof=0):.4f}（五個 fold mean 的 population SD）。",
        f"3. **B0 250-case OOF performance？** Mean Dice {b0_metrics['Mean Dice']:.4f}、Recall {b0_metrics['Recall']:.4f}、Precision {b0_metrics['Precision']:.4f}、F1 {b0_metrics['F1']:.4f}。",
        f"4. **KamelRep OOF performance？** Mean Dice {kamel_metrics['Mean Dice']:.4f}、Recall {kamel_metrics['Recall']:.4f}、Precision {kamel_metrics['Precision']:.4f}、F1 {kamel_metrics['F1']:.4f}。",
        f"5. **Mean Dice 相差多少？** Kamel − B0 = {mean_delta:+.4f}。",
        f"6. **Recall 相差多少？** {recall_delta:+.4f}。",
        f"7. **Precision / F1 相差多少？** {precision_delta:+.4f} / {kamel_metrics['F1'] - b0_metrics['F1']:+.4f}。",
        f"8. **Severe under 相差多少？** B0 {int(b0_metrics['Severe under'])}、Kamel {int(kamel_metrics['Severe under'])}，差 {int(kamel_metrics['Severe under'] - b0_metrics['Severe under']):+d}。",
        f"9. **Large partial 相差多少？** B0 {int(b0_metrics['Large partial coverage'])}、Kamel {int(kamel_metrics['Large partial coverage'])}，差 {int(kamel_metrics['Large partial coverage'] - b0_metrics['Large partial coverage']):+d}。",
        f"10. **Small lesion performance 差多少？** 1–5 mL Dice {small_kamel - small_b0:+.4f}；Recall {kamel_metrics['1-5 mL Recall'] - b0_metrics['1-5 mL Recall']:+.4f}。",
        "11. **哪些 cases 明顯受益？** 見上方 strongest beneficiaries，完整清單在 paired CSV。",
        "12. **哪些 cases 明顯退步？** 見上方 strongest regressions，完整清單在 paired CSV。",
        f"13. **Fold-to-fold improvement 是否一致？** Kamel 的 Mean Dice 在 {positive_folds}/5 folds 較高；fold deltas 為 {[round(float(value), 4) for value in fold_dice_deltas]}。",
        f"14. **Technical configuration 有哪些實質差異？** 有效設定差異數為 {audit['effective_training_configuration_difference_count']}；只有 trainer wrapper/class 名稱不同，目的僅是讓原 Fold-0-only split selector 可選 Fold 1–4。",
        f"15. **Kamel-style replication 是否造成 gain？** 不可歸因為方法 gain；兩組設定相同。本次證據分類為 **{verdict}**，代表觀察到的是獨立訓練軌跡差異。",
        "16. **下一步？** 保留 OOF 較佳的 baseline 作為工程基準，但研究結論應先聚焦 paired/failure stability；若要宣稱改良，下一輪必須引入單一、可稽核的 configuration change。這一輪不自動啟動後續實驗。",
        "",
        "## Output index",
        "",
        "- `B0_100_OOF_all_cases.csv` / `B0_100_OOF_summary.json`: B0 exact 250-case OOF.",
        "- `B0_100_vs_KamelRep_100_summary.csv`: requested aggregate comparison.",
        "- `paired_B0_vs_Kamel.csv`: requested 250-case paired table and Dice-change class.",
        "- `failure_transition_B0_vs_Kamel.csv` / `failure_transition_summary.csv`: case-level and aggregate failure transitions.",
        "- `foldwise_B0_vs_Kamel.csv`: requested fold consistency comparison.",
        "- `B0_vs_Kamel_configuration_audit.md`: exact pre-training configuration audit.",
    ]
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    status["overall"] = {
        "status": "complete",
        "oof_cases": 250,
        "paired_cases": 250,
        "verdict": verdict,
        "ensemble_used": False,
        "followup_model_started": False,
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "b0_oof_mean_dice": b0_metrics["Mean Dice"], "kamel_oof_mean_dice": kamel_metrics["Mean Dice"], "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
