"""Per-fold failure monitoring and flat metric extraction for Kamel replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from failure_analysis_v2 import component_analysis


ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
DEFAULT_ANALYSIS = ROOT / "outputs/analysis/kamel_replication"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("case\n", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_status(status_path: Path, fold: int, **updates) -> None:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status[f"fold_{fold}"].update(updates)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(status_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    output = args.analysis / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.metrics / "cases.json").read_text(encoding="utf-8"))
    summary = json.loads((args.metrics / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != 50:
        raise RuntimeError(f"Fold {args.fold} expected 50 evaluated cases")
    severe_details = []
    for row in rows:
        if row["gt_ml"] <= 0 or row["pred_gt_ratio"] >= 0.5:
            continue
        case = row["case"]
        gt_img = nib.load(PP / "gt_segmentations" / f"{case}.nii.gz")
        pred_img = nib.load(args.predictions / f"{case}.nii.gz")
        gt = gt_img.get_fdata(dtype=np.float32) > 0.5
        pred = pred_img.get_fdata(dtype=np.float32) > 0.5
        voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3], dtype=np.float64) / 1000)
        components, _ = component_analysis(gt, pred, voxel_ml)
        severe_details.append(
            {
                "fold": args.fold,
                "case": case,
                "gt_ml": row["gt_ml"],
                "pred_ml": row["pred_ml"],
                "pred_gt_ratio": row["pred_gt_ratio"],
                "dice": row["dice"],
                "gt_lesion_count": components["gt_lesion_count"],
                "pred_lesion_count": components["pred_lesion_count"],
                "largest_gt_component_ml": components["largest_gt_component_ml"],
                "largest_gt_component_coverage": components["largest_gt_component_coverage"],
            }
        )
    sorted_nonempty = [row for row in rows if row["dice"] is not None]
    outputs = {
        "top5_worst_dice.csv": sorted(sorted_nonempty, key=lambda row: (row["dice"], row["case"]))[:5],
        "top5_lowest_pred_gt.csv": sorted(
            [row for row in rows if row["pred_gt_ratio"] is not None],
            key=lambda row: (row["pred_gt_ratio"], row["case"]),
        )[:5],
        "top5_fn_lesion_count.csv": sorted(rows, key=lambda row: (-row["lesion_fn"], row["case"]))[:5],
        "top5_lowest_lesion_recall.csv": sorted(
            [row for row in rows if row["lesion_recall"] is not None],
            key=lambda row: (row["lesion_recall"], row["case"]),
        )[:5],
        "severe_under_cases.csv": severe_details,
    }
    for name, data in outputs.items():
        write_csv(output / name, data)
    ratios = [row["pred_gt_ratio"] for row in rows if row["pred_gt_ratio"] is not None]
    flat = {
        "fold": args.fold,
        "validation_cases": summary["cases"],
        "Mean Dice": summary["dice"]["mean"],
        "Median Dice": summary["dice"]["median"],
        "Dice Std": summary["dice"]["sd"],
        "Lesion Recall": summary["lesion_recall"],
        "Lesion Precision": summary["lesion_precision"],
        "Lesion F1": summary["lesion_f1"],
        "FP/case": summary["fp_lesions_per_case"],
        "FN/case": summary["lesion_fn"] / summary["cases"],
        "Volume MAE": summary["distributions"]["absolute_volume_error_ml"]["mean"],
        "Mean Relative Volume Error": float(
            np.mean([row["signed_volume_error_ml"] / row["gt_ml"] for row in rows if row["gt_ml"] > 0])
        ),
        "Mean Pred/GT": float(np.mean(ratios)),
        "Median Pred/GT": float(np.median(ratios)),
        "Severe under count": summary["severe_under_count"],
        "Dice<0.5": summary["dice_lt_05"],
        "Dice<0.3": summary["dice_lt_03"],
        "1-5 mL Dice": summary["subgroups"]["1-5"]["dice"]["mean"],
        "1-5 mL Recall": summary["subgroups"]["1-5"]["lesion_recall"],
        "large_partial_under_count": sum(
            detail["gt_ml"] >= 5 and 0 < next(row for row in rows if row["case"] == detail["case"])["voxel_recall"] < 0.6
            for detail in severe_details
        ),
    }
    for subgroup in ["<1", "1-5", "5-20", ">20"]:
        sub = summary["subgroups"][subgroup]
        flat[f"{subgroup} n"] = sub["cases"]
        flat[f"{subgroup} Mean Dice"] = sub["dice"].get("mean")
        flat[f"{subgroup} Median Dice"] = sub["dice"].get("median")
        flat[f"{subgroup} Lesion Recall"] = sub["lesion_recall"]
    (output / "fold_metrics_flat.json").write_text(json.dumps(flat, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        f"# Fold {args.fold} validation summary",
        "",
        f"- Mean Dice: {flat['Mean Dice']:.4f}",
        f"- Median Dice: {flat['Median Dice']:.4f}",
        f"- Dice SD: {flat['Dice Std']:.4f}",
        f"- Lesion recall / precision / F1: {flat['Lesion Recall']:.4f} / {flat['Lesion Precision']:.4f} / {flat['Lesion F1']:.4f}",
        f"- FP / FN lesions per case: {flat['FP/case']:.3f} / {flat['FN/case']:.3f}",
        f"- Severe-under cases: {flat['Severe under count']}",
        f"- Large partial-under cases: {flat['large_partial_under_count']}",
        "",
        "## Subgroups",
        "",
        "| Group | n | Mean Dice | Median Dice | Lesion recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for subgroup in ["<1", "1-5", "5-20", ">20"]:
        lines.append(
            f"| {subgroup} | {flat[f'{subgroup} n']} | {flat[f'{subgroup} Mean Dice']} | {flat[f'{subgroup} Median Dice']} | {flat[f'{subgroup} Lesion Recall']} |"
        )
    lines += [
        "",
        "Small subgroup sizes are shown explicitly and should not be overinterpreted.",
        "",
        "The four top-five lists and severe-under case details are saved beside this summary.",
    ]
    (output / "fold_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_status(args.analysis / "status.json", args.fold, status="complete", validation_status="complete")
    print(json.dumps(flat, indent=2), flush=True)


if __name__ == "__main__":
    main()
