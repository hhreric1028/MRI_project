"""Five-fold and exact 250-case OOF analysis for the approved Kamel replication."""

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
SUITE = ROOT / "experiments/kamel_replication_seed2022"
OUT = ROOT / "outputs/analysis/kamel_replication"
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
LITERATURE_MEAN = 0.810
LITERATURE_SD = 0.155
PROVENANCE = (
    "Fold 0 strict replication gate originally failed on lesion precision and FP/case.\n"
    "A dedicated FP audit subsequently found no source, split, inference, post-processing,\n"
    "or evaluation mismatch. Prediction masks remained highly overlapping, and most of the\n"
    "FP-count divergence was attributable to stochastic connected-component topology\n"
    "differences. After human review, the replication was accepted as stochastic-equivalent,\n"
    "and Fold 1–4 were approved without changing model or evaluation protocol。"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        path.write_text("case\n", encoding="utf-8-sig")
        return
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fold_paths(fold: int) -> tuple[Path, Path]:
    run = SUITE / f"fold_{fold}"
    trainer = "nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres"
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


def main() -> None:
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if any(status[f"fold_{fold}"]["status"] != "complete" for fold in range(5)):
        raise RuntimeError("All five folds must be complete before final analysis")
    check = json.loads((OUT / "fold0_replication_check.json").read_text(encoding="utf-8"))
    approval = json.loads((OUT / "human_approval_after_fold0_audit.json").read_text(encoding="utf-8"))
    if check["passed"]:
        raise RuntimeError("Recorded Fold 0 gate unexpectedly changed; expected preserved failure")
    if approval["decision"] != "approved_continue_folds_1_to_4" or approval["gate_redefined"]:
        raise RuntimeError("Human approval provenance is missing or invalid")

    flats = [json.loads((OUT / f"fold_{fold}/fold_metrics_flat.json").read_text(encoding="utf-8")) for fold in range(5)]
    topologies = [json.loads((OUT / f"fold_{fold}/fp_topology_summary.json").read_text(encoding="utf-8")) for fold in range(5)]
    subgroups = [subgroup_map(fold) for fold in range(5)]
    large_partial_by_fold = []
    for fold in range(5):
        metrics_dir, _ = fold_paths(fold)
        rows = json.loads((metrics_dir / "cases.json").read_text(encoding="utf-8"))
        large_partial_by_fold.append(sum(row["gt_ml"] > 20 and row["pred_gt_ratio"] < 0.5 for row in rows))

    fold_records = []
    for fold in range(5):
        fold_records.append({
            **flats[fold],
            "FP volume/case": topologies[fold]["fp_volume_per_case_ml"],
            "FP <0.1 mL / case": topologies[fold]["fp_components_lt_0p1_ml"] / 50,
            "FP >=1 mL / case": topologies[fold]["fp_components_ge_1_ml"] / 50,
            "Large partial coverage failure count": large_partial_by_fold[fold],
        })
    summary_metrics = [
        "Mean Dice", "Median Dice", "Lesion Recall", "Lesion Precision", "Lesion F1", "FP/case", "FN/case",
        "FP volume/case", "FP <0.1 mL / case", "FP >=1 mL / case", "1-5 mL Dice", "1-5 mL Recall",
        "Severe under count", "Large partial coverage failure count", "Dice<0.5", "Dice<0.3", "Volume MAE", "Mean Pred/GT",
    ]
    five_fold_rows = []
    for metric in summary_metrics:
        values = np.asarray([float(record[metric]) for record in fold_records], dtype=float)
        five_fold_rows.append({
            "Metric": metric,
            **{f"Fold{fold}": float(values[fold]) for fold in range(5)},
            "Mean": float(values.mean()),
            "Std": float(values.std(ddof=0)),
        })
    write_csv(OUT / "five_fold_summary.csv", five_fold_rows)

    split = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))
    oof_dir = SUITE / "OOF_predictions"
    oof_dir.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    manifest = []
    all_fp_volumes: list[float] = []
    seen = set()
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
                **row, **components, **topology, **probability_values,
                "fold": fold,
                "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum()),
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
                manifest.append({"case": case, "fold": fold, "file_type": extension, "source_path": str(source), "oof_path": str(destination)})

    all_ids = sorted(path.name[:-7] for path in (PP / "gt_segmentations").glob("*.nii.gz"))
    if seen != set(all_ids) or len(all_rows) != 250:
        raise RuntimeError("OOF set is not an exact one-per-case cover of all 250 cases")
    all_rows.sort(key=lambda row: row["case"])
    write_csv(OUT / "OOF_all_cases.csv", all_rows)
    write_csv(OUT / "OOF_prediction_manifest.csv", manifest)

    oof_summary = summarize(all_rows)
    oof_summary["subgroups"] = {
        name: summarize([row for row in all_rows if low <= row["gt_ml"] < high])
        for name, low, high in [("<1", 0, 1), ("1-5", 1, 5), ("5-20", 5, 20), (">20", 20, float("inf"))]
    }
    ratios = [row["pred_gt_ratio"] for row in all_rows if row["pred_gt_ratio"] is not None]
    oof_summary["mean_pred_gt_ratio"] = float(np.mean(ratios))
    oof_summary["median_pred_gt_ratio"] = float(np.median(ratios))
    oof_summary["fn_lesions_per_case"] = oof_summary["lesion_fn"] / oof_summary["cases"]
    oof_summary["fp_topology"] = {
        "total_fp_components": int(sum(row["total_fp_components"] for row in all_rows)),
        "fp_volume_per_case_ml": float(sum(row["fp_total_volume_ml"] for row in all_rows) / len(all_rows)),
        "median_fp_component_volume_ml": float(np.median(all_fp_volumes)) if all_fp_volumes else 0.0,
        "fp_components_lt_0p1_ml": int(sum(row["fp_components_lt_0p1_ml"] for row in all_rows)),
        "fp_components_lt_0p1_ml_per_case": float(sum(row["fp_components_lt_0p1_ml"] for row in all_rows) / len(all_rows)),
        "fp_components_ge_1_ml": int(sum(row["fp_components_ge_1_ml"] for row in all_rows)),
        "fp_components_ge_1_ml_per_case": float(sum(row["fp_components_ge_1_ml"] for row in all_rows) / len(all_rows)),
        "predicted_component_count": int(sum(row["predicted_component_count"] for row in all_rows)),
        "total_predicted_foreground_volume_ml": float(sum(row["total_predicted_foreground_volume_ml"] for row in all_rows)),
    }
    oof_summary["large_partial_coverage_failure_count"] = int(sum(row["large_partial_coverage_failure"] for row in all_rows))
    oof_summary["literature_reference"] = {
        "name": "Kamel et al. DWI-only nnU-Net", "mean_dice": LITERATURE_MEAN, "dice_sd": LITERATURE_SD,
        "warning": "Context only; evaluation protocol and split may not be identical.",
    }
    (OUT / "OOF_summary.json").write_text(json.dumps(oof_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    mode_counts = Counter(row["failure_mode"] for row in all_rows)
    label_counts = Counter(label for row in all_rows for label in row["failure_labels"].split(";") if label)
    requested = ["small_lesion_miss", "multifocal_miss", "large_partial_underseg", "boundary_underseg", "fp_dominant", "low_confidence", "mixed", "normal", "unclassified", "large_partial_coverage_failure"]
    write_csv(OUT / "OOF_failure_mode_counts.csv", [
        {"failure_mode": mode, "primary_count": mode_counts.get(mode, 0), "expanded_label_presence": label_counts.get(mode, 0)}
        for mode in requested
    ])
    large_partial_rows = [row for row in all_rows if row["large_partial_coverage_failure"]]
    large_fields = ["case", "fold", "gt_ml", "pred_ml", "pred_gt_ratio", "dice", "voxel_recall", "gt_lesion_count", "pred_lesion_count", "largest_gt_component_ml", "largest_gt_component_coverage", "failure_mode", "failure_labels"]
    write_csv(OUT / "OOF_large_partial_coverage_failures.csv", [{key: row.get(key) for key in large_fields} for row in large_partial_rows])

    mean_dices = np.asarray([record["Mean Dice"] for record in fold_records], dtype=float)
    recalls = np.asarray([record["Lesion Recall"] for record in fold_records], dtype=float)
    precisions = np.asarray([record["Lesion Precision"] for record in fold_records], dtype=float)
    fp_per_case = np.asarray([record["FP/case"] for record in fold_records], dtype=float)
    tiny_fp = np.asarray([record["FP <0.1 mL / case"] for record in fold_records], dtype=float)
    large_fp = np.asarray([record["FP >=1 mL / case"] for record in fold_records], dtype=float)
    small_dices = np.asarray([record["1-5 mL Dice"] for record in fold_records], dtype=float)
    small_recalls = np.asarray([record["1-5 mL Recall"] for record in fold_records], dtype=float)
    severe = np.asarray([record["Severe under count"] for record in fold_records], dtype=int)
    best_fold, worst_fold = int(np.argmax(mean_dices)), int(np.argmin(mean_dices))
    gap = LITERATURE_MEAN - oof_summary["dice"]["mean"]
    if mean_dices.max() < LITERATURE_MEAN and gap > 2 * mean_dices.std(ddof=0):
        variance_interpretation = "Fold variance alone is not sufficient to explain the literature-reference gap."
    elif mean_dices.min() <= LITERATURE_MEAN <= mean_dices.max():
        variance_interpretation = "The literature reference lies inside the observed fold range, so split effects may explain part of the gap."
    else:
        variance_interpretation = "Fold variance may explain part, but not all, of the literature-reference gap."
    variance = {
        "best_fold": best_fold, "best_fold_mean_dice": float(mean_dices[best_fold]),
        "worst_fold": worst_fold, "worst_fold_mean_dice": float(mean_dices[worst_fold]),
        "fold_mean_dice_mean": float(mean_dices.mean()), "fold_mean_dice_std": float(mean_dices.std(ddof=0)),
        "lesion_recall_mean": float(recalls.mean()), "lesion_recall_std": float(recalls.std(ddof=0)),
        "lesion_precision_mean": float(precisions.mean()), "lesion_precision_std": float(precisions.std(ddof=0)),
        "fp_per_case_mean": float(fp_per_case.mean()), "fp_per_case_std": float(fp_per_case.std(ddof=0)),
        "tiny_fp_per_case_mean": float(tiny_fp.mean()), "tiny_fp_per_case_std": float(tiny_fp.std(ddof=0)),
        "ge1ml_fp_per_case_mean": float(large_fp.mean()), "ge1ml_fp_per_case_std": float(large_fp.std(ddof=0)),
        "small_1_5ml_dice_mean": float(small_dices.mean()), "small_1_5ml_dice_std": float(small_dices.std(ddof=0)),
        "small_1_5ml_recall_mean": float(small_recalls.mean()), "small_1_5ml_recall_std": float(small_recalls.std(ddof=0)),
        "severe_under_by_fold": severe.tolist(), "large_partial_by_fold": large_partial_by_fold,
        "variance_interpretation": variance_interpretation,
    }
    (OUT / "fold_variance_analysis.json").write_text(json.dumps(variance, indent=2) + "\n", encoding="utf-8")
    status["overall"] = {"status": "complete", "oof_cases": len(all_rows), "oof_prediction_directory": str(oof_dir), "ensemble_used": False, "followup_model_started": False}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Kamel-style replication: five-fold final report", "", PROVENANCE, "",
        "## Protocol completion", "",
        "All five fixed folds completed 100 epochs with unchanged DWI-only B0 configuration. Each of the 250 cases appears exactly once in validation and contributes one checkpoint-final prediction from a model that did not train on that case. No ensemble inference was performed.", "",
        "The original Fold 0 gate remains recorded as failed; continuation was a documented human decision after the dedicated FP audit and did not redefine the gate.", "",
        "## Five-fold results", "",
        "| Fold | Mean Dice | Median Dice | Recall | Precision | FP/case | FP <0.1/case | FP >=1/case | Severe under | Large partial |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, record in enumerate(fold_records):
        lines.append(f"| {fold} | {record['Mean Dice']:.4f} | {record['Median Dice']:.4f} | {record['Lesion Recall']:.4f} | {record['Lesion Precision']:.4f} | {record['FP/case']:.3f} | {record['FP <0.1 mL / case']:.3f} | {record['FP >=1 mL / case']:.3f} | {int(record['Severe under count'])} | {large_partial_by_fold[fold]} |")
    lines += [
        "", f"Fold-level Mean Dice: **{mean_dices.mean():.4f} ± {mean_dices.std(ddof=0):.4f}** (population SD across five fold means). Best fold: {best_fold}; worst fold: {worst_fold}.", "",
        "## 250-case OOF aggregate", "",
        f"- Mean / median / SD Dice: {oof_summary['dice']['mean']:.4f} / {oof_summary['dice']['median']:.4f} / {oof_summary['dice']['sd']:.4f}",
        f"- Lesion recall / precision / F1: {oof_summary['lesion_recall']:.4f} / {oof_summary['lesion_precision']:.4f} / {oof_summary['lesion_f1']:.4f}",
        f"- FP / FN lesions per case: {oof_summary['fp_lesions_per_case']:.3f} / {oof_summary['fn_lesions_per_case']:.3f}",
        f"- FP volume/case: {oof_summary['fp_topology']['fp_volume_per_case_ml']:.4f} mL; FP <0.1 mL/case: {oof_summary['fp_topology']['fp_components_lt_0p1_ml_per_case']:.3f}; FP >=1 mL/case: {oof_summary['fp_topology']['fp_components_ge_1_ml_per_case']:.3f}",
        f"- Volume MAE: {oof_summary['distributions']['absolute_volume_error_ml']['mean']:.3f} mL; mean / median Pred/GT: {oof_summary['mean_pred_gt_ratio']:.4f} / {oof_summary['median_pred_gt_ratio']:.4f}",
        f"- Severe under / large partial coverage / Dice<0.5 / Dice<0.3: {oof_summary['severe_under_count']} / {len(large_partial_rows)} / {oof_summary['dice_lt_05']} / {oof_summary['dice_lt_03']}", "",
        "## Fold-to-fold variance", "",
        f"Recall SD {recalls.std(ddof=0):.4f}; precision SD {precisions.std(ddof=0):.4f}; FP/case SD {fp_per_case.std(ddof=0):.4f}; tiny-FP/case SD {tiny_fp.std(ddof=0):.4f}; >=1 mL FP/case SD {large_fp.std(ddof=0):.4f}.",
        f"Severe-under counts by fold: {severe.tolist()}. Large partial-coverage counts by fold: {large_partial_by_fold}.",
        f"1–5 mL Dice mean ± fold SD: {small_dices.mean():.4f} ± {small_dices.std(ddof=0):.4f}; recall mean ± fold SD: {small_recalls.mean():.4f} ± {small_recalls.std(ddof=0):.4f}.", "",
        "## Reassessment of Fold 0", "",
        f"Fold 0 Mean Dice is {mean_dices[0]:.4f}, versus the five-fold mean of fold means {mean_dices.mean():.4f}. Its recall/precision are {recalls[0]:.4f}/{precisions[0]:.4f}; comparison with the other folds is shown above. `case_0140` is {'one of the OOF large partial-coverage failures' if any(row['case'] == 'case_0140' for row in large_partial_rows) else 'not classified by the fixed >20 mL and Pred/GT <0.5 rule'}. Fold 0 tiny-FP and >=1 mL FP rates are {tiny_fp[0]:.3f} and {large_fp[0]:.3f} per case.", "",
        "## Literature context", "",
        f"The OOF mean Dice is {oof_summary['dice']['mean']:.4f}, versus the supplied Kamel et al. DWI-only reference of approximately {LITERATURE_MEAN:.3f} ± {LITERATURE_SD:.3f}; the mean difference is {oof_summary['dice']['mean'] - LITERATURE_MEAN:+.4f}. This is a method-level contextual comparison only: our result is five-fold OOF and the paper protocol/test set is not established as identical, so this is not a head-to-head superiority or inferiority claim.", "",
        "## Answers to the requested final questions", "",
        "1. **Fold 1–4 是否全部成功完成？** 是；status、checkpoint、100-epoch training summary 與 final-checkpoint validation 均完成。",
        "2. **每 fold Mean Dice / Recall / Precision？** 見 Five-fold results 表。",
        f"3. **5-fold Mean ± Std？** {mean_dices.mean():.4f} ± {mean_dices.std(ddof=0):.4f}。",
        f"4. **250-case OOF performance？** Mean Dice {oof_summary['dice']['mean']:.4f}、lesion recall {oof_summary['lesion_recall']:.4f}、precision {oof_summary['lesion_precision']:.4f}、F1 {oof_summary['lesion_f1']:.4f}。",
        f"5. **Fold 0 是否偏難？** Fold 0 Mean Dice 與五-fold mean 的差為 {mean_dices[0] - mean_dices.mean():+.4f}；其在五 fold 中排名 {int(np.argsort(-mean_dices).tolist().index(0) + 1)}/5。",
        f"6. **Small lesion performance 是否穩定？** 1–5 mL Dice/recall 的 fold SD 為 {small_dices.std(ddof=0):.4f}/{small_recalls.std(ddof=0):.4f}；各 fold n 明列於 subgroup outputs，小 n 不過度解讀。",
        f"7. **Severe under-segmentation 是否普遍？** OOF 共 {oof_summary['severe_under_count']} 例；各 fold為 {severe.tolist()}。",
        f"8. **0140-type large partial failure 有幾例？** 依固定 GT >20 mL 且 Pred/GT <0.5 規則共 {len(large_partial_rows)} 例。",
        f"9. **FP increase 是否主要為 tiny components？** OOF FP 中 <0.1 mL 共 {oof_summary['fp_topology']['fp_components_lt_0p1_ml']} 個（{oof_summary['fp_topology']['fp_components_lt_0p1_ml_per_case']:.3f}/case）；總 FP 為 {oof_summary['fp_topology']['total_fp_components']} 個。",
        f"10. **>=1 mL FP 是否常見？** 共 {oof_summary['fp_topology']['fp_components_ge_1_ml']} 個（{oof_summary['fp_topology']['fp_components_ge_1_ml_per_case']:.3f}/case）。",
        f"11. **Fold-to-fold stochastic / split variance？** Mean Dice fold SD {mean_dices.std(ddof=0):.4f}；recall/precision/FP-case SD 分別為 {recalls.std(ddof=0):.4f}/{precisions.std(ddof=0):.4f}/{fp_per_case.std(ddof=0):.4f}。",
        f"12. **與 literature reference 的距離？** Mean Dice 相差 {oof_summary['dice']['mean'] - LITERATURE_MEAN:+.4f}；{variance_interpretation}",
        "13. **下一步？** 先由人工依 OOF failure composition 與 literature protocol 差異決定：若重點是公平外部評估，優先建立 independent hold-out；若重點是提升現有系統，優先回到 failure-specific improvement。是否繼續模仿 paper protocol需先確認 protocol 可比性。本輪不自動啟動任何一項。", "",
        "## Output index", "",
        "- `five_fold_summary.csv`: fold metrics, fold mean, and fold SD.",
        "- `OOF_all_cases.csv`, `OOF_summary.json`: exact 250-case final-checkpoint OOF results.",
        "- `OOF_prediction_manifest.csv`, experiment `OOF_predictions/`: one correct-fold mask and probability file per case.",
        "- `OOF_failure_mode_counts.csv`, `OOF_large_partial_coverage_failures.csv`: failure analysis.",
        "- `fold_variance_analysis.json`: Dice, recall, precision, FP, tiny-FP, large-FP, and failure variance.",
        "- `fold_0` through `fold_4`: per-fold validation, topology, subgroup, severe-under, and Top-5 outputs.",
        "- `fold0_replication_check.json`: preserved strict gate failure; `human_approval_after_fold0_audit.json`: approval provenance.",
    ]
    (OUT / "final_report_5fold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "oof_cases": len(all_rows), "oof_mean_dice": oof_summary["dice"]["mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
