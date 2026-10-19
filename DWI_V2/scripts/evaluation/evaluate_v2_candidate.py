"""Full-volume, failure-priority evaluation and promotion decision for one V2 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

V2_SCRIPTS = Path(__file__).resolve().parents[1]
V1_SCRIPTS = V2_SCRIPTS.parent.parent / "DWI_V1" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))
sys.path.insert(0, str(V1_SCRIPTS))

from evaluate_experiment import metrics as case_metrics  # noqa: E402
from v2_common import (  # noqa: E402
    B0_100_PRED,
    E1_EXPERIMENT,
    S1_EXPERIMENT,
    V1_PP,
    V2_ROOT,
    now,
    save_json,
)


EXPERIMENTS = {"E1": E1_EXPERIMENT, "S1": S1_EXPERIMENT}
ANCHORS = ["case_0068", "case_0181", "case_0186", "case_0140"]
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 2022


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_case(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not set(np.unique(data)).issubset({0.0, 1.0}):
        raise RuntimeError(f"Invalid binary mask: {path}")
    return image, data > 0.5


def evaluate_folder(predictions: Path, ids: list[str]) -> list[dict]:
    rows = []
    gt_dir = V1_PP / "gt_segmentations"
    actual = {path.name.removesuffix(".nii.gz") for path in predictions.glob("case_*.nii.gz")}
    if actual != set(ids):
        raise RuntimeError(f"Prediction IDs differ from fixed Fold 0: missing={set(ids)-actual}, extra={actual-set(ids)}")
    for case in ids:
        gt_image, gt = load_case(gt_dir / f"{case}.nii.gz")
        pred_image, pred = load_case(predictions / f"{case}.nii.gz")
        if gt_image.shape != pred_image.shape or not np.allclose(gt_image.affine, pred_image.affine, atol=1e-3, rtol=0):
            raise RuntimeError(f"Prediction geometry mismatch: {case}")
        voxel_ml = float(abs(np.linalg.det(gt_image.affine[:3, :3])) / 1000)
        rows.append({"case": case, **case_metrics(gt, pred, gt_image.affine, voxel_ml)})
    return rows


def finite_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def finite_median(values) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def aggregate(rows: list[dict], fixed_worst5: list[str] | None = None, cohort_ids: list[str] | None = None) -> dict:
    lesion_tp = sum(row["lesion_tp"] for row in rows)
    lesion_fn = sum(row["lesion_fn"] for row in rows)
    lesion_fp = sum(row["lesion_fp"] for row in rows)
    result = {
        "cases": len(rows),
        "mean_dice": finite_mean(row["dice"] for row in rows),
        "median_dice": finite_median(row["dice"] for row in rows),
        "volume_mae_ml": finite_mean(row["absolute_volume_error_ml"] for row in rows),
        "signed_volume_error_ml": finite_mean(row["signed_volume_error_ml"] for row in rows),
        "hd95_mm": finite_mean(row["hd95_mm"] for row in rows),
        "lesion_tp": lesion_tp,
        "lesion_fn": lesion_fn,
        "lesion_fp": lesion_fp,
        "lesion_recall": lesion_tp / (lesion_tp + lesion_fn) if lesion_tp + lesion_fn else None,
        "lesion_precision": lesion_tp / (lesion_tp + lesion_fp) if lesion_tp + lesion_fp else None,
        "lesion_f1": 2 * lesion_tp / (2 * lesion_tp + lesion_fn + lesion_fp) if 2 * lesion_tp + lesion_fn + lesion_fp else None,
        "fp_lesions_per_case": lesion_fp / len(rows),
        "severe_under_count": sum(row["pred_gt_ratio"] is not None and row["pred_gt_ratio"] < 0.5 for row in rows),
        "severe_under_rate": sum(row["pred_gt_ratio"] is not None and row["pred_gt_ratio"] < 0.5 for row in rows) / len(rows),
        "dice_lt_0p5": sum(row["dice"] is not None and row["dice"] < 0.5 for row in rows),
        "dice_lt_0p3": sum(row["dice"] is not None and row["dice"] < 0.3 for row in rows),
        "catastrophic_miss_count": sum(row["lesion_recall"] == 0 for row in rows),
    }
    row_map = {row["case"]: row for row in rows}
    if fixed_worst5:
        selected = [row_map[case] for case in fixed_worst5]
        result["fixed_worst5_ids"] = fixed_worst5
        result["fixed_worst5_mean_dice"] = finite_mean(row["dice"] for row in selected)
        result["fixed_worst5_mean_voxel_recall"] = finite_mean(row["voxel_recall"] for row in selected)
        result["fixed_worst5_total_fn_ml"] = float(sum(row["voxel_fn"] * row["gt_ml"] / (row["voxel_tp"] + row["voxel_fn"]) for row in selected))
    if cohort_ids:
        selected = [row_map[case] for case in cohort_ids]
        result["coverage_cohort_ids"] = cohort_ids
        result["coverage_cohort_mean_dice"] = finite_mean(row["dice"] for row in selected)
        result["coverage_cohort_mean_voxel_recall"] = finite_mean(row["voxel_recall"] for row in selected)
        result["coverage_cohort_total_fn_ml"] = float(sum(row["voxel_fn"] * row["gt_ml"] / (row["voxel_tp"] + row["voxel_fn"]) for row in selected))
        result["coverage_cohort_mean_pred_gt_ratio"] = finite_mean(row["pred_gt_ratio"] for row in selected)
    return result


def metric_vector(rows: list[dict], key: str) -> float:
    if key == "mean_dice":
        return finite_mean(row["dice"] for row in rows)
    if key == "mean_voxel_recall":
        return finite_mean(row["voxel_recall"] for row in rows)
    if key == "lesion_recall":
        tp, fn = sum(row["lesion_tp"] for row in rows), sum(row["lesion_fn"] for row in rows)
        return tp / (tp + fn) if tp + fn else np.nan
    if key == "lesion_precision":
        tp, fp = sum(row["lesion_tp"] for row in rows), sum(row["lesion_fp"] for row in rows)
        return tp / (tp + fp) if tp + fp else np.nan
    if key == "severe_under_rate":
        return float(np.mean([row["pred_gt_ratio"] is not None and row["pred_gt_ratio"] < 0.5 for row in rows]))
    if key == "fp_per_case":
        return float(sum(row["lesion_fp"] for row in rows) / len(rows))
    raise KeyError(key)


def paired_bootstrap(base: list[dict], candidate: list[dict], ids: list[str], metrics: list[str], seed: int) -> dict:
    b = {row["case"]: row for row in base}
    c = {row["case"]: row for row in candidate}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in metrics}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(ids, size=len(ids), replace=True)
        b_rows = [b[case] for case in sample]
        c_rows = [c[case] for case in sample]
        for key in metrics:
            values[key].append(metric_vector(c_rows, key) - metric_vector(b_rows, key))
    result = {}
    b_rows = [b[case] for case in ids]
    c_rows = [c[case] for case in ids]
    for key in metrics:
        array = np.asarray(values[key], dtype=float)
        array = array[np.isfinite(array)]
        result[key] = {
            "delta": metric_vector(c_rows, key) - metric_vector(b_rows, key),
            "ci95": np.percentile(array, [2.5, 97.5]).tolist(),
            "valid_iterations": int(len(array)),
        }
    return result


def case0140_persistent_recovery(predictions: Path) -> dict:
    persistent_path = V2_ROOT / "outputs/predictions/case_0140/persistent_FN_all_models.nii.gz"
    persistent_image, persistent = load_case(persistent_path)
    pred_image, pred = load_case(predictions / "case_0140.nii.gz")
    if persistent_image.shape != pred_image.shape or not np.allclose(persistent_image.affine, pred_image.affine, atol=1e-3, rtol=0):
        raise RuntimeError("Persistent-FN geometry differs from candidate prediction")
    voxel_ml = float(abs(np.linalg.det(persistent_image.affine[:3, :3])) / 1000)
    recovered = persistent & pred
    return {
        "persistent_fn_reference_ml": float(persistent.sum() * voxel_ml),
        "persistent_fn_recovered_ml": float(recovered.sum() * voxel_ml),
        "persistent_fn_recovered_fraction": float(recovered.sum() / persistent.sum()),
    }


def decide(base: dict, candidate: dict, base_case: dict, candidate_case: dict, stability: dict, stage: str) -> dict:
    volume_margin = max(0.5, 0.10 * base["volume_mae_ml"])
    guards = {
        "severe_under_not_increased": candidate["severe_under_count"] <= base["severe_under_count"],
        "fixed_worst5_improved": candidate["fixed_worst5_mean_dice"] > base["fixed_worst5_mean_dice"] or candidate["fixed_worst5_mean_voxel_recall"] > base["fixed_worst5_mean_voxel_recall"],
        "anchor_or_cohort_coverage_improved": candidate_case["voxel_recall"] > base_case["voxel_recall"] or candidate["coverage_cohort_mean_voxel_recall"] > base["coverage_cohort_mean_voxel_recall"],
        "overall_lesion_recall_not_decreased": candidate["lesion_recall"] >= base["lesion_recall"],
        "precision_drop_within_0p05": candidate["lesion_precision"] >= base["lesion_precision"] - 0.05,
        "fp_per_case_increase_within_0p5": candidate["fp_lesions_per_case"] <= base["fp_lesions_per_case"] + 0.5,
        "volume_mae_not_materially_worse": candidate["volume_mae_ml"] <= base["volume_mae_ml"] + volume_margin,
        "numerically_stable": stability["total_nonfinite_gradient_events"] == 0 and stability["unrecovered_consecutive_overflows"] == 0,
    }
    anchor_improved = candidate_case["voxel_recall"] > base_case["voxel_recall"]
    cohort_improved = candidate["coverage_cohort_mean_voxel_recall"] > base["coverage_cohort_mean_voxel_recall"]
    worst_improved = guards["fixed_worst5_improved"]
    severe_improved = candidate["severe_under_count"] < base["severe_under_count"]
    all_pass = all(guards.values())
    if all_pass:
        classification = "FAILURE-ROBUSTNESS SIGNAL" if stage == "screen100" else "CANDIDATE FOR FULL TRAINING"
        promote = stage == "screen100"
    elif anchor_improved and not cohort_improved and not worst_improved and not severe_improved:
        classification = "ANCHOR-CASE SPECIFIC IMPROVEMENT"
        promote = False
    elif guards["anchor_or_cohort_coverage_improved"] and (
        not guards["precision_drop_within_0p05"] or not guards["fp_per_case_increase_within_0p5"]
    ):
        classification = "RECALL-PRECISION TRADE-OFF"
        promote = False
    else:
        classification = "STOP"
        promote = False
    return {
        "classification": classification,
        "promote_to_300": promote,
        "guards": guards,
        "volume_mae_allowed_margin_ml": volume_margin,
        "anchor_improved": anchor_improved,
        "coverage_cohort_improved": cohort_improved,
        "fixed_worst5_improved": worst_improved,
        "severe_under_improved": severe_improved,
    }


def markdown_report(experiment: str, stage: str, base: dict, candidate: dict, decision: dict, recovery: dict, path: Path) -> None:
    lines = [
        f"# V2-{experiment} {stage} failure-priority evaluation",
        "",
        f"Final classification: **{decision['classification']}**",
        "",
        "| Metric | B0-R2022-100 | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "mean_dice", "median_dice", "lesion_recall", "lesion_precision", "fp_lesions_per_case",
        "severe_under_count", "fixed_worst5_mean_dice", "fixed_worst5_mean_voxel_recall",
        "coverage_cohort_mean_dice", "coverage_cohort_mean_voxel_recall", "volume_mae_ml",
    ]:
        b, c = base[key], candidate[key]
        lines.append(f"| {key} | {b:.4f} | {c:.4f} | {c-b:+.4f} |")
    lines += [
        "",
        f"case_0140 persistent FN recovered: {recovery['persistent_fn_recovered_ml']:.3f} mL ({recovery['persistent_fn_recovered_fraction']:.1%}).",
        "",
        "## Promotion guards",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in decision["guards"].items())
    lines += ["", "No 1000-epoch training is authorized by this evaluation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=["screen100", "promote300"], required=True)
    args = parser.parse_args()
    root = EXPERIMENTS[args.experiment]
    train_completion = json.loads((root / args.stage / "train/completion.json").read_text(encoding="utf-8"))
    predictions = Path(train_completion["predictions"])
    output = root / args.stage / "evaluation"
    if output.exists():
        raise RuntimeError(f"Evaluation output exists; refusing overwrite: {output}")
    output.mkdir(parents=True)

    split = json.loads((V1_PP / "splits_final.json").read_text(encoding="utf-8"))[0]
    ids = list(split["val"])
    baseline_rows = evaluate_folder(B0_100_PRED, ids)
    candidate_rows = evaluate_folder(predictions, ids)
    baseline_map = {row["case"]: row for row in baseline_rows}
    candidate_map = {row["case"]: row for row in candidate_rows}
    fixed_worst5 = [row["case"] for row in sorted([row for row in baseline_rows if row["dice"] is not None], key=lambda row: row["dice"])[:5]]
    cohort_source = V2_ROOT / "outputs/comparisons/case_0140_like_cohort.csv"
    with cohort_source.open(encoding="utf-8-sig", newline="") as handle:
        cohort_ids = [row["case"] for row in csv.DictReader(handle) if row["selection_reason"] == "coverage_failure_neighbor"]
    if not cohort_ids or not set(cohort_ids).issubset(ids):
        raise RuntimeError(f"Invalid broader coverage cohort: {cohort_ids}")

    baseline_summary = aggregate(baseline_rows, fixed_worst5, cohort_ids)
    candidate_summary = aggregate(candidate_rows, fixed_worst5, cohort_ids)
    recovery = case0140_persistent_recovery(predictions)
    candidate_summary["case_0140_persistent_fn_recovery"] = recovery
    stability = {
        key: train_completion[key]
        for key in ["total_amp_skipped_steps", "total_nonfinite_gradient_events", "unrecovered_consecutive_overflows"]
    }
    decision = decide(
        baseline_summary,
        candidate_summary,
        baseline_map["case_0140"],
        candidate_map["case_0140"],
        stability,
        args.stage,
    )

    bootstrap = {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "candidate_minus_B0_R2022_100": paired_bootstrap(
            baseline_rows,
            candidate_rows,
            ids,
            ["mean_dice", "lesion_recall", "lesion_precision", "severe_under_rate", "fp_per_case"],
            BOOTSTRAP_SEED,
        ),
        "fixed_worst5_candidate_minus_baseline": paired_bootstrap(
            baseline_rows, candidate_rows, fixed_worst5, ["mean_dice", "mean_voxel_recall"], BOOTSTRAP_SEED + 1
        ),
        "coverage_cohort_candidate_minus_baseline": paired_bootstrap(
            baseline_rows, candidate_rows, cohort_ids, ["mean_dice", "mean_voxel_recall"], BOOTSTRAP_SEED + 2
        ),
        "horizon_note": "Matched at 100 epochs" if args.stage == "screen100" else "B0 is 100 epochs and therefore contextual; candidate 100-to-300 comparison is reported separately",
    }
    if args.stage == "promote300":
        prior_path = root / "screen100/evaluation/cases.csv"
        with prior_path.open(encoding="utf-8-sig", newline="") as handle:
            prior_rows = list(csv.DictReader(handle))
        numeric = {
            "dice", "voxel_tp", "voxel_fn", "voxel_fp", "voxel_recall", "voxel_precision", "gt_ml", "pred_ml",
            "signed_volume_error_ml", "absolute_volume_error_ml", "signed_percentage_error", "absolute_percentage_error",
            "pred_gt_ratio", "false_positive_ml", "hd95_mm", "lesion_tp", "lesion_fn", "lesion_fp", "lesion_recall", "lesion_precision", "lesion_f1",
        }
        for row in prior_rows:
            for key in numeric:
                if key in row:
                    row[key] = None if row[key] == "" else float(row[key])
            for key in ["lesion_tp", "lesion_fn", "lesion_fp", "voxel_tp", "voxel_fn", "voxel_fp"]:
                row[key] = int(row[key])
        bootstrap["candidate300_minus_candidate100"] = paired_bootstrap(
            prior_rows, candidate_rows, ids, ["mean_dice", "lesion_recall", "lesion_precision", "severe_under_rate", "fp_per_case"], BOOTSTRAP_SEED + 3
        )

    write_csv(output / "cases.csv", candidate_rows)
    write_csv(output / "baseline_cases.csv", baseline_rows)
    write_csv(output / "failure_anchors.csv", [candidate_map[case] for case in ANCHORS])
    write_csv(output / "fixed_worst5.csv", [candidate_map[case] for case in fixed_worst5])
    write_csv(output / "coverage_failure_cohort.csv", [candidate_map[case] for case in cohort_ids])
    save_json(output / "baseline_summary.json", baseline_summary, exclusive=True)
    save_json(output / "summary.json", candidate_summary, exclusive=True)
    save_json(output / "paired_bootstrap.json", bootstrap, exclusive=True)
    save_json(output / "decision.json", decision, exclusive=True)
    save_json(output / "completion.json", {"status": "complete", "time": now(), "classification": decision["classification"]}, exclusive=True)
    markdown_report(args.experiment, args.stage, baseline_summary, candidate_summary, decision, recovery, output / "report.md")
    print(json.dumps({"experiment": args.experiment, "stage": args.stage, "decision": decision, "recovery": recovery}, indent=2))


if __name__ == "__main__":
    main()

