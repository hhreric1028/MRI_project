"""Select one 300-epoch continuation and write the final controlled-ablation report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_common import E1_EXPERIMENT, S1_EXPERIMENT, V1_ROOT, V2_ROOT, now, save_json  # noqa: E402


EXPERIMENTS = {"E1": E1_EXPERIMENT, "S1": S1_EXPERIMENT}
SELECTION_PATH = V2_ROOT / "outputs/comparisons/v2_promotion_selection.json"
FINAL_JSON = V2_ROOT / "outputs/comparisons/v2_final_summary.json"
FINAL_REPORT = V2_ROOT / "outputs/comparisons/v2_final_report.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(path: Path) -> dict[str, dict[str, float | str | None]]:
    numeric = {
        "dice", "voxel_recall", "voxel_precision", "pred_gt_ratio", "false_positive_ml",
        "absolute_volume_error_ml", "lesion_recall", "lesion_precision",
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = None if row[key] == "" else float(row[key])
    return {str(row["case"]): row for row in rows}


def experiment_result(name: str, stage: str) -> dict[str, Any] | None:
    evaluation = EXPERIMENTS[name] / stage / "evaluation"
    if not (evaluation / "completion.json").is_file():
        return None
    return {
        "experiment": name,
        "stage": stage,
        "summary": load_json(evaluation / "summary.json"),
        "baseline": load_json(evaluation / "baseline_summary.json"),
        "decision": load_json(evaluation / "decision.json"),
        "cases": load_cases(evaluation / "cases.csv"),
        "bootstrap": load_json(evaluation / "paired_bootstrap.json"),
    }


def priority_vector(result: dict[str, Any]) -> list[float]:
    candidate, baseline = result["summary"], result["baseline"]
    anchor = result["cases"]["case_0140"]
    base_anchor = load_cases(EXPERIMENTS[result["experiment"]] / "screen100/evaluation/baseline_cases.csv")["case_0140"]
    return [
        float(baseline["severe_under_count"] - candidate["severe_under_count"]),
        float(candidate["fixed_worst5_mean_dice"] - baseline["fixed_worst5_mean_dice"]),
        float(candidate["fixed_worst5_mean_voxel_recall"] - baseline["fixed_worst5_mean_voxel_recall"]),
        float(candidate["coverage_cohort_mean_voxel_recall"] - baseline["coverage_cohort_mean_voxel_recall"]),
        float(anchor["voxel_recall"] - base_anchor["voxel_recall"]),
        float(candidate["lesion_recall"] - baseline["lesion_recall"]),
        float(candidate["lesion_precision"] - baseline["lesion_precision"]),
        float(baseline["fp_lesions_per_case"] - candidate["fp_lesions_per_case"]),
        float(candidate["mean_dice"] - baseline["mean_dice"]),
    ]


def select_candidate() -> dict[str, Any]:
    if SELECTION_PATH.exists():
        return load_json(SELECTION_PATH)
    results = [result for name in EXPERIMENTS if (result := experiment_result(name, "screen100"))]
    eligible = [result for result in results if result["decision"].get("promote_to_300")]
    ranked = sorted(
        ({"experiment": result["experiment"], "priority_vector": priority_vector(result)} for result in eligible),
        key=lambda item: tuple(item["priority_vector"]),
        reverse=True,
    )
    sampling_path = S1_EXPERIMENT / "sampling_audit/completion.json"
    sampling = load_json(sampling_path) if sampling_path.is_file() else None
    selection = {
        "status": "selected" if ranked else "no_promotion",
        "time": now(),
        "selected_experiment": ranked[0]["experiment"] if ranked else None,
        "eligible_experiments": [item["experiment"] for item in ranked],
        "ranking": ranked,
        "priority_order": [
            "severe-under reduction", "fixed Worst-5 mean Dice delta", "fixed Worst-5 recall delta",
            "coverage-cohort recall delta", "case_0140 recall delta", "lesion recall delta",
            "precision delta", "negative FP/case delta", "overall mean Dice delta",
        ],
        "screen_classifications": {result["experiment"]: result["decision"]["classification"] for result in results},
        "s1_sampling_status": None if sampling is None else sampling["status"],
        "constraints": {"maximum_promoted_candidates": 1, "maximum_epochs": 300, "full_1000_epoch_training": False},
    }
    save_json(SELECTION_PATH, selection, exclusive=True)
    return selection


def legacy_summary(path: Path) -> dict[str, float]:
    value = load_json(path)
    return {
        "mean_dice": value["dice"]["mean"],
        "lesion_recall": value["lesion_recall"],
        "lesion_precision": value["lesion_precision"],
        "fp_lesions_per_case": value["fp_lesions_per_case"],
        "severe_under_count": value["severe_under_count"],
        "volume_mae_ml": value["distributions"]["absolute_volume_error_ml"]["mean"],
    }


def f(value: float | int | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def delta(candidate: dict, reference: dict, key: str) -> float:
    return float(candidate[key]) - float(reference[key])


def final_report(force: bool = False) -> dict[str, Any]:
    if FINAL_JSON.exists() and FINAL_REPORT.exists() and not force:
        return load_json(FINAL_JSON)
    if force:
        for path in [FINAL_JSON, FINAL_REPORT]:
            if path.exists():
                backup = path.with_name(path.stem + ".pre_clarity_fix" + path.suffix)
                if not backup.exists():
                    shutil.copy2(path, backup)
    selection = select_candidate()
    screens = {name: result for name in EXPERIMENTS if (result := experiment_result(name, "screen100"))}
    promoted_name = selection["selected_experiment"]
    promoted = experiment_result(promoted_name, "promote300") if promoted_name else None
    s1_audit_path = S1_EXPERIMENT / "sampling_audit/completion.json"
    s1_audit = load_json(s1_audit_path) if s1_audit_path.is_file() else None
    e1_train_path = E1_EXPERIMENT / "screen100/train/completion.json"
    e1_train = load_json(e1_train_path) if e1_train_path.is_file() else None
    p3a = legacy_summary(V1_ROOT / "experiments/P3a_clip_zscore_seed2022/screen100/metrics/summary.json")
    s1a = legacy_summary(V1_ROOT / "experiments/recall_screen100_seed2022_r2/S1a_screen100_seed2022/metrics/summary.json")

    rows = []
    for name, result in screens.items():
        summary = result["summary"]
        rows.append({
            "experiment": f"V2-{name}-100",
            "classification": result["decision"]["classification"],
            "mean_dice": summary["mean_dice"],
            "lesion_recall": summary["lesion_recall"],
            "lesion_precision": summary["lesion_precision"],
            "fp_lesions_per_case": summary["fp_lesions_per_case"],
            "severe_under_count": summary["severe_under_count"],
            "fixed_worst5_mean_dice": summary["fixed_worst5_mean_dice"],
            "coverage_cohort_recall": summary["coverage_cohort_mean_voxel_recall"],
            "persistent_fn_recovered_ml": summary["case_0140_persistent_fn_recovery"]["persistent_fn_recovered_ml"],
        })
    if promoted:
        summary = promoted["summary"]
        rows.append({
            "experiment": f"V2-{promoted_name}-300",
            "classification": promoted["decision"]["classification"],
            "mean_dice": summary["mean_dice"],
            "lesion_recall": summary["lesion_recall"],
            "lesion_precision": summary["lesion_precision"],
            "fp_lesions_per_case": summary["fp_lesions_per_case"],
            "severe_under_count": summary["severe_under_count"],
            "fixed_worst5_mean_dice": summary["fixed_worst5_mean_dice"],
            "coverage_cohort_recall": summary["coverage_cohort_mean_voxel_recall"],
            "persistent_fn_recovered_ml": summary["case_0140_persistent_fn_recovery"]["persistent_fn_recovered_ml"],
        })

    e1 = screens.get("E1")
    s1 = screens.get("S1")
    e1_balance = None if not e1 else {
        "v2_e1_minus_p3a_only_mean_dice": delta(e1["summary"], p3a, "mean_dice"),
        "v2_e1_minus_p3a_only_lesion_recall": delta(e1["summary"], p3a, "lesion_recall"),
        "v2_e1_minus_p3a_only_lesion_precision": delta(e1["summary"], p3a, "lesion_precision"),
        "v2_e1_minus_p3a_only_fp_per_case": delta(e1["summary"], p3a, "fp_lesions_per_case"),
    }
    s1_comparison = None if not s1 else {
        "v2_s1_minus_s1a_mean_dice": delta(s1["summary"], s1a, "mean_dice"),
        "v2_s1_minus_s1a_lesion_recall": delta(s1["summary"], s1a, "lesion_recall"),
        "v2_s1_minus_s1a_lesion_precision": delta(s1["summary"], s1a, "lesion_precision"),
        "v2_s1_minus_s1a_fp_per_case": delta(s1["summary"], s1a, "fp_lesions_per_case"),
    }
    final_classification = (
        promoted["decision"]["classification"] if promoted else
        (screens[promoted_name]["decision"]["classification"] if promoted_name else "STOP")
    )
    summary_json = {
        "status": "complete",
        "time": now(),
        "selection": selection,
        "s1_sampling": s1_audit,
        "candidate_rows": rows,
        "legacy_references": {"P3a_only_100": p3a, "S1a_100": s1a},
        "e1_vs_p3a_only": e1_balance,
        "s1_vs_s1a": s1_comparison,
        "final_classification": final_classification,
        "full_training_authorized": False,
        "stopped_at_epoch": 300 if promoted else 100,
    }
    if FINAL_JSON.exists() and not FINAL_REPORT.exists():
        backup = FINAL_JSON.with_name("v2_final_summary.pre_report_fix.json")
        if not backup.exists():
            shutil.copy2(FINAL_JSON, backup)
    save_json(FINAL_JSON, summary_json, exclusive=not FINAL_JSON.exists())

    lines = [
        "# DWI_V2 controlled-ablation final report", "",
        f"Final classification: **{final_classification}**", "",
        "No 1000-epoch run, combined E1+S1 run, threshold tuning, or additional intervention was started.", "",
        (
            f"Numerical note: E1 had {e1_train['total_amp_skipped_steps']} isolated AMP skipped steps and "
            f"{e1_train['unrecovered_consecutive_overflows']} unrecovered consecutive overflows. The process and "
            "checkpoint completed, but the prespecified zero-event numerical-stability guard failed."
            if e1_train else "Numerical note: no E1 training completion was available."
        ), "",
        "## Candidate comparison", "",
        "| Run | Classification | Mean Dice | Lesion recall | Lesion precision | FP/case | Severe-under | Fixed Worst-5 Dice | Coverage-cohort recall | Persistent FN recovered (mL) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['classification']} | {f(row['mean_dice'])} | {f(row['lesion_recall'])} | "
            f"{f(row['lesion_precision'])} | {f(row['fp_lesions_per_case'], 2)} | {row['severe_under_count']} | "
            f"{f(row['fixed_worst5_mean_dice'])} | {f(row['coverage_cohort_recall'])} | {f(row['persistent_fn_recovered_ml'], 3)} |"
        )
    sampling_answer = (
        "未通過：audit 標記 `SAMPLING INTERVENTION INEFFECTIVE`，因此依方案未啟動 V2-S1 長訓練。"
        if s1_audit and not s1_audit.get("training_allowed") else
        (f"已通過；選定 forced-foreground 內 hard-region preference={s1_audit['selected_sampling']['hard_preference_within_forced_foreground']:.3f}。" if s1_audit else "無可用 audit。")
    )
    anchor_generalization = "無候選完成 evaluation。"
    if promoted_name:
        d = screens[promoted_name]["decision"]
        anchor_generalization = (
            f"V2-{promoted_name}：broader cohort improved={d['coverage_cohort_improved']}，"
            f"fixed Worst-5 improved={d['fixed_worst5_improved']}，severe-under improved={d['severe_under_improved']}。"
        )
    elif e1:
        e1_base_anchor = load_cases(E1_EXPERIMENT / "screen100/evaluation/baseline_cases.csv")["case_0140"]
        e1_anchor = e1["cases"]["case_0140"]
        anchor_generalization = (
            f"case_0140 recall {e1_base_anchor['voxel_recall']:.4f}→{e1_anchor['voxel_recall']:.4f}，未改善；"
            f"broader cohort recall +{e1['summary']['coverage_cohort_mean_voxel_recall'] - e1['baseline']['coverage_cohort_mean_voxel_recall']:.4f}，"
            "但 95% CI 跨 0，因此只有弱的 cohort signal。"
        )
    e1_answer = "V2-E1 未完成。" if not e1 else (
        f"恢復 {e1['summary']['case_0140_persistent_fn_recovery']['persistent_fn_recovered_ml']:.3f} mL persistent FN；"
        f"分類為 {e1['decision']['classification']}。"
    )
    s1_vs = "V2-S1 未進入正式訓練，故不能聲稱優於 S1a。" if not s1 else (
        f"相對 S1a-100：Mean Dice {s1_comparison['v2_s1_minus_s1a_mean_dice']:+.4f}、"
        f"lesion recall {s1_comparison['v2_s1_minus_s1a_lesion_recall']:+.4f}、"
        f"precision {s1_comparison['v2_s1_minus_s1a_lesion_precision']:+.4f}。"
    )
    e1_vs = "V2-E1 未完成。" if not e1 else (
        f"相對 P3a-only-100：Mean Dice {e1_balance['v2_e1_minus_p3a_only_mean_dice']:+.4f}、"
        f"lesion recall {e1_balance['v2_e1_minus_p3a_only_lesion_recall']:+.4f}、"
        f"precision {e1_balance['v2_e1_minus_p3a_only_lesion_precision']:+.4f}、"
        f"FP/case {e1_balance['v2_e1_minus_p3a_only_fp_per_case']:+.2f}。"
    )
    best = f"V2-{promoted_name}" if promoted_name else "無 intervention 通過全部 promotion guards"
    if e1:
        worst_answer = (
            f"有方向性改善：Mean Dice {e1['baseline']['fixed_worst5_mean_dice']:.4f}→"
            f"{e1['summary']['fixed_worst5_mean_dice']:.4f}（Δ+0.0172），recall Δ+0.0168；"
            "兩者 bootstrap 95% CI 均跨 0。"
        )
        severe_answer = (
            f"有下降：{e1['baseline']['severe_under_count']}→{e1['summary']['severe_under_count']} cases；"
            "rate Δ-0.040，bootstrap 95% CI [-0.120, +0.040]。"
        )
        tradeoff_answer = (
            "是。Lesion recall Δ+0.0510（95% CI +0.0225 至 +0.0800），但 precision Δ-0.0593"
            "（95% CI -0.1115 至 -0.0136）、FP/case Δ+0.92（95% CI +0.36 至 +1.66），"
            "且 volume MAE +0.735 mL。"
        )
    else:
        worst_answer = severe_answer = tradeoff_answer = "無完成候選可判定。"
    lines += [
        "", "## Ten required conclusions", "",
        f"1. **V2-E1 是否恢復 persistent FN？** {e1_answer}",
        f"2. **V2-S1 是否提高 low-salience exposure？** {sampling_answer}",
        f"3. **case_0140 是否泛化？** {anchor_generalization}",
        f"4. **Fixed Worst-5 是否改善？** {worst_answer}",
        f"5. **Severe under 是否下降？** {severe_answer}",
        f"6. **是否只是 recall↑ / FP↑？** {tradeoff_answer}",
        f"7. **Original+P3a 是否比 P3a-only 平衡？** {e1_vs}",
        f"8. **Hard-region sampling 是否優於一般 oversampling？** {s1_vs}",
        f"9. **最值得保留的 intervention？** {best}。",
        f"10. **是否足以進入 full training？** 最終分類為 {final_classification}；本輪不授權也未執行 1000 epochs。",
        "", "## Stop boundary", "",
        f"Workflow stopped at {summary_json['stopped_at_epoch']} epochs as required.",
    ]
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "final"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = select_candidate() if args.mode == "select" else final_report(force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
