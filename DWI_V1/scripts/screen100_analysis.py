"""Paired case-level analysis and promotion decisions for 100-epoch screening."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from recall_analysis import bootstrap, load_cases, stat
from recall_common import ROOT, save


SUITE = ROOT / "experiments/recall_screen100_seed2022_r2"
REUSED_SAMPLING_AUDIT = ROOT / "experiments/recall_screen100_seed2022/sampling_audit/summary.json"
BASELINE = "B0_R2022_screen100"
CANDIDATES = ["S1a_screen100_seed2022", "L2a_screen100_seed2022"]

# Operational screening guards. These are triage thresholds, not claims of significance.
MAX_PRECISION_DROP = 0.05
MAX_FP_LESIONS_PER_CASE_INCREASE = 0.5


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_summary(name: str) -> dict:
    return json.loads((SUITE / name / "metrics/summary.json").read_text())


def curve_status(name: str) -> dict:
    path = SUITE / name / "train/epoch_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    numeric_keys = ["train_loss", "validation_loss", "patch_pseudo_dice", "learning_rate"]
    finite = len(rows) == 100 and all(np.isfinite(row[key]) for row in rows for key in numeric_keys)
    lr = np.asarray([row["learning_rate"] for row in rows], dtype=float)
    expected_final = 0.01 * (1 - 99 / 1000) ** 0.9
    completion = json.loads((SUITE / name / "train/completion.json").read_text())
    return {
        "epochs_recorded": len(rows),
        "finite_metrics": bool(finite),
        "lr_monotonic_nonincreasing": bool(np.all(np.diff(lr) <= 1e-14)),
        "final_lr": float(lr[-1]),
        "expected_final_lr_for_epoch_index_99_of_1000": expected_final,
        "final_lr_matches": bool(np.isclose(lr[-1], expected_final, rtol=1e-12, atol=1e-14)),
        "unrecovered_consecutive_overflows": completion["unrecovered_consecutive_overflows"],
        "total_amp_skipped_steps": completion["total_amp_skipped_steps"],
        "normal": bool(
            finite
            and np.all(np.diff(lr) <= 1e-14)
            and np.isclose(lr[-1], expected_final, rtol=1e-12, atol=1e-14)
            and completion["unrecovered_consecutive_overflows"] == 0
        ),
    }


def subgroup_ids(cases: dict, low: float, high: float) -> list[str]:
    return sorted(case for case, row in cases.items() if low <= row["gt_ml"] < high)


def metric_snapshot(summary: dict) -> dict:
    small = summary["subgroups"]["1-5"]
    return {
        "mean_dice": summary["dice"]["mean"],
        "median_dice": summary["dice"]["median"],
        "dice_sd": summary["dice"]["sd"],
        "hd95_mean_mm": summary["distributions"]["hd95_mm"]["mean"],
        "volume_mae_ml": summary["distributions"]["absolute_volume_error_ml"]["mean"],
        "signed_volume_error_ml": summary["distributions"]["signed_volume_error_ml"]["mean"],
        "lesion_recall": summary["lesion_recall"],
        "lesion_precision": summary["lesion_precision"],
        "lesion_f1": summary["lesion_f1"],
        "lesion_fn": summary["lesion_fn"],
        "lesion_fp": summary["lesion_fp"],
        "fp_lesions_per_case": summary["fp_lesions_per_case"],
        "severe_under_count": summary["severe_under_count"],
        "dice_lt_05": summary["dice_lt_05"],
        "dice_lt_03": summary["dice_lt_03"],
        "catastrophic_miss_count": summary["catastrophic_miss_count"],
        "small_1_5_mean_dice": small["dice"]["mean"],
        "small_1_5_median_dice": small["dice"]["median"],
        "small_1_5_lesion_recall": small["lesion_recall"],
        "small_1_5_lesion_fn": small["lesion_fn"],
        "small_1_5_volume_mae_ml": small["distributions"]["absolute_volume_error_ml"]["mean"],
    }


def decision(baseline: dict, candidate: dict, curve: dict, is_l2a: bool) -> dict:
    eps = 1e-12
    recall_improved = candidate["lesion_recall"] > baseline["lesion_recall"] + eps
    severe_under_improved = candidate["severe_under_count"] < baseline["severe_under_count"]
    small_dice_delta = candidate["small_1_5_mean_dice"] - baseline["small_1_5_mean_dice"]
    small_recall_delta = candidate["small_1_5_lesion_recall"] - baseline["small_1_5_lesion_recall"]
    small_improved = small_dice_delta >= -eps and small_recall_delta >= -eps and (
        small_dice_delta > eps or small_recall_delta > eps
    )
    core = {
        "lesion_recall_improved": recall_improved,
        "severe_under_count_reduced": severe_under_improved,
        "small_1_5_dice_recall_consistent_improvement": small_improved,
    }
    core_count = sum(core.values())
    precision_delta = candidate["lesion_precision"] - baseline["lesion_precision"]
    fp_delta = candidate["fp_lesions_per_case"] - baseline["fp_lesions_per_case"]
    guards = {
        "precision_drop_within_0.05": precision_delta >= -MAX_PRECISION_DROP,
        "fp_lesions_per_case_increase_within_0.5": fp_delta <= MAX_FP_LESIONS_PER_CASE_INCREASE,
        "training_curve_and_numerics_normal": curve["normal"],
    }
    if core_count >= 2 and all(guards.values()):
        if is_l2a and candidate["mean_dice"] < baseline["mean_dice"]:
            label = "PROMISING BUT NOT CONVERGED"
        else:
            label = "PROMOTE TO 300 EPOCHS"
    else:
        label = "STOP"
    return {
        "decision": label,
        "core_signal_count": core_count,
        "core_signals": core,
        "guards": guards,
        "deltas": {
            "lesion_recall": candidate["lesion_recall"] - baseline["lesion_recall"],
            "severe_under_count": candidate["severe_under_count"] - baseline["severe_under_count"],
            "small_1_5_mean_dice": small_dice_delta,
            "small_1_5_lesion_recall": small_recall_delta,
            "lesion_precision": precision_delta,
            "fp_lesions_per_case": fp_delta,
            "mean_dice": candidate["mean_dice"] - baseline["mean_dice"],
        },
        "policy": {
            "required_core_signals": 2,
            "max_absolute_precision_drop": MAX_PRECISION_DROP,
            "max_fp_lesions_per_case_increase": MAX_FP_LESIONS_PER_CASE_INCREASE,
            "automatic_followup": False,
        },
    }


def fmt(value, digits=3) -> str:
    return "目前資料未提供" if value is None else f"{value:.{digits}f}"


def nullable_delta(candidate, baseline):
    """Return candidate - baseline only when both case-level metrics exist."""
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def main() -> None:
    out = SUITE / "screening_comparison"
    out.mkdir(parents=True, exist_ok=False)
    cases = {name: load_cases(SUITE / name / "metrics") for name in [BASELINE, *CANDIDATES]}
    ids = sorted(cases[BASELINE])
    if len(ids) != 50 or any(set(cases[name]) != set(ids) for name in CANDIDATES):
        raise RuntimeError("Screening validation case IDs differ")
    small_ids = subgroup_ids(cases[BASELINE], 1, 5)
    summaries = {name: read_summary(name) for name in [BASELINE, *CANDIDATES]}
    snapshots = {name: metric_snapshot(summary) for name, summary in summaries.items()}
    curves = {name: curve_status(name) for name in [BASELINE, *CANDIDATES]}

    comparisons = {}
    for name in CANDIDATES:
        overall_bootstrap = bootstrap(cases[BASELINE], cases[name], ids, n=10000)
        small_bootstrap = bootstrap(cases[BASELINE], cases[name], small_ids, n=10000)
        candidate_decision = decision(
            snapshots[BASELINE], snapshots[name], curves[name], is_l2a=name.startswith("L2a")
        )
        paired_rows = []
        for case in ids:
            base = cases[BASELINE][case]
            cand = cases[name][case]
            paired_rows.append(
                {
                    "case": case,
                    "gt_ml": base["gt_ml"],
                    "gt_volume_group": "1-5 mL" if 1 <= base["gt_ml"] < 5 else "other",
                    "baseline_dice": base["dice"],
                    "candidate_dice": cand["dice"],
                    "delta_dice": nullable_delta(cand["dice"], base["dice"]),
                    "baseline_pred_gt_ratio": base["pred_gt_ratio"],
                    "candidate_pred_gt_ratio": cand["pred_gt_ratio"],
                    "delta_pred_gt_ratio": nullable_delta(
                        cand["pred_gt_ratio"], base["pred_gt_ratio"]
                    ),
                    "baseline_lesion_fn": base["lesion_fn"],
                    "candidate_lesion_fn": cand["lesion_fn"],
                    "delta_lesion_fn": cand["lesion_fn"] - base["lesion_fn"],
                    "baseline_lesion_fp": base["lesion_fp"],
                    "candidate_lesion_fp": cand["lesion_fp"],
                    "delta_lesion_fp": cand["lesion_fp"] - base["lesion_fp"],
                    "baseline_lesion_recall": base["lesion_recall"],
                    "candidate_lesion_recall": cand["lesion_recall"],
                    "delta_lesion_recall": (
                        cand["lesion_recall"] - base["lesion_recall"]
                        if cand["lesion_recall"] is not None and base["lesion_recall"] is not None
                        else None
                    ),
                }
            )
        write_csv(out / f"paired_cases_{name}.csv", paired_rows)
        result = {
            "candidate": name,
            "difference": f"{name} minus {BASELINE}",
            "iterations": 10000,
            "seed": 2022,
            "resampling_unit": "case ID; lesion TP/FP/FN re-aggregated after each paired resample",
            "overall": overall_bootstrap,
            "small_1_5_ml": {"cases": len(small_ids), "bootstrap": small_bootstrap},
            "decision": candidate_decision,
        }
        save(out / f"paired_bootstrap_{name}.json", result)
        comparisons[name] = result

    sampling_path = SUITE / "sampling_audit/summary.json"
    if not sampling_path.exists():
        sampling_path = REUSED_SAMPLING_AUDIT
    sampling = json.loads(sampling_path.read_text()) if sampling_path.exists() else None
    save(
        out / "screening_summary.json",
        {
            "label": "100-epoch screening experiment; not final performance",
            "baseline": BASELINE,
            "candidates": CANDIDATES,
            "snapshots": snapshots,
            "training_curves": curves,
            "comparisons": comparisons,
            "sampling_audit": sampling,
            "limitations": (
                "Fold 0 only; single seed screening; same-seed is not fully deterministic. "
                "The 100-epoch checkpoint is an early screening state using the first 100 epochs "
                "of the 1000-epoch PolyLR trajectory, not a final model. Bootstrap CIs describe "
                "case-level uncertainty in this validation cohort."
            ),
        },
    )

    rows = []
    display_metrics = [
        ("Mean Dice", "mean_dice", 3),
        ("Median Dice", "median_dice", 3),
        ("Lesion recall", "lesion_recall", 3),
        ("Lesion precision", "lesion_precision", 3),
        ("Severe under", "severe_under_count", 0),
        ("Dice <0.5", "dice_lt_05", 0),
        ("1–5 mL Dice", "small_1_5_mean_dice", 3),
        ("1–5 mL recall", "small_1_5_lesion_recall", 3),
        ("Volume MAE (mL)", "volume_mae_ml", 2),
        ("Signed volume error (mL)", "signed_volume_error_ml", 2),
    ]
    for label, key, digits in display_metrics:
        base = snapshots[BASELINE][key]
        s1 = snapshots[CANDIDATES[0]][key]
        l2 = snapshots[CANDIDATES[1]][key]
        rows.append([label, fmt(base, digits), fmt(s1, digits), fmt(s1 - base, digits), fmt(l2, digits), fmt(l2 - base, digits)])

    lines = [
        "# ISLES2022 DWI-only Fold 0：100-epoch screening",
        "",
        "本報告為 screening experiment，不代表 final performance。三個模型皆只執行原 1000-epoch PolyLR 軌跡的前 100 epochs。",
        "",
        "| Metric | B0-100 | S1a-100 | Δ | L2a-100 | Δ |",
        "|---|---:|---:|---:|---:|---:|",
        *["| " + " | ".join(row) + " |" for row in rows],
        "",
        "## Paired bootstrap 重點",
        "",
    ]
    for name in CANDIDATES:
        comparison = comparisons[name]
        overall = comparison["overall"]
        small = comparison["small_1_5_ml"]["bootstrap"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Δ Mean Dice：{fmt(overall['mean_dice']['delta'])}，95% CI {overall['mean_dice']['ci95']}",
                f"- Δ Lesion recall：{fmt(overall['lesion_recall']['delta'])}，95% CI {overall['lesion_recall']['ci95']}",
                f"- Δ Severe-under rate：{fmt(overall['severe_under_rate']['delta'])}，95% CI {overall['severe_under_rate']['ci95']}",
                f"- 1–5 mL Δ Mean Dice：{fmt(small['mean_dice']['delta'])}，95% CI {small['mean_dice']['ci95']}",
                f"- 1–5 mL Δ Lesion recall：{fmt(small['lesion_recall']['delta'])}，95% CI {small['lesion_recall']['ci95']}",
                f"- Decision：**{comparison['decision']['decision']}**",
                f"- Core signals：{comparison['decision']['core_signals']}",
                f"- Safety/trade-off guards：{comparison['decision']['guards']}",
                "",
            ]
        )
    if sampling:
        delta = sampling["delta_050_minus_033"]
        lines.extend(
            [
                "## Sampling exposure audit",
                "",
                f"- Overall 0.50−0.33：{delta['overall']}",
                f"- 1–5 mL 0.50−0.33：{delta['1–5 mL']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision policy",
            "",
            "至少兩項核心方向一致改善，且 lesion precision 降幅不超過 0.05、FP lesions/case 增幅不超過 0.5、訓練與數值正常，才可進入 300-epoch 候選。",
            "L2a 若 recall／severe-under／small-lesion 有改善但 overall Dice 尚未追上，標記為 PROMISING BUT NOT CONVERGED。",
            "本流程不會自動啟動 300 或 1000 epochs。",
            "",
            "## Limitations",
            "",
            "Fold 0 only；single seed screening；same-seed 不等於 fully deterministic。100 epochs 為早期篩選狀態；bootstrap CI 僅反映目前 validation cohort 的 case-level uncertainty。",
        ]
    )
    with (out / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({name: comparisons[name]["decision"] for name in CANDIDATES}, indent=2), flush=True)


if __name__ == "__main__":
    main()
