"""Paired case-level trend analysis for S1a-300 against S1a-100 and B0-100."""

import csv
import json
from pathlib import Path

from recall_analysis import bootstrap, load_cases, stat
from recall_common import ROOT, save


SUITE = ROOT / "experiments/S1a_promote300_seed2022_from100"
SCREEN = ROOT / "experiments/recall_screen100_seed2022_r2"
REFERENCES = {
    "S1a_100": SCREEN / "S1a_screen100_seed2022/metrics",
    "B0_100": SCREEN / "B0_R2022_screen100/metrics",
}


def nullable_delta(candidate, baseline):
    return None if candidate is None or baseline is None else candidate - baseline


def main() -> None:
    out = SUITE / "trend_analysis"
    out.mkdir(parents=True, exist_ok=False)
    candidate = load_cases(SUITE / "metrics")
    ids = sorted(candidate)
    results = {}
    for label, folder in REFERENCES.items():
        baseline = load_cases(folder)
        if set(baseline) != set(candidate):
            raise RuntimeError(f"Validation cases differ for {label}")
        overall = bootstrap(baseline, candidate, ids, n=10000)
        small_ids = [case for case in ids if 1 <= baseline[case]["gt_ml"] < 5]
        small = bootstrap(baseline, candidate, small_ids, n=10000)
        rows = []
        for case in ids:
            base, cand = baseline[case], candidate[case]
            rows.append({
                "case": case,
                "gt_ml": base["gt_ml"],
                "baseline_dice": base["dice"],
                "candidate_dice": cand["dice"],
                "delta_dice": nullable_delta(cand["dice"], base["dice"]),
                "baseline_pred_gt_ratio": base["pred_gt_ratio"],
                "candidate_pred_gt_ratio": cand["pred_gt_ratio"],
                "delta_pred_gt_ratio": nullable_delta(cand["pred_gt_ratio"], base["pred_gt_ratio"]),
                "baseline_lesion_fn": base["lesion_fn"],
                "candidate_lesion_fn": cand["lesion_fn"],
                "baseline_lesion_fp": base["lesion_fp"],
                "candidate_lesion_fp": cand["lesion_fp"],
            })
        with (out / f"paired_cases_vs_{label}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        results[label] = {
            "iterations": 10000,
            "seed": 2022,
            "resampling_unit": "case ID; identical sampled cases used for both models",
            "overall": overall,
            "small_1_5_ml": {"cases": len(small_ids), "bootstrap": small},
        }
    s100 = results["S1a_100"]["overall"]
    improvement = {
        "mean_dice_increased": s100["mean_dice"]["delta"] > 0,
        "lesion_recall_increased": s100["lesion_recall"]["delta"] > 0,
        "severe_under_rate_reduced": s100["severe_under_rate"]["delta"] < 0,
        "volume_mae_reduced": s100["volume_mae_ml"]["delta"] < 0,
        "precision_drop_within_0.05": s100["lesion_precision"]["delta"] >= -0.05,
        "fp_lesions_per_case_increase_within_0.5": s100["fp_lesions_per_case"]["delta"] <= 0.5,
    }
    core = sum(improvement[key] for key in [
        "mean_dice_increased", "lesion_recall_increased", "severe_under_rate_reduced", "volume_mae_reduced"
    ])
    guards = improvement["precision_drop_within_0.05"] and improvement["fp_lesions_per_case_increase_within_0.5"]
    trend = "CONTINUED IMPROVEMENT" if core >= 2 and guards else "MIXED OR NO CONTINUED IMPROVEMENT"
    summary = {
        "label": "S1a checkpoint continuation from epoch 100 to total epoch 300",
        "primary_comparison": "S1a_300 minus S1a_100",
        "trend": trend,
        "signals": improvement,
        "comparisons": results,
        "limitations": (
            "Fold 0 only; single seed; controlled checkpoint continuation is not bitwise deterministic. "
            "The B0-100 comparison is contextual because there is no same-horizon B0-300 control."
        ),
        "automatic_followup_started": False,
    }
    save(out / "trend_summary.json", summary)
    lines = [
        "# S1a 100→300 epoch trend analysis",
        "",
        f"Decision: **{trend}**",
        "",
        "Primary comparison is S1a-300 minus S1a-100 using paired case bootstrap, 10,000 iterations, seed 2022.",
        "",
        "| Metric | S1a-100 | S1a-300 | Delta | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["mean_dice", "median_dice", "lesion_recall", "lesion_precision", "volume_mae_ml", "signed_volume_error_ml", "severe_under_rate", "dice_below_05_rate", "fp_lesions_per_case"]:
        row = s100[key]
        lines.append(f"| {key} | {row['baseline']:.3f} | {row['candidate']:.3f} | {row['delta']:.3f} | {row['ci95']} |")
    lines.extend(["", "## Signals", "", f"`{json.dumps(improvement, ensure_ascii=False)}`", "", summary["limitations"]])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"trend": trend, "signals": improvement}, indent=2), flush=True)


if __name__ == "__main__":
    main()

