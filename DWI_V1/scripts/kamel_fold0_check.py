"""Gate later folds on Fold 0 replication agreement with the existing B0-100."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100/metrics"
NEW = ROOT / "experiments/kamel_replication_seed2022/fold_0/metrics"
OUT = ROOT / "outputs/analysis/kamel_replication"
STATUS = OUT / "status.json"


def main() -> None:
    old_summary = json.loads((OLD / "summary.json").read_text(encoding="utf-8"))
    new_summary = json.loads((NEW / "summary.json").read_text(encoding="utf-8"))
    old_cases = json.loads((OLD / "cases.json").read_text(encoding="utf-8"))
    new_cases = json.loads((NEW / "cases.json").read_text(encoding="utf-8"))
    if [row["case"] for row in old_cases] != [row["case"] for row in new_cases]:
        raise RuntimeError("Fold 0 case IDs/order do not match existing B0-100")
    metric_values = {
        "Mean Dice": (old_summary["dice"]["mean"], new_summary["dice"]["mean"]),
        "Median Dice": (old_summary["dice"]["median"], new_summary["dice"]["median"]),
        "Lesion Recall": (old_summary["lesion_recall"], new_summary["lesion_recall"]),
        "Lesion Precision": (old_summary["lesion_precision"], new_summary["lesion_precision"]),
        "FP/case": (old_summary["fp_lesions_per_case"], new_summary["fp_lesions_per_case"]),
        "Volume MAE": (
            old_summary["distributions"]["absolute_volume_error_ml"]["mean"],
            new_summary["distributions"]["absolute_volume_error_ml"]["mean"],
        ),
        "Severe under count": (old_summary["severe_under_count"], new_summary["severe_under_count"]),
    }
    comparison = [
        {"Metric": metric, "Existing_B0_100": old, "New_Fold0_100": new, "Delta_new_minus_existing": new - old}
        for metric, (old, new) in metric_values.items()
    ]
    with (OUT / "fold0_replication_comparison.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    paired_dice = [
        abs(new["dice"] - old["dice"])
        for old, new in zip(old_cases, new_cases)
        if old["dice"] is not None and new["dice"] is not None
    ]
    tolerances = {
        "absolute_mean_dice_delta_le_0p015": abs(metric_values["Mean Dice"][1] - metric_values["Mean Dice"][0]) <= 0.015,
        "absolute_median_dice_delta_le_0p02": abs(metric_values["Median Dice"][1] - metric_values["Median Dice"][0]) <= 0.02,
        "absolute_lesion_recall_delta_le_0p03": abs(metric_values["Lesion Recall"][1] - metric_values["Lesion Recall"][0]) <= 0.03,
        "absolute_lesion_precision_delta_le_0p03": abs(metric_values["Lesion Precision"][1] - metric_values["Lesion Precision"][0]) <= 0.03,
        "absolute_fp_per_case_delta_le_0p30": abs(metric_values["FP/case"][1] - metric_values["FP/case"][0]) <= 0.30,
        "absolute_severe_under_count_delta_le_2": abs(metric_values["Severe under count"][1] - metric_values["Severe under count"][0]) <= 2,
        "paired_mean_absolute_dice_delta_le_0p05": float(np.mean(paired_dice)) <= 0.05,
    }
    passed = all(tolerances.values())
    result = {
        "passed": passed,
        "interpretation": "replicated within predeclared stochastic-equivalence tolerances" if passed else "material Fold 0 mismatch; stop before Fold 1",
        "why_not_bitwise": "The B0 data augmenter records fixed worker seeds but uses nondeterministic worker arrival order, so bitwise equality is not promised.",
        "tolerances": tolerances,
        "paired_mean_absolute_dice_delta": float(np.mean(paired_dice)),
        "paired_median_absolute_dice_delta": float(np.median(paired_dice)),
        "max_absolute_dice_delta": float(np.max(paired_dice)),
    }
    (OUT / "fold0_replication_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    status["fold_0"]["replication_check"] = "passed" if passed else "failed"
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps({"comparison": comparison, **result}, indent=2), flush=True)
    if not passed:
        raise SystemExit("Fold 0 replication gate failed; later folds must not start")


if __name__ == "__main__":
    main()
