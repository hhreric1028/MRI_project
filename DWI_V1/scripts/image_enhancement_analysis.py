"""Paired evaluation, promotion, failure tracking, and final reporting for enhancements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from image_enhancement_common import (
    BASE_METRICS,
    BASE_PREDICTIONS,
    BASE_RAW,
    BOOTSTRAP_ITERATIONS,
    EXPERIMENTS,
    KNOWN_FAILURES,
    ROOT,
    SEED,
    SUITE,
    experiment_dir,
)
from recall_analysis import bootstrap, load_cases
from recall_common import save


MAX_PRECISION_DROP = 0.05
MAX_FP_PER_CASE_INCREASE = 0.5
MAX_VOLUME_MAE_INCREASE_ML = 0.5
MAX_ABS_SIGNED_BIAS_WORSENING_ML = 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot(summary: dict) -> dict:
    small = summary["subgroups"]["1-5"]
    return {
        "mean_dice": summary["dice"]["mean"],
        "median_dice": summary["dice"]["median"],
        "sd_dice": summary["dice"]["sd"],
        "hd95_mean_mm": summary["distributions"]["hd95_mm"]["mean"],
        "volume_mae_ml": summary["distributions"]["absolute_volume_error_ml"]["mean"],
        "signed_volume_error_ml": summary["distributions"]["signed_volume_error_ml"]["mean"],
        "lesion_recall": summary["lesion_recall"],
        "lesion_precision": summary["lesion_precision"],
        "lesion_f1": summary["lesion_f1"],
        "lesion_tp": summary["lesion_tp"],
        "lesion_fn": summary["lesion_fn"],
        "lesion_fp": summary["lesion_fp"],
        "fp_lesions_per_case": summary["fp_lesions_per_case"],
        "severe_under_count": summary["severe_under_count"],
        "severe_under_rate": summary["severe_under_count"] / summary["cases"],
        "catastrophic_miss_count": summary["catastrophic_miss_count"],
        "dice_lt_05": summary["dice_lt_05"],
        "dice_lt_03": summary["dice_lt_03"],
        "small_cases": small["cases"],
        "small_mean_dice": small["dice"].get("mean"),
        "small_median_dice": small["dice"].get("median"),
        "small_recall": small["lesion_recall"],
        "small_precision": small["lesion_precision"],
        "small_fn": small["lesion_fn"],
        "small_fp": small["lesion_fp"],
        "small_volume_mae_ml": small["distributions"]["absolute_volume_error_ml"].get("mean"),
    }


def curve_status(experiment: str, stage: str) -> dict:
    path = experiment_dir(experiment) / stage / "train/epoch_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected_rows = 100 if stage == "screen100" else 200
    expected_first = 1 if stage == "screen100" else 101
    expected_last = 100 if stage == "screen100" else 300
    finite_keys = ["train_loss", "validation_loss", "patch_pseudo_dice", "learning_rate"]
    finite = len(rows) == expected_rows and all(np.isfinite(r[k]) for r in rows for k in finite_keys)
    epochs = [r["epoch"] for r in rows]
    lr = np.asarray([r["learning_rate"] for r in rows])
    completion = json.loads((experiment_dir(experiment) / stage / "train/completion.json").read_text())
    expected_lr = 0.01 * (1 - (expected_last - 1) / 1000) ** 0.9
    normal = bool(
        finite and epochs[0] == expected_first and epochs[-1] == expected_last
        and np.all(np.diff(lr) <= 1e-14)
        and np.isclose(lr[-1], expected_lr, rtol=1e-12, atol=1e-14)
        and completion["unrecovered_consecutive_overflows"] == 0
    )
    return {
        "rows": len(rows), "first_epoch": epochs[0], "last_epoch": epochs[-1],
        "finite": bool(finite), "final_lr": float(lr[-1]), "expected_final_lr": expected_lr,
        "amp_overflow_count": completion["total_amp_skipped_steps"],
        "amp_overflow_events_file": str(experiment_dir(experiment) / stage / "train/amp_events.jsonl"),
        "unrecovered_consecutive_overflows": completion["unrecovered_consecutive_overflows"],
        "normal": normal,
    }


def decision(base: dict, candidate: dict, curve: dict) -> dict:
    eps = 1e-12
    core = {
        "mean_dice_improved": candidate["mean_dice"] > base["mean_dice"] + eps,
        "lesion_recall_improved": candidate["lesion_recall"] > base["lesion_recall"] + eps,
        "severe_under_reduced": candidate["severe_under_count"] < base["severe_under_count"],
        "small_1_5_ml_dice_improved": candidate["small_mean_dice"] > base["small_mean_dice"] + eps,
        "small_1_5_ml_recall_improved": candidate["small_recall"] > base["small_recall"] + eps,
    }
    guards = {
        "precision_drop_le_0.05": candidate["lesion_precision"] - base["lesion_precision"] >= -MAX_PRECISION_DROP,
        "fp_per_case_increase_le_0.5": candidate["fp_lesions_per_case"] - base["fp_lesions_per_case"] <= MAX_FP_PER_CASE_INCREASE,
        "volume_mae_increase_le_0.5_ml": candidate["volume_mae_ml"] - base["volume_mae_ml"] <= MAX_VOLUME_MAE_INCREASE_ML,
        "absolute_signed_bias_worsening_le_1_ml": abs(candidate["signed_volume_error_ml"]) - abs(base["signed_volume_error_ml"]) <= MAX_ABS_SIGNED_BIAS_WORSENING_ML,
        "training_numerically_stable": curve["normal"],
    }
    count = sum(core.values())
    recall_up = core["lesion_recall_improved"]
    specificity_failure = not guards["precision_drop_le_0.05"] or not guards["fp_per_case_increase_le_0.5"]
    if count >= 2 and all(guards.values()):
        label = "PROMOTE"
    elif recall_up and specificity_failure:
        label = "RECALL-PRECISION TRADE-OFF"
    else:
        label = "STOP"
    return {"decision": label, "core_signal_count": count, "core": core, "guards": guards}


def required_bootstrap(base: dict, candidate: dict, ids: list[str]) -> dict:
    raw = bootstrap(base, candidate, ids, n=BOOTSTRAP_ITERATIONS)
    keys = {
        "delta_mean_dice": "mean_dice",
        "delta_lesion_recall": "lesion_recall",
        "delta_precision": "lesion_precision",
        "delta_severe_under_rate": "severe_under_rate",
        "delta_fp_lesions_per_case": "fp_lesions_per_case",
        "delta_volume_mae_ml": "volume_mae_ml",
    }
    return {out: raw[source] for out, source in keys.items()}


def paired_case_rows(base: dict, candidate: dict) -> list[dict]:
    rows = []
    for case in sorted(base):
        b, c = base[case], candidate[case]
        delta_dice = None if b["dice"] is None or c["dice"] is None else c["dice"] - b["dice"]
        rows.append({
            "case": case,
            "gt_ml": b["gt_ml"],
            "baseline_dice": b["dice"], "candidate_dice": c["dice"], "delta_dice": delta_dice,
            "baseline_pred_gt": b["pred_gt_ratio"], "candidate_pred_gt": c["pred_gt_ratio"],
            "baseline_recall": b["lesion_recall"], "candidate_recall": c["lesion_recall"],
            "baseline_precision": b["lesion_precision"], "candidate_precision": c["lesion_precision"],
            "baseline_fn": b["lesion_fn"], "candidate_fn": c["lesion_fn"], "fn_improvement": b["lesion_fn"] - c["lesion_fn"],
            "baseline_fp": b["lesion_fp"], "candidate_fp": c["lesion_fp"], "fp_increase": c["lesion_fp"] - b["lesion_fp"],
            "baseline_volume_error_ml": b["signed_volume_error_ml"], "candidate_volume_error_ml": c["signed_volume_error_ml"],
        })
    return rows


def failure_selection(rows: list[dict]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    dice_rows = [r for r in rows if r["delta_dice"] is not None]
    ranked_dice = sorted(dice_rows, key=lambda r: r["delta_dice"])
    ranked_fn = sorted(dice_rows, key=lambda r: (r["fn_improvement"], r["delta_dice"]), reverse=True)
    ranked_fp = sorted(dice_rows, key=lambda r: (r["fp_increase"], -r["delta_dice"]), reverse=True)
    groups = [
        ("known_failure", KNOWN_FAILURES),
        ("top5_dice_decline", [r["case"] for r in ranked_dice[:5]]),
        ("top5_dice_improvement", [r["case"] for r in ranked_dice[-5:]]),
        ("largest_fn_improvement", [r["case"] for r in ranked_fn[:5] if r["fn_improvement"] > 0]),
        ("largest_fp_increase", [r["case"] for r in ranked_fp[:5] if r["fp_increase"] > 0]),
    ]
    for label, cases in groups:
        for case in cases:
            reasons.setdefault(case, []).append(label)
    return reasons


def case_analysis(experiment: str, stage: str, rows: list[dict], candidate_predictions: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = experiment_dir(experiment) / "case_analysis" / stage
    out.mkdir(parents=True, exist_ok=False)
    reasons = failure_selection(rows)
    selected_rows = []
    row_map = {r["case"]: r for r in rows}
    for case, why in sorted(reasons.items()):
        selected_rows.append({**row_map[case], "selection_reasons": ";".join(why)})
    write_csv(out / "tracked_cases.csv", selected_rows)
    save(out / "selection.json", [{"case": case, "reasons": why} for case, why in sorted(reasons.items())])
    figure_dir = out / "figures"
    figure_dir.mkdir()
    spec = EXPERIMENTS[experiment]
    enhanced_raw = ROOT / "datasets/ISLES2022/nnUNet_raw" / spec["dataset_name"]
    for case, why in sorted(reasons.items()):
        paths = [
            BASE_RAW / "imagesTr" / f"{case}_0000.nii.gz",
            enhanced_raw / "imagesTr" / f"{case}_0001.nii.gz",
            BASE_RAW / "labelsTr" / f"{case}.nii.gz",
            BASE_PREDICTIONS / f"{case}.nii.gz",
            candidate_predictions / f"{case}.nii.gz",
        ]
        images = [nib.load(p) for p in paths]
        if not all(i.shape == images[0].shape and np.allclose(i.affine, images[0].affine, atol=1e-3) for i in images):
            raise RuntimeError(f"Panel geometry mismatch: {case}")
        arrays = [nib.as_closest_canonical(i).get_fdata(dtype=np.float32) for i in images]
        original, enhanced = arrays[:2]
        gt, b0, candidate = [x > 0.5 for x in arrays[2:]]
        slices = [int(np.argmax(gt.sum((0, 1)))), int(np.argmax((b0 != candidate).sum((0, 1))))]
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        for row_index, z in enumerate(slices):
            panels = [
                ("Original", original, None), ("Enhanced", enhanced, None),
                ("GT", original, gt), ("B0-100", original, b0), (f"{spec['id']}-{stage[-3:]}", original, candidate),
            ]
            for column, (title, image, mask) in enumerate(panels):
                ax = axes[row_index, column]
                nonzero = image[image != 0]
                lo, hi = np.percentile(nonzero, [1, 99]) if nonzero.size else (image.min(), image.max())
                ax.imshow(image[:, :, z].T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
                if mask is not None:
                    overlay = np.ma.masked_where(~mask[:, :, z].T, mask[:, :, z].T)
                    ax.imshow(overlay, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.55)
                    if gt[:, :, z].any():
                        ax.contour(gt[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.6)
                ax.set_title(f"{title} | z={z}")
                ax.axis("off")
            axes[row_index, 0].set_ylabel("GT max" if row_index == 0 else "B0/candidate max disagreement")
        fig.suptitle(f"{case} | {', '.join(why)}")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{case}.png", dpi=120)
        plt.close(fig)


def analyse(experiment: str, stage: str) -> dict:
    exp = experiment_dir(experiment)
    metric_dir = exp / "metrics" / stage
    base = load_cases(BASE_METRICS)
    candidate = load_cases(metric_dir)
    ids = sorted(base)
    small = [case for case in ids if 1 <= base[case]["gt_ml"] < 5]
    base_summary = snapshot(json.loads((BASE_METRICS / "summary.json").read_text()))
    candidate_summary = snapshot(json.loads((metric_dir / "summary.json").read_text()))
    curve = curve_status(experiment, stage)
    stage_decision = decision(base_summary, candidate_summary, curve)
    bootstrap_dir = exp / "bootstrap" / stage
    bootstrap_dir.mkdir(parents=True, exist_ok=False)
    boot = {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": SEED,
        "resampling_unit": "paired case; lesion TP/FP/FN re-aggregated within each resample",
        "candidate_minus_b0_100": required_bootstrap(base, candidate, ids),
        "small_1_5_ml_candidate_minus_b0_100": {
            "delta_mean_dice": bootstrap(base, candidate, small, n=BOOTSTRAP_ITERATIONS)["mean_dice"],
            "delta_lesion_recall": bootstrap(base, candidate, small, n=BOOTSTRAP_ITERATIONS)["lesion_recall"],
        },
    }
    if stage == "promote300":
        prior = load_cases(exp / "metrics/screen100")
        boot["candidate_300_minus_candidate_100"] = required_bootstrap(prior, candidate, ids)
        boot["small_1_5_ml_300_minus_100"] = {
            "delta_mean_dice": bootstrap(prior, candidate, small, n=BOOTSTRAP_ITERATIONS)["mean_dice"],
            "delta_lesion_recall": bootstrap(prior, candidate, small, n=BOOTSTRAP_ITERATIONS)["lesion_recall"],
        }
        boot["matched_b0_300_available"] = False
        boot["limitation"] = "B0-R2022 epoch 300 is unavailable; B0-100 comparison is contextual and not horizon matched."
    save(bootstrap_dir / "paired_bootstrap.json", boot)
    rows = paired_case_rows(base, candidate)
    write_csv(exp / "case_analysis" / f"paired_cases_{stage}.csv", rows)
    completion = json.loads((exp / stage / "train/completion.json").read_text())
    case_analysis(experiment, stage, rows, Path(completion["predictions"]))
    result = {
        "experiment": experiment,
        "stage": stage,
        "comparison": f"{EXPERIMENTS[experiment]['id']}-{100 if stage == 'screen100' else 300} minus B0-R2022-100",
        "baseline": base_summary,
        "candidate": candidate_summary,
        "curve": curve,
        "decision": stage_decision,
        "bootstrap": str(bootstrap_dir / "paired_bootstrap.json"),
    }
    save(exp / stage / "analysis.json", result)
    save(exp / stage / "decision.json", stage_decision)
    lines = [
        f"# {EXPERIMENTS[experiment]['id']}-{100 if stage == 'screen100' else 300} result",
        "",
        "Comparison is candidate minus B0-R2022-100." + (" B0-300 is unavailable, so this is contextual rather than horizon matched." if stage == "promote300" else ""),
        "",
        "| Metric | B0-100 | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ["mean_dice", "median_dice", "lesion_recall", "lesion_precision", "fp_lesions_per_case", "severe_under_count", "volume_mae_ml", "signed_volume_error_ml", "small_mean_dice", "small_recall"]:
        b, c = base_summary[key], candidate_summary[key]
        lines.append(f"| {key} | {b:.3f} | {c:.3f} | {c-b:+.3f} |")
    lines += ["", f"Decision: **{stage_decision['decision']}**", "", f"Core directions improved: {stage_decision['core_signal_count']}/5. Safety guards: {stage_decision['guards']}"]
    (exp / stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def promotion_rank(screen_results: dict[str, dict]) -> list[str]:
    promoted = [name for name, result in screen_results.items() if result["decision"]["decision"] == "PROMOTE"]
    def score(name: str):
        result = screen_results[name]
        b, c = result["baseline"], result["candidate"]
        return (
            b["severe_under_count"] - c["severe_under_count"],
            c["lesion_recall"] - b["lesion_recall"],
            c["small_mean_dice"] - b["small_mean_dice"],
            c["small_recall"] - b["small_recall"],
            c["mean_dice"] - b["mean_dice"],
            -(c["fp_lesions_per_case"] - b["fp_lesions_per_case"]),
            c["lesion_precision"] - b["lesion_precision"],
        )
    return sorted(promoted, key=score, reverse=True)[:2]


def final_status(experiment: str, promoted: bool) -> str:
    exp = experiment_dir(experiment)
    screen = json.loads((exp / "screen100/analysis.json").read_text())
    if not promoted:
        return screen["decision"]["decision"] if screen["decision"]["decision"] != "PROMOTE" else "EARLY SIGNAL ONLY"
    result300 = json.loads((exp / "promote300/analysis.json").read_text())
    s100 = screen["candidate"]
    s300 = result300["candidate"]
    trend_stable = (
        s300["mean_dice"] >= s100["mean_dice"] - 0.005
        and s300["lesion_recall"] >= s100["lesion_recall"] - 0.01
        and s300["severe_under_count"] <= s100["severe_under_count"]
        and s300["small_mean_dice"] >= s100["small_mean_dice"] - 0.01
    )
    contextual = result300["decision"]["decision"]
    if contextual == "PROMOTE" and trend_stable:
        return "CANDIDATE FOR FULL TRAINING"
    if contextual == "RECALL-PRECISION TRADE-OFF":
        return "RECALL-PRECISION TRADE-OFF"
    return "EARLY SIGNAL ONLY"


def external_interventions() -> dict:
    paths = {
        "S1a-100": ROOT / "experiments/recall_screen100_seed2022_r2/S1a_screen100_seed2022/metrics/summary.json",
        "L2a-100": ROOT / "experiments/recall_screen100_seed2022_r2/L2a_screen100_seed2022/metrics/summary.json",
        "P3a-100": ROOT / "experiments/P3a_clip_zscore_seed2022/screen100/metrics/summary.json",
        "P3a-300": ROOT / "experiments/P3a_clip_zscore_seed2022/promote300/metrics/summary.json",
    }
    return {name: snapshot(json.loads(path.read_text())) for name, path in paths.items() if path.is_file()}


def write_final_report(promoted: list[str]) -> dict:
    baseline = snapshot(json.loads((BASE_METRICS / "summary.json").read_text()))
    qc = json.loads((SUITE / "qc/parameter_selection.json").read_text())
    final = {}
    table = []
    small_table = []
    for name, spec in EXPERIMENTS.items():
        screen = json.loads((experiment_dir(name) / "screen100/analysis.json").read_text())
        s100 = screen["candidate"]
        s300 = None
        if name in promoted:
            s300 = json.loads((experiment_dir(name) / "promote300/analysis.json").read_text())["candidate"]
        status = final_status(name, name in promoted)
        final[name] = status
        table.append({"model": spec["id"], "enhancement": spec["method"], "s100": s100, "s300": s300, "decision": status})
        small_table.append({"model": spec["id"], "s100": s100, "s300": s300, "decision": status})
    interventions = external_interventions()
    def final_rank(row: dict):
        metrics = row["s300"] or row["s100"]
        return (
            baseline["severe_under_count"] - metrics["severe_under_count"],
            metrics["lesion_recall"] - baseline["lesion_recall"],
            metrics["small_mean_dice"] - baseline["small_mean_dice"],
            metrics["small_recall"] - baseline["small_recall"],
            metrics["mean_dice"] - baseline["mean_dice"],
            -(metrics["fp_lesions_per_case"] - baseline["fp_lesions_per_case"]),
            metrics["lesion_precision"] - baseline["lesion_precision"],
        )
    best = max(table, key=final_rank)
    best_metrics = best["s300"] or best["s100"]
    balanced = (
        best_metrics["lesion_recall"] > baseline["lesion_recall"]
        and best_metrics["lesion_precision"] >= baseline["lesion_precision"] - MAX_PRECISION_DROP
        and best_metrics["fp_lesions_per_case"] <= baseline["fp_lesions_per_case"] + MAX_FP_PER_CASE_INCREASE
    )
    final_stage = "promote300" if best["s300"] is not None else "screen100"
    tracked_path = experiment_dir(next(name for name, spec in EXPERIMENTS.items() if spec["id"] == best["model"])) / "case_analysis" / final_stage / "tracked_cases.csv"
    with tracked_path.open(encoding="utf-8-sig", newline="") as handle:
        tracked = {row["case"]: row for row in csv.DictReader(handle)}
    improved_failures = [case for case in KNOWN_FAILURES if float(tracked[case]["delta_dice"]) > 0 or float(tracked[case]["fn_improvement"]) > 0]
    unresolved_failures = [case for case in KNOWN_FAILURES if case not in improved_failures]
    comparison_lines = []
    for intervention in ["S1a-100", "L2a-100"]:
        if intervention not in interventions:
            continue
        m = interventions[intervention]
        enhancement_is_more_balanced = (
            balanced
            and best_metrics["lesion_recall"] - baseline["lesion_recall"] >= m["lesion_recall"] - baseline["lesion_recall"]
            and best_metrics["lesion_precision"] - baseline["lesion_precision"] >= m["lesion_precision"] - baseline["lesion_precision"]
            and best_metrics["fp_lesions_per_case"] - baseline["fp_lesions_per_case"] <= m["fp_lesions_per_case"] - baseline["fp_lesions_per_case"]
        )
        comparison_lines.append(f"{intervention}: {'yes' if enhancement_is_more_balanced else 'no'} under the joint recall/precision/FP definition")
    lines = [
        "# DWI image enhancement controlled ablation — final report",
        "",
        "All runs stopped at or before 300 epochs. No 1000-epoch training, enhancement combination, sampling/loss/spacing/modality/architecture change was started.",
        "",
        "## Final comparison",
        "",
        "| Model | Enhancement | 100 Dice | 100 Recall | 100 Precision | 100 FP/case | 100 Severe Under | 300 Dice | 300 Recall | 300 Precision | 300 FP/case | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for row in table:
        a, b = row["s100"], row["s300"]
        lines.append(
            f"| {row['model']} | {row['enhancement']} | {fmt(a['mean_dice'])} | {fmt(a['lesion_recall'])} | {fmt(a['lesion_precision'])} | {fmt(a['fp_lesions_per_case'])} | {a['severe_under_count']} | "
            f"{fmt(None if b is None else b['mean_dice'])} | {fmt(None if b is None else b['lesion_recall'])} | {fmt(None if b is None else b['lesion_precision'])} | {fmt(None if b is None else b['fp_lesions_per_case'])} | {row['decision']} |"
        )
    lines += [
        "", "## 1–5 mL subgroup", "",
        "| Model | 100 Mean Dice | 100 Median Dice | 100 Recall | 100 Precision | 100 FN | 100 FP | 100 Volume MAE | 300 Mean Dice | 300 Recall | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in small_table:
        a, b = row["s100"], row["s300"]
        lines.append(
            f"| {row['model']} | {fmt(a['small_mean_dice'])} | {fmt(a['small_median_dice'])} | {fmt(a['small_recall'])} | {fmt(a['small_precision'])} | {a['small_fn']} | {a['small_fp']} | {fmt(a['small_volume_mae_ml'])} | "
            f"{fmt(None if b is None else b['small_mean_dice'])} | {fmt(None if b is None else b['small_recall'])} | {row['decision']} |"
        )
    lines += [
        "", "## Research questions", "",
        f"1. **Does enhancement improve lesion detectability?** Best observed candidate: {best['model']}; lesion recall delta versus B0-100 = {best_metrics['lesion_recall']-baseline['lesion_recall']:+.3f}. Interpret with its paired bootstrap interval.",
        f"2. **Recurring severe under-segmentation?** Best-candidate count changed from {baseline['severe_under_count']} to {best_metrics['severe_under_count']}; fixed-case details are in each case-analysis table and panel set.",
        f"3. **1–5 mL lesions?** Mean Dice delta = {best_metrics['small_mean_dice']-baseline['small_mean_dice']:+.3f}; recall delta = {best_metrics['small_recall']-baseline['small_recall']:+.3f}.",
        f"4. **Only recall through more FP?** {'No for the best candidate under the prespecified precision/FP guards.' if balanced else 'The best candidate did not demonstrate a guard-passing balanced sensitivity gain.'}",
        f"5. **More balanced than sampling/loss interventions?** For the selected best representation, {'; '.join(comparison_lines)}. This definition requires at least as much recall gain, no worse precision change, and no greater FP/case change versus B0-100.",
        f"6. **Improved failure phenotypes?** Fixed failures with Dice or lesion-FN improvement: {improved_failures or ['none']}. The largest Dice/FN improvement cohorts and images are in `{best['model']}/case_analysis/{final_stage}`.",
        f"7. **Unresolved cases?** Fixed failures without Dice or FN improvement: {unresolved_failures or ['none']}. Top-five Dice declines are retained in the same tracked-case table.",
        f"8. **Enough evidence for 1000 epochs?** Final labels: {final}. This workflow does not start 1000 epochs under any label.",
        "", "## QC parameter freeze", "",
        f"Gamma={qc['gamma']['selected']}; CLAHE clip={qc['clahe']['selected_clip_limit']}, kernel={qc['clahe']['kernel_size_voxels']}, axial slice-wise; intensity scale a={qc['intensity_scale']['scale']}, b={qc['intensity_scale']['shift']}.",
        "", "## Context: prior interventions", "",
    ]
    for name, metrics in interventions.items():
        lines.append(f"- {name}: Dice {metrics['mean_dice']:.3f}, recall {metrics['lesion_recall']:.3f}, precision {metrics['lesion_precision']:.3f}, FP/case {metrics['fp_lesions_per_case']:.3f}, severe-under {metrics['severe_under_count']}.")
    lines += [
        "", "## Limitations", "",
        "Fold 0 and one seed only. Same-seed GPU training is not bitwise deterministic. Paired case bootstrap quantifies this validation cohort, not cross-fold, center or seed generalization. B0-R2022-300 is unavailable, so 300-epoch comparisons to B0-100 are contextual rather than horizon matched.",
    ]
    (SUITE / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "status": "image_enhancement_batch_complete",
        "promoted_to_300": promoted,
        "final_decisions": final,
        "automatic_1000_started": False,
        "matched_b0_300_available": False,
    }
    save(SUITE / "final_decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--stage", choices=["screen100", "promote300"])
    parser.add_argument("--select-promotions", action="store_true")
    parser.add_argument("--final-report", action="store_true")
    args = parser.parse_args()
    if args.experiment and args.stage:
        print(json.dumps(analyse(args.experiment, args.stage)["decision"], indent=2), flush=True)
    elif args.select_promotions:
        screens = {name: json.loads((experiment_dir(name) / "screen100/analysis.json").read_text()) for name in EXPERIMENTS}
        selected = promotion_rank(screens)
        save(SUITE / "promotion_selection.json", {"selected_maximum_two": selected, "ranking_priorities": ["severe under reduction", "recall", "1-5 mL", "mean Dice", "FP penalty", "precision"]})
        print(json.dumps(selected), flush=True)
    elif args.final_report:
        selected = json.loads((SUITE / "promotion_selection.json").read_text())["selected_maximum_two"]
        print(json.dumps(write_final_report(selected), indent=2), flush=True)
    else:
        parser.error("Choose an analysis action")


if __name__ == "__main__":
    main()
