"""P3a paired evaluation, promotion decision, and fixed-slice case review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from recall_analysis import bootstrap, load_cases
from recall_common import ROOT, save


SUITE = ROOT / "experiments/P3a_clip_zscore_seed2022"
B0 = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100"
B0_METRICS = B0 / "metrics"
B0_PRED = Path(json.loads((B0 / "train/completion.json").read_text())["predictions"])
RAW_B0 = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI"
RAW_P3 = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset013_ISLES2022_DWI_ClipZScore"
KNOWN_FAILURES = ["case_0068", "case_0182", "case_0181", "case_0140", "case_0212"]
MAX_PRECISION_DROP = 0.05
MAX_FP_PER_CASE_INCREASE = 0.5
MAX_VOLUME_MAE_INCREASE_ML = 0.5


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nullable_delta(candidate, baseline):
    return None if candidate is None or baseline is None else candidate - baseline


def snapshot(summary: dict) -> dict:
    small = summary["subgroups"]["1-5"]
    return {
        "mean_dice": summary["dice"]["mean"],
        "median_dice": summary["dice"]["median"],
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
        "small_cases": small["cases"],
        "small_mean_dice": small["dice"].get("mean"),
        "small_median_dice": small["dice"].get("median"),
        "small_recall": small["lesion_recall"],
        "small_precision": small["lesion_precision"],
        "small_volume_mae_ml": small["distributions"]["absolute_volume_error_ml"].get("mean"),
        "small_severe_under_count": small["severe_under_count"],
    }


def curve_status(stage: str) -> dict:
    rows = [json.loads(x) for x in (SUITE / stage / "train/epoch_metrics.jsonl").read_text().splitlines() if x.strip()]
    expected_rows, expected_last = (100, 100) if stage == "screen100" else (200, 300)
    keys = ["train_loss", "validation_loss", "patch_pseudo_dice", "learning_rate"]
    finite = len(rows) == expected_rows and all(np.isfinite(r[k]) for r in rows for k in keys)
    epochs = [r["epoch"] for r in rows]
    lr = np.asarray([r["learning_rate"] for r in rows])
    expected_lr = 0.01 * (1 - (expected_last - 1) / 1000) ** 0.9
    completion = json.loads((SUITE / stage / "train/completion.json").read_text())
    normal = bool(
        finite and epochs[-1] == expected_last and np.all(np.diff(lr) <= 1e-14)
        and np.isclose(lr[-1], expected_lr, rtol=1e-12, atol=1e-14)
        and completion["unrecovered_consecutive_overflows"] == 0
    )
    return {
        "rows": len(rows), "first_epoch": epochs[0], "last_epoch": epochs[-1],
        "finite": bool(finite), "final_lr": float(lr[-1]), "expected_final_lr": expected_lr,
        "total_amp_skipped_steps": completion["total_amp_skipped_steps"],
        "unrecovered_consecutive_overflows": completion["unrecovered_consecutive_overflows"],
        "normal": normal,
    }


def promotion_decision(base: dict, cand: dict, curve: dict) -> dict:
    eps = 1e-12
    small_balanced = (
        cand["small_mean_dice"] >= base["small_mean_dice"] - eps
        and cand["small_recall"] >= base["small_recall"] - eps
        and (cand["small_mean_dice"] > base["small_mean_dice"] + eps or cand["small_recall"] > base["small_recall"] + eps)
    )
    core = {
        "mean_dice_improved": cand["mean_dice"] > base["mean_dice"] + eps,
        "lesion_recall_improved": cand["lesion_recall"] > base["lesion_recall"] + eps,
        "severe_under_count_reduced": cand["severe_under_count"] < base["severe_under_count"],
        "small_dice_recall_balanced_improvement": small_balanced,
    }
    guards = {
        "precision_drop_within_0.05": cand["lesion_precision"] - base["lesion_precision"] >= -MAX_PRECISION_DROP,
        "fp_per_case_increase_within_0.5": cand["fp_lesions_per_case"] - base["fp_lesions_per_case"] <= MAX_FP_PER_CASE_INCREASE,
        "volume_mae_increase_within_0.5_ml": cand["volume_mae_ml"] - base["volume_mae_ml"] <= MAX_VOLUME_MAE_INCREASE_ML,
        "training_and_numerics_normal": curve["normal"],
    }
    core_count = sum(core.values())
    recall_up = core["lesion_recall_improved"]
    specificity_tradeoff = not guards["precision_drop_within_0.05"] or not guards["fp_per_case_increase_within_0.5"]
    if core_count >= 2 and all(guards.values()):
        label = "PROMOTE TO 300"
    elif recall_up and specificity_tradeoff and not core["mean_dice_improved"]:
        label = "RECALL-PRECISION TRADE-OFF"
    else:
        label = "STOP"
    return {"decision": label, "core_signal_count": core_count, "core": core, "guards": guards}


def paired_rows(base: dict, cand: dict) -> list[dict]:
    rows = []
    for case in sorted(base):
        b, c = base[case], cand[case]
        rows.append({
            "case": case, "gt_ml": b["gt_ml"], "baseline_dice": b["dice"], "candidate_dice": c["dice"],
            "delta_dice": nullable_delta(c["dice"], b["dice"]),
            "baseline_lesion_recall": b["lesion_recall"], "candidate_lesion_recall": c["lesion_recall"],
            "delta_lesion_recall": nullable_delta(c["lesion_recall"], b["lesion_recall"]),
            "baseline_lesion_fn": b["lesion_fn"], "candidate_lesion_fn": c["lesion_fn"],
            "baseline_lesion_fp": b["lesion_fp"], "candidate_lesion_fp": c["lesion_fp"],
            "baseline_pred_gt_ratio": b["pred_gt_ratio"], "candidate_pred_gt_ratio": c["pred_gt_ratio"],
            "baseline_signed_volume_error_ml": b["signed_volume_error_ml"],
            "candidate_signed_volume_error_ml": c["signed_volume_error_ml"],
        })
    return rows


def case_panels(out: Path, rows: list[dict], candidate_pred: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = [r for r in rows if r["delta_dice"] is not None]
    ranked.sort(key=lambda r: r["delta_dice"])
    reasons = {}
    for label, cases in [
        ("top5_decline", [r["case"] for r in ranked[:5]]),
        ("top5_improve", [r["case"] for r in ranked[-5:]]),
        ("known_failure", KNOWN_FAILURES),
    ]:
        for case in cases:
            reasons.setdefault(case, []).append(label)
    figdir = out / "case_analysis/figures"
    figdir.mkdir(parents=True, exist_ok=False)
    details = []
    for case, why in sorted(reasons.items()):
        paths = [RAW_B0 / "imagesTr" / f"{case}_0000.nii.gz", RAW_P3 / "imagesTr" / f"{case}_0000.nii.gz",
                 RAW_B0 / "labelsTr" / f"{case}.nii.gz", B0_PRED / f"{case}.nii.gz", candidate_pred / f"{case}.nii.gz"]
        imgs = [nib.load(p) for p in paths]
        if not all(i.shape == imgs[0].shape and np.allclose(i.affine, imgs[0].affine, atol=1e-3) for i in imgs):
            raise RuntimeError(f"Case panel geometry mismatch: {case}")
        arrays = [nib.as_closest_canonical(i).get_fdata(dtype=np.float32) for i in imgs]
        original, clipped = arrays[:2]
        gt, b0, p3 = [x > .5 for x in arrays[2:]]
        z_gt = int(np.argmax(gt.sum((0, 1))))
        z_diff = int(np.argmax((b0 != p3).sum((0, 1))))
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        for row, z in enumerate([z_gt, z_diff]):
            for col, (name, image, mask) in enumerate([
                ("Original DWI", original, None), ("P3a clipped DWI", clipped, None), ("GT", original, gt),
                ("B0-100", original, b0), ("P3a", original, p3),
            ]):
                ax = axes[row, col]
                foreground = image[image != 0]
                lo, hi = np.percentile(foreground, [1, 99]) if foreground.size else (image.min(), image.max())
                ax.imshow(image[:, :, z].T, origin="lower", cmap="gray", vmin=lo, vmax=hi)
                if mask is not None:
                    overlay = np.ma.masked_where(~mask[:, :, z].T, mask[:, :, z].T)
                    ax.imshow(overlay, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=.55)
                    if gt[:, :, z].any():
                        ax.contour(gt[:, :, z].T, levels=[.5], colors=["cyan"], linewidths=.6)
                ax.set_title(f"{name} | z={z}")
                ax.axis("off")
            axes[row, 0].set_ylabel("GT max" if row == 0 else "B0/P3 max disagreement")
        fig.suptitle(f"{case} | {', '.join(why)} | cyan=GT boundary")
        fig.tight_layout()
        fig.savefig(figdir / f"{case}.png", dpi=120)
        plt.close(fig)
        details.append({"case": case, "reasons": why, "slices": [z_gt, z_diff]})
    save(out / "case_analysis/selection.json", details)


def fmt(value, digits=3):
    return "目前資料未提供" if value is None else f"{value:.{digits}f}"


def analyse_screen100() -> None:
    out = SUITE / "screen100/analysis"
    out.mkdir(parents=True, exist_ok=False)
    base = load_cases(B0_METRICS)
    cand = load_cases(SUITE / "screen100/metrics")
    ids = sorted(base)
    small = [i for i in ids if 1 <= base[i]["gt_ml"] < 5]
    rows = paired_rows(base, cand)
    write_csv(out / "paired_cases.csv", rows)
    base_summary = snapshot(json.loads((B0_METRICS / "summary.json").read_text()))
    cand_summary = snapshot(json.loads((SUITE / "screen100/metrics/summary.json").read_text()))
    curve = curve_status("screen100")
    decision = promotion_decision(base_summary, cand_summary, curve)
    result = {
        "comparison": "P3a-100 minus matched B0-R2022-100", "iterations": 10000, "seed": 2022,
        "resampling_unit": "paired validation case ID; lesion counts re-aggregated per resample",
        "baseline": base_summary, "candidate": cand_summary, "curve": curve,
        "overall_bootstrap": bootstrap(base, cand, ids, n=10000),
        "small_1_5_ml_bootstrap": bootstrap(base, cand, small, n=10000),
        "promotion": decision,
    }
    save(out / "comparison.json", result)
    save(out / "decision.json", decision)
    completion = json.loads((SUITE / "screen100/train/completion.json").read_text())
    case_panels(out, rows, Path(completion["predictions"]))
    lines = ["# P3a-100 result", "", "Primary comparison: matched B0-R2022-100 vs P3a-100; candidate minus baseline.", "",
             "| Metric | B0-100 | P3a-100 | Delta |", "|---|---:|---:|---:|"]
    for key in ["mean_dice", "median_dice", "lesion_recall", "lesion_precision", "fp_lesions_per_case", "severe_under_count", "volume_mae_ml", "small_mean_dice", "small_recall"]:
        lines.append(f"| {key} | {fmt(base_summary[key])} | {fmt(cand_summary[key])} | {fmt(cand_summary[key]-base_summary[key])} |")
    lines += ["", f"## Promotion decision: {decision['decision']}", "", f"Core signals: {decision['core_signal_count']}/4. Guards: {decision['guards']}", "",
              "Fold 0 only; single-seed screening; same seed is not fully deterministic. Paired bootstrap CIs describe this validation cohort, not cross-fold generalization."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


def analyse_promote300() -> None:
    out = SUITE / "promote300/analysis"
    out.mkdir(parents=True, exist_ok=False)
    p100 = load_cases(SUITE / "screen100/metrics")
    p300 = load_cases(SUITE / "promote300/metrics")
    ids = sorted(p100)
    small = [i for i in ids if 1 <= p100[i]["gt_ml"] < 5]
    rows = paired_rows(p100, p300)
    write_csv(out / "paired_cases_p3a100_vs_p3a300.csv", rows)
    s100 = snapshot(json.loads((SUITE / "screen100/metrics/summary.json").read_text()))
    s300 = snapshot(json.loads((SUITE / "promote300/metrics/summary.json").read_text()))
    b100 = snapshot(json.loads((B0_METRICS / "summary.json").read_text()))
    curve = curve_status("promote300")
    contextual = promotion_decision(b100, s300, curve)
    trend_balanced = (
        s300["mean_dice"] >= s100["mean_dice"] - 0.005
        and s300["lesion_recall"] >= s100["lesion_recall"] - 0.01
        and s300["severe_under_count"] <= s100["severe_under_count"]
        and s300["small_mean_dice"] >= s100["small_mean_dice"] - 0.01
    )
    tradeoff = (
        s300["lesion_recall"] > b100["lesion_recall"]
        and (s300["lesion_precision"] - b100["lesion_precision"] < -MAX_PRECISION_DROP
             or s300["fp_lesions_per_case"] - b100["fp_lesions_per_case"] > MAX_FP_PER_CASE_INCREASE)
    )
    if contextual["decision"] == "PROMOTE TO 300" and trend_balanced:
        final = "CANDIDATE FOR FULL TRAINING"
    elif tradeoff:
        final = "RECALL-PRECISION TRADE-OFF"
    else:
        final = "STOP"
    result = {
        "primary_trend": "P3a-300 minus P3a-100", "iterations": 10000, "seed": 2022,
        "p3a100": s100, "p3a300": s300, "b0_100_context_only": b100, "curve": curve,
        "trend_bootstrap": bootstrap(p100, p300, ids, n=10000),
        "small_1_5_ml_trend_bootstrap": bootstrap(p100, p300, small, n=10000),
        "matched_b0_300_available": False,
        "limitation": "No B0-R2022 epoch-300 checkpoint exists; B0-100 is context only and is not a matched 300-epoch primary comparator.",
        "final_decision": final, "automatic_1000_started": False,
    }
    save(out / "trend_and_final_decision.json", result)
    completion = json.loads((SUITE / "promote300/train/completion.json").read_text())
    case_panels(out, rows, Path(completion["predictions"]))
    lines = ["# P3a-300 result", "", "Primary trend comparison: P3a-100 vs P3a-300.",
             "B0-R2022-300 is unavailable; B0-100 is contextual only, not a horizon-matched primary comparison.", "",
             "| Metric | P3a-100 | P3a-300 | Delta |", "|---|---:|---:|---:|"]
    for key in ["mean_dice", "median_dice", "lesion_recall", "lesion_precision", "fp_lesions_per_case", "severe_under_count", "volume_mae_ml", "small_mean_dice", "small_recall"]:
        lines.append(f"| {key} | {fmt(s100[key])} | {fmt(s300[key])} | {fmt(s300[key]-s100[key])} |")
    lines += ["", f"## Final decision: {final}", "", "No 1000-epoch run was started."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"final_decision": final}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen100", "promote300"], required=True)
    args = parser.parse_args()
    analyse_screen100() if args.stage == "screen100" else analyse_promote300()
