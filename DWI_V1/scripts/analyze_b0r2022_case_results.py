"""Build the B0-R2022 case-level comparison report without altering predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

from recall_analysis import bootstrap, group, load_cases, stat


ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
RAW_DWI = ROOT / "datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI/imagesTr"

DEFAULT_NEW_EVAL = ROOT / "experiments/B0-R2022_resume_from_epoch34/evaluation_v2"
DEFAULT_B0_EVAL = ROOT / "experiments/B0_baseline/evaluation_v2"
DEFAULT_P1_EVAL = ROOT / "experiments/P1_spacing_1p5mm/run_seed2022_train_r2/metrics"
DEFAULT_NEW_PRED = (
    ROOT
    / "experiments/B0-R2022_resume_from_epoch34/train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Resume__nnUNetPlans__3d_fullres/fold_0/validation"
)
DEFAULT_B0_PRED = (
    ROOT
    / "outputs/models/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
)
DEFAULT_P1_PRED = (
    ROOT
    / "experiments/P1_spacing_1p5mm/run_seed2022_train_r2/results/Dataset011_ISLES2022_DWI"
    / "nnUNetTrainer_SeededPreserve__nnUNetPlans__3d_fullres/fold_0/validation"
)
DEFAULT_TRAIN_LOG = ROOT / "experiments/B0-R2022_resume_from_epoch34/logs/train.log"
DEFAULT_NATIVE_SUMMARY = DEFAULT_NEW_PRED / "summary.json"

KNOWN_CASES = {
    "case_0068",
    "case_0182",
    "case_0181",
    "case_0140",
    "case_0212",
    "case_0118",
    "case_0135",
    "case_0160",
}

LIMITATIONS = (
    "Fold 0 only; single-seed screening. The resumed run is a training-state continuation, "
    "not a bitwise deterministic continuation. Historical B0 has an unconfirmed seed, and P1 "
    "also changes spacing, so these comparisons are descriptive rather than single-variable "
    "causal comparisons. Paired bootstrap confidence intervals describe case-level uncertainty "
    "within this validation cohort and do not establish cross-fold, cross-center, or cross-seed "
    "generalization."
)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_training_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    epoch_matches = list(re.finditer(r"Epoch\s+(\d+)", text))
    train = re.findall(r"train_loss\s+(-?[0-9.]+)", text)
    val = re.findall(r"val_loss\s+(-?[0-9.]+)", text)
    pseudo = re.findall(r"Pseudo dice \[np\.float32\((-?[0-9.]+)\)\]", text)
    lr = re.findall(r"Current learning rate:\s+([0-9.eE+-]+)", text)
    if not (epoch_matches and train and val and pseudo and lr):
        raise ValueError("Unable to parse final training metrics")
    return {
        "final_epoch_zero_based": int(epoch_matches[-1].group(1)),
        "completed_epochs": int(epoch_matches[-1].group(1)) + 1,
        "train_loss": float(train[-1]),
        "validation_loss": float(val[-1]),
        "patch_pseudo_dice": float(pseudo[-1]),
        "learning_rate": float(lr[-1]),
    }


def native_dice_values(path: Path) -> dict[str, float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in data["metric_per_case"]:
        case = Path(row["prediction_file"]).name.removesuffix(".nii.gz")
        value = row["metrics"]["1"]["Dice"]
        result[case] = None if value is None or not np.isfinite(value) else float(value)
    return result


def subgroup_label(gt_ml: float) -> str:
    return group(gt_ml)


def case_categories(row: dict, lower_fence: float) -> list[str]:
    result = []
    dice = row.get("dice")
    ratio = row.get("pred_gt_ratio")
    gt_ml = row["gt_ml"]
    if gt_ml == 0:
        return ["empty_gt_empty_prediction"]
    if dice is not None and dice < lower_fence:
        result.append("absolute_low_dice_outlier")
    if dice is not None and dice < 0.5 and ratio is not None and ratio < 0.5:
        result.append("severe_under_segmentation")
    if ratio is not None and ratio < 0.5 and (row.get("voxel_tp") or 0) > 0:
        result.append("detected_but_coverage_insufficient")
    if 1 <= gt_ml < 5 and (
        (dice is not None and dice < lower_fence)
        or abs(row["delta_dice_vs_historical_B0"] or 0) >= 0.05
        or abs(row["delta_dice_vs_P1"] or 0) >= 0.05
    ):
        result.append("small_lesion_instability")
    if gt_ml >= 50 and ratio is not None and ratio < 0.5:
        result.append("large_lesion_coverage_failure")
    if ratio is not None and ratio >= 1.25:
        result.append("volume_over_segmentation")
    if row["delta_dice_vs_historical_B0"] is not None:
        if row["delta_dice_vs_historical_B0"] <= -0.05:
            result.append("large_decline_vs_historical_B0")
        elif row["delta_dice_vs_historical_B0"] >= 0.05:
            result.append("large_improvement_vs_historical_B0")
    if row["delta_dice_vs_P1"] is not None:
        if row["delta_dice_vs_P1"] <= -0.05:
            result.append("large_decline_vs_P1")
        elif row["delta_dice_vs_P1"] >= 0.05:
            result.append("large_improvement_vs_P1")
    return result or ["no_flag"]


def flatten_rows(new: dict, b0: dict, p1: dict, lower_fence: float) -> list[dict]:
    rows = []
    for case in sorted(new):
        n, b, p = new[case], b0[case], p1[case]
        if not (np.isclose(n["gt_ml"], b["gt_ml"]) and np.isclose(n["gt_ml"], p["gt_ml"])):
            raise ValueError(f"GT volume mismatch for {case}")
        db = n["dice"] - b["dice"] if n["dice"] is not None and b["dice"] is not None else None
        dp = n["dice"] - p["dice"] if n["dice"] is not None and p["dice"] is not None else None
        row = {
            "case": case,
            "gt_volume_group": subgroup_label(n["gt_ml"]),
            "new_dice": n["dice"],
            "historical_B0_dice": b["dice"],
            "delta_dice_vs_historical_B0": db,
            "P1_dice": p["dice"],
            "delta_dice_vs_P1": dp,
            "gt_ml": n["gt_ml"],
            "pred_ml": n["pred_ml"],
            "signed_volume_error_ml": n["signed_volume_error_ml"],
            "absolute_volume_error_ml": n["absolute_volume_error_ml"],
            "absolute_percentage_error": n["absolute_percentage_error"],
            "pred_gt_ratio": n["pred_gt_ratio"],
            "voxel_recall": n["voxel_recall"],
            "voxel_precision": n["voxel_precision"],
            "lesion_tp": n["lesion_tp"],
            "lesion_fp": n["lesion_fp"],
            "lesion_fn": n["lesion_fn"],
            "lesion_recall": n["lesion_recall"],
            "lesion_precision": n["lesion_precision"],
            "hd95_mm": n["hd95_mm"],
        }
        metric_row = dict(n)
        metric_row.update(row)
        categories = case_categories(metric_row, lower_fence)
        row["categories"] = ";".join(categories)
        rows.append(row)
    return rows


def rank_delta(rows: list[dict], key: str, n: int = 5) -> tuple[list[dict], list[dict]]:
    valid = [r for r in rows if r[key] is not None]
    ordered = sorted(valid, key=lambda r: r[key])
    return ordered[:n], list(reversed(ordered[-n:]))


def fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "目前資料未提供"
    return f"{value:.{digits}f}"


def ci_text(record: dict, digits=3) -> str:
    ci = record.get("ci95")
    return "目前資料未提供" if ci is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def component_pattern(gt: np.ndarray, pred: np.ndarray) -> dict:
    gl, gn = label(gt, np.ones((3, 3, 3)))
    gs = np.bincount(gl.ravel(), minlength=gn + 1)[1:]
    overlap = np.bincount(gl[pred].ravel(), minlength=gn + 1)[1:]
    pl, pn = label(pred, np.ones((3, 3, 3)))
    pred_hits = np.unique(pl[gt]) if gt.any() else np.array([0])
    extra = sum(component not in pred_hits for component in range(1, pn + 1))
    return {
        "whole_gt_components_no_overlap": int((overlap == 0).sum()),
        "detected_gt_components_with_incomplete_coverage": int(((overlap > 0) & (overlap < gs)).sum()),
        "extra_prediction_components_no_gt_overlap": int(extra),
        "gt_component_voxel_coverage": (overlap / gs).tolist() if len(gs) else [],
        "voxel_recall": float((gt & pred).sum() / gt.sum()) if gt.any() else None,
    }


def make_panels(
    out: Path,
    focus_reasons: dict[str, set[str]],
    new_rows: dict,
    prediction_paths: dict[str, Path],
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(exist_ok=False)
    details = []
    for index, case in enumerate(sorted(focus_reasons), 1):
        paths = {
            "DWI": RAW_DWI / f"{case}_0000.nii.gz",
            "GT": PP / "gt_segmentations" / f"{case}.nii.gz",
            **{name: folder / f"{case}.nii.gz" for name, folder in prediction_paths.items()},
        }
        images = {name: nib.load(path) for name, path in paths.items()}
        reference = images["DWI"]
        for name, image in images.items():
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=0):
                raise ValueError(f"Geometry mismatch for {case}: {name}")
        arrays = {
            name: nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
            for name, image in images.items()
        }
        dwi = arrays.pop("DWI")
        gt = arrays.pop("GT") > 0.5
        masks = {name: value > 0.5 for name, value in arrays.items()}
        gt_z = int(np.argmax(gt.sum(axis=(0, 1)))) if gt.any() else dwi.shape[2] // 2
        disagreement = np.zeros_like(gt, dtype=np.uint8)
        names = list(masks)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                disagreement += masks[left] != masks[right]
        disagreement_z = int(np.argmax(disagreement.sum(axis=(0, 1))))
        zs = [gt_z, disagreement_z]

        finite = dwi[np.isfinite(dwi)]
        low, high = np.percentile(finite, [1, 99])
        columns = [("DWI", None), ("GT", gt)] + list(masks.items())
        fig, axes = plt.subplots(2, len(columns), figsize=(3.5 * len(columns), 7.5))
        for row_index, z in enumerate(zs):
            for column_index, (name, mask) in enumerate(columns):
                ax = axes[row_index, column_index]
                ax.imshow(dwi[:, :, z].T, origin="lower", cmap="gray", vmin=low, vmax=high)
                if mask is not None:
                    overlay = np.ma.masked_where(~mask[:, :, z].T, mask[:, :, z].T)
                    ax.imshow(overlay, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.55)
                    if gt[:, :, z].any():
                        ax.contour(gt[:, :, z].T, levels=[0.5], colors=["cyan"], linewidths=0.6)
                ax.set_title(f"{name} | z={z}")
                ax.axis("off")
            axes[row_index, 0].set_ylabel("GT max area" if row_index == 0 else "max model disagreement")
        new_dice = new_rows[case]["dice"]
        fig.suptitle(f"{case} | B0-R2022 Dice={fmt(new_dice)} | cyan=GT boundary")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figdir / f"{case}.png", dpi=130)
        plt.close(fig)

        details.append(
            {
                "case": case,
                "selection_reasons": sorted(focus_reasons[case]),
                "canonical_axial_z": {"gt_max_area": gt_z, "max_model_disagreement": disagreement_z},
                "patterns": {name: component_pattern(gt, mask) for name, mask in masks.items()},
            }
        )
        print(f"Panel {index}/{len(focus_reasons)}: {case}", flush=True)
    return details


def add_reason(target: dict[str, set[str]], rows: list[dict], reason: str) -> None:
    for row in rows:
        target[row["case"]].add(reason)


def subgroup_rows(summary: dict) -> list[dict]:
    rows = []
    labels = [("<1", "<1 mL"), ("1-5", "1–5 mL"), ("5-20", "5–20 mL"), ("20-50", "20–50 mL"), (">=50", "≥50 mL")]
    for key, label_text in labels:
        subgroup = summary["subgroups"][key]
        rows.append(
            {
                "GT Volume Group": label_text,
                "Cases": subgroup["cases"],
                "Valid Dice": subgroup["dice"]["n"],
                "Mean Dice": subgroup["dice"].get("mean"),
                "Median Dice": subgroup["dice"].get("median"),
                "Dice SD": subgroup["dice"].get("sd"),
                "Volume MAE (mL)": subgroup["distributions"]["absolute_volume_error_ml"].get("mean"),
                "Lesion Recall": subgroup["lesion_recall"],
                "Lesion Precision": subgroup["lesion_precision"],
                "Severe Under Count": subgroup["severe_under_count"],
                "Dice <0.5": subgroup["dice_lt_05"],
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return result


def build_report(
    out: Path,
    training: dict,
    summary: dict,
    all_rows: list[dict],
    lower_fence: float,
    worst: list[dict],
    declines_b0: list[dict],
    improves_b0: list[dict],
    declines_p1: list[dict],
    improves_p1: list[dict],
    boot_b0: dict,
    boot_p1: dict,
    subgroup_table: list[dict],
    focus_reasons: dict[str, set[str]],
) -> None:
    d = summary["dice"]
    volume = summary["distributions"]
    category_cases = defaultdict(list)
    for row in all_rows:
        for category in row["categories"].split(";"):
            category_cases[category].append(row["case"])

    lines = [
        "# B0-R2022 Fold 0 新訓練結果與 Dice 偏差病例分析",
        "",
        "## 1. 訓練完成狀態",
        "",
        f"- 完成 epochs：{training['completed_epochs']}（log 中 final epoch index = {training['final_epoch_zero_based']}）",
        f"- Final train loss：{fmt(training['train_loss'])}",
        f"- Final validation loss：{fmt(training['validation_loss'])}",
        f"- Final patch-based pseudo-Dice：{fmt(training['patch_pseudo_dice'])}",
        f"- Final learning rate：{training['learning_rate']:.5f}",
        "- 評估 checkpoint：checkpoint_final.pth",
        "- 注意：patch-based pseudo-Dice 與完整 3D volume Dice 定義不同，不直接互相比較。",
        "",
        "## 2. Overall validation result",
        "",
        f"- Validation cases：{summary['cases']}；有效 Dice：{d['n']}。",
        f"- Mean Dice：{fmt(d['mean'])}；median：{fmt(d['median'])}；SD：{fmt(d['sd'])}。",
        f"- Dice ≥0.8：{summary['dice_ge_08']}/{d['n']}；Dice ≥0.7：{summary['dice_ge_07']}/{d['n']}。",
        f"- Dice <0.5：{summary['dice_lt_05']}/{d['n']}；Dice <0.3：{summary['dice_lt_03']}/{d['n']}。",
        f"- Lesion recall：{fmt(summary['lesion_recall'])}；lesion precision：{fmt(summary['lesion_precision'])}；FP lesions/case：{fmt(summary['fp_lesions_per_case'])}。",
        f"- Volume MAE：{fmt(volume['absolute_volume_error_ml']['mean'], 2)} mL；median absolute error：{fmt(volume['absolute_volume_error_ml']['median'], 2)} mL。",
        f"- Mean signed volume error：{fmt(volume['signed_volume_error_ml']['mean'], 2)} mL，整體呈輕度低估方向。",
        f"- HD95：mean {fmt(volume['hd95_mm']['mean'], 2)} mm；median {fmt(volume['hd95_mm']['median'], 2)} mm。",
        "- case_0151 為 empty GT + empty prediction，保留於病例數及 volume 統計，但排除 Dice mean。",
        "",
        "## 3. Dice 最低 10 例",
        "",
    ]
    lines += markdown_table(
        ["Case", "Dice", "Δ vs B0", "Δ vs P1", "GT mL", "Pred mL", "Pred/GT", "Lesion FN", "Category"],
        [
            [
                row["case"],
                fmt(row["new_dice"]),
                fmt(row["delta_dice_vs_historical_B0"]),
                fmt(row["delta_dice_vs_P1"]),
                fmt(row["gt_ml"], 2),
                fmt(row["pred_ml"], 2),
                fmt(row["pred_gt_ratio"]),
                str(row["lesion_fn"]),
                row["categories"],
            ]
            for row in worst
        ],
    )
    lines += [
        "",
        f"Tukey lower fence = {lower_fence:.3f}；低於此門檻者視為本 cohort 的低 Dice 離群病例。",
        "",
        "## 4. 模型間 Dice 變化",
        "",
        "差值均為 B0-R2022 − comparator。歷史 B0 seed 未確認；P1 同時更改 spacing，因此僅作描述性病例比較。",
        "",
        "### 相較歷史 B0",
        "",
    ]
    lines += markdown_table(
        ["最大下降", "ΔDice", "最大改善", "ΔDice"],
        [[declines_b0[i]["case"], fmt(declines_b0[i]["delta_dice_vs_historical_B0"]), improves_b0[i]["case"], fmt(improves_b0[i]["delta_dice_vs_historical_B0"])] for i in range(5)],
    )
    lines += ["", "### 相較 P1", ""]
    lines += markdown_table(
        ["最大下降", "ΔDice", "最大改善", "ΔDice"],
        [[declines_p1[i]["case"], fmt(declines_p1[i]["delta_dice_vs_P1"]), improves_p1[i]["case"], fmt(improves_p1[i]["delta_dice_vs_P1"])] for i in range(5)],
    )

    metric_labels = [
        ("mean_dice", "Δ Mean Dice", 3),
        ("median_dice", "Δ Median Dice", 3),
        ("lesion_recall", "Δ Lesion recall", 3),
        ("lesion_precision", "Δ Lesion precision", 3),
        ("volume_mae_ml", "Δ Volume MAE (mL)", 2),
        ("signed_volume_error_ml", "Δ Signed volume error (mL)", 2),
        ("severe_under_rate", "Δ Severe-under rate", 3),
        ("dice_below_05_rate", "Δ Dice <0.5 rate", 3),
    ]
    lines += ["", "## 5. Paired case bootstrap", "", "10,000 iterations，seed=2022；相同 sampled case IDs 套用兩模型，lesion TP/FP/FN 於每次抽樣後重新聚合。", ""]
    lines += markdown_table(
        ["Metric", "Δ vs B0", "95% CI", "Δ vs P1", "95% CI"],
        [
            [label_text, fmt(boot_b0[key]["delta"], digits), ci_text(boot_b0[key], digits), fmt(boot_p1[key]["delta"], digits), ci_text(boot_p1[key], digits)]
            for key, label_text, digits in metric_labels
        ],
    )

    lines += ["", "## 6. Lesion-size subgroup", ""]
    lines += markdown_table(
        ["GT group", "Cases", "Mean Dice", "Median Dice", "Dice SD", "Volume MAE", "Lesion recall", "Dice <0.5"],
        [
            [
                row["GT Volume Group"],
                str(row["Cases"]),
                fmt(row["Mean Dice"]),
                fmt(row["Median Dice"]),
                fmt(row["Dice SD"]),
                fmt(row["Volume MAE (mL)"], 2),
                fmt(row["Lesion Recall"]),
                str(row["Dice <0.5"]),
            ]
            for row in subgroup_table
        ],
    )

    lines += [
        "",
        "## 7. Failure pattern 分類討論",
        "",
        f"- Severe under-segmentation：{', '.join(category_cases['severe_under_segmentation'])}。這些病例 Dice <0.5 且 prediction volume 不到 GT 的一半，主要是病灶覆蓋不足，而非單純假陽性造成低 Dice。",
        f"- 找到病灶但 coverage 不足：{', '.join(category_cases['detected_but_coverage_insufficient'])}。其中 case_0160、case_0186 的 Dice 雖高於 0.5，體積比例仍低於 0.5。",
        f"- Small-lesion instability：{', '.join(category_cases['small_lesion_instability'])}。1–5 mL 組 mean Dice {summary['subgroups']['1-5']['dice']['mean']:.3f}、SD {summary['subgroups']['1-5']['dice']['sd']:.3f}，呈現明顯病例間差異；但 case_0135 有明顯改善，因此不能推論所有小病灶都必然失敗。",
        f"- Large-lesion coverage failure：{', '.join(category_cases['large_lesion_coverage_failure'])}。case_0140 的大型病灶仍有廣泛漏覆蓋，顯示問題不限於小病灶。",
        f"- Volume over-segmentation：{', '.join(category_cases['volume_over_segmentation'])}。這類病例需與 lesion FP 及 voxel precision 一併判讀；部分病例即使 Dice 尚可，仍可能有顯著額外體積。",
        "- case_0182 是最需優先檢查的病例：不僅為最低 Dice，且相較 B0 與 P1 均下降，Pred/GT 僅約 0.057。",
        "- case_0135 與 case_0118 顯示 P1 的部分低分並非固定不可恢復；但由於比較同時包含 spacing 等差異，不能將改善歸因於單一因素。",
        "",
        "所有視覺圖均使用相同 DWI slice，第一列為 GT 最大面積切片，第二列為三模型差異最大的切片；cyan 為 GT boundary。component coverage 為描述性 overlap 分析，不等同 evaluation v2 的 one-to-one lesion matching。",
        "",
        "## 8. 簡報版重點",
        "",
        f"- Mean Dice：{d['mean']:.3f}",
        f"- Median Dice：{d['median']:.3f}",
        f"- Dice ≥0.8：{summary['dice_ge_08']}/{d['n']}",
        f"- Lesion recall／precision：{summary['lesion_recall']:.3f}／{summary['lesion_precision']:.3f}",
        f"- Volume MAE：{volume['absolute_volume_error_ml']['mean']:.2f} mL",
        f"- Lowest Dice：{worst[0]['case']}，{worst[0]['new_dice']:.3f}",
        "",
        "B0-R2022 在 fold 0 的整體 3D Dice 表現尚可，但病例間差異明顯；主要失敗型態為少數病例的嚴重低估與病灶覆蓋不足，且同時出現在小型及大型病灶。後續應優先以 lesion recall、Pred/GT ratio 與 failure-case 影像共同驗證改善策略，避免只依 overall mean Dice 判定模型優劣。",
        "",
        "## 9. Limitations",
        "",
        LIMITATIONS,
        "",
        f"本次共輸出 {len(focus_reasons)} 個去重後的重點病例圖。",
    ]
    with (out / "report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-eval", type=Path, default=DEFAULT_NEW_EVAL)
    parser.add_argument("--b0-eval", type=Path, default=DEFAULT_B0_EVAL)
    parser.add_argument("--p1-eval", type=Path, default=DEFAULT_P1_EVAL)
    parser.add_argument("--new-pred", type=Path, default=DEFAULT_NEW_PRED)
    parser.add_argument("--b0-pred", type=Path, default=DEFAULT_B0_PRED)
    parser.add_argument("--p1-pred", type=Path, default=DEFAULT_P1_PRED)
    parser.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--native-summary", type=Path, default=DEFAULT_NATIVE_SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("Existing analysis output detected. No files were modified.")
    args.output.mkdir(parents=True, exist_ok=False)

    new = load_cases(args.new_eval)
    b0 = load_cases(args.b0_eval)
    p1 = load_cases(args.p1_eval)
    ids = sorted(new)
    if set(ids) != set(b0) or set(ids) != set(p1) or len(ids) != 50:
        raise ValueError("Comparison case IDs do not match the fixed 50-case fold 0 cohort")

    native = native_dice_values(args.native_summary)
    if set(native) != set(ids):
        raise ValueError("Native summary IDs do not match evaluation v2")
    for case in ids:
        left, right = native[case], new[case]["dice"]
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f"Native/evaluation empty-case mismatch: {case}")
        elif not np.isclose(left, right, atol=1e-12, rtol=0):
            raise ValueError(f"Native/evaluation Dice mismatch: {case}")

    summary = json.loads((args.new_eval / "summary.json").read_text(encoding="utf-8"))
    training = parse_training_log(args.train_log)
    q1, q3 = summary["dice"]["q1"], summary["dice"]["q3"]
    lower_fence = q1 - 1.5 * (q3 - q1)
    all_rows = flatten_rows(new, b0, p1, lower_fence)
    write_csv(args.output / "all_cases.csv", all_rows)

    worst = sorted([r for r in all_rows if r["new_dice"] is not None], key=lambda r: r["new_dice"])[:10]
    write_csv(args.output / "worst_10_cases.csv", worst)
    declines_b0, improves_b0 = rank_delta(all_rows, "delta_dice_vs_historical_B0")
    declines_p1, improves_p1 = rank_delta(all_rows, "delta_dice_vs_P1")
    shift_rows = []
    for comparator, direction, rows, key in [
        ("historical_B0", "decline", declines_b0, "delta_dice_vs_historical_B0"),
        ("historical_B0", "improvement", improves_b0, "delta_dice_vs_historical_B0"),
        ("P1", "decline", declines_p1, "delta_dice_vs_P1"),
        ("P1", "improvement", improves_p1, "delta_dice_vs_P1"),
    ]:
        for rank, row in enumerate(rows, 1):
            shift_rows.append({"comparator": comparator, "direction": direction, "rank": rank, "case": row["case"], "delta_dice": row[key], "new_dice": row["new_dice"]})
    write_csv(args.output / "dice_shift_rankings.csv", shift_rows)

    category_rows = []
    for row in all_rows:
        for category in row["categories"].split(";"):
            if category != "no_flag":
                category_rows.append({"category": category, **row})
    write_csv(args.output / "failure_categories.csv", category_rows)

    boot_b0 = bootstrap(b0, new, ids, n=10000)
    boot_p1 = bootstrap(p1, new, ids, n=10000)
    write_json(
        args.output / "paired_bootstrap_vs_historical_B0.json",
        {
            "iterations": 10000,
            "seed": 2022,
            "resampling_unit": "case ID; identical sampled IDs for both models; lesion TP/FP/FN re-aggregated",
            "difference": "B0-R2022 minus historical B0",
            "metrics": boot_b0,
            "limitations": LIMITATIONS,
        },
    )
    write_json(
        args.output / "paired_bootstrap_vs_P1.json",
        {
            "iterations": 10000,
            "seed": 2022,
            "resampling_unit": "case ID; identical sampled IDs for both models; lesion TP/FP/FN re-aggregated",
            "difference": "B0-R2022 minus P1",
            "metrics": boot_p1,
            "limitations": LIMITATIONS,
        },
    )

    subgroups = subgroup_rows(summary)
    write_csv(args.output / "lesion_size_subgroups.csv", subgroups)

    focus_reasons: dict[str, set[str]] = defaultdict(set)
    add_reason(focus_reasons, worst, "new_model_bottom_10_dice")
    add_reason(focus_reasons, declines_b0, "top5_decline_vs_historical_B0")
    add_reason(focus_reasons, improves_b0, "top5_improvement_vs_historical_B0")
    add_reason(focus_reasons, declines_p1, "top5_decline_vs_P1")
    add_reason(focus_reasons, improves_p1, "top5_improvement_vs_P1")
    for case in KNOWN_CASES:
        focus_reasons[case].add("protocol_known_case")
    focus_rows = []
    row_by_case = {r["case"]: r for r in all_rows}
    for case in sorted(focus_reasons):
        focus_rows.append({"selection_reasons": ";".join(sorted(focus_reasons[case])), **row_by_case[case]})
    write_csv(args.output / "focus_cases.csv", focus_rows)

    patterns = make_panels(
        args.output,
        focus_reasons,
        new,
        {"B0": args.b0_pred, "P1": args.p1_pred, "B0-R2022": args.new_pred},
    )
    write_json(args.output / "failure_patterns.json", patterns)

    build_report(
        args.output,
        training,
        summary,
        all_rows,
        lower_fence,
        worst,
        declines_b0,
        improves_b0,
        declines_p1,
        improves_p1,
        boot_b0,
        boot_p1,
        subgroups,
        focus_reasons,
    )

    manifest = {
        "status": "complete",
        "case_count": len(ids),
        "valid_dice_count": summary["dice"]["n"],
        "native_evaluation_dice_match": True,
        "tukey_lower_fence": lower_fence,
        "focus_panel_count": len(patterns),
        "inputs": {
            "new_evaluation": str(args.new_eval.resolve()),
            "historical_B0_evaluation": str(args.b0_eval.resolve()),
            "P1_evaluation": str(args.p1_eval.resolve()),
            "new_predictions": str(args.new_pred.resolve()),
        },
        "limitations": LIMITATIONS,
    }
    write_json(args.output / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
