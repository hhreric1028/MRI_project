"""Run the unchanged fixed native-grid evaluation for one fixed validation fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_experiment import metrics, summarize


ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
DEFAULT_STATUS = ROOT / "outputs/analysis/kamel_replication/status.json"


def update_status(status_path: Path, fold: int, **updates) -> None:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status[f"fold_{fold}"].update(updates)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(status_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Existing fold evaluation output detected; no files modified")
    ids = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))[args.fold]["val"]
    nii_ids = {path.name[:-7] for path in args.predictions.glob("*.nii.gz")}
    npz_ids = {path.stem for path in args.predictions.glob("*.npz")}
    if nii_ids != set(ids) or npz_ids != set(ids):
        raise ValueError(f"Fold {args.fold} prediction/probability case IDs differ from fixed validation split")
    rows = []
    for index, case in enumerate(ids, start=1):
        gt_image = nib.load(PP / "gt_segmentations" / f"{case}.nii.gz")
        pred_image = nib.load(args.predictions / f"{case}.nii.gz")
        if gt_image.shape != pred_image.shape or not np.allclose(gt_image.affine, pred_image.affine, atol=1e-3, rtol=0):
            raise ValueError(f"{case} geometry mismatch")
        if gt_image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{case} expected mm units")
        gt = gt_image.get_fdata()
        pred = pred_image.get_fdata()
        if not np.isfinite(gt).all() or not np.isfinite(pred).all():
            raise ValueError(f"{case} nonfinite mask")
        if (
            not np.allclose(gt, np.round(gt), atol=1e-8, rtol=0)
            or not set(np.unique(np.round(gt))).issubset({0, 1})
            or not set(np.unique(pred)).issubset({0, 1})
        ):
            raise ValueError(f"{case} nonbinary mask")
        voxel_ml = float(np.prod(gt_image.header.get_zooms(), dtype=np.float64) / 1000)
        rows.append({"fold": args.fold, "case": case, **metrics(gt > 0.5, pred > 0.5, gt_image.affine, voxel_ml)})
        print(f"fold {args.fold} {index}/50 {case} Dice={rows[-1]['dice']}", flush=True)
    summary = summarize(rows)
    summary["subgroups"] = {
        name: summarize([row for row in rows if low <= row["gt_ml"] < high])
        for name, low, high in [
            ("<1", 0, 1),
            ("1-5", 1, 5),
            ("5-20", 5, 20),
            ("20-50", 20, 50),
            (">=50", 50, float("inf")),
            (">20", 20, float("inf")),
        ]
    }
    summary["definitions"] = {
        "connectivity": 26,
        "matching": "maximum-cardinality then maximum-IoU one-to-one overlap",
        "hd95": "max of directed surface P95; infinite represented by status/count",
        "sd_ddof": 0,
        "checkpoint": "checkpoint_final.pth",
        "fold": args.fold,
        "threshold": 0.5,
        "postprocessing": "none",
        "metric_code": "scripts/evaluate_experiment.py metrics() and summarize()",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [("cases.json", rows), ("summary.json", summary)]:
        (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [key for key in rows[0] if not key.endswith("component_ml")]
    with (args.output / "cases.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    update_status(args.status, args.fold, status="evaluated", validation_status="fixed_evaluation_complete")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
