"""Inference-only validation for existing B0 and replication best checkpoints.

This script never calls training and writes only beneath the dedicated FP audit
directory. Existing checkpoints, predictions, splits, and metric outputs are read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from recall_common import PROTECTED, ROOT, protect_baseline
from run_kamel_fold import make_trainer as make_replication_trainer
from run_screen100_experiment import make_trainer as make_screening_trainer


OUT = ROOT / "outputs/analysis/kamel_replication/fold0_fp_audit"
OLD_ROOT = ROOT / "experiments/recall_screen100_seed2022_r2/B0_R2022_screen100"
NEW_ROOT = ROOT / "experiments/kamel_replication_seed2022/fold_0"
OLD_FOLD = (
    OLD_ROOT
    / "train/results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres/fold_0"
)
NEW_FOLD = (
    NEW_ROOT
    / "results/Dataset001_ISLES2022_DWI"
    / "nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres/fold_0"
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def infer(label: str, checkpoint: Path) -> dict:
    destination = OUT / "checkpoint_inference" / label
    destination.mkdir(parents=True, exist_ok=False)
    os.environ["nnUNet_results"] = str(destination / "results")
    torch.set_num_threads(8)
    trainer = (
        make_screening_trainer("B0_R2022_screen100")
        if label == "B0_best"
        else make_replication_trainer(0)
    )
    trainer.load_checkpoint(str(checkpoint))
    if trainer.current_epoch not in (68, 94):
        raise RuntimeError(f"Unexpected best-checkpoint epoch for {label}: {trainer.current_epoch}")
    trainer.perform_actual_validation(save_probabilities=True)
    prediction_folder = Path(trainer.output_folder) / "validation"
    nii = sorted(prediction_folder.glob("*.nii.gz"))
    npz = sorted(prediction_folder.glob("*.npz"))
    if len(nii) != 50 or len(npz) != 50:
        raise RuntimeError(f"Incomplete inference-only output for {label}")
    result = {
        "label": label,
        "mode": "inference_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_epoch": int(trainer.current_epoch),
        "trainer_class": trainer.__class__.__name__,
        "allowed_mirroring_axes": list(trainer.inference_allowed_mirroring_axes),
        "predictions": str(prediction_folder),
        "prediction_masks": len(nii),
        "probability_files": len(npz),
        "training_called": False,
    }
    (destination / "inference_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Audit output already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    PROTECTED.extend([OLD_ROOT, NEW_ROOT])
    protect_baseline()
    results = [
        infer("B0_best", OLD_FOLD / "checkpoint_best.pth"),
        infer("Replication_best", NEW_FOLD / "checkpoint_best.pth"),
    ]
    (OUT / "checkpoint_inference_manifest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
