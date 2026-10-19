"""Fast protocol invariants for the DWI enhancement batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from image_enhancement_analysis import decision, promotion_rank
from image_enhancement_common import (
    BASE_PP,
    BASE_RAW,
    EXPERIMENTS,
    RING_OUTER_MM,
    clahe_enhance,
    foreground_mask,
    foreground_zscore,
    gamma_enhance,
    intensity_scaled,
    local_ring,
)


def synthetic_summary(**updates):
    result = {
        "mean_dice": 0.70, "median_dice": 0.72, "sd_dice": 0.2, "hd95_mean_mm": 10.0,
        "volume_mae_ml": 4.0, "signed_volume_error_ml": -0.5,
        "lesion_recall": 0.60, "lesion_precision": 0.70, "lesion_f1": 0.64,
        "lesion_tp": 60, "lesion_fn": 40, "lesion_fp": 25, "fp_lesions_per_case": 0.5,
        "severe_under_count": 8, "severe_under_rate": 0.16,
        "catastrophic_miss_count": 1, "dice_lt_05": 5, "dice_lt_03": 2,
        "small_cases": 12, "small_mean_dice": 0.55, "small_median_dice": 0.58,
        "small_recall": 0.50, "small_precision": 0.65, "small_fn": 10, "small_fp": 5,
        "small_volume_mae_ml": 1.0,
    }
    result.update(updates)
    return result


def test_transforms() -> None:
    grid = np.indices((20, 22, 12)).sum(axis=0).astype(np.float32)
    brain = np.zeros_like(grid, dtype=bool)
    brain[2:18, 3:20, 1:11] = True
    image = np.zeros_like(grid)
    image[brain] = grid[brain] + 10
    z = foreground_zscore(image, brain)
    gamma = gamma_enhance(image, brain, 1.2)
    clahe = clahe_enhance(image, brain, 0.01, (8, 8))
    scaled = intensity_scaled(image, brain, 0.8, 0)
    for output in [z, gamma, clahe, scaled]:
        assert np.isfinite(output).all()
        assert np.all(output[~brain] == 0)
    assert np.allclose(scaled[brain], 0.8 * z[brain], atol=1e-6)
    assert not np.allclose(gamma[brain], image[brain])
    assert clahe[brain].min() >= 0 and clahe[brain].max() <= 1


def test_ring() -> None:
    lesion = np.zeros((11, 11, 11), dtype=bool)
    lesion[5, 5, 5] = True
    brain = np.ones_like(lesion)
    ring = local_ring(lesion, brain, (2.0, 2.0, 2.0))
    assert not ring[5, 5, 5]
    assert ring[8, 5, 5]
    assert not ring[9, 5, 5]
    assert RING_OUTER_MM == 6.0


def test_decisions_and_cap() -> None:
    base = synthetic_summary()
    good = synthetic_summary(mean_dice=0.72, lesion_recall=0.64, severe_under_count=7)
    curve = {"normal": True}
    assert decision(base, good, curve)["decision"] == "PROMOTE"
    tradeoff = synthetic_summary(lesion_recall=0.70, lesion_precision=0.60, fp_lesions_per_case=1.2)
    assert decision(base, tradeoff, curve)["decision"] == "RECALL-PRECISION TRADE-OFF"
    screens = {}
    for index, name in enumerate(EXPERIMENTS):
        candidate = synthetic_summary(mean_dice=0.71 + index * 0.01, lesion_recall=0.62 + index * 0.01, severe_under_count=7 - index)
        screens[name] = {"baseline": base, "candidate": candidate, "decision": {"decision": "PROMOTE"}}
    selected = promotion_rank(screens)
    assert len(selected) == 2


def test_dataset_and_split_invariants() -> None:
    ids = [spec["dataset_id"] for spec in EXPERIMENTS.values()]
    assert ids == [21, 22, 23] and len(ids) == len(set(ids))
    split = json.loads((BASE_PP / "splits_final.json").read_text())[0]
    assert len(split["train"]) == 200 and len(split["val"]) == 50
    assert not set(split["train"]) & set(split["val"])
    assert (BASE_RAW / "dataset.json").is_file()


def test_no_full_training_dispatch() -> None:
    pipeline = Path(__file__).with_name("image_enhancement_pipeline.py").read_text(encoding="utf-8")
    assert '"screen100"' in pipeline and '"promote300"' in pipeline
    assert "1000-epoch run started" in pipeline
    assert "full1000" not in pipeline and "run_1000" not in pipeline


def main() -> None:
    test_transforms()
    test_ring()
    test_decisions_and_cap()
    test_dataset_and_split_invariants()
    test_no_full_training_dispatch()
    print("image enhancement protocol tests: PASS", flush=True)


if __name__ == "__main__":
    main()

