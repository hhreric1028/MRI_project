import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_e1_dataset_test", ROOT / "scripts/preprocessing/prepare_e1_dataset.py")
SAMPLING = load("analyze_s1_sampling_test", ROOT / "scripts/preprocessing/analyze_s1_sampling.py")
EVALUATION = load("evaluate_v2_candidate_test", ROOT / "scripts/evaluation/evaluate_v2_candidate.py")


class V2ProtocolTests(unittest.TestCase):
    def test_p3a_clip_preserves_background_and_clips_foreground(self):
        image = np.zeros((3, 3, 3), dtype=np.float32)
        image[1, 1, :] = [1, 5, 100]
        clipped, foreground, lower, upper = PREPARE.p3a_clip(image)
        self.assertTrue(np.all(clipped[~foreground] == 0))
        self.assertEqual(clipped.dtype, np.float32)
        self.assertGreater(lower, 0)
        self.assertLess(upper, 100)
        self.assertTrue(np.all(clipped[foreground] >= lower))
        self.assertTrue(np.all(clipped[foreground] <= upper))

    def test_exact_low_rank_mask_is_componentwise_and_deterministic(self):
        seg = np.zeros((5, 5, 5), dtype=bool)
        seg[0, 0, 0:5] = True
        seg[4, 4, 0:5] = True
        data = np.arange(seg.size, dtype=np.float32).reshape(seg.shape)
        first = SAMPLING.exact_low_rank_mask(data, seg, 20)
        second = SAMPLING.exact_low_rank_mask(data, seg, 20)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), 2)
        self.assertTrue(first[0, 0, 0])
        self.assertTrue(first[4, 4, 0])

    def test_percentile_selection_uses_training_distribution_adequacy(self):
        summaries = [
            {
                "percentile": 20,
                "all_cases_have_hard_voxels": True,
                "median_hard_fraction": 0.20,
                "p10_hard_voxels": 11,
                "mean_standardized_easy_minus_hard": 2.2,
            },
            {
                "percentile": 30,
                "all_cases_have_hard_voxels": True,
                "median_hard_fraction": 0.30,
                "p10_hard_voxels": 17,
                "mean_standardized_easy_minus_hard": 2.3,
            },
        ]
        percentile, _ = SAMPLING.select_percentile(summaries)
        self.assertEqual(percentile, 20)

    def test_promotion_requires_all_guards(self):
        base = {
            "volume_mae_ml": 4.0, "severe_under_count": 8,
            "fixed_worst5_mean_dice": 0.30, "fixed_worst5_mean_voxel_recall": 0.25,
            "coverage_cohort_mean_voxel_recall": 0.40, "lesion_recall": 0.60,
            "lesion_precision": 0.72, "fp_lesions_per_case": 2.0,
        }
        candidate = dict(base)
        candidate.update({
            "severe_under_count": 7, "fixed_worst5_mean_dice": 0.35,
            "coverage_cohort_mean_voxel_recall": 0.45, "lesion_recall": 0.61,
            "lesion_precision": 0.69, "fp_lesions_per_case": 2.3, "volume_mae_ml": 4.2,
        })
        stable = {"total_nonfinite_gradient_events": 0, "unrecovered_consecutive_overflows": 0}
        decision = EVALUATION.decide(
            base, candidate, {"voxel_recall": 0.3}, {"voxel_recall": 0.35}, stable, "screen100"
        )
        self.assertTrue(decision["promote_to_300"])
        candidate["lesion_precision"] = 0.60
        decision = EVALUATION.decide(
            base, candidate, {"voxel_recall": 0.3}, {"voxel_recall": 0.35}, stable, "screen100"
        )
        self.assertFalse(decision["promote_to_300"])
        self.assertEqual(decision["classification"], "RECALL-PRECISION TRADE-OFF")


if __name__ == "__main__":
    unittest.main()
