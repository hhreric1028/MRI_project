import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "failure_analysis" / "analyze_case_0140.py"
SPEC = importlib.util.spec_from_file_location("failure_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FailureAnalysisTests(unittest.TestCase):
    def test_mask_metrics_exact_counts(self):
        gt = np.zeros((3, 3, 3), dtype=bool)
        pred = np.zeros_like(gt)
        gt[0, 0, 0] = True
        gt[0, 0, 1] = True
        pred[0, 0, 0] = True
        pred[2, 2, 2] = True
        result = MODULE.mask_metrics(gt, pred, 0.008)
        self.assertAlmostEqual(result["dice"], 0.5)
        self.assertAlmostEqual(result["voxel_recall"], 0.5)
        self.assertAlmostEqual(result["voxel_precision"], 0.5)
        self.assertAlmostEqual(result["fn_ml"], 0.008)
        self.assertAlmostEqual(result["fp_ml"], 0.008)

    def test_connected_component_detection(self):
        gt = np.zeros((6, 6, 6), dtype=bool)
        pred = np.zeros_like(gt)
        gt[0, 0, 0] = True
        gt[5, 5, 5] = True
        pred[0, 0, 0] = True
        result = MODULE.mask_metrics(gt, pred, 1.0)
        self.assertEqual(result["gt_components"], 2)
        self.assertEqual(result["lesion_tp"], 1)
        self.assertEqual(result["lesion_fn"], 1)
        self.assertAlmostEqual(result["lesion_recall"], 0.5)

    def test_output_root_is_separate_from_v1(self):
        _, v1, v2 = MODULE.load_config()
        self.assertNotEqual(v1, v2)
        self.assertTrue(v2.is_relative_to(v1))


if __name__ == "__main__":
    unittest.main()
