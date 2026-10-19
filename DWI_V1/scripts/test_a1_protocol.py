"""Protocol guardrails for the A1-100 single-variable experiment."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from a1_trainers import (  # noqa: E402
    A1_INTENSITY_AUGMENTATION,
    B0_INTENSITY_AUGMENTATION,
    nnUNetTrainer_A1_DWIIntensityAug_Screen100,
)


class TestA1Protocol(unittest.TestCase):
    def test_only_declared_intensity_branches_differ(self):
        self.assertEqual(A1_INTENSITY_AUGMENTATION["gaussian_blur"], B0_INTENSITY_AUGMENTATION["gaussian_blur"])
        self.assertEqual(A1_INTENSITY_AUGMENTATION["gaussian_noise"]["variance"], [0.0, 0.1])
        self.assertEqual(A1_INTENSITY_AUGMENTATION["multiplicative_brightness"]["range"], [0.85, 1.15])
        self.assertEqual(A1_INTENSITY_AUGMENTATION["contrast"]["range"], [0.75, 1.25])
        self.assertEqual(A1_INTENSITY_AUGMENTATION["gamma"]["range"], [0.7, 1.5])
        self.assertEqual(A1_INTENSITY_AUGMENTATION["gamma_invert"]["apply_probability"], 0.0)

    def test_training_controls_inherited(self):
        self.assertEqual(nnUNetTrainer_A1_DWIIntensityAug_Screen100.screening_epochs, 100)
        self.assertEqual(nnUNetTrainer_A1_DWIIntensityAug_Screen100.lr_schedule_max_epochs, 1000)
        self.assertEqual(nnUNetTrainer_A1_DWIIntensityAug_Screen100.train_foreground, 0.33)

    def test_b0_plans_unchanged(self):
        plans = json.loads(
            (ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/nnUNetPlans.json").read_text()
        )
        cfg = plans["configurations"]["3d_fullres"]
        self.assertEqual(cfg["spacing"], [2.0, 2.0, 2.0])
        self.assertEqual(cfg["patch_size"], [80, 96, 80])
        self.assertEqual(cfg["batch_size"], 8)

    def test_preview_gate_passed(self):
        preview = json.loads(
            (ROOT / "outputs/analysis/A1_augmentation_preview/sanity_check.json").read_text()
        )
        self.assertEqual(preview["quantitative_status"], "passed")
        self.assertEqual(preview["visual_review_status"], "passed")
        self.assertTrue(preview["all_masks_identical"])
        self.assertTrue(preview["all_outputs_finite"])


if __name__ == "__main__":
    unittest.main()
