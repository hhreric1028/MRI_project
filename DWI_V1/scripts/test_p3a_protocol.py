"""CPU-level protocol tests for the P3a controlled preprocessing ablation."""

import json
import unittest

import numpy as np

from p3a_trainers import P3_PP
from recall_common import ROOT, digest
from run_p3a_experiment import make_trainer


class P3aProtocolTests(unittest.TestCase):
    def test_dataset_and_split_identity(self):
        base = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
        self.assertEqual(digest(base / "splits_final.json"), digest(P3_PP / "splits_final.json"))
        self.assertEqual(len(list((P3_PP / "nnUNetPlans_3d_fullres").glob("case_*.pkl"))), 250)
        self.assertEqual(len(list((P3_PP / "gt_segmentations").glob("case_*.nii.gz"))), 250)

    def test_plans_match_controlled_baseline(self):
        base = json.loads((ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI/nnUNetPlans.json").read_text())
        p3 = json.loads((P3_PP / "nnUNetPlans.json").read_text())
        keys = ["batch_size", "patch_size", "spacing", "normalization_schemes", "use_mask_for_norm", "architecture", "batch_dice"]
        for key in keys:
            self.assertEqual(base["configurations"]["3d_fullres"][key], p3["configurations"]["3d_fullres"][key])

    def test_screen100_and_promote300_lr_horizon(self):
        for stage, epochs in [("screen100", 100), ("promote300", 300)]:
            trainer = make_trainer(stage, "cpu")
            self.assertEqual(trainer.num_epochs, epochs)
            self.assertEqual(trainer.lr_schedule_max_epochs, 1000)
            self.assertEqual(trainer.train_foreground, 0.33)
            trainer.initialize()
            trainer.lr_scheduler.step(99)
            expected = 0.01 * (1 - 99 / 1000) ** 0.9
            self.assertTrue(np.isclose(trainer.optimizer.param_groups[0]["lr"], expected, rtol=1e-12, atol=1e-14))
            train, validation = trainer.raw_loaders()
            self.assertEqual(train.oversample_foreground_percent, 0.33)
            self.assertEqual(validation.oversample_foreground_percent, 0.33)


if __name__ == "__main__":
    unittest.main()
