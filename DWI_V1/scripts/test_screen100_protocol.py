"""CPU-level protocol tests for 100-epoch screening."""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from recall_trainers import ForegroundTverskyCE
from run_screen100_experiment import make_trainer, seed_all
from screen100_analysis import nullable_delta
from screen100_trainers import SCREEN_TRAINERS


class Screen100ProtocolTests(unittest.TestCase):
    def test_nullable_case_metric_delta(self):
        self.assertAlmostEqual(nullable_delta(0.75, 0.50), 0.25)
        self.assertIsNone(nullable_delta(None, 0.50))
        self.assertIsNone(nullable_delta(0.50, None))
        self.assertIsNone(nullable_delta(None, None))

    def test_unique_screening_trainers(self):
        self.assertEqual(
            set(SCREEN_TRAINERS),
            {"B0_R2022_screen100", "S1a_screen100_seed2022", "L2a_screen100_seed2022"},
        )
        self.assertEqual(len(set(SCREEN_TRAINERS.values())), 3)

    def test_seed_reproducibility_in_main_process(self):
        seed_all()
        first = (np.random.rand(), torch.rand(3))
        seed_all()
        second = (np.random.rand(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_lr_horizon_and_loader_ratios(self):
        with tempfile.TemporaryDirectory(prefix="screen100_protocol_") as tmp:
            os.environ["nnUNet_results"] = tmp
            for name, ratio in [
                ("B0_R2022_screen100", 0.33),
                ("S1a_screen100_seed2022", 0.50),
                ("L2a_screen100_seed2022", 0.33),
            ]:
                trainer = make_trainer(name, "cpu")
                self.assertEqual(trainer.num_epochs, 100)
                self.assertTrue(hasattr(trainer, "execution_started"))
                self.assertTrue(np.isfinite(trainer.execution_started))
                trainer.initialize()
                self.assertEqual(trainer.lr_scheduler.max_steps, 1000)
                trainer.lr_scheduler.step(99)
                expected = 0.01 * (1 - 99 / 1000) ** 0.9
                self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], expected, places=14)
                self.assertGreater(expected, 0.009)
                train, validation = trainer.raw_loaders()
                self.assertEqual(train.oversample_foreground_percent, ratio)
                self.assertEqual(validation.oversample_foreground_percent, 0.33)

    def test_tversky_direction_and_finite_gradients(self):
        loss = ForegroundTverskyCE()
        gt = torch.zeros(1, 1, 8, 8, 8)
        gt[:, :, 2:6, 2:6, 2:6] = 1
        for prediction in [gt.clone(), torch.zeros_like(gt), torch.ones_like(gt)]:
            logits = torch.cat([1 - prediction, prediction], 1).mul(80).sub(40).requires_grad_()
            value = loss(logits, gt)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(logits.grad).all())
        g = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        fp = g.clone()
        fp[0, 4] = 1
        fn = g.clone()
        fn[0, 0] = 0
        self.assertGreater(loss.tversky(fn, g), loss.tversky(fp, g))


if __name__ == "__main__":
    unittest.main()
