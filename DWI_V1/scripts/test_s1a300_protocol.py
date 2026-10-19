"""Protocol checks for S1a epoch-100 to epoch-300 continuation."""

import json
import unittest

import numpy as np
import torch

from recall_common import digest
from run_s1a300_experiment import SOURCE_CHECKPOINT, SOURCE_RUN, make_trainer, source_state


class S1a300ProtocolTests(unittest.TestCase):
    def test_source_checkpoint_and_promotion(self):
        state = source_state()
        completion = json.loads((SOURCE_RUN / "train/completion.json").read_text())
        self.assertEqual(state["checkpoint_epoch"], 100)
        self.assertEqual(state["promotion_decision"], "PROMOTE TO 300 EPOCHS")
        self.assertEqual(digest(SOURCE_CHECKPOINT), completion["checkpoint_sha256"])

    def test_trainer_configuration_and_lr_continuity(self):
        trainer = make_trainer("cpu")
        self.assertEqual(trainer.num_epochs, 300)
        self.assertEqual(trainer.lr_schedule_max_epochs, 1000)
        self.assertEqual(trainer.train_foreground, 0.50)
        trainer.initialize()
        trainer.lr_scheduler.step(100)
        expected = 0.01 * (1 - 100 / 1000) ** 0.9
        self.assertTrue(np.isclose(trainer.optimizer.param_groups[0]["lr"], expected, rtol=1e-12, atol=1e-14))
        train, validation = trainer.raw_loaders()
        self.assertEqual(train.oversample_foreground_percent, 0.50)
        self.assertEqual(validation.oversample_foreground_percent, 0.33)

    def test_checkpoint_has_optimizer_and_scaler_state(self):
        checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["current_epoch"], 100)
        self.assertTrue(checkpoint["optimizer_state"]["state"])
        self.assertIsNotNone(checkpoint["grad_scaler_state"])


if __name__ == "__main__":
    unittest.main()
