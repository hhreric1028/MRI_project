"""Static protocol guards for the Original B0 full-five-fold run."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b0_full5fold_trainers import B0_FULL5FOLD_TRAINER  # noqa: E402
from kamel_trainers import KAMEL_TRAINER  # noqa: E402
from recall_common import PP  # noqa: E402
from recall_trainers import nnUNetTrainer_B0R2022  # noqa: E402
from screen100_trainers import Screen100Mixin  # noqa: E402


class TestB0FullFiveFoldProtocol(unittest.TestCase):
    def test_inheritance_keeps_original_b0_and_screen100(self):
        self.assertTrue(issubclass(B0_FULL5FOLD_TRAINER, Screen100Mixin))
        self.assertTrue(issubclass(B0_FULL5FOLD_TRAINER, nnUNetTrainer_B0R2022))
        self.assertEqual(B0_FULL5FOLD_TRAINER.screening_epochs, 100)
        self.assertEqual(B0_FULL5FOLD_TRAINER.lr_schedule_max_epochs, 1000)

    def test_only_split_selector_differs_from_kamel_wrapper(self):
        self.assertIs(B0_FULL5FOLD_TRAINER.raw_loaders, nnUNetTrainer_B0R2022.raw_loaders)
        self.assertIs(B0_FULL5FOLD_TRAINER.get_dataloaders, nnUNetTrainer_B0R2022.get_dataloaders)
        self.assertIs(B0_FULL5FOLD_TRAINER.configure_optimizers, Screen100Mixin.configure_optimizers)
        self.assertIs(B0_FULL5FOLD_TRAINER.raw_loaders, KAMEL_TRAINER.raw_loaders)

    def test_frozen_five_fold_cover(self):
        splits = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))
        cases = {path.name[:-7] for path in (PP / "gt_segmentations").glob("*.nii.gz")}
        self.assertEqual(len(splits), 5)
        self.assertEqual([len(split["train"]) for split in splits], [200] * 5)
        self.assertEqual([len(split["val"]) for split in splits], [50] * 5)
        self.assertEqual({case for split in splits for case in split["val"]}, cases)
        self.assertEqual(len([case for split in splits for case in split["val"]]), len(cases))


if __name__ == "__main__":
    unittest.main()
