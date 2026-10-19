"""Guardrails for the fixed five-fold Kamel-style replication."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kamel_trainers import KAMEL_TRAINER  # noqa: E402
from recall_trainers import nnUNetTrainer_B0R2022  # noqa: E402


class TestKamelProtocol(unittest.TestCase):
    def test_fixed_folds_are_exact_partition(self):
        pp = ROOT / "datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI"
        splits = json.loads((pp / "splits_final.json").read_text())
        all_cases = {path.name[:-7] for path in (pp / "gt_segmentations").glob("*.nii.gz")}
        counts = {case: 0 for case in all_cases}
        self.assertEqual(len(splits), 5)
        for fold, split in enumerate(splits):
            self.assertEqual(len(split["train"]), 200)
            self.assertEqual(len(split["val"]), 50)
            self.assertFalse(set(split["train"]) & set(split["val"]))
            self.assertEqual(set(split["train"]) | set(split["val"]), all_cases)
            instance = KAMEL_TRAINER.__new__(KAMEL_TRAINER)
            instance.fold = fold
            self.assertEqual(instance.do_split(), (split["train"], split["val"]))
            for case in split["val"]:
                counts[case] += 1
        self.assertTrue(all(value == 1 for value in counts.values()))

    def test_b0_controls_inherited(self):
        self.assertEqual(KAMEL_TRAINER.screening_epochs, 100)
        self.assertEqual(KAMEL_TRAINER.lr_schedule_max_epochs, 1000)
        self.assertEqual(KAMEL_TRAINER.train_foreground, 0.33)
        self.assertIs(KAMEL_TRAINER.raw_loaders, nnUNetTrainer_B0R2022.raw_loaders)
        self.assertNotIn("A1", " ".join(cls.__name__ for cls in KAMEL_TRAINER.__mro__))

    def test_audit_passed(self):
        audit = json.loads((ROOT / "outputs/analysis/kamel_replication/configuration_audit.json").read_text())
        self.assertTrue(audit["b0_configuration_matches_current_plans"])
        self.assertTrue(audit["split"]["every_case_validated_exactly_once"])
        self.assertFalse(audit["split"]["leakage_detected"])


if __name__ == "__main__":
    unittest.main()
