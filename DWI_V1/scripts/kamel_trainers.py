"""Unmodified-B0 trainer for fixed five-fold Kamel-style replication."""

from __future__ import annotations

import json

from recall_common import PP
from recall_trainers import nnUNetTrainer_B0R2022
from screen100_trainers import Screen100Mixin


class nnUNetTrainer_KamelRep_Screen100(Screen100Mixin, nnUNetTrainer_B0R2022):
    """B0-100 for any of the five existing fixed folds; no augmentation override."""

    def do_split(self):
        splits = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))
        if not isinstance(self.fold, int) or not 0 <= self.fold < len(splits):
            raise ValueError(f"Kamel replication requires fixed fold 0-{len(splits) - 1}; got {self.fold}")
        split = splits[self.fold]
        if set(split["train"]) & set(split["val"]):
            raise RuntimeError(f"Fold {self.fold} train/validation leakage")
        if len(split["train"]) != 200 or len(split["val"]) != 50:
            raise RuntimeError(f"Unexpected fold {self.fold} sizes")
        return split["train"], split["val"]


KAMEL_TRAINER = nnUNetTrainer_KamelRep_Screen100
