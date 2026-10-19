"""Original B0-100 trainer extended only to select fixed folds 0-4."""

from __future__ import annotations

import json

from recall_common import PP
from recall_trainers import nnUNetTrainer_B0R2022
from screen100_trainers import Screen100Mixin


class nnUNetTrainer_B0Full5fold_Screen100(Screen100Mixin, nnUNetTrainer_B0R2022):
    """B0-100 with the original settings and the requested fixed fold selector."""

    def do_split(self):
        splits = json.loads((PP / "splits_final.json").read_text(encoding="utf-8"))
        if not isinstance(self.fold, int) or not 0 <= self.fold < len(splits):
            raise ValueError(f"B0 full five-fold run requires fold 0-{len(splits) - 1}; got {self.fold}")
        split = splits[self.fold]
        if set(split["train"]) & set(split["val"]):
            raise RuntimeError(f"Fold {self.fold} train/validation leakage")
        if len(split["train"]) != 200 or len(split["val"]) != 50:
            raise RuntimeError(f"Unexpected fold {self.fold} sizes")
        return split["train"], split["val"]


B0_FULL5FOLD_TRAINER = nnUNetTrainer_B0Full5fold_Screen100
