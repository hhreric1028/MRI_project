"""Dataset-specific trainers for the dual-channel enhancement batch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from image_enhancement_common import BASE_PP, EXPERIMENTS, ROOT
from recall_common import save
from recall_trainers import nnUNetTrainer_B0R2022, trainer_module
from screen100_trainers import Screen100Mixin


class EnhancementTrainerBase(nnUNetTrainer_B0R2022):
    train_foreground = 0.33
    experiment_name: str

    @property
    def enhancement_pp(self) -> Path:
        return ROOT / "datasets/ISLES2022/processed" / EXPERIMENTS[self.experiment_name]["dataset_name"]

    def do_split(self):
        split = json.loads((self.enhancement_pp / "splits_final.json").read_text())[0]
        baseline = json.loads((BASE_PP / "splits_final.json").read_text())[0]
        if split != baseline or self.fold != 0 or set(split["train"]) & set(split["val"]):
            raise RuntimeError("Enhancement split differs from the frozen fold-0 baseline")
        return split["train"], split["val"]

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.set_deep_supervision_enabled(True)
        self.print_plans()
        trainer_module.empty_cache(self.device)
        base = Path(self.output_folder_base)
        save(base / "plans.json", self.plans_manager.plans)
        save(base / "dataset.json", self.dataset_json)
        shutil.copyfile(self.enhancement_pp / "dataset_fingerprint.json", base / "dataset_fingerprint.json")
        self._save_debug_information()
        self.print_to_log_file(f"{self.experiment_name}: isolated preprocessed data; unpack explicitly disabled.")


class E1Base(EnhancementTrainerBase):
    experiment_name = "E1_gamma"


class E2Base(EnhancementTrainerBase):
    experiment_name = "E2_clahe"


class E3Base(EnhancementTrainerBase):
    experiment_name = "E3_intensity_scale"


class nnUNetTrainer_E1Gamma_Screen100(Screen100Mixin, E1Base):
    screening_epochs = 100


class nnUNetTrainer_E1Gamma_Promote300(Screen100Mixin, E1Base):
    screening_epochs = 300


class nnUNetTrainer_E2CLAHE_Screen100(Screen100Mixin, E2Base):
    screening_epochs = 100


class nnUNetTrainer_E2CLAHE_Promote300(Screen100Mixin, E2Base):
    screening_epochs = 300


class nnUNetTrainer_E3Scale_Screen100(Screen100Mixin, E3Base):
    screening_epochs = 100


class nnUNetTrainer_E3Scale_Promote300(Screen100Mixin, E3Base):
    screening_epochs = 300


TRAINERS = {
    "E1_gamma": {
        "screen100": nnUNetTrainer_E1Gamma_Screen100,
        "promote300": nnUNetTrainer_E1Gamma_Promote300,
    },
    "E2_clahe": {
        "screen100": nnUNetTrainer_E2CLAHE_Screen100,
        "promote300": nnUNetTrainer_E2CLAHE_Promote300,
    },
    "E3_intensity_scale": {
        "screen100": nnUNetTrainer_E3Scale_Screen100,
        "promote300": nnUNetTrainer_E3Scale_Promote300,
    },
}

