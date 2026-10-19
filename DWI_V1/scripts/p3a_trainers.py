"""P3a trainers using Dataset013 clipped DWI with otherwise baseline-equivalent settings."""

import json
import shutil
from pathlib import Path

from recall_common import save
from recall_trainers import nnUNetTrainer_B0R2022, trainer_module
from screen100_trainers import Screen100Mixin


ROOT = Path(__file__).resolve().parents[1]
P3_PP = ROOT / "datasets/ISLES2022/processed/Dataset013_ISLES2022_DWI_ClipZScore"


class nnUNetTrainer_P3a(nnUNetTrainer_B0R2022):
    """Baseline trainer redirected to Dataset013; foreground sampling remains 0.33."""

    train_foreground = 0.33

    def do_split(self):
        split = json.loads((P3_PP / "splits_final.json").read_text())[0]
        assert self.fold == 0 and not set(split["train"]) & set(split["val"])
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
        shutil.copyfile(P3_PP / "dataset_fingerprint.json", base / "dataset_fingerprint.json")
        self._save_debug_information()
        self.print_to_log_file("Dataset013 P3a read-only reuse: unpack explicitly disabled.")


class nnUNetTrainer_P3a_Screen100(Screen100Mixin, nnUNetTrainer_P3a):
    screening_epochs = 100


class nnUNetTrainer_P3a_Promote300(Screen100Mixin, nnUNetTrainer_P3a):
    screening_epochs = 300
