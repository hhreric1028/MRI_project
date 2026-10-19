"""V2-E1 and V2-S1 trainers with all uncontrolled settings inherited from B0-R2022."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

V2_SCRIPTS = Path(__file__).resolve().parents[1]
V1_SCRIPTS = V2_SCRIPTS.parent.parent / "DWI_V1" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))
sys.path.insert(0, str(V1_SCRIPTS))

from recall_common import save  # noqa: E402
from recall_trainers import nnUNetTrainer_B0R2022, trainer_module  # noqa: E402
from screen100_trainers import Screen100Mixin  # noqa: E402
from v2_common import E1_PP, S1_EXPERIMENT, S1_HARD, V1_PP  # noqa: E402


class DatasetWithHardLocations:
    def __init__(self, base_dataset, identifiers: list[str]):
        self.base_dataset = base_dataset
        self.identifiers = list(identifiers)
        self.hard_locations = {}
        for case in self.identifiers:
            path = S1_HARD / f"{case}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing training-only hard locations: {path}")
            self.hard_locations[case] = np.load(path)["locations"].astype(np.int32)

    def load_case(self, identifier):
        data, seg, seg_prev, properties = self.base_dataset.load_case(identifier)
        properties = dict(properties)
        properties["class_locations"] = dict(properties["class_locations"])
        properties["class_locations"][2] = self.hard_locations[identifier]
        return data, seg, seg_prev, properties


class HardRegionDataLoader(trainer_module.nnUNetDataLoader):
    def __init__(self, *args, hard_preference: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < hard_preference <= 1:
            raise ValueError(hard_preference)
        self.hard_preference = float(hard_preference)

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        if force_fg:
            if 1 not in class_locations or 2 not in class_locations:
                raise RuntimeError("Hard-region loader requires foreground class 1 and hard pseudo-class 2")
            overwrite_class = 2 if np.random.uniform() < self.hard_preference else 1
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)


class nnUNetTrainer_V2E1(nnUNetTrainer_B0R2022):
    """Only controlled change: Dataset101 contains original + exact P3a representation."""

    train_foreground = 0.33

    def do_split(self):
        split = json.loads((E1_PP / "splits_final.json").read_text(encoding="utf-8"))[0]
        assert self.fold == 0 and len(split["train"]) == 200 and len(split["val"]) == 50
        assert not set(split["train"]) & set(split["val"])
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
        shutil.copyfile(E1_PP / "dataset_fingerprint.json", base / "dataset_fingerprint.json")
        self._save_debug_information()
        self.print_to_log_file("DWI_V2 Dataset101 isolated reuse: unpack explicitly disabled.")


class nnUNetTrainer_V2S1(nnUNetTrainer_B0R2022):
    """Baseline data/settings with training-only hard preference inside forced-FG crops."""

    train_foreground = 0.33

    def _hard_preference(self) -> float:
        completion = json.loads((S1_EXPERIMENT / "sampling_audit/completion.json").read_text(encoding="utf-8"))
        if not completion.get("training_allowed") or completion.get("status") != "AUDIT PASSED":
            raise RuntimeError("V2-S1 sampling audit did not authorize training")
        return float(completion["selected_sampling"]["hard_preference_within_forced_foreground"])

    def raw_loaders(self):
        if not hasattr(self, "batch_size"):
            self._set_batch_size_and_oversample()
        if self.dataset_class is None:
            self.dataset_class = trainer_module.infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        ds = self._get_deep_supervision_scales()
        rotation, dummy, initial, mirrors = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        assert not dummy and len(ds) == 4 and self.label_manager.ignore_label is None
        common = dict(
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=None,
        )
        train_transform = self.get_training_transforms(
            patch,
            rotation,
            ds,
            mirrors,
            dummy,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            **common,
        )
        val_transform = self.get_validation_transforms(ds, **common)
        train_dataset, val_dataset = self.get_tr_and_val_datasets()
        train_dataset = DatasetWithHardLocations(train_dataset, train_dataset.identifiers)
        train_loader = HardRegionDataLoader(
            train_dataset,
            self.batch_size,
            initial,
            patch,
            self.label_manager,
            oversample_foreground_percent=0.33,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=train_transform,
            probabilistic_oversampling=self.probabilistic_oversampling,
            hard_preference=self._hard_preference(),
        )
        val_loader = trainer_module.nnUNetDataLoader(
            val_dataset,
            self.batch_size,
            patch,
            patch,
            self.label_manager,
            oversample_foreground_percent=0.33,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transform,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        return train_loader, val_loader

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"V2-S1 training-only hard-region preference={self._hard_preference():.6f}; "
            "overall foreground framework remains 0.33."
        )


class nnUNetTrainer_V2E1_Screen100(Screen100Mixin, nnUNetTrainer_V2E1):
    screening_epochs = 100


class nnUNetTrainer_V2E1_Promote300(Screen100Mixin, nnUNetTrainer_V2E1):
    screening_epochs = 300


class nnUNetTrainer_V2S1_Screen100(Screen100Mixin, nnUNetTrainer_V2S1):
    screening_epochs = 100


class nnUNetTrainer_V2S1_Promote300(Screen100Mixin, nnUNetTrainer_V2S1):
    screening_epochs = 300

