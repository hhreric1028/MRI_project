"""Isolated 100-epoch screening trainers with the original 1000-epoch PolyLR horizon."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from amp_resume_trainer import all_finite, amp_update, tensors_in
from recall_trainers import nnUNetTrainer_B0R2022, nnUNetTrainer_L2a, nnUNetTrainer_S1a


class Screen100Mixin:
    """Run 100 epochs while preserving the LR trajectory of epochs 1-100/1000."""

    screening_epochs = 100
    lr_schedule_max_epochs = 1000
    max_consecutive_overflows = 16

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.screening_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.lr_schedule_max_epochs,
            exponent=0.9,
        )
        return optimizer, scheduler

    def configure_screen_monitoring(self, folder: Path, mode: str) -> None:
        self.screen_folder = Path(folder)
        self.screen_mode = mode
        self.screen_started = time.monotonic()
        self.consecutive_overflows = 0
        self.total_skipped = 0
        self.total_nonfinite_gradient_events = 0
        self.last_amp_event = None
        self.overflow_log = self.screen_folder / "amp_events.jsonl"
        self.epoch_monitor = self.screen_folder / "epoch_metrics.jsonl"
        self.overflow_log.open("x").close()
        self.epoch_monitor.open("x").close()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.lr_scheduler.max_steps != self.lr_schedule_max_epochs:
            raise RuntimeError("Screening PolyLR denominator changed")
        expected = self.initial_lr * (1 - self.current_epoch / self.lr_schedule_max_epochs) ** 0.9
        actual = float(self.optimizer.param_groups[0]["lr"])
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
            raise RuntimeError(f"Screening LR mismatch: expected {expected}, got {actual}")
        self.screen_iteration = 0
        self.epoch_norms = []
        self.epoch_skipped = 0
        self.epoch_nonfinite = 0
        self.epoch_learning_rate = actual
        self.print_to_log_file(
            f"SCREENING epoch {self.current_epoch + 1}/{self.screening_epochs}; "
            f"PolyLR reference horizon={self.lr_schedule_max_epochs}; LR={actual:.12f}"
        )

    def train_step(self, batch):
        self.screen_iteration += 1
        event = {
            "epoch": self.current_epoch + 1,
            "nnunet_epoch_index": self.current_epoch,
            "iteration": self.screen_iteration,
            "cases": [str(k) for k in batch.get("keys", [])],
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        try:
            data = batch["data"].to(self.device, non_blocking=True)
            target = batch["target"]
            target = (
                [item.to(self.device, non_blocking=True) for item in target]
                if isinstance(target, list)
                else target.to(self.device, non_blocking=True)
            )
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type) if self.device.type == "cuda" else nullcontext():
                loss = self.loss(self.network(data), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            event["loss"] = float(loss.detach())
            event.update(amp_update(loss, self.network, self.optimizer, self.grad_scaler))
            self.last_amp_event = event
            self.epoch_nonfinite += int(not event["gradient_finite"])
            self.total_nonfinite_gradient_events += int(not event["gradient_finite"])
            if event["optimizer_step_skipped"]:
                self.total_skipped += 1
                self.epoch_skipped += 1
                self.consecutive_overflows += 1
            else:
                self.consecutive_overflows = 0
            event["consecutive_overflows"] = self.consecutive_overflows
            if event["gradient_norm_after_unscale"] is not None:
                self.epoch_norms.append(event["gradient_norm_after_unscale"])
            if not all_finite(self.network.parameters()) or not all_finite(self.network.buffers()):
                raise FloatingPointError("Nonfinite model state after step")
            if not all_finite(tensors_in(self.optimizer.state)):
                raise FloatingPointError("Nonfinite optimizer state after step")
            if event["optimizer_step_skipped"]:
                with self.overflow_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, allow_nan=False) + "\n")
                self.print_to_log_file("AMP step skipped; scale reduced:", event)
            if self.consecutive_overflows >= self.max_consecutive_overflows:
                raise FloatingPointError("Persistent nonfinite gradients: 16 consecutive AMP skips")
            return {"loss": loss.detach().cpu().numpy()}
        except BaseException as exc:
            event["failure"] = repr(exc)
            with self.overflow_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, allow_nan=False) + "\n")
            failure_batch = self.screen_folder / (
                f"failure_batch_epoch{self.current_epoch + 1}_iteration{self.screen_iteration}.pth"
            )
            with failure_batch.open("xb") as f:
                torch.save(batch, f)
            raise

    def on_epoch_end(self):
        super().on_epoch_end()
        completed = self.current_epoch
        elapsed = time.monotonic() - self.screen_started
        remaining = elapsed / max(completed, 1) * (self.screening_epochs - completed)
        row = {
            "epoch": completed,
            "train_loss": float(self.logger.get_value("train_losses", step=-1)),
            "validation_loss": float(self.logger.get_value("val_losses", step=-1)),
            "patch_pseudo_dice": float(self.logger.get_value("mean_fg_dice", step=-1)),
            "learning_rate": self.epoch_learning_rate,
            "grad_scaler_scale": float(self.grad_scaler.get_scale()) if self.grad_scaler else None,
            "skipped_optimizer_steps": self.epoch_skipped,
            "nonfinite_gradient_events": self.epoch_nonfinite,
            "mean_finite_unscaled_gradient_norm": float(np.mean(self.epoch_norms)) if self.epoch_norms else None,
            "max_finite_unscaled_gradient_norm": max(self.epoch_norms) if self.epoch_norms else None,
            "consecutive_overflows_at_epoch_end": self.consecutive_overflows,
        }
        with self.epoch_monitor.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        self.print_to_log_file(
            f"SCREENING progress {completed}/{self.screening_epochs}; ETA {remaining / 3600:.2f} hours; "
            f"epoch skipped steps={self.epoch_skipped}"
        )


class nnUNetTrainer_B0R2022_Screen100(Screen100Mixin, nnUNetTrainer_B0R2022):
    pass


class nnUNetTrainer_S1a_Screen100(Screen100Mixin, nnUNetTrainer_S1a):
    pass


class nnUNetTrainer_L2a_Screen100(Screen100Mixin, nnUNetTrainer_L2a):
    pass


SCREEN_TRAINERS = {
    "B0_R2022_screen100": nnUNetTrainer_B0R2022_Screen100,
    "S1a_screen100_seed2022": nnUNetTrainer_S1a_Screen100,
    "L2a_screen100_seed2022": nnUNetTrainer_L2a_Screen100,
}
