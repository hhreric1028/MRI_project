"""Isolated binary lesion trainers. No installed nnUNet files are modified."""
import json
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_spacing_experiment import nnUNetTrainer_SeededPreserve, trainer_module
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from recall_common import PP, save

DS_WEIGHTS = (4/7, 2/7, 1/7, 0.)

class ForegroundTverskyCE(nn.Module):
    """Per-sample foreground TI, float32, alpha=.3 FP / beta=.7 FN, unit CE."""
    def tversky(self, probability, target):
        p=probability.float(); g=target.float()
        axes=tuple(range(1,p.ndim))
        tp=(p*g).sum(axes); fp=(p*(1-g)).sum(axes); fn=((1-p)*g).sum(axes)
        return (1-(tp+1e-5)/(tp+.3*fp+.7*fn+1e-5)).mean()

    def forward(self, logits, target):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            x=logits.float(); y=target[:,0].long()
            if x.shape[1]!=2 or torch.any((y<0)|(y>1)):
                raise ValueError('L2a requires binary labels 0/1 and no ignore label')
            return self.tversky(torch.softmax(x,dim=1)[:,1],y)+F.cross_entropy(x,y)

class nnUNetTrainer_B0R2022(nnUNetTrainer_SeededPreserve):
    train_foreground=.33

    def do_split(self):
        split=json.loads((PP/'splits_final.json').read_text())[0]
        assert self.fold==0 and not set(split['train']) & set(split['val'])
        return split['train'],split['val']

    def raw_loaders(self):
        if not hasattr(self,'batch_size'): self._set_batch_size_and_oversample()
        if self.dataset_class is None:
            self.dataset_class=trainer_module.infer_dataset_class(self.preprocessed_dataset_folder)
        patch=self.configuration_manager.patch_size
        ds=self._get_deep_supervision_scales()
        rotation,dummy,initial,mirrors=self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        assert not dummy and len(ds)==4 and self.label_manager.ignore_label is None
        common=dict(is_cascaded=self.is_cascaded,foreground_labels=self.label_manager.foreground_labels,
                    regions=None,ignore_label=None)
        tr=self.get_training_transforms(patch,rotation,ds,mirrors,dummy,
                                       use_mask_for_norm=self.configuration_manager.use_mask_for_norm,**common)
        va=self.get_validation_transforms(ds,**common)
        datasets=self.get_tr_and_val_datasets()
        loaders=[]
        for dataset,size,transform,fg in zip(datasets,(initial,patch),(tr,va),(self.train_foreground,.33)):
            loaders.append(trainer_module.nnUNetDataLoader(dataset,self.batch_size,size,patch,self.label_manager,
                oversample_foreground_percent=fg,sampling_probabilities=None,pad_sides=None,
                transforms=transform,probabilistic_oversampling=self.probabilistic_oversampling))
        return tuple(loaders)

    def get_dataloaders(self):
        tr,va=self.raw_loaders()
        a=trainer_module.NonDetMultiThreadedAugmenter(tr,None,num_processes=4,num_cached=6,
            seeds=[2022,2023,2024,2025],pin_memory=self.device.type=='cuda',wait_time=.002)
        b=trainer_module.NonDetMultiThreadedAugmenter(va,None,num_processes=2,num_cached=3,
            seeds=[3022,3023],pin_memory=self.device.type=='cuda',wait_time=.002)
        next(a); next(b)
        return a,b

    def on_train_start(self):
        # Deliberately does NOT call official on_train_start (which calls unpack_dataset).
        if not self.was_initialized: self.initialize()
        self.dataloader_train,self.dataloader_val=self.get_dataloaders()
        Path(self.output_folder).mkdir(parents=True,exist_ok=True)
        self.set_deep_supervision_enabled(True)
        self.print_plans(); trainer_module.empty_cache(self.device)
        base=Path(self.output_folder_base)
        save(base/'plans.json',self.plans_manager.plans)
        save(base/'dataset.json',self.dataset_json)
        shutil.copyfile(PP/'dataset_fingerprint.json',base/'dataset_fingerprint.json')
        self._save_debug_information()
        self.print_to_log_file('Dataset001 read-only reuse: unpack explicitly disabled.')

    def train_step(self,batch):
        data=batch['data'].to(self.device,non_blocking=True)
        target=batch['target']
        target=[t.to(self.device,non_blocking=True) for t in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type) if self.device.type=='cuda' else nullcontext():
            loss=self.loss(self.network(data),target)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward(); self.grad_scaler.unscale_(self.optimizer)
            norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),12,error_if_nonfinite=True)
            self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
        else:
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),12,error_if_nonfinite=True)
            self.optimizer.step()
        self.last_gradient_norm=float(norm)
        return {'loss':loss.detach().cpu().numpy()}

class nnUNetTrainer_S1a(nnUNetTrainer_B0R2022):
    train_foreground=.50

class nnUNetTrainer_L2a(nnUNetTrainer_B0R2022):
    def _build_loss(self):
        assert self.enable_deep_supervision and len(self._get_deep_supervision_scales())==4
        assert not self.label_manager.has_regions and self.label_manager.ignore_label is None
        return DeepSupervisionWrapper(ForegroundTverskyCE(),DS_WEIGHTS)

TRAINERS={'B0-R2022':nnUNetTrainer_B0R2022,'S1a':nnUNetTrainer_S1a,'L2a':nnUNetTrainer_L2a}
