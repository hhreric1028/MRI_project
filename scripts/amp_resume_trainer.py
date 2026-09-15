"""AMP-aware continuation subclass; original experiment and installed classes unchanged."""
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from recall_trainers import nnUNetTrainer_B0R2022, trainer_module


def tensors_in(value):
    if isinstance(value,torch.Tensor):
        yield value
    elif isinstance(value,dict):
        for item in value.values(): yield from tensors_in(item)
    elif isinstance(value,(tuple,list)):
        for item in value: yield from tensors_in(item)


def all_finite(tensors):
    values=list(tensors)
    return not values or bool(torch.stack([torch.isfinite(t).all() for t in values]).all().item())


def amp_update(loss,network,optimizer,scaler):
    """One unscale; keep scaler's overflow detection and skip/update mechanism intact."""
    before=float(scaler.get_scale()) if scaler is not None else None
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    parameters=[p for p in network.parameters() if p.grad is not None]
    gradients=[p.grad for p in parameters]
    if not gradients:raise FloatingPointError('No gradients')
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(),2) for g in gradients]),2)
    finite_norm=bool(torch.isfinite(norm))
    finite_gradients=finite_norm or all_finite(gradients)
    if finite_gradients:
        if finite_norm:
            torch.nn.utils.clip_grad_norm_(parameters,12,error_if_nonfinite=True)
        else:
            # Distinguish finite entries with an overflowing FP32 reduction from bad gradients.
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().double(),2) for g in gradients]),2)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite double-precision gradient norm')
            coefficient=(12/(norm+1e-6)).clamp(max=1.)
            with torch.no_grad():
                for g in gradients:g.mul_(coefficient.to(g.dtype))
    elif scaler is None:
        raise FloatingPointError('Nonfinite unscaled gradients without enabled AMP')
    # Do not clip nonfinite gradients: unscale_ has already registered them with GradScaler.
    # Observe whether the optimizer actually ran, rather than infer skipping from its return value.
    steps=[]
    hook=optimizer.register_step_post_hook(lambda *args:steps.append(True))
    try:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    finally:
        hook.remove()
    skipped=not steps
    after=float(scaler.get_scale()) if scaler is not None else None
    if not finite_gradients and not skipped:raise FloatingPointError('Nonfinite gradients reached optimizer update')
    if skipped and (after is None or not math.isfinite(after) or not 0<after<before):
        raise FloatingPointError('Skipped AMP step failed to reduce a positive finite scale')
    return dict(scale_before=before,scale_after=after,optimizer_step_skipped=skipped,
        gradient_norm_after_unscale=float(norm) if torch.isfinite(norm) else None,
        gradient_norm_nonfinite=None if torch.isfinite(norm) else ('nan' if torch.isnan(norm) else 'inf'),
        gradient_finite=finite_gradients)


class nnUNetTrainer_B0R2022_Resume(nnUNetTrainer_B0R2022):
    """Same training configuration; permit transient AMP overflow, monitor continuation."""
    # Operational failure guard, not a model/optimization hyperparameter. Sixteen consecutive
    # skipped updates allow 2**16 scale backoff; any intervening successful update resets it.
    max_consecutive_overflows=16

    def configure_resume_monitoring(self,folder,start_epoch,mode):
        self.resume_folder=Path(folder)
        self.resume_start_epoch=start_epoch
        self.resume_mode=mode
        self.resume_started=time.monotonic()
        self.consecutive_overflows=0
        self.total_skipped=0
        self.last_amp_event=None
        self.overflow_log=self.resume_folder/'amp_events.jsonl'
        self.overflow_log.open('x').close()
        self.epoch_monitor=self.resume_folder/'early_epochs.jsonl'
        self.epoch_monitor.open('x').close()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        expected=self.initial_lr*(1-self.current_epoch/self.num_epochs)**self.lr_scheduler.exponent
        if any(not math.isclose(g['lr'],expected,rel_tol=1e-12,abs_tol=1e-14) for g in self.optimizer.param_groups):
            raise RuntimeError('LR continuity error')
        self.resume_iteration=0
        self.epoch_norms=[]
        self.epoch_skipped=0
        self.epoch_nonfinite=0
        self.print_to_log_file(f'Resume epoch {self.current_epoch+1}/1000 (nnUNet index {self.current_epoch})')

    def train_step(self,batch):
        self.resume_iteration+=1
        event=dict(epoch=self.current_epoch+1,nnunet_epoch_index=self.current_epoch,
                   iteration=self.resume_iteration,cases=[str(k) for k in batch.get('keys',[])],
                   learning_rate=float(self.optimizer.param_groups[0]['lr']))
        try:
            data=batch['data'].to(self.device,non_blocking=True)
            target=batch['target']
            target=[t.to(self.device,non_blocking=True) for t in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type) if self.device.type=='cuda' else nullcontext():
                loss=self.loss(self.network(data),target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            event['loss']=float(loss.detach())
            event.update(amp_update(loss,self.network,self.optimizer,self.grad_scaler))
            self.last_amp_event=event
            self.epoch_nonfinite+=int(not event['gradient_finite'])
            if event['optimizer_step_skipped']:
                self.total_skipped+=1;self.epoch_skipped+=1;self.consecutive_overflows+=1
            else:
                self.consecutive_overflows=0
            event['consecutive_overflows']=self.consecutive_overflows
            if event['gradient_norm_after_unscale'] is not None:self.epoch_norms.append(event['gradient_norm_after_unscale'])
            if not all_finite(self.network.parameters()) or not all_finite(self.network.buffers()):
                raise FloatingPointError('Nonfinite model state after step')
            if not all_finite(tensors_in(self.optimizer.state)):
                raise FloatingPointError('Nonfinite optimizer state after step')
            if event['optimizer_step_skipped']:
                with self.overflow_log.open('a',encoding='utf-8') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
                self.print_to_log_file('AMP step skipped; scale reduced:',event)
            if self.consecutive_overflows>=self.max_consecutive_overflows:
                raise FloatingPointError('Persistent nonfinite gradients: 16 consecutive AMP skips without recovery')
            return {'loss':loss.detach().cpu().numpy()}
        except BaseException as exc:
            event.update(failure=repr(exc))
            with self.overflow_log.open('a',encoding='utf-8') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
            # Preserve the exact failing augmented batch when available; never overwrite earlier evidence.
            path=self.resume_folder/f'failure_batch_epoch{self.current_epoch+1}_iteration{self.resume_iteration}.pth'
            with path.open('xb') as f:torch.save(batch,f)
            raise

    def on_epoch_end(self):
        # Use official epoch accounting; inherited ETA divides by historical epochs and is unsuitable.
        index=self.current_epoch
        trainer_module.nnUNetTrainer.on_epoch_end(self)
        elapsed=time.monotonic()-self.resume_started
        completed=self.current_epoch-self.resume_start_epoch
        remaining=elapsed/max(completed,1)*(self.num_epochs-self.current_epoch)
        self.print_to_log_file(f'Progress {self.current_epoch}/1000; resumed epochs {completed}; ETA {remaining/3600:.2f} hours')
        if index<self.resume_start_epoch+3:
            r=dict(epoch=index+1,train_loss=float(self.logger.get_value('train_losses',step=-1)),
                validation_loss=float(self.logger.get_value('val_losses',step=-1)),
                pseudo_dice=float(self.logger.get_value('mean_fg_dice',step=-1)),
                learning_rate=float(self.optimizer.param_groups[0]['lr']),grad_scaler_scale=float(self.grad_scaler.get_scale()),
                skipped_optimizer_steps=self.epoch_skipped,nonfinite_gradient_events=self.epoch_nonfinite,
                mean_finite_unscaled_gradient_norm=float(np.mean(self.epoch_norms)) if self.epoch_norms else None,
                max_finite_unscaled_gradient_norm=max(self.epoch_norms) if self.epoch_norms else None,
                nonfinite_loss_parameter_optimizer_events=0)
            with self.epoch_monitor.open('a',encoding='utf-8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
            self.print_to_log_file('Early continuation monitor:',r)
