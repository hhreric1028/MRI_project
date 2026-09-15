"""Restore the original epoch-34 checkpoint into exclusive smoke/train directories."""
import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
import torch
from amp_resume_trainer import nnUNetTrainer_B0R2022_Resume, all_finite, tensors_in
from recall_common import ROOT,PP,PROTECTED,save,digest,provenance,protect_baseline,verify_baseline,now
from run_recall_experiment import seed_all

DEST=ROOT/'experiments/B0-R2022_resume_from_epoch34'
ORIGINAL=ROOT/'experiments/recall_2mm_seed2022/B0-R2022/train'
SOURCE=ORIGINAL/'results/Dataset001_ISLES2022_DWI/nnUNetTrainer_B0R2022__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
EXPECTED_SHA='f67ae113f4a68bad8e4ed6da31191f3497d7588bfa2c7a1c291e75dff1f865bd'
LIMITATION='This run is a training-state continuation, not a bitwise deterministic continuation.'


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def checkpoint_schema(c):
    required=['network_weights','optimizer_state','grad_scaler_state','current_epoch','logging','_best_ema','init_args','trainer_name','inference_allowed_mirroring_axes']
    if any(k not in c for k in required):raise RuntimeError('Missing checkpoint state: '+repr([k for k in required if k not in c]))
    if c['current_epoch']!=34 or len(c['logging']['train_losses'])!=34 or len(c['logging']['val_losses'])!=34:
        raise RuntimeError('Checkpoint epoch/history mismatch')
    if c['trainer_name']!='nnUNetTrainer_B0R2022':raise RuntimeError('Unexpected source trainer')
    if not all_finite(tensors_in(c['network_weights'])) or not all_finite(tensors_in(c['optimizer_state'])):
        raise FloatingPointError('Source checkpoint contains nonfinite state')
    return dict(keys=list(c),model=dict(present=True,restorable=True,key='network_weights',tensors=len(list(tensors_in(c['network_weights'])))),
        optimizer=dict(present=True,restorable=True,key='optimizer_state',tensors=len(list(tensors_in(c['optimizer_state'])))),
        grad_scaler=dict(present=True,restorable=True,state=c['grad_scaler_state']),epoch=dict(present=True,restorable=True,value=34,first_resumed_epoch=35),
        lr_scheduler=dict(present=False,restorable=True,method='Reconstruct PolyLR(initial_lr=.01,max_steps=1000,exponent=.9), set explicit current_step=34; validate saved LR at step33'),
        trainer_state=dict(present=True,restorable=True,keys=['logging','_best_ema','init_args','inference_allowed_mirroring_axes']),
        rng=dict(present=False,restorable=False),dataloader_workers=dict(present=False,restorable=False),limitation=LIMITATION)


def restore(c,run):
    plans=json.loads((PP/'nnUNetPlans.json').read_text());plans['continue_training']=False
    trainer=nnUNetTrainer_B0R2022_Resume(plans,'3d_fullres',0,json.loads((PP/'dataset.json').read_text()),device=torch.device('cuda'))
    trainer.initialize()
    assert trainer.num_epochs==1000 and trainer.initial_lr==.01
    assert trainer.lr_scheduler.__class__.__name__=='PolyLRScheduler' and trainer.lr_scheduler.exponent==.9 and trainer.lr_scheduler.max_steps==1000
    trainer.network.load_state_dict(c['network_weights'],strict=True)
    trainer.optimizer.load_state_dict(c['optimizer_state'])
    trainer.grad_scaler.load_state_dict(c['grad_scaler_state'])
    trainer.current_epoch=c['current_epoch']
    trainer.my_init_kwargs=copy.deepcopy(c['init_args'])
    trainer.logger.load_checkpoint(copy.deepcopy(c['logging']))
    trainer._best_ema=c['_best_ema']
    trainer.inference_allowed_mirroring_axes=c['inference_allowed_mirroring_axes']
    # The optimizer/scaler are now restored objects, not fresh optimization state.
    checks=dict(model_exact=equal_state(c['network_weights'],trainer.network.state_dict()),
        optimizer_exact=equal_state(c['optimizer_state'],trainer.optimizer.state_dict()),
        scaler_exact=equal_state(c['grad_scaler_state'],trainer.grad_scaler.state_dict()),
        optimizer_tensors_on_device=all(t.device==trainer.device for t in tensors_in(trainer.optimizer.state)),
        model_finite=all_finite(trainer.network.parameters()),optimizer_finite=all_finite(tensors_in(trainer.optimizer.state)),
        restored_epoch=trainer.current_epoch,restored_history_length=len(trainer.logger.get_checkpoint()['train_losses']))
    if not all(checks[k] for k in ['model_exact','optimizer_exact','scaler_exact','optimizer_tensors_on_device','model_finite','optimizer_finite']):
        raise RuntimeError('Checkpoint restore mismatch: '+repr(checks))
    before=float(trainer.optimizer.param_groups[0]['lr'])
    expected_before=.01*(1-33/1000)**.9
    expected=.01*(1-34/1000)**.9
    if not math.isclose(before,expected_before,rel_tol=1e-12):raise RuntimeError('Saved LR does not match original epoch33 schedule')
    trainer.lr_scheduler.step(trainer.current_epoch)
    actual=float(trainer.optimizer.param_groups[0]['lr'])
    if not math.isclose(actual,expected,rel_tol=1e-12):raise RuntimeError('Resume LR continuity error')
    checks.update(lr_before_interruption=before,expected_lr_at_resume=expected,actual_lr_after_resume=actual,
        scheduler_state_reconstructed=trainer.lr_scheduler.state_dict(),grad_scaler=trainer.grad_scaler.state_dict())
    save(run/'restoration.json',checks)
    print(f'Loaded checkpoint: {SOURCE}\nCheckpoint epoch: 34\nResuming from epoch: 35\nInternal nnUNet epoch index: 34',flush=True)
    print(f'LR before interruption: {before:.12f}\nExpected LR at resume: {expected:.12f}\nActual LR after resume: {actual:.12f}',flush=True)
    print(LIMITATION,flush=True)
    return trainer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','train'],required=True);args=parser.parse_args()
    run=DEST/args.mode
    run.mkdir(parents=True,exist_ok=False)
    PROTECTED.append(ORIGINAL);protect_baseline()
    trainer=None;original_manifest=None
    try:
        if digest(SOURCE)!=EXPECTED_SHA:raise RuntimeError('Original checkpoint hash mismatch')
        original_manifest={str(p):digest(p) for p in ORIGINAL.rglob('*') if p.is_file()}
        save(run/'original_run_manifest.json',original_manifest)
        verify_baseline(run/'baseline_before.json')
        os.environ['nnUNet_results']=str(run/'results')
        seed_all();torch.set_num_threads(8)
        meta=provenance()
        for p in (ROOT/'scripts').glob('*amp*resume*'):
            if p.is_file():meta['source_sha256'][str(p)]=digest(p)
        old=json.loads((ORIGINAL/'experiment_config.json').read_text())
        differences={p:dict(original=h,current=meta['source_sha256'].get(p)) for p,h in old['source_sha256'].items() if meta['source_sha256'].get(p)!=h}
        meta['historical_source_differences']=differences
        if differences:raise RuntimeError('Original trainer/dependency sources changed')
        if meta['packages']!=old['packages']:raise RuntimeError('Environment packages differ from source run')
        if meta['plans_sha256']!=old['plans_sha256'] or meta['split_sha256']!=old['split_sha256']:raise RuntimeError('Plans/splits changed')
        if args.mode=='train':
            passed=json.loads((DEST/'smoke/completion.json').read_text())
            prior=json.loads((DEST/'smoke/resume_config.json').read_text())
            if passed['status']!='passed' or prior['source_sha256']!=meta['source_sha256'] or prior['packages']!=meta['packages']:
                raise RuntimeError('Resume smoke gate not satisfied')
            if prior['original_checkpoint_sha256']!=EXPECTED_SHA:raise RuntimeError('Smoke used a different source checkpoint')
        c=torch.load(SOURCE,map_location='cpu',weights_only=False)
        schema=checkpoint_schema(c);save(run/'checkpoint_schema.json',schema)
        trainer=restore(c,run)
        trainer.configure_resume_monitoring(run,34,args.mode)
        meta.update(mode=args.mode,original_checkpoint=str(SOURCE),original_checkpoint_sha256=EXPECTED_SHA,
            original_run_commit=old['git_commit'],source_epoch=34,first_resumed_epoch=35,internal_resume_epoch=34,
            resume_trainer=trainer.__class__.__name__,trainer_sha256=digest(ROOT/'scripts/amp_resume_trainer.py'),
            resume_start_time=now(),configuration=trainer.configuration_manager.configuration,
            seeds=dict(python=2022,numpy=2022,torch=2022,cuda=2022,train_workers=[2022,2023,2024,2025],validation_workers=[3022,3023]),
            worker_sequence='New worker streams at epoch35 under existing seed policy; historical iterator position unavailable',
            limitation=LIMITATION,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            restored_grad_scaler=trainer.grad_scaler.state_dict(),consecutive_overflow_stop_threshold=16,
            overflow_policy='One unscale; monitor unscaled gradients; skip clipping nonfinite entries; scaler.step then update; verify actual optimizer skip and scale decrease',
            formal_source='Original checkpoint freshly loaded, not smoke weights',fixed_hyperparameters=old['configuration'],
            gpu=torch.cuda.get_device_name(),evaluation_checkpoint='checkpoint_final.pth')
        save(run/'resume_config.json',meta)
        if args.mode=='smoke':
            trainer.on_train_start();trainer.on_train_epoch_start()
            events=[]
            for i in range(120):
                if i==20:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                trainer.train_step(next(trainer.dataloader_train));events.append(dict(trainer.last_amp_event))
                if (i+1)%10==0:print(f'Resume verification {i+1}/120 | loss={events[-1]["loss"]:.6f} | scale={events[-1]["scale_after"]} | skipped={trainer.total_skipped}',flush=True)
            torch.cuda.synchronize();seconds=time.monotonic()-started
            trainer.network.eval()
            with torch.no_grad():val=trainer.validation_step(next(trainer.dataloader_val))
            if trainer.consecutive_overflows:raise RuntimeError('Smoke ends in unrecovered overflow; formal continuation gated')
            save(run/'verification_iterations.json',events)
            result=dict(status='passed',warmup=20,measured=100,seconds=seconds,seconds_per_iteration=seconds/100,
                skipped_steps=trainer.total_skipped,finite_gradient_steps=sum(e['gradient_finite'] for e in events),
                recovered_at_end=trainer.consecutive_overflows==0,model_finite=all_finite(trainer.network.parameters()),
                optimizer_finite=all_finite(tensors_in(trainer.optimizer.state)),final_grad_scaler=trainer.grad_scaler.state_dict(),
                validation_loss=float(val['loss']),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            trainer.close_loaders()
        else:
            trainer.run_training()
            checkpoint=Path(trainer.output_folder)/'checkpoint_final.pth'
            save(run/'training_completed.json',dict(time=now(),epochs=trainer.current_epoch,checkpoint=str(checkpoint),sha256=digest(checkpoint)))
            trainer.perform_actual_validation(save_probabilities=True)
            result=dict(status='training_and_validation_complete',epochs=trainer.current_epoch,checkpoint=str(checkpoint),
                        checkpoint_sha256=digest(checkpoint),predictions=str(checkpoint.parent/'validation'))
        verify_baseline(run/'baseline_after.json')
        unchanged=all(Path(p).is_file() and digest(p)==h for p,h in original_manifest.items())
        current_files={str(p) for p in ORIGINAL.rglob('*') if p.is_file()}
        unchanged=unchanged and current_files==set(original_manifest)
        save(run/'source_preservation.json',dict(original_checkpoint_sha256=digest(SOURCE),original_run_unchanged=unchanged))
        if not unchanged:raise RuntimeError('Original run changed during resume')
        result.update(time=now(),original_checkpoint_sha256=EXPECTED_SHA)
        save(run/'completion.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(run/'failure.json',dict(time=now(),error=repr(e),traceback=traceback.format_exc()))
        if trainer:trainer.close_loaders()
        if not (run/'baseline_after.json').exists():verify_baseline(run/'baseline_after.json')
        if not (run/'source_preservation.json').exists():
            save(run/'source_preservation.json',dict(original_checkpoint_sha256=digest(SOURCE),matches_expected=digest(SOURCE)==EXPECTED_SHA))
        raise

if __name__=='__main__':main()
