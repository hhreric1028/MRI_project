"""Fresh-process GPU feasibility/full training, exclusive outputs, immutable baseline."""
import argparse
import json
import os
import random
import time
import traceback
import numpy as np
import torch
from recall_trainers import TRAINERS, DS_WEIGHTS
from recall_common import ROOT, PP, SUITE, save, digest, provenance, protect_baseline, verify_baseline, now

def seed_all():
    random.seed(2022);np.random.seed(2022);torch.manual_seed(2022);torch.cuda.manual_seed_all(2022)

def make_trainer(experiment,device='cuda'):
    plans=json.loads((PP/'nnUNetPlans.json').read_text());cfg=plans['configurations']['3d_fullres']
    assert cfg['patch_size']==[80,96,80] and cfg['batch_size']==8 and cfg['spacing']==[2.,2.,2.]
    plans['continue_training']=False
    trainer=TRAINERS[experiment](plans,'3d_fullres',0,json.loads((PP/'dataset.json').read_text()),device=torch.device(device))
    assert trainer.num_epochs==1000 and trainer.num_iterations_per_epoch==250 and trainer.num_val_iterations_per_epoch==50
    trainer.execution_started=time.monotonic()
    return trainer

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=TRAINERS,required=True);p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    run=SUITE/a.experiment/a.mode
    run.mkdir(parents=True,exist_ok=False)
    protect_baseline()
    trainer=None
    try:
        verify_baseline(run/'baseline_before.json')
        os.environ['nnUNet_results']=str(run/'results')
        seed_all();torch.set_num_threads(8)
        meta=provenance()
        if a.mode=='train':
            smoke=run.parent/'smoke'
            status=json.loads((smoke/'completion.json').read_text());prior=json.loads((smoke/'experiment_config.json').read_text())
            assert status['status']=='passed' and meta['source_sha256']==prior['source_sha256']
            assert meta['plans_sha256']==prior['plans_sha256'] and meta['packages']==prior['packages']
        trainer=make_trainer(a.experiment)
        meta.update(experiment=a.experiment,mode=a.mode,comparison='controlled same-seed comparison',
            seed_python=2022,seed_numpy=2022,seed_torch=2022,seed_cuda=2022,
            worker_seeds_train=[2022,2023,2024,2025],worker_seeds_val=[3022,3023],
            worker_order='NonDetMultiThreadedAugmenter; arrival order not deterministic',
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            train_foreground=trainer.train_foreground,validation_foreground=.33,
            configuration=trainer.configuration_manager.configuration,epochs=1000,pretrained_weights=None,
            early_stopping=False,evaluation_checkpoint='checkpoint_final.pth',fresh_initialization=True,
            deep_supervision_weights=DS_WEIGHTS,trainer_class=trainer.__class__.__name__,
            loss_definition='mean(1-TI)+CE; foreground per-sample; FP=.3 FN=.7 smooth=1e-5 float32' if a.experiment=='L2a' else 'DC_and_CE_loss: MemoryEfficientSoftDiceLoss (negative Dice) + RobustCrossEntropyLoss; weights1/1; per-sample; no background Dice; smooth1e-5; ignore=None',
            baseline_preprocessing='read-only Blosc2; unpack disabled; Python write audit guard',
            gradient_policy='unscale, finite norm assertion, clip12; fail-stop, no automatic hyperparameter adjustment')
        save(run/'experiment_config.json',meta)
        if a.mode=='smoke':
            trainer.on_train_start();trainer.on_train_epoch_start()
            norms=[];losses=[]
            for i in range(120):
                if i==20:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                result=trainer.train_step(next(trainer.dataloader_train));norms.append(trainer.last_gradient_norm);losses.append(float(result['loss']))
                if (i+1)%10==0:print(f'{a.experiment} feasibility {i+1}/120 | loss={losses[-1]:.5f} | grad={norms[-1]:.3f}',flush=True)
            torch.cuda.synchronize();seconds=time.monotonic()-started
            trainer.network.eval()
            with torch.no_grad(): val=trainer.validation_step(next(trainer.dataloader_val))
            result=dict(status='passed',warmup=20,measured=100,seconds=seconds,seconds_per_iteration=seconds/100,
                peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gb=torch.cuda.max_memory_reserved()/1024**3,
                losses=losses,unscaled_gradient_norms=norms,validation_loss=float(val['loss']))
            trainer.close_loaders()
        else:
            trainer.run_training()
            trainer.perform_actual_validation(save_probabilities=True)
            checkpoint=__import__('pathlib').Path(trainer.output_folder)/'checkpoint_final.pth'
            result=dict(status='training_and_validation_complete',epochs=trainer.current_epoch,
                checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),predictions=str(checkpoint.parent/'validation'))
        verify_baseline(run/'baseline_after.json')
        result['finished_at']=now();save(run/'completion.json',result)
    except BaseException as e:
        save(run/'failure.json',dict(time=now(),error=repr(e),traceback=traceback.format_exc()))
        if trainer:trainer.close_loaders()
        if not (run/'baseline_after.json').exists():verify_baseline(run/'baseline_after.json')
        raise

if __name__=='__main__':main()
