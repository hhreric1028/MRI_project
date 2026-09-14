"""Isolated, seeded official trainer with visible feasibility benchmark or full run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for key,rel in [('nnUNet_raw','datasets/ISLES2022/nnUNet_raw'),('nnUNet_preprocessed','datasets/ISLES2022/processed')]: os.environ[key]=str(ROOT/rel)
os.environ['nnUNet_n_proc_DA']='4'
os.environ['nnUNet_compile']='false'
os.environ['nnUNet_results']=str(ROOT/'experiments/unused_results_guard')
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_SeededPreserve(nnUNetTrainer):
    """Same optimization/architecture; explicit seeds, finite checks, no checkpoint deletion."""
    def get_dataloaders(self):
        factory=trainer_module.NonDetMultiThreadedAugmenter
        def seeded(*args,**kwargs):
            n=kwargs['num_processes']
            kwargs['seeds']=[2022+i for i in range(n)] if n==4 else [3022+i for i in range(n)]
            return factory(*args,**kwargs)
        with patch.object(trainer_module,'NonDetMultiThreadedAugmenter',seeded):
            return super().get_dataloaders()
    def train_step(self,batch):
        result=super().train_step(batch)
        if not np.isfinite(result['loss']).all(): raise RuntimeError('Nonfinite training loss')
        return result
    def validation_step(self,batch):
        result=super().validation_step(batch)
        if not np.isfinite(result['loss']).all(): raise RuntimeError('Nonfinite validation loss')
        return result
    def on_epoch_end(self):
        super().on_epoch_end()
        elapsed=time.monotonic()-self.execution_started
        remaining=elapsed/max(self.current_epoch,1)*(self.num_epochs-self.current_epoch)
        print(f'Progress {self.current_epoch}/{self.num_epochs}; ETA {remaining/3600:.2f} hours',flush=True)
    def on_train_end(self):
        self.current_epoch-=1
        self.save_checkpoint(str(Path(self.output_folder)/'checkpoint_final.pth'))
        self.current_epoch+=1
        self.close_loaders()
        self.print_to_log_file('Training done. All checkpoints retained.')
    def close_loaders(self):
        for loader in [self.dataloader_train,self.dataloader_val]:
            if hasattr(loader,'_finish'): loader._finish()

def save(path,data):
    with path.open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',type=int,choices=[11,12],required=True)
    ap.add_argument('--mode',choices=['smoke','train'],required=True);args=ap.parse_args()
    name=f'Dataset{args.id:03d}_ISLES2022_DWI'
    exp='P1_spacing_1p5mm' if args.id==11 else 'P2_spacing_1mm'
    run=ROOT/'experiments'/exp/('run_seed2022_'+args.mode)
    if run.exists(): raise SystemExit('Existing experiment output detected. No files were modified.')
    pp=Path(os.environ['nnUNet_preprocessed'])/name
    baseline=Path(os.environ['nnUNet_preprocessed'])/'Dataset001_ISLES2022_DWI'
    assert (pp/'splits_final.json').read_bytes()==(baseline/'splits_final.json').read_bytes()
    plans=json.loads((pp/'nnUNetPlans.json').read_text())
    cfg=plans['configurations']['3d_fullres']
    assert importlib.metadata.version('nnunetv2')=='2.8.1'
    if args.mode=='train':
        smoke=json.loads((run.parent/'run_seed2022_smoke'/'completion.json').read_text())
        assert smoke['status']=='passed'
        assert smoke['plans_sha256']==hashlib.sha256((pp/'nnUNetPlans.json').read_bytes()).hexdigest()
    git=lambda *a:subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*a],cwd=ROOT,text=True).strip()
    commit=git('rev-parse','HEAD');branch=git('branch','--show-current')
    for path in ['scripts/train_spacing_experiment.py']:
        if git('status','--porcelain','--',path): raise RuntimeError('Trainer runner must be committed before execution')
    random.seed(2022);np.random.seed(2022);torch.manual_seed(2022);torch.cuda.manual_seed_all(2022)
    torch.set_num_threads(8)
    run.mkdir(parents=True,exist_ok=False)
    os.environ['nnUNet_results']=str(run/'results')
    trainer_module.nnUNet_results=str(run/'results')
    meta=dict(experiment_id=exp,mode=args.mode,git_commit=commit,branch=branch,dataset=name,fold=0,
              seed=2022,worker_seeds_train=[2022,2023,2024,2025],worker_seeds_val=[3022,3023],
              worker_order='nondeterministic',plans=cfg,trainer='nnUNetTrainer_SeededPreserve',
              plans_sha256=hashlib.sha256((pp/'nnUNetPlans.json').read_bytes()).hexdigest(),
              split_sha256=hashlib.sha256((pp/'splits_final.json').read_bytes()).hexdigest(),
              started_at=datetime.now().astimezone().isoformat(),torch=torch.__version__,nnunet='2.8.1',
              gpu=torch.cuda.get_device_name(),evaluation_checkpoint='checkpoint_final.pth')
    save(run/'experiment_config.json',meta)
    trainer=None
    try:
        trainer=nnUNetTrainer_SeededPreserve(plans,'3d_fullres',0,json.loads((pp/'dataset.json').read_text()),device=torch.device('cuda'))
        trainer.execution_started=time.monotonic()
        if args.mode=='smoke':
            trainer.on_train_start();trainer.on_train_epoch_start()
            for i in range(120):
                if i==20: torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                trainer.train_step(next(trainer.dataloader_train))
                if (i+1)%10==0: print(f'Feasibility {i+1}/120',flush=True)
            torch.cuda.synchronize();seconds=time.monotonic()-started
            with torch.no_grad():
                trainer.network.eval()
                trainer.validation_step(next(trainer.dataloader_val))
            result=dict(status='passed',iterations=100,seconds=seconds,seconds_per_iteration=seconds/100,
                        estimated_training_seconds_per_epoch=seconds*2.5,
                        patches_per_second=100*cfg['batch_size']/seconds,
                        peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gb=torch.cuda.max_memory_reserved()/1024**3)
            trainer.close_loaders()
        else:
            trainer.run_training()
            trainer.perform_actual_validation(save_probabilities=True)
            result=dict(status='training_and_validation_complete',epochs=trainer.current_epoch)
        result.update(plans_sha256=meta['plans_sha256'],finished_at=datetime.now().astimezone().isoformat())
        save(run/'completion.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        save(run/'failure.json',dict(error=repr(exc),finished_at=datetime.now().astimezone().isoformat()))
        if trainer: trainer.close_loaders()
        raise

if __name__=='__main__':main()
