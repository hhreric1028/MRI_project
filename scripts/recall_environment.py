"""Current versus historical evidence; unavailable historical facts stay unknown."""
import json
import os
import torch
from recall_common import ROOT, SUITE, save, provenance, protect_baseline
from run_recall_experiment import make_trainer

def main():
    out=SUITE/'environment_audit';out.mkdir(exist_ok=False)
    protect_baseline();os.environ['nnUNet_results']=str(out/'inspection_only')
    current=provenance();trainer=make_trainer('B0-R2022','cpu');tr,va=trainer.raw_loaders()
    debug=json.loads((ROOT/'outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/debug.json').read_text())
    p1=json.loads((ROOT/'experiments/P1_spacing_1p5mm/run_seed2022_train_r2/experiment_config.json').read_text())
    evidence={k:v for k,v in debug.items() if any(word in k for word in ('version','torch','gpu','seed','cudnn','deterministic','benchmark'))}
    current.update(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        historical_B0=evidence,historical_P1={k:p1.get(k) for k in ['torch','nnunet','gpu','seed','worker_seeds_train','worker_seeds_val','worker_order']},
        P1_version_checks={'torch_equal':p1['torch']==torch.__version__,'nnunet_equal':p1['nnunet']==current['packages'].get('nnunetv2')},
        B0_train_augmentation_repr_equal=str(tr.transforms)==debug.get('dataloader_train.generator.transforms'),
        B0_validation_augmentation_repr_equal=str(va.transforms)==debug.get('dataloader_val.generator.transforms'),
        current_augmentation_train=str(tr.transforms),current_augmentation_validation=str(va.transforms),
        unavailable_historical=['B0 exact Python/NumPy/Torch/CUDA training seeds','B0/P1 deterministic flags at launch where absent','B0 installed source hashes at training time'],
        note='Current hashes establish prospective provenance, not historical source equivalence. CPU inspection flags are not used as GPU-run flags; actual GPU trainer records its flags separately.')
    save(out/'comparison.json',current)
    print(json.dumps({k:current[k] for k in ['torch','cuda','cudnn','P1_version_checks','B0_train_augmentation_repr_equal','B0_validation_augmentation_repr_equal','unavailable_historical']},indent=2),flush=True)

if __name__=='__main__':main()
