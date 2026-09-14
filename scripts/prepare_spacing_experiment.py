"""Create isolated spacing dataset, exact split copy, official planning/preprocessing."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for k,v in [('nnUNet_raw','datasets/ISLES2022/nnUNet_raw'),('nnUNet_preprocessed','datasets/ISLES2022/processed'),('nnUNet_results','outputs/models')]: os.environ[k]=str(ROOT/v)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--id',type=int,choices=[11,12],required=True); args=ap.parse_args()
    spacing=1.5 if args.id==11 else 1.
    name=f'Dataset{args.id:03d}_ISLES2022_DWI'
    raw=Path(os.environ['nnUNet_raw']); pp=Path(os.environ['nnUNet_preprocessed'])
    for root in [raw,pp,Path(os.environ['nnUNet_results'])]:
        if list(root.glob(f'Dataset{args.id:03d}_*')): raise SystemExit('Existing experiment output detected. No files were modified.')
    if shutil.disk_usage(ROOT).free<50*1024**3: raise RuntimeError('Less than 50GB free')
    old='Dataset001_ISLES2022_DWI'
    print('Copying DWI/GT to isolated dataset '+name,flush=True)
    shutil.copytree(raw/old,raw/name)
    (pp/name).mkdir()
    for f in ['dataset_fingerprint.json','splits_final.json']:
        shutil.copy2(pp/old/f,pp/name/f)
    # Unchanged image data allows reuse of fingerprint; do not re-fit any statistics.
    from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset,preprocess_dataset
    plan_experiment_dataset(args.id,gpu_memory_target_in_gb=8,overwrite_target_spacing=(spacing,)*3,overwrite_plans_name='nnUNetPlans')
    assert (pp/name/'splits_final.json').read_bytes()==(pp/old/'splits_final.json').read_bytes()
    print('Preprocessing only 3d_fullres; 4 workers',flush=True)
    preprocess_dataset(args.id,configurations=['3d_fullres'],num_processes=[4])
    cfg=json.loads((pp/name/'nnUNetPlans.json').read_text())['configurations']['3d_fullres']
    ids=json.loads((pp/name/'splits_final.json').read_text())[0]
    for case in ids['train']+ids['val']:
        for suffix in ['.b2nd','_seg.b2nd','.pkl']:
            if not (pp/name/cfg['data_identifier']/(case+suffix)).is_file(): raise RuntimeError('Missing preprocessed case '+case)
    print('PREPROCESS_COMPLETE '+name,flush=True)

if __name__=='__main__': main()
