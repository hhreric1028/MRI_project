"""Checked official fold-0 launch. Stops on reported nonfinite loss or runtime failure."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for key, relative in {'nnUNet_raw':'datasets/ISLES2022/nnUNet_raw','nnUNet_preprocessed':'datasets/ISLES2022/processed','nnUNet_results':'outputs/models'}.items():
    expected = str(ROOT / relative)
    if os.environ.get(key) != expected:
        raise RuntimeError(f'Path mismatch {key}: {os.environ.get(key)} != {expected}')
os.environ['nnUNet_n_proc_DA'] = '4'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/.matplotlib')
import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime
import torch
from nnunetv2.utilities.crossval_split import generate_crossval_split


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    name = 'Dataset001_ISLES2022_DWI'
    pp = Path(os.environ['nnUNet_preprocessed']) / name
    raw = Path(os.environ['nnUNet_raw']) / name
    output = Path(os.environ['nnUNet_results']) / name / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'fold_0'
    if output.exists():
        raise RuntimeError('Existing fold_0 output: explicit checkpoint/config review required before resume.')
    assert importlib.metadata.version('nnunetv2') == '2.8.1'
    assert torch.__version__ == '2.8.0+cu128' and torch.cuda.is_available()
    plans = json.loads((pp / 'nnUNetPlans.json').read_text())
    cfg = plans['configurations']['3d_fullres']
    assert cfg['batch_size'] == 8 and cfg['patch_size'] == [80,96,80] and cfg['spacing'] == [2,2,2]
    rows = list(csv.DictReader((ROOT/'datasets/ISLES2022/documentation/case_mapping.csv').open()))
    ids = sorted(row['nnunet_case'] for row in rows)
    assert len(ids) == len(set(ids)) == 250
    hashes = {}
    for row in rows:
        case = row['nnunet_case']
        assert digest(raw/'imagesTr'/f'{case}_0000.nii.gz') == row['dwi_sha256']
        assert digest(raw/'labelsTr'/f'{case}.nii.gz') == row['gt_sha256']
        for suffix in ('.b2nd','_seg.b2nd','.pkl'):
            path = pp/cfg['data_identifier']/(case+suffix)
            hashes[str(path.relative_to(ROOT))] = digest(path)
    splits = generate_crossval_split(ids, seed=12345, n_splits=5)
    splitfile = pp/'splits_final.json'
    if splitfile.exists():
        assert json.loads(splitfile.read_text()) == splits, 'Existing split differs'
    else:
        splitfile.write_text(json.dumps(splits,indent=2))
    train, val = splits[0]['train'], splits[0]['val']
    assert len(train)==200 and len(val)==50 and not set(train)&set(val)
    subjects = {r['nnunet_case']:r['original_case'].split('_ses-')[0] for r in rows}
    assert not {subjects[k] for k in train}&{subjects[k] for k in val}
    for filename in ('nnUNetPlans.json','dataset.json','splits_final.json'):
        hashes[str((pp/filename).relative_to(ROOT))] = digest(pp/filename)
    command = [str(Path(sys.executable).parent/'Scripts/nnUNetv2_train.exe'),'1','3d_fullres','0','-tr','nnUNetTrainer','-p','nnUNetPlans','-device','cuda','--npz']
    evidence = dict(started_at=datetime.now().astimezone().isoformat(),command=command,python=sys.version,
                    torch=torch.__version__,nnunetv2=importlib.metadata.version('nnunetv2'),gpu=torch.cuda.get_device_name(),
                    train=train,val=val,hashes=hashes,workers=4)
    metrics=ROOT/'outputs/metrics/training_fold0'
    (metrics/'training_preflight.json').write_text(json.dumps(evidence,indent=2))
    logpath=metrics/'fold0_console.log'
    statefile=metrics/'fold0_status.json'
    state=dict(status='running',started_at=evidence['started_at'],command=command)
    statefile.write_text(json.dumps(state,indent=2))
    with logpath.open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        state['pid']=proc.pid
        statefile.write_text(json.dumps(state,indent=2))
        for line in proc.stdout:
            log.write(line);log.flush()
            if re.search(r'(train_loss|val_loss).*\b(?:nan|[+-]?inf)\b',line,re.I):
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                state['stop_reason']=line.strip()
        code=proc.wait()
    state.update(status='training_and_validation_complete' if code==0 else 'failed',exit_code=code,finished_at=datetime.now().astimezone().isoformat())
    statefile.write_text(json.dumps(state,indent=2))
    print(json.dumps(state,indent=2),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
