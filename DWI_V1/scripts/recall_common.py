"""Shared provenance and exclusive-output helpers for the 2 mm recall experiment."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / 'datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI'
SUITE = ROOT / 'experiments/recall_2mm_seed2022'
PROTECTED = [PP, ROOT/'outputs/models/Dataset001_ISLES2022_DWI',
             ROOT/'outputs/metrics/training_fold0', ROOT/'experiments/B0_baseline']

def now():
    return datetime.now().astimezone().isoformat()

def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,text=True).strip()

def provenance():
    import importlib.metadata
    paths = list((ROOT/'scripts').glob('*recall*.py')) + [ROOT/'scripts/evaluate_experiment.py', ROOT/'scripts/train_spacing_experiment.py']
    import nnunetv2
    package = Path(nnunetv2.__file__).parent
    paths += list((package/'training').rglob('*.py'))
    import batchgeneratorsv2
    paths += [p for directory in batchgeneratorsv2.__path__ for p in Path(directory).rglob('*.py')]
    import batchgenerators, dynamic_network_architectures
    paths += list((Path(batchgenerators.__file__).parent/'dataloading').rglob('*.py'))
    paths += [p for directory in dynamic_network_architectures.__path__ for p in Path(directory).rglob('*.py')]
    return dict(time=now(), git_commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
                git_status=git('status','--short'), python=sys.version, executable=sys.executable,
                packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                source_sha256={str(p):digest(p) for p in paths},
                plans_sha256=digest(PP/'nnUNetPlans.json'), split_sha256=digest(PP/'splits_final.json'),
                environment={k:v for k,v in os.environ.items() if k.startswith(('nnUNet','CUDA','CUBLAS','OMP','MKL'))})

def verify_baseline(output):
    audit = json.loads((ROOT/'docs/improvement_planning_audit.json').read_text())
    expected = audit['baseline_sha256']
    failures=[]
    print(f'Checking {len(expected)} protected baseline hashes...', flush=True)
    for key, value in expected.items():
        path=ROOT/key
        if not path.is_file() or digest(path)!=value: failures.append(key)
    # Detect newly created preprocessing/model files as well as changed existing files.
    roots=PROTECTED[:3]
    known={str((ROOT/k).resolve()).lower() for k in expected}
    additions=[str(p) for root in roots for p in root.rglob('*') if p.is_file() and str(p.resolve()).lower() not in known]
    save(output, dict(time=now(), checked=len(expected), changed_or_missing=failures, unexpected_files=additions,
                      passed=not failures and not additions))
    if failures or additions: raise RuntimeError('Baseline integrity mismatch; see '+str(output))

def protect_baseline():
    """Deny Python-mediated writes; Blosc arrays are separately opened in mode='r'."""
    roots=[str(p.resolve()).lower()+os.sep for p in PROTECTED]
    def guarded(path):
        if not isinstance(path,(str,bytes,os.PathLike)): return False
        p=str(Path(os.fsdecode(path)).resolve()).lower()
        return any(p==r[:-1] or p.startswith(r) for r in roots)
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and guarded(path): raise PermissionError('Protected baseline write: '+str(path))
        elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'):
            if guarded(args[0]): raise PermissionError('Protected baseline mutation: '+str(args[0]))
        elif event in ('os.rename','os.link','os.symlink'):
            if any(guarded(p) for p in args[:2]): raise PermissionError('Protected baseline path mutation')
    sys.addaudithook(audit)
