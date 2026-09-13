"""Read baseline, calculate planner candidates, save new audit only (exclusive create)."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ['nnUNet_raw'] = str(ROOT / 'datasets/ISLES2022/nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(ROOT / 'datasets/ISLES2022/processed')
os.environ['nnUNet_results'] = str(ROOT / 'outputs/models')
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

def main():
    destination = ROOT / 'docs/improvement_planning_audit.json'
    if destination.exists():
        raise SystemExit('Existing experiment output detected. No files were modified.')
    print('Planning audit only. No preprocessing or training.', flush=True)
    base = ROOT / 'datasets/ISLES2022/processed/Dataset001_ISLES2022_DWI'
    model = ROOT / 'outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres'
    paths = list(base.rglob('*')) + list(model.rglob('*')) + list((ROOT / 'outputs/metrics/training_fold0').rglob('*'))
    paths = sorted(set(p for p in paths if p.is_file()))
    def digest(p):
        with p.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    print('Hashing baseline configuration, arrays, checkpoints and validation...', flush=True)
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    # Bypass __init__: it may copy splits. Use its constants without file mutation.
    import inspect
    from unittest.mock import patch
    with patch('nnunetv2.experiment_planning.experiment_planners.default_experiment_planner._maybe_copy_splits_file', side_effect=RuntimeError('Unexpected split write')):
        raw = ROOT / 'datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI/splits_final.json'
        if raw.exists():
            raise RuntimeError('Constructor split-copy path exists; audit refuses to run')
        planner = ExperimentPlanner('Dataset001_ISLES2022_DWI', gpu_memory_target_in_gb=8)
    results = []
    for spacing in (2., 1.5, 1.):
        target = np.array([spacing]*3)
        fp = planner.dataset_fingerprint
        shape = np.median([compute_new_shape(j, i, target) for i,j in zip(fp['spacings'], fp['shapes_after_crop'])], axis=0)
        cfg = planner.get_plans_for_configuration(target, shape, 'audit_only', float(np.prod(shape)*planner.dataset_json['numTraining']), {})
        row = dict(spacing=target.tolist(), patch=list(map(int,cfg['patch_size'])), batch=int(cfg['batch_size']), physical_fov=(np.array(cfg['patch_size'])*target).tolist(), stages=cfg['architecture']['arch_kwargs']['n_stages'])
        results.append(row)
        print(json.dumps(row), flush=True)
    unchanged = all(digest(ROOT / p) == value for p,value in before.items())
    if not unchanged:
        raise RuntimeError('Baseline hash verification failed')
    source = Path(inspect.getfile(ExperimentPlanner))
    audit = dict(memory_target_gb=8, memory_target_note='Default planner hypothesis; 2mm reproduction checked against baseline', results=results, baseline_sha256=before, baseline_unchanged=unchanged, planner_source=str(source), planner_sha256=digest(source))
    with destination.open('x', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)
    print('Audit complete. Baseline unchanged. Saved '+str(destination), flush=True)

if __name__ == '__main__':
    main()
