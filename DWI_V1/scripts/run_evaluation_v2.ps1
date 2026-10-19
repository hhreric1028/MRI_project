$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$dest = 'experiments/B0_baseline/evaluation_v2'
if (Test-Path $dest) { throw 'Existing experiment output detected. No files were modified.' }
$log = 'outputs/evaluation_v2_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log'
Start-Transcript -LiteralPath $log -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/evaluate_experiment.py --predictions outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation --output $dest
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $LASTEXITCODE" }
} finally { Stop-Transcript }
