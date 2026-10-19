$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$log='outputs/p1_training_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log'
Start-Transcript -LiteralPath $log -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/train_spacing_experiment.py --id 11 --mode train --revision r2
    if ($LASTEXITCODE -ne 0) { throw "P1 training failed: $LASTEXITCODE" }
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/evaluate_experiment.py --predictions experiments/P1_spacing_1p5mm/run_seed2022_train_r2/results/Dataset011_ISLES2022_DWI/nnUNetTrainer_SeededPreserve__nnUNetPlans__3d_fullres/fold_0/validation --output experiments/P1_spacing_1p5mm/run_seed2022_train_r2/metrics
    if ($LASTEXITCODE -ne 0) { throw "P1 evaluation failed: $LASTEXITCODE" }
} finally { Stop-Transcript }
