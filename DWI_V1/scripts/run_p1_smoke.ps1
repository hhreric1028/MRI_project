$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$log='outputs/p1_smoke_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log'
Start-Transcript -LiteralPath $log -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/train_spacing_experiment.py --id 11 --mode smoke
    if ($LASTEXITCODE -ne 0) { throw "P1 smoke failed: $LASTEXITCODE" }
} finally { Stop-Transcript }
