$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = 'outputs/spacing_preparation_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log'
Start-Transcript -LiteralPath $log -NoClobber
try {
    foreach ($dataset in @(11,12)) {
        & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/prepare_spacing_experiment.py --id $dataset
        if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $LASTEXITCODE" }
    }
} finally { Stop-Transcript }
