param([ValidateSet('all')][string]$Phase='all')
$ErrorActionPreference='Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle="ISLES2022 100-epoch screening r2 - $Phase"
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Transcript -Path "$projectRoot\outputs\screen100_r2_${Phase}_${stamp}.log" -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u "$projectRoot\scripts\screen100_pipeline.py" --phase $Phase
    if ($LASTEXITCODE -ne 0) { throw "Screening failed (exit $LASTEXITCODE). Outputs retained; no automatic retry." }
} finally {
    Stop-Transcript
}
