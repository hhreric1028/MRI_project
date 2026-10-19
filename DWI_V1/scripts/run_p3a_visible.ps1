$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$transcript = Join-Path $project "outputs\p3a_clip_zscore_$stamp.log"
Set-Location -LiteralPath $project
Start-Transcript -LiteralPath $transcript
try {
    & $python -u (Join-Path $project 'scripts\p3a_pipeline.py')
    if ($LASTEXITCODE -ne 0) { throw "P3a failed (exit $LASTEXITCODE). Outputs retained; no automatic retry." }
} finally {
    Stop-Transcript
    Write-Host "Transcript: $transcript"
}
