$ErrorActionPreference='Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle='ISLES2022 S1a promotion: epoch 100 to 300'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Transcript -Path "$projectRoot\outputs\s1a300_${stamp}.log" -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u "$projectRoot\scripts\s1a300_pipeline.py"
    if ($LASTEXITCODE -ne 0) {
        throw "S1a-300 failed (exit $LASTEXITCODE). Outputs retained; no automatic retry."
    }
} finally {
    Stop-Transcript
}
