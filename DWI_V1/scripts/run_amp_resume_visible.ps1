param([ValidateSet('smoke','train')][string]$Mode='smoke')
$ErrorActionPreference='Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle="ISLES2022 B0-R2022 AMP-safe resume - $Mode"
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Transcript -Path "$projectRoot\outputs\amp_resume_${Mode}_${stamp}.log" -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u -B "$projectRoot\scripts\amp_resume_dispatch.py" --mode $Mode
    if ($LASTEXITCODE -ne 0) { throw "Resume phase failed (exit $LASTEXITCODE). Outputs retained; no automatic retry." }
} finally { Stop-Transcript }
