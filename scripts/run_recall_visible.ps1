param([ValidateSet('preflight','baseline','candidates')][string]$Phase='preflight')
$ErrorActionPreference='Stop'
$projectRoot='C:\Users\User\Desktop\MRI_project_v1'
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle="ISLES2022 Recall 2mm - $Phase"
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Transcript -Path "$projectRoot\outputs\recall_${Phase}_${stamp}.log" -NoClobber
try {
    & 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u "$projectRoot\scripts\recall_pipeline.py" --phase $Phase
    if ($LASTEXITCODE -ne 0) { throw "Phase failed (exit $LASTEXITCODE). All outputs retained; no restart." }
} finally { Stop-Transcript }
