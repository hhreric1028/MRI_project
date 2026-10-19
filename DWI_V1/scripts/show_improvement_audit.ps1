$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Set-Location ..
Write-Host 'ISLES2022 improvement planning audit - no training' -ForegroundColor Cyan
if (Test-Path -LiteralPath 'docs/improvement_planning_audit.json') {
    Write-Host 'Existing experiment output detected. No files were modified.'
    return
}
$auditLog = Join-Path $env:TEMP ('isles-planning-' + [guid]::NewGuid().ToString() + '.log')
Start-Transcript -LiteralPath $auditLog -NoClobber
& 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe' -u scripts/audit_improvement_plan.py
Write-Host "Audit exit code: $LASTEXITCODE"
Stop-Transcript
