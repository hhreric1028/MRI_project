$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = 'C:\Users\User\anaconda3\envs\MRI_DWI\python.exe'
Set-Location -LiteralPath $ProjectRoot
& $PythonExe -u (Join-Path $PSScriptRoot 'image_enhancement_pipeline.py')
if ($LASTEXITCODE -ne 0) { throw "Image enhancement workflow failed with exit code $LASTEXITCODE" }
