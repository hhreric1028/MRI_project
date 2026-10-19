$projectRoot = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $projectRoot 'datasets/ISLES2022/nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $projectRoot 'datasets/ISLES2022/processed'
$env:nnUNet_results = Join-Path $projectRoot 'outputs/models'
# Session-local configuration: dot-source this file after conda activate MRI_DWI.
