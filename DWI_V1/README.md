# DWI_V1

This directory is an independent project root within `MRI_project_v1`.
Run all V1 commands from this directory so dataset, experiment and output paths
resolve locally.

Goal: DWI-based ischemic stroke lesion segmentation and lesion-level analysis.

```text
DWI MRI
↓
Lesion Segmentation
↓
Lesion Detection
↓
Lesion Volume
↓
Anatomical Localization
↓
Structured Clinical Description
```

Current phase: **Phase 1 — Build ISLES2022 DWI-only segmentation baseline.**
Current status (2026-09-09): fold 0 completed 1,000 epochs and 50 full-image validation cases.
Mean Dice: 0.778159 (49 valid cases; one both-empty case is N/A).
Median Dice: 0.829133. Mean absolute lesion volume error: 4.509421 mL.
Nine selected cases have completed visual QA. Training monitoring has stopped; no fold 1 started.
The single-GT lesion-volume test also passed. See [training report](docs/training_report.md).

## Start here / 結果入口

| 內容 | 位置（相對專案根目錄） |
|---|---|
| **依完整比較／100／300／1000 epochs 整理後的入口** | [PROJECT_RESULTS/README.md](PROJECT_RESULTS/README.md) |
| **B0-100 vs KamelRep-100 完整五折比較** | [outputs/analysis/B0_full5fold_100/final_report.md](outputs/analysis/B0_full5fold_100/final_report.md) |
| 完整訓練結果與限制 | [docs/training_report.md](docs/training_report.md) |
| 50 例逐例 Dice、體積誤差 | [outputs/metrics/training_fold0/fold0_validation_cases.csv](outputs/metrics/training_fold0/fold0_validation_cases.csv) |
| 訓練評估摘要 | [outputs/metrics/training_fold0/fold0_validation_summary.json](outputs/metrics/training_fold0/fold0_validation_summary.json) |
| Loss 等訓練曲線 | [progress.png](outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/progress.png) |
| GT／prediction 疊圖 | `outputs/figures/training_fold0/` |
| 官方模型、50 例預測與機率 | `outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`，預測位於其中 `validation/` |
| 資料夾用途與腳本索引 | [docs/project_structure.md](docs/project_structure.md) |

`docs/phase1_report.md` 與 `docs/preprocessing_report.md` 是歷史階段報告；其中「尚未訓練」描述的是當時狀態。

Future phases:

2. DWI + ADC comparison
3. DWI + ADC + FLAIR
4. Lesion volume and connected-component analysis
5. Brain atlas anatomical localization
6. Structured report generation

## Environment

Independent Conda environment: `MRI_DWI`, Python 3.11, PyTorch 2.8.0 with CUDA 12.8,
nnUNetv2 2.8.1. This is an official upstream installation with no old project dependencies.
`dynamic-network-architectures` is an upstream nnUNet dependency and is installed cleanly.
Exact installed Python packages are pinned in `requirements.txt`; Conda packages are in `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate MRI_DWI
. ./scripts/set_paths.ps1
```

For a staged install, create Python 3.11 first, install PyTorch before nnUNet:

```powershell
conda create -n MRI_DWI python=3.11 pip -y
conda activate MRI_DWI
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The GPU driver is managed separately. `nvidia-smi` reports the driver's maximum CUDA compatibility,
while `torch.version.cuda` reports the installed runtime. A separate CUDA Toolkit is not required for these wheels.

## Data and reproducibility

Official release: [Zenodo ISLES2022 v2.3.1](https://zenodo.org/records/7960856), 250 training cases.
See `datasets/ISLES2022/documentation/source.md` for provenance, citations and the source license discrepancy.

```powershell
python scripts/download_isles2022.py
python scripts/extract_isles2022.py
python src/data/inspect_isles2022.py
python src/data/visualize_samples.py
python src/data/convert_isles2022.py
python scripts/verify_dataset.py
```

Download resumes partial files and validates official MD5. Extraction refuses to overwrite raw files.
Run extraction/conversion once; they intentionally reject an existing destination rather than silently replacing data.
The inspection and visualization scripts are rerunnable and update generated reports only.
No case is deleted or excluded automatically. Conversion requires all 250 cases and compatible DWI/GT geometry.
It reads actual labels first and refuses unexpected label sets; no label remapping is performed.
Converted NIfTIs are byte-identical copies, checked with SHA256 and tracked in `case_mapping.csv`.

Some raw GT files have NIfTI slope/intercept scaling: nibabel float64 reads a foreground value
of `0.9999999997671694`. The official nnUNet `SimpleITKIO` reader reads exactly `1` (float32).
The audit records both value sets and validates near-integer native values with absolute tolerance `1e-8`.
`dataset.json` explicitly selects this official reader. No rounding or rewriting of mask files occurs.

`0000` is the only input channel and means DWI. ADC and FLAIR remain in raw data.
Native FLAIR may differ in shape, spacing and orientation; no automatic registration or resampling is applied.
Sample QC uses seed 2022. Matching geometry establishes grid compatibility, not a complete anatomical quality judgment.

`nnUNetv2_plan_and_preprocess -h` checks CLI availability. The verification script calls the official
`verify_dataset_integrity` function used by that command and resolves dataset ID 1 using nnUNet itself.
The full planning command, including `--no_pp`, also performs other work; it is deliberately not used for this integrity-only phase.

## Storage

- `datasets/ISLES2022/raw/`: official archive and all extracted modalities/annotations; immutable inputs.
- `datasets/ISLES2022/nnUNet_raw/Dataset001_ISLES2022_DWI/`: DWI/GT copies and dataset.json.
- `datasets/ISLES2022/processed/`: official fingerprint, plans, dataset metadata and preprocessed `2d`/`3d_fullres` cases.
- `datasets/ISLES2022/documentation/`: source records, raw hashes and case mapping.
- `outputs/metrics/data/`: source inventory, conversion and integrity-check records.
- `outputs/metrics/preprocessing/`: preprocessing QA, preservation checks and logs.
- `outputs/metrics/training_fold0/`: fold 0 metrics, status, audit and execution logs.
- `outputs/figures/data/`: five native-grid sample QC figures.
- `outputs/figures/preprocessing/`: preprocessing QA figures.
- `outputs/figures/training_fold0/`: final GT/prediction QA figures.
- `outputs/predictions/`: reserved for future predictions.
- `src/data/`, `src/preprocessing/`, `src/analysis/`: implemented data checks, preprocessing QA and single-mask volume analysis.
- `src/{training,inference,utils}/`, `configs/`, `notebooks/`: reserved directories.
- `docs/environment/`: environment decisions, versions and installation notes.
- `docs/logs/`: original setup/download/extraction logs.
- `docs/reference/`: official source snapshots and historical directory listing.

See `docs/phase1_report.md` for measured results and `docs/reference/directory-tree.txt` for the full directory tree.
The phase1 and preprocessing reports record earlier milestones; `docs/training_report.md` records the completed fold 0 baseline.

To rerun the read-only preprocessing audit and regenerate sample QC figures:

```powershell
python src/preprocessing/check_preprocessed_cases.py
```
