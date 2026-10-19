# MRI_project_v1

ISLES2022 DWI ischemic-stroke lesion segmentation research, organized as two
independent and reproducible project versions.

## Repository layout

```text
MRI_project_v1/
├── DWI_V1/       Original DWI-only baseline and follow-up experiments
├── DWI_V2/       Failure-driven second-generation workflow
├── .codex        Repository conventions for Codex-assisted work
├── .gitignore    Dataset, model-weight and generated-output policy
└── README.md     This entry point
```

| Version | Purpose | Start here |
|---|---|---|
| DWI_V1 | Original nnUNet DWI-only baseline, ablations, five-fold comparisons and clinical analyses | [DWI_V1/README.md](DWI_V1/README.md) |
| DWI_V2 | Failure-driven preprocessing and dual-channel research workflow | [DWI_V2/README.md](DWI_V2/README.md) |

## Storage policy

GitHub contains source code, environment files, experiment configuration,
small CSV/JSON metrics, and Markdown reports. The following stay on the local
machine and are ignored by Git:

- ISLES2022 images and preprocessed arrays;
- nnUNet checkpoints and exported model weights;
- predictions, inference volumes and per-case binary files;
- generated figures, logs, caches and build intermediates.

The ignored files have not been deleted. Dataset acquisition and expected local
paths are documented in [DWI_V1/datasets/README.md](DWI_V1/datasets/README.md)
and [DWI_V2/datasets/README.md](DWI_V2/datasets/README.md).

## Model weights

The repository does not store checkpoint binaries in Git history. The selected
release model is the DWI_V1 B0-100 five-fold ensemble used for the complete
250-case OOF evaluation. Its five final checkpoints are published separately as
GitHub Release assets; checksums and provenance are recorded in
[`DWI_V1/outputs/releases/v1-b0-100-5fold/`](DWI_V1/outputs/releases/v1-b0-100-5fold/).

DWI_V2 has no released checkpoint because its final decision was `STOP` with no
candidate passing every promotion guard.

## Quick start

Choose a version first; commands and relative paths are version-local.

```powershell
cd DWI_V1
conda env create -f environment.yml
conda activate MRI_DWI
. ./scripts/set_paths.ps1
```

For V2, begin with [DWI_V2/README.md](DWI_V2/README.md). Do not run training
from the repository root because each version owns its own `datasets/`,
`experiments/`, and `outputs/` trees.

## Publishing checklist

Before pushing, run:

```powershell
git status --short
git check-ignore -v DWI_V1/datasets/ISLES2022/raw/ISLES-2022.zip
git ls-files | Select-String -Pattern '\.(pth|pt|ckpt|nii|nii\.gz|npz|npy|pkl|b2nd)$'
```

The final command should produce no output.

