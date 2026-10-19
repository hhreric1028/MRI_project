# DWI_V1 dataset entry

Dataset files are intentionally local-only and ignored by Git.

Expected layout:

```text
datasets/
├── ISLES2022/
│   ├── raw/             Original ISLES2022 archive and extracted modalities
│   ├── nnUNet_raw/      Dataset001_ISLES2022_DWI
│   ├── processed/       nnUNet plans and preprocessed cases
│   └── documentation/   Provenance, licenses, hashes and case mapping
└── atlases/             Local anatomical atlases used by analysis scripts
```

The source dataset is ISLES2022 v2.3.1 (250 training cases). Use
`scripts/download_isles2022.py` and `scripts/extract_isles2022.py` from the
`DWI_V1` directory, then follow the preparation steps in the project README.
Do not commit patient imaging, archives, derived NIfTI files or preprocessing
arrays.

