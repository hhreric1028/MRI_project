# DWI_V2 dataset entry

Dataset files are intentionally local-only and ignored by Git. V2 owns its
derived datasets and must not write into the V1 dataset tree.

Expected layout:

```text
datasets/
├── nnUNet_raw/           V2 dataset definitions and local images
└── nnUNet_preprocessed/  V2 plans, fingerprints and preprocessed arrays
```

V2 preparation scripts may read the local V1 source data through explicitly
documented paths, but generated V2 files belong here. See
`../docs/v2_research_plan.md` and `../README.md` for the active protocol.

