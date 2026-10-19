# DWI_V1 B0-100 five-fold weights

Release tag: `weights-v1-b0-100-5fold`

This is the selected deployable baseline from the complete 250-case out-of-fold
evaluation. It contains the five `checkpoint_final.pth` files used by the fixed
evaluation protocol. The Kamel replication was not packaged because its
configuration was technically equivalent and its small Dice difference was
classified as stochastic variation; B0 had better precision, F1, FP/case and
volume MAE.

The checkpoint binaries are GitHub Release assets, not Git-tracked files. Their
local staging directory is:

```text
DWI_V1/outputs/release_assets/v1-b0-100-5fold/
```

All five checkpoints were inspected successfully: `current_epoch=100`,
`network_weights` is present, and their combined size is 662,555,411 bytes.
Verify downloads against `manifest.json` before use.

Fold 0 uses `nnUNetTrainer_B0R2022_Screen100`. Folds 1–4 use the
`nnUNetTrainer_B0Full5fold_Screen100` selector wrapper. The configuration audit
confirmed that the wrapper changes only fixed-fold selection; model and training
configuration remain equivalent.

No DWI_V2 checkpoint is included because the V2 workflow ended with
`STOP / no_promotion`.

