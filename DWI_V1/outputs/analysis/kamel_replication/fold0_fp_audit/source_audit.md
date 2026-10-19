# Fold 0 source audit

## B0-100

- Prediction folder: `C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\recall_screen100_seed2022_r2\B0_R2022_screen100\train\results\Dataset001_ISLES2022_DWI\nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres\fold_0\validation`
- Checkpoint used for the existing prediction: `C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\recall_screen100_seed2022_r2\B0_R2022_screen100\train\results\Dataset001_ISLES2022_DWI\nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth`
- Checkpoint SHA-256: `cfca704ff0c337887cc091e83da983de45bbfa8c402d436a21ae31ec17b58b3d`
- Training/inference runner command: `C:\Users\User\anaconda3\envs\MRI_DWI\python.exe -u C:\Users\User\Desktop\MRI_project_v1\DWI_V1\scripts\run_screen100_experiment.py --experiment B0_R2022_screen100 --mode train`
- Fixed metric command: `C:\Users\User\anaconda3\envs\MRI_DWI\python.exe -u C:\Users\User\Desktop\MRI_project_v1\DWI_V1\scripts\evaluate_experiment.py --predictions C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\recall_screen100_seed2022_r2\B0_R2022_screen100\train\results\Dataset001_ISLES2022_DWI\nnUNetTrainer_B0R2022_Screen100__nnUNetPlans__3d_fullres\fold_0\validation --output C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\recall_screen100_seed2022_r2\B0_R2022_screen100\metrics`
- Inference checkpoint: final, epoch 100
- Post-processing: none
- Binary decision: nnU-Net foreground argmax, equivalent to probability threshold 0.5 for two classes
- Connected components: 26-connectivity
- Metric implementation: `scripts/evaluate_experiment.py`

## Replication Fold 0-100

- Prediction folder: `C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\kamel_replication_seed2022\fold_0\results\Dataset001_ISLES2022_DWI\nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres\fold_0\validation`
- Checkpoint used for the existing prediction: `C:\Users\User\Desktop\MRI_project_v1\DWI_V1\experiments\kamel_replication_seed2022\fold_0\results\Dataset001_ISLES2022_DWI\nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth`
- Checkpoint SHA-256: `c3591dc0f15e9c9232ea280ce48614607adbf726322b8ccf61cc4fc032d54f70`
- Training/inference runner: `python scripts/run_kamel_fold.py --fold 0`
- Fixed metric runner: `python scripts/evaluate_cv_fold.py --fold 0 --predictions <folder> --output <folder>`
- Inference checkpoint: final, epoch 100
- Post-processing: none
- Binary decision: nnU-Net foreground argmax, equivalent to probability threshold 0.5 for two classes
- Connected components: 26-connectivity
- Metric implementation: `evaluate_cv_fold.py` imports `metrics()` and `summarize()` unchanged from `scripts/evaluate_experiment.py`

## Source comparison

- Plans JSON semantic equality: `True`
- Dataset JSON semantic equality: `True`
- Split SHA-256 equality: `True`
- Package manifest equality: `True`
- Recorded B0 evaluation source hash: `44a97e68800e66754ea07b2f00f3abf6c274489ee6ea0f8fd34740c81a091e64`
- Current/shared evaluation source hash: `44a97e68800e66754ea07b2f00f3abf6c274489ee6ea0f8fd34740c81a091e64`
- Evaluation source hash equality: `True`
- Both existing prediction sets used `checkpoint_final.pth`. Checkpoint selection is therefore matched for the failed gate.
- The trainer class name differs because the replication subclass permits fixed folds 0–4; its Fold 0 data, architecture, augmentation, loss, optimizer, and inference path inherit B0 behavior.
