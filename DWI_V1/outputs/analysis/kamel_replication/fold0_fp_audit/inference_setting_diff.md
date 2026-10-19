# Inference setting comparison

| Setting | B0-100 | Replication Fold 0 | Status | Evidence |
|---|---|---|---|---|
| Sliding-window step size | 0.5 | 0.5 | SAME | Shared nnUNetTrainer.perform_actual_validation implementation |
| Gaussian weighting | enabled | enabled | SAME | Shared predictor construction |
| Mirroring / TTA | enabled, axes (0,1,2) | enabled, axes (0,1,2) | SAME | Both checkpoint metadata |
| Device | CUDA | CUDA | SAME | Both runners instantiate trainer with cuda device |
| Probability export | enabled | enabled | SAME | Both call perform_actual_validation(save_probabilities=True) |
| Binary threshold | argmax, equivalent 0.5 | argmax, equivalent 0.5 | SAME | Same two-class label manager |
| Native-space probability resampling | resample_data_or_seg_to_shape | resample_data_or_seg_to_shape | SAME | Semantically identical plans |
| Probability interpolation | order 1, order_z 0 | order 1, order_z 0 | SAME | Semantically identical plans |
| Revert crop / transpose | nnU-Net export_prediction_from_logits | same | SAME | Shared nnU-Net 2.8.1 export implementation |
| Post-processing | none | none | SAME | Existing prediction manifests and evaluation definitions |
| Region / label order | background=0, infarct=1 | same | SAME | Semantically identical dataset JSON |

No inference-setting difference was found. The two added best-checkpoint runs used the same shared inference path and were inference-only.
