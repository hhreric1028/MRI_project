# Evaluation metric audit

| Item | B0-100 | Replication Fold 0 | Status |
|---|---|---|---|
| Connected-component connectivity | 26 | 26 | SAME |
| Minimum component size | none | none | SAME |
| GT lesion definition | each nonzero 26-connected GT component | same | SAME |
| Prediction lesion definition | each nonzero 26-connected prediction component | same | SAME |
| Component overlap threshold | at least one shared voxel | same | SAME |
| Matching rule | maximum-cardinality, then maximum-IoU one-to-one assignment | same imported function | SAME |
| Many-to-one handling | not allowed; unmatched prediction components count as FP | same | SAME |
| Empty prediction handling | voxel metrics unavailable where denominator is zero; lesion counts retained | same | SAME |
| Double-empty handling | Dice is missing; HD95 status is both_empty | same | SAME |
| Lesion precision | aggregate TP / (aggregate TP + aggregate FP) | same | SAME |
| FP/case | total unmatched prediction components / 50 cases | same | SAME |

The replication evaluator adds fold identity and an extra combined subgroup only. It imports the consequential component matching and summary functions directly from the B0 evaluator, so no evaluation-code mismatch explains the gate failure.
