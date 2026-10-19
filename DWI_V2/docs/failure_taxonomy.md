# Failure taxonomy

This taxonomy describes observable failure phenotypes. Labels are not causal
diagnoses, and a case may carry more than one label.

## Definitions

| Phenotype | Operational evidence |
|---|---|
| Large partial-coverage failure | GT ≥50 mL, lesion detected, Pred/GT and voxel recall both materially low |
| Low-salience lesion region | FN DWI distribution is lower or less separated from local tissue than TP |
| Representation-dominant confidence failure | Most FN probabilities are close to background rather than to the decision threshold |
| Threshold-related failure | A large share of FN probabilities is immediately below 0.5 and a small threshold change recovers coverage without disproportionate FP |
| Multifocal detection failure | One or more GT components have no prediction overlap |
| Dominant-component fragmentation | GT is dominated by one component but prediction is split into many components |
| Persistent-FN phenotype | The same GT territory is missed across materially different V1 interventions |
| Preprocessing-sensitive failure | Resampling/cropping/normalization measurably alters lesion geometry or suppresses signal beyond a simple affine intensity transform |

## case_0140 assignment

| Phenotype | Status | Evidence |
|---|---|---|
| Large partial-coverage failure | Confirmed | GT 141.368 mL; B0-R2022 Pred/GT 0.379; voxel recall 0.336; FN 93.896 mL |
| Dominant-component fragmentation | Confirmed | 3 GT components, largest 141.128 mL (99.83% of GT); prediction has 14 components |
| Multifocal detection failure | Not supported | One-to-one component matching detects all 3 GT components; lesion recall 1.0 |
| Low-salience lesion region | Supported | FN median DWI 312.687 versus TP 373.328; FN/TP median ratio 0.838; FN local CNR 1.032 versus TP 1.869 |
| Representation-dominant confidence failure | Strongly supported | FN median probability 0.000035; 94.55% of FN <0.10; only 0.91% of FN lies in 0.40–0.50 |
| Threshold-related failure | Rejected as primary mechanism | Lowering the diagnostic threshold from 0.5 to 0.1 raises voxel recall only from 0.336 to 0.372 |
| Persistent-FN phenotype | Confirmed | 72.896 mL, or 51.56% of GT, is missed by all 12 available model variants |
| Preprocessing-sensitive failure | Not supported by current artifacts | Native and processed spacing remain 2 mm; restored GT is identical; no resampling shape change; raw/processed within-brain correlation is ~1.0 |

The most concise phase-1 label is:

> **Large, dominant-component, low-salience partial-coverage failure with
> representation-dominant confidence and a large persistent-FN core.**

This is not a small-lesion detection miss. The model sees every GT component but
confidently assigns most of the dominant lesion territory to background.

## Cross-model phenotype stability

- P3a-100 provides the largest observed coverage rescue: voxel recall 0.427 and
  Dice 0.540, rescuing 13.920 mL (14.82%) of B0-R2022 FN.
- That rescue costs FP volume: 21.816 mL and voxel precision 0.735, versus B0-R2022
  FP 6.104 mL and precision 0.886.
- E3-Scale-300 is a smaller, more moderate shift: recall 0.365, Dice 0.513, and FP
  8.352 mL.
- P1, S1a, L2a, Gamma, and CLAHE do not resolve the anchor failure.

The high FN overlap across models and the 72.896 mL persistent core argue against
explaining the failure as a training-horizon or simple decision-threshold artifact.

## Similar-case status

No second Fold 0 case meets the strict large-lesion coverage-failure definition.
Therefore `case_0140` is an isolated large-lesion phenotype within this fold. The
broader coverage cohort shows that undercoverage repeats at other lesion sizes and
morphologies, but it cannot establish that the exact large-lesion mechanism repeats.

## Reserved anchors

These remain separate phenotypes and were not expanded in phase 1:

- `case_0068`: small severe miss.
- `case_0181`: partial coverage.
- `case_0186`: multifocal/high-FN case.

They must not be collapsed into the `case_0140` label merely because their Dice or
Pred/GT is low.

