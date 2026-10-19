# Evidence-gated V2 experiment matrix

Phase 1 supported exactly two proposals. Phase 2 has now completed the controlled
decision path; no general ablation sweep was added.

| ID | Status | Observed failure | Mechanistic hypothesis | Proposed intervention | Expected measurable change | Main risk/trade-off |
|---|---|---|---|---|---|---|
| V2-B0 | Reference only | B0-R2022 covers 33.6% of GT and leaves 93.896 mL FN | Existing V1 reference | Read existing B0-R2022 artifacts only | Fixed comparator | None; no new training |
| V2-E1 | Completed at 100; not promoted | FN DWI is less salient than TP; P3a-100 gives the largest rescue but increases FP | The original representation under-emphasizes lower-salience lesion territory, while clipped/z-score contrast exposes some of it but removes specificity cues | Two-channel input containing original DWI plus the frozen P3a clipped/z-score representation; keep architecture, loss, sampling, spacing, seed policy, and horizon matched | Higher voxel recall and lower FN/persistent-FN on the coverage cohort than B0, with less FP/precision loss than P3a-100 | Observed: recall improved, but precision, FP/case, volume MAE, and strict numerical-stability guards failed; classification `RECALL-PRECISION TRADE-OFF` |
| V2-S1 | Audit ineffective; training not started | Generic foreground oversampling (S1a) and recall-biased loss (L2a) did not help; missed voxels are low-salience parts inside a detected dominant component | The training sampler sees foreground but insufficiently emphasizes low-contrast intra-lesion regions and lesion edges of large/diffuse lesions | Training-derived low-salience hard-region patch sampling using only training DWI+GT | Increased coverage of low-CNR lesion voxels without reducing lesion-patch exposure | Observed: baseline already exposed the hard region in 99.2% of audited patches; all preferences gave zero gain, so the intervention was stopped before training |

## Phase 2 decision summary

| Metric | B0-R2022-100 | V2-E1-100 | Delta |
|---|---:|---:|---:|
| Mean Dice | 0.7645 | 0.7709 | +0.0064 |
| Lesion recall | 0.5942 | 0.6452 | +0.0510 |
| Lesion precision | 0.7147 | 0.6554 | -0.0593 |
| FP lesions/case | 2.14 | 3.06 | +0.92 |
| Severe-under cases | 8 | 6 | -2 |
| Fixed Worst-5 Mean Dice | 0.2814 | 0.2986 | +0.0172 |
| Coverage-cohort recall | 0.2729 | 0.3007 | +0.0278 |
| case_0140 persistent FN recovered | — | 0.104 mL | 0.1% of persistent FN |

Paired bootstrap supported the recall increase, but also supported a precision drop
and FP increase. Worst-5 and coverage-cohort confidence intervals crossed zero.
Therefore no candidate met all promotion guards, and the workflow stopped at 100
epochs.

## Required evaluation for V2-I1 and V2-I2

Each proposal must use a matched baseline and be evaluated on:

1. `case_0140`, reported but never used for checkpoint or parameter selection;
2. the prespecified broader coverage-failure cohort;
3. the complete fixed Fold 0 validation set; and
4. an additional prespecified large-lesion severe-coverage case outside this Fold 0
   set before any claim about the strict phenotype.

Measure baseline-FN rescue, newly missed baseline TP, persistent-FN volume,
voxel/component metrics, FP volume, and paired cohort uncertainty. A proposal is not
supported if improvement is confined to `case_0140` or is explained by uncontrolled
precision loss.

## Methods not currently supported

| Method | Why it should not be tried blindly now |
|---|---|
| Decision-threshold tuning | 94.55% of baseline FN is below probability 0.10 and only 0.91% is near 0.5; even threshold 0.1 barely improves recall |
| Generic foreground oversampling | S1a-100/300 does not improve anchor coverage over B0-R2022 |
| Generic recall-biased loss | L2a-100 is worse than B0-R2022 on anchor Dice and recall |
| Component-aware sampling | The dominant problem is incomplete coverage inside an already detected component, not failure to sample or detect a separate component |
| Spacing change | P1 does not help the anchor; phase-1 preprocessing shows no 2 mm resampling distortion in this case |
| Gamma or CLAHE | E1 Gamma and E2 CLAHE do not improve the anchor; CLAHE is the worst available variant here |
| Threshold calibration as the primary fix | The probability distribution is background-like, not near-threshold |
| Architecture/receptive-field change | The current evidence does not isolate receptive field as the mechanism |
| ADC/multimodal input | Potentially useful, but no ADC evidence was analyzed in phase 1; it remains a separate, untested hypothesis |

The proposals are intentionally limited to two. Adding a third method without new
evidence would return V2 to method-driven ablation rather than failure-driven work.
