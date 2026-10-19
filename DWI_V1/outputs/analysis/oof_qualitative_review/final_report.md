# OOF qualitative segmentation review / failure analysis

## Scope and data integrity

This review used the frozen Kamel-style 5-fold OOF set only: 250 predictions, 250 existing probability exports, raw DWI, and GT. No training, inference, threshold change, post-processing, split change, or prediction modification was performed. All probability argmax masks were verified to reproduce the stored prediction exactly. 31 unique representative cases were reviewed; no probability file required backfill.

## Executive finding

The dominant failure is **component-level omission in spatially dispersed multifocal disease**, frequently mixed with low-confidence missed tissue. The prespecified taxonomy marks 108 / 250 cases with multifocal component miss and 63 / 250 with background-confident missed GT; these labels overlap and are not mutually exclusive. The OOF lesion recall (0.5585) is therefore substantially weaker than case-level overlap suggests (mean Dice 0.7711, median 0.8491).

## Answers to the eleven review questions

1. **How are poor cases poor?** They are usually incomplete rather than grossly misplaced: missing satellites, missing low-salience parts of a dominant lesion, or a shrunken boundary. Extreme low-Dice cases add complete miss or near-complete miss; FP-heavy cases are usually many tiny fragments rather than one large false lesion.

2. **Most common failure modes.** The largest prespecified non-other mechanism by label presence is multifocal_component_miss (108 cases), followed by boundary_shrinkage (77 cases). Mixed failures occur in 104 cases. See `failure_taxonomy_summary.csv` for definitions and metric effects.

3. **Small-lesion miss pattern.** Tiny lesions are often represented by only a few voxels/components; missing a small number drives voxel recall and Dice sharply downward. The reviewed examples commonly have background-like missed probability rather than a narrow threshold-borderline response.

4. **Multifocal miss pattern.** The prediction usually captures the dominant or most conspicuous component but omits multiple satellites. This directly explains the gap between acceptable overlap on the largest lesion and low lesion-level recall.

5. **Large partial-coverage pattern.** All six fixed-rule cases were reviewed. 6/6 have at least 70% of missed GT below probability 0.2, and 6/6 meet the quantitative bright-core-preference rule. Their errors include incomplete axial extent, peripheral/heterogeneous tissue omission, and satellite loss rather than one uniform complete-miss mechanism.

6. **Probability of missed lesion.** Across reviewed cases with missed GT, the mean missed-voxel fraction below 0.2 is 0.880, versus 0.120 in 0.2-0.5. Thus missed tissue is more often treated as background than merely sitting just under the 0.5 threshold.

7. **Does the model only catch the brightest part?** This occurs in 13/31 reviewed cases under the prespecified intensity-separation rule, including 6/6 large-partial cases. It is a recurring subtype, not a universal explanation: many cases have no strong detected-versus-missed intensity separation and are better explained by dispersed morphology/context.

8. **Fold 4 commonality.** Fold 4 has mean Dice 0.7394 and its selected worst cases combine small/low-salience lesions, multifocal component misses, and two large-partial failures. Fold-4 label rates are: small 6/50, multifocal 22/50, low-confidence 15/50, large-partial 2/50. Other folds are respectively 12/200, 86/200, 48/200, and 4/200. The pattern is therefore quantitatively more severe in Fold 4 but not qualitatively unique.

9. **Good versus bad visual difference.** Good cases maintain probability and component coverage across the full lesion morphology. Bad cases lose low-salience periphery, axial extent, or satellite components; probability collapses toward background in those omitted regions. The nine side-by-side pair montages document this across large, small, and multifocal lesions.

10. **Single highest-priority next direction.** **Sampling**—specifically component-aware lesion sampling that increases exposure to small satellite components and heterogeneous lesion margins—is the one recommended next experiment category.

11. **Why sampling?** The evidence is dominated by component omission (multifocal labels in 108 cases) and low lesion recall, while many missed voxels are already background-confident. Threshold or post-processing cannot recover absent probability support, and FP fragmentation is not the primary aggregate limitation. A sampling-only experiment is the cleanest single-variable test of whether underexposed small components/margins drive the observed pattern.

## Large partial-coverage synthesis

The six cases are heterogeneous in total volume and multiplicity, but share a failure to sustain coverage over the full GT support. Background-confident missed tissue is common, and all six meet the prespecified within-case intensity-separation rule for preferential capture of brighter DWI regions. This supports a salience/representation component while weakening a threshold-only explanation. The detailed per-case evidence and nearest good-large comparator are in `large_partial_review.csv`.

## Fold 4 synthesis

Fold 4 contains difficult validation cases and a higher burden of the same mechanisms seen elsewhere; it does not introduce a separate failure class. Its worst-case review should therefore inform the same component-aware sampling hypothesis rather than motivate a fold-specific model change.

## Interpretation limits

Taxonomy counts are reproducible rule-based approximations and categories overlap. DWI intensity comparisons use within-case robust normalization and support, but do not prove, salience mechanisms. Spatial proximity to ventricles/cortex was not assigned without an anatomical atlas; the montages retain full DWI context for human review. These findings are hypothesis-generating and do not alter the primary OOF result.

## Deliverables

- 8 CSV tables, including a 250-case source manifest and a deduplicated reviewed-case table.
- 31 standardized case montages, each with three data-driven axial slices and six panels.
- 9 good-vs-bad pair montages across the three required comparison groups.
- Failure-category contact sheets, a navigable `index.md`, and this report.

No follow-up experiment was started.
