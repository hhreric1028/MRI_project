# DWI_V2 — failure-driven DWI segmentation research

This workspace investigates why selected DWI-only nnUNet cases fail and tests only
evidence-gated interventions. `case_0140` is the first anchor case; it is evidence
for hypothesis generation, never a tuning target.

## Guardrails

- Sibling `../DWI_V1` is a read-only evidence source for this workspace.
- All generated metrics, maps, figures, comparisons, and proposals live below
  `DWI_V2`.
- Phase 1 performs analysis only. It does not alter loss, sampling, spacing,
  preprocessing, inputs, checkpoints, or models, and it does not launch training.
- An intervention may be proposed only when the anchor evidence is checked against
  a similar-case cohort and the full Fold 0 validation set.

## Phase 2 outcome

- `V2-E1-100` (original DWI + exact P3a channel) completed on Fold 0 and was
  classified **RECALL-PRECISION TRADE-OFF**. It was not promoted.
- `V2-S1` failed its prespecified 1,000-patch exposure audit: the baseline crop
  already contained the selected low-salience region in 99.2% of patches, so hard
  centering produced zero exposure gain. Long training was not started.
- No 300- or 1000-epoch training was run. The workflow's final classification is
  **STOP**.
- See `outputs/comparisons/v2_final_report.md` for the fixed-cohort, bootstrap, and
  ten-question summary.

## Reproduce phase 1

From the V1 repository root, run:

```powershell
C:\Users\User\anaconda3\envs\MRI_DWI\python.exe DWI_V2\scripts\failure_analysis\analyze_case_0140.py
C:\Users\User\anaconda3\envs\MRI_DWI\python.exe -m unittest discover -s DWI_V2\tests -v
```

The source manifest and model paths are recorded in `configs/paths.yaml`. The
analysis refuses to write outside `DWI_V2`.

Phase 2 was run by the event-driven supervisor:

```powershell
C:\Users\User\anaconda3\envs\MRI_DWI\python.exe DWI_V2\scripts\run_v2_workflow.py
```

Child output is written to local stage logs; only process start/exit events are
emitted during a healthy run.
