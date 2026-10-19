@echo off
setlocal EnableExtensions
title Kamel Replication - Fixed 5-Fold CV
cd /d "%~dp0.."

set "PYTHON_EXE=C:\Users\User\anaconda3\envs\MRI_DWI\python.exe"
set "PYTHONPATH=%CD%\scripts"
set "SUITE=experiments\kamel_replication_seed2022"
set "PRED_REL=results\Dataset001_ISLES2022_DWI\nnUNetTrainer_KamelRep_Screen100__nnUNetPlans__3d_fullres"

echo [Kamel] Foreground CMD is active.
echo [Kamel] Fold 0 strict gate remains recorded as failed.
echo [Kamel] Human review accepted it as stochastic-equivalent and approved Fold 1-4.

if not exist "outputs\analysis\kamel_replication\human_approval_after_fold0_audit.json" goto approval_missing
if not exist "%SUITE%\fold_0\completion.json" goto fold0_failed
if not exist "%SUITE%\fold_0\metrics\summary.json" goto fold0_failed

echo [Kamel] Backfilling Fold 0 FP topology from existing predictions only...
if not exist "outputs\analysis\kamel_replication\fold_0\fp_topology_summary.json" (
  "%PYTHON_EXE%" scripts\kamel_fp_topology.py --fold 0 --metrics "%SUITE%\fold_0\metrics" --predictions "%SUITE%\fold_0\%PRED_REL%\fold_0\validation"
  if errorlevel 1 goto failed
)

for %%F in (1 2 3 4) do call :run_fold %%F
if errorlevel 1 goto failed

echo [Kamel] All five folds complete. Building exact 250-case OOF analysis...
"%PYTHON_EXE%" scripts\kamel_final_analysis.py
if errorlevel 1 goto failed
echo [Kamel] COMPLETE - five-fold CV and OOF analysis finished.
goto end

:run_fold
set "FOLD=%~1"
echo [Kamel] Starting Fold %FOLD% training and prediction...
"%PYTHON_EXE%" scripts\run_kamel_fold.py --fold %FOLD%
if errorlevel 1 exit /b 1
echo [Kamel] Fold %FOLD% prediction complete. Running fixed evaluation...
"%PYTHON_EXE%" scripts\evaluate_cv_fold.py --fold %FOLD% --predictions "%SUITE%\fold_%FOLD%\%PRED_REL%\fold_%FOLD%\validation" --output "%SUITE%\fold_%FOLD%\metrics"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" scripts\kamel_fold_analysis.py --fold %FOLD% --metrics "%SUITE%\fold_%FOLD%\metrics" --predictions "%SUITE%\fold_%FOLD%\%PRED_REL%\fold_%FOLD%\validation"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" scripts\kamel_fp_topology.py --fold %FOLD% --metrics "%SUITE%\fold_%FOLD%\metrics" --predictions "%SUITE%\fold_%FOLD%\%PRED_REL%\fold_%FOLD%\validation"
if errorlevel 1 exit /b 1
echo [Kamel] Fold %FOLD% evaluation complete.
exit /b 0

:fold0_failed
echo [Kamel] STOP - the already-running Fold 0 wrote failure.json.
goto failed

:approval_missing
echo [Kamel] STOP - human approval provenance file is missing.
goto failed

:failed
echo [Kamel] STOP - a command failed. Later folds were not started.

:end
echo.
echo This foreground window will remain open for review.
endlocal
