"""Serial fail-stop pipeline. Never resumes or overwrites a started run."""
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import traceback
from pathlib import Path
from recall_common import ROOT, SUITE, save, digest, verify_baseline, now

def stage(name,script,*args):
    logs=SUITE/'logs';logs.mkdir(exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]
    save(logs/(name+'.command.json'),dict(command=command,time=now()))
    # Child output is both visible and stored; combined log preserves stderr ordering.
    with (logs/(name+'.log')).open('x',encoding='utf-8') as log:
        print('\nSTAGE '+name,flush=True)
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'})
        for line in child.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=child.wait()
    save(logs/(name+'.exit.json'),dict(exit_code=code,time=now()))
    if code:raise RuntimeError(f'{name} failed ({code}); retained, no automatic retry')

def check_frozen():
    path=SUITE/'source_manifest_supplement.json'
    if not path.exists():path=SUITE/'source_manifest.json'
    manifest=json.loads(path.read_text())
    changed=[p for p,h in manifest.items() if digest(p)!=h]
    if changed:raise RuntimeError('Frozen sources changed: '+repr(changed))

def train_and_evaluate(name):
    check_frozen()
    stage(name+'_smoke','run_recall_experiment.py','--experiment',name,'--mode','smoke')
    check_frozen()
    stage(name+'_train','run_recall_experiment.py','--experiment',name,'--mode','train')
    completion=json.loads((SUITE/name/'train/completion.json').read_text())
    check_frozen()
    stage(name+'_evaluation','evaluate_experiment.py','--predictions',completion['predictions'],'--output',SUITE/name/'metrics')
    verify_baseline(SUITE/name/'baseline_after_evaluation.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','preflight_remaining','baseline','candidates'],required=True);a=p.parse_args()
    SUITE.mkdir(parents=True,exist_ok=True)
    # OS-backed lock survives process crashes without a stale lock that needs deletion.
    with (SUITE/'pipeline.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        state=SUITE/(a.phase+'_started.json');save(state,dict(time=now(),pid=os.getpid()))
        try:
            if a.phase=='preflight':
                paths=list((ROOT/'scripts').glob('*recall*.py'))+[ROOT/'scripts/evaluate_experiment.py',ROOT/'scripts/train_spacing_experiment.py']
                save(SUITE/'source_manifest.json',{str(p):digest(p) for p in paths})
                verify_baseline(SUITE/'baseline_initial.json')
                stage('protocol_tests','test_recall_protocol.py')
                stage('evaluation_tests','test_evaluation_v2.py')
                stage('environment_audit','recall_environment.py')
                stage('p1_failure_analysis','recall_analysis.py','--failure','--output',SUITE/'P1_failure_analysis')
                verify_baseline(SUITE/'baseline_after_failure_analysis.json')
                print('Preflight complete. Review failure panels before the baseline phase.',flush=True)
            elif a.phase=='preflight_remaining':
                # Continue only unfinished checks after metadata-only namespace-package failure.
                assert json.loads((SUITE/'baseline_initial.json').read_text())['passed']
                for name in ['protocol_tests','evaluation_tests']:
                    assert json.loads((SUITE/'logs'/(name+'.exit.json')).read_text())['exit_code']==0
                paths=list((ROOT/'scripts').glob('*recall*.py'))+[ROOT/'scripts/evaluate_experiment.py',ROOT/'scripts/train_spacing_experiment.py']
                save(SUITE/'source_manifest_supplement.json',{str(p):digest(p) for p in paths})
                stage('environment_audit_supplement','recall_environment.py')
                stage('p1_failure_analysis','recall_analysis.py','--failure','--output',SUITE/'P1_failure_analysis')
                verify_baseline(SUITE/'baseline_after_failure_analysis.json')
            elif a.phase=='baseline':
                assert (SUITE/'preflight_completed.json').exists() or (SUITE/'preflight_remaining_completed.json').exists()
                assert (SUITE/'failure_visual_review.md').exists(), 'Visual review required before B0-R2022'
                train_and_evaluate('B0-R2022')
                check_frozen()
                stage('sampling_audit','recall_sampling.py')
                print('Sampling complete. Review substantive exposure change (especially 1–5 mL) before recording S1a decision. Equality alone is not the criterion. L2a is not started ahead of this gate.',flush=True)
            else:
                assert (SUITE/'baseline_completed.json').exists()
                decision=json.loads((SUITE/'sampling_decision.json').read_text())
                assert decision['action'] in ['run_S1a','skip_S1a'] and decision.get('rationale')
                assert decision.get('audit_sha256')==digest(SUITE/'sampling_audit/summary.json')
                if decision['action']=='run_S1a':train_and_evaluate('S1a')
                train_and_evaluate('L2a')
                for name in ['S1a','L2a']:
                    if (SUITE/name/'metrics/summary.json').exists():
                        stage(name+'_paired_comparison','recall_analysis.py','--baseline',SUITE/'B0-R2022/metrics','--candidate',SUITE/name/'metrics','--output',SUITE/('comparison_'+name))
                verify_baseline(SUITE/'baseline_final.json')
            save(SUITE/(a.phase+'_completed.json'),dict(time=now(),status='phase_complete'))
        except BaseException as e:
            save(SUITE/(a.phase+'_failure.json'),dict(time=now(),error=repr(e),traceback=traceback.format_exc()))
            print('STOP: records preserved; no automatic retry or hyperparameter change.',flush=True)
            raise

if __name__=='__main__':main()
