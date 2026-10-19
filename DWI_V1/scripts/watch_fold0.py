"""Watch this run's log, stop nonfinite losses, and evaluate after official completion."""
import json,re,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
metrics=ROOT/'outputs/metrics/training_fold0'
folder=ROOT/'outputs/models/Dataset001_ISLES2022_DWI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0'


def main():
    while True:
        state=json.loads((metrics/'fold0_status.json').read_text())
        logs=sorted(folder.glob('training_log*.txt'))
        content=logs[-1].read_text(encoding='utf-8',errors='replace') if logs else ''
        bad=re.search(r'^.*(?:train_loss|val_loss).*\b(?:nan|[+-]?inf)\b.*$',content,re.I|re.M)
        times=[float(x) for x in re.findall(r'Epoch time: ([\d.]+) s',content)]
        epochs=re.findall(r'\bEpoch (\d+)\s*$',content,re.M)
        health=dict(checked_at=datetime.now().astimezone().isoformat(),status=state['status'],
                    completed_epochs=len(times),current_epoch=int(epochs[-1]) if epochs else None,
                    initial_10_epochs_passed=len(times)>=10 and not bool(bad),
                    estimated_training_hours_remaining=(1000-len(times))*sum(times[-10:])/len(times[-10:])/3600 if times else None)
        (metrics/'fold0_health.json').write_text(json.dumps(health,indent=2))
        if bad and state['status']=='running':
            (metrics/'fold0_watchdog_error.txt').write_text(bad[0])
            subprocess.run(['taskkill','/PID',str(state['pid']),'/T','/F'],capture_output=True)
            return
        if state['status']=='failed':return
        if state['status']=='training_and_validation_complete':
            with (metrics/'fold0_evaluation.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_fold0.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (metrics/'fold0_evaluation_status.json').write_text(json.dumps(dict(exit_code=result.returncode,visual_review_pending=result.returncode==0),indent=2))
            return
        time.sleep(15)


if __name__=='__main__':main()
