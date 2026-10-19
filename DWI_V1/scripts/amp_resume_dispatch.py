"""Visible, logged, single-process dispatch. Smoke and formal resume are separate actions."""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/B0-R2022_resume_from_epoch34'

def invoke(name,script,*args):
    logdir=DEST/'logs';logdir.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-B',str(ROOT/'scripts'/script),*args]
    with (logdir/(name+'.command.json')).open('x',encoding='utf-8') as f:
        json.dump(dict(command=command,time=datetime.now().astimezone().isoformat()),f,indent=2)
    with (logdir/(name+'.log')).open('x',encoding='utf-8') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
        code=p.wait()
    with (logdir/(name+'.exit.json')).open('x',encoding='utf-8') as f:json.dump(dict(exit_code=code),f)
    if code:raise RuntimeError(f'{name} failed. Records retained; no automatic retry or parameter change.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    DEST.mkdir(exist_ok=True)
    with (DEST/'dispatch.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if a.mode=='smoke':invoke('unit_tests','test_amp_resume.py')
        if a.mode=='train':
            assert json.loads((DEST/'smoke/completion.json').read_text())['status']=='passed'
            assert (DEST/'pre_resume_report.md').is_file(), 'Report smoke findings before formal resume'
        invoke(a.mode,'run_amp_resume.py','--mode',a.mode)

if __name__=='__main__':main()
