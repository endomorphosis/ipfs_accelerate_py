"""Start only bounded, read-only observers for one fresh authorized trial10."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
JOBS = A / 'grok-tune-mjcf-10/jobs/supervisor-full-tune-mjcf'
OUTPUT = A / 'grok-container/observer-coordinator-10.json'
PRIVATE = A / 'grok-container/private-observers-10'
PYTHON = '/home/barberb/.local/bin/python'
PROFILE = 'source384-5cpu-20gib-coding600@1'
INSPECT = '{"id":{{json .Id}},"name":{{json .Name}},"running":{{json .State.Running}}}'


def owned_case(jobs):
    if not jobs.exists():
        return None
    if jobs.resolve() != jobs.absolute() or not jobs.is_dir():
        raise ValueError('noncanonical_job_directory')
    entries = list(jobs.glob('tune-mjcf__*'))
    if len(entries) > 1:
        raise ValueError('ambiguous_owned_case')
    if not entries:
        return None
    case = entries[0]
    if (case.is_symlink() or not case.is_dir() or case.resolve() != case.absolute()
            or re.fullmatch(r'tune-mjcf__[A-Za-z0-9]+', case.name) is None):
        raise ValueError('invalid_owned_case')
    return case


def closed_progress():
    path = A / 'timeout-recovery/native-live-monitor-10.jsonl'
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return None
            stream.seek(max(0, info.st_size - 32768))
            rows = stream.read(32768).splitlines()
        if not rows:
            return None
        value = json.loads(rows[-1])
        if value.get('schema') != 'owned-native-live-phase-observation@1':
            return None
        failure = value.get('bridge_failures') or [{}]
        status = value.get('supervisor', {}).get('status')
        return dict(native_status=status if status in ('starting','running','stopped','failed','completed') else None,
            bridge_observation_status=value.get('bridge_observation', {}).get('status')
                if value.get('bridge_observation', {}).get('status') in ('observed','missing','invalid') else None,
            terminal_failure=failure[0].get('phase') == 'terminal_failure',
            native_returncode=failure[0].get('returncode') if type(failure[0].get('returncode')) is int else None,
            oom_kill_delta=value.get('oom_kill_delta_since_first_observation')
                if type(value.get('oom_kill_delta_since_first_observation')) is int else None)
    except (OSError, ValueError, TypeError, IndexError):
        return None


def main():
    if len(sys.argv) != 1:
        raise SystemExit('fixed authorized trial10 coordinator takes no arguments')
    if OUTPUT.exists() or PRIVATE.exists():
        raise SystemExit('fresh coordinator outputs required')
    PRIVATE.mkdir(mode=0o700)
    started = time.monotonic()
    deadline = started + 1195
    report = dict(schema='exact-owned-trial-observer-coordination@1', trial='10',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='waiting_for_owned_case', container_mutations=0, provider_calls=0,
        broad_container_search=False, task_body_read=False, authority=False, children=[])
    children = []
    handles = []
    def persist():
        temporary = OUTPUT.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        os.replace(temporary, OUTPUT)
    try:
        persist()
        cid = None
        while time.monotonic() < min(deadline, started + 180):
            case = owned_case(JOBS)
            if case is not None:
                name = case.name.lower() + '__env-main-1'
                try:
                    result = subprocess.run(['docker','inspect','--format',INSPECT,name],
                        capture_output=True, text=True, timeout=5)
                except subprocess.TimeoutExpired:
                    result = None
                if result is not None and result.returncode == 0:
                    value = json.loads(result.stdout)
                    if (type(value) is not dict or value.get('name') != '/' + name
                            or type(value.get('id')) is not str
                            or re.fullmatch('[0-9a-f]{64}', value['id']) is None):
                        raise ValueError('owned_container_identity_mismatch')
                    if value.get('running') is True:
                        cid = value['id']
                        report.update(container_id=cid, container_name=name,
                            owned_case=case.name, captured_after_seconds=round(time.monotonic()-started,3))
                        break
            time.sleep(1)
        if cid is None:
            report['status'] = 'owned_container_unavailable_before_deadline'
            return 1
        commands = [
            ('native', [PYTHON,'-B',str(A/'timeout-recovery/monitor_owned_native_trial.py'),cid,'10']),
            ('resources',[PYTHON,'-B',str(A/'grok-container/observe_coding600_resources.py'),
                '--container-id',cid,'--attempt','10','--resource-profile',PROFILE]),
            ('tools',[PYTHON,'-B',str(A/'grok-container/monitor_closed_tools.py'),
                '--container-id',cid,'--trial-name','grok-tune-mjcf-10']),
        ]
        for label, command in commands:
            handle = os.fdopen(os.open(PRIVATE/(label+'.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb')
            handles.append(handle)
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                start_new_session=True, cwd=A)
            children.append((label,process))
            report['children'].append(dict(label=label,pid=process.pid,status='running',exit_code=None))
        report['status']='observing';persist()
        print(json.dumps({'trial':'10','status':'observers_started','container_id':cid,
            'captured_after_seconds':report['captured_after_seconds'],'observer_count':len(children)}),flush=True)
        last_progress = None
        last_print = time.monotonic()
        while time.monotonic() < deadline:
            changed = False
            for row, (_label, process) in zip(report['children'],children):
                code = process.poll()
                if code is not None and row['exit_code'] is None:
                    row.update(status='exited',exit_code=code);changed=True
            progress=closed_progress()
            if progress is not None and (progress != last_progress or time.monotonic()-last_print>=30):
                print(json.dumps({'trial':'10','elapsed_seconds':round(time.monotonic()-started,3),**progress}),flush=True)
                last_progress=progress;last_print=time.monotonic()
            if changed:persist()
            if all(process.poll() is not None for _label,process in children):
                report['status']='observers_exited';break
            time.sleep(2)
        else:
            report['status']='coordinator_deadline'
        return 0 if report['status']=='observers_exited' else 1
    except Exception as error:
        report['status']='coordinator_failed'
        report['error_type']=type(error).__name__ if type(error).__name__ in {'ValueError','OSError','FileNotFoundError','TimeoutExpired','JSONDecodeError'} else 'other'
        return 1
    finally:
        # Only exact Popen children created above may receive a signal. No
        # container process, service or independent observer is mutated.
        for label,process in children:
            if process.poll() is None:
                process.terminate()
        for row,(_label,process) in zip(report['children'],children):
            try:code=process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill();code=process.wait(timeout=1)
            row.update(status='exited',exit_code=code)
        for handle in handles:handle.close()
        report['elapsed_seconds']=round(time.monotonic()-started,3)
        report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        report['monitor_absence_is_not_success']=True
        persist()
        print(json.dumps({'trial':'10','status':report['status'],
            'elapsed_seconds':report['elapsed_seconds'],'observer_exit_codes':{r['label']:r['exit_code'] for r in report['children']}}),flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
