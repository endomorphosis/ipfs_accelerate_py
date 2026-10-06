"""Observe only native tool counters inside one selected disposable trial."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

REMOTE = r'''import collections,json,pathlib,stat
root=pathlib.Path('/opt/ipfs-supervisor/worker-home/.grok/logs')
allowed={'read_file','search_replace','grep','list_dir','todo_write','run_terminal_cmd','search_tool','use_tool'}
counts=collections.Counter();calls=collections.Counter();files=0;rows=0;total=0;truncated=False
paths=sorted(root.rglob('*')) if root.is_dir() and not root.is_symlink() else []
for p in paths:
 if p.is_symlink() or not p.is_file() or p.suffix not in {'.json','.jsonl'}:continue
 if files>=16:truncated=True;break
 try:
  info=p.stat()
  if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=16*1024*1024:continue
  if total+info.st_size>64*1024*1024:truncated=True;break
  raw=p.read_bytes()
  if len(raw)>16*1024*1024:truncated=True;continue
 except OSError:continue
 total+=len(raw);files+=1
 for line in raw.splitlines():
  if len(line)>1024*1024:truncated=True;continue
  try:value=json.loads(line)
  except (ValueError,RecursionError):continue
  if type(value) is not dict:continue
  rows+=1;ctx=value.get('ctx')
  if type(ctx) is not dict:continue
  n=ctx.get('tool_count')
  if type(n) is int and 0<=n<=1024:counts[str(n)]+=1
  if 'tool_name' in ctx:
   tool=ctx.get('tool_name');tool=tool if type(tool) is str and tool in allowed else 'other_tool'
   success='success' if ctx.get('success') is True else 'failed' if ctx.get('success') is False else 'unknown'
   calls[tool+':'+success]+=1
state=pathlib.Path('/opt/ipfs-supervisor/state/benchmark')
markers={name:(state/name).is_file() for name in ('planning-result.json','admission.json','context-result.json')}
print(json.dumps({'phase_markers':markers,'native_log_files':files,'native_json_records':rows,'advertised_tool_count_frequencies':dict(counts),'tool_outcome_counts':dict(calls),'bounded_scan_truncated':truncated,'raw_logs_exported':False,'credential_files_read':False,'provider_calls':0}))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-id',required=True)
    parser.add_argument('--trial-name',required=True)
    args=parser.parse_args()
    if re.fullmatch('[0-9a-f]{64}',args.container_id) is None or re.fullmatch('grok-tune-mjcf-[0-9]{2}',args.trial_name) is None:
        parser.error('exact container id and trial label required')
    output=Path(__file__).resolve().parent/('native-tools-'+args.trial_name+'.json')
    if output.exists():raise SystemExit('fresh telemetry output required')
    started=time.monotonic();started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());checks=0;last=None;status='observing';failure=None;windows=[];omitted=0
    try:
        while time.monotonic()-started<1100:
            checks+=1
            process=subprocess.run(['docker','exec',args.container_id,'python3','-I','-c',REMOTE],capture_output=True,text=True,timeout=15)
            if process.returncode:
                status='container_unavailable';break
            value=json.loads(process.stdout)
            if value!=last:
                windows.append(dict(seconds=time.monotonic()-started,observation=value))
                if len(windows)>64:windows.pop(0);omitted+=1
            last=value
            time.sleep(3)
        else:status='monitor_deadline'
    except BaseException as exc:
        status='monitor_failed';failure=type(exc).__name__ if type(exc).__name__ in {'TimeoutExpired','OSError','ValueError','JSONDecodeError','KeyboardInterrupt','SystemExit'} else 'other'
    result=dict(schema='grok-native-tool-observation@1',trial_name=args.trial_name,status=status,error_type=failure,
        seconds=time.monotonic()-started,checks=checks,last_observation=last,
        started_utc=started_utc,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        observation_windows=windows,windows_omitted=omitted,
        phase_interpretation='Marker existence and native advertised tool counts are observations; they confer no START or task completion claim.',
        missing_native_logs_are_unknown=True,
        provider_calls=0,container_mutations=0,raw_logs_exported=False,raw_model_source_or_credential_data_exported=False,
        observation_completeness='last_successful_bounded_snapshot_before_container_unavailable',task_completion_authority=False)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
