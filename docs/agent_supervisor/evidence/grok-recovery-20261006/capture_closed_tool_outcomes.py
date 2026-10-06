"""Capture bounded native completion events without arguments or message text."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

REMOTE = r'''from pathlib import Path
import json,math
root=Path('/opt/ipfs-supervisor/worker-home/.grok/logs')
allowed={'read_file','search_replace','grep','list_dir','todo_write','run_terminal_cmd','run_terminal_command'}
rows=[];files=0;total=0;records=0;truncated=False
for path in sorted(root.rglob('*')) if root.is_dir() else []:
 if path.is_symlink() or not path.is_file() or path.suffix not in {'.json','.jsonl'}:continue
 if files>=16:truncated=True;break
 size=path.stat().st_size
 if size>16777216 or total+size>67108864:truncated=True;continue
 raw=path.read_bytes();files+=1;total+=len(raw)
 for line in raw.splitlines():
  if len(line)>1048576:truncated=True;continue
  try:value=json.loads(line)
  except (ValueError,RecursionError):continue
  records+=1;ctx=value.get('ctx') if type(value)is dict else None
  if type(ctx)is not dict or 'tool_name' not in ctx:continue
  tool=ctx.get('tool_name');tool=tool.strip().lower().split('.')[-1] if type(tool)is str else None
  elapsed=ctx.get('elapsed_ms')
  rows.append({'record_index':records,'tool':tool if tool in allowed else 'other_tool',
    'success':ctx.get('success') if type(ctx.get('success'))is bool else None,
    'elapsed_ms':elapsed if type(elapsed)in(int,float) and math.isfinite(elapsed) and 0<=elapsed<=3600000 else None,
    'explicit_start_field_present':any(key in ctx for key in ('started_at','start_time','start_timestamp')),
    'command_category_field_present':'command_category' in ctx})
print(json.dumps({'native_log_files':files,'native_json_records':records,'completion_event_count':len(rows),
 'events':rows[-128:],'events_omitted':max(0,len(rows)-128),'bounded_scan_truncated':truncated}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--trial-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.container_id) or not re.fullmatch('grok-tune-mjcf-[0-9]{2}', args.trial_name):
        parser.error('exact owned container and trial label required')
    root = Path(__file__).resolve().parent.parent / args.trial_name / 'jobs/supervisor-full-tune-mjcf'
    trials = [path for path in root.glob('tune-mjcf__*') if path.is_dir()]
    if len(trials) != 1:
        raise SystemExit('one exact trial directory required')
    name = subprocess.check_output(['docker', 'inspect', '--format', '{{.Name}}',
        args.container_id], text=True, timeout=10).strip()
    if name != '/' + trials[0].name.lower() + '__env-main-1':
        raise SystemExit('exact owned container required')
    observed = json.loads(subprocess.check_output(['docker', 'exec', args.container_id,
        'python3', '-I', '-c', REMOTE], text=True, timeout=15))
    terminal = [row for row in observed['events'] if row['tool'] in {'run_terminal_cmd', 'run_terminal_command'}]
    elapsed = [row['elapsed_ms'] for row in terminal if row['elapsed_ms'] is not None]
    result = dict(schema='grok-closed-native-tool-completions@1', trial_name=args.trial_name,
        observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        observation=observed, native_terminal_alias_observed=(
            'run_terminal_command' if any(row['tool'] == 'run_terminal_command' for row in terminal) else None),
        terminal_completion_events=len(terminal),
        terminal_success_events=sum(row['success'] is True for row in terminal),
        terminal_failure_events=sum(row['success'] is False for row in terminal),
        terminal_max_completed_elapsed_ms=max(elapsed) if elapsed else None,
        terminal_last_completed_elapsed_ms=terminal[-1]['elapsed_ms'] if terminal else None,
        terminal_active_at_provider_deadline='not_established_from_completion_events',
        terminal_command_category='unknown',
        command_arguments_or_messages_inspected=False,
        raw_task_model_verifier_or_credential_data_exported=False,
        provider_calls=0, container_mutations=0, task_completion_authority=False)
    target = Path(__file__).resolve().parent / ('closed-tool-completions-' + args.trial_name + '.json')
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'observation'}, sort_keys=True))


if __name__ == '__main__':
    main()
