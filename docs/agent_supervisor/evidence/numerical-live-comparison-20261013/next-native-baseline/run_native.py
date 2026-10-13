"""Launch one independently frozen native Codex baseline; retain private logs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

R = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock_path = R / 'execution-lock.json'
lock = json.loads(lock_path.read_bytes())
assert lock['schema'] == 'numerical-native-baseline-execution-lock@1'
assert lock['frozen_before_launch'] is True
for path, digest in lock['files'].items():
    assert sha(Path(path)) == digest, path
assert str(Path(__file__).resolve()) in lock['files']
recipe = json.loads((R / 'recipe.json').read_bytes())
env_record = json.loads((R / 'environment.json').read_bytes())
assert sha(R / 'environment.json') == recipe['environment_sha256']
trial = R / 'native-codex-01'
command = recipe['execute_argv']
assert command[-3:] == ['execute', '--output', str(trial)]
environment = dict(os.environ)
for key in ('OPENAI_API_KEY', 'XAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN',
            'OPENAI_BASE_URL', 'OPENAI_API_BASE'):
    environment.pop(key, None)
environment.update(env_record['environment_overrides'])
record = dict(schema='numerical-native-baseline-launch@1', argv=command,
    cwd=recipe['cwd'], execution_lock_sha256=sha(lock_path),
    script_sha256=sha(Path(__file__)), recipe_sha256=sha(R / 'recipe.json'),
    environment_sha256=sha(R / 'environment.json'),
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (trial / 'launch-start.json').open('x') as stream:
    json.dump(record, stream, indent=2)
start = time.monotonic()
print(json.dumps(dict(arm='native-codex', status='launching', execution_lock_sha256=sha(lock_path))), flush=True)
with (trial / 'execute.stdout').open('xb') as out, (trial / 'execute.stderr').open('xb') as err:
    completed = subprocess.run(command, cwd=recipe['cwd'], env=environment, stdout=out, stderr=err)
receipt_path = trial / 'receipt.json'
receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
result = dict(schema='numerical-native-baseline-launch-result@1', exit_code=completed.returncode,
    seconds=time.monotonic()-start, stdout_sha256=sha(trial / 'execute.stdout'),
    stderr_sha256=sha(trial / 'execute.stderr'), receipt_sha256=sha(receipt_path) if receipt else None,
    original_task_inputs_unchanged=receipt.get('original_task_inputs_unchanged'),
    complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'),
    trial_count=receipt.get('trial_count'),
    results=[dict(reward=t.get('reward'), exception_type=t.get('exception_type')) for t in receipt.get('trials', [])])
with (trial / 'launch-result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result), flush=True)
raise SystemExit(completed.returncode)
