"""Execute exactly one arm from the prelaunch matched-pair lock."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
arm = sys.argv[1]
assert arm in ('full', 'no-index')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock_path = R / 'pair-lock.json'
lock = json.loads(lock_path.read_bytes())
assert lock['schema'] == 'numerical-live-supervisor-pair-lock@1'
assert lock['frozen_before_launch'] is True
for key, path in {'runner': Path(__file__), 'recipe': R / 'launch-recipe-candidates.json',
                  'environment': R / 'environment.json'}.items():
    binding = lock['launch_inputs'][key]
    assert str(path.resolve()) == binding['path']
    assert sha(path) == binding['sha256']
rows = [row for row in lock['attempts'] if row['arm'] == arm]
assert len(rows) == 1
row = rows[0]
trial = Path(row['directory'])
for name in ('preparation', 'configuration'):
    assert sha(Path(row[name]['path'])) == row[name]['sha256']
assert sha(R / 'collect_pair.py') == lock['implementations']['collector']['sha256']
recipe_path = R / 'launch-recipe-candidates.json'
recipe = json.loads(recipe_path.read_bytes())
route = lock['common']['provider_profile']
assert route == 'codex-gpt-6.1-sol-0.160.0@1'
command = recipe['commands_by_provider'][route][arm]['execute_argv']
assert command[-3:] == ['execute', '--output', str(trial)]
environment_path = R / 'environment.json'
assert sha(environment_path) == recipe['environment_manifest_sha256']
environment = dict(os.environ)
for key in ('OPENAI_API_KEY', 'XAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'):
    environment.pop(key, None)
environment.update(json.loads(environment_path.read_bytes())['environment_overrides'])
start = time.monotonic()
record = dict(schema='numerical-live-pair-launch@1', arm=arm, argv=command,
    cwd=recipe['cwd'], pair_lock_sha256=sha(lock_path),
    recipe_sha256=sha(recipe_path), script_sha256=sha(Path(__file__)),
    environment_manifest_sha256=sha(environment_path),
    preparation_sha256=row['preparation']['sha256'],
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    inherited_environment_values_retained=False)
with (trial / 'launch-start.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(dict(arm=arm, status='launching', pair_lock_sha256=record['pair_lock_sha256'])), flush=True)
with (trial / 'execute.stdout').open('xb') as out, (trial / 'execute.stderr').open('xb') as err:
    completed = subprocess.run(command, cwd=recipe['cwd'], env=environment, stdout=out, stderr=err)
receipt_path = trial / 'receipt.json'
receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
result = dict(schema='numerical-live-pair-launch-result@1', arm=arm,
    exit_code=completed.returncode, seconds=time.monotonic()-start,
    preparation_unchanged=sha(Path(row['preparation']['path'])) == row['preparation']['sha256'],
    stdout_sha256=sha(trial / 'execute.stdout'), stderr_sha256=sha(trial / 'execute.stderr'),
    receipt_sha256=sha(receipt_path) if receipt else None,
    complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'),
    trial_count=receipt.get('trial_count'),
    results=[dict(reward=t.get('reward'), exception_type=t.get('exception_type')) for t in receipt.get('trials', [])])
with (trial / 'launch-result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result), flush=True)
raise SystemExit(completed.returncode)
