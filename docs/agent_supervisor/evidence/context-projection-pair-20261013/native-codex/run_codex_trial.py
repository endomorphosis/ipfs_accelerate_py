"""Execute one already-frozen native trial; retain logs without echoing bodies."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).parent.resolve()
manifest_path = root / 'live-preparation' / sys.argv[1]
manifest = json.loads(manifest_path.read_bytes())
name = sys.argv[2]
assert name in manifest['commands'] and Path(name).name == name
trial = root / 'live-trials' / name
prepared_path = trial / 'preparation.json'
prepared = json.loads(prepared_path.read_bytes())
assert prepared['prepared'] and prepared['provider_profile'] == 'codex-gpt-6.1-sol-0.160.0@1'
assert prepared['source_view_profile'] == 'identity@1'
assert prepared.get('semantic_metadata_view', 'legacy') in {'legacy', 'common-bindings@1'}
argv = manifest['commands'][name]['execute_argv']
assert argv[-2:] == ['--output', str(trial)] and argv[-3] == 'execute'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(prepared_path)
record = dict(schema='source-context-native-launch@1', command=argv,
    cwd=manifest['cwd'], command_manifest_sha256=sha(manifest_path),
    command_manifest=str(manifest_path), preparation_sha256=before,
    script_sha256=sha(Path(__file__)), started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    provider_profile=prepared['provider_profile'], source_view_profile=prepared['source_view_profile'],
    semantic_metadata_view=prepared.get('semantic_metadata_view', 'legacy'),
    inherited_environment_values_retained=False)
with (trial / 'launch.json').open('x') as stream:
    json.dump(record, stream, indent=2)
environment = os.environ.copy()
environment.update(manifest['environment_overrides'])
started = time.monotonic()
with (trial / 'execute.stdout').open('xb') as stdout, (trial / 'execute.stderr').open('xb') as stderr:
    completed = subprocess.run(argv, cwd=manifest['cwd'], env=environment, stdout=stdout, stderr=stderr)
receipt_path = trial / 'receipt.json'
receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
summary = dict(schema='source-context-native-launch-result@1', name=name,
    exit_code=completed.returncode, seconds=time.monotonic() - started,
    preparation_unchanged=sha(prepared_path) == before,
    stdout_sha256=sha(trial / 'execute.stdout'), stderr_sha256=sha(trial / 'execute.stderr'),
    receipt_sha256=sha(receipt_path) if receipt else None,
    complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'),
    trial_count=receipt.get('trial_count'),
    results=[dict(trial=r.get('trial'), reward=r.get('reward'), exception_type=r.get('exception_type'))
             for r in receipt.get('trials', [])])
(trial / 'launch-result.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
raise SystemExit(completed.returncode)
