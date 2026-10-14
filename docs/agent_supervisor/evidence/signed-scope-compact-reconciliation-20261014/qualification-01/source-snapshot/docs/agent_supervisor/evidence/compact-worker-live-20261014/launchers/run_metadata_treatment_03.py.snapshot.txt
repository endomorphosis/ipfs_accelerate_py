"""Verify or execute the separately released metadata treatment, without retries."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

A = Path(__file__).resolve().parent
LABEL = 'comments-metadata'

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'execute'])
    args = parser.parse_args()
    lock_path = A / 'metadata-execution-lock-03.json'
    lock = read(lock_path)
    assert lock['schema'] == 'compact-metadata-live-execution-lock@1'
    assert lock['frozen_before_launch'] is True and lock['label'] == LABEL
    for row in lock['files'].values():
        assert sha(row['path']) == row['sha256']
    files = lock['files']
    assert files['runner']['path'] == str(Path(__file__).resolve())
    release = read(files['release']['path'])
    assert release['schema'] == 'compact-metadata-treatment-preparation-release@1'
    assert release['label'] == LABEL
    for row in [*release['files'].values(), release['archive']]:
        assert sha(row['path']) == row['sha256']
    for path, digest in release['host_source_sha256'].items():
        assert sha(path) == digest
    for key in ['environment', 'recipe', 'preparation', 'configuration', 'runtime_manifest']:
        assert files[key] == release['files'][key]
    assert files['archive'] == release['archive']
    baseline = read(files['baseline_launch_result']['path'])
    assert baseline['schema'] == 'repeated-compact-live-launch-result@1'
    assert baseline['label'] == 'comments' and baseline['arm'] == 'full'
    assert baseline['complete_single_trial_receipt'] is True
    assert baseline['preparation_unchanged'] is True and baseline['configuration_unchanged'] is True
    assert baseline['host_source_unchanged'] is True
    recipe = read(files['recipe']['path'])
    trial = Path(release['treatment_directory'])
    command = recipe['arm']['execute_argv']
    assert command == release['execute_argv']
    assert command[-3:] == ['execute', '--output', str(trial)]
    assert recipe['cwd'] == release['execute_cwd']
    assert files['preparation']['path'] == str(trial / 'preparation.json')
    assert files['configuration']['path'] == str(trial / 'config.json')
    assert not any((trial / name).exists() for name in ['launch-start.json', 'invocation.json', 'receipt.json'])
    if args.action == 'verify':
        print(json.dumps(dict(status='ready_no_execution', label=LABEL,
                             execution_lock_sha256=sha(lock_path), provider_calls=0)))
        return 0
    configuration = read(files['environment']['path'])
    env = dict(os.environ)
    for name in configuration['scrub_inherited_names']:
        env.pop(name, None)
    env.update(configuration['environment_overrides'])
    start = dict(schema='repeated-compact-live-launch@1', label=LABEL, arm='full',
                 argv=command, cwd=recipe['cwd'], execution_lock_sha256=sha(lock_path),
                 metadata_release_sha256=files['release']['sha256'],
                 baseline_pair_lock_sha256=files['baseline_pair_lock']['sha256'],
                 runner_sha256=sha(__file__), recipe_sha256=files['recipe']['sha256'],
                 environment_manifest_sha256=files['environment']['sha256'],
                 preparation_sha256=files['preparation']['sha256'],
                 configuration_sha256=files['configuration']['sha256'],
                 started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 inherited_environment_values_retained=False)
    write_new(trial / 'launch-start.json', start)
    print(json.dumps(dict(label=LABEL, status='launching', execution_lock_sha256=start['execution_lock_sha256'])), flush=True)
    begin = time.monotonic()
    with (trial / 'execute.stdout').open('xb') as out, (trial / 'execute.stderr').open('xb') as err:
        completed = subprocess.run(command, cwd=recipe['cwd'], env=env, stdout=out, stderr=err)
    receipt_path = trial / 'receipt.json'
    receipt = read(receipt_path) if receipt_path.is_file() else {}
    result = dict(schema='repeated-compact-live-launch-result@1', label=LABEL, arm='full',
                  exit_code=completed.returncode, seconds=time.monotonic() - begin,
                  execution_lock_sha256=sha(lock_path), metadata_release_sha256=files['release']['sha256'],
                  preparation_unchanged=sha(files['preparation']['path']) == files['preparation']['sha256'],
                  configuration_unchanged=sha(files['configuration']['path']) == files['configuration']['sha256'],
                  host_source_unchanged=all(sha(p) == h for p, h in release['host_source_sha256'].items()),
                  stdout_sha256=sha(trial / 'execute.stdout'), stderr_sha256=sha(trial / 'execute.stderr'),
                  receipt_sha256=sha(receipt_path) if receipt else None,
                  complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'),
                  trial_count=receipt.get('trial_count'),
                  results=[dict(reward=t.get('reward'), exception_type=t.get('exception_type'))
                           for t in receipt.get('trials', [])])
    write_new(trial / 'launch-result.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return completed.returncode

if __name__ == '__main__':
    raise SystemExit(main())
