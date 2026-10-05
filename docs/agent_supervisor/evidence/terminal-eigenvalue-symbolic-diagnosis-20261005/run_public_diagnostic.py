"""Launch exactly one frozen public diagnostic; retain bounded runtime evidence."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parent
PUBLIC = Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval')
NAME = 'ipfs-eigen-diagnostic-20261005-replay02'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def docker(*args, check=True):
    return subprocess.run(['docker', *args], capture_output=True, text=True,
                          check=check, timeout=30)


def binding(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    receipt_path = ROOT / 'runtime-receipt.json'
    assert not receipt_path.exists(), 'one-run receipt already exists'
    assert (ROOT / 'output').is_dir()
    frozen = json.loads((ROOT / 'frozen-inputs.json').read_text())
    initial_inputs = [binding(row['path']) for row in frozen['inputs']]
    assert initial_inputs == frozen['inputs'], 'frozen inputs changed'
    image_id = (ROOT / 'rebuilt-image.id').read_text().strip()
    image = json.loads(docker('image', 'inspect', image_id).stdout)[0]
    base = json.loads(docker('image', 'inspect',
        'sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed').stdout)[0]
    assert image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] == base['RootFS']['Layers']
    receipt = {
        'schema': 'public-eigenvalue-diagnostic-runtime@1',
        'created_at_utc': now(),
        'interpretation': 'One public diagnostic in a rebuilt runtime; not an official benchmark rerun or exact original-environment reproduction.',
        'model_calls': 0, 'official_verifier_calls': 0,
        'candidate_modifications': 0, 'optimization_iterations': 0,
        'original_image_available': False,
        'replacement_image': {key: image.get(key) for key in ('Id', 'Architecture', 'Os', 'Created', 'RepoDigests')},
        'cached_base_id': base['Id'],
        'replacement_rootfs_extends_inspected_cached_base': True,
        'input_bindings_before': initial_inputs,
        'producer_bindings': [binding(__file__), binding(ROOT / 'frozen-inputs.json')],
        'hidden_inputs_mounted': False, 'credentials_mounted': False,
    }
    command = [
        'create', '--name', NAME, '--label', 'purpose=public-eigenvalue-diagnostic-20261005-replay02',
        '--cpus', '5', '--memory', '16g', '--memory-swap', '32g',
        '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,exec,nosuid,size=512m',
        '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '--mount', f'type=bind,src={ARTIFACT / ".private/native-eigen-final.py"},dst=/app/eigen.py,readonly',
        '--mount', f'type=bind,src={PUBLIC / "environment/src/eval.py"},dst=/app/eval.py,readonly',
        '--mount', f'type=bind,src={PUBLIC / "instruction.md"},dst=/diagnostic/instruction.md,readonly',
        '--mount', f'type=bind,src={ROOT / "replay.py"},dst=/diagnostic/replay.py,readonly',
        '--mount', f'type=bind,src={ROOT / "plan.json"},dst=/diagnostic/plan.json,readonly',
        '--mount', f'type=bind,src={ROOT / "output"},dst=/diagnostic/output',
        image_id, 'python', '/diagnostic/replay.py',
    ]
    container_id = docker(*command).stdout.strip()
    receipt['container_id'] = container_id
    receipt['container_name'] = NAME
    receipt['create_command'] = ['docker', *command]
    try:
        inspect = json.loads(docker('inspect', container_id).stdout)[0]
        receipt['actual_before'] = {
            'Image': inspect['Image'],
            'host_config': {key: inspect['HostConfig'].get(key) for key in (
                'NanoCpus', 'Memory', 'MemorySwap', 'CpuQuota', 'CpuPeriod',
                'CpusetCpus', 'NetworkMode', 'ReadonlyRootfs', 'Tmpfs')},
            'mounts': [{key: row.get(key) for key in ('Type', 'Source', 'Destination', 'RW')}
                       for row in inspect['Mounts']],
        }
        assert inspect['Image'] == image_id
        assert inspect['HostConfig']['NanoCpus'] == 5_000_000_000
        assert inspect['HostConfig']['Memory'] == 17_179_869_184
        assert inspect['HostConfig']['MemorySwap'] == 34_359_738_368
        assert inspect['HostConfig']['NetworkMode'] == 'none'
        receipt['started_at_utc'] = now()
        write(receipt_path, receipt)
        started = time.perf_counter()
        with (ROOT / 'run.log').open('wb') as log:
            try:
                result = subprocess.run(['docker', 'start', '-a', container_id],
                    stdout=log, stderr=subprocess.STDOUT, timeout=360)
                receipt['attach_returncode'] = result.returncode
                receipt['host_watchdog_expired'] = False
            except subprocess.TimeoutExpired:
                receipt['attach_returncode'] = None
                receipt['host_watchdog_expired'] = True
        receipt['outer_seconds'] = time.perf_counter() - started
        after = json.loads(docker('inspect', container_id).stdout)[0]
        receipt['actual_after_state'] = {key: after['State'].get(key) for key in (
            'Status', 'Running', 'ExitCode', 'OOMKilled', 'StartedAt', 'FinishedAt', 'Error')}
        receipt['output_bindings'] = [binding(path) for path in sorted((ROOT / 'output').glob('*.json'))]
        receipt['run_log_binding'] = binding(ROOT / 'run.log')
    finally:
        removed = docker('rm', '-f', container_id, check=False)
        receipt['container_cleanup_returncode'] = removed.returncode
        absent = docker('inspect', container_id, check=False)
        receipt['owned_container_absent_after_cleanup'] = absent.returncode != 0 and 'no such object' in absent.stderr.lower()
        receipt['input_bindings_after'] = [binding(row['path']) for row in frozen['inputs']]
        receipt['frozen_inputs_unchanged'] = receipt['input_bindings_after'] == initial_inputs
        receipt['finished_at_utc'] = now()
        write(receipt_path, receipt)
    print(json.dumps({'runtime_receipt': str(receipt_path),
        'returncode': receipt.get('attach_returncode'),
        'seconds': receipt.get('outer_seconds'),
        'container_removed': receipt['owned_container_absent_after_cleanup']}))


if __name__ == '__main__':
    main()
