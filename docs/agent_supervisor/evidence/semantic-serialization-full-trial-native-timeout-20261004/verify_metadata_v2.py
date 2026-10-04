"""Independently verify the bounded v2 projection from whitelisted outer metadata.

Never follow receipt paths or export raw messages, source, verifier, auth or model
bodies. The already-sanitized production traceback is bounded to 20 frames.
"""
from pathlib import Path
import hashlib
import json
import math
import re

OUT = Path(__file__).resolve().parent
INPUT = OUT / 'trial-01/receipt.json'
TARGET = OUT / 'full-trial-metadata-v2.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, bound):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= bound
    return json.loads(path.read_text())


def scalar(value):
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float) and math.isfinite(value):
        return value
    if type(value) is str and len(value) <= 200 and re.fullmatch(r'[A-Za-z0-9_:/@.+=-]*', value):
        return value
    return '[excluded_non_scalar_or_body]'


def pick(value, fields):
    return {key: scalar(value[key]) for key in fields if type(value) is dict and key in value}


def numbers(value):
    if type(value) is not dict or len(value) > 64:
        return {}
    return {key: item for key, item in value.items() if type(key) is str
        and re.fullmatch(r'[A-Za-z0-9_]{1,80}', key) and type(item) in (int, float)
        and math.isfinite(item)}


receipt = read(INPUT, 16 * 1024 * 1024)
result = dict(schema='bounded-full-trial-metadata-inspection@1',
    receipt=pick(receipt, ('schema', 'arm', 'task', 'trial_count', 'parallel_workers', 'harbor_returncode',
        'invocation_seconds', 'original_task_inputs_unchanged', 'planning_strategy', 'benchmark_advantage_claimed')),
    exported_verifier_bodies=False, exported_model_bodies=False, exported_credentials=False, trials=[])
assert type(receipt.get('trials')) is list and len(receipt['trials']) == 1
for trial in receipt['trials']:
    supervisor = trial['supervisor']
    providers = supervisor.get('provider_invocations')
    assert type(providers) is list
    doctors = supervisor.get('doctor_invocations')
    assert doctors is None or type(doctors) is list
    row = dict(trial=scalar(trial.get('trial')), reward=numbers(trial.get('reward')),
        durations_seconds=numbers(trial.get('durations_seconds')),
        supervisor=pick(supervisor, ('schema', 'arm', 'task_completed', 'production_activation',
            'benchmark_advantage_claimed', 'max_total_agent_seconds', 'reserved_cleanup_seconds',
            'work_cutoff_seconds', 'seconds', 'error_phase', 'error_type', 'worker_cleanup_returncode')),
        phases=numbers(supervisor.get('phases')),
        provider_invocation_count=len(providers),
        doctor_invocation_count=len(doctors) if doctors is not None else None)
    row['planning'] = pick(supervisor.get('planning'), ('qualified', 'goals', 'tasks', 'provider_calls', 'elapsed_seconds', 'planning_strategy'))
    doctor = supervisor.get('doctor_dispatch', {})
    row['doctor'] = pick(doctor, ('status', 'route', 'provider_calls', 'task_cid'))
    workflow = doctor.get('contract_workflow', {})
    row['contract'] = pick(workflow, ('status', 'whole_program_proved', 'canonical_source_edits', 'publication_authority', 'completion_authority'))
    row['proof'] = pick(workflow.get('proof'), ('status', 'whole_program_proved', 'security_ir_obligations_discharged', 'provider_calls'))
    row['proof_receipt'] = pick(workflow.get('proof', {}).get('proof'), ('disposition', 'receipt_id', 'property_id', 'write_authority', 'uniqueness_satisfied'))
    initial = supervisor.get('initial_context', {})
    row['initial_context'] = pick(initial, ('seconds', 'indexed_symbols', 'full_capsules'))
    row['initial_context_phases'] = numbers(initial.get('nonoverlapping_seconds'))
    row['source384'] = pick(initial.get('source384_context'), ('seconds', 'inference_executed', 'native_worker_executed'))
    row['failure_admission'] = pick(supervisor.get('failure_admission'), ('status', 'reason', 'error_type', 'observation_boundary', 'causal_proof'))
    row['failure_resources'] = numbers(supervisor.get('failure_resources'))
    admission = supervisor.get('failure_admission')
    if (type(admission) is dict and admission.get('schema') == 'terminal-admission-failure-observation@1'
            and len(json.dumps(admission).encode('utf-8')) <= 4096):
        row['failure_admission_observation'] = admission
    row['doctor_dispatch_present'] = 'doctor_dispatch' in supervisor
    row['supervisor']['error_type'] = scalar(supervisor.get('error', {}).get('type'))
    for key in ('start', 'stop', 'task_state', 'coding_dispatch_possible', 'context_refresh', 'implementation_route', 'remaining_processes'):
        value = supervisor.get(key)
        assert value is None or type(value) in (bool, int, float, str)
        row[key] = scalar(value)
    frames = supervisor.get('error_traceback', {}).get('frames')
    assert type(frames) is list and len(frames) <= 20
    row['traceback_frames'] = []
    for frame in frames:
        assert type(frame) is dict and set(frame) == {'file', 'function', 'line'}
        assert type(frame['line']) is int and 0 <= frame['line'] < 1000000
        assert type(frame['file']) is str and len(frame['file']) <= 1024
        assert frame['file'] == '<string>' or frame['file'].startswith('/opt/ipfs-supervisor/')
        assert type(frame['function']) is str and len(frame['function']) <= 100
        assert re.fullmatch(r'[A-Za-z0-9_<>]+', frame['function'])
        row['traceback_frames'].append(frame)
    result['trials'].append(row)

assert result == read(TARGET, 65536), 'v2 projection differs from independent whitelist projection'
verification = dict(schema='bounded-full-trial-v2-verification@1', valid=True,
    receipt_sha256=sha(INPUT), projection_sha256=sha(TARGET), verifier_sha256=sha(Path(__file__)),
    exact_whitelisted_projection_equal=True, frame_count=len(result['trials'][0]['traceback_frames']),
    doctor_dispatch_present=True, doctor_invocation_count=None,
    absent_doctor_list_not_treated_as_zero=True,
    original_inspector='Historical v1 producer; not authoritative for the corrected v2 projection.',
    no_verifier_auth_model_or_source_body_paths_followed=True)
with (OUT / 'metadata-v2-verification.json').open('x') as stream:
    stream.write(json.dumps(verification, indent=2, sort_keys=True) + '\n')
print(json.dumps({'valid': True, 'verification_sha256': sha(OUT / 'metadata-v2-verification.json')}))
