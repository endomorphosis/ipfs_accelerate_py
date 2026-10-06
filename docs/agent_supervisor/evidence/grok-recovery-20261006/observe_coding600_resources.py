"""Bind a reviewed20GiB profile to preparation and observe its actual limits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
MEMORY_BYTES = 21_474_836_480
PROFILES = frozenset({"source384-5cpu-20gib-planner180@1", "source384-5cpu-20gib-coding600@1"})
REMOTE = '''from pathlib import Path
import json
p=Path('/sys/fs/cgroup')
def number(name):
 value=(p/name).read_text().strip()
 return int(value) if value.isdecimal() else None
cpu=(p/'cpu.max').read_text().split()
high_path=p/'memory.high'
high=high_path.read_text().strip() if high_path.is_file() else None
high_kind='absent' if high is None else 'unlimited' if high=='max' else 'finite' if high.isdecimal() else 'invalid'
print(json.dumps({'memory_max_bytes':number('memory.max'),'memory_current_bytes':number('memory.current'),
 'memory_high_kind':high_kind,'memory_high_bytes':int(high) if high_kind=='finite' else None,
 'cpu_quota_us':int(cpu[0]) if cpu[0].isdecimal() else None,
 'cpu_period_us':int(cpu[1]) if len(cpu)==2 and cpu[1].isdecimal() else None}))
'''


def resource_limits_match(observed, actual):
    docker_cpu = (observed.get('nano_cpus') == 5_000_000_000 or
        type(observed.get('cpu_period_us')) is int and observed['cpu_period_us'] > 0 and
        observed.get('cpu_quota_us') == 5 * observed['cpu_period_us'])
    cgroup_cpu = (type(actual.get('cpu_period_us')) is int and actual['cpu_period_us'] > 0
        and actual.get('cpu_quota_us') == 5 * actual['cpu_period_us'])
    high_kind, high_bytes = actual.get('memory_high_kind'), actual.get('memory_high_bytes')
    high_allows_limit = ((high_kind in {'absent', 'unlimited'} and high_bytes is None)
        or (high_kind == 'finite' and type(high_bytes) is int and high_bytes >= MEMORY_BYTES))
    return (observed.get('running') is True and observed.get('memory_bytes') == MEMORY_BYTES
        and actual.get('memory_max_bytes') == MEMORY_BYTES and high_allows_limit and docker_cpu and cgroup_cpu)



def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def prepared_profile_binding(root, trial, profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError('reviewed20GiB profile required')
    path = root / trial / 'preparation.json'
    if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('bounded canonical preparation metadata required')
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('preparation metadata too large')
    def invalid_constant(value):
        raise ValueError('nonfinite preparation metadata')
    prepared = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid_constant)
    if (type(prepared) is not dict or prepared.get('schema') != 'terminal-full-supervisor-preparation@1'
            or prepared.get('task') != 'tune-mjcf' or prepared.get('arm') != 'full'
            or prepared.get('resource_profile') != profile):
        raise ValueError('prepared task/resource selection differs')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), resource_profile=profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--resource-profile', choices=sorted(PROFILES), required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.container_id) or not re.fullmatch('[0-9]{2}', args.attempt):
        parser.error('exact container ID and attempt required')
    trial = 'grok-tune-mjcf-' + args.attempt
    preparation = prepared_profile_binding(A, trial, args.resource_profile)
    roots = list((A / trial / 'jobs/supervisor-full-tune-mjcf').glob('tune-mjcf__*'))
    roots = [path for path in roots if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit('one exact owned trial directory required')
    expected_name = '/' + roots[0].name.lower() + '__env-main-1'
    template = '{"name":{{json .Name}},"memory_bytes":{{json .HostConfig.Memory}},"nano_cpus":{{json .HostConfig.NanoCpus}},"cpu_quota_us":{{json .HostConfig.CpuQuota}},"cpu_period_us":{{json .HostConfig.CpuPeriod}},"running":{{json .State.Running}}}'
    observed = json.loads(subprocess.check_output(['docker', 'inspect', '--format', template, args.container_id], text=True, timeout=10))
    if observed.pop('name', None) != expected_name:
        raise SystemExit('container does not match exact selected trial')
    actual = json.loads(subprocess.check_output(['docker', 'exec', args.container_id, 'python3', '-I', '-c', REMOTE], text=True, timeout=10))
    qualified = resource_limits_match(observed, actual)
    result = dict(schema='grok-explicit-container-resource-observation@1', trial_name=trial,
        observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        resource_profile=args.resource_profile, preparation_binding=preparation,
        expected_memory_bytes=MEMORY_BYTES, expected_cpus=5,
        exact_owned_container_verified=True, docker=observed, cgroup=actual,
        qualified=qualified, provider_calls=0, container_mutations=0,
        source_or_credential_data_read=False, same_resource_comparison_claim=False,
        same_coding_budget_performance_comparison_claim=False)
    target = A / 'grok-container' / ('resource-observation-' + args.attempt + '.json')
    with target.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if qualified else 1)


if __name__ == '__main__':
    main()
