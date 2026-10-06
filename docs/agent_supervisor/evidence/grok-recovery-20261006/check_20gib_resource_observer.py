"""Synthetic admission-limit observations; no container or provider calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'grok-container/observe_20gib_resources.py'
spec = importlib.util.spec_from_file_location('resource_observer', SCRIPT)
observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)


def main():
    maximum = observer.MEMORY_BYTES
    docker = dict(running=True, memory_bytes=maximum, nano_cpus=5_000_000_000,
                  cpu_quota_us=0, cpu_period_us=0)
    group = dict(memory_max_bytes=maximum, memory_current_bytes=1000,
                 cpu_quota_us=500000, cpu_period_us=100000,
                 memory_high_kind='unlimited', memory_high_bytes=None)
    checks = []

    def check(name, result):
        assert result, name
        checks.append(name)

    check('unlimited high permits exact max', observer.resource_limits_match(docker, group))
    check('absent high permits exact max', observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'absent'}))
    check('equal finite high permits exact max', observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'finite', 'memory_high_bytes': maximum}))
    check('higher finite high permits exact max', observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'finite', 'memory_high_bytes': maximum + 1}))
    check('smaller finite high rejects effective limit', not observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'finite', 'memory_high_bytes': maximum - 1}))
    check('malformed high rejects qualification', not observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'invalid'}))
    check('missing high observation rejects qualification', not observer.resource_limits_match(docker, {key: value for key, value in group.items() if not key.startswith('memory_high')}))
    check('contradictory absent high rejects qualification', not observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'absent', 'memory_high_bytes': 1}))
    check('boolean finite high rejects qualification', not observer.resource_limits_match(docker, {**group, 'memory_high_kind': 'finite', 'memory_high_bytes': True}))
    check('lower max rejects qualification', not observer.resource_limits_match(docker, {**group, 'memory_max_bytes': maximum - 1}))
    check('different Docker memory rejects qualification', not observer.resource_limits_match({**docker, 'memory_bytes': maximum - 1}, group))
    check('different effective CPU quota rejects qualification', not observer.resource_limits_match(docker, {**group, 'cpu_quota_us': 400000}))
    check('Docker quota-period CPU representation supported', observer.resource_limits_match({**docker, 'nano_cpus': 0, 'cpu_quota_us': 500000, 'cpu_period_us': 100000}, group))
    result = dict(schema='explicit20gib-observer-safety-review@1', passed=True,
        check_count=len(checks), checks=checks, provider_calls=0, containers_launched=0,
        script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (ROOT / 'qualification/explicit20gib-observer-safety-review-01.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'provider_calls': 0}))


if __name__ == '__main__':
    main()
