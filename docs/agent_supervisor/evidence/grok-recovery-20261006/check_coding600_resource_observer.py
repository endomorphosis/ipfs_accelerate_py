"""Synthetic admission-limit observations; no container or provider calls."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'grok-container/observe_coding600_resources.py'
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
    def rejected(call):
        try:
            call()
        except ValueError:
            return True
        return False

    with tempfile.TemporaryDirectory(prefix='coding600-resource-observer-fixture-') as temporary:
        root = Path(temporary)
        trial = 'grok-tune-mjcf-06'
        folder = root / trial; folder.mkdir()
        path = folder / 'preparation.json'
        profile = 'source384-5cpu-20gib-coding600@1'
        prepared = dict(schema='terminal-full-supervisor-preparation@1', task='tune-mjcf', arm='full', resource_profile=profile)
        path.write_text(json.dumps(prepared))
        binding = observer.prepared_profile_binding(root, trial, profile)
        check('coding600 actual selection bound before observation', binding['resource_profile'] == profile and binding['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest())
        check('300 profile cannot label600 preparation', rejected(lambda:observer.prepared_profile_binding(root, trial, 'source384-5cpu-20gib-planner180@1')))
        check('16GiB profile rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, 'source384-5cpu-16gib-planner180@1')))
        check('malformed profile rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, {'private':'MARKER'})))
        for name, replacement in [('wrong task', {'task':'another-task'}), ('wrong arm', {'arm':'baseline'}), ('wrong schema', {'schema':'another-schema'})]:
            path.write_text(json.dumps({**prepared, **replacement}))
            check(name+' preparation rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, profile)))
        path.write_text(json.dumps(prepared)[:-1]+',"resource_profile":"'+profile+'"}')
        check('duplicate metadata rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, profile)))
        path.write_text(json.dumps(prepared)[:-1]+',"x":NaN}')
        check('nonfinite metadata rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, profile)))
        original = folder / 'original.json'; original.write_text(json.dumps(prepared))
        path.unlink(); path.symlink_to(original)
        check('symlink preparation rejected', rejected(lambda:observer.prepared_profile_binding(root, trial, profile)))
        path.unlink(); path.write_text(' ' * (8 * 1024 * 1024 + 1))
        check('oversize metadata rejected before parsing', rejected(lambda:observer.prepared_profile_binding(root, trial, profile)))
        original_profile = 'source384-5cpu-20gib-planner180@1'
        path.write_text(json.dumps({**prepared, 'resource_profile':original_profile}))
        check('original20GiB profile remains distinct supported selection', observer.prepared_profile_binding(root, trial, original_profile)['resource_profile'] == original_profile)
    result = dict(schema='coding600-observer-safety-review@1', passed=True,

        check_count=len(checks), checks=checks, provider_calls=0, containers_launched=0,
        script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (ROOT / 'qualification/coding600-observer-safety-review-01.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'provider_calls': 0}))


if __name__ == '__main__':
    main()
