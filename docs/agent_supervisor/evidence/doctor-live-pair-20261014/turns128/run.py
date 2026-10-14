"""One frozen live pilot; distinct immutable step receipts, no automatic retry."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

A = Path(__file__).resolve().parent
R = json.loads((A / 'recipe.json').read_text())
W = Path(R['cwd'])

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def check_sources():
    assert sha(A / 'environment.json') == R['environment_sha256']
    for row in R['roots'].values():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=row['path'], text=True).strip() == row['commit']
        assert not subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=row['path'], text=True)
    for path, digest in R['host_source_sha256'].items():
        assert sha(path) == digest, path

def environment():
    value = json.loads((A / 'environment.json').read_text())
    env = dict(os.environ)
    for key in value['scrub_inherited_names']:
        env.pop(key, None)
    env.update(value['environment_overrides'])
    return env

def binding(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)

def verify():
    directory = A / 'runtime-01'
    manifest_path, archive = directory / 'manifest.json', directory / 'runtime.tar.gz'
    manifest = json.loads(manifest_path.read_text())
    assert sha(archive) == manifest['archive_sha256']
    rows = {row['path']: row for row in manifest['files']}
    assert len(rows) == len(manifest['files'])
    seen, total = set(), 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            assert member.name in rows and member.name not in seen and member.isfile()
            row = rows[member.name]
            assert member.size == row['bytes'] and member.mode == row['mode']
            with stream.extractfile(member) as content:
                assert hashlib.file_digest(content, 'sha256').hexdigest() == row['sha256']
            seen.add(member.name)
            total += member.size
    assert seen == set(rows)
    env = environment()
    os.environ.clear()
    os.environ.update(env)
    sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_coding_policy import require_doctor_coding_archive
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
    from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import require_worker_capability
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import require_runtime_provider_profile
    from benchmarks.agent_supervisor.container_coding.terminal_deployment import validate_source384_binding, validate_torch_cpu_wheel_binding
    require_doctor_coding_archive(manifest, 2, 'doctor-coding-two-attempts@1')
    require_source_view_archive(manifest, 'identity@1', 'source384-5cpu-20gib-formalization@1')
    require_worker_capability(manifest, 'source384-5cpu-20gib-formalization@1')
    require_runtime_provider_profile(manifest, 'grok-4.7-cli-1.0.46@1')
    assert validate_source384_binding(manifest) is not None
    assert validate_torch_cpu_wheel_binding(manifest) is not None
    return dict(archive=binding(archive), manifest=binding(manifest_path), all_archive_members_match=True,
        file_count=len(seen), uncompressed_bytes=total, current_doctor_policy_sources_match=True,
        provider_calls=0)

def run(step):
    assert step in ('bundle', 'verify', 'prepare', 'execute', 'collect')
    check_sources()
    assert not (A / (step + '-start.json')).exists(), 'immutable step already started'
    if step in ('prepare', 'execute', 'collect'):
        verified = json.loads((A / 'verify-result.json').read_text())
        for key in ('archive', 'manifest'):
            assert binding(verified[key]['path']) == verified[key]
    if step == 'execute':
        reporting_path = A / 'reporting-freeze.json'
        reporting = json.loads(reporting_path.read_text())
        for row in reporting['files']:
            assert binding(row['path']) == row
        readiness_path = A / 'provider-readiness/grok-metadata-precheck.json'
        readiness = json.loads(readiness_path.read_text())
        assert readiness['provider_generation_calls'] == 0 and readiness['host_auth_unchanged'] is True
        assert readiness['container_and_tmpfs_removed'] is True and readiness['host_temporary_auth_copy_removed'] is True
        assert readiness['observation']['status'] == 'ready' and readiness['observation']['selected_model_observed'] is True
        out = Path(R['output'])
        assert not (out / 'invocation.json').exists(), 'no replay or automatic retry'
        prep, config = out / 'preparation.json', out / 'config.json'
        prepared = json.loads(prep.read_text())
        assert prepared['prepared'] and prepared['config_sha256'] == sha(config)
        assert prepared['candidate_rounds'] == 2 and prepared['doctor_coding_policy'] == 'doctor-coding-two-attempts@1'
        assert prepared['archive_sha256'] == verified['archive']['sha256']
        write(A / 'launch-lock.json', dict(preparation=binding(prep), configuration=binding(config),
            runtime=verified, recipe=binding(A / 'recipe.json'), runner=binding(Path(__file__)),
            reporting=binding(reporting_path), readiness=binding(readiness_path),
            conditional_second_attempt=True, automatic_retries=0, provider_fallback=False))
    if step == 'collect':
        lock = json.loads((A / 'launch-lock.json').read_text())
        for key in ('preparation', 'configuration', 'recipe', 'runner', 'reporting', 'readiness'):
            assert binding(lock[key]['path']) == lock[key]
        for row in json.loads(Path(lock['reporting']['path']).read_text())['files']:
            assert binding(row['path']) == row
    free = shutil.disk_usage(A).free
    if step in ('bundle', 'execute'):
        assert free >= R['minimum_free_bytes'], 'physical free-space prerequisite'
    write(A / (step + '-start.json'), dict(step=step, started_at=datetime.now(timezone.utc).isoformat(),
        runner_sha256=sha(Path(__file__)), recipe_sha256=sha(A / 'recipe.json'), free_bytes=free,
        argv=R['commands'].get(step), cwd=str(W)))
    print(json.dumps(dict(step=step, status='started')), flush=True)
    before = time.monotonic()
    if step == 'verify':
        result, code = verify(), 0
    else:
        with (A / (step + '.stdout')).open('xb') as out, (A / (step + '.stderr')).open('xb') as err:
            process = subprocess.run(R['commands'][step], cwd=W, env=environment(), stdin=subprocess.DEVNULL,
                stdout=out, stderr=err)
        code = process.returncode
        result = dict(stdout=binding(A / (step + '.stdout')), stderr=binding(A / (step + '.stderr')))
    check_sources()
    result.update(step=step, exit_code=code, seconds=time.monotonic()-before, source_pins_unchanged=True)
    write(A / (step + '-result.json'), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
