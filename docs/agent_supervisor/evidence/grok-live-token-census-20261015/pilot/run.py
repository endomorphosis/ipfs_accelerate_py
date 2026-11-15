"""Public-validation diagnostic pilot; immutable steps, no retries or fallback."""
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
    assert R['status'] == 'frozen_after_root_commit'
    frozen = json.loads((A / 'source-freeze.json').read_text())
    assert frozen['root_authorized_source_freeze'] is True
    assert frozen['source_commit'] == R['roots']['source']['commit']
    for key, name in (('recipe', 'recipe.json'), ('runner', 'run.py'),
                      ('qualification', 'qualification.json'), ('reporting', 'reporting-freeze-v2.json')):
        assert binding(A / name) == frozen[key]
    qualification = json.loads((A / 'qualification.json').read_text())
    assert qualification['schema'] == 'grok-coding-route-qualification@1'
    assert qualification['passed_cases'] == 415
    for relative, digest in qualification['source_sha256'].items():
        assert sha(W / relative) == digest
    previous_path = Path(R['previous_run']['cleanup'])
    assert any(row == binding(previous_path) for row in R['input_bindings'])
    previous = json.loads(previous_path.read_text())
    assert previous['container_cleanup_observed'] is True and previous['remaining_container_ids'] == []
    assert json.loads((previous_path.parent / 'collect-result.json').read_text())['exit_code'] == 0
    assert sha(A / 'environment.json') == R['environment_sha256']
    for row in R['input_bindings']:
        assert binding(row['path']) == row
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
    directory = Path(R['runtime_directory'])
    manifest_path, archive = directory / 'manifest.json', directory / 'runtime.tar.gz'
    manifest = json.loads(manifest_path.read_text())
    assert sha(archive) == manifest['archive_sha256']
    rows = {row['path']: row for row in manifest['files']}
    assert len(rows) == len(manifest['files'])
    repaired_files = {}
    for relative in ('ipfs_accelerate_py/llm_router.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
            'ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py',
            'ipfs_accelerate_py/cli_runtime/grok_stream_capture.py',
            'ipfs_accelerate_py/agent_supervisor/todo_daemon/bridge_failure_diagnostics.py',
            'benchmarks/agent_supervisor/container_coding/terminal_context_audit.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/candidate_public_validation.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/candidate_execution.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/candidate_round_lifecycle.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/candidate_round_policy.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py',
            'ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py'):
        host = W / relative
        retained = rows['source/' + relative]
        assert retained['sha256'] == sha(host) and retained['bytes'] == host.stat().st_size
        repaired_files[relative] = dict(host=binding(host), archive_member=retained)
    excluded_host_tests = []
    for path, digest in R['host_source_sha256'].items():
        host = Path(path)
        relative = host.relative_to(W)
        member_name = 'source/' + str(relative)
        assert sha(host) == digest
        # Match terminal_deployment.py's exact immediate benchmark-test exclusion.
        if (relative.parent == Path('benchmarks/agent_supervisor/container_coding')
                and relative.name.startswith('test_') and relative.suffix == '.py'):
            assert member_name not in rows
            excluded_host_tests.append(str(relative))
            continue
        retained = rows[member_name]
        assert retained['sha256'] == digest and retained['bytes'] == host.stat().st_size
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
    require_doctor_coding_archive(manifest, 2, 'doctor-coding-public-validation-two-attempts@1')
    require_source_view_archive(manifest, 'identity@1', 'source384-5cpu-20gib-formalization@1')
    require_worker_capability(manifest, 'source384-5cpu-20gib-formalization@1')
    require_runtime_provider_profile(manifest, 'grok-4.7-cli-1.0.46@1')
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import require_grok_coding_archive
    require_grok_coding_archive(manifest, 8)
    assert validate_source384_binding(manifest) is not None
    assert validate_torch_cpu_wheel_binding(manifest) is not None
    return dict(archive=binding(archive), manifest=binding(manifest_path), all_archive_members_match=True,
        file_count=len(seen), uncompressed_bytes=total, current_doctor_policy_sources_match=True,
        excluded_host_benchmark_test_files=excluded_host_tests,
        repaired_production_files_match=repaired_files,
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
        reporting_path = A / 'reporting-freeze-v2.json'
        reporting = json.loads(reporting_path.read_text())
        for row in reporting['files']:
            assert binding(row['path']) == row
        readiness_path = Path(R['readiness']['path'])
        assert binding(readiness_path) == R['readiness']
        previous = json.loads(Path(R['previous_run']['cleanup']).read_text())
        assert previous['container_cleanup_observed'] is True and previous['remaining_container_ids'] == []
        frozen = json.loads((A / 'prepared-lock.json').read_text())
        for row in frozen['files']:
            assert binding(row['path']) == row
        readiness = json.loads(readiness_path.read_text())
        assert readiness['provider_generation_calls'] == 0 and readiness['host_auth_unchanged'] is True
        assert readiness['container_and_tmpfs_removed'] is True and readiness['host_temporary_auth_copy_removed'] is True
        assert readiness['observation']['status'] == 'ready' and readiness['observation']['selected_model_observed'] is True
        out = Path(R['output'])
        assert not (out / 'invocation.json').exists(), 'no replay or automatic retry'
        prep, config = out / 'preparation.json', out / 'config.json'
        prepared = json.loads(prep.read_text())
        assert prepared['prepared'] and prepared['config_sha256'] == sha(config)
        assert prepared['candidate_rounds'] == 2 and prepared['doctor_coding_policy'] == 'doctor-coding-public-validation-two-attempts@1'
        assert prepared['archive_sha256'] == verified['archive']['sha256']
        assert prepared['grok_coding_max_turns'] == 8
        config_value = json.loads(config.read_text())
        assert config_value['n_attempts'] == config_value['n_concurrent_trials'] == 1
        assert config_value['retry']['max_retries'] == 0
        controls = config_value['agents'][0]['kwargs']
        assert controls['grok_coding_max_turns'] == 8
        assert controls['planning_mode'] == R['limits']['planning_mode'] == 'declared-contract@1'
        assert controls['arm'] == 'full' and controls['source_view_profile'] == 'identity@1'
        assert controls.get('source_edit_mode', 'disabled') == 'disabled'
        assert controls.get('numerical_formalization_profile') is None
        assert controls['doctor_coding_policy'] == R['limits']['coding_policy']
        assert controls['task_profile'] == json.loads((A/'largest-eigenval-public-profile.json').read_text())
        assert controls['task_profile']['schema'] == 'terminal-public-task-profile@4'
        assert controls['task_profile']['public_validation_profile'] == R['limits']['public_validation_profile']
        assert controls['provider_profile'] == 'grok-4.7-cli-1.0.46@1'
        assert controls['resource_profile'] == 'source384-5cpu-20gib-formalization@1'
        assert config_value['environment']['override_cpus'] == 5
        assert config_value['environment']['override_memory_mb'] == 20480
        assert R['limits']['coding_deadline_seconds'] == 1800
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
