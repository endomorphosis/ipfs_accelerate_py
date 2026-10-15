"""Immutable public-input coding diagnostic, disabled until qualified freeze.

Direct Task lifecycle with official verification disabled and one coding
invocation of at most eight turns. No automatic retry or provider fallback.
Raw owner output stays private; only closed telemetry is exported.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

A = Path(__file__).resolve().parent
os.umask(0o077)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_bytes())


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def environment():
    manifest = read(A / 'environment.json')
    env = dict(os.environ)
    for key in manifest['scrub_inherited_names']:
        env.pop(key, None)
    env.update(manifest['environment_overrides'])
    return env


def setup_imports():
    env = environment()
    os.environ.clear()
    os.environ.update(env)
    sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)


def validate_diagnostic_controls(recipe):
    expected = dict(working_set_census_only=False, semantic_working_set='legacy',
        candidate_rounds=1, doctor_coding_policy=None, grok_coding_max_turns=8,
        disable_intent_autoencoder=True, planning_mode='declared-contract@1',
        semantic_metadata_view='legacy', coding_reply_mode='legacy',
        source_view_profile='identity@1', source_edit_mode='disabled',
        semantic_transport_schema='supervisor-semantic-router-input@1',
        provider_profile='grok-4.7-cli-1.0.46@1',
        resource_profile='source384-5cpu-20gib-formalization@1',
        max_coding_invocations=1, provider_fallback=False, automatic_retries=0,
        official_verifier_enabled=False, expected_progress_schema='native-grok-stream-progress@5',
        expected_tool_event_schema='native-grok-tool-events@2')
    assert all(type(recipe.get(key)) is type(value) and recipe[key] == value for key, value in expected.items())


def freeze(qualification):
    """Root supplies a passing exact-source qualification; no execution follows."""
    recipe = read(A / 'recipe-draft.json')
    validate_diagnostic_controls(recipe)
    assert recipe['status'] == 'disabled_pending_qualification_and_source_freeze'
    source = Path(recipe['source'])
    assert not git(source, 'status', '--porcelain=v1')
    passed = read(qualification)
    assert passed['passed'] is True and type(passed['passed_cases']) is int and passed['passed_cases'] > 0
    assert passed['runtime_commit'] == git(source, 'rev-parse', 'HEAD')
    assert passed['failures'] == passed['errors'] == passed['skipped'] == 0
    assert passed['source_sha256']
    assert 'ipfs_accelerate_py/cli_runtime/grok_tool_events.py' in passed['source_sha256']
    for relative, digest in passed['source_sha256'].items():
        assert sha(source / relative) == digest
    roots = {'source': dict(path=str(source), commit=git(source, 'rev-parse', 'HEAD')),
             'datasets': recipe['datasets'], 'kit': recipe['kit']}
    for root in roots.values():
        assert git(root['path'], 'rev-parse', 'HEAD') == root['commit']
        assert not git(root['path'], 'status', '--porcelain=v1', '--untracked-files=no')
    files = [binding(A / name) for name in ('run.py', 'prepare_draft.py', 'recipe-draft.json',
             'environment.json', 'largest-eigenval-public-profile.json', 'closed_export.py',
             'local-asset-readiness-draft01.json', 'test_run.py', 'test_closed_export.py', 'scaffold-origin.json')]
    files.append(binding(qualification))
    inputs = [recipe['source384_config'], *recipe['public_input_bindings']]
    assets = read(A / 'local-asset-readiness-draft01.json')
    assert assets['source384_config'] == recipe['source384_config']
    inputs.extend([assets['checkpoint'], assets['grok_binary'], assets['torch_cpu_wheel'],
        *assets['embedding_assets'], *assets['extensions'], *assets['public_environment_inputs']])
    for row in inputs:
        assert binding(row['path']) == row
    recipe.update(status='frozen_after_root_qualification', roots=roots)
    write(A / 'recipe.json', recipe)
    write(A / 'source-freeze.json', dict(schema='tool-event-public-diagnostic-source-freeze@1',
        roots=roots, files=files, recipe=binding(A / 'recipe.json'), inputs=inputs,
        qualification=binding(qualification), source_sha256=passed['source_sha256'],
        source_frozen_before_execution=True, provider_calls=0, docker_calls=0))
    print(json.dumps(dict(status='frozen', provider_calls=0, docker_calls=0)))


def check():
    recipe = read(A / 'recipe.json')
    validate_diagnostic_controls(recipe)
    frozen = read(A / 'source-freeze.json')
    assert recipe['status'] == 'frozen_after_root_qualification'
    assert binding(A / 'recipe.json') == frozen['recipe']
    for row in [*frozen['files'], *frozen['inputs']]:
        assert binding(row['path']) == row
    for root in frozen['roots'].values():
        assert git(root['path'], 'rev-parse', 'HEAD') == root['commit']
        assert not git(root['path'], 'status', '--porcelain=v1',
            *([] if root['path'] == recipe['source'] else ['--untracked-files=no']))
    for relative, digest in frozen['source_sha256'].items():
        assert sha(Path(recipe['source']) / relative) == digest
    return recipe


def verify(recipe):
    """Stream every archive member and match its manifest before any launch."""
    root = Path(recipe['runtime'])
    archive, manifest_path = root / 'runtime.tar.gz', root / 'manifest.json'
    manifest = read(manifest_path)
    assert sha(archive) == manifest['archive_sha256']
    assert manifest['credentials_in_archive'] is False and manifest['task_inputs_in_archive'] is False
    rows = {row['path']: row for row in manifest['files']}
    assert len(rows) == len(manifest['files'])
    seen, total = set(), 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            assert member.isfile() and member.name not in seen and member.name in rows
            row = rows[member.name]
            assert member.size == row['bytes'] and member.mode == row['mode']
            with stream.extractfile(member) as content:
                assert hashlib.file_digest(content, 'sha256').hexdigest() == row['sha256']
            seen.add(member.name)
            total += member.size
    assert seen == set(rows)
    setup_imports()
    from benchmarks.agent_supervisor.container_coding.terminal_semantic_working_set_policy import require_semantic_working_set_archive
    from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
    from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import require_planning_archive
    from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import require_worker_capability
    from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import require_runtime_provider_profile
    from benchmarks.agent_supervisor.container_coding.terminal_deployment import validate_source384_binding, validate_torch_cpu_wheel_binding
    require_semantic_working_set_archive(manifest, recipe['semantic_working_set'])
    require_source_view_archive(manifest, recipe['source_view_profile'], recipe['resource_profile'])
    require_planning_archive(manifest, recipe['planning_mode'])
    require_worker_capability(manifest, recipe['resource_profile'])
    require_runtime_provider_profile(manifest, recipe['provider_profile'])
    from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import require_grok_coding_archive
    require_grok_coding_archive(manifest, recipe['grok_coding_max_turns'])
    source384 = validate_source384_binding(manifest)
    assert source384 is not None
    assert validate_torch_cpu_wheel_binding(manifest) is not None
    assets = read(A / 'local-asset-readiness-draft01.json')
    assert source384['config']['checkpoint_sha256'] == assets['checkpoint']['sha256']
    assert source384['source_training_data_included'] is False
    for relative, digest in read(A / 'source-freeze.json')['source_sha256'].items():
        archived = rows.get('source/' + relative)
        if Path(relative).name.startswith('test_'):
            continue
        assert archived is not None and archived['sha256'] == digest
    return dict(archive=binding(archive), manifest=binding(manifest_path),
        all_archive_members_verified=True, archive_members=len(seen), uncompressed_bytes=total,
        provider_calls=0, docker_calls=0)


async def execute(recipe):
    validate_diagnostic_controls(recipe)
    setup_imports()
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.agent.context import AgentContext
    from harbor.environments.docker.docker import DockerEnvironment
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import validate_task_profile
    from benchmarks.agent_supervisor.container_coding.terminal_source384_qualification import resource_options, observe_resources
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import execution_budget
    from closed_export import export_report

    assert 'working_set_census_only' in inspect.signature(FullSupervisorAgent.__init__).parameters
    verified = read(A / 'verify-result.json')
    for key in ('archive', 'manifest'):
        assert binding(verified[key]['path']) == verified[key]
    assert verified['exit_code'] == 0 and verified['all_archive_members_verified'] is True
    fresh = A / 'private-attempt-01'
    fresh.mkdir(mode=0o700)
    task = Task(Path(recipe['task']), disable_verification=True)
    assert not task.has_steps
    profile = validate_task_profile(read(recipe['public_profile']), instruction=task.instruction)
    assert profile['schema'] == 'terminal-public-task-profile@4'
    assert hashlib.sha256(task.instruction.encode()).hexdigest() == profile['instruction_sha256']
    paths = TrialPaths(fresh / 'harbor')
    paths.mkdir()
    session = 'tool-event-coverage-diagnostic-' + uuid.uuid4().hex[:16]
    env = DockerEnvironment(environment_dir=task.paths.environment_dir, environment_name=task.short_name,
        session_id=session, trial_paths=paths, task_env_config=task.config.environment,
        keep_containers=False, **resource_options(recipe['resource_profile']))
    controls = {key: recipe[key] for key in ('provider_profile', 'resource_profile', 'planning_mode',
        'semantic_working_set', 'semantic_metadata_view', 'coding_reply_mode', 'source_view_profile',
        'source_edit_mode', 'semantic_transport_schema', 'candidate_rounds', 'doctor_coding_policy',
        'grok_coding_max_turns', 'disable_intent_autoencoder', 'working_set_census_only')}
    agent = FullSupervisorAgent(logs_dir=paths.agent_dir, model_name='grok-4.7',
        runtime_archive=recipe['runtime'], arm='full', task_profile=profile, **controls)
    agent.session_id = session
    context = AgentContext()
    cid, phase = None, 'container_start'
    closed = dict(schema='terminal-public-tool-event-diagnostic@1', qualified=False,
        scope='fresh-source384-single-coding-invocation-telemetry', task_name='largest-eigenval',
        official_verifier_executed=False, official_reward=None, task_completed=False,
        provider_billing_total_verified=False, complete_run_token_savings_claimed=False,
        source_freeze_sha256=sha(A / 'source-freeze.json'), public_profile_sha256=sha(recipe['public_profile']),
        instruction_sha256=profile['instruction_sha256'], archive_sha256=verified['archive']['sha256'],
        public_input_bindings=recipe['public_input_bindings'], controls=controls,
        expected_progress_schema=recipe['expected_progress_schema'],
        expected_tool_event_schema=recipe['expected_tool_event_schema'],
        max_coding_invocations=1, provider_fallback=False, automatic_retries=0,
        inline_evidence_dispatch_enabled=False, semantic_pack_projection_dispatch_enabled=False)
    started = time.monotonic()
    try:
        await asyncio.wait_for(env.start(force_build=True), timeout=900)
        container = await env._platform._resolve_service_container('main')
        cid = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}', container], text=True).strip()
        write(fresh / 'container-identity.json', dict(container_id=cid, session_id=session))
        closed['container_id_sha256'] = hashlib.sha256(cid.encode()).hexdigest()
        phase = 'deployment'
        await asyncio.wait_for(agent.setup(env), timeout=1800)
        phase = 'resource_observation'
        closed['resources'] = await observe_resources(env, output=fresh, profile=recipe['resource_profile'])
        phase = 'declared_planning_and_coding'
        await asyncio.wait_for(agent.run(task.instruction, env, context),
            timeout=execution_budget(recipe['resource_profile'])['exec_seconds'])
    except BaseException as error:
        closed['outer_failure'] = dict(phase=phase, error_type=type(error).__name__)
    finally:
        try:
            await asyncio.wait_for(env.stop(delete=True), timeout=60)
            closed['container_stop_returned'] = True
        except BaseException as error:
            closed['container_stop_returned'] = False
            closed['container_stop_error_type'] = type(error).__name__
        observations = []
        for _ in range(2):
            try:
                found = [] if cid is None else subprocess.check_output(['docker', 'container', 'ls', '--all',
                    '--quiet', '--no-trunc', '--filter', 'id=' + cid], text=True, timeout=15).splitlines()
                observations.append(None if cid is None else cid not in found)
            except BaseException:
                observations.append(None)
            await asyncio.sleep(.2)
        closed['exact_container_absence_observations'] = observations
        try:
            check()
            closed['frozen_inputs_unchanged'] = True
        except BaseException:
            closed['frozen_inputs_unchanged'] = False
        result_path = paths.agent_dir / 'supervisor-result.json'
        if result_path.exists():
            closed['driver_report_sha256'] = sha(result_path)
            try:
                closed['observation'] = export_report(read(result_path))
            except BaseException as error:
                closed['export_error_type'] = type(error).__name__
        closed['seconds'] = time.monotonic() - started
        closed['qualified'] = (closed.get('observation', {}).get('qualified') is True
            and not closed.get('outer_failure') and closed['frozen_inputs_unchanged'] is True
            and closed['container_stop_returned'] is True and observations == [True, True])
        write(A / 'actual-public-diagnostic-01.json', closed)
    return dict(qualified=closed['qualified'], observation=binding(A / 'actual-public-diagnostic-01.json'),
        exact_container_absence_observations=observations, outer_failure=closed.get('outer_failure'))


def step(name):
    recipe = check()
    free = shutil.disk_usage(A).free
    assert free >= recipe['minimum_free_bytes']
    assert name in ('bundle', 'verify', 'execute')
    write(A / (name + '-start.json'), dict(step=name, free_bytes=free,
        minimum_free_bytes=recipe['minimum_free_bytes'], started_at_utc=datetime.now(timezone.utc).isoformat(),
        native_admission_still_required=True, source_freeze_sha256=sha(A / 'source-freeze.json')))
    started = time.monotonic()
    code, result = 1, {}
    try:
        if name == 'bundle':
            assert not Path(recipe['runtime']).exists()
            with (A / 'bundle.stdout').open('xb') as out, (A / 'bundle.stderr').open('xb') as err:
                code = subprocess.run(recipe['bundle_argv'], cwd=recipe['source'], env=environment(),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=1800).returncode
            result = dict(stdout=binding(A / 'bundle.stdout'), stderr=binding(A / 'bundle.stderr'))
        elif name == 'verify':
            result, code = verify(recipe), 0
        else:
            result = asyncio.run(execute(recipe))
            code = 0 if result['qualified'] else 1
        check()
        result['frozen_inputs_unchanged'] = True
    except BaseException as error:
        code = 1
        result['error_type'] = type(error).__name__
    result.update(step=name, exit_code=code, seconds=time.monotonic() - started)
    write(A / (name + '-result.json'), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('freeze', 'bundle', 'verify', 'execute'))
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    if args.step == 'freeze':
        assert args.qualification is not None
        freeze(args.qualification)
    else:
        raise SystemExit(step(args.step))
