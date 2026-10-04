from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_planning_qualification
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE


def _prepare_implementation_fixture(output):
    """Declare native pre-merge checks using the fixed launcher vocabulary."""
    from benchmarks.agent_supervisor.container_coding.local_planning_qualification import prepare_local_task
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    repository = output / 'repository'
    repository.mkdir(parents=True)
    (repository / 'answer.py').write_text('def answer():\n    return 1\n')
    (repository / 'test_answer.py').write_text('from answer import answer\nassert answer() == 2\n')
    for args in [('init', '-q'), ('add', '.'),
                 ('-c', 'user.name=Native implementation test', '-c',
                  'user.email=native@example.invalid', 'commit', '-qm', 'Declared public acceptance')]:
        subprocess.run(['git', '-C', str(repository), *args], check=True, capture_output=True)
    with IntentRepository(output / 'intent.duckdb') as intent:
        return prepare_local_task(repository=repository, state=output / 'state', intent=intent,
            scope_paths=['answer.py', 'test_answer.py'], output_path='answer.py',
            validation_argv=['python3', '-B', 'test_answer.py'],
            objective='Return two and pass the immutable public answer check')


@pytest.fixture
def admitted(tmp_path, monkeypatch, request):
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'ambient-unrelated-state'))
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepare = (_prepare_implementation_fixture if getattr(request, 'param', 'configured') == 'implementation'
               else prepare_local_planning_qualification)
    prepared = prepare(tmp_path / 'task')
    verified = verify_local_benchmark_admission(prepared['admission'])
    from pathlib import Path
    with open_existing_native_owner(
        database=Path(prepared['intent_database']), checkout=Path(prepared['repository']),
        state_dir=tmp_path / 'owner', repository_id=verified['manifest']['repository_cid'],
        execution_routes={prepared['task_id']: GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        bundle = write_task_context_bundle(repository=Path(prepared['repository']), prepared=[{
            'schema': 'supervisor-task-context-preparation@1', 'task_cid': prepared['task_cid'],
            'task_id': prepared['task_id'], 'metadata': {},
        }], output=Path(prepared['repository']) / '.runtime/context.json')
        worktrees = tmp_path / 'allocated-worktrees'
        worktrees.mkdir(mode=0o750)
        from ipfs_accelerate_py.agent_supervisor.task_sources import board_control_plane as board
        if getattr(request, 'param', 'configured') == 'unset':
            monkeypatch.delenv(board.ORCHESTRATION_DIR_ENV, raising=False)
        if getattr(request, 'param', 'configured') in {'extended-proof', 'extended-startup'}:
            monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
            monkeypatch.setenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH', str(tmp_path / 'shared-scheduler.json'))
        parent_orchestration = os.environ.get(board.ORCHESTRATION_DIR_ENV)
        def forbidden_account_scan():
            raise AssertionError('isolated runtime construction must not migrate account catalogs')
        monkeypatch.setattr(board, 'adopt_legacy_platform_databases', forbidden_account_scan)
        implementation = {}
        if getattr(request, 'param', 'configured') == 'implementation':
            implementation = dict(implement=True, implementation_command=shlex.join([
                sys.executable, '-B', '-c',
                "from pathlib import Path; Path('answer.py').write_text('def answer():\\n    return 2\\n')",
            ]))
        runtime = AdmittedBenchmarkRuntime.create(
            tmp_path / 'launch', admission=prepared['admission'], server=owner.server,
            source=owner.source, context_bundle=bundle, timeout_ms=20_000,
            worker_worktree_root=worktrees,
            **implementation,
            **({'lifetime_seconds':900} if getattr(request, 'param', 'configured') in {'extended-proof', 'extended-startup'} else {}),
            **({'start_timeout_ms':120_000} if getattr(request, 'param', 'configured') == 'extended-startup' else {}),
        )
        try:
            assert os.environ.get(board.ORCHESTRATION_DIR_ENV) == parent_orchestration
            yield runtime, owner, prepared
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()


@pytest.mark.parametrize('admitted', ['implementation'], indirect=True)
def test_actual_native_implementation_binds_owner_validation_and_completes_authored_repair(admitted):
    """Exercise the actual implementation path with a declared repair and no model."""
    runtime, owner, prepared = admitted
    original_check = (runtime.repository / 'test_answer.py').read_bytes()
    assert type(runtime.completion_service) is dict
    assert runtime.completion_service['bound'] is True
    assert runtime.completion_service['completion_authority'] is False
    assert callable(owner.server._command_gateway._local_task_validation_handler)
    assert owner.source.get_task(prepared['task_cid']).status == 'ready'
    started = runtime.start()
    assert started.succeeded, started.error
    deadline = time.monotonic() + 90
    while True:
        task = owner.source.get_task(prepared['task_cid'])
        if task.status in {'completed', 'failed', 'blocked', 'cancelled'}:
            break
        assert time.monotonic() < deadline, (task.status, task.revision)
        assert runtime.process.snapshot(runtime.profile).members
        time.sleep(.25)
    assert task.status == 'completed', (task.status, task.revision)
    assert (runtime.repository / 'answer.py').read_text() == 'def answer():\n    return 2\n'
    assert (runtime.repository / 'test_answer.py').read_bytes() == original_check
    assert runtime.bootstrap_receipts and runtime.bootstrap_errors == []
    assert runtime.stop().succeeded
    assert not runtime.process.snapshot(runtime.profile).members


@pytest.mark.parametrize('failure', ['before-binding', 'after-binding', 'duplicate-binding'])
def test_failed_implementation_constructor_closes_unlaunched_native_resources(tmp_path, monkeypatch, failure):
    """A binding refusal owns no child, but must dispose its real run lease/thread."""
    from pathlib import Path
    from ipfs_accelerate_py.agent_supervisor.runtime import local_completion_bridge as bridge
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LocalPlanningError
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_database_coordinator
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TypedStateOwnerProtocolError
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepared = prepare_local_planning_qualification(tmp_path / 'task')
    verified = verify_local_benchmark_admission(prepared['admission'])
    captured = []
    original_bootstrap = AdmittedBenchmarkRuntime._serve_bootstrap
    def observe_bootstrap(runtime):
        captured.append(runtime)
        original_bootstrap(runtime)
    original_binding = bridge.bind_owner_local_completion_service
    def refuse_binding(**kwargs):
        if failure != 'before-binding':
            original_binding(**kwargs)
        raise LocalPlanningError('deliberate owner binding refusal')
    monkeypatch.setattr(AdmittedBenchmarkRuntime, '_serve_bootstrap', observe_bootstrap)
    monkeypatch.setattr(bridge, 'bind_owner_local_completion_service', refuse_binding)
    with open_existing_native_owner(
        database=Path(prepared['intent_database']), checkout=Path(prepared['repository']),
        state_dir=tmp_path / 'owner', repository_id=verified['manifest']['repository_cid'],
        execution_routes={prepared['task_id']: GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        gateway = owner.server._command_gateway
        existing_handler = lambda *_args: None
        if failure == 'duplicate-binding':
            gateway.bind_local_task_validation_handler(existing_handler)
        error, message = {
            'before-binding': (LocalPlanningError, 'deliberate owner binding refusal'),
            'after-binding': (RuntimeError, 'constructor cleanup unproven'),
            'duplicate-binding': (TypedStateOwnerProtocolError, 'already bound'),
        }[failure]
        with pytest.raises(error, match=message):
            AdmittedBenchmarkRuntime.create(tmp_path / 'launch', admission=prepared['admission'],
                server=owner.server, source=owner.source, implement=True,
                implementation_command=shlex.join([sys.executable, '-B', '-c', 'pass']))
        assert len(captured) == 1
        runtime = captured[0]
        assert not runtime._bootstrap_thread.is_alive()
        assert runtime._bootstrap_stop.is_set() and runtime._listener.fileno() == -1
        assert runtime._children == [] and runtime.bootstrap_receipts == []
        assert not runtime.process.snapshot(runtime.profile).members
        if failure == 'after-binding':
            assert callable(gateway._local_task_validation_handler)
            assert getattr(runtime, 'completion_service', None) is None
            assert runtime._construction_cleanup_failed
        else:
            assert gateway._local_task_validation_handler is (existing_handler if failure == 'duplicate-binding' else None)
        assert owner.source.get_task(prepared['task_cid']).status == 'ready'
        coordinator = (runtime.coordinator if failure == 'after-binding'
                       else open_database_coordinator(runtime.state / 'coordination.duckdb'))
        try:
            assert coordinator.get_lease(runtime.lease.lease_id).state.value == (
                'accepted' if failure == 'after-binding' else 'released')
        finally:
            if failure != 'after-binding':
                coordinator.close()
        receipt_path = runtime.state / 'construction-cleanup.json'
        if failure == 'after-binding':
            assert not receipt_path.exists()
        else:
            receipt = json.loads(receipt_path.read_text())
            assert receipt['process_observation'] == 'empty_native_tree'
            assert receipt['run_lease_released'] and receipt['bootstrap_stopped']
            assert receipt['native_STOP_proved'] is False
    if failure == 'after-binding':
        # Test teardown releases its retained lease only after the actual
        # owner has closed. Production rollback never retires this callback.
        coordinator = runtime.coordinator
        try:
            coordinator.release(runtime.lease, expected_fencing_token=runtime.lease.fencing_token,
                                expected_fence_epoch=runtime.lease.fence_epoch)
        finally:
            coordinator.close()


def test_legacy_gateway_ordinary_binding_and_retirement_refusal(tmp_path):
    from test.api.test_owner_completion_service_retirement import owner_fixture
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import bind_owner_local_completion_service
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import LocalPlanningError
    server, _gateway, repository = owner_fixture(tmp_path)
    class LegacyGateway:
        def __init__(self):
            self.bound = []
        def bind_local_task_validation_handler(self, handler):
            self.bound.append(handler)
    gateway = server._command_gateway = LegacyGateway()
    kwargs = dict(server=server, repo_root=repository, portal_attempt_root=tmp_path / 'attempts',
        merge_queue_dir=tmp_path / 'queue', board_namespace='legacy-fixture', target_branch='main')
    with pytest.raises(LocalPlanningError, match='does not support completion retirement'):
        bind_owner_local_completion_service(**kwargs, retirable=True)
    assert gateway.bound == []
    service = bind_owner_local_completion_service(**kwargs, retirable=False)
    assert type(service) is dict and service['bound'] and len(gateway.bound) == 1


def test_actual_native_owner_child_start_heartbeat_and_stop(admitted):
    runtime, owner, prepared = admitted
    original = owner.source.get_task(prepared['task_cid'])
    started = runtime.start()
    assert started.succeeded, started.error
    assert len(runtime.bootstrap_receipts) == 1
    assert runtime.manifest['lifetime_seconds'] == 300
    assert runtime.manifest['worker_worktree_root'].endswith('/allocated-worktrees')
    assert runtime.profile.argv[runtime.profile.argv.index('--worktree-root') + 1] == runtime.manifest['worker_worktree_root']
    assert runtime.bootstrap_receipts[0]['grant_expires_at_ms'] <= runtime.lease.expires_at_ms
    assert runtime.bootstrap_receipts[0]['run_lease_expires_at_ms'] == runtime.lease.expires_at_ms
    assert runtime.bootstrap_errors == []
    first = runtime.observe()
    assert first['healthy'] and not first['provider_dispatch_allowed']
    assert first['native_heartbeat']['owner_read_sequence'] >= 1
    time.sleep(.6)
    second = runtime.observe()
    assert second['healthy']
    assert second['native_heartbeat']['owner_read_sequence'] > first['native_heartbeat']['owner_read_sequence']
    assert second['native_heartbeat']['completion_authority'] is False
    assert second['native_heartbeat']['task_progress_authority'] is False
    members = second['process_tree']['members']
    child = next(x for x in members if x['pid'] == second['native_heartbeat']['daemon_pid'])
    assert '--state-store-id' in child['argv']
    assert '--task-context-bundle-artifact' in child['argv']
    assert child['pid'] == runtime.bootstrap_receipts[0]['pid']
    assert child['start_time_ticks'] > 0
    assert owner.source.get_task(prepared['task_cid']) == original
    completed_pass = json.loads((runtime.state / 'run/admitted_database_daemon_pass_heartbeat.json').read_text())
    assert completed_pass['selection_idle_reason'] == 'implementation_disabled'
    assert runtime.stop().succeeded
    assert runtime.observe()['process_tree']['members'] == []
    assert not runtime.observe()['healthy']
    assert owner.source.get_task(prepared['task_cid']) == original


@pytest.mark.parametrize('admitted', ['configured', 'unset'], indirect=True,
                         ids=['ambient-override', 'account-default'])
def test_signed_run_owns_orchestration_path_without_legacy_account_scan(admitted, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import board_control_plane as board
    runtime, _owner, _prepared = admitted
    expected = str(runtime.state / 'orchestration')
    environment = dict(runtime.profile.environment)
    assert environment[board.ORCHESTRATION_DIR_ENV] == expected
    assert expected != os.environ.get(board.ORCHESTRATION_DIR_ENV)
    assert dict(runtime.manifest['environment'])[board.ORCHESTRATION_DIR_ENV] == expected
    assert (runtime.state / 'orchestration').stat().st_mode & 0o777 == 0o700
    monkeypatch.setenv(board.ORCHESTRATION_DIR_ENV, environment[board.ORCHESTRATION_DIR_ENV])
    def forbidden_scan():
        raise AssertionError('isolated native startup must not scan or migrate ambient account state')
    monkeypatch.setattr(board, 'adopt_legacy_platform_databases', forbidden_scan)
    assert str(board.orchestration_state_root()) == expected
    runtime._verify()
    # Environment changes cannot borrow the signed launch after preparation.
    runtime.profile = replace(runtime.profile, profile_id="", environment=tuple(
        (key, '/tmp/unrelated-orchestration') if key == board.ORCHESTRATION_DIR_ENV else (key, value)
        for key, value in runtime.profile.environment))
    with pytest.raises(ValueError, match='launch environment'):
        runtime._verify()


def test_foreign_request_cannot_borrow_admitted_launch(admitted):
    runtime, _owner, _prepared = admitted
    request = runtime.request(Operation.START)
    altered = replace(request, parameters={**request.parameters, 'run_id': 'foreign'})
    result = runtime.service.execute(altered)
    assert not result.succeeded
    assert not runtime.process.snapshot(runtime.profile).members
    assert not runtime.bootstrap_receipts


def test_stale_native_lease_does_not_launch_or_issue_child_grant(admitted):
    runtime, _owner, _prepared = admitted
    request = runtime.request(Operation.START)
    lease = runtime.lease
    runtime.lease = replace(lease, fencing_token=lease.fencing_token + 1)
    try:
        assert not runtime.service.execute(request).succeeded
        assert not runtime.process.snapshot(runtime.profile).members
        assert not runtime.bootstrap_receipts
    finally:
        runtime.lease = lease


def test_changed_context_digest_is_rejected_before_launch(admitted):
    runtime, _owner, _prepared = admitted
    context = runtime.repository / runtime.context_bundle['artifact']
    original = context.read_bytes()
    context.write_bytes(original + b' ')
    try:
        with pytest.raises(ValueError, match='digest differs'):
            runtime.start()
        assert not runtime.bootstrap_receipts
    finally:
        context.write_bytes(original)


@pytest.mark.parametrize('lifetime', [True, 119, 901, 300.0])
def test_launch_lifetime_is_an_explicit_bounded_integer(tmp_path, lifetime):
    with pytest.raises(ValueError, match='lifetime_seconds'):
        AdmittedBenchmarkRuntime.create(tmp_path / 'launch', admission=None, server=None,
                                        source=None, lifetime_seconds=lifetime)


@pytest.mark.parametrize('admitted', ['extended-proof'], indirect=True)
def test_extended_launch_binds_exact_shared_scheduler_and_nine_hundred_second_lease(admitted):
    runtime,_owner,_prepared=admitted
    expected={key:os.environ[key] for key in ('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','IPFS_DATASETS_RESOURCE_SCHEDULER_PATH')}
    assert runtime.manifest['lifetime_seconds']==900
    assert all(dict(runtime.profile.environment)[key]==value for key,value in expected.items())
    assert all(dict(runtime.manifest['environment'])[key]==value for key,value in expected.items())
    runtime._verify()
    runtime.profile=replace(runtime.profile,profile_id='',environment=tuple(
        (key,value+'-foreign') if key=='IPFS_DATASETS_RESOURCE_SCHEDULER_PATH' else (key,value)
        for key,value in runtime.profile.environment))
    with pytest.raises(ValueError,match='launch environment'):
        runtime._verify()


def test_default_proof_profile_does_not_forward_an_ambient_scheduler(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import _bounded_proof_resource_environment
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE',raising=False)
    monkeypatch.setenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH','/tmp/unrelated-scheduler.json')
    assert _bounded_proof_resource_environment()=={}


@pytest.mark.parametrize('invalid',['unknown-profile','missing-path','relative-path','directory','symlink','newline'])
def test_selected_proof_profile_requires_exact_canonical_shared_ledger(tmp_path,monkeypatch,invalid):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import _bounded_proof_resource_environment
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','local-benchmark@1')
    value=str(tmp_path/'scheduler.json')
    if invalid=='unknown-profile':
        monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','other@1')
    elif invalid=='missing-path':value=''
    elif invalid=='relative-path':value='scheduler.json'
    elif invalid=='directory':value=str(tmp_path)
    elif invalid=='newline':value+='\n'
    elif invalid=='symlink':
        target=tmp_path/'actual.json';target.write_text('{}');link=tmp_path/'alias.json';link.symlink_to(target);value=str(link)
    monkeypatch.setenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH',value)
    with pytest.raises(ValueError):
        _bounded_proof_resource_environment()


def test_candidate_git_environment_retains_closed_configuration():
    from pathlib import Path
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as module
    environment = module._candidate_git_environment()
    assert environment['GIT_CONFIG_NOSYSTEM'] == '1'
    assert environment['GIT_CONFIG_GLOBAL'] == '/dev/null'
    values = {environment[f'GIT_CONFIG_KEY_{index}']: environment[f'GIT_CONFIG_VALUE_{index}']
              for index in range(int(environment['GIT_CONFIG_COUNT']))}
    assert values['core.hooksPath'] == '/dev/null'
    assert values['core.fsmonitor'] == 'false'
    if 'safe.directory' in values:
        assert values['safe.directory'] == str(Path(module.__file__).absolute().parents[3])
        assert '*' not in values['safe.directory']
