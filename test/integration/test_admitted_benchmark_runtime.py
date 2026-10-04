from __future__ import annotations

import json
import os
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


@pytest.fixture
def admitted(tmp_path, monkeypatch, request):
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'ambient-unrelated-state'))
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepared = prepare_local_planning_qualification(tmp_path / 'task')
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
        if getattr(request, 'param', 'configured') == 'extended-proof':
            monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
            monkeypatch.setenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH', str(tmp_path / 'shared-scheduler.json'))
        parent_orchestration = os.environ.get(board.ORCHESTRATION_DIR_ENV)
        def forbidden_account_scan():
            raise AssertionError('isolated runtime construction must not migrate account catalogs')
        monkeypatch.setattr(board, 'adopt_legacy_platform_databases', forbidden_account_scan)
        runtime = AdmittedBenchmarkRuntime.create(
            tmp_path / 'launch', admission=prepared['admission'], server=owner.server,
            source=owner.source, context_bundle=bundle, timeout_ms=20_000,
            worker_worktree_root=worktrees,
            **({'lifetime_seconds':900} if getattr(request, 'param', 'configured') == 'extended-proof' else {}),
        )
        try:
            assert os.environ.get(board.ORCHESTRATION_DIR_ENV) == parent_orchestration
            yield runtime, owner, prepared
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()


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
