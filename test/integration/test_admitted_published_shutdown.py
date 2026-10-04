"""Stop an exact native run after publication without granting acceptance."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import native_published_transition, typed_claim


@pytest.fixture(autouse=True)
def isolated_lifecycle_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')


@pytest.fixture
def published_runtime(scenario, tmp_path):
    admission = local.admit_local_benchmark_plan(graph=scenario['graph'], manifest=scenario['manifest'])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        runtime = AdmittedBenchmarkRuntime.create(
            tmp_path / 'launch', admission=admission, server=owner.server,
            source=owner.source, implement=False, timeout_ms=30_000,
        )
        try:
            started = runtime.start()
            assert started.succeeded, started.error
            assert runtime.observe()['healthy']
            transition = native_published_transition(scenario, tmp_path, owner, attempt)
            assert transition['payload']['baseline_commit'] != transition['payload']['published_commit']
            yield runtime, owner, attempt
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()


def _observation_count(owner, attempt):
    return owner.server._connection.execute(
        'SELECT COUNT(*) FROM validation_results WHERE task_cid = ?', [attempt.task_cid],
    ).fetchone()[0]


def test_actual_published_run_stops_before_owner_validation_without_accepting_source(published_runtime):
    runtime, owner, attempt = published_runtime
    before = owner.source.get_task(attempt.task_cid)
    assert _observation_count(owner, attempt) == 0
    assert len(runtime.process.snapshot(runtime.profile).members) >= 2
    with pytest.raises(local.LocalPlanningError, match='baseline drift'):
        runtime.start()
    with pytest.raises(local.LocalPlanningError, match='no exact native owner observation'):
        runtime.observe()
    stopped = runtime.stop()
    assert stopped.succeeded, stopped.error
    assert not runtime.process.snapshot(runtime.profile).members
    assert owner.source.get_task(attempt.task_cid) == before
    assert before.status == 'in_progress'
    assert _observation_count(owner, attempt) == 0
    with pytest.raises(local.LocalPlanningError, match='no exact native owner observation'):
        runtime.observe()


def test_published_shutdown_retains_launch_identity_signature_policy_and_fence(published_runtime, monkeypatch):
    runtime, owner, attempt = published_runtime
    original_profile = runtime.profile
    identities = {member.identity_id for member in runtime.process.snapshot(original_profile).members}
    (runtime.repository / 'another-cwd').mkdir()
    changes = [
        {'argv': (*original_profile.argv, '--foreign-option')},
        {'cwd': str(runtime.repository / 'another-cwd')},
        {'environment': (*original_profile.environment, ('FOREIGN_OWNER_ENV', '1'))},
        {'target_id': 'foreign-target'},
        {'run_id': 'foreign-run'},
        {'configuration_root': 'foreign-configuration'},
    ]
    for change in changes:
        runtime.profile = replace(original_profile, **change, profile_id='')
        try:
            with pytest.raises(ValueError, match='lifecycle profile differs'):
                runtime.stop()
        finally:
            runtime.profile = original_profile
        assert {member.identity_id for member in runtime.process.snapshot(original_profile).members} == identities
    tree_id = runtime.tree_id
    runtime.tree_id = '0' * 40
    try:
        with pytest.raises(ValueError, match='shutdown launch grant changed'):
            runtime.stop()
    finally:
        runtime.tree_id = tree_id
    grant = runtime.state / 'local-process-grant.json'
    original_grant = grant.read_bytes()
    forged = json.loads(original_grant)
    forged['manifest']['run_id'] = 'foreign-run'
    grant.write_text(json.dumps(forged))
    try:
        with pytest.raises(ValueError, match='persisted admitted shutdown grant changed'):
            runtime.stop()
    finally:
        grant.write_bytes(original_grant)
    original_signature = runtime.signature
    runtime.signature = deepcopy(original_signature)
    signature = runtime.signature['signature']
    runtime.signature['signature'] = ('0' if signature[0] != '0' else '1') + signature[1:]
    grant.write_text(json.dumps({'manifest': runtime.manifest, 'signature': runtime.signature}))
    try:
        with pytest.raises(ValueError):
            runtime.stop()
    finally:
        runtime.signature = original_signature
        grant.write_bytes(original_grant)
    request = runtime.request(Operation.STOP)
    foreign = replace(request, parameters={**request.parameters, 'run_id': 'foreign-run'})
    assert not runtime.service.execute(foreign).succeeded
    lease = runtime.lease
    runtime.lease = replace(lease, fencing_token=lease.fencing_token + 1)
    try:
        assert not runtime.service.execute(request).succeeded
    finally:
        runtime.lease = lease
    # Exercise the native handler's stale revision boundary itself. Its
    # exception must prevent entering candidate cleanup at all. This sentinel
    # is not a candidate runner and cannot authorize a successful operation.
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
    from ipfs_accelerate_py.agent_supervisor.control.control_plane import TransactionConflictError
    original_manifest = runtime.manifest
    runtime.manifest = {**original_manifest, 'candidate_runner': {'refusal-probe': True}}
    def forbidden_cleanup(_binding):
        pytest.fail('a refused lifecycle operation entered worker cleanup')
    with monkeypatch.context() as patch:
        patch.setattr(candidate_execution, 'verify_candidate_runner', forbidden_cleanup)
        try:
            stale = replace(request, parameters={**request.parameters, 'expected_revision': 0})
            with pytest.raises(TransactionConflictError):
                runtime._bounded_lifecycle_response(stale)
        finally:
            runtime.manifest = original_manifest
    assert {member.identity_id for member in runtime.process.snapshot(original_profile).members} == identities
    assert _observation_count(owner, attempt) == 0
    assert runtime.stop().succeeded
    assert not runtime.process.snapshot(original_profile).members
