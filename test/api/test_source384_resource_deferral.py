"""Source384 admission deferral controls with explicit authored pressure.

The local scheduler, resource leases and exception observations are real. The
pressure sampler and source-unit report are small authored fixtures; these
tests do not reproduce host PSI, execute a model, or qualify benchmark work.
"""
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources


@pytest.fixture
def native_child_timeout(tmp_path, monkeypatch):
    """Obtain the real error at the same child-admission boundary as R."""
    monkeypatch.setenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", "local-benchmark@1")
    healthy = ProofHostResources(8, 32768, 32768)
    readings = [healthy]
    config = resources.ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "isolated-proof-resources.json",
        proof_resource_profile="local-benchmark@1",
        proof_memory_stall_percent=10.0,
        proof_recovery_enabled=True,
        proof_recovery_grants=1,
        proof_resource_sampler=lambda: readings[0],
        total_child_process_slots=4,
        lane_reservations={},
        auto_renew_leases=False,
        proof_backoff_seconds=0.01,
        poll_interval_seconds=0.005,
    )
    scheduler = resources.GlobalResourceScheduler(config)
    cid = cid_for_structured({"authored_source_unit_fixture": True})
    head = CodebaseHead("fixture", 1, cid, cid, f"rev:fixture:snapshot:{cid}", cid)
    saved = {"key": {"source_head": head.to_dict()}}
    report = {"artifact": {"authored": True}, "report": saved,
              "native_worker_executed": False, "inference_executed": False, **units.FALSE}
    observed = []
    index = SimpleNamespace(observe_current=lambda *args, **kwargs: observed.append(True))
    registry = object()

    def owners(actual_index, actual_registry):
        assert actual_index is index and actual_registry is registry

    with monkeypatch.context() as patch:
        patch.setattr(units.shared, "_owners", owners)
        with scheduler.acquire("snapshot_evaluation", cpu_slots=3, memory_mb=6144,
                               child_process_slots=3, timeout=2) as parent:
            readings[0] = replace(healthy, memory_stall_percent=27.62)
            with pytest.raises(resources.LeaseTimeoutError) as caught:
                units.validate_shared_parent_units(
                    index, tmp_path, report, registry=registry,
                    embedding_snapshot="authored-no-model", parent_lease=parent,
                    timeout_seconds=0.05, memory_mb=4096,
                )
            inside = scheduler.snapshot()
            assert inside["active_root_lease_count"] == 1
            assert inside["active_lease_count"] == 1
            assert inside["waiting_request_count"] == 0
    state = scheduler.snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0
    assert observed == []
    return caught.value


def test_actual_source_unit_child_refusal_preserves_pressure_evidence(native_child_timeout):
    error = native_child_timeout
    observation = error.admission_observation
    assert observation["terminal"] == "timeout"
    assert observation["primary_gate"]["reason"] == "proof_memory_stall"
    sample = observation["last_sample"]
    assert sample["host"]["available_memory_mb"] == 32768
    assert sample["host"]["memory_stall_percent"] == 27.62
    assert sample["thresholds"]["memory_stall_percent"] == 10.0
    assert sample["additional_request_memory_mb"] == 0
    assert sample["reserved_root_memory_mb"] == 6144
    assert error.proof_refusal_observation["reason"] == "proof_memory_stall"


@pytest.fixture
def nomination_case(tmp_path, monkeypatch):
    """Use the production task envelope and live nomination loader."""
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    metadata = {"semantic context artifact": "semantic.json", "semantic context sha256": "b" * 64}
    task = impl.PortalTask(
        task_id="RESOURCE-001", title="Revalidate captured source", status="ready",
        completion="manual", priority="P1", track="context", outputs=["module.py"],
        validation=[], acceptance="Use only current captured source", metadata={},
        canonical_task_cid="task:resource-fixture",
    )
    receipt = {"schema": "terminal-source384-repository-context@1",
               "artifact": "source384.json", "sha256": "a" * 64,
               "completion_authority": False}
    payload = {"schema": bundles.SOURCE384_SCHEMA, "completion_authority": False,
               "tasks": [{"task_id": task.task_id, "task_cid": task.canonical_task_cid,
                          "metadata": metadata, "source384_context": receipt}]}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "context.json").write_bytes(raw)
    daemon = object.__new__(impl.TodoImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon._task_context_nomination_bundle = {
        "artifact": "context.json", "sha256": hashlib.sha256(raw).hexdigest()}
    case = SimpleNamespace(daemon=daemon, task=task, receipt=receipt, metadata=metadata,
                           failure=None, calls=[])

    def validate(**kwargs):
        case.calls.append(kwargs)
        if case.failure is not None:
            raise case.failure
        return deepcopy(receipt)

    monkeypatch.setattr(source, "validate_source384_context", validate)
    case.read = lambda: daemon._task_metadata_snapshot(task, include_context=True)
    token = impl._source384_callback_deadline.set(time.monotonic() + 60.)
    try:
        yield case
    finally:
        impl._source384_callback_deadline.reset(token)


def test_native_child_timeout_becomes_closed_pre_dispatch_deferral(
    nomination_case, native_child_timeout,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = nomination_case
    case.failure = native_child_timeout
    original = deepcopy(native_child_timeout.admission_observation)
    with pytest.raises(impl.ImplementationRetryDeferred) as caught:
        case.read()
    assert type(caught.value).__name__ == "Source384ResourceDeferred"
    assert caught.value.reason == "source384_resource_admission_deferred"
    assert caught.value.backoff_seconds == 5
    assert caught.value.__cause__ is native_child_timeout
    assert native_child_timeout.admission_observation == original
    assert case.task.metadata == {}
    assert len(case.calls) == 1


def test_unscoped_portal_cannot_turn_resource_timeout_into_unbounded_retry(
    nomination_case, native_child_timeout,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = nomination_case
    case.failure = native_child_timeout
    token = impl._source384_callback_deadline.set(None)
    try:
        with pytest.raises(resources.LeaseTimeoutError) as caught:
            case.read()
        assert caught.value is native_child_timeout
        assert len(case.calls) == 1
    finally:
        impl._source384_callback_deadline.reset(token)


@pytest.mark.parametrize("failure", [
    resources.LeaseCancelledError("cancelled authored preparation"),
    TimeoutError("ordinary non-admission timeout"),
    ValueError("selected source digest changed"),
    resources.LeaseTimeoutError("native type without its diagnostic"),
])
def test_other_nomination_failures_never_gain_resource_retry_authority(nomination_case, failure):
    nomination_case.failure = failure
    with pytest.raises(type(failure)) as caught:
        nomination_case.read()
    assert caught.value is failure
    assert nomination_case.task.metadata == {}


@pytest.mark.parametrize("damage", ["cancelled_terminal", "unknown_field", "nonfinite", "boolean_threshold"])
def test_damaged_native_observation_cannot_authorize_retry(
    nomination_case, native_child_timeout, damage,
):
    observation = deepcopy(native_child_timeout.admission_observation)
    if damage == "cancelled_terminal":
        observation["terminal"] = "cancelled"
    elif damage == "unknown_field":
        observation["unbounded_raw_body"] = "must not become public diagnostics"
    elif damage == "nonfinite":
        observation["last_sample"]["host"]["memory_stall_percent"] = float("nan")
    else:
        observation["last_sample"]["thresholds"]["memory_stall_percent"] = True
    native_child_timeout.admission_observation = observation
    nomination_case.failure = native_child_timeout
    with pytest.raises(resources.LeaseTimeoutError) as caught:
        nomination_case.read()
    assert caught.value is native_child_timeout
    assert nomination_case.task.metadata == {}


def test_recovered_nomination_revalidates_again_without_caching_refused_context(
    nomination_case, native_child_timeout,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = nomination_case
    case.failure = native_child_timeout
    with pytest.raises(impl.ImplementationRetryDeferred):
        case.read()
    case.failure = None
    assert dict(case.read()) == case.metadata
    assert len(case.calls) == 2
    assert case.task.metadata == {}


@pytest.mark.parametrize("stop", ["cancelled", "expired"])
def test_stop_arriving_during_admission_cannot_authorize_retry(
    nomination_case, native_child_timeout, monkeypatch, stop,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = nomination_case
    case.failure = native_child_timeout
    token = None
    if stop == "cancelled":
        case.daemon.implementation_cancelled = lambda: True
    else:
        clock = iter([100., 102.])
        monkeypatch.setattr(impl, "time", SimpleNamespace(monotonic=lambda: next(clock)))
        token = impl._source384_callback_deadline.set(101.)
    try:
        with pytest.raises(resources.LeaseTimeoutError) as caught:
            case.read()
        assert caught.value is native_child_timeout
        assert len(case.calls) == 1
    finally:
        if token is not None:
            impl._source384_callback_deadline.reset(token)


@pytest.fixture
def portal_case(nomination_case, monkeypatch):
    """Actual Portal claim and prompt cleanup with nomination as the prompt seam."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = nomination_case
    root = case.daemon.repo_root
    todo = root / "tasks.todo.md"
    todo.write_text("# Authored resource deferral control\n")
    daemon = impl.TodoImplementationDaemon(
        todo_path=todo, state_path=root / "state/tasks.json", strategy_path=root / "state/strategy.json",
        events_path=root / "state/events.jsonl", repo_root=root, task_header_prefix="## RESOURCE-",
        implement=True,
    )
    daemon._task_context_nomination_bundle = case.daemon._task_context_nomination_bundle
    case.daemon = daemon
    monkeypatch.setattr(daemon, "_require_primary_provider_readiness", lambda task: None)
    monkeypatch.setattr(daemon, "_build_implementation_prompt",
                        lambda task, attempt: daemon._task_metadata_snapshot(task, include_context=True))
    monkeypatch.setattr(daemon, "_try_acquire_implementation_lock",
                        lambda *args, **kwargs: pytest.fail("resource deferral must precede execution lock"))
    case.state = impl.PortalTaskState()
    case.run = lambda: daemon._run_implementation(case.task, case.state)
    try:
        yield case
    finally:
        daemon.close_event_runtime()


def test_portal_resource_deferral_releases_its_exact_dispatch_claim(portal_case, native_child_timeout):
    case = portal_case
    case.failure = native_child_timeout
    result = case.run()
    assert result["deferred"] is True
    assert result["reason"] == "source384_resource_admission_deferred"
    assert result["attempt_consumed"] is False
    assert result["provider_dispatched"] is False
    assert result["backoff_seconds"] == 5
    assert "source384_resource_deferral" in result
    claim_path = case.daemon._implementation_task_claim_path(
        case.task.task_id, canonical_task_cid=case.daemon._canonical_ref(case.task))
    assert not claim_path.exists()
    assert not case.state.implementation_in_progress
    assert case.state.implementation_attempts == {}
    assert case.state.implementation_attempts_by_cid == {}


@pytest.mark.parametrize("cleanup", ["false", "foreign_claim", "retained_fence"])
def test_failed_exact_claim_cleanup_cannot_emit_resource_retry_authority(
    portal_case, native_child_timeout, monkeypatch, cleanup,
):
    case = portal_case
    case.failure = native_child_timeout
    refused = []
    release = case.daemon._release_implementation_task_claim

    def retain_claim(path, metadata):
        refused.append((path, metadata))
        if cleanup == "foreign_claim":
            observed = json.loads(path.read_bytes())
            observed["lease_id"] = "foreign-authored-lease"
            path.write_text(json.dumps(observed))
            return release(path, metadata)
        # Native cleanup can report success while retaining a protected fence.
        return cleanup == "retained_fence"

    monkeypatch.setattr(case.daemon, "_release_implementation_task_claim", retain_claim)
    with pytest.raises(Exception):
        case.run()
    assert len(refused) == 1
    assert refused[0][0].exists()
    if cleanup == "foreign_claim":
        assert json.loads(refused[0][0].read_bytes())["lease_id"] == "foreign-authored-lease"
    assert not case.state.implementation_in_progress


@pytest.mark.parametrize("timeout", [True, 0, -1, 90.01, float("nan"), float("inf"), "10"])
def test_invalid_nomination_timeout_refuses_before_reading_history(nomination_case, monkeypatch, timeout):
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles

    case = nomination_case
    monkeypatch.setattr(bundles, "read_task_context_historical_selection",
                        lambda **kwargs: pytest.fail("invalid bound reached historical read"))
    with pytest.raises(ValueError):
        bundles.load_task_context_selection(
            repository=case.daemon.repo_root,
            artifact=case.daemon._task_context_nomination_bundle["artifact"],
            expected_sha256=case.daemon._task_context_nomination_bundle["sha256"],
            task_cid=case.task.canonical_task_cid, task_id=case.task.task_id,
            source384_timeout_seconds=timeout,
        )


@pytest.mark.parametrize("prefix,validation,refused", [(3., 1., False), (10., 0., True), (2., 9., True)])
def test_historical_read_and_validation_share_original_nomination_deadline(
    nomination_case, monkeypatch, prefix, validation, refused,
):
    """Simulated monotonic time; this is deadline plumbing, not measured recovery."""
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles

    case = nomination_case
    now = [100.]
    monkeypatch.setattr(bundles, "time", SimpleNamespace(monotonic=lambda: now[0]))
    read_history = bundles.read_task_context_historical_selection

    def read(**kwargs):
        result = read_history(**kwargs)
        now[0] += prefix
        return result

    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        now[0] += validation

    monkeypatch.setattr(bundles, "read_task_context_historical_selection", read)
    monkeypatch.setattr(source, "validate_source384_context", validate)
    arguments = dict(repository=case.daemon.repo_root,
                     artifact=case.daemon._task_context_nomination_bundle["artifact"],
                     expected_sha256=case.daemon._task_context_nomination_bundle["sha256"],
                     task_cid=case.task.canonical_task_cid, task_id=case.task.task_id,
                     source384_timeout_seconds=10.)
    if refused:
        with pytest.raises(TimeoutError):
            bundles.load_task_context_selection(**arguments)
    else:
        assert bundles.load_task_context_selection(**arguments)["metadata"] == case.metadata
    if prefix >= 10:
        assert calls == []
    else:
        assert len(calls) == 1 and calls[0]["timeout_seconds"] == 10 - prefix


def test_default_nomination_preserves_implicit_source384_budget(nomination_case):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl
    case = nomination_case
    token = impl._source384_callback_deadline.set(None)
    try:
        assert dict(case.read()) == case.metadata
        assert len(case.calls) == 1 and "timeout_seconds" not in case.calls[0]
    finally:
        impl._source384_callback_deadline.reset(token)


@pytest.fixture
def typed_attempt(tmp_path, monkeypatch):
    """Real local typed owner, kernel-bound grant and durable native claim.

    The Quack extension transport is the project's declared fake transport;
    typed command RPC, DuckDB stores, grant checks and claim mutations are real.
    No provider completion or external model evidence is manufactured.
    """
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnerLiveness
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport, build_server
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackStateClient
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import StateOwnerBootstrapCredentials
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import DETERMINISTIC_ONLY_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource, daemon_required_owner_operations, daemon_required_owner_command_operations,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV, TYPED_STATE_OWNER_TOKEN_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate

    database = tmp_path / "typed.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize({
        "repository_tree_id": "tree:authored-resource-control", "plan_root_cid": "plan:authored-resource-control",
        "goals": [{"goal_cid": "goal:authored-resource-control", "goal_alias": "RESOURCE-G001",
                   "title": "Preserve exact pre-dispatch custody"}],
        "tasks": [{"task_cid": "task:resource-fixture", "task_id": "RESOURCE-001",
                   "goal_cid": "goal:authored-resource-control", "status": "ready"}],
    })
    source.close()
    server = build_server(
        database_path=database, state_dir=tmp_path / "owner", repository_id="repository:resource-control",
        store_id="resource-control", transport=FakeQuackTransport(), capability_probe=_capability,
        migrate=_migrate, connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    client_id = "database-implementation-daemon:resource-control"
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id, process_birth_id=identity.process_birth_id,
        allowed_operations=daemon_required_owner_operations(),
        allowed_command_operations=daemon_required_owner_command_operations(), peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path()))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(owner_id=client_id, store_id=identity.store_id,
                             process_birth_id=identity.process_birth_id)
    adapter = daemon = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        route = unsealed.seal_execution_route_policy({"RESOURCE-001": DETERMINISTIC_ONLY_EXECUTION_MODE})
        unsealed.close()
        adapter = TypedDatabaseTaskSource(client, execution_route_policy=route)
        credentials = StateOwnerBootstrapCredentials(
            endpoint=identity.listen_uri, socket_path=str(server.typed_command_socket_path()),
            store_id=identity.store_id, server_id=identity.server_id, client_id=client_id,
            process_birth_id=identity.process_birth_id, token=token, execution_route_policy=route,
        )
        daemon = DatabaseImplementationDaemon(
            database_path=database, coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb", owner_session_id="session:resource-control",
            process_instance_id=identity.process_birth_id, authority_mode="quack", task_source_kind="duckdb",
            quack_uri=identity.listen_uri, task_source=adapter, close_task_source=False,
            state_owner_bootstrap_credentials=credentials, lease_ms=30000,
            max_task_attempts=4, strict_task_sharding=True, require_real_execution=True,
        ).open()
        attempt = daemon.claim_next()
        assert attempt is not None
        yield SimpleNamespace(daemon=daemon, attempt=attempt, adapter=adapter, grant=grant)
    finally:
        if daemon is not None:
            daemon.close()
        if adapter is not None:
            adapter.close()
        client.close()
        server.stop()


@pytest.fixture
def native_deferral_bridge(typed_attempt, portal_case, native_child_timeout, tmp_path):
    """Native bridge consumes the actual prompt-boundary result.

    Its Portal adapter exposes the existing authored database-authority test
    protocol. The prompt guard, local claims, typed task and outer callback
    dispatch records use their production implementations.
    """
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from test.api.test_agent_supervisor_database_portal_bridge import _CompletingPortal

    case = portal_case
    case.failure = native_child_timeout
    calls = []

    class ContextPortal(_CompletingPortal):
        def run_once(self):
            calls.append(True)
            if case.failure is None:
                case.daemon._task_metadata_snapshot(case.task, include_context=True)
                return super().run_once()
            return {"implementation_result": case.run()}

        def close_event_runtime(self):
            pass  # The actual Portal fixture owns this runtime's cleanup.

    bridge = DatabasePortalExecutionBridge(
        task_source=typed_attempt.adapter, attempt_root=tmp_path / "bridge-attempts",
        portal_factory=lambda paths, alias: ContextPortal(paths, alias), max_task_attempts=4,
    )
    typed_attempt.daemon._provider_fn = bridge.run_provider
    return SimpleNamespace(bridge=bridge, calls=calls, owner=typed_attempt, portal=case)


def test_exact_native_resource_deferral_retries_same_claim_once(
    native_deferral_bridge, record_property, monkeypatch,
):
    case = native_deferral_bridge
    daemon, attempt = case.owner.daemon, case.owner.attempt
    task_before = case.owner.adapter.get(attempt.task_cid)
    control_binding = deepcopy(attempt.body["control_binding"])
    started = time.monotonic()
    result = daemon._resume_attempt_without_process_crash(attempt)
    assert result["deferred"] is True
    assert result["reason"] == "source384_resource_admission_deferred"
    key = f"provider:{attempt.attempt_id}"
    prior = dict(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key))
    assert prior["callback_state"] == "not_dispatched"
    assert prior["process_instance_id"] == daemon.process_instance_id
    assert prior["retry_not_before_ms"] < prior["retry_deadline_ms"]
    assert case.owner.adapter.get(attempt.task_cid) == task_before
    assert daemon.get_attempt(attempt.attempt_id).body["control_binding"] == control_binding
    assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
    assert case.calls == [True]

    early = daemon._resume_attempt_without_process_crash(daemon.get_attempt(attempt.attempt_id))
    assert early["deferred"] is True
    assert case.calls == [True]
    assert dict(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)) == prior
    remaining = max(0., (prior["retry_not_before_ms"] - daemon._now_ms()) / 1000.)
    time.sleep(remaining + 0.02)
    accepted_calls = []
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationAuthorityError
    with pytest.raises(DatabaseImplementationAuthorityError):
        daemon.run_provider(daemon.get_attempt(attempt.attempt_id),
                            provider_fn=lambda current: pytest.fail("foreign callback gained retry authority"))
    assert dict(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)) == prior

    def accepted_response(**kwargs):
        accepted_calls.append(kwargs["attempt"].attempt_id)
        return {"status": "ok", "accepted": True, "authored_control": True}

    # Keep the exact bridge callback. Only accepted external work evidence is
    # authored here; this control ends at provider phase, never task completion.
    monkeypatch.setattr(case.bridge, "_acceptance_receipt", accepted_response)
    case.portal.failure = None
    current = daemon.get_attempt(attempt.attempt_id)
    updated, returned, duplicated = daemon.run_provider(current)
    assert duplicated is False and returned["accepted"] is True
    assert accepted_calls == [attempt.attempt_id]
    assert updated.attempt_id == attempt.attempt_id and updated.claim_id == attempt.claim_id
    assert updated.body["control_binding"] == control_binding
    replayed, repeated, duplicated = daemon.run_provider(updated)
    assert duplicated is True and repeated == returned
    assert accepted_calls == [attempt.attempt_id]
    assert case.calls == [True, True]
    assert replayed.attempt_id == attempt.attempt_id
    assert case.owner.adapter.get(attempt.task_cid) == task_before
    assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
    record_property("source384_native_retry", json.dumps({
        "actual_scheduler_child_timeout": True, "authored_pressure_sampler": True,
        "typed_command_rpc": True, "quack_extension_transport": "declared_fixture",
        "real_cooldown_wait_seconds": remaining + 0.02,
        "elapsed_seconds": time.monotonic() - started,
        "same_claim_and_control": True, "authored_provider_acceptances": len(accepted_calls),
        "external_provider_or_model_calls": 0, "task_completion_claimed": False,
    }, sort_keys=True))


@pytest.mark.parametrize("damage", ["missing_expiry", "expired_monotonic", "foreign_callback", "missing_callback", "committed_callback"])
def test_native_settlement_refuses_unproved_expiry_or_changed_callback_custody(
    native_deferral_bridge, monkeypatch, damage,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = native_deferral_bridge
    daemon, attempt = case.owner.daemon, case.owner.attempt
    task_before = case.owner.adapter.get(attempt.task_cid)
    key = f"provider:{attempt.attempt_id}"
    settle = daemon._settle_source384_resource_deferral
    observed = []

    def changed(current, **kwargs):
        connection = daemon._require_connection()
        receipt = daemon.provider_invocation_recorded(current.attempt_id, idempotency_key=key)
        assert receipt["callback_state"] == "started_outcome_unknown"
        if damage == "missing_expiry":
            daemon._source384_retry_deadline_ms = None
        elif damage == "expired_monotonic":
            daemon._source384_retry_deadline_monotonic = time.monotonic() - 1.
        elif damage == "foreign_callback":
            receipt = {**receipt, "claim_id": "claim:foreign-authored-custody"}
            connection.execute("UPDATE provider_invocations SET result_json=? WHERE attempt_id=?",
                               [impl._database_daemon_json(receipt), current.attempt_id])
        elif damage == "missing_callback":
            connection.execute("DELETE FROM provider_invocations WHERE attempt_id=?", [current.attempt_id])
            receipt = None
        else:
            receipt = {"status": "ok", "accepted": True, "authored_committed_custody": True}
            connection.execute("UPDATE provider_invocations SET result_json=? WHERE attempt_id=?",
                               [impl._database_daemon_json(receipt), current.attempt_id])
        observed.append(deepcopy(receipt))
        return settle(current, **kwargs)

    monkeypatch.setattr(daemon, "_settle_source384_resource_deferral", changed)
    expected = TimeoutError if damage == "expired_monotonic" else impl.DatabaseImplementationAuthorityError
    with pytest.raises(expected):
        daemon._resume_attempt_without_process_crash(attempt)
    assert len(observed) == 1
    assert daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key) == observed[0]
    assert case.calls == [True]
    assert case.owner.adapter.get(attempt.task_cid) == task_before
    assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"


@pytest.mark.parametrize("changed_context", [False, True])
def test_repeated_native_deferral_cannot_renew_deadline_or_change_context(
    native_deferral_bridge, monkeypatch, changed_context, record_property,
):
    """Six seconds of owner wall time are simulated; no elapsed recovery claim."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as impl

    case = native_deferral_bridge
    daemon, attempt = case.owner.daemon, case.owner.attempt
    key = f"provider:{attempt.attempt_id}"
    grant_deadline = (daemon._source384_retry_deadline_ms, daemon._source384_retry_deadline_monotonic)
    first = daemon._resume_attempt_without_process_crash(attempt)
    assert first["deferred"] is True
    prior = deepcopy(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key))
    prior_claim_expiry = daemon.coordinator.get_task_claim(attempt.claim_id).expires_at_ms
    actual_now = daemon._now_ms
    monkeypatch.setattr(daemon, "_now_ms", lambda: actual_now() + 6000)
    if changed_context:
        settle = daemon._settle_source384_resource_deferral

        def change(current, **kwargs):
            kwargs["failure"].no_dispatch["context"]["context_sha256"] = "f" * 64
            return settle(current, **kwargs)

        monkeypatch.setattr(daemon, "_settle_source384_resource_deferral", change)
        with pytest.raises(impl.DatabaseImplementationAuthorityError):
            daemon._resume_attempt_without_process_crash(daemon.get_attempt(attempt.attempt_id))
        current = daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)
        assert current["callback_state"] == "started_outcome_unknown"
        assert "retry_count" not in current
    else:
        second = daemon._resume_attempt_without_process_crash(daemon.get_attempt(attempt.attempt_id))
        assert second["deferred"] is True
        current = daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)
        assert current["retry_count"] == prior["retry_count"] + 1
        assert current["retry_not_before_ms"] > prior["retry_not_before_ms"]
        assert current["retry_deadline_ms"] == prior["retry_deadline_ms"]
        assert current["context"] == prior["context"]
        assert daemon.coordinator.get_task_claim(attempt.claim_id).expires_at_ms > prior_claim_expiry
    assert (daemon._source384_retry_deadline_ms, daemon._source384_retry_deadline_monotonic) == grant_deadline
    assert case.calls == [True, True]
    record_property("source384_repeat_scope", json.dumps({
        "simulated_owner_clock_advance_seconds": 6, "grant_deadline_unchanged": True,
        "context_change_refused": changed_context, "actual_host_pressure_recovery": False,
    }, sort_keys=True))


def test_raw_admission_failure_inside_callback_retains_unknown_custody(
    native_deferral_bridge, native_child_timeout, monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalBridgeDeferred
    from test.api.test_agent_supervisor_database_portal_bridge import _DatabaseAttemptAuthorityPortal

    case = native_deferral_bridge
    daemon, attempt = case.owner.daemon, case.owner.attempt
    entered = []

    class UnresolvedPortal(_DatabaseAttemptAuthorityPortal):
        def run_once(self):
            entered.append(True)
            raise native_child_timeout

        def close_event_runtime(self):
            pass

    monkeypatch.setattr(case.bridge, "portal_factory", lambda paths, alias: UnresolvedPortal())
    with pytest.raises(resources.LeaseTimeoutError) as caught:
        daemon._resume_attempt_without_process_crash(attempt)
    assert caught.value is native_child_timeout
    key = f"provider:{attempt.attempt_id}"
    receipt = deepcopy(daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key))
    assert receipt["callback_state"] == "started_outcome_unknown"
    with pytest.raises(DatabasePortalBridgeDeferred, match="provider_callback_outcome_unknown"):
        daemon.run_provider(daemon.get_attempt(attempt.attempt_id))
    assert entered == [True]
    assert daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key) == receipt
