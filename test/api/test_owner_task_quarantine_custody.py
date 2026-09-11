"""Native retained rows survive ordinary claim/expiry and callback paths."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from ipfs_accelerate_py.agent_supervisor.merge import owner_task_quarantine as custody
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    owner_task_quarantine as local,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.owner_task_quarantine import (
    QuarantineDenied,
)
from test.api.test_agent_supervisor_database_portal_bridge import (
    _seed_interrupted_database_portal_attempt,
)


def entered(tmp_path):
    repo, daemon, bridge, attempt, paths = _seed_interrupted_database_portal_attempt(
        tmp_path, seed_nested_state=False
    )
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    daemon._record_database_portal_attempt_binding(
        attempt, json.loads(paths.binding.read_text()), "portal_entered"
    )
    return repo, daemon, bridge, attempt, paths


def install_custody(daemon, attempt):
    observed = local.capture(daemon, attempt)
    coordinator = daemon.coordinator
    with coordinator._lock:
        connection = coordinator._require()
        coordinator._begin(connection)
        custody.install(
            connection,
            task_cid=attempt.task_cid,
            event_id="event:test-native-capture",
            retained_state_cid=observed["retained"]["retained_state_cid"],
            expected_snapshot=observed["coordination"],
        )
        coordinator._commit_if_idle(connection)
    return observed


def test_native_capture_requires_entered_unknown_not_missing_file_alone(tmp_path):
    _, daemon, _, attempt, paths = _seed_interrupted_database_portal_attempt(
        tmp_path, seed_nested_state=False
    )
    try:
        with pytest.raises(QuarantineDenied, match="binding_missing"):
            local.capture(daemon, attempt)
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt, json.loads(paths.binding.read_text()), "portal_entered"
        )
        observed = local.capture(daemon, attempt)
        assert observed["retained"]["diagnosis"] == "entered_callback_state_missing"
        assert not paths.state.exists()
        assert local.capture(daemon, attempt) == observed
    finally:
        daemon.close()


def test_expiry_sweep_preserves_custody_while_independent_claim_succeeds(tmp_path):
    _, daemon, _, attempt, _ = entered(tmp_path)
    try:
        before = install_custody(daemon, attempt)
        coordinator = daemon.coordinator
        claim = coordinator.get_task_claim(attempt.claim_id)
        coordinator.register_task(task_cid="task:independent", task_id="PCTDD-008")
        next_claim = coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            now_ms=claim.expires_at_ms + 1,
            exclude_task_cids={attempt.task_cid},
        )
        assert next_claim is not None and next_claim.task_cid == "task:independent"
        assert (
            custody.snapshot(coordinator._require(), attempt.task_cid)
            == before["coordination"]
        )
        assert (
            local.execution_snapshot(daemon._require_connection(), attempt.task_cid)
            == before["execution"]
        )
        assert coordinator.get_task_claim(attempt.claim_id) == claim
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "operation", ["release", "renew", "protect", "resume", "expire"]
)
def test_direct_retained_mutations_cannot_escape_commit_fence(tmp_path, operation):
    _, daemon, _, attempt, _ = entered(tmp_path)
    try:
        before = install_custody(daemon, attempt)
        coordinator = daemon.coordinator
        claim = coordinator.get_task_claim(attempt.claim_id)
        with pytest.raises(QuarantineDenied):
            if operation == "release":
                coordinator.release(claim.as_fenced_lease())
            elif operation == "renew":
                coordinator.renew(claim.as_fenced_lease())
            elif operation == "protect":
                daemon._protect_attempt_write(attempt)
            elif operation == "resume":
                daemon.resume_attempt(
                    attempt,
                    provider_fn=lambda _: pytest.fail("retained provider invoked"),
                )
            else:
                coordinator.expire_task_claim(claim, now_ms=claim.expires_at_ms + 1)
        assert (
            custody.snapshot(coordinator._require(), attempt.task_cid)
            == before["coordination"]
        )
        assert (
            local.execution_snapshot(daemon._require_connection(), attempt.task_cid)
            == before["execution"]
        )
    finally:
        daemon.close()


@pytest.mark.parametrize("admission", ["explicit", "automatic"])
def test_native_daemon_and_real_quack_keep_unknown_attempt_while_claiming_independent(
    tmp_path, monkeypatch, admission
):
    import shutil
    import threading
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        probe_quack_capabilities,
    )
    from test.api.test_agent_supervisor_quack_owner_mutation import (
        _isolation_receipt,
        _isolation_server_kwargs,
        _admitted_observation,
    )

    repo, daemon, _, attempt, _ = entered(tmp_path)
    server = None
    consumer = None
    stopping = threading.Event()
    try:
        daemon.task_source.intent.upsert_task(
            task_cid="task:independent",
            task_alias="PCTDD-008",
            goal_cid="goal:pctdd",
            status="ready",
        )
        (tmp_path / "control").mkdir()
        database = tmp_path / "control" / "control.duckdb"
        shutil.copyfile(repo / "control.duckdb", database)
        receipt_path, receipt = _isolation_receipt(tmp_path)
        server = build_server(
            database_path=database,
            state_dir=receipt_path.parent,
            **_isolation_server_kwargs(receipt),
            store_id=str(database),
            repository_id="repository:test",
            isolation_receipt_path=receipt_path,
            isolation_observer=_admitted_observation,
            capability_probe=lambda **_: probe_quack_capabilities(),
        )
        identity = server.start()
        for key, value in {
            "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": identity.secret_handle,
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": str(database),
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "test-quarantine",
            "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": "test-quarantine",
            "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(
                identity.generation
            ),
            "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(
                identity.schema_revision
            ),
            "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path),
        }.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)

        def consume():
            while not stopping.is_set():
                server.service_mutation_inbox(max_requests=8)
                stopping.wait(0.01)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        daemon._task_source.close()
        daemon._task_source = DatabaseTaskSource(
            identity.listen_uri, install_schema=False, owner_id=daemon.owner_session_id
        )
        daemon._database_portal_bridge.task_source = daemon._task_source
        before = local.capture(daemon, attempt)
        if admission == "automatic":
            # The disposable source image is now owned by the real Quack
            # producer/consumer above. Exercise the normal Quack recovery policy.
            daemon.authority_mode = "quack"
            first = daemon.reconcile_quiesced_database_portal_attempts(
                trigger="automatic-first-pass", force=True
            )
            assert first["owner_task_quarantines_admitted"] == [attempt.attempt_id]
            assert first["blocked"] is True and first["continuation_required"] is True
            ack = json.loads(
                daemon._require_connection()
                .execute(
                    "SELECT value FROM daemon_execution_metadata WHERE key = ?",
                    [local.ACK_KEY + ":" + attempt.attempt_id],
                )
                .fetchone()[0]
            )
        else:
            ack = daemon.acknowledge_owner_task_quarantine(attempt.attempt_id)
        assert (
            ack["head"]["retained_state_cid"]
            == before["retained"]["retained_state_cid"]
        )
        assert (
            daemon._current_owner_task_quarantines()[attempt.attempt_id] == ack["head"]
        )
        assert [item.attempt_id for item in daemon.list_running_attempts()] == [
            attempt.attempt_id
        ]
        assert daemon._running_attempts_for_independent_work() == []
        with pytest.raises(
            QuarantineDenied, match="remaining_population_audit_required"
        ):
            daemon.claim_next()
        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="test-independent", force=True
        )
        assert result["blocked"] is False
        next_attempt = daemon.claim_next()
        assert next_attempt is not None and next_attempt.task_cid == "task:independent"
        assert result["independent_work_admitted"] is True
        assert result["reconciled"] is False and result["safe_to_restart"] is False
        from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine

        bridge = daemon._database_portal_bridge
        fresh_leases = []

        def native_provider(current):
            paths, _ = bridge._ensure_attempt_projection(
                current, daemon.task_source.get(current.task_cid)
            )
            portal = bridge.portal_factory(paths, current.task_alias)
            frozen = workspace_quarantine.verify(repo, bridge.workspace_root)
            assert portal.worktree_root == Path(frozen["fresh_root"])
            lease = portal.worktree_pool.acquire(
                cache_key="independent-producer",
                branch_name="test/independent-provider",
            )
            assert not workspace_quarantine.within(lease.path, bridge.workspace_root)
            fresh_leases.append(lease)
            return {
                "status": "workspace_created",
                "accepted": True,
                "workspace_path": str(lease.path),
            }

        next_attempt = daemon.commit_phase(
            next_attempt, "context", body={"context": "bounded test provider"}
        )
        _, provider_result, duplicated = daemon.run_provider(
            next_attempt, provider_fn=native_provider
        )
        assert provider_result["status"] == "workspace_created" and not duplicated
        assert len(fresh_leases) == 1
        assert local.capture(daemon, attempt) == before
        with pytest.raises(QuarantineDenied):
            daemon.resume_attempt(
                attempt, provider_fn=lambda _: pytest.fail("retained provider called")
            )
        assert local.capture(daemon, attempt) == before
        daemon._require_connection().execute(
            "UPDATE daemon_execution_metadata SET value = ? WHERE key = ?",
            ["{}", local.ACK_KEY + ":" + attempt.attempt_id],
        )
        with pytest.raises(QuarantineDenied, match="local_ack_changed"):
            daemon.run_provider(
                next_attempt,
                provider_fn=lambda _: pytest.fail("provider escaped incomplete ack"),
            )
    finally:
        daemon.close()
        stopping.set()
        if consumer is not None:
            consumer.join(timeout=5)
        if server is not None:
            server.stop()


def test_actual_legacy_finalizer_conflict_is_captured_without_repairing_old_rows(
    tmp_path, monkeypatch
):
    import types
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as native,
    )

    _, daemon, bridge, attempt, _ = _seed_interrupted_database_portal_attempt(tmp_path)
    try:
        original = daemon.task_source.get(attempt.task_cid)
        _, binding = bridge._ensure_attempt_projection(attempt, original)
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="deferred",
        )
        daemon._record_database_portal_attempt_binding(
            attempt, binding, "portal_entered"
        )
        release = daemon.coordinator.release

        def crash_after_cas(*args, **kwargs):
            raise RuntimeError("after canonical CAS")

        monkeypatch.setattr(daemon.coordinator, "release", crash_after_cas)
        with pytest.raises(RuntimeError, match="after canonical CAS"):
            daemon._finalize_failed_attempt(attempt, reason="old ordinary failure")
        finalized = daemon.task_source.get(attempt.task_cid)
        monkeypatch.setattr(daemon.coordinator, "release", release)
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:replacement-before-saga",
                "tasks": [
                    {
                        "task_cid": attempt.task_cid,
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "title": "Later replacement",
                        "status": "ready",
                        "validation_commands": ["pytest replacement.py"],
                    }
                ],
            }
        )
        finalize = daemon._finalize_failed_attempt

        def crash_after_barrier(*args, **kwargs):
            saga = daemon._database_portal_terminal_reconciliation_saga(attempt)
            assert saga["stage"] == "commit_barrier"
            assert saga["intended_database_disposition"] == "superseded_attempt_revoked"
            raise RuntimeError("after supersession barrier")

        monkeypatch.setattr(daemon, "_finalize_failed_attempt", crash_after_barrier)
        with pytest.raises(RuntimeError, match="after supersession barrier"):
            daemon.reconcile_quiesced_database_portal_attempts(
                trigger="test", force=True
            )
        replacement = daemon.task_source.get(attempt.task_cid)
        daemon._cas_task_status_database(
            attempt.task_cid,
            expected_revision=replacement.revision,
            new_status=finalized.status,
            receipt=dict(finalized.body["completion_receipt"]),
        )
        # Execute the frozen original native producer. No failed phase, saga,
        # callback, or canonical receipt is manufactured by the test.
        fixture = (
            Path(__file__).parent / "fixtures" / "pctdd_legacy_finalizer_5abf57b9.py"
        )
        namespace = dict(vars(native))
        exec(compile(fixture.read_text(), str(fixture), "exec"), namespace)
        monkeypatch.setattr(
            daemon,
            "_finalize_failed_attempt",
            types.MethodType(namespace["_finalize_failed_attempt"], daemon),
        )
        daemon._reconcile_database_portal_post_cas_transitions(
            bridge=bridge, trigger="native-legacy-reproduction"
        )
        monkeypatch.setattr(daemon, "_finalize_failed_attempt", finalize)
        retained = daemon.get_attempt(attempt.attempt_id)
        assert retained.status == "failed"
        observed = local.capture(daemon, retained)
        assert observed["retained"]["diagnosis"] == "terminal_disposition_conflict"
        before = install_custody(daemon, retained)
        audit = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="unacknowledged-conflict", force=True
        )
        assert audit["blocked"] is True
        assert local.capture(daemon, retained) == before
        assert (
            daemon._database_portal_terminal_reconciliation_saga(retained)["stage"]
            == "commit_barrier"
        )
    finally:
        daemon.close()


def test_retained_expired_database_resource_denies_same_scope_successor(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationError,
    )

    _, daemon, _, attempt, _ = entered(tmp_path)
    try:
        coordinator = daemon.coordinator
        retained = coordinator.claim_resource(
            resource_kind="path",
            resource_id="src/retained",
            path="src/retained",
            repository_id="repository:test",
            owner_session_id=daemon.owner_session_id,
            task_cid=attempt.task_cid,
            lease_ms=5000,
        )
        before = install_custody(daemon, attempt)
        with pytest.raises(DatabaseCoordinationError):
            coordinator.claim_resource(
                resource_kind="path",
                resource_id="src/retained",
                path="src/retained",
                repository_id="repository:test",
                owner_session_id="independent-owner",
                task_cid="task:independent",
                now_ms=retained.expires_at_ms + 1,
            )
        assert (
            custody.snapshot(coordinator._require(), attempt.task_cid)
            == before["coordination"]
        )
    finally:
        daemon.close()
