"""Stale coordination rows must not authorize inactive owner tasks."""
from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport, _allocate_loopback_port,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


@pytest.fixture(params=["embedded", "quack"])
def daemon(request, tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "control.duckdb"
    population = {
        "repository_tree_id": "tree:claim-eligibility-test",
        "objectives": [{"objective_id": "objective:test", "objective_alias": "TEST-O",
            "goal_cid": "goal:test", "goal_alias": "TEST-G", "title": "Eligibility test", "status": "open"}],
        "tasks": [{"task_cid": f"task:{i}", "task_id": f"TEST-{i:03}", "goal_cid": "goal:test",
            "ordinal": i, "title": f"Temporary task {i}", "status": "ready", "priority": "P0"}
            for i in (1, 2)],
    }
    with DatabaseTaskSource(database) as source:
        source.materialize(population)
    owner, transport = None, None
    endpoint = ""
    try:
        if request.param == "quack":
            owner = duckdb.connect(str(database), config={"autoinstall_known_extensions": False})
            try:
                owner.execute("LOAD quack")
            except Exception as exc:
                pytest.skip(f"installed Quack unavailable: {type(exc).__name__}")
            token, port = secrets.token_urlsafe(32), _allocate_loopback_port()
            monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
            endpoint = f"quack:127.0.0.1:{port}"
            transport = InProcessQuackTransport()
            identity = SimpleNamespace(server_id="eligibility-test", store_id="eligibility-store",
                database_uuid="eligibility-database", schema_revision=1, schema_fingerprint="test-schema",
                generation=1, process_birth_id="test-birth")
            transport.start(owner, host="127.0.0.1", port=port, token=token, identity=identity)
        instance = DatabaseImplementationDaemon(database_path=database,
            authority_mode=request.param, task_source_kind="duckdb", quack_uri=endpoint,
            owner_session_id="eligibility-test", require_real_execution=True)
        try:
            instance.sync_ready_tasks_into_coordination()
            yield instance
        finally:
            instance.close()
    finally:
        try:
            if transport is not None:
                transport.stop(owner)
        finally:
            if owner is not None:
                owner.close()


def change_status(daemon, status):
    task = daemon.task_source.get("task:1")
    return daemon.task_source.compare_and_set_status(task.task_cid, task.revision, status,
        receipt={"operation": "test-authoritative-eligibility-change"})


@pytest.mark.parametrize("status", ["blocked", "cancelled"])
def test_stale_registered_inactive_task_is_preserved_and_independent_task_runs(daemon, status):
    changed = change_status(daemon, status).task
    claim = daemon.claim_next()
    assert claim.task_cid == "task:2"
    preserved = daemon.task_source.get("task:1")
    assert preserved.status == status and preserved.revision == changed.revision
    assert daemon.task_source.get("task:2").status == "in_progress"
    assert len(daemon.list_running_attempts()) == 1


def test_changed_dependency_and_preflight_block_override_stale_registry(daemon):
    daemon.task_source.intent.set_task_dependencies("task:1", ["task:2"])
    daemon.task_source.intent.block_task(task_cid="task:2", blocker_kind="preflight",
        blocker_id="missing-test-tool", reason="Temporary test prerequisite unavailable")
    assert not daemon.task_source.ready_tasks().tasks
    assert daemon.claim_next() is None
    assert not daemon.coordinator.list_active_leases()
    assert not daemon.list_running_attempts()
    assert daemon.task_source.get("task:1").status == "ready"
    assert daemon.task_source.get("task:2").status == "blocked"


def test_dependency_revocation_after_coordination_claim_releases_only_that_claim(daemon, monkeypatch):
    original = daemon.coordinator.claim_ready_task
    def revoke_dependency(**kwargs):
        claim = original(**kwargs)
        assert claim.task_cid == "task:1"
        daemon.task_source.intent.set_task_dependencies("task:1", ["task:2"])
        return claim
    monkeypatch.setattr(daemon.coordinator, "claim_ready_task", revoke_dependency)
    assert daemon.claim_next() is None
    assert not daemon.coordinator.list_active_leases()
    assert not daemon.list_running_attempts()
    assert daemon.task_source.get("task:1").status == "ready"
    monkeypatch.setattr(daemon.coordinator, "claim_ready_task", original)
    assert daemon.claim_next().task_cid == "task:2"


def test_peer_status_change_between_ready_read_and_cas_is_not_overwritten(daemon, monkeypatch):
    original = daemon._cas_task_status_database
    def peer_cancels(task_cid, **kwargs):
        assert task_cid == "task:1"
        # This opens a separate authoritative connection, including a second
        # remote SQL session for the genuine Quack parameterization.
        change_status(daemon, "cancelled")
        return original(task_cid, **kwargs)
    monkeypatch.setattr(daemon, "_cas_task_status_database", peer_cancels)
    assert daemon.claim_next() is None
    assert daemon.task_source.get("task:1").status == "cancelled"
    assert daemon.task_source.get("task:2").status == "ready"
    assert not daemon.coordinator.list_active_leases()
    assert not daemon.list_running_attempts()
    monkeypatch.setattr(daemon, "_cas_task_status_database", original)
    assert daemon.claim_next().task_cid == "task:2"


@pytest.mark.parametrize("status", ["blocked", "cancelled"])
def test_expiry_does_not_requeue_an_intervening_authoritative_status(daemon, monkeypatch, status):
    attempt = daemon.claim_next()
    assert attempt.task_cid == "task:1"
    changed = change_status(daemon, status).task
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    monkeypatch.setattr(daemon, "_clock_ms", lambda: claim.expires_at_ms + 1)
    outcomes = daemon.reconcile_expired_running_attempts()
    assert len(outcomes) == 1 and outcomes[0]["status"] == "expired"
    preserved = daemon.task_source.get("task:1")
    assert preserved.status == status and preserved.revision == changed.revision
    assert daemon.task_source.get("task:2").status == "ready"
    assert not daemon.list_running_attempts()


@pytest.mark.parametrize("body", [{"review only": "true"}, {"is schedulable": "false"},
                                  {"review_only": True}, {"completion": "manual"}])
def test_native_metadata_cannot_make_inactive_task_automatic(body):
    assert DatabaseImplementationDaemon._automatic_claim_forbidden(SimpleNamespace(body=body))
