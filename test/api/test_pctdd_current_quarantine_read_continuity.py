"""Independent-work admission owns one complete quarantine/binding read."""
from __future__ import annotations

import duckdb
import pytest
import shutil
import json
import os
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
from ipfs_accelerate_py.agent_supervisor.task_sources import owner_task_quarantine as quarantine
from ipfs_accelerate_py.agent_supervisor.task_sources import quack_read_continuity as continuity
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepositoryReadUnavailableError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationProviderDispatchError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.owner_task_quarantine import ACK_KEY
from test.api.test_pctdd_quarantine_read_continuity import owned, withdraw, journal  # noqa: F401
from test.api.test_owner_task_quarantine import apply, quarantine_request
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population
from test.api.test_pctdd_task_read_continuity import (
    _admitted_observation, _isolation_receipt, _isolation_server_kwargs,
    build_server, probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationExpiredError


@pytest.fixture
def provider_owned(tmp_path, monkeypatch):
    """Move the exact private claimed source into a real Quack fixture owner."""
    local = tmp_path / "daemon"
    local.mkdir()
    clock = {"now": 1000}
    daemon = _open_daemon(local, clock_ms=lambda: clock["now"])
    server = None
    try:
        population = _population(1)
        population["tasks"][0]["task_cid"] = "task:test"
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        daemon.task_source.intent.close()
        database = tmp_path / "control" / "control.duckdb"
        database.parent.mkdir()
        shutil.copyfile(local / "control.duckdb", database)
        receipt_path, receipt = _isolation_receipt(tmp_path)
        server = build_server(database_path=database, state_dir=receipt_path.parent,
            **_isolation_server_kwargs(receipt), store_id=str(database),
            repository_id="repository:test", isolation_receipt_path=receipt_path,
            isolation_observer=_admitted_observation,
            capability_probe=lambda **_: probe_quack_capabilities())
        identity = server.start()
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        for key, value in {
            "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": identity.secret_handle,
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": str(database),
            "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(identity.generation),
            "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(identity.schema_revision),
            "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path),
        }.items():
            monkeypatch.setenv(key, value)
        repo = IntentRepository(identity.listen_uri, install_schema=False)
        daemon.task_source._intent = repo
        yield server, repo, lambda: (daemon, attempt, clock)
    finally:
        daemon.close()
        if server is not None:
            server.stop()
        references = []
        for fd in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == str(tmp_path) or target.startswith(str(tmp_path) + "/"):
                references.append(target)
        assert not references and not list(tmp_path.rglob(".git"))
        details = tmp_path.stat()
        with (tmp_path.parent / "closed-provider-fixtures.jsonl").open("a") as ledger:
            ledger.write(json.dumps({"path": str(tmp_path), "device": details.st_dev,
                "inode": details.st_ino, "owner_closed": True,
                "own_open_references": references}) + "\n")
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize("select", [1, 2, 3])
def test_current_guard_replica_loss_does_not_replay_provider(provider_owned, monkeypatch, select):
    _, _, make_daemon = provider_owned
    daemon, attempt, _ = make_daemon()
    callbacks = []
    unknown = RuntimeError("provider entered; durable outcome unavailable")

    def provider(current):
        callbacks.append(current.attempt_id)
        raise unknown

    calls, handles, closed, deadlines = withdraw(provider_owned, monkeypatch, select)
    with pytest.raises(DatabaseImplementationProviderDispatchError,
                       match="callback raised without admissible return evidence") as caught:
        daemon.run_provider(attempt, provider_fn=provider)
    assert caught.value.__cause__ is unknown
    assert callbacks == [attempt.attempt_id]
    assert len(calls[0]) == select and len(calls[1]) == 3
    assert "count(*)" in calls[1][0] and "ORDER BY global_sequence" in calls[1][1]
    assert "control_plane_metadata" in calls[1][2]
    assert deadlines[0] == deadlines[1] and deadlines[0] is not None
    assert handles == closed
    before = journal(daemon)
    assert len(before) == 1
    with pytest.raises(DatabaseImplementationProviderDispatchError, match="outcome is unknown"):
        daemon.run_provider(attempt, provider_fn=provider)
    assert callbacks == [attempt.attempt_id] and journal(daemon) == before


def test_current_guard_retry_does_not_bypass_expired_claim(provider_owned, monkeypatch):
    _, _, make_daemon = provider_owned
    daemon, attempt, clock = make_daemon()
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    callbacks = []
    withdraw(provider_owned, monkeypatch, 3,
             after_loss=lambda: clock.update(now=claim.expires_at_ms + 1))
    with pytest.raises(DatabaseCoordinationExpiredError):
        daemon.run_provider(attempt, provider_fn=lambda current: callbacks.append(current))
    assert callbacks == [] and journal(daemon) == []


@pytest.mark.parametrize("field", sorted(continuity._BINDING_FIELDS))
def test_current_guard_refuses_changed_owner_before_provider(owned, monkeypatch, field):
    _, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    callbacks = []
    _, handles, closed, _ = withdraw(owned, monkeypatch, 2, drift=field)
    with pytest.raises(IntentRepositoryReadUnavailableError, match="binding"):
        daemon.run_provider(attempt, provider_fn=lambda current: callbacks.append(current))
    assert callbacks == [] and journal(daemon) == [] and handles == closed


def test_current_guard_observes_new_local_fence_without_inventing_ack(owned, monkeypatch):
    server, _, make_daemon = owned
    daemon, attempt, _ = make_daemon()
    callbacks = []
    token = server._vault.resolve(server.identity.secret_handle)

    def install_fence():
        apply(server, quarantine_request(server, token,
            execution_store_id=daemon.execution_store_identity,
            execution_owner_id=daemon.owner_session_id))

    withdraw(owned, monkeypatch, 2, after_loss=install_fence)
    # The synthetic local daemon has no native Quack owner/ack custody. It must
    # fail its existing owner gate before any callback or acknowledgement write.
    with pytest.raises(quarantine.QuarantineDenied):
        daemon.run_provider(attempt, provider_fn=lambda current: callbacks.append(current))
    assert callbacks == [] and journal(daemon) == []
    assert daemon._require_connection().execute(
        "SELECT count(*) FROM daemon_execution_metadata WHERE key LIKE ?", [ACK_KEY + ":%"]
    ).fetchone()[0] == 0


def test_observation_returns_same_handle_binding_and_complete_heads(owned, monkeypatch):
    server, repo, make_daemon = owned
    daemon, _, _ = make_daemon()
    token = server._vault.resolve(server.identity.secret_handle)
    apply(server, quarantine_request(server, token))
    _, handles, closed, _ = withdraw(owned, monkeypatch, 3)
    heads, binding = repo.owner_task_quarantine_observation()
    assert heads["attempt:retained"]["owner_binding"] == binding == server._mutation_binding()
    assert handles == closed and binding is not handles[-1]._quack_mutation_binding
    assert daemon._current_owner_task_quarantines() == {}  # foreign execution store


@pytest.mark.parametrize("drift", ["binding", "binding_type", "transaction", "pending"])
def test_successful_projection_cannot_return_changed_handle_custody(owned, monkeypatch, drift):
    _, repo, _ = owned
    original = state._open_quack_transport_connection_once
    handles = []

    def opened(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles.append(connection)
        return connection

    heads = quarantine.heads
    def changed(connection):
        result = heads(connection)
        if drift == "binding":
            connection._quack_mutation_binding["generation"] += 1
        elif drift == "binding_type":
            connection._quack_mutation_binding["generation"] = True
        elif drift == "transaction":
            connection.execute("BEGIN")
        else:
            connection._quack_pending_mutations.append({"unknown": True})
        return result

    monkeypatch.setattr(state, "_open_quack_transport_connection_once", opened)
    monkeypatch.setattr(quarantine, "heads", changed)
    with pytest.raises(IntentRepositoryReadUnavailableError, match="binding changed"):
        repo.owner_task_quarantine_observation()
    assert len(handles) == 1


@pytest.mark.parametrize("select", [1, 2, 3])
def test_peer_receive_loss_retries_owned_projection_only(owned, monkeypatch, select):
    server, repo, _ = owned
    original = state._open_quack_transport_connection_once
    handles, transport_errors = [], []

    def opened(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles.append(connection)
        execute = connection.execute
        if len(handles) == 1:
            count = 0
            def query(*a, **kw):
                nonlocal count
                count += 1
                if count != select:
                    return execute(*a, **kw)
                server._stop_transport_connection(observe_closed=True)
                try:
                    execute(*a, **kw)
                except duckdb.IOException as error:
                    transport_errors.append(error)
                    # The fixture closes the endpoint before SELECT, whereas
                    # the retained production failure was mid-response.
                    raise duckdb.IOException(
                        "IO Error: Failed to send message: IO Error: Failure when receiving data from the peer\n"
                        "error for HTTP POST to 'http://127.0.0.1:1/quack'") from error
                finally:
                    server._refresh_read_replica()
            connection.execute = query
        return connection

    monkeypatch.setattr(state, "_open_quack_transport_connection_once", opened)
    assert repo.owner_task_quarantine_observation()[0] == {}
    assert len(handles) == 2 and len(transport_errors) == 1


@pytest.mark.parametrize("error", [
    RuntimeError("IO Error: Failed to send message: IO Error: Failure when receiving data from the peer"),
    duckdb.IOException("IO Error: Failure when receiving data from the peer"),
    duckdb.IOException("IO Error: Failed to send message: IO Error: unrelated failure"),
    duckdb.IOException("read_replica_refresh_unknown_outcome canonical_effects_present=True"),
])
def test_peer_retry_classifier_does_not_accept_other_failures(error):
    assert continuity._replica_connection_lost(error) is False
