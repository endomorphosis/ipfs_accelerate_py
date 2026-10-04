"""Bounded attempt-budget and routing-selection evidence.

The ACK budget cases consume a lease from actual native SQL under the new
fenced callback. Routing cases exercise the selector with an inert typed
binding sentinel and an actual AF_UNIX socket; they do not attest a bootstrap,
invoke a proof observer, start a coding worker, or qualify native publication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator, duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError, DatabaseImplementationDaemon,
)


@pytest.mark.skipif(not duckdb_available(), reason="native DuckDB is required")
@pytest.mark.parametrize("remaining_ms", [5001, 5000, 4999])
def test_ack_budget_uses_actual_guarded_sql_lease(tmp_path, monkeypatch, remaining_ms):
    now_ms = 1_000_000
    calls = []
    with DatabaseCoordinator(tmp_path / "coordination.duckdb", clock_ms=lambda: now_ms) as owner:
        owner.register_task(task_cid="task:ack-budget", task_id="ACK-BUDGET")
        claim = owner.claim_task(task_cid="task:ack-budget", owner_session_id="session:owner",
                                 lease_ms=20_000)

        def candidate(lease):
            calls.append(lease)
            monkeypatch.setattr(dispatch.time, "time_ns",
                lambda: (lease.expires_at_ms - remaining_ms) * 1_000_000)
            dispatch.OwnerFiniteProofQueryDispatchBroker._attempt_ack_budget(lease)
            return "ack-budget-only-no-child"

        if remaining_ms > 5000:
            assert owner.execute_with_task_claim_fence(
                claim, candidate, minimum_remaining_ms=5000) == "ack-budget-only-no-child"
        else:
            with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="full child acknowledgement"):
                owner.execute_with_task_claim_fence(claim, candidate, minimum_remaining_ms=5000)
        assert len(calls) == 1
        assert calls[0].lease_id == claim.lease_id
        assert calls[0].claim_id == claim.claim_id
        assert calls[0].attempt_id == claim.attempt_id
        assert owner.protect_task_claim(claim).lease_id == claim.lease_id


def _selector():
    # This inert object is only selector-input coverage. It cannot bootstrap a
    # daemon, install schema, provide actual native authority or dispatch work.
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._typed_quack_authority_binding = {"structural-test-only": True}
    daemon.authority_mode = "quack"
    daemon.task_shard_count, daemon.task_shard_index = 1, 0
    daemon._require_live_typed_owner = lambda: None
    return daemon


def test_no_finite_markers_keeps_default_selector(monkeypatch):
    monkeypatch.delenv(dispatch.SOCKET_ENV, raising=False)
    monkeypatch.delenv(dispatch.CONTEXT_ENV, raising=False)
    daemon = _selector()
    daemon._typed_quack_authority_binding = None
    assert daemon._finite_proof_query_serialized_coordinator() is False


@pytest.mark.parametrize("absent", [dispatch.SOCKET_ENV, dispatch.CONTEXT_ENV])
def test_partial_finite_routing_is_refused(monkeypatch, absent):
    monkeypatch.setenv(dispatch.SOCKET_ENV, "/not-an-authority.sock")
    monkeypatch.setenv(dispatch.CONTEXT_ENV, "structural-test-context")
    monkeypatch.delenv(absent)
    with pytest.raises(DatabaseImplementationAuthorityError, match="complete typed owner routing"):
        _selector()._finite_proof_query_serialized_coordinator()


@pytest.mark.parametrize("invalid", ["untyped", "multiple-lanes", "nonprivate-socket", "nonprivate-directory"])
def test_finite_selector_requires_complete_private_single_lane_inputs(tmp_path, monkeypatch, invalid):
    receipt = {"schema": "finite-attempt-selector-structural-socket@1",
               "control": invalid, "genuine_bootstrap_attested": False,
               "native_worker_born": False}
    with tempfile.TemporaryDirectory(prefix="fpl-", dir="/tmp") as directory:
        parent = Path(directory)
        endpoint = parent / "dispatch.sock"
        monkeypatch.setenv(dispatch.SOCKET_ENV, str(endpoint))
        monkeypatch.setenv(dispatch.CONTEXT_ENV, "structural-test-context")
        daemon = _selector()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            os.chmod(parent, 0o700)
            metadata = endpoint.lstat()
            receipt.update(socket_path=str(endpoint), socket_inode=metadata.st_ino,
                           socket_device=metadata.st_dev, socket_uid=metadata.st_uid,
                           socket_mode=metadata.st_mode & 0o777,
                           directory_mode=parent.lstat().st_mode & 0o777)
            assert daemon._finite_proof_query_serialized_coordinator() is True
            receipt["complete_structural_inputs_selected"] = True
            if invalid == "untyped":
                daemon._typed_quack_authority_binding = None
            elif invalid == "multiple-lanes":
                daemon.task_shard_count = 2
            elif invalid == "nonprivate-socket":
                os.chmod(endpoint, 0o660)
            else:
                os.chmod(parent, 0o750)
            receipt.update(control_socket_mode=endpoint.lstat().st_mode & 0o777,
                           control_directory_mode=parent.lstat().st_mode & 0o777)
            with pytest.raises(DatabaseImplementationAuthorityError):
                daemon._finite_proof_query_serialized_coordinator()
            receipt["invalid_structural_input_refused"] = True
        receipt["listener_closed"] = listener.fileno() == -1
    receipt.update(socket_removed=not endpoint.exists(), directory_removed=not parent.exists())
    assert receipt["listener_closed"] and receipt["socket_removed"] and receipt["directory_removed"]
    (tmp_path / "selector-socket-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


def test_attempt_authority_file_custody_refuses_same_filesystem_hardlink(tmp_path):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "coordination.duckdb"
    path.write_bytes(b"inert-file-custody-only-no-native-schema")
    alias = tmp_path / "alias.duckdb"
    broker = object.__new__(dispatch.OwnerFiniteProofQueryDispatchBroker)
    before = broker._attempt_authority_file_current(path)
    os.link(path, alias)
    try:
        with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="single-link regular"):
            broker._attempt_authority_file_current(path)
    finally:
        alias.unlink()
    assert broker._attempt_authority_file_current(path) == before


def test_attempt_authority_source_is_pinned_only_by_proof_query_profile():
    from ipfs_accelerate_py.agent_supervisor.merge import database_coordination
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution

    expected = hashlib.sha256(Path(database_coordination.__file__).read_bytes()).hexdigest()
    proof_query = finite_repository_execution._pins(proof_query=True)
    ordinary = finite_repository_execution._pins(proof_query=False)
    assert proof_query[database_coordination.__name__] == expected
    assert database_coordination.__name__ not in ordinary
