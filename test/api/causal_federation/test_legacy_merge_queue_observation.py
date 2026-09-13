"""Real disposable owner/queue tests for observation without recovery authority."""
from __future__ import annotations

import copy
import fcntl
import json
import os
import shutil
from pathlib import Path

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.task_sources import legacy_merge_queue_observation as observation
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    STATUS_BOOTSTRAP_CLIENT_ID, TypedStateOwnerConnection, TypedStateOwnerError,
)

CID = "task:typed-owner"
ALIAS = "CASF-TYPED"
REPOSITORY = "repository:sha256:" + "a" * 64


def enqueue(queue, ordinal, *, alias=ALIAS, cid=CID, bindings=None):
    return queue.enqueue(branch_name=f"candidate/{ordinal}", task_id=alias,
                         canonical_task_id=cid, canonical_task_key=cid,
                         commit_sha=f"{ordinal + 1:040x}",
                         metadata={"completion_task_cids": bindings or {alias: cid}})


@pytest.fixture
def native(tmp_path):
    database = tmp_path / "control.duckdb"
    _install(database)
    gateway, connection = _gateway(database, tmp_path / "owner.sock")
    token = gateway.configure_status_bootstrap()
    connection.execute("UPDATE tasks SET plan_cid=?, identity_json=?, body_json=?",
                       ["plan:sealed", json.dumps({"repository_tree_id": "tree:sealed"}),
                        json.dumps({"board_namespace": "board:sealed"})])
    gateway.bind_database_status_scope(board_namespace="board:sealed", plan_root_cid="plan:sealed",
                                       repository_tree_id="tree:sealed", task_cids=[CID])
    queue = MergeQueue(tmp_path / "queue", target_repository_id=REPOSITORY,
                       target_branch="main", require_target_binding=True)
    first = enqueue(queue, 1)
    claimed = queue.dequeue(consumer_id="consumer:first")
    assert claimed is not None and claimed.request_id == first.request_id
    queue.fail(claimed, reason="retained first quarantine")
    second = enqueue(queue, 2)
    clients = []

    def client(*, bind=True):
        if bind:
            gateway.bind_legacy_merge_queue_status_scope(queue_dir=queue.queue_dir,
                target_repository_id=REPOSITORY, target_branch="main")
        value = TypedStateOwnerConnection(socket_path=gateway.socket_path, token=token,
            client_id=STATUS_BOOTSTRAP_CLIENT_ID, process_birth_id="birth:queue-observer",
            store_id="control.duckdb", status_bootstrap=True)
        clients.append(value)
        return value

    yield gateway, connection, queue, first, second, client
    for value in clients:
        value.close()
    gateway.stop()
    connection.close()
    # This fixture owns these reproducible databases and starts threads only.
    # Retain a small closed-scope ledger; no source or live board state belongs
    # below this explicit pytest directory.
    assert not any(thread.is_alive() for thread in gateway._clients)
    assert gateway._thread is None or not gateway._thread.is_alive()
    assert not list(tmp_path.rglob(".git"))
    details = tmp_path.stat()
    with (tmp_path.parent / "closed-fixtures.jsonl").open("a") as ledger:
        ledger.write(json.dumps({"path": str(tmp_path), "device": details.st_dev,
                                "inode": details.st_ino, "owner_threads_closed": True}) + "\n")
    shutil.rmtree(tmp_path)


def observe(client):
    return client.legacy_merge_queue_task_observation([CID], task_cid=CID)


def filesystem(root):
    return {str(path.relative_to(root)): (path.stat().st_ino, path.stat().st_size,
             path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*") if path.is_file()}


def test_native_observation_preserves_both_candidates_claims_and_complete_population(native):
    gateway, connection, queue, first, second, make_client = native
    other = enqueue(queue, 3, alias="OTHER", cid="task:other")
    client = make_client()
    before_queue = filesystem(queue.queue_dir)
    before_control = connection.execute("SELECT * FROM tasks").fetchall()
    result = observe(client)
    assert filesystem(queue.queue_dir) == before_queue
    assert connection.execute("SELECT * FROM tasks").fetchall() == before_control
    assert result["queue_settlement"]["row_count"] == len(result["population"]) == 3
    assert [row["request_id"] for row in result["matching_rows"]] == [first.request_id, second.request_id]
    assert other.request_id in [row["request_id"] for row in result["population"]]
    assert [row["status"] for row in result["matching_rows"]] == ["quarantined", "pending"]
    assert result["matching_rows"][0]["claim_generation"] > 0
    assert result["matching_rows"][0]["claim_token"] == ""
    assert result["matching_rows"][0]["failure_reason"] == "retained first quarantine"
    assert result["matching_rows"][1]["consumer_id"] == ""
    assert result["matching_rows"][1]["commit_sha"] == second.commit_sha
    assert result["queue_settlement"]["settled"] is False
    assert all(result[name] is False for name in ("guard_retained", "claim_authority", "retry_authorized", "completion_authority"))
    assert client.grant["allowed_command_operations"] == []
    assert gateway._committed_transactions == 0
    with pytest.raises(TypedStateOwnerError):
        client.execute_operation("txn_cas_task_status", ["done", "now", CID, 0])


def test_observation_is_opt_in_and_socket_cannot_nominate_path(native):
    _, _, _, _, _, make_client = native
    client = make_client(bind=False)
    assert observation.OPERATION not in client.grant["allowed_operations"]
    with pytest.raises(TypedStateOwnerError):
        observe(client)
    request = observation.observation_request(client.identity, [CID], task_cid=CID)
    request["queue_dir"] = "/some/other/queue"
    with pytest.raises(TypedStateOwnerError):
        client._request(observation.OPERATION, observation_request=request)


def test_bound_queue_cannot_be_rebound_or_database_substituted(native):
    gateway, _, queue, _, _, make_client = native
    client = make_client()
    with pytest.raises(TypedStateOwnerError):
        gateway.bind_legacy_merge_queue_status_scope(queue_dir=queue.queue_dir,
            target_repository_id=REPOSITORY, target_branch="main")
    original = queue.database_path
    replacement = original.with_suffix(".replacement")
    replacement.write_bytes(original.read_bytes())
    os.replace(replacement, original)
    with pytest.raises(TypedStateOwnerError):
        observe(client)


def test_queue_lock_is_held_during_owner_snapshot_and_released_on_return(native, monkeypatch):
    gateway, _, queue, _, _, make_client = native
    client = make_client()
    original = gateway._completion_progress_snapshot
    checked = []

    def inspect(*args, **kwargs):
        with (queue.queue_dir / ".merge_queue.duckdb.lock").open("rb") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(gateway, "_completion_progress_snapshot", inspect)
    observe(client)
    assert checked == [True]
    with (queue.queue_dir / ".merge_queue.duckdb.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("change", ["source", "owner", "wal", "missing_binding", "unknown_state"])
def test_uncertain_source_or_queue_never_becomes_empty_observation(native, change):
    gateway, connection, queue, _, second, make_client = native
    client = make_client()
    if change == "source":
        connection.execute("UPDATE tasks SET plan_cid='plan:changed'")
    elif change == "owner":
        client.identity = dict(client.identity) | {"generation": client.identity["generation"] + 1}
    elif change == "wal":
        Path(str(queue.database_path) + ".wal").write_bytes(b"not settled")
    else:
        with queue._connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            if change == "unknown_state":
                conn.execute("UPDATE merge_requests SET status='unknown' WHERE request_id=?", [second.request_id])
            else:
                conn.execute("UPDATE merge_requests SET metadata_json='{}' WHERE request_id=?", [second.request_id])
            conn.commit()
    with pytest.raises(TypedStateOwnerError):
        observe(client)


def test_unrelated_malformed_binding_does_not_silently_drop_population(native):
    _, _, queue, _, _, make_client = native
    third = enqueue(queue, 3, alias="OTHER", cid="task:other")
    claimed = queue.claim_pending_request(third.request_id, consumer_id="third")
    assert claimed is not None
    queue.fail(claimed, reason="unrelated quarantine")
    with queue._connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("UPDATE merge_requests SET metadata_json='{}' WHERE request_id=?", [third.request_id])
        conn.commit()
    client = make_client()
    with pytest.raises(TypedStateOwnerError):
        observe(client)


def test_secondary_task_binding_is_included_in_complete_matching_population(native):
    _, _, queue, _, _, make_client = native
    third = enqueue(queue, 3, alias="OTHER", cid="task:other", bindings={"OTHER": "task:other", ALIAS: CID})
    result = observe(make_client())
    assert third.request_id in [row["request_id"] for row in result["matching_rows"]]
    assert len(result["matching_rows"]) == 3


@pytest.mark.parametrize("limit", ["MAX_POPULATION", "MAX_INPUT_BYTES", "MAX_MATCHES", "MAX_RESULT_BYTES"])
def test_bounded_observation_fails_instead_of_truncating(native, monkeypatch, limit):
    _, _, _, _, _, make_client = native
    client = make_client()
    monkeypatch.setattr(observation, limit, 1)
    with pytest.raises(TypedStateOwnerError):
        observe(client)


def test_client_rejects_omitted_row_even_with_resealed_envelope(native):
    _, _, _, _, _, make_client = native
    client = make_client()
    request = observation.observation_request(client.identity, [CID], task_cid=CID)
    result = copy.deepcopy(observe(client))
    result["matching_rows"].pop()
    result["observation_cid"] = observation.content_identity({k:v for k,v in result.items() if k != "observation_cid"})
    with pytest.raises(TypedStateOwnerError):
        observation.validate_observation(result, request=request)


@pytest.mark.parametrize("dedupe_key", [None, "x" * 4097])
def test_nullable_dedupe_key_remains_null_or_rejects_oversize(native, dedupe_key):
    _, _, queue, _, second, make_client = native
    with queue._connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("UPDATE merge_requests SET dedupe_key=? WHERE request_id=?", [dedupe_key, second.request_id])
        conn.commit()
    client = make_client()
    if dedupe_key is None:
        rows = observe(client)["matching_rows"]
        row = next(r for r in rows if r["request_id"] == second.request_id)
        assert row["dedupe_key"] is None
        decoded = observation.decode_observed_queue_row(row)
        assert type(decoded["enqueued_at"]) is float
        assert decoded["enqueued_at"] == second.enqueued_at
        assert type(decoded["claim_generation"]) is int
    else:
        with pytest.raises(TypedStateOwnerError):
            observe(client)


def test_matching_integer_authority_fields_reject_bool_and_float(native):
    _, _, _, _, _, make_client = native
    row = observe(make_client())["matching_rows"][0]
    for invalid in (True, 1.0):
        changed = dict(row, claim_generation=invalid)
        with pytest.raises(TypedStateOwnerError):
            observation.decode_observed_queue_row(changed)


def test_final_guard_recheck_rejects_wal_created_during_owner_snapshot(native, monkeypatch):
    gateway, _, queue, _, _, make_client = native
    client = make_client()
    original = gateway._completion_progress_snapshot

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        Path(str(queue.database_path) + ".wal").write_bytes(b"uncertain writer")
        return result

    monkeypatch.setattr(gateway, "_completion_progress_snapshot", changed)
    with pytest.raises(TypedStateOwnerError):
        observe(client)
    with (queue.queue_dir / ".merge_queue.duckdb.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
