"""Retained source provenance survives a forward cooldown and cold replay."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.retained_callback_cooldown import (
    FIELD,
    build_binding,
    payload_from_binding,
    require_claim_binding,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TypedStateOwnerError,
    _validated_stored_retry_cooldown,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    EXECUTION,
    IDENTITY,
    ROUTE,
)


def fixture():
    f = json.loads(
        (Path(__file__).parent / "fixtures/retained_callback_suffix.json").read_text()
    )
    h = {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1",
        "task_cid": f["task_cid"],
        "revisions": f["revisions"],
    }
    source = h["revisions"][7]["body"]["completion_receipt"]
    seed = {
        **{k: source[k] for k in IDENTITY},
        "schema": "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@2",
        "task_cid": f["task_cid"],
        "task_alias": f["task_alias"],
        "source_task_revision": 8,
        "recovery_control_revision": 15,
        "request_id": "request:retained",
        "candidate_commit": "a" * 40,
        "qualified_target_commit": "b" * 40,
        "qualification_kind": "callback_integration",
        "qualification_receipt_id": "receipt:qualified",
        "queue_source_attempt_id": source["attempt_id"],
        "queue_source_claim_id": source["claim_id"],
        "queue_source_lease_id": source["lease_id"],
        "queue_source_fencing_token": source["fencing_token"],
        "queue_source_fence_epoch": source["fence_epoch"],
        "queue_source_binding_id": "sha256:" + "c" * 64,
        "queue_source_projection_immutable_digest": "sha256:" + "d" * 64,
        "recovery_evidence_id": "sha256:" + "e" * 64,
        "terminal_reason": source["reason"],
    }
    seed["seed_id"] = "sha256:" + hashlib.sha256(canonical_json_bytes(seed)).hexdigest()
    receipt = {
        **{k: source[k] for k in IDENTITY | EXECUTION | ROUTE},
        "operation": "database_post_merge_declared_outputs_callback_integration_recovery",
        "control_expected_revision": 15,
        "control_expected_status": "blocked",
        "post_merge_completion_recovery_seed": seed,
        "request_id": seed["request_id"],
        "candidate_commit": seed["candidate_commit"],
        "qualified_target_commit": seed["qualified_target_commit"],
        "source_binding_id": seed["queue_source_binding_id"],
        "source_projection_immutable_digest": seed[
            "queue_source_projection_immutable_digest"
        ],
        "callback_requalification_receipt_id": seed["qualification_receipt_id"],
        "callback_reconciliation_evidence_id": seed["recovery_evidence_id"],
        "source_integration_commit": "f" * 40,
        "source_train_receipt_id": "sha256:" + "a" * 64,
        "backoff_ms": 0,
        "retry_not_before_ms": 1,
        "queue_receipt": {},
        "coordination": {
            k: source[k] for k in ("attempt_id", "claim_id", "attempt_number")
        },
        "queue_reason": "database_post_merge_declared_outputs_callback_integration:"
        + seed["request_id"]
        + ":"
        + seed["qualification_receipt_id"],
    }
    body = {
        k: v for k, v in h["revisions"][-1]["body"].items() if k != "completion_receipt"
    }
    h["revisions"].append(
        {
            "revision": 16,
            "status": "retrying",
            "body": {**body, "completion_receipt": receipt},
        }
    )
    h["projection_cid"] = content_identity(h)
    task = {
        **h["revisions"][-1],
        "task_cid": f["task_cid"],
        "task_alias": f["task_alias"],
    }
    middle = h["revisions"][11]["body"]["completion_receipt"]
    e = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": f["task_cid"],
        **{k: middle[k] for k in IDENTITY},
        "expected_task_revision": 11,
        "delay_ms": middle["backoff_ms"],
        "started_at_ms": middle["retry_not_before_ms"] - middle["backoff_ms"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
        "reason": middle["queue_reason"],
        "selection_penalty": 7,
        "consecutive_failures": 3,
        "expected_queue_revision": 2,
        "expected_queue_attempt": 2,
    }
    return task, h, queue_row(e, 3)


def queue_row(e, revision):
    return {
        "task_cid": e["task_cid"],
        "claim_cid": e["claim_id"],
        "resolution_cid": content_identity(
            {"typed_retry_cooldown": e, "started_at_ms": e["started_at_ms"]}
        ),
        "claimant_did": e["owner_session_id"],
        "logical_epoch": e["fence_epoch"],
        "fencing_token": e["fencing_token"],
        "expires_at_ms": 0,
        "attempt": e["attempt_number"],
        "state": "released",
        "started_at_ms": e["started_at_ms"],
        "release_reason": e["reason"],
        "retry_not_before_ms": e["retry_not_before_ms"],
        "owner_session_id": e["owner_session_id"],
        "fence_epoch": e["fence_epoch"],
        "revision": revision,
        "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "extension_json": canonical_json_bytes(e).decode(),
    }


def forward(task, h, q):
    binding = build_binding(task=task, history=h, prior_queue=q)
    payload = payload_from_binding(binding)
    e = {
        k: v for k, v in payload.items() if k not in {"now_ms", "expected_task_status"}
    }
    e.update(
        schema=TYPED_RETRY_COOLDOWN_SCHEMA,
        started_at_ms=payload["now_ms"],
        retry_not_before_ms=payload["now_ms"] + payload["delay_ms"],
        consecutive_failures=4,
        expected_queue_revision=3,
        expected_queue_attempt=3,
    )
    return payload, queue_row(e, 4)


def test_forward_binding_preserves_source_and_floor():
    task, h, q = fixture()
    before = copy.deepcopy(task)
    payload, after = forward(task, h, q)
    row = _validated_stored_retry_cooldown(after, task_cid=task["task_cid"])
    TypedDatabaseTaskSource._validate_retrying_cooldown_binding(
        SimpleNamespace(**task), row
    )
    assert task == before
    assert task["body"]["completion_receipt"]["attempt_number"] == 2
    assert row["attempt"] == row["fencing_token"] == row["fence_epoch"] == 4
    assert row["extension"]["selection_penalty"] == 7
    assert row["extension"]["consecutive_failures"] == 4
    assert row["extension"][FIELD]["prior_queue"] == q
    assert (
        payload["retained_callback_binding"]["source_seed_id"]
        == task["body"]["completion_receipt"]["post_merge_completion_recovery_seed"][
            "seed_id"
        ]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "semantic",
        "gap",
        "guard",
        "queue_claim",
        "queue_fence",
        "queue_deadline",
        "active",
        "seed",
        "control",
        "nested",
    ],
)
def test_forward_binding_rejects_drift(mutation):
    task, h, q = fixture()
    if mutation == "source":
        h["revisions"][7]["body"]["completion_receipt"]["claim_id"] = "foreign"
    elif mutation == "semantic":
        h["revisions"][10]["body"]["title"] = "foreign"
    elif mutation == "gap":
        h["revisions"].pop(9)
    elif mutation == "guard":
        h["revisions"][14]["body"]["completion_receipt"]["reason"] = "other"
    elif mutation == "queue_claim":
        q["claim_cid"] = "foreign"
    elif mutation == "queue_fence":
        q["fencing_token"] = 2
    elif mutation == "queue_deadline":
        q["retry_not_before_ms"] += 1
    elif mutation == "active":
        task["status"] = "in_progress"
    elif mutation == "seed":
        task["body"]["completion_receipt"]["post_merge_completion_recovery_seed"][
            "seed_id"
        ] = "foreign"
    elif mutation == "control":
        task["revision"] = 17
    elif mutation == "nested":
        q["extension_json"] = json.dumps({**json.loads(q["extension_json"]), FIELD: {}})
    h.pop("projection_cid")
    h["projection_cid"] = content_identity(h)
    with pytest.raises(TypedStateOwnerError):
        build_binding(task=task, history=h, prior_queue=q)


@pytest.mark.parametrize(
    "mutation", [None, "attempt", "fence", "epoch", "seed", "missing", "history"]
)
def test_owner_reservation_binds_forward_floor_and_original_seed(tmp_path, mutation):
    import duckdb

    task, h, q = fixture()
    payload, after = forward(task, h, q)
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE leases AS SELECT * FROM read_json_auto(?)",
        [str(_write(tmp_path / "queue.json", [after]))],
    )
    conn.execute(
        "CREATE TABLE task_revisions(revision BIGINT,status VARCHAR,body_json VARCHAR,task_cid VARCHAR)"
    )
    for r in h["revisions"]:
        conn.execute(
            "INSERT INTO task_revisions VALUES (?,?,?,?)",
            [r["revision"], r["status"], json.dumps(r["body"]), task["task_cid"]],
        )
    seed = task["body"]["completion_receipt"]["post_merge_completion_recovery_seed"]
    claim = {
        **payload[FIELD]["identity"],
        "attempt_id": "new-attempt",
        "claim_id": "new-claim",
        "lease_id": "new-lease",
        "attempt_number": 5,
        "fencing_token": 5,
        "fence_epoch": 5,
        "post_merge_completion_recovery_seed": seed,
        "post_merge_completion_recovery_source_attempt_id": seed["attempt_id"],
    }
    if mutation == "attempt":
        claim["attempt_number"] = 4
    elif mutation == "fence":
        claim["fencing_token"] = 4
    elif mutation == "epoch":
        claim["fence_epoch"] = 4
    elif mutation == "seed":
        claim["post_merge_completion_recovery_seed"] = {}
    elif mutation == "missing":
        conn.execute("DELETE FROM leases")
    elif mutation == "history":
        conn.execute("UPDATE task_revisions SET status='in_progress' WHERE revision=15")
    try:
        if mutation:
            with pytest.raises(TypedStateOwnerError):
                require_claim_binding(conn, task=task, next_receipt=claim)
        else:
            require_claim_binding(conn, task=task, next_receipt=claim)
    finally:
        conn.close()


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize(
    "mutation", [None, "live_history", "live_queue", "payload_floor"]
)
@pytest.mark.parametrize("generation_refresh", [False, True])
def test_exclusive_owner_forward_write_and_replay(tmp_path, monkeypatch, mutation, generation_refresh):
    import duckdb

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        FakeQuackTransport,
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
        TransactionError,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        probe_quack_capabilities,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackClientError,
    )
    from test.api.test_agent_supervisor_quack_owner_mutation import (
        _seed,
        _typed_task_source,
    )

    if generation_refresh:
        from test.api.test_retained_callback_generation import refresh_queue_fixture
        task, h, q = refresh_queue_fixture()
        payload = payload_from_binding(build_binding(task=task, history=h, prior_queue=q))
    else:
        task, h, q = fixture()
        payload, _after = forward(task, h, q)
    floor = 5 if generation_refresh else 4
    database = tmp_path / "control.duckdb"
    _seed(database)
    connection = duckdb.connect(str(database))
    connection.execute(
        "UPDATE tasks SET task_cid=?,task_alias=?,status=?,revision=?,body_json=?",
        [
            task["task_cid"],
            task["task_alias"],
            "retrying",
            task["revision"],
            json.dumps(task["body"]),
        ],
    )
    for row in h["revisions"]:
        connection.execute(
            "INSERT INTO task_revisions VALUES (?,?,?,?,?)",
            [
                task["task_cid"],
                row["revision"],
                row["status"],
                json.dumps(row["body"]),
                "test",
            ],
        )
    keys = list(q)
    connection.execute(
        "INSERT INTO leases ("
        + ",".join(keys)
        + ") VALUES ("
        + ",".join("?" for _ in keys)
        + ")",
        [q[k] for k in keys],
    )
    if mutation == "live_history":
        connection.execute(
            "UPDATE task_revisions SET status='in_progress' WHERE task_cid=? AND revision=15",
            [task["task_cid"]],
        )
    if mutation == "live_queue":
        connection.execute("UPDATE leases SET release_reason=?", ["foreign"])
    if mutation == "payload_floor":
        payload["attempt_number"] = 3
    connection.close()

    def launch():
        server = build_server(
            database_path=database,
            state_dir=tmp_path / "state",
            store_id="retained-cooldown-test",
            repository_id="repository:test",
            transport=FakeQuackTransport(),
            capability_probe=lambda **kw: probe_quack_capabilities(),
        )
        identity = server.start()
        source = _typed_task_source(
            server,
            identity,
            monkeypatch,
            client_id="database-implementation-daemon:retained-test",
            allowed_command_operations=(
                "task.retry.cooldown.record",
                "task.status.cas.receipt",
            ),
        )
        return server, source

    server, source = launch()
    try:
        if mutation:
            with pytest.raises(
                (QuackClientError, TypedStateOwnerError, TransactionError)
            ):
                source.record_task_retry_cooldown(**payload)
        else:
            repairs = source.repair_retrying_cooldown_bindings()
            assert len(repairs) == 1 and repairs[0]["changed"] is True
            assert source.record_task_retry_cooldown(**payload).changed is False
            task_record = source.get_task(task["task_cid"])
            assert task_record.revision == task["revision"] and task_record.body == task["body"]
            entry = source.validate_retrying_task_cooldown(
                task["task_cid"],
                expected_attempt_identity={
                    k: task["body"]["completion_receipt"][k] for k in IDENTITY
                },
            )
            assert entry.attempt == floor
            from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
                DatabaseImplementationDaemon,
            )

            daemon = object.__new__(DatabaseImplementationDaemon)
            daemon.open = lambda: daemon
            daemon._task_source = source
            assert daemon._typed_authoritative_attempt_floor(task_record) == floor
    finally:
        source.close()
        server.stop()
    if mutation:
        return
    server, source = launch()
    try:
        assert source.record_task_retry_cooldown(**payload).changed is False
        entry = source.validate_retrying_task_cooldown(task["task_cid"])
        assert entry.attempt == floor
        assert source.repair_retrying_cooldown_bindings() == ()
    finally:
        source.close()
        server.stop()
