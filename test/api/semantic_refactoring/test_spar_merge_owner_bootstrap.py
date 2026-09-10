"""Disposable real DuckDB/typed socket role; fake only Quack network here."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ops.agent_supervisor import spar_merge_owner as native
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue import (
    OwnerMergeQueueClient,
    SERVICE_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
)
from test.api.test_agent_supervisor_quack_state_server import _compatible_report
from test.api.test_spar_replica_copy_writer_lock import (
    writer_lock,
    assert_other_writer_blocked,
)


@pytest.fixture
def preserved(tmp_path):
    source = tmp_path / "offline"
    queue = MergeQueue(
        source,
        target_repository_id="repo:spar",
        target_branch="main",
        require_target_binding=True,
    )
    pending = queue.enqueue(
        branch_name="work/pending", task_id="SPAR-001", commit_sha="a" * 40
    )
    unknown = queue.enqueue(
        branch_name="work/unknown", task_id="SPAR-031", commit_sha="b" * 40
    )
    unknown = queue.claim_pending_request(unknown, consumer_id="retained-old-consumer")
    manifest = {
        "schema": native.SCHEMA,
        "database": "merge_queue.duckdb",
        "wal": None,
        "repository_id": "repo:spar",
        "target_branch": "main",
        "store_id": "spar-legacy-queue",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "scope_bindings": [
            {
                "board_namespace": "SPAR",
                "config_cid": "sha256:" + "c" * 64,
                "plan_cid": "sha256:" + "d" * 64,
                "lane_id": "0",
                "attempt_root": str(tmp_path / "attempts"),
            }
        ],
        "queue_policy": dict(native.DEFAULT_QUEUE_POLICY),
        "receipt_imports": [],
        "cursor_imports": [],
        "files": [],
    }
    for path in sorted(source.rglob("*")):
        if path.is_file():
            body = path.read_bytes()
            manifest["files"].append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size_bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
    before = {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    return source, manifest, pending, unknown, before


def start(prepared, directory):
    return native.start_queue_owner(
        prepared,
        state_dir=directory,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_: _compatible_report(),
    )


def attach(server, consumer="native-reader"):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )

    birth = process_birth_id(current_process_birth())
    token = server.issue_typed_client_grant(
        client_id=consumer,
        process_birth_id=birth,
        allowed_operations=tuple(SERVICE_OPERATIONS),
        peer_pid=os.getpid(),
        entity_scopes={
            "repository_id": "repo:spar",
            "target_branch": "main",
            "consumer_id": consumer,
        },
    )
    conn = TypedStateOwnerConnection(
        socket_path=server.typed_command_socket_path(),
        token=token,
        client_id=consumer,
        process_birth_id=birth,
        store_id=server.identity.store_id,
    )
    return conn, OwnerMergeQueueClient(
        conn, repository_id="repo:spar", target_branch="main", consumer_id=consumer
    )


def test_two_retained_roles_in_one_pid_preserve_legacy_queue_and_restart(
    preserved, tmp_path
):
    source, manifest, pending, unknown, before = preserved
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "clone", manifest=manifest
    )
    other = native.prepare_offline_clone(
        offline_root=source,
        destination=tmp_path / "other-clone",
        manifest={**manifest, "store_id": "spar-other-queue"},
    )
    first = second = replacement = None
    clients = []
    try:
        first = start(prepared, tmp_path / "owner-one")
        second = start(other, tmp_path / "owner-two")
        assert first.identity.process_birth == second.identity.process_birth
        for field in ("database_uuid", "server_id", "listen_uri", "store_id"):
            assert getattr(first.identity, field) != getattr(second.identity, field)
        assert first.typed_command_socket_path() != second.typed_command_socket_path()
        assert first.owner_lock_path() != second.owner_lock_path()
        for owner in (first, second):
            assert owner.ready()["ready"] is True
            conn, api = attach(owner)
            clients.append(conn)
            row = json.loads(
                api.call("get", request_id=unknown.request_id)["request_json"]
            )
            assert row["claim_token"] == unknown.claim_token
            assert row["claim_generation"] == unknown.claim_generation
            assert row["consumer_id"] == "retained-old-consumer"
            assert row["status"] == "processing"
            assert (
                json.loads(
                    api.call("get", request_id=pending.request_id)["request_json"]
                )["status"]
                == "pending"
            )
            with owner._owner_transaction_lock:
                for table in ("tasks", "objectives", "goals"):
                    assert (
                        owner._connection.execute(
                            'SELECT COUNT(*) FROM "' + table + '"'
                        ).fetchone()[0]
                        == 0
                    )
            owner.checkpoint()
            assert writer_lock(owner.config.database_path)
            assert_other_writer_blocked(owner.config.database_path)
        old_identity = first.identity
        clients[0].close()
        first.stop()
        replacement = start(prepared, tmp_path / "owner-one")
        assert replacement.identity.database_uuid == old_identity.database_uuid
        assert replacement.identity.generation == old_identity.generation + 1
        with replacement._owner_transaction_lock:
            native.require_preserved(
                prepared.preserved_inventory, native.inventory(replacement._connection)
            )
    finally:
        for client in clients:
            client.close()
        for owner in (replacement, second, first):
            if owner is not None:
                owner.stop()
    assert before == {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }


def refresh_manifest(source, manifest):
    manifest["files"] = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            body = path.read_bytes()
            manifest["files"].append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size_bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )


def add_preserved_imports(source, manifest):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        _POST_MERGE_RECOVERY_CURSOR_SCHEMA,
        _canonical_json,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    scope = manifest["scope_bindings"][0]
    cursor = {
        "schema": _POST_MERGE_RECOVERY_CURSOR_SCHEMA,
        "target_repository_id": manifest["repository_id"],
        "target_branch": manifest["target_branch"],
        "attempt_root": scope["attempt_root"],
        "cursors": {stage: "preserved:" + stage for stage in native.STAGES},
    }
    cursor["state_id"] = content_identity(cursor)
    key = hashlib.sha256(
        _canonical_json(
            {
                "target_repository_id": manifest["repository_id"],
                "target_branch": manifest["target_branch"],
                "attempt_root": scope["attempt_root"],
            }
        )
    ).hexdigest()
    cursor_path = source / "train" / "post-merge-recovery-cursors" / (key + ".json")
    cursor_path.parent.mkdir(parents=True)
    cursor_path.write_text(json.dumps(cursor))
    manifest["cursor_imports"] = [
        {
            "path": cursor_path.relative_to(source).as_posix(),
            "scope_cid": native.recovery_scope_cid(
                store_id=manifest["store_id"],
                repository_id=manifest["repository_id"],
                target_branch=manifest["target_branch"],
                scope_binding=scope,
            ),
        }
    ]
    receipts = [
        {"stage": "prepared", "duration": 0.125},
        {"stage": "observed", "unknown_callback": "SPAR-031"},
    ]
    for revision, receipt in enumerate(receipts, 1):
        path = f"train/receipt-{revision}.json"
        (source / path).write_text(json.dumps(receipt))
        manifest["receipt_imports"].append(
            {
                "path": path,
                "receipt_key": "train:preserved",
                "revision": revision,
                "receipt_cid": native._cid(receipt),
            }
        )
    refresh_manifest(source, manifest)
    return cursor, receipts


def attach_recovery(server, manifest, consumer="native-recovery-reader"):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge import (
        owner_recovery_runtime as recovery,
    )

    birth = process_birth_id(current_process_birth())
    scope = native.recovery_scope_cid(
        store_id=manifest["store_id"],
        repository_id=manifest["repository_id"],
        target_branch=manifest["target_branch"],
        scope_binding=manifest["scope_bindings"][0],
    )
    token = server.issue_typed_client_grant(
        client_id=consumer,
        process_birth_id=birth,
        allowed_operations=tuple(recovery.SERVICE_OPERATIONS),
        peer_pid=os.getpid(),
        entity_scopes={
            "repository_id": manifest["repository_id"],
            "target_branch": manifest["target_branch"],
            "consumer_id": consumer,
            "recovery_scope_cid": scope,
        },
    )
    connection = TypedStateOwnerConnection(
        socket_path=server.typed_command_socket_path(),
        token=token,
        client_id=consumer,
        process_birth_id=birth,
        store_id=manifest["store_id"],
    )
    return connection, recovery.OwnerRecoveryRuntimeClient(
        connection,
        repository_id=manifest["repository_id"],
        target_branch=manifest["target_branch"],
        consumer_id=consumer,
        recovery_scope_cid=scope,
    )


def test_explicit_five_stage_cursor_and_receipt_history_survive_owner_restart(
    preserved, tmp_path
):
    source, manifest, *_ = preserved
    cursor, receipts = add_preserved_imports(source, manifest)
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "clone", manifest=manifest
    )
    for generation in (1, 2):
        server = start(prepared, tmp_path / "owner")
        client = None
        try:
            assert server.identity.generation == generation
            client, api = attach_recovery(server, manifest)
            assert api.load_cursors()["cursors"] == cursor["cursors"]
            for revision, receipt in enumerate(receipts, 1):
                assert (
                    api.get_receipt("train:preserved", revision=revision)["receipt"]
                    == receipt
                )
            assert api.get_receipt("train:preserved")["revision"] == 2
        finally:
            if client is not None:
                client.close()
            server.stop()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_stage",
        "foreign_scope",
        "duplicate_cursor",
        "cursor_hash",
        "receipt_gap",
        "receipt_hash",
        "wrong_cursor_path",
        "unimported_cursor",
    ],
)
def test_preserved_import_invalidity_is_not_empty_state(preserved, tmp_path, failure):
    source, manifest, *_ = preserved
    cursor, _ = add_preserved_imports(source, manifest)
    if failure == "missing_stage":
        del cursor["cursors"][native.STAGES[0]]
        (source / manifest["cursor_imports"][0]["path"]).write_text(json.dumps(cursor))
        refresh_manifest(source, manifest)
    elif failure == "foreign_scope":
        manifest["cursor_imports"][0]["scope_cid"] = "sha256:" + "0" * 64
    elif failure == "duplicate_cursor":
        manifest["cursor_imports"].append(dict(manifest["cursor_imports"][0]))
    elif failure == "cursor_hash":
        cursor["state_id"] = "invalid"
        (source / manifest["cursor_imports"][0]["path"]).write_text(json.dumps(cursor))
        refresh_manifest(source, manifest)
    elif failure == "receipt_gap":
        manifest["receipt_imports"][1]["revision"] = 3
    elif failure == "receipt_hash":
        manifest["receipt_imports"][0]["receipt_cid"] = "invalid"
    elif failure == "wrong_cursor_path":
        spec = manifest["cursor_imports"][0]
        original = source / spec["path"]
        renamed = original.with_name("0" * 64 + ".json")
        original.rename(renamed)
        spec["path"] = renamed.relative_to(source).as_posix()
        refresh_manifest(source, manifest)
    else:
        manifest["cursor_imports"] = []
    with pytest.raises(native.SparMergeOwnerError):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )


def test_undeclared_wal_is_rejected(preserved, tmp_path):
    source, manifest, *_ = preserved
    (source / "merge_queue.duckdb.wal").write_bytes(b"must never be silently ignored")
    with pytest.raises(native.SparMergeOwnerError, match="inventory"):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )


def test_explicit_database_wal_clone_replays_only_disposable_input(preserved, tmp_path):
    import duckdb

    source, manifest, *_ = preserved
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import os, sys
sys.path.insert(0, sys.argv[2])
import duckdb
c = duckdb.connect(sys.argv[1], config={'threads': 1})
c.execute("CREATE TABLE offline_wal_evidence(value VARCHAR)")
c.execute("INSERT INTO offline_wal_evidence VALUES ('preserve committed WAL')")
os._exit(0)
""",
            str(source / "merge_queue.duckdb"),
            str(Path(duckdb.__file__).resolve().parent.parent),
        ],
        env={"PATH": os.defpath},
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert (source / "merge_queue.duckdb.wal").is_file()
    manifest["wal"] = "merge_queue.duckdb.wal"
    refresh_manifest(source, manifest)
    before = {
        entry["path"]: (source / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "clone", manifest=manifest
    )
    server = start(prepared, tmp_path / "owner")
    try:
        with server._owner_transaction_lock:
            assert (
                server._connection.execute(
                    "SELECT value FROM offline_wal_evidence"
                ).fetchone()[0]
                == "preserve committed WAL"
            )
        assert writer_lock(prepared.database_path)
    finally:
        server.stop()
    assert before == {name: (source / name).read_bytes() for name in before}


@pytest.mark.parametrize(
    "failure", ["hash", "size", "wal", "duplicate", "scope", "fifo", "symlink"]
)
def test_offline_manifest_refusals_before_owner_birth(preserved, tmp_path, failure):
    source, manifest, *_ = preserved
    value = json.loads(json.dumps(manifest))
    if failure == "hash":
        value["files"][0]["sha256"] = "0" * 64
    elif failure == "size":
        value["files"][0]["size_bytes"] += 1
    elif failure == "wal":
        value["wal"] = "merge_queue.duckdb.wal"
    elif failure == "duplicate":
        value["files"].append(value["files"][0])
    elif failure == "scope":
        value["scope_bindings"][0]["attempt_root"] = "relative"
    else:
        path = source / value["files"][0]["path"]
        path.unlink()
        if failure == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(tmp_path / "absent")
    with pytest.raises(native.SparMergeOwnerError):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=value
        )


def test_existing_partial_owner_schema_is_not_minted_a_new_identity(
    preserved, tmp_path
):
    source, manifest, *_ = preserved
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    with open_duckdb_connection(source / "merge_queue.duckdb") as conn:
        conn.execute("CREATE TABLE control_plane_metadata (key VARCHAR,value VARCHAR)")
    for entry in manifest["files"]:
        body = (source / entry["path"]).read_bytes()
        entry.update(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    with pytest.raises(native.SparMergeOwnerError, match="partial"):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )


def test_runtime_refuses_wrong_preserved_owner_uuid(preserved, tmp_path):
    source, manifest, *_ = preserved
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "clone", manifest=manifest
    )
    with pytest.raises(native.SparMergeOwnerError, match="UUID"):
        start(replace(prepared, database_uuid="foreign"), tmp_path / "owner")


def test_two_cycle_offline_qualification_is_not_live_admission(preserved, tmp_path):
    source, manifest, *_ = preserved
    add_preserved_imports(source, manifest)
    report = native.qualify_offline_bundle(
        offline_root=source,
        destination=tmp_path / "qualified",
        manifest=manifest,
        transport_factory=FakeQuackTransport,
        capability_probe=lambda **_: _compatible_report(),
    )
    assert report["qualified"] is True, report
    assert report["live_custody_qualified"] is False
    assert report["completion_authority"] is False
    assert report["source_admission"] is False
    assert report["transport"] == "injected_test_transport"
    assert [c["generation"] for c in report["cycles"]] == [1, 2]
    assert all(
        c["closed"]
        and c["preserved"]
        and c["startup_and_checkpoint_writer_lock"]
        and c["scopes_read"] == 1
        for c in report["cycles"]
    )

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not {"token", "queue_token", "recovery_token"}.intersection(keys(report))


def test_failure_after_successful_cycles_never_leaves_qualified_true(
    preserved, tmp_path, monkeypatch
):
    source, manifest, *_ = preserved
    original = native.file_inventory
    calls = 0

    def fail_final(root):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise native.SparMergeOwnerError("final input changed")
        return original(root)

    monkeypatch.setattr(native, "file_inventory", fail_final)
    report = native.qualify_offline_bundle(
        offline_root=source,
        destination=tmp_path / "qualified",
        manifest=manifest,
        transport_factory=FakeQuackTransport,
        capability_probe=lambda **_: _compatible_report(),
    )
    assert len(report["cycles"]) == 2 and all(c["closed"] for c in report["cycles"])
    assert report["qualified"] is False
    assert report["error_type"] == "SparMergeOwnerError"


def test_source_changes_during_copy_refuse_before_database_open(
    preserved, tmp_path, monkeypatch
):
    source, manifest, *_ = preserved
    original = native.copy_entry
    fired = False

    def interfere(root, entry, destination, **kwargs):
        nonlocal fired
        result = original(root, entry, destination, **kwargs)
        if destination is not None and not fired:
            fired = True
            (source / "unexpected-after-copy").write_bytes(b"concurrent work")
        return result

    monkeypatch.setattr(native, "copy_entry", interfere)

    def forbidden(*args, **kwargs):
        pytest.fail("opened database before stable input admission")

    monkeypatch.setattr(native, "open_duckdb_connection", forbidden)
    with pytest.raises(native.SparMergeOwnerError, match="changed"):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )
    assert (source / "unexpected-after-copy").read_bytes() == b"concurrent work"


def test_known_owner_table_without_metadata_is_partial_not_fresh(preserved, tmp_path):
    source, manifest, *_ = preserved
    with native.open_duckdb_connection(source / "merge_queue.duckdb") as connection:
        connection.execute("CREATE TABLE schema_contracts (evidence VARCHAR)")
    refresh_manifest(source, manifest)
    with pytest.raises(native.SparMergeOwnerError, match="partial"):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )


def test_observed_locked_input_is_refused_without_dropping_writer(preserved, tmp_path):
    import duckdb

    source, manifest, *_ = preserved
    database = source / "merge_queue.duckdb"
    connection = duckdb.connect(str(database), config={"threads": 1})
    try:
        assert writer_lock(database)
        with pytest.raises(native.SparMergeOwnerError, match="kernel lock"):
            native.prepare_offline_clone(
                offline_root=source, destination=tmp_path / "refused", manifest=manifest
            )
        assert writer_lock(database)
        assert_other_writer_blocked(database)
    finally:
        connection.close()


def test_real_preinstalled_quack_qualifies_disposable_role(preserved, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        probe_quack_capabilities,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        QuackCapabilityStatus,
    )

    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip("real preinstalled Quack unavailable")
    source, manifest, *_ = preserved
    report = native.qualify_offline_bundle(
        offline_root=source, destination=tmp_path / "real-qualified", manifest=manifest
    )
    assert report["qualified"] is True, report
    assert report["transport"] == "real_quack"
    assert len(report["cycles"]) == 2
    assert all(
        c["closed"] and c["startup_and_checkpoint_writer_lock"]
        for c in report["cycles"]
    )


def test_foreign_rows_are_exact_not_append_only(tmp_path):
    with native.open_duckdb_connection(tmp_path / "offline.duckdb") as connection:
        connection.execute("CREATE TABLE foreign_evidence (value VARCHAR)")
        connection.execute("INSERT INTO foreign_evidence VALUES ('preserved')")
        before = native.inventory(connection)
        connection.execute(
            "INSERT INTO foreign_evidence VALUES ('not an admitted owner append')"
        )
        with pytest.raises(native.SparMergeOwnerError, match="foreign"):
            native.require_preserved(before, native.inventory(connection))


@pytest.mark.parametrize("bound", ["rows", "digest_bytes"])
def test_inventory_uses_aggregate_not_per_table_budget(tmp_path, monkeypatch, bound):
    with native.open_duckdb_connection(tmp_path / "offline.duckdb") as connection:
        for table in ("first", "second"):
            connection.execute(
                "CREATE TABLE "
                + table
                + " AS SELECT value FROM range(3) AS data(value)"
            )
        if bound == "rows":
            monkeypatch.setattr(native, "MAX_ROWS", 4)
        else:
            monkeypatch.setattr(native, "MAX_INVENTORY_DIGEST_BYTES", 4 * 64)
        with pytest.raises(native.SparMergeOwnerError, match="inventory exceeds"):
            native.inventory(connection)


def test_inventory_deterministic_pages_cover_duplicate_rows(tmp_path):
    with native.open_duckdb_connection(tmp_path / "offline.duckdb") as connection:
        connection.execute(
            "CREATE TABLE unordered_rows AS SELECT i%7 AS key, i%3 AS value FROM range(3001) AS data(i)"
        )
        first = native.inventory(connection)
        connection.execute(
            "CREATE TABLE replacement AS SELECT * FROM unordered_rows ORDER BY key DESC,value DESC"
        )
        connection.execute("DROP TABLE unordered_rows")
        connection.execute("ALTER TABLE replacement RENAME TO unordered_rows")
        assert native.inventory(connection) == first
        assert len(first["unordered_rows"]["rows"]) == 3001


def test_nonmain_schema_is_explicitly_refused(preserved, tmp_path):
    source, manifest, *_ = preserved
    with native.open_duckdb_connection(source / "merge_queue.duckdb") as connection:
        connection.execute("CREATE SCHEMA foreign_namespace")
        connection.execute(
            "CREATE TABLE foreign_namespace.preserved AS SELECT 1 AS evidence"
        )
    refresh_manifest(source, manifest)
    with pytest.raises(native.SparMergeOwnerError, match="non-main"):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )


@pytest.mark.parametrize(
    "table,reason", [("foreign_new", "undeclared"), ("tasks", "non-owner")]
)
def test_new_foreign_table_or_domain_rows_cannot_be_called_preserved(
    tmp_path, table, reason
):
    with native.open_duckdb_connection(tmp_path / "offline.duckdb") as connection:
        before = native.inventory(connection)
        connection.execute(
            "CREATE TABLE " + table + " AS SELECT 'not owner metadata' AS value"
        )
        with pytest.raises(native.SparMergeOwnerError, match=reason):
            native.require_preserved(before, native.inventory(connection))


@pytest.mark.parametrize("length", [41, 48, 63])
def test_source_object_id_requires_exact_git_format(preserved, length):
    _, manifest, *_ = preserved
    manifest["source_commit"] = "a" * length
    with pytest.raises(native.SparMergeOwnerError, match="source generation"):
        native.validate_manifest(manifest)
