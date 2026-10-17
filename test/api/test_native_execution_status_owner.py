"""Disposable native-backend fixtures; these tests grant no native authority.

All databases and writer locks are newly created below pytest's charged
temporary directory. No campaign/control store, provider or native admission
is used. Production DDL is read as source for fixture creation, never executed
by the observation owner. No daemon object or forged old owner handle exists.
"""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import time
from functools import lru_cache
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_execution_status_owner import (
    MAX_JSON_BYTES,
    NativeExecutionStatusBinding,
    NativeExecutionStatusError,
    NativeExecutionStatusOwner,
)


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("fixture tests must not spawn providers, Docker or subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def identity(path):
    info = path.stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_fixture_ddl():
    # Validate the genuine source digest on every fixture. The outer source
    # owner additionally pins that producer for the complete test lifetime.
    spec = importlib.util.find_spec("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")
    path = Path(spec.origin)
    source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _source_validated_native_fixture_ddl(str(path), source_digest)


@lru_cache(maxsize=1)
def _source_validated_native_fixture_ddl(path_text, source_digest):
    # Cache only immutable DDL text keyed by exact path+source digest. A changed
    # producer cannot reuse old DDL, including outside the guarded test owner.
    source = Path(path_text).read_bytes()
    assert hashlib.sha256(source).hexdigest() == source_digest, "fixture producer changed during source read"
    tree = ast.parse(source.decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_DAEMON_EXECUTION_SQL" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("actual native producer DDL not found")


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / "lane-2"
    directory.mkdir()
    path = directory / "fixture.execution.shard-0002-of-0003.duckdb"
    lock = directory / f".{path.name}.writer.lock"
    lock.touch(mode=0o600)
    source = tmp_path / "fixture_producer.py"
    source.write_text("# explicit disposable source-binding fixture\n", encoding="utf-8")
    metadata = {
        "interface": "DatabaseImplementationDaemon@1",
        "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
        "authority_mode": "quack", "logical_owner_session_id": "embedded-store:fixture-original",
        "process_instance_id": "process:original-fixture", "state_schema_revision": "",
        "control_schema_profile_id": "fixture-profile", "control_schema_fingerprint": "fixture-schema-fingerprint",
        "execution_state_dir": str(directory), "execution_state_prefix": "fixture",
        "task_shard_count": "3", "task_shard_index": "2", "strict_task_sharding": "true",
    }
    connection = duckdb.connect(str(path), config={"threads": "1", "memory_limit": "256MB"})  # Fixture writer, never the product owner.
    try:
        connection.execute(native_fixture_ddl())
        connection.executemany("INSERT INTO daemon_execution_metadata VALUES (?,?)", list(metadata.items()))
        connection.execute("INSERT INTO database_task_attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            "attempt:fixture2", "claim:fixture2", "task:fixture", "FIXTURE-1", 2,
            "embedded-store:fixture-original", 2, 2, "lease:fixture2", "context", "running",
            1000, None, 2, json.dumps({"fixture": True, "feature_score": 0.75}),
        ])
        connection.executemany("INSERT INTO attempt_phases VALUES (?,?,?,?,?,?,?)", [
            ["attempt:fixture2", "claimed", 1000, 2, 2, 1, "{}"],
            ["attempt:fixture2", "context", 1001, 2, 2, 2, "{}"],
        ])
        connection.execute("INSERT INTO provider_invocations VALUES (?,?,?,?,?,?,?)", [
            "invocation:fixture", "attempt:fixture2", "task:fixture", "provider:attempt:fixture2",
            "embedded-store:fixture-original", 1002,
            json.dumps({"callback_state": "started_outcome_unknown", "provider_effect_state": "unknown_may_have_started", "numeric_diagnostic": 0.5}),
        ])
    finally:
        connection.close()
    known = {key: value for key, value in metadata.items() if key not in {
        "process_instance_id", "state_schema_revision", "control_schema_profile_id", "control_schema_fingerprint",
    }}
    producer_path = Path(importlib.util.find_spec("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon").origin)
    reader_path = Path(importlib.import_module(NativeExecutionStatusOwner.__module__).__file__)
    policy_path = Path(duckdb_state.__file__)
    binding = NativeExecutionStatusBinding(
        backend="duckdb", execution_path=str(path), execution_file_identity=identity(path),
        writer_lock_identity=identity(lock), expected_metadata=known,
        source_bindings={str(item): digest(item) for item in (source, producer_path, reader_path, policy_path)}, control_store_id="fixture-primary-uuid",
        control_schema_version=5, task_cid="task:fixture", control_revision=12,
        expected_control_status="in_progress", attempt_id="attempt:fixture2", attempt_number=2,
        claim_id="claim:fixture2", lease_id="lease:fixture2", owner_session_id="embedded-store:fixture-original",
        fencing_token=2, fence_epoch=2,
    )
    return {"path": path, "lock": lock, "source": source, "metadata": metadata, "binding": binding}


def fixture_write(store, sql, values=None):
    # Deliberate mutation of NEW fixture data before any observer opens it.
    connection = duckdb.connect(str(store["path"]), config={"threads": "1", "memory_limit": "256MB"})
    try:
        connection.execute(sql, values or [])
    finally:
        connection.close()
    store["binding"] = replace(store["binding"], execution_file_identity=identity(store["path"]))


def owner(store):
    return NativeExecutionStatusOwner(store["binding"], deadline_monotonic=time.monotonic() + 60)


def assert_released(instance, store):
    assert instance._connection is None
    assert instance._database_fd is None
    assert instance._lock_fd is None
    fd = os.open(store["lock"], os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def control_task(binding):
    return {
        "task_cid": binding.task_cid, "task_alias": "FIXTURE-1", "status": "in_progress", "revision": 12,
        "body": {"completion_receipt": {"operation": "database_claim", "claimed_from_revision": 11, **{
            name: getattr(binding, name) for name in (
                "attempt_id", "attempt_number", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch",
            )
        }}},
    }


def test_real_readonly_snapshot_no_mutation_and_honest_unknown(store):
    before = digest(store["path"]), identity(store["path"]), store["lock"].read_bytes(), identity(store["lock"])
    instance = owner(store)
    with instance:
        result = instance.observe_attempt()
        assert result["attempt"]["attempt_number"] == 2
        assert result["attempt"]["body"]["feature_score"] == 0.75
        assert [row["phase"] for row in result["phase_history"]] == ["claimed", "context"]
        assert result["provider_records"][0]["result"]["numeric_diagnostic"] == 0.5
        assert result["provider_effect_state"] == "unknown"
        assert result["native_store_uuid"] is None
        assert result["preknown_original_process_instance_id"] is False
        assert result["observed_process_instance_id"] == "process:original-fixture"
        assert result["original_process_impersonated"] is False
        assert result["physical_identity_provenance"] == "current_server_selected_identity_not_historical_inode_attestation"
        assert result["completion_authority"] is result["automatic_retry_admitted"] is result["semantic_admission"] is False
        assert result["native_callback_authority_observed"] is result["coordination_authority_observed"] is False
    assert_released(instance, store)
    assert before == (digest(store["path"]), identity(store["path"]), store["lock"].read_bytes(), identity(store["lock"]))
    assert not Path(str(store["path"]) + ".wal").exists()


def test_actual_connection_policy_and_engine_readonly(store, monkeypatch):
    genuine = duckdb.connect
    calls = []

    def capture(*args, **kwargs):
        calls.append((args[0], copy.deepcopy(kwargs)))
        return genuine(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", capture)
    before = digest(store["path"])
    with owner(store) as instance:
        with pytest.raises(duckdb.Error):
            instance._connection.execute("UPDATE database_task_attempts SET status='failed'")
    assert len(calls) == 1 and calls[0][0] == str(store["path"])
    config = calls[0][1]
    assert config["read_only"] is True
    assert config["config"]["threads"] == "1" and config["config"]["memory_limit"] == "256MB"
    assert all(config["config"][name] == "false" for name in (
        "autoinstall_known_extensions", "autoload_known_extensions", "enable_external_access", "allow_unsigned_extensions",
    ))
    assert config["config"]["lock_configuration"] == "true"
    assert digest(store["path"]) == before


def test_binding_roundtrip_is_detached_and_does_not_mint_authority(store):
    binding = store["binding"]
    encoded = binding.to_dict()
    reopened = NativeExecutionStatusBinding.from_dict(encoded)
    assert reopened.binding_cid == binding.binding_cid
    encoded["expected_metadata"]["schema"] = "changed"
    encoded["source_bindings"].clear()
    assert reopened.to_dict() == binding.to_dict()
    with pytest.raises(TypeError):
        binding.expected_metadata["schema"] = "changed"


@pytest.mark.parametrize("backend", ["sqlite", "quack", "ducklake", "DuckDB", ""])
def test_unsupported_backends_refuse(store, backend):
    with pytest.raises(NativeExecutionStatusError, match="unsupported_execution_backend"):
        replace(store["binding"], backend=backend)


@pytest.mark.parametrize("changed", ["missing", "extra", "wrong_schema", "bool_revision"])
def test_closed_binding_shape(store, changed):
    encoded = store["binding"].to_dict()
    if changed == "missing":
        del encoded["claim_id"]
    elif changed == "extra":
        encoded["sql"] = "SELECT 1"
    elif changed == "wrong_schema":
        encoded["schema"] = "old"
    else:
        encoded["control_revision"] = True
    with pytest.raises(NativeExecutionStatusError):
        NativeExecutionStatusBinding.from_dict(encoded)


@pytest.mark.parametrize("field", ["task_cid", "revision", "status", "attempt_id", "claim_id", "lease_id", "owner_session_id", "attempt_number", "fencing_token", "fence_epoch"])
def test_control_join_exact_and_pure(store, field):
    binding = store["binding"]
    task = control_task(binding)
    binding.validate_control_task(task)
    target = task if field in {"task_cid", "revision", "status"} else task["body"]["completion_receipt"]
    target[field] = 99 if type(target[field]) is int else "different"
    with pytest.raises(NativeExecutionStatusError):
        binding.validate_control_task(task)


def test_control_join_rejects_missing_or_unrelated_receipt(store):
    for body in ({}, {"completion_receipt": {}}, {"completion_receipt": {"operation": "recovery"}}):
        task = control_task(store["binding"])
        task["body"] = body
        with pytest.raises(NativeExecutionStatusError):
            store["binding"].validate_control_task(task)


def test_genuine_receipt_without_taskcid_and_contradictory_optional_duplicate(store):
    task = control_task(store["binding"])
    assert "task_cid" not in task["body"]["completion_receipt"]
    store["binding"].validate_control_task(task)
    task["body"]["completion_receipt"]["task_cid"] = "task:wrong"
    with pytest.raises(NativeExecutionStatusError, match="control_claim_join_mismatch"):
        store["binding"].validate_control_task(task)


@pytest.mark.parametrize("missing", ["database", "writer_lock", "source"])
def test_missing_existing_file_never_creates_replacement(store, missing):
    path = store[{"database": "path", "writer_lock": "lock", "source": "source"}[missing]]
    path.unlink()
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="original_file_unavailable"):
        instance.open()
    assert not path.exists()
    assert instance._connection is None and instance._lock_fd is None


@pytest.mark.parametrize("which,alias", [("path", "symlink"), ("path", "hardlink"), ("lock", "symlink"), ("lock", "hardlink")])
def test_aliases_refuse_without_following_or_repair(store, which, alias):
    path = store[which]
    retained = path.with_name(path.name + ".retained")
    path.rename(retained)
    if alias == "symlink":
        path.symlink_to(retained)
    else:
        os.link(retained, path)
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="aliased_original_path|invalid_original_file"):
        instance.open()
    assert retained.exists()
    assert instance._connection is None and instance._lock_fd is None


def test_ancestor_alias_refuses(store):
    directory = store["path"].parent
    retained = directory.with_name(directory.name + "-retained")
    directory.rename(retained)
    directory.symlink_to(retained, target_is_directory=True)
    with pytest.raises(NativeExecutionStatusError, match="aliased_original_path"):
        owner(store).open()


def test_current_identity_drift_refuses_instead_of_inventing_historical_inode(store):
    replacement = store["path"].with_name("different.duckdb")
    replacement.write_bytes(store["path"].read_bytes())
    os.replace(replacement, store["path"])
    with pytest.raises(NativeExecutionStatusError, match="original_file_identity_drift"):
        owner(store).open()


def test_source_binding_drift_refuses(store):
    store["source"].write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(NativeExecutionStatusError, match="source_binding_drift"):
        owner(store).open()


@pytest.mark.parametrize("which", ["reader", "policy", "producer"])
def test_actual_reader_policy_and_producer_sources_are_mandatory(store, which):
    binding = store["binding"]
    names = {
        "reader": "native_execution_status_owner.py",
        "policy": "duckdb_state.py", "producer": "implementation_daemon.py",
    }
    pins = {path: value for path, value in binding.source_bindings.items() if Path(path).name != names[which]}
    with pytest.raises(NativeExecutionStatusError, match="source_not_bound|producer_source_binding"):
        store["binding"] = replace(binding, source_bindings=pins)
        owner(store).open()


def test_substituted_policy_callable_is_refused_before_native_connect(store, monkeypatch):
    def never_authorized(*_args, **_kwargs):
        raise AssertionError("substituted callable must never run")

    monkeypatch.setattr(duckdb_state, "connect_duckdb_with_policy", never_authorized)
    with pytest.raises(NativeExecutionStatusError, match="loaded_backend_policy_source_not_bound"):
        owner(store).open()


@pytest.mark.parametrize("which", ["source", "lock", "path"])
def test_regular_to_fifo_race_cannot_block_before_native_open(store, which, monkeypatch):
    module = importlib.import_module(NativeExecutionStatusOwner.__module__)
    genuine = os.open
    target = store[which]
    raced = []

    def race(path, flags, *args, **kwargs):
        if Path(path) == target and not raced:
            assert flags & os.O_NONBLOCK
            target.unlink()
            os.mkfifo(target, mode=0o600)
            raced.append(True)
        return genuine(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", race)
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="invalid_opened_original_file"):
        instance.open()
    assert raced and instance._connection is None and instance._lock_fd is None


@pytest.mark.parametrize("sql", [
    "DROP TABLE effect_claims",
    "ALTER TABLE daemon_execution_events ADD COLUMN unexpected VARCHAR",
    "ALTER TABLE database_task_attempts ALTER task_alias DROP NOT NULL",
    "ALTER TABLE daemon_execution_metadata ALTER value TYPE BIGINT USING 1",
])
def test_schema_absence_or_drift_never_installs_or_repairs(store, sql):
    if sql == "ALTER TABLE database_task_attempts ALTER task_alias DROP NOT NULL":
        # Genuine producer indexes make DuckDB refuse ALTER before the reader
        # can observe this deliberate fixture schema defect. Drop only these
        # three fixture indexes; keep the production nullability check intact.
        indexes = (
            "database_task_attempts_task_idx",
            "database_task_attempts_owner_idx",
            "database_task_attempts_claim_idx",
        )
        actual_ddl = native_fixture_ddl()
        assert all(f"CREATE INDEX IF NOT EXISTS {name}\n    ON database_task_attempts(" in actual_ddl for name in indexes)
        sql = "\n".join([*(f"DROP INDEX {name};" for name in indexes), sql])
    fixture_write(store, sql)
    before = digest(store["path"])
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="execution_schema"):
        instance.open()
    assert digest(store["path"]) == before
    assert_released(instance, store)


def test_missing_primary_key_refuses_even_when_rows_look_valid(store):
    fixture_write(store, "CREATE TABLE phases_copy AS SELECT * FROM attempt_phases")
    fixture_write(store, "DROP TABLE attempt_phases")
    fixture_write(store, "ALTER TABLE phases_copy RENAME TO attempt_phases")
    # CTAS also drops NOT NULL. Restore it so this targets the constraint test.
    for column in ("attempt_id", "phase", "committed_at_ms", "fencing_token", "fence_epoch", "revision", "body_json"):
        fixture_write(store, f"ALTER TABLE attempt_phases ALTER {column} SET NOT NULL")
    with pytest.raises(NativeExecutionStatusError, match="execution_schema_constraint_mismatch"):
        owner(store).open()


@pytest.mark.parametrize("key", ["schema", "logical_owner_session_id", "execution_state_dir", "execution_state_prefix", "task_shard_index", "strict_task_sharding"])
def test_wrong_native_identity_or_lane_refuses(store, key):
    fixture_write(store, "UPDATE daemon_execution_metadata SET value=? WHERE key=?", ["wrong", key])
    with pytest.raises(NativeExecutionStatusError, match="native_metadata_binding_mismatch"):
        owner(store).open()


@pytest.mark.parametrize("key", ["process_instance_id", "control_schema_profile_id", "control_schema_fingerprint"])
def test_incomplete_native_identity_refuses(store, key):
    fixture_write(store, "DELETE FROM daemon_execution_metadata WHERE key=?", [key])
    with pytest.raises(NativeExecutionStatusError, match="native_identity_incomplete"):
        owner(store).open()


def test_scoped_process_discovery_and_optional_exact_process_pin(store):
    binding = store["binding"]
    with owner(store) as instance:
        observed = instance.observe_attempt()
    assert observed["preknown_original_process_instance_id"] is False
    store["binding"] = replace(binding, expected_metadata={**binding.expected_metadata, "process_instance_id": "process:wrong"})
    with pytest.raises(NativeExecutionStatusError, match="native_metadata_binding_mismatch"):
        owner(store).open()


@pytest.mark.parametrize("profile,fingerprint", [("datasets-authoritative-operational-control-plane@1", ""), ("", "")])
def test_real_producer_permitted_empty_control_metadata_is_observed_not_fabricated(store, profile, fingerprint):
    fixture_write(store, "UPDATE daemon_execution_metadata SET value=? WHERE key='control_schema_profile_id'", [profile])
    fixture_write(store, "UPDATE daemon_execution_metadata SET value=? WHERE key='control_schema_fingerprint'", [fingerprint])
    with owner(store) as instance:
        result = instance.observe_attempt()
    assert result["native_metadata"]["control_schema_profile_id"] == profile
    assert result["native_metadata"]["control_schema_fingerprint"] == fingerprint
    assert result["native_control_schema_profile_recorded"] is bool(profile)
    assert result["native_control_schema_fingerprint_recorded"] is False


def test_existing_writer_fails_bounded_and_does_not_unlock_original_writer(store):
    fd = os.open(store["lock"], os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        instance = owner(store)
        with pytest.raises(NativeExecutionStatusError, match="original_execution_writer_active"):
            instance.open()
        assert instance._connection is None and instance._lock_fd is None
    finally:
        os.close(fd)


def test_wal_is_retained_and_never_recovered_writably(store):
    wal = Path(str(store["path"]) + ".wal")
    wal.write_bytes(b"explicit fixture sentinel, not a real WAL")
    with pytest.raises(NativeExecutionStatusError, match="execution_wal_requires_native_review"):
        owner(store).open()
    assert wal.read_bytes() == b"explicit fixture sentinel, not a real WAL"


@pytest.mark.parametrize("field,value", [
    ("claim_id", "claim:other"), ("lease_id", "lease:other"),
    ("owner_session_id", "owner:other"), ("task_cid", "task:other"),
    ("attempt_number", 1), ("fencing_token", 3), ("fence_epoch", 3),
])
def test_wrong_attempt_join_refuses_and_closes(store, field, value):
    fixture_write(store, f"UPDATE database_task_attempts SET {field}=?", [value])
    instance = owner(store).open()
    with pytest.raises(NativeExecutionStatusError, match="native_attempt_join_mismatch"):
        instance.observe_attempt()
    assert_released(instance, store)


@pytest.mark.parametrize("raw", ["{", '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', "[]", '{"x":"' + "z" * MAX_JSON_BYTES + '"}'],
    ids=["malformed", "duplicate_key", "nan", "exponent_overflow", "array", "oversized"])
def test_malformed_nonfinite_duplicate_or_oversized_json_never_becomes_empty_evidence(store, raw):
    fixture_write(store, "UPDATE provider_invocations SET result_json=?", [raw])
    instance = owner(store).open()
    with pytest.raises(NativeExecutionStatusError):
        instance.observe_attempt()
    assert_released(instance, store)


def test_missing_attempt_is_unknown_and_incomplete_not_nondispatch(store):
    for table in ("database_task_attempts", "attempt_phases", "provider_invocations"):
        fixture_write(store, f"DELETE FROM {table}")
    with owner(store) as instance:
        result = instance.observe_attempt()
    assert result["attempt"] is None and result["attempt_observation"] == "missing_record"
    assert result["provider_records"] == result["effect_records"] == []
    assert result["provider_effect_state"] == "unknown"
    assert result["automatic_retry_admitted"] is False


def test_phase_fence_mismatch_refuses(store):
    fixture_write(store, "UPDATE attempt_phases SET fence_epoch=3 WHERE phase='context'")
    with owner(store) as instance:
        with pytest.raises(NativeExecutionStatusError, match="native_phase_join_mismatch"):
            instance.observe_attempt()


def test_phase_cursor_inconsistency_refuses(store):
    fixture_write(store, "DELETE FROM attempt_phases WHERE phase='context'")
    with owner(store) as instance:
        with pytest.raises(NativeExecutionStatusError, match="native_phase_cursor_inconsistent"):
            instance.observe_attempt()


def test_history_bound_refuses_instead_of_truncating(store):
    for index in range(9):
        fixture_write(store, "INSERT INTO attempt_phases VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING", [
            "attempt:fixture2", f"fixture-extra-{index}", 2000 + index, 2, 2, 2, "{}",
        ])
    with owner(store) as instance:
        with pytest.raises(NativeExecutionStatusError, match="native_result_bound_exceeded"):
            instance.observe_attempt()


def test_provider_record_bound_refuses_instead_of_truncating(store):
    for index in range(8):
        fixture_write(store, "INSERT INTO provider_invocations VALUES (?,?,?,?,?,?,?)", [
            f"invocation:extra{index}", "attempt:fixture2", "task:fixture", f"provider:extra{index}",
            "embedded-store:fixture-original", 2000 + index, "{}",
        ])
    with owner(store) as instance:
        with pytest.raises(NativeExecutionStatusError, match="native_result_bound_exceeded"):
            instance.observe_attempt()


def test_provider_identity_mismatch_refuses(store):
    fixture_write(store, "UPDATE provider_invocations SET owner_session_id='other'")
    with owner(store) as instance:
        with pytest.raises(NativeExecutionStatusError, match="native_idempotency_join_mismatch"):
            instance.observe_attempt()


@pytest.mark.parametrize("table,column", [
    ("daemon_execution_metadata", "key"), ("daemon_execution_metadata", "value"),
    ("database_task_attempts", "task_alias"), ("provider_invocations", "invocation_id"),
    ("provider_invocations", "idempotency_key"),
])
def test_variable_scalar_bytes_are_bounded_before_python_materialization(store, table, column):
    if table == "daemon_execution_metadata":
        fixture_write(store, f"UPDATE {table} SET {column}=? WHERE key='state_schema_revision'", ["z" * 65536])
    else:
        fixture_write(store, f"UPDATE {table} SET {column}=?", ["z" * 65536])
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError):
        with instance:
            instance.observe_attempt()
    assert_released(instance, store)


def test_returned_nested_results_do_not_mutate_next_observation(store):
    with owner(store) as instance:
        first = instance.observe_attempt()
        first["attempt"]["body"]["fixture"] = False
        first["provider_records"][0]["result"].clear()
        first["native_metadata"].clear()
        again = instance.observe_attempt()
        assert again["attempt"]["body"]["fixture"] is True
        assert again["provider_records"][0]["result"]["callback_state"] == "started_outcome_unknown"
        assert again["native_metadata"]["schema"] == store["metadata"]["schema"]


def test_closed_getter_and_reopen_never_rewrite_owner(store):
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="status_owner_not_open"):
        instance.observe_attempt()
    instance.open()
    instance.close()
    before = digest(store["path"])
    for action in (instance.observe_attempt, instance.open):
        with pytest.raises(NativeExecutionStatusError, match="status_owner_closed"):
            action()
    assert digest(store["path"]) == before
    assert_released(instance, store)


def test_cancellation_closes_native_connection_and_lock(store):
    instance = owner(store).open()
    instance.cancel()
    assert_released(instance, store)
    with pytest.raises(NativeExecutionStatusError, match="status_owner_closed"):
        instance.observe_attempt()


@pytest.mark.parametrize("deadline", [0, True, float("inf"), float("nan"), "90"])
def test_invalid_deadline_cannot_extend_owner_lifetime(store, deadline):
    with pytest.raises(NativeExecutionStatusError, match="invalid_status_deadline"):
        NativeExecutionStatusOwner(store["binding"], deadline_monotonic=deadline)


def test_expired_deadline_refuses_query_and_closes_on_context_exit(store, monkeypatch):
    instance = owner(store).open()
    module = importlib.import_module(NativeExecutionStatusOwner.__module__)
    monkeypatch.setattr(module.time, "monotonic", lambda: instance.deadline_monotonic + 1)
    try:
        with pytest.raises(NativeExecutionStatusError, match="status_deadline_expired"):
            instance.observe_attempt()
    finally:
        instance.close()
    assert_released(instance, store)


def test_native_open_failure_is_sanitized_and_releases_custody(store, monkeypatch):
    def explicit_fixture_failure(*_args, **_kwargs):
        raise RuntimeError("private fixture path must not be exported")

    monkeypatch.setattr(duckdb, "connect", explicit_fixture_failure)
    instance = owner(store)
    with pytest.raises(NativeExecutionStatusError, match="^native_read_only_open_failed$"):
        instance.open()
    assert_released(instance, store)


def test_query_failure_closes_and_does_not_claim_an_empty_result(store, monkeypatch):
    genuine = duckdb.connect

    class ExplicitFixtureFailureConnection:
        def __init__(self, native):
            self.native = native

        def execute(self, sql, values=None):
            if "FROM main.database_task_attempts" in sql:
                raise RuntimeError("explicit unit fixture query fault")
            return self.native.execute(sql, values or [])

        def interrupt(self):
            self.native.interrupt()

        def close(self):
            self.native.close()

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: ExplicitFixtureFailureConnection(genuine(*args, **kwargs)))
    instance = owner(store).open()
    with pytest.raises(NativeExecutionStatusError, match="native_read_failed"):
        instance.observe_attempt()
    assert_released(instance, store)


def test_unverified_native_close_retains_writer_guard_until_real_close(store, monkeypatch):
    genuine = duckdb.connect

    class ExplicitFixtureCloseFailure:
        def __init__(self, native):
            self.native = native
            self.fail_once = True

        def execute(self, sql, values=None):
            return self.native.execute(sql, values or [])

        def interrupt(self):
            self.native.interrupt()

        def close(self):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("explicit fixture uncertain close")
            self.native.close()

    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: ExplicitFixtureCloseFailure(genuine(*args, **kwargs)))
    instance = owner(store).open()
    with pytest.raises(NativeExecutionStatusError, match="native_connection_close_unverified"):
        instance.close()
    assert instance._connection is not None and instance._lock_fd is not None
    fd = os.open(store["lock"], os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    instance.close()
    assert_released(instance, store)


@pytest.mark.parametrize("failed_closes", [0, 2])
def test_policy_verification_failure_retains_raw_handle_until_verified_close(store, monkeypatch, failed_closes):
    genuine = duckdb.connect
    captured = []

    class ExplicitFixturePolicyFailure:
        def __init__(self, native):
            self.native = native
            self.remaining_failed_closes = failed_closes

        def execute(self, sql, values=None):
            if "current_setting(" in sql:
                raise RuntimeError("explicit fixture policy verification fault")
            return self.native.execute(sql, values or [])

        def interrupt(self):
            self.native.interrupt()

        def close(self):
            if self.remaining_failed_closes:
                self.remaining_failed_closes -= 1
                raise RuntimeError("explicit fixture uncertain factory cleanup")
            self.native.close()

    def connect(*args, **kwargs):
        wrapped = ExplicitFixturePolicyFailure(genuine(*args, **kwargs))
        captured.append(wrapped)
        return wrapped

    monkeypatch.setattr(duckdb, "connect", connect)
    instance = owner(store)
    reason = "native_connection_close_unverified" if failed_closes else "native_read_only_open_failed"
    with pytest.raises(NativeExecutionStatusError, match="^" + reason + "$"):
        instance.open()
    assert len(captured) == 1
    if failed_closes:
        assert instance._connection is captured[0]
        assert instance._database_fd is not None and instance._lock_fd is not None
        assert captured[0].native.execute("SELECT 1").fetchone() == (1,)
        fd = os.open(store["lock"], os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        instance.close()
    assert_released(instance, store)


def test_no_public_mutation_dispatch_or_sql_surface(store):
    instance = owner(store)
    assert {name for name, value in vars(type(instance)).items() if not name.startswith("_") and callable(value)} == {"open", "observe_attempt", "close", "cancel"}
    with pytest.raises(AttributeError):
        instance.binding = replace(store["binding"], control_revision=13)
    with pytest.raises(AttributeError):
        instance.deadline_monotonic = time.monotonic() + 300
    with pytest.raises(TypeError):
        instance.observe_attempt(sql="SELECT 1")
    with pytest.raises(TypeError):
        instance.observe_attempt(attempt_id="attempt:other")
    instance.close()
