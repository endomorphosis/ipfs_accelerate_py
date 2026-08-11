from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import ipfs_accelerate_py.agent_supervisor.merge.lease_coordination as coordination_module
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import (
    LeaseCoordinator,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    MergeResolverRegistry,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DUCKDB_CONNECTION_POLICY_SETTINGS,
    DuckDBConnectionPolicyError,
    connect_duckdb_with_policy,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
    materialize_duckdb_task_source,
)


def _bundle() -> dict[str, object]:
    return {
        "bundle_key": "objective/connection-policy/lease",
        "parallel_lane": "policy-lane-a",
        "todo_path": "policy.todo.md",
        "source_todo": "policy.todo.md",
        "tasks": [{"task_id": "ASE3-032"}],
    }


def _assert_connect_config(config: object) -> None:
    assert isinstance(config, dict)
    expected = {
        name: configured
        for name, configured, _value in DUCKDB_CONNECTION_POLICY_SETTINGS
        if name != "lock_configuration"
    }
    expected.update({"threads": "1", "memory_limit": "256MB"})
    expected["lock_configuration"] = "true"
    assert config == expected
    assert tuple(config) == tuple(expected)
    assert all(type(key) is str for key in config)
    assert all(type(value) is str for value in config.values())


def _formal_source() -> dict[str, object]:
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": "tree:connection-policy",
        "objectives": [
            {
                "goal_id": "G32",
                "goal_cid": "goal:cid:g32",
                "owner_actor_id": "owner:supervisor",
                "title": "Seal DuckDB connection policy",
                "acceptance_criteria": ["Every connection is policy-bound."],
            }
        ],
        "taskboard": [
            {
                "task_id": "ASE3-032",
                "task_cid": "task:cid:ase3-032",
                "goal_id": "G32",
                "actor_id": "agent:policy",
                "resource_needs": ["duckdb"],
                "changed_ast_scopes": ["symbol:cid:duckdb-policy"],
                "acceptance_criteria": ["connection-policy tests pass"],
                "validation_commands": [
                    "pytest test_agent_supervisor_duckdb_connection_policy.py"
                ],
            }
        ],
        "ast_records": [
            {
                "symbol_cid": "symbol:cid:duckdb-policy",
                "tree_cid": "tree:connection-policy",
                "task_cid": "task:cid:ase3-032",
                "symbol": "connect_duckdb_with_policy",
            }
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:ase3-032",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
        },
        "evidence_records": [
            {
                "evidence_cid": "evidence:cid:duckdb-policy",
                "task_cid": "task:cid:ase3-032",
                "kind": "test",
            }
        ],
    }


def _install_populated_legacy_coordination_schema(path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    with LeaseCoordinator(path):
        pass
    connection = duckdb.connect(str(path))
    try:
        connection.execute("DROP TABLE tasks")
        connection.execute(
            """
            CREATE TABLE tasks(
              task_cid TEXT PRIMARY KEY,
              goal_cid TEXT NOT NULL,
              subgoal_cid TEXT NOT NULL,
              task_id TEXT NOT NULL,
              bundle_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
            [
                "task:legacy",
                "goal:legacy",
                "subgoal:legacy",
                "ASE3-LEGACY",
                "{}",
            ],
        )
        connection.execute("DROP TABLE leases")
        connection.execute(
            """
            CREATE TABLE leases(
              task_cid TEXT PRIMARY KEY,
              claim_cid TEXT NOT NULL,
              resolution_cid TEXT NOT NULL,
              claimant_did TEXT NOT NULL,
              logical_epoch BIGINT NOT NULL,
              fencing_token BIGINT NOT NULL,
              expires_at_ms BIGINT NOT NULL,
              attempt BIGINT NOT NULL,
              state TEXT NOT NULL,
              started_at_ms BIGINT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "task:legacy",
                "claim:legacy",
                "resolution:legacy",
                "did:web:legacy.example",
                1,
                1,
                10_000,
                1,
                "accepted",
                1_000,
            ],
        )
    finally:
        connection.close()


def _runtime_connection_site(
    frames: tuple[tuple[str, str], ...],
    *,
    database: str,
    read_only: bool,
) -> str:
    functions: dict[str, set[str]] = {}
    for filename, function in frames:
        functions.setdefault(filename, set()).add(function)
    if "merge_queue.py" in functions:
        if "_init_database" in functions["merge_queue.py"]:
            return "merge_queue.initialize"
        if "_import_legacy_files" in functions["merge_queue.py"]:
            return "merge_queue.legacy_import"
        return "merge_queue.operation"
    if "merge_resolver.py" in functions:
        if "_connect" not in functions["merge_resolver.py"]:
            return "merge_resolver.initialize"
        return "merge_resolver.operation"
    if "duckdb_task_source.py" in functions:
        if "materialize" in functions["duckdb_task_source.py"]:
            return "duckdb_task_source.materialize"
        return "duckdb_task_source.snapshot"
    if "lease_coordination.py" in functions:
        lease_functions = functions["lease_coordination.py"]
        if "compact" in lease_functions:
            if read_only:
                return "lease_compaction.source_read_only"
            if "_database_operation" in lease_functions:
                return "lease_compaction.target_initialize"
            assert ".compact-" in database
            return "lease_compaction.target_write"
        if "__init__" in lease_functions:
            return "lease_coordinator.initialize"
        return "lease_coordinator.operation"
    raise AssertionError(f"unrecognized DuckDB runtime connection stack: {frames}")


def test_policy_is_atomic_verified_on_returned_connection_and_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    original_connect = duckdb.connect
    observed: list[dict[str, Any]] = []

    def capture_connect(*args: object, **kwargs: Any) -> Any:
        observed.append(dict(kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", capture_connect)
    connection = connect_duckdb_with_policy(
        duckdb,
        ":memory:",
    )
    try:
        assert len(observed) == 1
        assert observed[0]["read_only"] is False
        _assert_connect_config(observed[0]["config"])
        row = connection.execute(
            """
            SELECT current_setting('autoinstall_known_extensions'),
                   current_setting('autoload_known_extensions'),
                   current_setting('enable_external_access'),
                   current_setting('allow_unsigned_extensions'),
                   current_setting('lock_configuration')
            """
        ).fetchone()
        assert row == (False, False, False, False, True)
        assert all(type(value) is bool for value in row)
        for statement in (
            "SET autoinstall_known_extensions=true",
            "SET autoload_known_extensions=true",
            "SET enable_external_access=true",
            "SET allow_unsigned_extensions=true",
            "SET lock_configuration=false",
            "RESET enable_external_access",
            "SET GLOBAL enable_external_access=true",
        ):
            with pytest.raises(duckdb.InvalidInputException, match="locked"):
                connection.execute(statement)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("configuration", "error_type"),
    [
        ({"autoinstall_known_extensions": "true"}, ValueError),
        ({"AUTOLOAD_KNOWN_EXTENSIONS": "false"}, ValueError),
        ({" enable_external_access ": "false"}, ValueError),
        ({"allow_unsigned_extensions": "false"}, ValueError),
        ({"lock_configuration": "true"}, ValueError),
        ({"threads": True}, TypeError),
        ({"threads": "1"}, TypeError),
        ({"threads": 0}, ValueError),
        ({"threads": 257}, ValueError),
        ({"memory_limit": object()}, TypeError),
        ({"memory_limit": 256}, TypeError),
        ({"memory_limit": ""}, ValueError),
        ({"memory_limit": "0MB"}, ValueError),
        ({"memory_limit": "257MB"}, ValueError),
        ({"memory_limit": "1GB"}, ValueError),
        ({"memory_limit": "256MB; SET enable_external_access=true"}, ValueError),
        ({"memory_limit": " 256MB"}, ValueError),
        ({"memory_limit": "256mb"}, ValueError),
        ({"home_directory": "/tmp/duckdb"}, ValueError),
        ({1: "threads"}, TypeError),
        ({" THREADS ": 1}, ValueError),
        ({"threads": 1, " THREADS ": 1}, ValueError),
    ],
)
def test_caller_cannot_override_or_smuggle_connection_configuration(
    configuration: dict[object, object],
    error_type: type[Exception],
) -> None:
    class NeverConnect:
        @staticmethod
        def connect(*_args: object, **_kwargs: object) -> Any:
            raise AssertionError("invalid configuration reached duckdb.connect")

    with pytest.raises(error_type):
        connect_duckdb_with_policy(  # type: ignore[arg-type]
            NeverConnect,
            ":memory:",
            configuration=configuration,
        )


def test_policy_verification_rejects_integer_bool_spoof_and_closes() -> None:
    class IntegerPolicyConnection:
        def __init__(self) -> None:
            self.closed = False
            self.statements: list[str] = []

        def execute(self, statement: str) -> IntegerPolicyConnection:
            self.statements.append(statement)
            return self

        @staticmethod
        def fetchone() -> tuple[int, int, int, int, int]:
            return (0, 0, 0, 0, 1)

        def close(self) -> None:
            self.closed = True

    connection = IntegerPolicyConnection()

    class LyingDuckDB:
        @staticmethod
        def connect(*_args: object, **_kwargs: object) -> IntegerPolicyConnection:
            return connection

    with pytest.raises(DuckDBConnectionPolicyError, match="verification failed"):
        connect_duckdb_with_policy(LyingDuckDB, ":memory:")
    assert connection.closed
    assert len(connection.statements) == 1
    assert connection.statements[0].startswith("SELECT current_setting")


@pytest.mark.parametrize(
    "statement",
    [
        "INSTALL httpfs",
        "LOAD httpfs",
        "SELECT * FROM read_csv_auto('http://127.0.0.1:9/not-reached.csv')",
    ],
)
def test_policy_blocks_dynamic_extension_fetch_load_and_http_access(
    statement: str,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        with pytest.raises(duckdb.PermissionException):
            connection.execute(statement)
    finally:
        connection.close()


@pytest.mark.parametrize("extension", ["json", "parquet"])
def test_reviewed_statically_linked_modules_do_not_cross_external_byte_boundary(
    extension: str,
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        # These modules are compiled into the separately pinned native payload;
        # loading one does not fetch or map new extension bytes.
        connection.execute(f"LOAD {extension}")
        with pytest.raises(duckdb.PermissionException):
            connection.execute(
                f"SELECT * FROM read_parquet('{tmp_path / 'external.parquet'}')"
            )
    finally:
        connection.close()


def test_only_in_memory_attach_is_nonexternal(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    external = tmp_path / "external.duckdb"
    duckdb.connect(str(external)).close()
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        connection.execute("ATTACH ':memory:' AS transient_memory")
        connection.execute("CREATE TABLE transient_memory.main.probe(value INTEGER)")
        connection.execute("DETACH transient_memory")
        with pytest.raises(duckdb.PermissionException):
            connection.execute(f"ATTACH '{external}' AS external_file")
    finally:
        connection.close()


def test_policy_blocks_arbitrary_local_extension_load(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    payload = tmp_path / "malicious.duckdb_extension"
    payload.write_bytes(b"not an extension")
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        with pytest.raises(duckdb.PermissionException):
            connection.execute(f"LOAD '{payload}'")
    finally:
        connection.close()


def test_every_accepted_runtime_connection_site_uses_the_canonical_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    original_connect = duckdb.connect
    observed: list[dict[str, Any]] = []

    def capture_connect(*args: object, **kwargs: Any) -> Any:
        assert len(args) == 1
        assert type(args[0]) is str
        assert type(kwargs.get("read_only")) is bool
        frames = tuple(
            (Path(frame.filename).name, frame.function)
            for frame in inspect.stack()[1:]
            if "ipfs_accelerate_py/agent_supervisor" in frame.filename
        )
        observed.append(
            {
                "site": _runtime_connection_site(
                    frames,
                    database=args[0],
                    read_only=kwargs["read_only"],
                ),
                "args": args,
                **kwargs,
            }
        )
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", capture_connect)

    queue = MergeQueue(tmp_path / "merge-queue")
    assert queue.pending_count() == 0

    resolver = MergeResolverRegistry(tmp_path / "merge-resolver")
    assert resolver.active_attempt("missing-fingerprint") is None

    task_source = materialize_duckdb_task_source(
        tmp_path / "task-source.duckdb",
        _formal_source(),
    )
    assert task_source.snapshot().task_count == 1

    coordination_path = tmp_path / "coordination.duckdb"
    with LeaseCoordinator(coordination_path) as coordinator:
        coordinator.register_bundle(_bundle(), created_at_ms=1_000)
        coordinator.compact()

    assert Counter(item["site"] for item in observed) == Counter(
        {
            "merge_queue.initialize": 1,
            "merge_queue.legacy_import": 1,
            "merge_queue.operation": 1,
            "merge_resolver.initialize": 1,
            "merge_resolver.operation": 1,
            "duckdb_task_source.materialize": 1,
            "duckdb_task_source.snapshot": 1,
            "lease_coordinator.initialize": 1,
            "lease_coordinator.operation": 1,
            "lease_compaction.target_initialize": 1,
            "lease_compaction.source_read_only": 1,
            "lease_compaction.target_write": 1,
        }
    )
    for item in observed:
        _assert_connect_config(item.get("config"))


@pytest.mark.parametrize(
    ("case", "statements"),
    [
        ("main_table", ("CREATE TABLE foreign_main(value INTEGER)",)),
        ("main_view", ("CREATE VIEW foreign_main AS SELECT 1 AS value",)),
        ("empty_schema", ("CREATE SCHEMA foreign_schema",)),
        (
            "foreign_schema_table",
            (
                "CREATE SCHEMA foreign_schema",
                "CREATE TABLE foreign_schema.state(value INTEGER)",
            ),
        ),
        (
            "foreign_schema_view",
            (
                "CREATE SCHEMA foreign_schema",
                "CREATE VIEW foreign_schema.state AS SELECT 1 AS value",
            ),
        ),
        ("sequence", ("CREATE SEQUENCE foreign_sequence",)),
        ("macro", ("CREATE MACRO foreign_macro(value) AS value + 1",)),
        ("custom_type", ("CREATE TYPE foreign_type AS ENUM ('a', 'b')",)),
        ("index", ("CREATE INDEX foreign_index ON tasks(task_id)",)),
        (
            "check_constraint",
            (
                "DROP TABLE worker_capability_receipts",
                """
                CREATE TABLE worker_capability_receipts(
                  receipt_id TEXT PRIMARY KEY,
                  worker_id TEXT NOT NULL CHECK(length(worker_id)>0),
                  expires_at_ms BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                )
                """,
            ),
        ),
        (
            "unique_constraint",
            (
                "DROP TABLE worker_capability_receipts",
                """
                CREATE TABLE worker_capability_receipts(
                  receipt_id TEXT PRIMARY KEY,
                  worker_id TEXT NOT NULL UNIQUE,
                  expires_at_ms BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                )
                """,
            ),
        ),
        (
            "collation",
            (
                "DROP TABLE worker_capability_receipts",
                """
                CREATE TABLE worker_capability_receipts(
                  receipt_id TEXT PRIMARY KEY,
                  worker_id TEXT COLLATE nocase NOT NULL,
                  expires_at_ms BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                )
                """,
            ),
        ),
        (
            "regular_column",
            ("ALTER TABLE tasks ADD COLUMN foreign_column BIGINT DEFAULT 7",),
        ),
        (
            "generated_column",
            (
                "DROP TABLE tasks",
                """
                CREATE TABLE tasks(
                  task_cid TEXT PRIMARY KEY,
                  goal_cid TEXT NOT NULL,
                  subgoal_cid TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  bundle_json TEXT NOT NULL,
                  registered_at_ms BIGINT NOT NULL DEFAULT 0,
                  updated_at_ms BIGINT GENERATED ALWAYS AS
                    (registered_at_ms + 1)
                )
                """,
            ),
        ),
        ("table_comment", ("COMMENT ON TABLE tasks IS 'foreign schema state'",)),
    ],
)
def test_compaction_rejects_any_foreign_persistent_catalog_without_source_change(
    case: str,
    statements: tuple[str, ...],
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / f"coordination-{case}.duckdb"
    with LeaseCoordinator(path):
        pass
    connection = duckdb.connect(str(path))
    try:
        for statement in statements:
            connection.execute(statement)
    finally:
        connection.close()

    with LeaseCoordinator(path) as coordinator:
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match="coordination compaction"):
            coordinator.compact()
        assert path.read_bytes() == before
    assert not list(tmp_path.glob(f".{path.name}.compact-*"))


def test_populated_legacy_additive_schema_upgrades_idempotently_and_compacts(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "legacy-additive.duckdb"
    _install_populated_legacy_coordination_schema(path)

    for _reopen in range(2):
        with LeaseCoordinator(path):
            pass
    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute(
            "SELECT registered_at_ms, updated_at_ms FROM tasks"
        ).fetchone() == (0, 0)
        assert connection.execute(
            "SELECT release_reason, retry_not_before_ms FROM leases"
        ).fetchone() == (None, 0)
    finally:
        connection.close()

    with LeaseCoordinator(path) as coordinator:
        coordinator.compact()
    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute("SELECT * FROM tasks").fetchone() == (
            "task:legacy",
            "goal:legacy",
            "subgoal:legacy",
            "ASE3-LEGACY",
            "{}",
            0,
            0,
        )
        assert connection.execute("SELECT * FROM leases").fetchone() == (
            "task:legacy",
            "claim:legacy",
            "resolution:legacy",
            "did:web:legacy.example",
            1,
            1,
            10_000,
            1,
            "accepted",
            1_000,
            None,
            0,
        )
    finally:
        connection.close()


def test_legacy_additive_schema_failure_between_steps_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "legacy-additive-rollback.duckdb"
    _install_populated_legacy_coordination_schema(path)

    def fail_after_default_backfill(
        connection: Any,
        *,
        table: str,
        column: str,
    ) -> None:
        assert (table, column) == ("tasks", "registered_at_ms")
        connection.execute(
            'ALTER TABLE "tasks" ADD COLUMN "registered_at_ms" BIGINT DEFAULT 0'
        )
        raise RuntimeError("injected additive-schema interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            coordination_module,
            "_add_coordination_not_null_default_column",
            fail_after_default_backfill,
        )
        with pytest.raises(RuntimeError, match="additive-schema interruption"):
            LeaseCoordinator(path)

    connection = duckdb.connect(str(path), read_only=True)
    try:
        task_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        assert "registered_at_ms" not in task_columns
        assert connection.execute("SELECT * FROM tasks").fetchone() == (
            "task:legacy",
            "goal:legacy",
            "subgoal:legacy",
            "ASE3-LEGACY",
            "{}",
        )
    finally:
        connection.close()

    with LeaseCoordinator(path) as coordinator:
        coordinator.compact()


def test_compaction_partial_copy_failure_keeps_authoritative_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    path = tmp_path / "coordination.duckdb"
    with LeaseCoordinator(path) as coordinator:
        registered = coordinator.register_bundle(_bundle(), created_at_ms=1_000)
        before = path.read_bytes()

        def fail_after_partial_target_write(_source: Any, target: Any) -> None:
            target.execute("DELETE FROM coordination_metadata")
            target.execute(
                "INSERT INTO coordination_metadata VALUES (?, ?)",
                ("partial", "uncommitted"),
            )
            raise RuntimeError("injected partial compaction copy")

        monkeypatch.setattr(
            coordination_module,
            "_copy_coordination_store_rows",
            fail_after_partial_target_write,
        )
        with pytest.raises(RuntimeError, match="injected partial"):
            coordinator.compact()

        assert path.read_bytes() == before
        assert coordinator.claimability(registered["task_cid"])["claimable"]
    assert not list(tmp_path.glob(".coordination.duckdb.compact-*"))


def test_compaction_atomic_replace_failure_keeps_authoritative_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    path = tmp_path / "coordination.duckdb"
    with LeaseCoordinator(path) as coordinator:
        registered = coordinator.register_bundle(_bundle(), created_at_ms=1_000)
        before = path.read_bytes()

        def fail_replace(_source: object, target: object) -> None:
            assert Path(target) == path
            raise OSError("injected atomic replacement failure")

        monkeypatch.setattr(coordination_module.os, "replace", fail_replace)
        with pytest.raises(OSError, match="atomic replacement"):
            coordinator.compact()

        assert path.read_bytes() == before
        assert coordinator.claimability(registered["task_cid"])["claimable"]
    assert not list(tmp_path.glob(".coordination.duckdb.compact-*"))
