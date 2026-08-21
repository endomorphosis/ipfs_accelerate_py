from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path
from types import ModuleType

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_architecture_refactorer.py"


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcar_bootstrap_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owner_mutation_vocabulary_is_closed() -> None:
    operator = _operator()

    assert operator._normalized_owner_dml(" update tasks set status = ? ").startswith(
        "UPDATE "
    )
    with pytest.raises(operator.OperatorError, match="closed owner-DML"):
        operator._normalized_owner_dml("SELECT * FROM tasks")
    with pytest.raises(operator.OperatorError, match="exactly one SQL statement"):
        operator._normalized_owner_dml("UPDATE tasks SET status='ready'; DELETE FROM tasks")


def test_atomic_json_is_private_and_canonical(tmp_path: Path) -> None:
    operator = _operator()
    target = tmp_path / "receipt.json"

    operator._atomic_json(target, {"z": 2, "a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "z": 2}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_task_status_distinguishes_dependency_waiting_from_blocked() -> None:
    operator = _operator()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE tasks (task_cid VARCHAR, task_alias VARCHAR, ordinal BIGINT, "
            "status VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE task_dependencies (task_cid VARCHAR, dependency_task_cid VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE task_blocks (task_cid VARCHAR, state VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?)",
            [
                ("cid:000", "PCAR-000", 1, "todo"),
                ("cid:001", "PCAR-001", 2, "todo"),
                ("cid:002", "PCAR-002", 3, "blocked"),
            ],
        )
        connection.execute(
            "INSERT INTO task_dependencies VALUES ('cid:001', 'cid:000')"
        )

        observed = operator._task_status(connection)
    finally:
        connection.close()

    assert observed["dependency_ready_task_ids"] == ["PCAR-000"]
    assert observed["blocked_count"] == 1
    assert observed["active_task_ids"] == []
    assert observed["task_count"] == 3


def test_token_path_uses_only_the_opaque_handle(tmp_path: Path) -> None:
    operator = _operator()

    path = operator._token_path(tmp_path, "handle:pcar-v1")

    assert path == tmp_path / "handle_pcar-v1.quack-token"
    assert "token-value" not in str(path)


def test_state_owner_verifies_the_canonical_full_control_plane(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    operator = _operator()
    database = tmp_path / "control.duckdb"
    with DatabaseTaskSource(database, owner_id="pcar-full-schema-test"):
        pass

    report = operator._verify_control_plane(database)

    assert report.from_version == 1
    assert report.to_version == 1
    assert report.changed is False
    assert report.schema_fingerprint
    assert report.catalog_fingerprint
