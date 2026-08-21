from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_residual_intelligence.py"
CONFIG_PATH = ROOT / "config/agent_supervisor_residual_intelligence_scheduler.json"


def _operator():
    spec = importlib.util.spec_from_file_location("vrif_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bold_and_plain_board_metadata_are_equivalent() -> None:
    operator = _operator()
    assert operator._metadata_value("completed") == "completed"
    assert operator._metadata_value("** completed") == "completed"
    goals = operator._goal_blocks("## VRIF-G000 Root\n- **Status:** active\n- **Parent:**\n")
    assert goals == [("VRIF-G000", "Root", {"status": "active", "parent": ""})]


def test_runtime_config_closes_authority_and_training_fallbacks() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    projection = config["initial_projection"]
    assert projection["task_count"] == 33
    assert projection["task_dependency_count"] == 111
    assert projection["goal_count"] == 9
    assert projection["ready_task_ids"] == [
        "VRIF-009",
        "VRIF-010",
        "VRIF-011",
        "VRIF-012",
    ]
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["training_policy"]["training_enabled_at_bootstrap"] is False


def test_mutation_inbox_is_closed_to_owner_dml() -> None:
    operator = _operator()
    assert operator._normalized_owner_dml(
        "UPDATE tasks SET status = ?, revision = ? "
        "WHERE task_cid = ? AND revision = ?",
        ["claimed", 2, "cid", 1],
    )
    with pytest.raises(operator.OperatorError):
        operator._normalized_owner_dml("SELECT * FROM tasks")
    with pytest.raises(operator.OperatorError):
        operator._normalized_owner_dml("UPDATE tasks SET status = 'x'; DELETE FROM tasks")
    with pytest.raises(operator.OperatorError):
        operator._normalized_owner_dml("UPDATE tasks SET status = 'completed'")
    with pytest.raises(operator.OperatorError):
        operator._normalized_owner_dml(
            "DELETE FROM completion_receipts WHERE receipt_cid = ?", ["receipt"]
        )


def test_mutation_inbox_requires_signed_store_bound_envelope(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        quack_owner_mutation_signature,
    )

    operator = _operator()
    request_id = "a" * 32
    token = "vrif_test_token_0123456789"
    payload = {
        "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": "supervisor-process:1234",
        "store_id": "data/agent_supervisor/residual_intelligence_foundry/control.duckdb",
        "store_generation": "vrif-v1",
        "sql": (
            "UPDATE tasks SET status = ?, revision = ? "
            "WHERE task_cid = ? AND revision = ?"
        ),
        "parameters": ["claimed", 2, "task-cid", 1],
    }
    payload["signature"] = quack_owner_mutation_signature(payload, token=token)
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    class _Result:
        description = None
        rowcount = 1

    class _Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def execute(self, sql: str, parameters: object) -> _Result:
            self.calls.append((sql, parameters))
            return _Result()

    class _Server:
        def __init__(self) -> None:
            self._connection = _Connection()

    server = _Server()
    operator._process_mutations(
        server,
        tmp_path,
        token=token,
        expected_store_id=payload["store_id"],
        expected_store_generation=payload["store_generation"],
        seen_request_ids=set(),
    )
    assert len(server._connection.calls) == 1
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result == {"ok": True, "rowcount": 1}


def test_mutation_inbox_rejects_unsigned_and_stale_requests(tmp_path: Path) -> None:
    operator = _operator()
    request_id = "b" * 32
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(
        json.dumps(
            {
                "sql": "UPDATE tasks SET status = 'completed'",
                "parameters": None,
            }
        ),
        encoding="utf-8",
    )

    class _Connection:
        def execute(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("unauthorized SQL must never execute")

    class _Server:
        _connection = _Connection()

    operator._process_mutations(
        _Server(),
        tmp_path,
        token="vrif_test_token_0123456789",
        expected_store_id="store",
        expected_store_generation="generation",
        seen_request_ids=set(),
    )
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result["ok"] is False
    assert result["error"] == "OperatorError: mutation rejected"
