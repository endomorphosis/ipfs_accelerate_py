"""Board/config consistency tests for the PCPC supervisor program."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / "scripts/validate_agent_supervisor_procedure_compiler_board.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("pcpc_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(report: dict[str, object], name: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return next(item for item in checks if isinstance(item, dict) and item.get("name") == name)


def test_board_validator_accepts_sealed_program() -> None:
    report = _validator_module().validate_program()
    assert report["valid"] is True, json.dumps(report["errors"], indent=2)
    assert report["task_count"] == 32
    assert report["goal_count"] == 5
    assert report["blocked_task_ids"] == []
    assert report["ready_task_ids"] == ["PCPC-009", "PCPC-011", "PCPC-013"]
    assert _check(report, "self_contained_normative_vocabulary")["passed"] is True
    assert _check(report, "task_parallel_lanes")["passed"] is True
    assert _check(report, "concurrency_dependency_safety")["passed"] is True


def test_board_validator_rejects_lane_metadata_drift(tmp_path, monkeypatch) -> None:
    module = _validator_module()
    original = module.TODO_PATH.read_text(encoding="utf-8")
    corrupted = original.replace(
        "- Parallel lane: pcpc-lane-1", "- Parallel lane: pcpc-lane-0", 1
    )
    todo_path = tmp_path / "todo.md"
    todo_path.write_text(corrupted, encoding="utf-8")
    monkeypatch.setattr(module, "TODO_PATH", todo_path)
    monkeypatch.setattr(module, "REPO_ROOT", Path("/"))

    report = module.validate_program()
    check = _check(report, "task_parallel_lanes")
    assert report["valid"] is False
    assert check["passed"] is False
    assert check["detail"]["PCPC-000"] == {
        "expected": "pcpc-lane-1",
        "observed": "pcpc-lane-0",
    }


def test_board_validator_rejects_concurrency_with_transitive_dependent(
    tmp_path, monkeypatch
) -> None:
    module = _validator_module()
    original = module.TODO_PATH.read_text(encoding="utf-8")
    corrupted = original.replace(
        "- Allow concurrent with:\n", "- Allow concurrent with: PCPC-031\n", 1
    )
    todo_path = tmp_path / "todo.md"
    todo_path.write_text(corrupted, encoding="utf-8")
    monkeypatch.setattr(module, "TODO_PATH", todo_path)
    monkeypatch.setattr(module, "REPO_ROOT", Path("/"))

    report = module.validate_program()
    check = _check(report, "concurrency_dependency_safety")
    assert report["valid"] is False
    assert check["passed"] is False
    assert check["detail"] == [
        {
            "task_id": "PCPC-000",
            "peer": "PCPC-031",
            "relation": "peer_depends_on_task",
        }
    ]


def test_ducklake_is_explicitly_non_authoritative() -> None:
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["ducklake_projection_program"]["scheduling_prerequisite"] is False
    assert config["ducklake_projection_program"]["extension_files_sha256"] == {
        "ducklake.duckdb_extension": (
            "d0b57c8e261b89a1ae367c7224f0857cfde72ab6cf2609f188e0de9b897b1088"
        ),
        "ducklake.duckdb_extension.info": (
            "14c3385450437fee5570ff21b53de687536a75b4590e33f351887df194ef9393"
        ),
    }
    assert config["ducklake_projection_program"]["extension_install_policy"] == "forbidden"
    assert config["ducklake_projection_program"]["network_access"] is False


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("authority", True),
        ("scheduling_prerequisite", True),
        ("extension_install_policy", "allowed"),
        ("network_access", True),
        ("unknown_normative_field", False),
    ),
)
def test_board_validator_rejects_unsafe_ducklake_projection_config(
    mutation: str,
    value: object,
) -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    corrupted = deepcopy(config)
    corrupted["ducklake_projection_program"][mutation] = value
    assert module._ducklake_projection_is_valid(
        corrupted["ducklake_projection_program"],
        owner_extension_hashes=corrupted["quack_owner_isolation"][
            "extension_files_sha256"
        ],
    ) is False


def test_board_validator_rejects_ducklake_extension_digest_drift() -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_projection_program"]["extension_files_sha256"][
        "ducklake.duckdb_extension"
    ] = "0" * 64
    assert module._ducklake_projection_is_valid(
        config["ducklake_projection_program"],
        owner_extension_hashes=config["quack_owner_isolation"]["extension_files_sha256"],
    ) is False


@pytest.mark.parametrize(
    ("catalog_path", "data_path"),
    (
        (
            "docs/redteam.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            ".",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
        ),
    ),
)
def test_board_validator_rejects_non_exact_or_overlapping_ducklake_paths(
    catalog_path: str, data_path: str
) -> None:
    module = _validator_module()
    config = json.loads(
        (
            ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_projection_program"]["catalog_path"] = catalog_path
    config["ducklake_projection_program"]["data_path"] = data_path

    assert module._ducklake_projection_is_valid(
        config["ducklake_projection_program"],
        owner_extension_hashes=config["quack_owner_isolation"]["extension_files_sha256"],
    ) is False
