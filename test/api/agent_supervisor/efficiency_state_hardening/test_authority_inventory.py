"""Schema and negative checks for the ASEH authority inventory and ADR."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[4]
INVENTORY_PATH = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/authority_inventory.json"
ADR_PATH = ROOT / "docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md"

EXPECTED_SCOPE = {
    ".gitignore",
    "config/agent_supervisor_efficiency_state_hardening_scheduler.json",
    "docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_PLAN.md",
    "docs/architecture/agent_supervisor/PROGRAMS.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.objectives.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/bootstrap_baseline.json",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/process_security.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "scripts/validate_agent_supervisor_efficiency_state_hardening_board.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
    "test/api/test_agent_supervisor_todo_daemon_port.py",
}
REQUIRED_ENTRY_FIELDS = {
    "path", "entry_points", "capabilities", "owner", "status", "store",
    "bypass", "metrics", "disposition", "evidence",
}
REQUIRED_CAPABILITIES = {
    "mutator", "launcher", "validator", "prover", "patcher", "merger",
    "terminalizer", "receipt_writer", "policy_updater", "fallback", "recovery",
}
VALID_STATUSES = {"available", "available_with_caveats", "non_authoritative_input", "test_only"}
VALID_DISPOSITIONS = {
    "canonical", "compatibility_adapter", "deprecated", "test_fixture",
    "non_authoritative_input",
}


def _load() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_inventory(payload: dict[str, Any]) -> None:
    assert payload["schema"] == "ipfs_accelerate_py/agent-supervisor/aseh-authority-inventory@1"
    assert payload["program_id"] == "agent-supervisor-efficiency-and-state-hardening-v1"
    assert payload["task_id"] == "ASEH-000"
    captured = payload["captured_from"]
    assert captured["repository"] == "ipfs_accelerate_py"
    assert captured["source_identity_kind"] == "captured_committed_tree"
    assert captured["tree"] == _git("rev-parse", f"{captured['commit']}^{{tree}}")

    scope = payload["scope"]
    assert scope["kind"] == "closed_bootstrap_handoff_and_protected_authority_boundary"
    assert set(scope["paths"]) == EXPECTED_SCOPE
    assert len(scope["paths"]) == scope["path_count"] == len(EXPECTED_SCOPE)
    assert all((ROOT / path).is_file() for path in scope["paths"])
    for path in scope["paths"]:
        subprocess.run(
            ["git", "cat-file", "-e", f"{captured['commit']}:{path}"],
            cwd=ROOT,
            check=True,
        )

    vocabulary = payload["vocabulary"]
    assert set(vocabulary["required_entry_fields"]) == REQUIRED_ENTRY_FIELDS
    assert set(vocabulary["statuses"]) == VALID_STATUSES
    assert set(vocabulary["dispositions"]) == VALID_DISPOSITIONS

    entries = payload["entries"]
    assert len(entries) == len(EXPECTED_SCOPE)
    by_path = {entry["path"]: entry for entry in entries}
    assert set(by_path) == EXPECTED_SCOPE
    assert len(by_path) == len(entries), "duplicate path entries hide an unclassified source"
    observed_capabilities: set[str] = set()
    for path, entry in by_path.items():
        assert set(entry) == REQUIRED_ENTRY_FIELDS, path
        assert isinstance(entry["entry_points"], list) and entry["entry_points"], path
        assert isinstance(entry["capabilities"], list) and entry["capabilities"], path
        assert set(entry["capabilities"]) <= REQUIRED_CAPABILITIES, path
        observed_capabilities.update(entry["capabilities"])
        assert isinstance(entry["owner"], str) and entry["owner"], path
        assert entry["status"] in VALID_STATUSES, path
        assert isinstance(entry["store"], str) and entry["store"], path
        assert isinstance(entry["bypass"], str) and entry["bypass"], path
        assert isinstance(entry["metrics"], list) and entry["metrics"], path
        assert entry["disposition"] in VALID_DISPOSITIONS, path
        assert isinstance(entry["evidence"], str) and entry["evidence"], path
    assert REQUIRED_CAPABILITIES <= observed_capabilities

    authorities = payload["canonical_authorities"]
    assert set(authorities) == {
        "supervisor_handoff", "state_machine", "routing", "context_pack", "ducklake",
    }
    for name, text in authorities.items():
        assert isinstance(text, str) and len(text) > 80, name
    assert "TypedStateOwnerGateway@1" in authorities["state_machine"]
    assert "QuackStateServer@1" in authorities["supervisor_handoff"]
    assert "deterministic-first" in authorities["routing"]
    assert "ipfs_datasets_py" in authorities["context_pack"]
    assert "ipfs_kit_py" in authorities["context_pack"]
    assert "non-authoritative" in authorities["ducklake"]

    boundary = payload["cross_repository_boundary"]
    assert set(boundary) == {"ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"}
    for repository, record in boundary.items():
        assert set(record) == {"owns", "must_not_own"}, repository
        assert all(isinstance(value, str) and value for value in record["owns"])
        assert all(isinstance(value, str) and value for value in record["must_not_own"])


def test_inventory_schema_and_exact_current_tree_coverage() -> None:
    _validate_inventory(_load())


def test_inventory_rejects_unclassified_path_and_unknown_status() -> None:
    missing = copy.deepcopy(_load())
    missing["entries"].pop()
    with pytest.raises(AssertionError):
        _validate_inventory(missing)

    invalid = copy.deepcopy(_load())
    invalid["entries"][0]["status"] = "unknown"
    with pytest.raises(AssertionError):
        _validate_inventory(invalid)


def test_adr_binds_all_required_authorities() -> None:
    text = ADR_PATH.read_text(encoding="utf-8")
    required_phrases = (
        "Canonical supervisor handoff",
        "State-machine authority",
        "Routing authority",
        "ContextPack and cross-repository authority",
        "TypedStateOwnerGateway@1",
        "QuackStateServer@1",
        "DatabaseTaskSource@1",
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py",
        "DuckLake",
        "deterministic-first ladder",
        "operator-authorized, CAS-protected",
    )
    assert all(phrase in text for phrase in required_phrases)
