from __future__ import annotations

from scripts.ops.agent_supervisor.validate_ipfs_kit_vfs_symbolic_assurance import (
    OBJECTIVE_PATH,
    TODO_PATH,
    validate,
)


def test_vfs_rollout_gate_requires_differential_and_control_evidence() -> None:
    result = validate(OBJECTIVE_PATH, TODO_PATH)

    assert result["valid"] is True
    assert result["errors"] == []


def test_vfs_rollout_gate_rejects_missing_audited_dependencies(tmp_path) -> None:
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    todo_text = todo_text.replace(
        "VFS-024, VFS-027, VFS-028, VFS-030, VFS-033, VFS-034",
        "VFS-024, VFS-028, VFS-030, VFS-034",
        1,
    )
    assert todo_text != TODO_PATH.read_text(encoding="utf-8")
    tampered_todo_path = tmp_path / TODO_PATH.name
    tampered_todo_path.write_text(todo_text, encoding="utf-8")

    result = validate(OBJECTIVE_PATH, tampered_todo_path)

    assert result["valid"] is False
    assert (
        "VFS-036 missing required dependencies: ['VFS-027', 'VFS-033']"
        in result["errors"]
    )
