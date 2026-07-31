from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    record_reconciliation_guardrail_findings,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)
from scripts.ops.agent_supervisor.validate_ipfs_kit_vfs_symbolic_assurance import (
    OBJECTIVE_PATH,
    TODO_PATH,
    validate,
)


def _vfs_board_before_reconciliation_guardrail(tmp_path):
    repo = tmp_path / "repo"
    todo_path = repo / "docs" / "architecture" / TODO_PATH.name
    todo_path.parent.mkdir(parents=True)
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    todo_text = todo_text.split("\n## VFS-065 ", 1)[0].rstrip() + "\n"
    todo_path.write_text(todo_text, encoding="utf-8")
    return repo, todo_path


def _preflight_conflict_result(count: int = 3):
    return {
        "attempted": True,
        "processed": [
            {
                "branch": f"implementation/vfs-{55 + index:03d}-attempt-1",
                "path": f"/tmp/worktree-{index}",
                "target_ref": "main",
                "preflight_result": {
                    "mergeable": False,
                    "reason": "preflight_merge_conflict",
                    "conflict_paths": ["docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"],
                },
            }
            for index in range(count)
        ],
    }


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


def test_reconciliation_guardrail_inherits_root_vfs_board_profile(tmp_path) -> None:
    repo, todo_path = _vfs_board_before_reconciliation_guardrail(tmp_path)
    external_discovery = tmp_path / "external-state" / "discovery"

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=tmp_path / "state" / "strategy.json",
        discovery_dir=external_discovery,
        reconciliation_result=_preflight_conflict_result(),
        task_prefix="VFS-",
        discovery_output_path=str(external_discovery),
        repo_root=repo,
    )

    assert [item["follow_up_task_id"] for item in findings] == ["VFS-065"]
    task = parse_task_file(todo_path, "## VFS-")[-1]
    assert task.metadata["goal id"] == "VFS-G000"
    assert task.metadata["board namespace"] == "ipfs-kit-vfs-symbolic-assurance-v1"
    assert task.metadata["bundle"] == "vfs-assurance/foundation"
    assert task.metadata["parallel lane"] == "assurance-contracts"
    assert task.metadata["resource class"] == "cpu-small"
    assert task.outputs == ["docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"]
    assert validate(OBJECTIVE_PATH, todo_path)["valid"] is True


def test_reconciliation_guardrail_refresh_repairs_and_reopens_vfs_block(tmp_path) -> None:
    repo, todo_path = _vfs_board_before_reconciliation_guardrail(tmp_path)
    external_discovery = tmp_path / "external-state" / "discovery"
    stale_discovery = external_discovery / "vfs-065.md"
    stale_discovery.parent.mkdir(parents=True)
    stale_discovery.write_text("# stale\n", encoding="utf-8")
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").rstrip()
        + f"""

## VFS-065 Resolve 1 preflight-conflicting backlogged worktree merges

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Dedupe key: reconciliation_guardrail:preflight_merge_conflict
- Depends on:
- Outputs: {external_discovery}, {todo_path}
- Validation: test -f {stale_discovery}
- Acceptance: stale
""",
        encoding="utf-8",
    )

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=tmp_path / "state" / "strategy.json",
        discovery_dir=external_discovery,
        reconciliation_result=_preflight_conflict_result(),
        task_prefix="VFS-",
        discovery_output_path=str(external_discovery),
        repo_root=repo,
    )

    assert len(findings) == 1
    assert findings[0]["refreshed"] is True
    assert findings[0]["reopened"] is True
    task = parse_task_file(todo_path, "## VFS-")[-1]
    assert task.status == "blocked"
    assert task.metadata["goal id"] == "VFS-G000"
    assert task.metadata["board namespace"] == "ipfs-kit-vfs-symbolic-assurance-v1"
    assert task.outputs == ["docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"]
    result = validate(OBJECTIVE_PATH, todo_path)
    assert result["valid"] is True, result["errors"]
