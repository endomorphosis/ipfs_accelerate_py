"""Focused regressions for implementation-provider write authority."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    parse_task_file,
)


def test_parsed_predicted_directory_reaches_general_task_edit_policy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    protected_path = "config/operator-owned.json"
    protected = repo / protected_path
    protected.parent.mkdir()
    protected.write_text("{}\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## PCTDD-001 Implement production scope

- Status: ready
- Completion: manual
- Priority: P1
- Track: runtime
- Outputs: test/api/test_production_scope.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon, config/operator-owned.json
- Allowed paths: docs/implementation
- Validation: python -m pytest test/api/test_production_scope.py -q
- Acceptance: Implement production code and its exact regression coverage.
""",
        encoding="utf-8",
    )
    task = parse_task_file(
        todo_path,
        task_header_prefix="## PCTDD-",
    )[0]
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PCTDD-",
        implementation_protected_paths=[protected_path],
    )

    try:
        declared_scope = daemon._proposal_scope_paths_for(
            task,
            repo_root=None,
            include_ast_companions=False,
        )
        assert "ipfs_accelerate_py/agent_supervisor/todo_daemon" in declared_scope
        assert "docs/implementation" in declared_scope
        assert protected_path in declared_scope

        result = daemon._compile_implementation_context(task, attempt=1)
        edit_policy = result.capsule.authority["edit_policy"]
        expected_allowed_paths = (
            "docs/implementation",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon",
            "test/api/test_production_scope.py",
        )
        assert edit_policy["allowed_paths"] == expected_allowed_paths
        assert result.capsule.scope["allowed_edit_paths"] == expected_allowed_paths
        assert protected_path not in edit_policy["allowed_paths"]
        assert edit_policy["protected_paths"] == (protected_path,)

        # Predicted and allowed paths grant write scope, but only exact Outputs
        # remain mandatory artifacts for completion.
        assert result.capsule.scope["expected_outputs"] == (
            "test/api/test_production_scope.py",
        )
        assert daemon._path_matches_scope(
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/worker.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon",
        )
        assert not daemon._path_matches_scope(
            "ipfs_accelerate_py/agent_supervisor/context/worker.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon",
        )
    finally:
        daemon.close_event_runtime()
