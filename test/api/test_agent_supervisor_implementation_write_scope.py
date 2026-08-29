"""Focused regressions for implementation-provider write authority."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
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


def test_resource_claims_use_the_complete_proposal_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PCTDD-",
        worktree_submodule_paths=("external/ipfs_accelerate",),
    )

    def task(
        task_id: str,
        *,
        output: str,
        predicted_directory: str,
    ) -> PortalTask:
        return PortalTask(
            task_id=task_id,
            title=f"Implement {task_id}",
            status="ready",
            completion="manual",
            priority="P1",
            track="runtime",
            outputs=[output],
            metadata={
                "predicted files": predicted_directory,
                "allowed paths": f"{predicted_directory}/generated",
            },
        )

    shared = "external/ipfs_accelerate/ipfs_accelerate_py/shared_runtime"
    first = task(
        "PCTDD-001",
        output="docs/task-one.md",
        predicted_directory=shared,
    )
    second = task(
        "PCTDD-002",
        output="test/task-two.json",
        predicted_directory=shared,
    )
    alpha = task(
        "PCTDD-003",
        output="docs/task-three.md",
        predicted_directory=(
            "external/ipfs_accelerate/ipfs_accelerate_py/runtime/alpha"
        ),
    )
    beta = task(
        "PCTDD-004",
        output="docs/task-four.md",
        predicted_directory=(
            "external/ipfs_accelerate/ipfs_accelerate_py/runtime/beta"
        ),
    )

    try:
        # Distinct repository-root artifacts do not become shared claims, but
        # the identical predicted production directory does.
        assert daemon._task_implementation_resource_paths(first) == (shared,)
        assert daemon._task_implementation_resource_paths(second) == (shared,)

        # Child Allowed paths collapse beneath their predicted parent, while
        # genuinely disjoint production directories retain distinct claims.
        alpha_claims = daemon._task_implementation_resource_paths(alpha)
        beta_claims = daemon._task_implementation_resource_paths(beta)
        assert alpha_claims == (
            "external/ipfs_accelerate/ipfs_accelerate_py/runtime/alpha",
        )
        assert beta_claims == (
            "external/ipfs_accelerate/ipfs_accelerate_py/runtime/beta",
        )
        assert alpha_claims != beta_claims
    finally:
        daemon.close_event_runtime()
