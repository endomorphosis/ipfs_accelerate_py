from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    DaemonLoopHook,
    ImplementationDaemonDefaults,
    ImplementationDaemonRunContext,
    apply_portal_implementation_daemon_defaults,
    build_portal_implementation_daemon_from_args,
    implementation_state_paths,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_args


def test_implementation_state_paths_follow_state_prefix(tmp_path: Path):
    parsed = argparse.Namespace(state_dir=tmp_path / "state", state_prefix="example")

    paths = implementation_state_paths(parsed)

    assert paths["state_path"] == tmp_path / "state" / "example_task_state.json"
    assert paths["strategy_path"] == tmp_path / "state" / "example_strategy.json"
    assert paths["events_path"] == tmp_path / "state" / "example_events.jsonl"


def test_apply_portal_implementation_daemon_defaults_preserves_user_values(tmp_path: Path):
    args = apply_portal_implementation_daemon_defaults(
        ["--state-prefix", "custom", "--worktree-submodule-path", "user-module"],
        defaults=ImplementationDaemonDefaults(
            todo_path=tmp_path / "tasks.todo.md",
            state_dir=tmp_path / "state",
            task_prefix="## EX-",
            state_prefix="example",
            worktree_root=tmp_path / "worktrees",
            objective_path=tmp_path / "objective.md",
            objective_bundle_dir=tmp_path / "bundles",
            worktree_submodule_paths=("module-a", "module-b"),
        ),
    )

    assert args[args.index("--state-prefix") + 1] == "custom"
    assert args.count("--worktree-submodule-path") == 1
    parsed = parse_args(args)
    assert parsed.todo_path == tmp_path / "tasks.todo.md"
    assert parsed.state_dir == tmp_path / "state"
    assert parsed.task_prefix == "## EX-"
    assert parsed.state_prefix == "custom"
    assert parsed.worktree_root == tmp_path / "worktrees"
    assert parsed.objective_path == tmp_path / "objective.md"
    assert parsed.objective_bundle_dir == tmp_path / "bundles"
    assert parsed.worktree_submodule_path == ["user-module"]


def test_build_portal_implementation_daemon_from_args_applies_defaults(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = parse_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-prefix",
            "## EX-",
            "--state-prefix",
            "example",
            "--worktree-root",
            str(tmp_path / "worktrees"),
        ]
    )

    daemon, context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=tmp_path,
        default_worktree_submodule_paths=("module-a", "module-b"),
        default_objective_path=tmp_path / "objective.md",
        default_objective_bundle_dir=tmp_path / "bundles",
    )

    assert daemon.todo_path == board
    assert daemon.repo_root == tmp_path
    assert daemon.task_header_prefix == "## EX-"
    assert daemon.worktree_submodule_paths == ("module-a", "module-b")
    assert daemon.objective_path == tmp_path / "objective.md"
    assert daemon.objective_bundle_dir == tmp_path / "bundles"
    assert context.state_path == tmp_path / "state" / "example_task_state.json"
    assert context.strategy_path == tmp_path / "state" / "example_strategy.json"
    assert context.events_path == tmp_path / "state" / "example_events.jsonl"


def test_run_portal_implementation_daemon_loop_runs_hooks_once(caplog):
    class FakeDaemon:
        def __init__(self) -> None:
            self.count = 0

        def run_once(self) -> dict[str, int]:
            self.count += 1
            return {"count": self.count}

    parsed = argparse.Namespace(once=True, interval=999)
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=Path("state.json"),
        strategy_path=Path("strategy.json"),
        events_path=Path("events.jsonl"),
    )
    calls: list[str] = []

    def before(ctx: ImplementationDaemonRunContext) -> list[str]:
        calls.append(f"before:{ctx.pass_index}")
        return ["before-result"]

    def after(ctx: ImplementationDaemonRunContext) -> list[str]:
        calls.append(f"after:{ctx.pass_index}")
        return ["after-result"]

    logger = logging.getLogger("test-daemon-runner")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        run_portal_implementation_daemon_loop(
            FakeDaemon(),
            context,
            logger=logger,
            hooks=(
                DaemonLoopHook("before", "before hook: %s", before),
                DaemonLoopHook("after", "after hook: %s", after),
            ),
            pass_complete_message="fake pass complete: %s",
        )

    assert calls == ["before:0", "after:0"]
    assert "before hook: ['before-result']" in caplog.text
    assert "after hook: ['after-result']" in caplog.text
