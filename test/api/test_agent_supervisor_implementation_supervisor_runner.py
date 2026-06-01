from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner import (
    CodebaseRefillDefaults,
    ImplementationSupervisorRunContext,
    ImplementationSupervisorDefaults,
    ObjectiveRefillDefaults,
    SupervisorRunHook,
    apply_portal_implementation_supervisor_defaults,
    build_portal_implementation_supervisor_from_args,
    run_portal_implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import parse_args


def test_apply_portal_implementation_supervisor_defaults_preserves_user_values(tmp_path: Path):
    args = apply_portal_implementation_supervisor_defaults(
        ["--state-prefix", "custom", "--objective-scan-max-findings", "99"],
        defaults=ImplementationSupervisorDefaults(
            todo_path=tmp_path / "tasks.todo.md",
            state_dir=tmp_path / "state",
            task_prefix="## EX-",
            state_prefix="example",
            worktree_root=tmp_path / "worktrees",
            daemon_script_path=tmp_path / "daemon.py",
            supervisor_script_path=tmp_path / "supervisor.py",
            llm_merge_resolver_command="codex exec -",
            worktree_submodule_paths=("module-a", "module-b"),
        ),
        objective=ObjectiveRefillDefaults(
            objective_path=tmp_path / "objective.md",
            objective_graph_path=tmp_path / "objective.json",
            objective_bundle_dir=tmp_path / "bundles",
            objective_dataset_dir=tmp_path / "datasets",
            objective_discovery_dir=tmp_path / "discovery",
            objective_discovery_output_path="data/discovery",
            objective_scan_min_open_tasks=3,
            objective_scan_max_findings=7,
            objective_scan_cooldown_seconds=60,
            objective_todo_vector_index_path=tmp_path / "bundles" / "todo_vector_index.json",
            objective_surplus_findings_per_goal=4,
            objective_surplus_min_terms_per_todo=2,
            objective_interoperability_focus=("hallucinate_app",),
            seed_interoperability_goals=True,
        ),
        codebase=CodebaseRefillDefaults(
            codebase_scan_discovery_dir=tmp_path / "codebase",
            codebase_scan_discovery_output_path="data/codebase",
            codebase_scan_min_open_tasks=1,
            codebase_scan_max_findings=5,
            codebase_scan_cooldown_seconds=120,
            codebase_scan_skip_prefixes=("data/state/", "scripts/"),
        ),
    )

    assert args[args.index("--state-prefix") + 1] == "custom"
    assert args[args.index("--objective-scan-max-findings") + 1] == "99"
    assert args.count("--worktree-submodule-path") == 2
    assert args.count("--codebase-scan-skip-prefix") == 2
    assert "--objective-refill-scan" in args
    assert "--objective-seed-interoperability-goals" in args
    assert "--codebase-refill-scan" in args
    parsed = parse_args(args)
    assert parsed.todo_path == tmp_path / "tasks.todo.md"
    assert parsed.state_prefix == "custom"
    assert parsed.objective_scan_max_findings == 99
    assert parsed.codebase_scan_cooldown_seconds == 120


def test_build_portal_implementation_supervisor_from_args_applies_defaults(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    daemon_script = tmp_path / "daemon.py"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon_script.write_text("print('daemon')\n", encoding="utf-8")
    parsed = parse_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "example",
            "--task-prefix",
            "## EX-",
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--once",
        ]
    )

    supervisor, context = build_portal_implementation_supervisor_from_args(
        parsed,
        repo_root=tmp_path,
        daemon_script_path=daemon_script,
        worktree_submodule_paths=("module-a", "module-b"),
    )

    assert supervisor.config.todo_path == board
    assert supervisor.config.repo_root == tmp_path
    assert supervisor.config.daemon_script_path == daemon_script
    assert supervisor.config.worktree_submodule_paths == ("module-a", "module-b")
    assert context.state_path == tmp_path / "state" / "example_task_state.json"
    assert context.strategy_path == tmp_path / "state" / "example_strategy.json"
    assert context.events_path == tmp_path / "state" / "example_supervisor_events.jsonl"
    assert context.daemon_events_path == tmp_path / "state" / "example_events.jsonl"


def test_run_portal_implementation_supervisor_runs_before_and_after_once_hooks(caplog):
    class FakeSupervisor:
        def run_once(self) -> dict[str, int]:
            calls.append("run_once")
            return {"checks": 1}

        def run_forever(self) -> None:
            calls.append("run_forever")

    calls: list[str] = []
    parsed = argparse.Namespace(once=True)
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=Path("state.json"),
        strategy_path=Path("strategy.json"),
        events_path=Path("supervisor-events.jsonl"),
        daemon_events_path=Path("daemon-events.jsonl"),
    )

    def before(ctx: ImplementationSupervisorRunContext) -> list[str]:
        calls.append(f"before:{ctx.state_path}")
        return ["before-result"]

    def after(ctx: ImplementationSupervisorRunContext) -> list[str]:
        calls.append(f"after:{ctx.daemon_events_path}")
        return ["after-result"]

    logger = logging.getLogger("test-supervisor-runner")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = run_portal_implementation_supervisor(
            FakeSupervisor(),
            context,
            logger=logger,
            hooks=(
                SupervisorRunHook("before", "before hook: %s", before),
                SupervisorRunHook("after_once", "after hook: %s", after),
            ),
            once_complete_message="fake supervisor complete: %s",
        )

    assert result == {"checks": 1}
    assert calls == ["before:state.json", "run_once", "after:daemon-events.jsonl"]
    assert "before hook: ['before-result']" in caplog.text
    assert "after hook: ['after-result']" in caplog.text


def test_run_portal_implementation_supervisor_can_delegate_ensure_running():
    parsed = argparse.Namespace(once=False)
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=Path("state.json"),
        strategy_path=Path("strategy.json"),
        events_path=Path("supervisor-events.jsonl"),
        daemon_events_path=Path("daemon-events.jsonl"),
    )

    def ensure(ctx: ImplementationSupervisorRunContext) -> dict[str, str]:
        return {"state": str(ctx.state_path)}

    result = run_portal_implementation_supervisor(
        object(),
        context,
        logger=logging.getLogger("test-supervisor-ensure"),
        ensure_running=True,
        ensure_running_callback=ensure,
    )

    assert result == {"state": "state.json"}
