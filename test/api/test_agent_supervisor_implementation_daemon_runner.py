from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    ConfiguredDaemonBootstrapRunner,
    ConfiguredImplementationDaemonRunner,
    DaemonLoopHook,
    ImplementationDaemonDefaults,
    ImplementationDaemonRunContext,
    apply_portal_implementation_daemon_defaults,
    build_configured_daemon_bootstrap_runner,
    build_configured_implementation_daemon_runner,
    build_namespace_daemon_bootstrap_runner,
    build_namespace_configured_implementation_daemon_runner,
    build_daemon_codebase_scan_refill_callback,
    build_daemon_objective_refill_callback,
    build_portal_implementation_daemon_from_args,
    build_daemon_refill_hooks,
    build_daemon_refill_hooks_factory_from_recorders,
    build_daemon_refill_hooks_from_recorders,
    build_daemon_retry_budget_refill_callback,
    implementation_state_artifact_paths,
    namespace_implementation_state_artifact_paths,
    implementation_state_paths,
    run_configured_portal_implementation_daemon,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.checkout_lock import checkout_lock_owner_is_active
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    default_llm_merge_resolver_command,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    parse_args as parse_supervisor_args,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    normalize_validation_command_text,
    split_validation_commands,
)
from ipfs_accelerate_py.agent_supervisor.wrapper_utils import agent_supervisor_namespace_paths


def test_validation_command_helpers_unwrap_markdown_inline_code():
    assert (
        normalize_validation_command_text(
            "`cd swissknife && npm run test:e2e:app-improvement -- --all`."
        )
        == "cd swissknife && npm run test:e2e:app-improvement -- --all"
    )
    assert split_validation_commands(
        "`cd swissknife && npm test`; `cd external/ipfs_datasets && pytest -q`"
    ) == [
        "cd swissknife && npm test",
        "cd external/ipfs_datasets && pytest -q",
    ]


def test_supervisor_propagates_explicit_merge_target_branch(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    target_branch = "automation/virtual-desktop-app-improvement"
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--merge-target-branch",
            target_branch,
        ]
    )

    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    command = PortalImplementationSupervisor(config)._build_daemon_command()

    assert config.merge_target_branch == target_branch
    assert command[command.index("--merge-target-branch") + 1] == target_branch


def test_supervisor_propagates_execution_slice_task_ids_to_daemon(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    parsed = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--execution-slice-task-id",
            "HSSL-BENCH-011",
        ]
    )

    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    command = PortalImplementationSupervisor(config)._build_daemon_command()

    assert config.execution_slice_task_ids == ("HSSL-BENCH-011",)
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--execution-slice-task-id"
    ] == ["HSSL-BENCH-011"]


def test_daemon_execution_slice_cannot_select_an_earlier_ready_bundle_member(
    tmp_path: Path,
):
    board = tmp_path / "tasks.todo.md"
    board.write_text(
        """# HSSL benchmark tasks

## HSSL-BENCH-001 Earlier ready bundle member

- Status: todo
- Completion: manual
- Priority: P1
- Track: protocol

## HSSL-BENCH-011 Leased execution-slice member

- Status: todo
- Completion: manual
- Priority: P1
- Track: protocol
""",
        encoding="utf-8",
    )
    parsed = parse_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-prefix",
            "## HSSL-BENCH-",
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "0",
            "--execution-slice-task-id",
            "HSSL-BENCH-011",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=tmp_path,
    )

    result = daemon.run_once()
    state = PortalTaskState.load(Path(result["state_path"]))

    assert set(daemon.execution_slice_task_ids) == {"HSSL-BENCH-011"}
    assert result["active_task_id"] == "HSSL-BENCH-011"
    assert state.selectable_ready_task_ids == ["HSSL-BENCH-011"]
    assert "HSSL-BENCH-001" not in state.selectable_ready_task_ids


def test_daemon_uses_explicit_merge_target_branch_and_rejects_missing_branch(tmp_path: Path, monkeypatch):
    target_branch = "automation/virtual-desktop-app-improvement"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        merge_target_branch=target_branch,
    )
    monkeypatch.setattr(daemon, "_git_ref_exists", lambda ref: ref == target_branch)
    assert daemon._main_branch_name() == target_branch

    monkeypatch.setattr(daemon, "_git_ref_exists", lambda _ref: False)
    try:
        daemon._main_branch_name()
    except RuntimeError as error:
        assert target_branch in str(error)
    else:
        raise AssertionError("missing configured merge target must not fall back to main")


def test_daemon_uses_packaged_merge_resolver_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", raising=False)
    expected = f"{sys.executable} -m ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )

    assert default_llm_merge_resolver_command() == expected
    assert daemon.llm_merge_resolver_command == expected
    assert parse_args([]).llm_merge_resolver_command == expected


def test_daemon_explicit_merge_resolver_overrides_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", "custom-resolver")
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        llm_merge_resolver_command="explicit-resolver",
    )

    assert default_llm_merge_resolver_command() == "custom-resolver"
    assert daemon.llm_merge_resolver_command == "explicit-resolver"


def test_daemon_resolves_relative_worktree_root_for_runner_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=Path("tmp") / "implementation-worktrees",
    )

    workspace = daemon.worktree_root / "svd-223-attempt-1"
    command = daemon._build_implementation_command(workspace)

    assert daemon.worktree_root == (tmp_path / "tmp" / "implementation-worktrees").resolve()
    assert command[command.index("-C") + 1] == str(workspace.resolve())


def test_daemon_links_shared_dependencies_from_configured_source_root(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "branch-checkout"
    worktree_root = repo_root / "worktrees"
    worktree_path = worktree_root / "svd-174-attempt-5"
    dependency_root = tmp_path / "dependency-checkout"
    dependency_path = dependency_root / "swissknife" / "node_modules"
    dependency_path.mkdir(parents=True)
    worktree_path.mkdir(parents=True)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_SHARED_WORKTREE_SOURCE_ROOT",
        str(dependency_root),
    )

    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo_root,
        worktree_root=worktree_root,
    )

    daemon._link_shared_worktree_paths(worktree_path)

    target = worktree_path / "swissknife" / "node_modules"
    assert target.is_symlink()
    assert target.resolve() == dependency_path.resolve()


def test_secondary_task_shard_skips_repository_wide_git_gc(tmp_path: Path, monkeypatch):
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        task_shard_count=2,
        task_shard_index=1,
    )
    monkeypatch.setattr(daemon, "_cleanup_stale_worktrees", lambda: {})
    monkeypatch.setattr(daemon, "_cleanup_stale_locks", lambda: {})
    monkeypatch.setattr(daemon, "_reset_persistently_dirty_submodules", lambda: {})
    monkeypatch.setattr(daemon.task_queue, "compact", lambda _active_ids: 0)
    monkeypatch.setattr(
        daemon.git_gc,
        "run_if_needed",
        lambda: (_ for _ in ()).throw(AssertionError("secondary shard must not run Git GC")),
    )

    result = daemon._periodic_maintenance()

    assert result["git_gc"] == {"ran": False, "reason": "non_primary_shard"}


def test_periodic_maintenance_obeys_daemon_cooldown(tmp_path: Path, monkeypatch):
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        maintenance_interval_seconds=300,
    )
    calls: list[str] = []
    monkeypatch.setattr(daemon, "_cleanup_stale_worktrees", lambda: calls.append("worktrees") or {})
    monkeypatch.setattr(daemon, "_cleanup_stale_locks", lambda: calls.append("locks") or {})
    monkeypatch.setattr(
        daemon,
        "_reset_persistently_dirty_submodules",
        lambda: calls.append("submodules") or {},
    )
    monkeypatch.setattr(daemon.git_gc, "run_if_needed", lambda: calls.append("gc") or {})
    monkeypatch.setattr(
        daemon.task_queue,
        "compact",
        lambda _active_ids: calls.append("queue") or 0,
    )

    first = daemon._periodic_maintenance()
    second = daemon._periodic_maintenance()

    assert first["ran"] is True
    assert second["ran"] is False
    assert second["reason"] == "cooldown"
    assert calls == ["worktrees", "locks", "submodules", "gc", "queue"]


def test_task_claim_liveness_accepts_module_style_daemon_invocation(tmp_path: Path, monkeypatch):
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.process_is_running",
        lambda _pid: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.process_command_line",
        lambda _pid: "python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    )

    assert daemon._lock_owner_is_active(
        {
            "kind": "implementation_task_claim",
            "pid": 1234,
            "owner_script": "implementation_daemon.py",
        },
        expected_kind="implementation_task_claim",
    )


def test_shared_checkout_lock_liveness_accepts_module_style_invocation(tmp_path: Path):
    assert checkout_lock_owner_is_active(
        {
            "kind": "checkout-mutation",
            "pid": 1234,
            "owner_script": "implementation_daemon.py",
            "repo_root": str(tmp_path),
        },
        expected_kind="checkout-mutation",
        expected_repo_root=tmp_path,
        process_is_running=lambda _pid: True,
        process_command_line=lambda _pid: "python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    )


def test_implementation_state_paths_follow_state_prefix(tmp_path: Path):
    parsed = argparse.Namespace(state_dir=tmp_path / "state", state_prefix="example")

    paths = implementation_state_paths(parsed)

    assert paths["state_path"] == tmp_path / "state" / "example_task_state.json"
    assert paths["strategy_path"] == tmp_path / "state" / "example_strategy.json"
    assert paths["events_path"] == tmp_path / "state" / "example_events.jsonl"

    supervisor_paths = implementation_state_artifact_paths(
        tmp_path / "state",
        "example",
        supervisor_events=True,
    )
    assert supervisor_paths["state_path"] == tmp_path / "state" / "example_task_state.json"
    assert supervisor_paths["strategy_path"] == tmp_path / "state" / "example_strategy.json"
    assert supervisor_paths["events_path"] == tmp_path / "state" / "example_supervisor_events.jsonl"
    assert supervisor_paths["daemon_events_path"] == tmp_path / "state" / "example_events.jsonl"

    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")
    namespace_state_paths = namespace_implementation_state_artifact_paths(namespace_paths)
    assert namespace_state_paths["state_path"] == (
        tmp_path / "data" / "agent_supervisor" / "state" / "agent_supervisor_task_state.json"
    )
    assert namespace_state_paths["strategy_path"] == (
        tmp_path / "data" / "agent_supervisor" / "state" / "agent_supervisor_strategy.json"
    )
    assert namespace_state_paths["events_path"] == (
        tmp_path / "data" / "agent_supervisor" / "state" / "agent_supervisor_events.jsonl"
    )

    overridden_namespace_state_paths = namespace_implementation_state_artifact_paths(
        namespace_paths,
        state_prefix="agent",
        state_dir=tmp_path / "custom-state",
        supervisor_events=True,
    )
    assert overridden_namespace_state_paths["events_path"] == tmp_path / "custom-state" / "agent_supervisor_events.jsonl"
    assert overridden_namespace_state_paths["daemon_events_path"] == tmp_path / "custom-state" / "agent_events.jsonl"


def test_apply_portal_implementation_daemon_defaults_preserves_user_values(tmp_path: Path):
    args = apply_portal_implementation_daemon_defaults(
        [
            "--state-prefix",
            "custom",
            "--worktree-submodule-path",
            "user-module",
            "--llm-merge-resolver-command",
            "user-resolver",
        ],
        defaults=ImplementationDaemonDefaults(
            todo_path=tmp_path / "tasks.todo.md",
            state_dir=tmp_path / "state",
            task_prefix="## EX-",
            state_prefix="example",
            worktree_root=tmp_path / "worktrees",
            objective_path=tmp_path / "objective.md",
            objective_bundle_dir=tmp_path / "bundles",
            llm_merge_resolver_command="default-resolver",
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
    assert parsed.llm_merge_resolver_command == "user-resolver"


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


def test_run_configured_portal_implementation_daemon_builds_and_runs_once(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    calls: list[Path] = []

    def before(ctx: ImplementationDaemonRunContext) -> list[str]:
        calls.append(ctx.parsed.todo_path)
        return []

    run_configured_portal_implementation_daemon(
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
            "--once",
        ],
        repo_root=tmp_path,
        logger=logging.getLogger("test-configured-daemon-runner"),
        hooks=(DaemonLoopHook("before", "before: %s", before),),
    )

    assert calls == [board]


def test_configured_daemon_runner_resolves_bootstrap_paths(monkeypatch, tmp_path: Path):
    calls: list[object] = []
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
        "objective_path": tmp_path / "objective.md",
    }
    hook = DaemonLoopHook("before", "hook: %s", lambda ctx: [])
    captured: dict[str, object] = {}

    def fake_run_configured_from_paths(self, argv, resolved_paths, **kwargs):
        calls.append("run")
        captured["argv"] = tuple(argv)
        captured["paths"] = resolved_paths
        captured["kwargs"] = kwargs
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredImplementationDaemonRunner,
        "run_configured_from_paths",
        fake_run_configured_from_paths,
    )

    def ensure_paths():
        calls.append("paths")
        return paths

    def enter_runtime():
        calls.append("enter")

    def path_callback(resolved_paths):
        calls.append(("path-callback", resolved_paths["todo_path"]))

    def hooks_factory(resolved_paths):
        calls.append(("hooks", resolved_paths["objective_path"]))
        return (hook,)

    runner = build_configured_implementation_daemon_runner(
        repo_root=tmp_path,
        logger=logging.getLogger("test-bootstrap-daemon-runner"),
    )
    result = runner.run_configured_from_bootstrap(
        ["--once"],
        ensure_paths=ensure_paths,
        enter_runtime_environment=enter_runtime,
        path_callbacks=(path_callback,),
        hooks_factory=hooks_factory,
        task_prefix="## EX-",
        state_prefix="example",
        objective_path_key="objective_path",
        llm_merge_resolver_command=lambda: "resolver-command",
        pass_complete_message="complete: %s",
    )

    assert result == {"ran": True}
    assert calls == [
        "paths",
        "enter",
        ("path-callback", paths["todo_path"]),
        ("hooks", paths["objective_path"]),
        "run",
    ]
    assert captured["argv"] == ("--once",)
    assert captured["paths"] == paths
    assert captured["kwargs"]["task_prefix"] == "## EX-"
    assert captured["kwargs"]["state_prefix"] == "example"
    assert captured["kwargs"]["objective_path_key"] == "objective_path"
    assert captured["kwargs"]["llm_merge_resolver_command"] == "resolver-command"
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["pass_complete_message"] == "complete: %s"

    calls.clear()
    runner.run_configured_from_bootstrap(
        [],
        ensure_paths=ensure_paths,
        enter_runtime_environment=enter_runtime,
        enter_runtime_before_paths=True,
        task_prefix="## EX-",
        state_prefix="example",
    )
    assert calls[0:2] == ["enter", "paths"]


def test_namespace_configured_daemon_runner_uses_namespace_defaults(tmp_path: Path):
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")

    runner = build_namespace_configured_implementation_daemon_runner(
        repo_root=tmp_path,
        logger=logging.getLogger("test-namespace-daemon-runner"),
        namespace_paths=namespace_paths,
        default_worktree_submodule_paths=("module-a", "module-b"),
        default_objective_path=tmp_path / "objective.md",
        pass_complete_message="namespace pass complete: %s",
    )

    assert runner.repo_root == tmp_path
    assert runner.default_worktree_submodule_paths == ("module-a", "module-b")
    assert runner.default_objective_path == tmp_path / "objective.md"
    assert runner.default_objective_bundle_dir == namespace_paths.objective_bundle_dir
    assert runner.pass_complete_message == "namespace pass complete: %s"


def test_configured_daemon_bootstrap_runner_dispatches_without_namespace(
    monkeypatch,
    tmp_path: Path,
):
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
    }
    captured: dict[str, object] = {}

    def fake_run_configured_from_bootstrap(self, argv, **kwargs):
        captured["payload"] = {"self": self, "argv": tuple(argv), "kwargs": kwargs}
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredImplementationDaemonRunner,
        "run_configured_from_bootstrap",
        fake_run_configured_from_bootstrap,
    )
    configured_runner = build_configured_implementation_daemon_runner(
        repo_root=tmp_path,
        logger=logging.getLogger("test-configured-bootstrap-daemon"),
    )
    bootstrap = build_configured_daemon_bootstrap_runner(
        runner=configured_runner,
        ensure_paths=lambda: paths,
        task_prefix="## EX-",
        state_prefix="example",
        worktree_submodule_paths=("module-a",),
    )

    assert isinstance(bootstrap, ConfiguredDaemonBootstrapRunner)
    assert bootstrap.run(["--once"]) == {"ran": True}

    payload = captured["payload"]
    assert payload["self"] is configured_runner
    assert payload["argv"] == ("--once",)
    assert payload["kwargs"]["ensure_paths"]() == paths
    assert payload["kwargs"]["task_prefix"] == "## EX-"
    assert payload["kwargs"]["state_prefix"] == "example"
    assert payload["kwargs"]["worktree_submodule_paths"] == ("module-a",)

    monkeypatch.setattr(sys, "argv", ["example-daemon.py", "--status"])
    assert bootstrap.run() == {"ran": True}
    assert captured["payload"]["argv"] == ("--status",)


def test_namespace_daemon_bootstrap_runner_dispatches_namespace_defaults(
    monkeypatch,
    tmp_path: Path,
):
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
        "objective_heap_path": tmp_path / "objective.md",
        "objective_bundle_dir": tmp_path / "runtime-bundles",
    }
    captured: dict[str, object] = {}

    def fake_run_namespace_configured_from_bootstrap(self, argv, **kwargs):
        captured["payload"] = {"self": self, "argv": tuple(argv), "kwargs": kwargs}
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredImplementationDaemonRunner,
        "run_namespace_configured_from_bootstrap",
        fake_run_namespace_configured_from_bootstrap,
    )
    bootstrap = build_namespace_daemon_bootstrap_runner(
        repo_root=tmp_path,
        logger=logging.getLogger("test-namespace-bootstrap-daemon"),
        namespace_paths=namespace_paths,
        ensure_paths=lambda: paths,
        task_prefix="## EX-",
        state_prefix="example",
        default_worktree_submodule_paths=("module-a", "module-b"),
        default_objective_path=tmp_path / "default-objective.md",
        pass_complete_message="namespace pass complete: %s",
        use_bootstrap_keys=True,
        objective_path_key="objective_heap_path",
    )

    assert isinstance(bootstrap, ConfiguredDaemonBootstrapRunner)
    assert bootstrap.runner.default_worktree_submodule_paths == ("module-a", "module-b")
    assert bootstrap.runner.default_objective_path == tmp_path / "default-objective.md"
    assert bootstrap.runner.pass_complete_message == "namespace pass complete: %s"
    assert bootstrap.run(["--once"]) == {"ran": True}

    payload = captured["payload"]
    assert payload["argv"] == ("--once",)
    assert payload["kwargs"]["ensure_paths"]() == paths
    assert payload["kwargs"]["namespace_paths"] is namespace_paths
    assert payload["kwargs"]["use_bootstrap_keys"] is True
    assert payload["kwargs"]["task_prefix"] == "## EX-"
    assert payload["kwargs"]["state_prefix"] == "example"
    assert payload["kwargs"]["objective_path_key"] == "objective_heap_path"
    assert payload["kwargs"]["worktree_submodule_paths"] == ("module-a", "module-b")


def test_namespace_configured_daemon_bootstrap_applies_objective_bundle_defaults(
    monkeypatch,
    tmp_path: Path,
):
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
        "objective_heap_path": tmp_path / "objective.md",
        "objective_bundle_dir": tmp_path / "runtime-bundles",
    }
    captured: list[dict[str, object]] = []

    def fake_run_configured_from_bootstrap(self, argv, **kwargs):
        captured.append({"argv": tuple(argv), "kwargs": kwargs})
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredImplementationDaemonRunner,
        "run_configured_from_bootstrap",
        fake_run_configured_from_bootstrap,
    )

    runner = build_configured_implementation_daemon_runner(
        repo_root=tmp_path,
        logger=logging.getLogger("test-namespace-daemon-bootstrap"),
    )

    assert runner.run_namespace_configured_from_bootstrap(
        ["--once"],
        ensure_paths=lambda: paths,
        namespace_paths=namespace_paths,
        task_prefix="## EX-",
        state_prefix="example",
        objective_path=tmp_path / "static-objective.md",
        worktree_submodule_paths=("module-a",),
    ) == {"ran": True}

    kwargs = captured[-1]["kwargs"]
    assert captured[-1]["argv"] == ("--once",)
    assert kwargs["task_prefix"] == "## EX-"
    assert kwargs["state_prefix"] == "example"
    assert kwargs["objective_path"] == tmp_path / "static-objective.md"
    assert kwargs["objective_bundle_dir"] == namespace_paths.objective_bundle_dir
    assert kwargs["worktree_submodule_paths"] == ("module-a",)

    runner.run_namespace_configured_from_bootstrap(
        [],
        ensure_paths=lambda: paths,
        namespace_paths=namespace_paths,
        task_prefix="## EX-",
        state_prefix="example",
        use_bootstrap_keys=True,
        objective_path_key="objective_heap_path",
    )

    kwargs = captured[-1]["kwargs"]
    assert kwargs["objective_path_key"] == "objective_heap_path"
    assert kwargs["objective_bundle_dir_key"] == "objective_bundle_dir"
    assert kwargs["objective_bundle_dir"] is None


def test_build_daemon_refill_hooks_formats_standard_messages():
    def callback(ctx: ImplementationDaemonRunContext) -> list[str]:
        return [str(ctx.pass_index)]

    hooks = build_daemon_refill_hooks(
        (
            ("objective-goal", callback),
            ("codebase-scan", callback),
            ("retry-budget", callback),
        ),
        scope_label="Hallucinate",
        after_order=("retry-budget", "objective-goal"),
    )

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Hallucinate objective-goal findings before daemon pass: %s"),
        ("before", "Recorded Hallucinate codebase-scan findings before daemon pass: %s"),
        ("before", "Recorded Hallucinate retry-budget findings before daemon pass: %s"),
        ("after", "Recorded Hallucinate retry-budget findings after daemon pass: %s"),
        ("after", "Recorded Hallucinate objective-goal findings after daemon pass: %s"),
        ("after", "Recorded Hallucinate codebase-scan findings after daemon pass: %s"),
    ]


def test_build_daemon_context_refill_callbacks(tmp_path: Path):
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
    )
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    objective_hook = build_daemon_objective_refill_callback(
        recorder("objective"),
        discovery_dir=tmp_path / "discovery",
        objective_path=tmp_path / "objective.md",
        repo_root=tmp_path,
    )
    codebase_hook = build_daemon_codebase_scan_refill_callback(
        recorder("codebase"),
        discovery_dir=tmp_path / "discovery",
        repo_root=tmp_path,
    )
    retry_hook = build_daemon_retry_budget_refill_callback(
        recorder("retry"),
        discovery_dir=tmp_path / "discovery",
    )

    assert objective_hook(context) == ["objective"]
    assert codebase_hook(context) == ["codebase"]
    assert retry_hook(context) == ["retry"]
    assert captured["objective"]["objective_path"] == tmp_path / "objective.md"
    assert captured["objective"]["state_path"] == tmp_path / "state.json"
    assert captured["objective"]["repo_root"] == tmp_path
    assert captured["codebase"]["strategy_path"] == tmp_path / "strategy.json"
    assert captured["retry"]["events_path"] == tmp_path / "events.jsonl"
    assert captured["retry"]["task_header_prefix"] == "## EX-"


def test_build_daemon_refill_hooks_from_recorders(tmp_path: Path):
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
    )
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    hooks = build_daemon_refill_hooks_from_recorders(
        objective_recorder=recorder("objective"),
        codebase_scan_recorder=recorder("codebase"),
        retry_budget_recorder=recorder("retry"),
        discovery_dir=tmp_path / "discovery",
        objective_path=tmp_path / "objective.md",
        repo_root=tmp_path,
        retry_budget_extra_kwargs={"discovery_output_path": "data/discovery"},
        scope_label="Example",
        after_order=("retry-budget", "objective-goal"),
    )

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Example objective-goal findings before daemon pass: %s"),
        ("before", "Recorded Example codebase-scan findings before daemon pass: %s"),
        ("before", "Recorded Example retry-budget findings before daemon pass: %s"),
        ("after", "Recorded Example retry-budget findings after daemon pass: %s"),
        ("after", "Recorded Example objective-goal findings after daemon pass: %s"),
        ("after", "Recorded Example codebase-scan findings after daemon pass: %s"),
    ]
    assert hooks[0].callback(context) == ["objective"]
    assert hooks[1].callback(context) == ["codebase"]
    assert hooks[2].callback(context) == ["retry"]
    assert captured["objective"]["objective_path"] == tmp_path / "objective.md"
    assert captured["objective"]["repo_root"] == tmp_path
    assert captured["codebase"]["repo_root"] == tmp_path
    assert captured["retry"]["discovery_output_path"] == "data/discovery"


def test_build_daemon_refill_hooks_factory_from_recorders(tmp_path: Path):
    paths = {
        "discovery_dir": tmp_path / "discovery",
        "objective_path": tmp_path / "objective.md",
    }
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
    )
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    hooks_factory = build_daemon_refill_hooks_factory_from_recorders(
        objective_recorder=recorder("objective"),
        codebase_scan_recorder=recorder("codebase"),
        retry_budget_recorder=recorder("retry"),
        discovery_dir_key="discovery_dir",
        objective_path_key="objective_path",
        repo_root=tmp_path,
        retry_budget_extra_kwargs_factory=lambda resolved: {
            "discovery_output_path": f"data/{Path(resolved['discovery_dir']).name}",
        },
        scope_label="Example",
        after_order=("retry-budget", "objective-goal"),
    )

    hooks = hooks_factory(paths)

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Example objective-goal findings before daemon pass: %s"),
        ("before", "Recorded Example codebase-scan findings before daemon pass: %s"),
        ("before", "Recorded Example retry-budget findings before daemon pass: %s"),
        ("after", "Recorded Example retry-budget findings after daemon pass: %s"),
        ("after", "Recorded Example objective-goal findings after daemon pass: %s"),
        ("after", "Recorded Example codebase-scan findings after daemon pass: %s"),
    ]
    assert hooks[0].callback(context) == ["objective"]
    assert hooks[1].callback(context) == ["codebase"]
    assert hooks[2].callback(context) == ["retry"]
    assert captured["objective"]["objective_path"] == paths["objective_path"]
    assert captured["objective"]["repo_root"] == tmp_path
    assert captured["codebase"]["discovery_dir"] == paths["discovery_dir"]
    assert captured["retry"]["discovery_output_path"] == "data/discovery"
