from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    ConfiguredImplementationDaemonRunner,
    DaemonLoopHook,
    ImplementationDaemonDefaults,
    ImplementationDaemonRunContext,
    apply_portal_implementation_daemon_defaults,
    build_configured_implementation_daemon_runner,
    build_namespace_configured_implementation_daemon_runner,
    build_daemon_codebase_scan_refill_callback,
    build_daemon_objective_refill_callback,
    build_portal_implementation_daemon_from_args,
    build_daemon_refill_hooks,
    build_daemon_refill_hooks_factory_from_recorders,
    build_daemon_refill_hooks_from_recorders,
    build_daemon_retry_budget_refill_callback,
    implementation_state_artifact_paths,
    implementation_state_paths,
    run_configured_portal_implementation_daemon,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_args
from ipfs_accelerate_py.agent_supervisor.wrapper_utils import agent_supervisor_namespace_paths


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
