from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner import (
    CodebaseRefillDefaults,
    ConfiguredSupervisorBootstrapRunner,
    ConfiguredSupervisorRuntime,
    ImplementationSupervisorRunContext,
    ImplementationSupervisorDefaults,
    ObjectiveRefillDefaults,
    SupervisorRunHook,
    apply_portal_implementation_supervisor_defaults,
    build_codebase_refill_defaults_factory,
    build_configured_supervisor_bootstrap_runner,
    build_configured_supervisor_runtime,
    build_objective_refill_defaults_factory,
    build_portal_implementation_supervisor_from_args,
    build_script_supervisor_bootstrap_runner,
    build_script_supervisor_runtime,
    build_supervisor_codebase_scan_refill_callback,
    build_supervisor_objective_refill_callback,
    build_supervisor_refill_hooks,
    build_supervisor_refill_hooks_factory_from_recorders,
    build_supervisor_refill_hooks_from_recorders,
    build_supervisor_retry_budget_refill_callback,
    build_supervisor_runtime_callbacks,
    run_configured_portal_implementation_supervisor,
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


def test_build_supervisor_refill_default_factories_resolve_bootstrap_paths(tmp_path: Path):
    paths = {
        "objective_path": tmp_path / "objective.md",
        "objective_graph_path": tmp_path / "objective.json",
        "bundle_dir": tmp_path / "bundles",
        "dataset_dir": tmp_path / "datasets",
        "discovery_dir": tmp_path / "discovery",
        "todo_vector_index_path": tmp_path / "bundles" / "todo_vector_index.json",
    }

    objective_factory = build_objective_refill_defaults_factory(
        objective_path_key="objective_path",
        objective_graph_path_key="objective_graph_path",
        objective_bundle_dir_key="bundle_dir",
        objective_dataset_dir_key="dataset_dir",
        objective_discovery_dir_key="discovery_dir",
        objective_discovery_output_path_factory=lambda resolved: f"out/{Path(resolved['discovery_dir']).name}",
        objective_todo_vector_index_path_key="todo_vector_index_path",
        objective_interoperability_focus=("hallucinate_app",),
        objective_scan_max_findings=11,
        seed_interoperability_goals=True,
    )
    codebase_factory = build_codebase_refill_defaults_factory(
        codebase_scan_discovery_dir_key="discovery_dir",
        codebase_scan_discovery_output_path_factory=lambda resolved: f"scan/{Path(resolved['discovery_dir']).name}",
        codebase_scan_skip_prefixes=("data/state/",),
    )

    objective = objective_factory(paths)
    codebase = codebase_factory(paths)

    assert objective.objective_path == paths["objective_path"]
    assert objective.objective_graph_path == paths["objective_graph_path"]
    assert objective.objective_bundle_dir == paths["bundle_dir"]
    assert objective.objective_dataset_dir == paths["dataset_dir"]
    assert objective.objective_discovery_output_path == "out/discovery"
    assert objective.objective_todo_vector_index_path == paths["todo_vector_index_path"]
    assert objective.objective_interoperability_focus == ("hallucinate_app",)
    assert objective.objective_scan_max_findings == 11
    assert objective.seed_interoperability_goals is True
    assert codebase.codebase_scan_discovery_dir == paths["discovery_dir"]
    assert codebase.codebase_scan_discovery_output_path == "scan/discovery"
    assert codebase.codebase_scan_skip_prefixes == ("data/state/",)


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


def test_run_configured_portal_implementation_supervisor_builds_and_runs_once(tmp_path: Path):
    board = tmp_path / "tasks.todo.md"
    daemon_script = tmp_path / "daemon.py"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon_script.write_text("print('daemon')\n", encoding="utf-8")
    calls: list[Path] = []

    def before(ctx: ImplementationSupervisorRunContext) -> list[str]:
        calls.append(ctx.parsed.todo_path)
        return []

    result = run_configured_portal_implementation_supervisor(
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
            "--daemon-script-path",
            str(daemon_script),
            "--once",
        ],
        repo_root=tmp_path,
        logger=logging.getLogger("test-configured-supervisor-runner"),
        hooks=(SupervisorRunHook("before", "before: %s", before),),
    )

    assert calls == [board]
    assert isinstance(result, dict)


def test_configured_supervisor_runtime_resolves_bootstrap_paths(monkeypatch, tmp_path: Path):
    calls: list[object] = []
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
        "objective_path": tmp_path / "objective.md",
    }
    daemon_script = tmp_path / "daemon.py"
    supervisor_script = tmp_path / "supervisor.py"
    hook = SupervisorRunHook("before", "hook: %s", lambda ctx: [])
    objective = ObjectiveRefillDefaults(objective_path=paths["objective_path"])
    codebase = CodebaseRefillDefaults(codebase_scan_discovery_dir=tmp_path / "discovery")
    captured: dict[str, object] = {}

    def fake_run_configured_from_paths(self, argv, resolved_paths, **kwargs):
        calls.append("run")
        captured["argv"] = tuple(argv)
        captured["paths"] = resolved_paths
        captured["kwargs"] = kwargs
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredSupervisorRuntime,
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

    def objective_factory(resolved_paths):
        calls.append(("objective", resolved_paths["objective_path"]))
        return objective

    def codebase_factory(resolved_paths):
        calls.append(("codebase", resolved_paths["state_dir"]))
        return codebase

    def hooks_factory(resolved_paths):
        calls.append(("hooks", resolved_paths["worktree_root"]))
        return (hook,)

    runtime = build_configured_supervisor_runtime(
        repo_root=tmp_path,
        script_path=supervisor_script,
    )
    result = runtime.run_configured_from_bootstrap(
        ["--ensure-running", "--once"],
        logger=logging.getLogger("test-bootstrap-supervisor-runtime"),
        ensure_paths=ensure_paths,
        enter_runtime_environment=enter_runtime,
        path_callbacks=(path_callback,),
        objective_factory=objective_factory,
        codebase_factory=codebase_factory,
        hooks_factory=hooks_factory,
        task_prefix="## EX-",
        state_prefix="example",
        daemon_script_path=daemon_script,
        supervisor_script_path=supervisor_script,
        once_complete_message="complete: %s",
        ensure_running_message="ensure: %s",
    )

    assert result == {"ran": True}
    assert calls == [
        "paths",
        "enter",
        ("path-callback", paths["todo_path"]),
        ("objective", paths["objective_path"]),
        ("codebase", paths["state_dir"]),
        ("hooks", paths["worktree_root"]),
        "run",
    ]
    assert captured["argv"] == ("--once",)
    assert captured["paths"] == paths
    assert captured["kwargs"]["task_prefix"] == "## EX-"
    assert captured["kwargs"]["state_prefix"] == "example"
    assert captured["kwargs"]["daemon_script_path"] == daemon_script
    assert captured["kwargs"]["supervisor_script_path"] == supervisor_script
    assert captured["kwargs"]["objective"] == objective
    assert captured["kwargs"]["codebase"] == codebase
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["ensure_running"] is True
    assert captured["kwargs"]["once_complete_message"] == "complete: %s"
    assert captured["kwargs"]["ensure_running_message"] == "ensure: %s"

    calls.clear()
    runtime.run_configured_from_bootstrap(
        [],
        logger=logging.getLogger("test-bootstrap-supervisor-runtime"),
        ensure_paths=ensure_paths,
        enter_runtime_environment=enter_runtime,
        enter_runtime_before_paths=True,
        task_prefix="## EX-",
        state_prefix="example",
        daemon_script_path=daemon_script,
    )
    assert calls[0:2] == ["enter", "paths"]


def test_configured_supervisor_bootstrap_runner_dispatches_runtime(monkeypatch, tmp_path: Path):
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
    }
    daemon_script = tmp_path / "daemon.py"
    supervisor_script = tmp_path / "supervisor.py"
    objective = ObjectiveRefillDefaults(objective_path=tmp_path / "objective.md")
    codebase = CodebaseRefillDefaults(codebase_scan_discovery_dir=tmp_path / "discovery")
    hook = SupervisorRunHook("before", "hook: %s", lambda ctx: [])
    captured: dict[str, object] = {}
    calls: list[str] = []

    def fake_run_configured_from_bootstrap(self, argv, **kwargs):
        captured["runtime"] = self
        captured["argv"] = tuple(argv)
        captured["kwargs"] = kwargs
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredSupervisorRuntime,
        "run_configured_from_bootstrap",
        fake_run_configured_from_bootstrap,
    )

    runtime = build_configured_supervisor_runtime(
        repo_root=tmp_path,
        script_path=supervisor_script,
    )
    runner = build_configured_supervisor_bootstrap_runner(
        runtime=runtime,
        logger=logging.getLogger("test-configured-supervisor-bootstrap-runner"),
        ensure_paths=lambda: paths,
        enter_runtime_environment=lambda: calls.append("enter"),
        path_callbacks=(lambda resolved: calls.append(str(resolved["todo_path"])),),
        objective_factory=lambda _paths: objective,
        codebase_factory=lambda _paths: codebase,
        hooks_factory=lambda _paths: (hook,),
        task_prefix="## EX-",
        state_prefix="example",
        daemon_script_path=daemon_script,
        supervisor_script_path=supervisor_script,
        todo_path_flag="--task-board",
        llm_merge_resolver_command=lambda: "codex exec resolver",
        worktree_submodule_paths=("module-a",),
        once_complete_message="complete: %s",
        ensure_running_message="ensure: %s",
        repair_runtime_message="repair: %s",
    )

    assert runner.run(["--once"]) == {"ran": True}
    assert captured["runtime"] is runtime
    assert captured["argv"] == ("--once",)
    kwargs = captured["kwargs"]
    assert kwargs["logger"] is runner.logger
    assert kwargs["ensure_paths"]() == paths
    assert kwargs["task_prefix"] == "## EX-"
    assert kwargs["state_prefix"] == "example"
    assert kwargs["daemon_script_path"] == daemon_script
    assert kwargs["supervisor_script_path"] == supervisor_script
    assert kwargs["todo_path_flag"] == "--task-board"
    assert kwargs["llm_merge_resolver_command"] == "codex exec resolver"
    assert kwargs["worktree_submodule_paths"] == ("module-a",)
    assert kwargs["objective_factory"](paths) == objective
    assert kwargs["codebase_factory"](paths) == codebase
    assert kwargs["hooks_factory"](paths) == (hook,)
    assert kwargs["once_complete_message"] == "complete: %s"
    assert kwargs["ensure_running_message"] == "ensure: %s"
    assert kwargs["repair_runtime_message"] == "repair: %s"
    kwargs["enter_runtime_environment"]()
    kwargs["path_callbacks"][0](paths)
    assert calls == ["enter", str(paths["todo_path"])]


def test_build_script_supervisor_bootstrap_runner_builds_runtime_and_dispatches(
    monkeypatch,
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    script_path = repo / "scripts" / "example_supervisor.py"
    daemon_script = repo / "scripts" / "example_daemon.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# supervisor\n", encoding="utf-8")
    daemon_script.write_text("# daemon\n", encoding="utf-8")
    paths = {
        "todo_path": tmp_path / "tasks.todo.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
    }
    captured: dict[str, object] = {}

    def fake_run_configured_from_bootstrap(self, argv, **kwargs):
        captured["runtime"] = self
        captured["argv"] = tuple(argv)
        captured["kwargs"] = kwargs
        return {"ran": True}

    monkeypatch.setattr(
        ConfiguredSupervisorRuntime,
        "run_configured_from_bootstrap",
        fake_run_configured_from_bootstrap,
    )
    runner = build_script_supervisor_bootstrap_runner(
        repo_root=repo,
        script_path=script_path,
        logger=logging.getLogger("test-script-supervisor-bootstrap-runner"),
        ensure_paths=lambda: paths,
        prepare_environment=lambda: None,
        extra_process_match_any=("example_autopilot.py",),
        task_prefix="## EX-",
        state_prefix="example",
        daemon_script_path=daemon_script,
        worktree_submodule_paths=("module-a",),
    )

    assert isinstance(runner, ConfiguredSupervisorBootstrapRunner)
    assert runner.runtime.process_match_any == ("example_supervisor.py", "example_autopilot.py")
    assert runner.run(["--once"]) == {"ran": True}
    assert captured["runtime"] is runner.runtime
    assert captured["argv"] == ("--once",)
    assert captured["kwargs"]["ensure_paths"]() == paths
    assert captured["kwargs"]["task_prefix"] == "## EX-"
    assert captured["kwargs"]["state_prefix"] == "example"
    assert captured["kwargs"]["daemon_script_path"] == daemon_script
    assert captured["kwargs"]["worktree_submodule_paths"] == ("module-a",)


def test_build_script_supervisor_runtime_derives_script_marker(tmp_path: Path):
    repo = tmp_path / "repo"
    script_path = repo / "scripts" / "example_supervisor.py"
    repo.mkdir()
    script_path.parent.mkdir()
    script_path.write_text("# supervisor\n", encoding="utf-8")

    runtime = build_script_supervisor_runtime(
        repo_root=repo,
        script_path=script_path,
        extra_process_match_any=("example_autopilot.py",),
        startup_delay_seconds=0.5,
    )

    assert runtime.repo_root == repo
    assert runtime.script_path == script_path.resolve()
    assert runtime.process_match_any == ("example_supervisor.py", "example_autopilot.py")
    assert runtime.startup_delay_seconds == 0.5


def test_build_supervisor_refill_hooks_formats_standard_messages():
    def callback(ctx: ImplementationSupervisorRunContext) -> list[str]:
        return [str(ctx.state_path)]

    hooks = build_supervisor_refill_hooks(
        (
            ("objective-goal", callback),
            ("codebase-scan", callback),
            ("retry-budget", callback),
        ),
        scope_label="Hallucinate",
    )

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Hallucinate objective-goal findings before supervisor pass: %s"),
        ("before", "Recorded Hallucinate codebase-scan findings before supervisor pass: %s"),
        ("before", "Recorded Hallucinate retry-budget findings before supervisor pass: %s"),
        ("after_once", "Recorded Hallucinate objective-goal findings after supervisor pass: %s"),
        ("after_once", "Recorded Hallucinate codebase-scan findings after supervisor pass: %s"),
        ("after_once", "Recorded Hallucinate retry-budget findings after supervisor pass: %s"),
    ]


def test_build_supervisor_context_refill_callbacks(tmp_path: Path):
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
        objective_bundle_dir=tmp_path / "bundles",
        objective_dataset_dir=tmp_path / "datasets",
        objective_todo_vector_index_path=tmp_path / "bundles" / "todo_vector_index.json",
        objective_scan_min_open_tasks=3,
        objective_scan_max_findings=7,
        objective_scan_cooldown_seconds=60,
        objective_surplus_findings_per_goal=4,
        objective_surplus_min_terms_per_todo=2,
        codebase_scan_min_open_tasks=1,
        codebase_scan_max_findings=5,
        codebase_scan_cooldown_seconds=120,
    )
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "supervisor-events.jsonl",
        daemon_events_path=tmp_path / "daemon-events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    objective_hook = build_supervisor_objective_refill_callback(
        recorder("objective"),
        discovery_dir=tmp_path / "discovery",
        objective_path=tmp_path / "objective.md",
        repo_root=tmp_path,
    )
    codebase_hook = build_supervisor_codebase_scan_refill_callback(
        recorder("codebase"),
        discovery_dir=tmp_path / "discovery",
        repo_root=tmp_path,
    )
    retry_hook = build_supervisor_retry_budget_refill_callback(
        recorder("retry"),
        discovery_dir=tmp_path / "discovery",
    )

    assert objective_hook(context) == ["objective"]
    assert codebase_hook(context) == ["codebase"]
    assert retry_hook(context) == ["retry"]
    assert captured["objective"]["objective_path"] == tmp_path / "objective.md"
    assert captured["objective"]["bundle_dir"] == tmp_path / "bundles"
    assert captured["objective"]["surplus_findings_per_goal"] == 4
    assert captured["codebase"]["max_findings"] == 5
    assert captured["retry"]["events_path"] == tmp_path / "daemon-events.jsonl"
    assert captured["retry"]["task_header_prefix"] == "## EX-"


def test_build_supervisor_refill_hooks_from_recorders(tmp_path: Path):
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
        objective_bundle_dir=tmp_path / "bundles",
        objective_dataset_dir=tmp_path / "datasets",
        objective_todo_vector_index_path=tmp_path / "bundles" / "todo_vector_index.json",
        objective_scan_min_open_tasks=3,
        objective_scan_max_findings=7,
        objective_scan_cooldown_seconds=60,
        objective_surplus_findings_per_goal=4,
        objective_surplus_min_terms_per_todo=2,
        codebase_scan_min_open_tasks=1,
        codebase_scan_max_findings=5,
        codebase_scan_cooldown_seconds=120,
    )
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "supervisor-events.jsonl",
        daemon_events_path=tmp_path / "daemon-events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    hooks = build_supervisor_refill_hooks_from_recorders(
        objective_recorder=recorder("objective"),
        codebase_scan_recorder=recorder("codebase"),
        retry_budget_recorder=recorder("retry"),
        discovery_dir=tmp_path / "discovery",
        objective_path=tmp_path / "objective.md",
        repo_root=tmp_path,
        retry_budget_extra_kwargs={"discovery_output_path": "data/discovery"},
        scope_label="Example",
        after_once_order=("retry-budget", "objective-goal"),
    )

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Example objective-goal findings before supervisor pass: %s"),
        ("before", "Recorded Example codebase-scan findings before supervisor pass: %s"),
        ("before", "Recorded Example retry-budget findings before supervisor pass: %s"),
        ("after_once", "Recorded Example retry-budget findings after supervisor pass: %s"),
        ("after_once", "Recorded Example objective-goal findings after supervisor pass: %s"),
        ("after_once", "Recorded Example codebase-scan findings after supervisor pass: %s"),
    ]
    assert hooks[0].callback(context) == ["objective"]
    assert hooks[1].callback(context) == ["codebase"]
    assert hooks[2].callback(context) == ["retry"]
    assert captured["objective"]["objective_path"] == tmp_path / "objective.md"
    assert captured["objective"]["bundle_dir"] == tmp_path / "bundles"
    assert captured["codebase"]["max_findings"] == 5
    assert captured["codebase"]["repo_root"] == tmp_path
    assert captured["retry"]["discovery_output_path"] == "data/discovery"


def test_build_supervisor_refill_hooks_factory_from_recorders(tmp_path: Path):
    paths = {
        "discovery_dir": tmp_path / "discovery",
        "objective_path": tmp_path / "objective.md",
    }
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
        objective_bundle_dir=tmp_path / "bundles",
        objective_dataset_dir=tmp_path / "datasets",
        objective_todo_vector_index_path=tmp_path / "bundles" / "todo_vector_index.json",
        objective_scan_min_open_tasks=3,
        objective_scan_max_findings=7,
        objective_scan_cooldown_seconds=60,
        objective_surplus_findings_per_goal=4,
        objective_surplus_min_terms_per_todo=2,
        codebase_scan_min_open_tasks=1,
        codebase_scan_max_findings=5,
        codebase_scan_cooldown_seconds=120,
    )
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "supervisor-events.jsonl",
        daemon_events_path=tmp_path / "daemon-events.jsonl",
    )
    captured: dict[str, dict[str, object]] = {}

    def recorder(label: str):
        def callback(**kwargs: object) -> list[str]:
            captured[label] = kwargs
            return [label]

        return callback

    hooks_factory = build_supervisor_refill_hooks_factory_from_recorders(
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
        after_once_order=("retry-budget", "objective-goal"),
    )

    hooks = hooks_factory(paths)

    assert [(hook.phase, hook.message) for hook in hooks] == [
        ("before", "Recorded Example objective-goal findings before supervisor pass: %s"),
        ("before", "Recorded Example codebase-scan findings before supervisor pass: %s"),
        ("before", "Recorded Example retry-budget findings before supervisor pass: %s"),
        ("after_once", "Recorded Example retry-budget findings after supervisor pass: %s"),
        ("after_once", "Recorded Example objective-goal findings after supervisor pass: %s"),
        ("after_once", "Recorded Example codebase-scan findings after supervisor pass: %s"),
    ]
    assert hooks[0].callback(context) == ["objective"]
    assert hooks[1].callback(context) == ["codebase"]
    assert hooks[2].callback(context) == ["retry"]
    assert captured["objective"]["objective_path"] == paths["objective_path"]
    assert captured["codebase"]["discovery_dir"] == paths["discovery_dir"]
    assert captured["retry"]["discovery_output_path"] == "data/discovery"


def test_build_supervisor_runtime_callbacks_repairs_stale_markers(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    prefix = "example"
    stale_pid = 99999999
    managed_pid = state_dir / f"{prefix}_managed_daemon.pid"
    wrapper_pid = state_dir / f"{prefix}_supervisor_wrapper.pid"
    status_path = state_dir / f"{prefix}_supervisor_status.json"
    lock_path = state_dir / "implementation.lock"
    managed_pid.write_text(f"{stale_pid}\n", encoding="utf-8")
    wrapper_pid.write_text(f"{stale_pid}\n", encoding="utf-8")
    lock_path.write_text('{"pid": 99999999}\n', encoding="utf-8")
    status_path.write_text('{"status": "running", "supervisor_pid": 99999999}\n', encoding="utf-8")
    parsed = argparse.Namespace(state_dir=state_dir, state_prefix=prefix)
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=object(),
        state_path=state_dir / f"{prefix}_task_state.json",
        strategy_path=state_dir / f"{prefix}_strategy.json",
        events_path=state_dir / f"{prefix}_supervisor_events.jsonl",
        daemon_events_path=state_dir / f"{prefix}_events.jsonl",
    )

    callbacks = build_supervisor_runtime_callbacks(
        ["--once"],
        repo_root=tmp_path,
        script_path=tmp_path / "supervisor.py",
        process_match_any=("supervisor.py",),
    )
    repairs = callbacks.repair_runtime(context)

    assert str(managed_pid) in repairs["removed"]
    assert str(wrapper_pid) in repairs["removed"]
    assert str(lock_path) in repairs["removed"]
    assert repairs["updated_status"] is True
