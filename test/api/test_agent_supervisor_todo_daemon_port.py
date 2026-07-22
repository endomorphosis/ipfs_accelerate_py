from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    build_arg_parser,
    discovery_fingerprints,
    run_objective_daemon,
)
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    build_arg_parser as build_bundle_arg_parser,
    plan_bundle_lanes,
    run_bundle_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveGoal,
    objective_heap_schedule,
    parse_goal_heap,
    scan_objective_gaps,
)
from ipfs_accelerate_py.agent_supervisor.todo_vector_index import parse_todo_vector_records, write_todo_vector_index
from ipfs_accelerate_py.agent_supervisor.objective_tracker import fibonacci_priority, run_goal_validation
from ipfs_accelerate_py.agent_supervisor.validation_commands import split_validation_commands
from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    reconciliation_guardrail_plan,
    reconciliation_guardrail_records,
    record_reconciliation_guardrail_findings,
)
from ipfs_accelerate_py.agent_supervisor import merge_resolver
from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    ConfiguredMergeResolverRunner,
    MergeResolverNamespaceSpec,
    build_configured_merge_resolver_runner,
    build_namespace_merge_resolver_runner_from_spec,
)
from ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback import (
    llm_merge_resolver_fallback_command,
)
from ipfs_accelerate_py.agent_supervisor import task_proposal_router
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    ConfiguredTaskProposalRouterRunner,
    TaskProposalRouteSpec,
    build_configured_task_proposal_router_runner,
    build_repo_task_proposal_route_runner,
    build_repo_task_proposal_route_runner_from_spec,
    build_repo_task_proposal_router_runner,
    run_configured_task_proposal_router_cli,
    standard_task_proposal_requested_outputs,
)
from ipfs_accelerate_py.agent_supervisor import multi_supervisor_runner
from ipfs_accelerate_py.agent_supervisor.multi_supervisor_runner import (
    ConfiguredMultiSupervisorCliRunner,
    ConfiguredMultiSupervisorLauncher,
    ImplementationSupervisorNamespaceTrackSpec,
    ImplementationSupervisorTrackConfig,
    build_arg_parser as build_multi_supervisor_arg_parser,
    build_configured_multi_supervisor_launcher,
    build_configured_multi_supervisor_cli_runner,
    build_repo_implementation_multi_supervisor_launcher,
    common_args_from_parsed_args,
    implementation_multi_supervisor_env_defaults,
    implementation_supervisor_compact_track_spec,
    implementation_supervisor_compact_track_specs,
    implementation_supervisor_common_args,
    implementation_supervisor_namespace_track_config,
    implementation_supervisor_namespace_track_configs,
    implementation_supervisor_track_spec,
    parse_implementation_track_spec,
    parse_track_spec,
    run_supervisor_tracks,
    supervisor_track_payload,
    tracks_from_parsed_args,
)
from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    ConfiguredImplementationDaemonRunner,
    ImplementationDaemonDefaults,
    ImplementationDaemonRunContext,
    apply_portal_implementation_daemon_defaults,
    apply_portal_implementation_daemon_defaults_from_paths,
    build_daemon_codebase_scan_refill_callback,
    build_daemon_objective_refill_callback,
    build_configured_implementation_daemon_runner,
    build_implementation_daemon_defaults_from_paths,
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor import implementation_daemon_runner
from ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner import (
    CodebaseRefillDefaults,
    ImplementationSupervisorDefaults,
    ObjectiveRefillDefaults,
    apply_portal_implementation_supervisor_defaults,
    apply_portal_implementation_supervisor_defaults_from_paths,
    build_codebase_refill_defaults_from_paths,
    build_configured_supervisor_runtime,
    build_implementation_supervisor_defaults_from_paths,
    build_namespace_codebase_refill_defaults_factory,
    build_namespace_objective_refill_defaults_factory,
    build_objective_refill_defaults_from_paths,
)
from ipfs_accelerate_py.agent_supervisor import implementation_supervisor_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as implementation_daemon_module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoTaskState,
    TodoImplementationDaemon,
    parse_task_file,
    parse_args as parse_implementation_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
    parse_args as parse_implementation_supervisor_args,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec, pid_alive, stop_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.runner import TodoDaemonRunner
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor import (
    SupervisorStatusContext,
    worktree_phase_worker_status,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
    SupervisorLoopDecision,
    SupervisorLoopResult,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    ChildSummaryHealthSpec,
    ConfiguredSupervisorEntrypoint,
    RestartPolicy,
    SupervisedChild,
    SupervisedChildSpec,
    background_supervisor_args,
    adopt_supervised_child,
    build_configured_implementation_supervisor_entrypoint,
    build_module_implementation_supervisor_entrypoint,
    build_supervisor_runtime_operations,
    child_exit_should_restart,
    install_stop_signal_handlers,
    implementation_supervisor_args,
    launch_process_child,
    launch_supervised_child,
    repair_supervisor_runtime,
    run_process_group_capture,
    supervisor_is_running,
    supervisor_runtime_paths,
    supervised_child_group_succeeded,
    supervised_child_succeeded,
    summarize_child_summary_files,
    timestamp_age_seconds,
    terminate_process_group,
    terminate_processes_with_grace,
    terminate_process_with_grace,
    terminate_supervised_child,
)
from ipfs_accelerate_py.agent_supervisor.wrapper_utils import (
    AGENT_SUPERVISOR_DIRECTORY_BOOTSTRAP_KEYS,
    AgentSupervisorNamespaceContext,
    AgentSupervisorNamespacePaths,
    AgentSupervisorRuntimeBootstrapCallbacks,
    BootstrapPathCallbacks,
    BootstrapPathSpec,
    CodebaseScanEnvSettings,
    DEFAULT_CODEBASE_SCAN_DATA_SUBDIRS,
    DEFAULT_REPO_DOCS_DIR,
    RepoScriptBootstrap,
    RuntimeEnvironmentCallbacks,
    android_validation_command_needs_environment,
    android_validation_environment_contract,
    agent_supervisor_bootstrap_path_entries,
    agent_supervisor_namespace_paths,
    apply_env_defaults,
    apply_environment_contract,
    bootstrap_runtime_environment,
    build_android_validation_callbacks,
    build_agent_supervisor_bootstrap_path_callbacks,
    build_agent_supervisor_namespace_context,
    build_agent_supervisor_runtime_bootstrap_callbacks,
    build_bootstrap_path_ensurer,
    build_bootstrap_path_resolver,
    build_default_llm_merge_resolver_command_callback,
    build_prefixed_bootstrap_path_callbacks,
    build_prefixed_default_llm_merge_resolver_command_callback,
    build_repo_runtime_environment_callbacks,
    build_repo_script_bootstrap,
    build_runtime_environment_callback,
    build_runtime_environment_callbacks,
    csv_tuple,
    data_namespace_scan_skip_prefixes,
    default_llm_merge_resolver_command,
    enforce_android_validation_environment,
    ensure_named_directories,
    ensure_runtime_pythonpath,
    ensure_sys_path,
    environment_assignment_prefix,
    env_csv_tuple,
    env_int,
    env_path,
    env_str,
    ObjectiveRefillEnvSettings,
    prefixed_bootstrap_path_spec,
    prefixed_bootstrap_path_specs,
    prefixed_codebase_scan_env_settings,
    prefixed_env_csv_tuple,
    prefixed_env_int,
    prefixed_env_path,
    prefixed_env_str,
    prefixed_env_var,
    prefixed_interoperability_focus,
    prefixed_objective_refill_env_settings,
    repo_doc_path,
    repo_external_package_root,
    repo_external_package_roots,
    repo_root_from_env,
    repo_relative_or_default,
    repo_script_command,
    repo_script_path,
    repo_task_board_path,
    resolve_and_ensure_bootstrap_paths,
    resolve_bootstrap_paths,
    rewrite_validation_commands,
    task_board_env_var,
    task_board_filename,
    task_board_path_key,
    task_board_path_option,
    unique_path_entries,
    with_android_validation_environment,
    with_default,
    with_exclusive_flag_default,
    with_flag_default,
    with_repeated_default,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    source = repo / "src" / "control_surface.py"
    source.parent.mkdir()
    source.write_text(
        """class VoiceCommandSurface:
    def route_click(self, event):
        return event
""",
        encoding="utf-8",
    )
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G010 Meta display control bridge

- Status: active
- Parent:
- Fib priority: 1
- Track: mobile
- Priority: P1
- Bundle: objective/mobile/meta-display
- Goal: Prove the glasses control bridge.
- Evidence: VoiceCommandSurface.route_click, missing_gesture_policy
- Outputs: src, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing gesture policy proof.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/control_surface.py")
    _git(repo, "commit", "-m", "seed objective heap")
    return repo, objective_path, todo_path


def test_wrapper_utils_apply_defaults_and_runtime_paths(monkeypatch, tmp_path):
    assert with_default(["--flag", "caller"], "--flag", "default") == ["--flag", "caller"]
    assert with_default(["tail"], "--flag", "default") == ["--flag", "default", "tail"]
    assert with_flag_default(["tail"], "--enabled") == ["--enabled", "tail"]
    assert with_flag_default(["--enabled"], "--enabled") == ["--enabled"]
    assert with_exclusive_flag_default(["tail"], "--implement", ("--no-implement",)) == [
        "--implement",
        "tail",
    ]
    assert with_exclusive_flag_default(["--implement"], "--implement", ("--no-implement",)) == ["--implement"]
    assert with_exclusive_flag_default(["--no-implement"], "--implement", ("--no-implement",)) == [
        "--no-implement"
    ]
    assert with_repeated_default(["tail"], "--path", ("a", "b")) == [
        "--path",
        "a",
        "--path",
        "b",
        "tail",
    ]
    assert with_repeated_default(["--path", "caller"], "--path", ("a", "b")) == ["--path", "caller"]
    assert csv_tuple(["a,b", "b,c"]) == ("a", "b", "c")
    assert DEFAULT_CODEBASE_SCAN_DATA_SUBDIRS == (
        "discovery",
        "objective_bundles",
        "objective_datasets",
        "state",
        "worktrees",
    )
    assert data_namespace_scan_skip_prefixes(
        {
            "virtual_ai_os": ("discovery", "state"),
            "meta_glasses_display_widgets": ("discovery",),
        },
        include_scripts=True,
    ) == (
        "scripts/",
        "data/virtual_ai_os/discovery/",
        "data/virtual_ai_os/state/",
        "data/meta_glasses_display_widgets/discovery/",
    )
    assert data_namespace_scan_skip_prefixes(
        ("agent_supervisor",),
        root="runtime/data/",
        subdirs=("state",),
        script_prefix="tools",
        include_scripts=True,
        extra_prefixes=("runtime/data/agent_supervisor/state/",),
    ) == (
        "tools/",
        "runtime/data/agent_supervisor/state/",
    )
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")
    assert isinstance(namespace_paths, AgentSupervisorNamespacePaths)
    assert namespace_paths.namespace_root == tmp_path / "data" / "agent_supervisor"
    assert namespace_paths.discovery_dir == tmp_path / "data" / "agent_supervisor" / "discovery"
    assert namespace_paths.state_dir == tmp_path / "data" / "agent_supervisor" / "state"
    assert namespace_paths.worktree_root == tmp_path / "data" / "agent_supervisor" / "worktrees"
    assert namespace_paths.objective_graph_path == tmp_path / "data" / "agent_supervisor" / "objective_graph.json"
    assert namespace_paths.objective_bundle_dir == tmp_path / "data" / "agent_supervisor" / "objective_bundles"
    assert namespace_paths.objective_dataset_dir == tmp_path / "data" / "agent_supervisor" / "objective_datasets"
    assert namespace_paths.objective_todo_vector_index_path == (
        tmp_path / "data" / "agent_supervisor" / "objective_bundles" / "todo_vector_index.json"
    )
    assert namespace_paths.repo_relative_path("discovery_dir", "fallback") == "data/agent_supervisor/discovery"
    assert namespace_paths.discovery_output_path() == "data/agent_supervisor/discovery"
    assert namespace_paths.discovery_output_path("fallback") == "data/agent_supervisor/discovery"
    custom_namespace_paths = agent_supervisor_namespace_paths(
        tmp_path,
        "agent_supervisor",
        data_root="runtime",
        objective_bundle_subdir="bundles",
        objective_dataset_subdir="datasets",
        todo_vector_index_filename="index.json",
    )
    assert custom_namespace_paths.objective_bundle_dir == tmp_path / "runtime" / "agent_supervisor" / "bundles"
    assert custom_namespace_paths.objective_dataset_dir == tmp_path / "runtime" / "agent_supervisor" / "datasets"
    assert custom_namespace_paths.objective_todo_vector_index_path == (
        tmp_path / "runtime" / "agent_supervisor" / "bundles" / "index.json"
    )
    assert custom_namespace_paths.discovery_output_path() == "runtime/agent_supervisor/discovery"
    assert AGENT_SUPERVISOR_DIRECTORY_BOOTSTRAP_KEYS == (
        "state_dir",
        "worktree_root",
        "discovery_dir",
        "objective_bundle_dir",
        "objective_dataset_dir",
    )
    assert agent_supervisor_bootstrap_path_entries(
        tmp_path / "tasks.md",
        namespace_paths,
        todo_key="task_board_path",
        todo_setting="todo_path",
        objective_path=tmp_path / "objective.md",
        namespace_keys=("state_dir", "discovery_dir", "objective_bundle_dir"),
    ) == (
        ("task_board_path", tmp_path / "tasks.md", "todo_path"),
        ("objective_heap_path", tmp_path / "objective.md"),
        ("state_dir", namespace_paths.state_dir),
        ("discovery_dir", namespace_paths.discovery_dir),
        ("objective_bundle_dir", namespace_paths.objective_bundle_dir),
    )
    bootstrap_callbacks = build_agent_supervisor_bootstrap_path_callbacks(
        tmp_path,
        "WRAPPER_UTILS",
        tmp_path / "tasks.md",
        namespace_paths,
        todo_key="task_board_path",
        todo_setting="todo_path",
        objective_path=tmp_path / "objective.md",
        namespace_keys=("state_dir", "discovery_dir", "objective_graph_path"),
    )
    assert bootstrap_callbacks.specs == (
        BootstrapPathSpec("task_board_path", tmp_path / "tasks.md", "WRAPPER_UTILS_TODO_PATH"),
        BootstrapPathSpec("objective_heap_path", tmp_path / "objective.md", "WRAPPER_UTILS_OBJECTIVE_HEAP_PATH"),
        BootstrapPathSpec("state_dir", namespace_paths.state_dir, "WRAPPER_UTILS_STATE_DIR"),
        BootstrapPathSpec("discovery_dir", namespace_paths.discovery_dir, "WRAPPER_UTILS_DISCOVERY_DIR"),
        BootstrapPathSpec(
            "objective_graph_path",
            namespace_paths.objective_graph_path,
            "WRAPPER_UTILS_OBJECTIVE_GRAPH_PATH",
        ),
    )
    ensured = bootstrap_callbacks.ensure(None)
    assert ensured["state_dir"].is_dir()
    assert ensured["discovery_dir"].is_dir()
    assert not ensured["objective_graph_path"].exists()
    monkeypatch.setenv("WRAPPER_UTILS_CSV", "alpha,beta")
    assert env_csv_tuple("WRAPPER_UTILS_CSV", "fallback") == ("alpha", "beta")
    monkeypatch.delenv("WRAPPER_UTILS_STRING", raising=False)
    assert env_str("WRAPPER_UTILS_STRING", "fallback") == "fallback"
    monkeypatch.setenv("WRAPPER_UTILS_STRING", " custom ")
    assert env_str("WRAPPER_UTILS_STRING", "fallback") == "custom"
    target_env = {"EXISTING": "caller", "EMPTY": ""}
    assert apply_env_defaults(
        {"EXISTING": "default", "MISSING": "default", "EMPTY": "default"},
        environ=target_env,
    ) == {"EXISTING": "caller", "MISSING": "default", "EMPTY": ""}
    assert target_env["MISSING"] == "default"
    assert apply_env_defaults({"EMPTY": "filled"}, environ=target_env, replace_empty=True)["EMPTY"] == "filled"
    monkeypatch.delenv("WRAPPER_UTILS_INT", raising=False)
    assert env_int("WRAPPER_UTILS_INT", 7) == 7
    monkeypatch.setenv("WRAPPER_UTILS_INT", "11")
    assert env_int("WRAPPER_UTILS_INT", 7, minimum=10, maximum=12) == 11
    monkeypatch.delenv("WRAPPER_UTILS_PATH", raising=False)
    assert env_path("WRAPPER_UTILS_PATH", tmp_path / "default-path") == tmp_path / "default-path"
    monkeypatch.setenv("WRAPPER_UTILS_PATH", str(tmp_path / "custom-path"))
    assert env_path("WRAPPER_UTILS_PATH", tmp_path / "default-path") == tmp_path / "custom-path"
    monkeypatch.setenv("WRAPPER_UTILS_PATH", "")
    assert env_path("WRAPPER_UTILS_PATH", tmp_path / "default-path") == Path(".")
    assert repo_root_from_env(environ={}, fallback=tmp_path / "fallback") == (tmp_path / "fallback").resolve()
    assert repo_root_from_env(environ={"REPO_ROOT": ""}, fallback=tmp_path / "fallback") == (
        tmp_path / "fallback"
    ).resolve()
    assert repo_root_from_env(environ={"REPO_ROOT": str(tmp_path / "override")}, fallback=tmp_path / "fallback") == (
        tmp_path / "override"
    ).resolve()
    script_path = tmp_path / "bootstrap-repo" / "scripts" / "run.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    bootstrap_sys_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(bootstrap_sys_path))
    bootstrap = build_repo_script_bootstrap(
        script_path,
        environ={"REPO_ROOT": str(tmp_path / "runtime-repo")},
    )
    assert isinstance(bootstrap, RepoScriptBootstrap)
    assert bootstrap.script_path == script_path.resolve()
    assert bootstrap.script_repo_root == tmp_path / "bootstrap-repo"
    assert bootstrap.repo_root == (tmp_path / "runtime-repo").resolve()
    assert bootstrap.package_root == tmp_path / "bootstrap-repo" / "external" / "ipfs_accelerate"
    assert bootstrap.script_dir == script_path.parent
    assert sys.path[0] == str(bootstrap.package_root)
    monkeypatch.setattr(sys, "path", list(bootstrap_sys_path))
    bootstrap_with_script_dir = build_repo_script_bootstrap(
        script_path,
        include_script_dir=True,
        environ={},
    )
    assert bootstrap_with_script_dir.script_dir == script_path.parent
    assert sys.path[:2] == [
        str(bootstrap_with_script_dir.package_root),
        str(script_path.parent),
    ]
    monkeypatch.setattr(sys, "path", list(bootstrap_sys_path))
    custom_bootstrap = build_repo_script_bootstrap(
        script_path,
        package_name="accelerator_pkg",
        external_dir="vendor",
        ensure_package_path=False,
        environ={},
    )
    assert custom_bootstrap.package_root == tmp_path / "bootstrap-repo" / "vendor" / "accelerator_pkg"
    assert custom_bootstrap.repo_root == (tmp_path / "bootstrap-repo").resolve()
    assert sys.path == bootstrap_sys_path
    assert repo_external_package_root(tmp_path, "ipfs_accelerate") == tmp_path / "external" / "ipfs_accelerate"
    assert repo_external_package_root(tmp_path, "custom", external_dir="vendor") == tmp_path / "vendor" / "custom"
    absolute_package = tmp_path / "absolute-package"
    assert repo_external_package_root(tmp_path, absolute_package) == absolute_package
    assert repo_external_package_roots(tmp_path, ("ipfs_accelerate", "ipfs_datasets")) == (
        tmp_path / "external" / "ipfs_accelerate",
        tmp_path / "external" / "ipfs_datasets",
    )
    assert repo_script_path(tmp_path, "run.sh") == tmp_path / "scripts" / "run.sh"
    assert repo_script_path(tmp_path, "tools/run.sh") == tmp_path / "tools" / "run.sh"
    assert repo_script_path(tmp_path, tmp_path / "absolute-run.sh") == tmp_path / "absolute-run.sh"
    assert repo_script_path(tmp_path, "run.sh", scripts_dir="bin") == tmp_path / "bin" / "run.sh"
    assert repo_script_command(tmp_path, "scripts/run.sh") == f"bash {tmp_path / 'scripts' / 'run.sh'}"
    assert repo_script_command(tmp_path, "scripts/path with space.sh", command=("python3", "-u")) == (
        "python3 -u " + shlex.quote(str(tmp_path / "scripts" / "path with space.sh"))
    )
    assert prefixed_env_var("wrapper_utils", "state_dir") == "WRAPPER_UTILS_STATE_DIR"
    assert prefixed_env_var("WRAPPER_UTILS_", "_worktree_root", "") == "WRAPPER_UTILS_WORKTREE_ROOT"
    assert prefixed_bootstrap_path_spec("state_dir", "state", "WRAPPER_UTILS") == BootstrapPathSpec(
        "state_dir",
        "state",
        "WRAPPER_UTILS_STATE_DIR",
    )
    assert prefixed_bootstrap_path_spec(
        "task_board_path",
        "tasks.md",
        "WRAPPER_UTILS",
        "todo_path",
    ) == BootstrapPathSpec(
        "task_board_path",
        "tasks.md",
        "WRAPPER_UTILS_TODO_PATH",
    )
    assert prefixed_bootstrap_path_specs(
        "WRAPPER_UTILS",
        (
            ("state_dir", "state"),
            ("task_board_path", "tasks.md", "todo_path"),
        ),
    ) == (
        BootstrapPathSpec("state_dir", "state", "WRAPPER_UTILS_STATE_DIR"),
        BootstrapPathSpec("task_board_path", "tasks.md", "WRAPPER_UTILS_TODO_PATH"),
    )
    with pytest.raises(ValueError, match="2 or 3 fields"):
        prefixed_bootstrap_path_specs("WRAPPER_UTILS", (("bad", "path", "setting", "extra"),))
    monkeypatch.setenv("WRAPPER_UTILS_TYPED_CSV", "one,two")
    assert prefixed_env_csv_tuple("WRAPPER_UTILS", "TYPED_CSV") == ("one", "two")
    monkeypatch.setenv("WRAPPER_UTILS_TYPED_STRING", " typed ")
    assert prefixed_env_str("WRAPPER_UTILS", "TYPED_STRING", "fallback") == "typed"
    monkeypatch.setenv("WRAPPER_UTILS_INTEROPERABILITY_FOCUS", "hallucinate_app,swissknife")
    assert prefixed_interoperability_focus("WRAPPER_UTILS", "fallback") == ("hallucinate_app", "swissknife")
    monkeypatch.setenv("WRAPPER_UTILS_TYPED_INT", "12")
    assert prefixed_env_int("WRAPPER_UTILS", "TYPED_INT", 7, minimum=10, maximum=20) == 12
    monkeypatch.setenv("WRAPPER_UTILS_TYPED_PATH", str(tmp_path / "typed-path"))
    assert prefixed_env_path("WRAPPER_UTILS", "TYPED_PATH", tmp_path / "default-path") == tmp_path / "typed-path"
    monkeypatch.setenv("WRAPPER_UTILS_CODEBASE_SCAN_MIN_OPEN_TASKS", "6")
    monkeypatch.setenv("WRAPPER_UTILS_CODEBASE_SCAN_MAX_FINDINGS", "8")
    monkeypatch.setenv("WRAPPER_UTILS_CODEBASE_SCAN_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("WRAPPER_UTILS_CODEBASE_REFILL_TIMEOUT_SECONDS", "601")
    assert prefixed_codebase_scan_env_settings("WRAPPER_UTILS") == CodebaseScanEnvSettings(
        min_open_tasks=6,
        max_findings=8,
        cooldown_seconds=120,
        timeout_seconds=601,
    )
    codebase_settings = prefixed_codebase_scan_env_settings("WRAPPER_UTILS")
    assert codebase_settings.recorder_kwargs() == {
        "min_open_tasks": 6,
        "max_findings": 8,
        "cooldown_seconds": 120,
    }
    assert codebase_settings.codebase_refill_kwargs() == {
        "codebase_scan_min_open_tasks": 6,
        "codebase_scan_max_findings": 8,
        "codebase_scan_cooldown_seconds": 120,
        "codebase_refill_timeout_seconds": 601,
    }
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_MIN_OPEN_TASKS", "21")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_MAX_FINDINGS", "13")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_COOLDOWN_SECONDS", "901")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_REFILL_TIMEOUT_SECONDS", "601")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "7")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "5")
    assert prefixed_objective_refill_env_settings("WRAPPER_UTILS") == ObjectiveRefillEnvSettings(
        min_open_tasks=21,
        max_findings=13,
        cooldown_seconds=901,
        timeout_seconds=601,
        surplus_findings_per_goal=7,
        surplus_min_terms_per_todo=5,
    )
    objective_settings = prefixed_objective_refill_env_settings("WRAPPER_UTILS")
    assert objective_settings.recorder_kwargs() == {
        "min_open_tasks": 21,
        "max_findings": 13,
        "cooldown_seconds": 901,
        "surplus_findings_per_goal": 7,
        "surplus_min_terms_per_todo": 5,
    }
    assert objective_settings.objective_refill_kwargs() == {
        "objective_scan_min_open_tasks": 21,
        "objective_scan_max_findings": 13,
        "objective_scan_cooldown_seconds": 901,
        "objective_refill_timeout_seconds": 601,
        "objective_surplus_findings_per_goal": 7,
        "objective_surplus_min_terms_per_todo": 5,
    }
    assert task_board_env_var("WRAPPER_UTILS") == "WRAPPER_UTILS_TODO_PATH"
    assert task_board_env_var("WRAPPER_UTILS_") == "WRAPPER_UTILS_TODO_PATH"
    assert task_board_filename("roadmap") == "roadmap.todo.md"
    assert task_board_filename("roadmap", ".markdown") == "roadmap.todo.markdown"
    assert DEFAULT_REPO_DOCS_DIR == "implementation_plan/docs"
    assert repo_doc_path(tmp_path, "roadmap.md") == tmp_path / "implementation_plan" / "docs" / "roadmap.md"
    assert repo_doc_path(tmp_path, "docs/roadmap.md") == tmp_path / "docs" / "roadmap.md"
    assert repo_doc_path(tmp_path, tmp_path / "absolute.md") == tmp_path / "absolute.md"
    assert repo_doc_path(tmp_path, "roadmap.md", docs_dir="project/docs") == (
        tmp_path / "project" / "docs" / "roadmap.md"
    )
    assert repo_task_board_path(tmp_path, "roadmap") == (
        tmp_path / "implementation_plan" / "docs" / "roadmap.todo.md"
    )
    assert repo_task_board_path(tmp_path, "roadmap", docs_dir="project/docs", suffix=".markdown") == (
        tmp_path / "project" / "docs" / "roadmap.todo.markdown"
    )
    assert task_board_path_option() == "--todo-path"
    assert task_board_path_key() == "todo_path"
    monkeypatch.setenv("WRAPPER_UTILS_PRIMARY_MERGE_COMMAND", "primary-merge")
    merge_command_callback = build_default_llm_merge_resolver_command_callback(
        primary_env_var="WRAPPER_UTILS_PRIMARY_MERGE_COMMAND"
    )
    assert merge_command_callback() == "primary-merge"
    monkeypatch.setenv("WRAPPER_UTILS_LLM_MERGE_RESOLVER_COMMAND", "prefixed-merge")
    prefixed_merge_command_callback = build_prefixed_default_llm_merge_resolver_command_callback("WRAPPER_UTILS")
    assert prefixed_merge_command_callback() == "prefixed-merge"
    assert unique_path_entries(["a", "", "b", "a"]) == ["a", "b"]
    assert repo_relative_or_default(tmp_path / "first", tmp_path, "fallback") == "first"
    assert repo_relative_or_default(Path("/"), tmp_path / "repo", "fallback") == "fallback"
    monkeypatch.setenv("WRAPPER_UTILS_STATE_DIR", str(tmp_path / "custom-state"))
    bootstrap_paths = resolve_bootstrap_paths(
        tmp_path,
        (
            BootstrapPathSpec("state_dir", "state", "WRAPPER_UTILS_STATE_DIR"),
            BootstrapPathSpec("worktree_root", tmp_path / "worktrees"),
        ),
    )
    assert bootstrap_paths == {
        "repo_root": tmp_path,
        "state_dir": tmp_path / "custom-state",
        "worktree_root": tmp_path / "worktrees",
    }
    paths = {"state_dir": tmp_path / "state", "worktree_root": tmp_path / "worktrees"}
    ensured = ensure_named_directories(paths, ("state_dir", "worktree_root"))
    assert ensured == paths
    assert paths["state_dir"].is_dir()
    assert paths["worktree_root"].is_dir()

    ensured_bootstrap = resolve_and_ensure_bootstrap_paths(
        tmp_path,
        (BootstrapPathSpec("cache_dir", "cache"),),
        ("cache_dir",),
    )
    assert ensured_bootstrap["cache_dir"] == tmp_path / "cache"
    assert ensured_bootstrap["cache_dir"].is_dir()
    provided_bootstrap = resolve_and_ensure_bootstrap_paths(
        tmp_path,
        (),
        ("cache_dir",),
        paths={"cache_dir": tmp_path / "provided-cache"},
    )
    assert provided_bootstrap["cache_dir"].is_dir()
    resolver = build_bootstrap_path_resolver(
        tmp_path,
        (BootstrapPathSpec("cache_dir", "callback-cache"),),
    )
    assert resolver()["cache_dir"] == tmp_path / "callback-cache"
    ensurer = build_bootstrap_path_ensurer(
        tmp_path,
        (BootstrapPathSpec("cache_dir", "callback-cache"),),
        ("cache_dir",),
    )
    ensured_callback = ensurer()
    assert ensured_callback["cache_dir"].is_dir()
    provided_callback = ensurer({"cache_dir": tmp_path / "provided-callback-cache"})
    assert provided_callback["cache_dir"].is_dir()
    callback_bundle = build_prefixed_bootstrap_path_callbacks(
        tmp_path,
        "WRAPPER_UTILS_CALLBACKS",
        (
            ("state_dir", "state"),
            ("task_board_path", "tasks.md", "todo_path"),
        ),
        ("state_dir",),
    )
    assert isinstance(callback_bundle, BootstrapPathCallbacks)
    assert callback_bundle.repo_root == tmp_path
    assert callback_bundle.specs == (
        BootstrapPathSpec("state_dir", "state", "WRAPPER_UTILS_CALLBACKS_STATE_DIR"),
        BootstrapPathSpec("task_board_path", "tasks.md", "WRAPPER_UTILS_CALLBACKS_TODO_PATH"),
    )
    assert callback_bundle.resolve()["task_board_path"] == tmp_path / "tasks.md"
    assert callback_bundle.output_path("task_board_path", "fallback.md") == "tasks.md"
    assert callback_bundle.output_path("state_dir", "fallback", {"state_dir": Path("/")}) == "fallback"
    task_board_output_path = callback_bundle.output_path_factory("task_board_path", "fallback.md")
    assert task_board_output_path() == "tasks.md"
    assert task_board_output_path({"task_board_path": Path("/")}) == "fallback.md"
    task_board_output_kwargs = callback_bundle.output_path_kwargs_factory(
        "discovery_output_path",
        "task_board_path",
        "fallback.md",
    )
    assert task_board_output_kwargs() == {"discovery_output_path": "tasks.md"}
    assert task_board_output_kwargs({"task_board_path": Path("/")}) == {
        "discovery_output_path": "fallback.md",
    }
    ensured_callback_bundle = callback_bundle.ensure()
    assert ensured_callback_bundle["state_dir"].is_dir()

    contract = {
        "env": {"JAVA_HOME": "/jdk", "ANDROID_HOME": "/sdk"},
        "path_entries": ["/jdk/bin", "/sdk/bin", "/jdk/bin"],
    }
    target_env = {"PATH": "/usr/bin"}
    applied = apply_environment_contract(contract, environ=target_env)
    assert target_env["JAVA_HOME"] == "/jdk"
    assert target_env["PATH"].split(os.pathsep) == ["/jdk/bin", "/sdk/bin", "/usr/bin"]
    assert applied["effective_path"] == target_env["PATH"]
    assert environment_assignment_prefix(
        contract,
        env_keys=("JAVA_HOME", "ANDROID_HOME", "ANDROID_SDK_ROOT"),
    ) == "JAVA_HOME=/jdk ANDROID_HOME=/sdk PATH=/jdk/bin:/sdk/bin:$PATH"

    todo_path = tmp_path / "tasks.todo.md"
    todo_path.write_text(
        "# Tasks\n\n## TST-001 Build\n\n- Validation: first; second\n",
        encoding="utf-8",
    )
    assert rewrite_validation_commands(todo_path, lambda command: command.upper())
    assert "- Validation: FIRST; SECOND" in todo_path.read_text(encoding="utf-8")
    assert not rewrite_validation_commands(todo_path, lambda command: command)
    quoted_python_validation = (
        "python3 -c 'import pathlib, sys; "
        'p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        "' src/config.yml"
    )
    todo_path.write_text(
        f"# Tasks\n\n## TST-002 Build\n\n- Validation: {quoted_python_validation}; gradle test\n",
        encoding="utf-8",
    )
    seen_commands: list[str] = []

    def transform_command(command: str) -> str:
        seen_commands.append(command)
        return command if command.startswith("python3 -c") else command.upper()

    assert rewrite_validation_commands(todo_path, transform_command)
    assert seen_commands == [quoted_python_validation, "gradle test"]
    assert f"- Validation: {quoted_python_validation}; GRADLE TEST" in todo_path.read_text(encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(second), "existing"]))
    original_sys_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_sys_path))

    ensure_sys_path((second, first))
    assert sys.path[:2] == [str(second), str(first)]
    monkeypatch.setattr(sys, "path", list(original_sys_path))

    ensure_runtime_pythonpath((first, second))

    assert sys.path[:2] == [str(first), str(second)]
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(first), str(second), "existing"]

    cwd = Path.cwd()
    repo = tmp_path / "repo"
    third = tmp_path / "third"
    repo.mkdir()
    try:
        bootstrap_runtime_environment(repo, (third,))
        assert Path.cwd() == repo
        assert sys.path[0] == str(third)
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(third)
    finally:
        os.chdir(cwd)

    fourth = tmp_path / "fourth"
    callback = build_runtime_environment_callback(repo, (fourth,), chdir=False)
    callback()
    assert Path.cwd() == cwd
    assert sys.path[0] == str(fourth)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(fourth)

    fifth = tmp_path / "fifth"
    sixth = tmp_path / "sixth"
    callbacks = build_runtime_environment_callbacks(
        repo,
        (fifth, sixth),
        primary_import_paths=(fifth,),
    )
    assert isinstance(callbacks, RuntimeEnvironmentCallbacks)
    callbacks.ensure_primary_pythonpath()
    assert Path.cwd() == cwd
    assert sys.path[0] == str(fifth)
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    callbacks.ensure_pythonpath()
    assert sys.path[:2] == [str(fifth), str(sixth)]
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    repo_callbacks = build_repo_runtime_environment_callbacks(
        repo,
        ("pkg_a", "pkg_b"),
        primary_package_names=("pkg_a",),
    )
    repo_callbacks.ensure_primary_pythonpath()
    assert sys.path[0] == str(repo / "external" / "pkg_a")
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    repo_callbacks.ensure_pythonpath()
    assert sys.path[:2] == [
        str(repo / "external" / "pkg_a"),
        str(repo / "external" / "pkg_b"),
    ]
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    runtime_bootstrap = build_agent_supervisor_runtime_bootstrap_callbacks(
        repo,
        "WRAPPER_UTILS_RUNTIME",
        repo / "runtime-tasks.md",
        agent_supervisor_namespace_paths(repo, "runtime_namespace"),
        objective_path=repo / "objective.md",
        namespace_keys=("state_dir", "worktree_root", "discovery_dir"),
        runtime_package_names=("pkg_a", "pkg_b"),
        runtime_primary_package_names=("pkg_a",),
    )
    assert isinstance(runtime_bootstrap, AgentSupervisorRuntimeBootstrapCallbacks)
    assert runtime_bootstrap.specs == (
        BootstrapPathSpec("todo_path", repo / "runtime-tasks.md", "WRAPPER_UTILS_RUNTIME_TODO_PATH"),
        BootstrapPathSpec("objective_heap_path", repo / "objective.md", "WRAPPER_UTILS_RUNTIME_OBJECTIVE_HEAP_PATH"),
        BootstrapPathSpec(
            "state_dir",
            repo / "data" / "runtime_namespace" / "state",
            "WRAPPER_UTILS_RUNTIME_STATE_DIR",
        ),
        BootstrapPathSpec(
            "worktree_root",
            repo / "data" / "runtime_namespace" / "worktrees",
            "WRAPPER_UTILS_RUNTIME_WORKTREE_ROOT",
        ),
        BootstrapPathSpec(
            "discovery_dir",
            repo / "data" / "runtime_namespace" / "discovery",
            "WRAPPER_UTILS_RUNTIME_DISCOVERY_DIR",
        ),
    )
    assert runtime_bootstrap.resolve()["todo_path"] == repo / "runtime-tasks.md"
    ensured_runtime_bootstrap = runtime_bootstrap.ensure(None)
    assert ensured_runtime_bootstrap["state_dir"].is_dir()
    assert ensured_runtime_bootstrap["worktree_root"].is_dir()
    assert ensured_runtime_bootstrap["discovery_dir"].is_dir()
    runtime_bootstrap.ensure_primary_pythonpath()
    assert sys.path[0] == str(repo / "external" / "pkg_a")
    monkeypatch.setattr(sys, "path", list(original_sys_path))
    runtime_bootstrap.ensure_pythonpath()
    assert sys.path[:2] == [
        str(repo / "external" / "pkg_a"),
        str(repo / "external" / "pkg_b"),
    ]
    try:
        callbacks.enter()
        assert Path.cwd() == repo
    finally:
        os.chdir(cwd)


def test_wrapper_utils_namespace_context_binds_standard_wrapper_layout(tmp_path):
    repo = tmp_path / "repo"
    objective_path = repo / "docs" / "objective.md"
    context = build_agent_supervisor_namespace_context(
        repo,
        "WRAPPER_UTILS_CONTEXT",
        namespace="context_namespace",
        task_board_stem="roadmap",
        task_board_key="task_board_path",
        task_board_setting="todo_path",
        objective_path=objective_path,
        namespace_keys=("state_dir", "worktree_root", "objective_bundle_dir"),
        runtime_package_names=("pkg_a", "pkg_b"),
        runtime_primary_package_names=("pkg_a",),
    )

    assert isinstance(context, AgentSupervisorNamespaceContext)
    assert context.repo_root == repo
    assert context.env_prefix == "WRAPPER_UTILS_CONTEXT"
    assert context.task_board_path == repo / "implementation_plan" / "docs" / "roadmap.todo.md"
    assert context.task_board_path_key == "task_board_path"
    assert context.task_board_path_option == "--todo-path"
    assert context.namespace_paths == agent_supervisor_namespace_paths(repo, "context_namespace")
    assert context.bootstrap_paths is context.runtime_bootstrap.bootstrap_paths
    assert context.runtime_environment is context.runtime_bootstrap.runtime_environment
    assert context.specs == (
        BootstrapPathSpec(
            "task_board_path",
            repo / "implementation_plan" / "docs" / "roadmap.todo.md",
            "WRAPPER_UTILS_CONTEXT_TODO_PATH",
        ),
        BootstrapPathSpec("objective_heap_path", objective_path, "WRAPPER_UTILS_CONTEXT_OBJECTIVE_HEAP_PATH"),
        BootstrapPathSpec(
            "state_dir",
            repo / "data" / "context_namespace" / "state",
            "WRAPPER_UTILS_CONTEXT_STATE_DIR",
        ),
        BootstrapPathSpec(
            "worktree_root",
            repo / "data" / "context_namespace" / "worktrees",
            "WRAPPER_UTILS_CONTEXT_WORKTREE_ROOT",
        ),
        BootstrapPathSpec(
            "objective_bundle_dir",
            repo / "data" / "context_namespace" / "objective_bundles",
            "WRAPPER_UTILS_CONTEXT_OBJECTIVE_BUNDLE_DIR",
        ),
    )
    ensured_paths = context.bootstrap_paths.ensure(None)
    assert ensured_paths["state_dir"].is_dir()
    assert ensured_paths["worktree_root"].is_dir()
    assert ensured_paths["objective_bundle_dir"].is_dir()


def test_default_llm_merge_resolver_command_prefers_env(monkeypatch):
    monkeypatch.setenv("PRIMARY_RESOLVER_COMMAND", "primary-command")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", "fallback-command")
    assert default_llm_merge_resolver_command(primary_env_var="PRIMARY_RESOLVER_COMMAND") == "primary-command"

    monkeypatch.delenv("PRIMARY_RESOLVER_COMMAND")
    assert default_llm_merge_resolver_command(primary_env_var="PRIMARY_RESOLVER_COMMAND") == "fallback-command"

    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    assert default_llm_merge_resolver_command(codex_args=("exec", "-")) == "/usr/bin/codex exec -"
    assert (
        default_llm_merge_resolver_command()
        == "/usr/bin/codex exec --ignore-user-config --dangerously-bypass-approvals-and-sandbox "
        "-c 'model_reasoning_effort=\"high\"' -C . -"
    )


def test_wrapper_utils_android_validation_environment_contract(tmp_path):
    java = tmp_path / ".tools" / "jdk17" / "jdk-17.0.18+8" / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.touch()
    (tmp_path / ".tools" / "android-sdk" / "platform-tools").mkdir(parents=True)

    contract = android_validation_environment_contract(tmp_path)

    assert contract["env"]["JAVA_HOME"] == str(java.parents[1])
    assert contract["env"]["ANDROID_HOME"] == str(tmp_path / ".tools" / "android-sdk")
    assert contract["path_entries"][:2] == [
        str(java.parent),
        str(tmp_path / ".tools" / "android-sdk" / "platform-tools"),
    ]
    assert contract["missing"] == []
    assert android_validation_command_needs_environment("cd mobile/android && ./gradlew test")
    assert not android_validation_command_needs_environment("cd mobile/android && JAVA_HOME=/jdk ./gradlew test")
    assert not android_validation_command_needs_environment("npm test")

    command = "cd mobile/android && ./gradlew :app:assembleDebug"
    wrapped = with_android_validation_environment(command, tmp_path)

    assert "env JAVA_HOME=" in wrapped
    assert "ANDROID_SDK_ROOT=" in wrapped
    assert wrapped.endswith("./gradlew :app:assembleDebug")

    todo_path = tmp_path / "tasks.md"
    todo_path.write_text(
        "# Tasks\n\n## TST-001 Android\n\n- Validation: cd mobile/android && ./gradlew build; true\n",
        encoding="utf-8",
    )
    assert enforce_android_validation_environment(todo_path, tmp_path)
    updated = todo_path.read_text(encoding="utf-8")
    assert "env JAVA_HOME=" in updated
    assert not enforce_android_validation_environment(todo_path, tmp_path)

    callback_todo_path = tmp_path / "callback-tasks.md"
    callback_todo_path.write_text(
        "# Tasks\n\n## TST-002 Android\n\n- Validation: cd mobile/android && ./gradlew test\n",
        encoding="utf-8",
    )
    callbacks = build_android_validation_callbacks(tmp_path, todo_path=callback_todo_path)
    assert callbacks.environment_contract()["env"]["JAVA_HOME"] == str(java.parents[1])
    assert callbacks.wrap_command(command).startswith("cd mobile/android && env JAVA_HOME=")
    assert callbacks.enforce_todo()
    assert "ANDROID_SDK_ROOT=" in callback_todo_path.read_text(encoding="utf-8")
    applied = callbacks.apply_environment()
    assert applied["env"]["JAVA_HOME"] == str(java.parents[1])


def test_supervisor_runtime_repairs_stale_markers(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    prefix = "agent"
    paths = supervisor_runtime_paths(state_dir, prefix)
    stale_pid = 99999999
    paths["managed_daemon_pid"].write_text(f"{stale_pid}\n", encoding="utf-8")
    paths["wrapper_pid"].write_text(f"{stale_pid}\n", encoding="utf-8")
    paths["implementation_lock"].write_text(json.dumps({"pid": stale_pid}), encoding="utf-8")
    paths["supervisor_status"].write_text(
        json.dumps({"status": "running", "supervisor_pid": stale_pid, "daemon_pid": stale_pid}),
        encoding="utf-8",
    )

    repairs = repair_supervisor_runtime(state_dir, prefix)

    assert str(paths["managed_daemon_pid"]) in repairs["removed"]
    assert str(paths["wrapper_pid"]) in repairs["removed"]
    assert str(paths["implementation_lock"]) in repairs["removed"]
    assert repairs["updated_status"] is True
    assert not paths["managed_daemon_pid"].exists()
    assert not paths["wrapper_pid"].exists()
    assert not paths["implementation_lock"].exists()
    status = json.loads(paths["supervisor_status"].read_text(encoding="utf-8"))
    assert status["status"] == "stale"
    assert status["repair_reason"] == "supervisor_pid_not_running"
    assert not supervisor_is_running(state_dir, prefix, process_match_any=("definitely-not-this-process",))

    paths["wrapper_pid"].write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert supervisor_is_running(state_dir, prefix, process_predicate=lambda pid: pid == os.getpid())


def test_background_supervisor_args_removes_once_and_defaults_implement():
    assert background_supervisor_args(["--once", "--flag"]) == ["--implement", "--flag"]
    assert background_supervisor_args(["--no-implement", "--once"]) == ["--no-implement"]


def test_implementation_supervisor_args_defaults_implement_without_removing_once():
    assert implementation_supervisor_args(["--once", "--flag"]) == ["--implement", "--once", "--flag"]
    assert implementation_supervisor_args(["--implement", "--once"]) == ["--implement", "--once"]
    assert implementation_supervisor_args(["--no-implement", "--once"]) == ["--no-implement", "--once"]


def test_configured_implementation_supervisor_entrypoint_defaults_and_dispatches(monkeypatch):
    captured: dict[str, object] = {}

    def supervisor_main(argv: list[str]) -> str:
        captured["argv"] = argv
        return "ran"

    entrypoint = build_configured_implementation_supervisor_entrypoint(supervisor_main)

    assert isinstance(entrypoint, ConfiguredSupervisorEntrypoint)
    assert entrypoint.with_defaults(["--once"]) == ["--implement", "--once"]
    assert entrypoint.with_defaults(["--no-implement", "--once"]) == ["--no-implement", "--once"]
    assert entrypoint.run(["--once"]) == "ran"
    assert captured["argv"] == ["--implement", "--once"]
    monkeypatch.setattr(sys, "argv", ["supervisor-autopilot", "--once"])
    assert entrypoint.run() == "ran"
    assert captured["argv"] == ["--implement", "--once"]


def test_module_implementation_supervisor_entrypoint_imports_main(tmp_path, monkeypatch):
    module_path = tmp_path / "example_supervisor.py"
    output_path = tmp_path / "argv.txt"
    module_path.write_text(
        "from pathlib import Path\n\n"
        f"OUTPUT_PATH = {str(output_path)!r}\n\n"
        "def main(argv):\n"
        "    Path(OUTPUT_PATH).write_text('|'.join(argv), encoding='utf-8')\n"
        "    return 'module-ran'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    entrypoint = build_module_implementation_supervisor_entrypoint("example_supervisor")

    assert isinstance(entrypoint, ConfiguredSupervisorEntrypoint)
    assert entrypoint.run(["--once"]) == "module-ran"
    assert output_path.read_text(encoding="utf-8") == "--implement|--once"


def test_build_implementation_daemon_defaults_from_paths(tmp_path):
    paths = {
        "task_board_path": tmp_path / "tasks.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
        "objective_heap_path": tmp_path / "objective-heap.md",
    }

    defaults = build_implementation_daemon_defaults_from_paths(
        paths,
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        todo_path_flag="--task-board",
        objective_path_key="objective_heap_path",
        objective_bundle_dir=tmp_path / "bundles",
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    )

    assert defaults == ImplementationDaemonDefaults(
        todo_path=tmp_path / "tasks.md",
        state_dir=tmp_path / "state",
        task_prefix="## AUTO-",
        state_prefix="agent",
        worktree_root=tmp_path / "worktrees",
        todo_path_flag="--task-board",
        objective_path=tmp_path / "objective-heap.md",
        objective_bundle_dir=tmp_path / "bundles",
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    )
    assert apply_portal_implementation_daemon_defaults_from_paths(
        ["--once"],
        paths,
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        todo_path_flag="--task-board",
        objective_path_key="objective_heap_path",
        objective_bundle_dir=tmp_path / "bundles",
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    ) == apply_portal_implementation_daemon_defaults(["--once"], defaults=defaults)


def test_implementation_daemon_skips_unauthenticated_copilot_fallback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Todos\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in {"codex", "copilot"} else None,
    )
    monkeypatch.setattr(implementation_daemon_module, "_copilot_has_auth", lambda: False)

    command = daemon._build_implementation_command(repo)

    assert command[:5] == [
        "/usr/local/bin/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(repo),
    ]
    assert command[-1] == "-"
    assert ["-c", "model_context_window=200000"] == command[5:7]
    assert "-c" in command
    assert 'model_reasoning_effort="high"' in command
    assert "agents.max_threads=10" in command
    assert "agents.max_depth=2" in command


def test_implementation_daemon_uses_authenticated_copilot_fallback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Todos\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in {"codex", "copilot"} else None,
    )
    monkeypatch.setattr(implementation_daemon_module, "_copilot_has_auth", lambda: True)

    command = daemon._build_implementation_command(repo)

    assert command[:2] == ["bash", "-lc"]
    assert "falling back to copilot" in command[2]
    assert command[3:7] == ["bash", "/usr/local/bin/codex", "/usr/local/bin/copilot", str(repo)]
    assert command[7:] == ["", "200000", "high", "10", "2", "", "high", "long_context", "30"]


def test_implementation_daemon_links_shared_dependencies_only_in_managed_worktrees(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "swissknife" / "node_modules"
    source.mkdir(parents=True)
    worktree_root = tmp_path / "worktrees"
    worktree = worktree_root / "task-attempt"
    (worktree / "swissknife").mkdir(parents=True)
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_root=worktree_root,
    )

    daemon._link_shared_worktree_paths(worktree)

    target = worktree / "swissknife" / "node_modules"
    assert target.is_symlink()
    assert target.resolve() == source.resolve()

    outside = tmp_path / "outside-worktree"
    outside.mkdir()
    daemon._link_shared_worktree_paths(outside)
    assert not (outside / "swissknife" / "node_modules").exists()


def test_implementation_daemon_never_nests_shared_dependency_links_inside_their_source(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "swissknife" / "node_modules"
    source.mkdir(parents=True)
    worktree_root = tmp_path / "worktrees"
    worktree = worktree_root / "task-attempt"
    worktree.mkdir(parents=True)
    (worktree / "swissknife").symlink_to(source, target_is_directory=True)
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_root=worktree_root,
    )

    daemon._link_shared_worktree_paths(worktree)

    assert source.is_dir()
    assert not (source / "node_modules").exists()


def test_build_configured_implementation_daemon_runner_reuses_binding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    objective_path = repo / "objective.md"
    bundle_dir = repo / "bundles"
    repo.mkdir()
    captured: dict[str, object] = {}

    def fake_run_configured_portal_implementation_daemon(argv, **kwargs):
        captured["argv"] = tuple(argv)
        captured["kwargs"] = kwargs
        return "daemon-ran"

    monkeypatch.setattr(
        implementation_daemon_runner,
        "run_configured_portal_implementation_daemon",
        fake_run_configured_portal_implementation_daemon,
    )

    runner = build_configured_implementation_daemon_runner(
        repo_root=str(repo),
        logger=logging.getLogger("test-configured-daemon-runner"),
        default_worktree_submodule_paths=("external/custom",),
        default_objective_path=str(objective_path),
        default_objective_bundle_dir=str(bundle_dir),
        pass_complete_message="custom complete: %s",
    )
    hook = implementation_daemon_runner.DaemonLoopHook("before", "hook: %s", lambda context: None)

    result = runner.run_configured(["--once"], hooks=(hook,))

    assert isinstance(runner, ConfiguredImplementationDaemonRunner)
    assert result == "daemon-ran"
    assert captured["argv"] == ("--once",)
    assert captured["kwargs"]["repo_root"] == repo
    assert captured["kwargs"]["logger"] == runner.logger
    assert captured["kwargs"]["default_worktree_submodule_paths"] == ("external/custom",)
    assert captured["kwargs"]["default_objective_path"] == objective_path
    assert captured["kwargs"]["default_objective_bundle_dir"] == bundle_dir
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["pass_complete_message"] == "custom complete: %s"

    runner.run_configured(["--once"], pass_complete_message="override complete: %s")
    assert captured["kwargs"]["pass_complete_message"] == "override complete: %s"

    paths = {
        "task_board_path": repo / "tasks.md",
        "state_dir": repo / "state",
        "worktree_root": repo / "worktrees",
        "objective_heap_path": objective_path,
        "objective_bundle_dir": bundle_dir,
    }

    runner.run_configured_from_paths(
        ["--once"],
        paths,
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        todo_path_flag="--task-board",
        objective_path_key="objective_heap_path",
        objective_bundle_dir_key="objective_bundle_dir",
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("external/path-default",),
        hooks=(hook,),
        pass_complete_message="paths complete: %s",
    )

    forwarded = captured["argv"]
    assert forwarded[0:2] == ("--worktree-submodule-path", "external/path-default")
    assert forwarded[forwarded.index("--task-board") + 1] == str(repo / "tasks.md")
    assert forwarded[forwarded.index("--state-dir") + 1] == str(repo / "state")
    assert forwarded[forwarded.index("--task-prefix") + 1] == "## AUTO-"
    assert forwarded[forwarded.index("--state-prefix") + 1] == "agent"
    assert forwarded[forwarded.index("--worktree-root") + 1] == str(repo / "worktrees")
    assert forwarded[forwarded.index("--objective-path") + 1] == str(objective_path)
    assert forwarded[forwarded.index("--objective-bundle-dir") + 1] == str(bundle_dir)
    assert forwarded[forwarded.index("--llm-merge-resolver-command") + 1] == "resolve-conflict"
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["pass_complete_message"] == "paths complete: %s"


def test_daemon_hooks_timeout_and_continue(tmp_path):
    events_path = tmp_path / "state" / "events.jsonl"
    parsed = argparse.Namespace(daemon_hook_timeout_seconds=0.01)
    context = implementation_daemon_runner.ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=events_path,
    )
    calls: list[str] = []
    hooks = (
        implementation_daemon_runner.DaemonLoopHook(
            "before",
            "slow hook: %s",
            lambda _context: time.sleep(1),
        ),
        implementation_daemon_runner.DaemonLoopHook(
            "before",
            "fast hook: %s",
            lambda _context: calls.append("fast") or {"ok": True},
        ),
    )

    implementation_daemon_runner._run_hooks(  # pylint: disable=protected-access
        hooks,
        phase="before",
        context=context,
        logger=logging.getLogger("test-daemon-hook-timeout"),
    )

    assert calls == ["fast"]
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "daemon_hook_timeout"
    assert events[0]["phase"] == "before"
    assert events[0]["timeout_seconds"] == 0.01


def test_supervisor_runtime_child_exit_restart_policy() -> None:
    assert child_exit_should_restart(
        exit_code=2,
        restart_count=0,
        restart_limit=3,
    )
    assert child_exit_should_restart(
        exit_code=-signal.SIGKILL,
        restart_count=2,
        restart_limit=3,
    )
    assert not child_exit_should_restart(
        exit_code=0,
        restart_count=0,
        restart_limit=3,
    )
    assert child_exit_should_restart(
        exit_code=0,
        restart_count=0,
        restart_limit=3,
        restart_on_clean_exit=True,
    )
    assert not child_exit_should_restart(
        exit_code=None,
        restart_count=0,
        restart_limit=3,
    )
    assert not child_exit_should_restart(
        exit_code=2,
        restart_count=3,
        restart_limit=3,
    )
    assert not child_exit_should_restart(
        exit_code=2,
        restart_count=0,
        restart_limit=3,
        stop_requested=True,
    )


def test_supervisor_runtime_child_success_accounting() -> None:
    assert supervised_child_succeeded(
        child_id="worker-1",
        exit_code=0,
        runner_terminated_child_ids=[],
    )
    assert not supervised_child_succeeded(
        child_id="worker-1",
        exit_code=0,
        runner_terminated_child_ids=["worker-1"],
    )
    assert supervised_child_succeeded(
        child_id="worker-1",
        exit_code=-signal.SIGTERM,
        runner_terminated_child_ids=["worker-1"],
        allow_runner_terminated=True,
    )
    assert not supervised_child_succeeded(
        child_id="worker-1",
        exit_code=-signal.SIGTERM,
        runner_terminated_child_ids=["worker-1"],
        allow_runner_terminated=True,
        stop_requested=True,
    )
    assert supervised_child_group_succeeded(
        {
            "worker-1": -signal.SIGTERM,
            "worker-2": 0,
        },
        runner_terminated_child_ids=["worker-1"],
        allow_runner_terminated=True,
    )
    assert not supervised_child_group_succeeded(
        {
            "worker-1": 2,
            "worker-2": 0,
        },
        runner_terminated_child_ids=["worker-1"],
        allow_runner_terminated=True,
    )


def test_supervisor_runtime_launch_process_child_normalizes_runtime_defaults(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    stdout_marker = object()
    stderr_marker = object()

    class FakeProcess:
        pid = 2468

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(supervisor_runtime.subprocess, "Popen", FakeProcess)
    monkeypatch.setenv("SUPERVISOR_TEST_BASE", "base")

    process = launch_process_child(
        ("python", "worker.py"),
        cwd=tmp_path,
        env={"SUPERVISOR_TEST_EXTRA": 12},
        stdin=None,
        stdout=stdout_marker,
        stderr=stderr_marker,
        start_new_session=False,
        text=True,
    )

    assert process.pid == 2468
    assert captured["command"] == ["python", "worker.py"]
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["SUPERVISOR_TEST_BASE"] == "base"
    assert kwargs["env"]["SUPERVISOR_TEST_EXTRA"] == "12"
    assert kwargs["stdin"] is None
    assert kwargs["stdout"] is stdout_marker
    assert kwargs["stderr"] is stderr_marker
    assert kwargs["start_new_session"] is False
    assert kwargs["text"] is True


def test_supervisor_runtime_launch_supervised_child_uses_shared_launcher(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    calls: list[dict[str, object]] = []

    class FakeProcess:
        pid = 3579

    def fake_launch_process_child(command, **kwargs):
        calls.append({"command": command, "kwargs": kwargs})
        return FakeProcess()

    monkeypatch.setattr(supervisor_runtime, "launch_process_child", fake_launch_process_child)

    child = launch_supervised_child(
        SupervisedChildSpec(
            repo_root=repo,
            command=("python", "worker.py"),
            log_path=Path("logs/child.log"),
            child_pid_path=Path("state/child.pid"),
            env={"SUPERVISOR_TEST_EXTRA": "1"},
            stdin_devnull=False,
            start_new_session=False,
        )
    )

    assert child.pid == 3579
    assert (repo / "state" / "child.pid").read_text(encoding="utf-8").strip() == "3579"
    assert len(calls) == 1
    assert calls[0]["command"] == ("python", "worker.py")
    kwargs = calls[0]["kwargs"]
    assert kwargs["cwd"] == str(repo)
    assert kwargs["env"] == {"SUPERVISOR_TEST_EXTRA": "1"}
    assert kwargs["stdin"] is None
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["start_new_session"] is False


def test_supervisor_runtime_adopts_matching_child_pid_marker(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    pid_path.parent.mkdir()
    pid_path.write_text("2468\n", encoding="utf-8")

    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda pid: int(pid) == 2468)
    monkeypatch.setattr(
        supervisor_runtime,
        "process_args",
        lambda pid: "python worker.py --state-dir state --implement" if int(pid) == 2468 else "",
    )

    child = adopt_supervised_child(
        SupervisedChildSpec(
            repo_root=repo,
            command=("python", "worker.py", "--state-dir", "state", "--implement"),
            log_path=Path("logs/child.log"),
            child_pid_path=Path("state/child.pid"),
        )
    )

    assert child is not None
    assert child.pid == 2468
    assert child.child_pid_path == pid_path


def test_supervisor_loop_adopts_existing_child_before_launch(tmp_path, monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_loop as supervisor_loop_module

    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=("python", "worker.py"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    adopted = SupervisedChild(
        pid=2468,
        command=("python", "worker.py"),
        log_path=state_dir / "child.log",
        child_pid_path=state_dir / "child.pid",
    )

    monkeypatch.setattr(supervisor_loop_module, "adopt_supervised_child", lambda _spec: adopted)
    monkeypatch.setattr(
        supervisor_loop_module,
        "launch_supervised_child",
        lambda _spec: pytest.fail("supervisor loop launched a duplicate child"),
    )
    monkeypatch.setattr(supervisor_loop_module, "_poll_child_exit", lambda _child: None)
    monkeypatch.setattr(supervisor_loop_module, "terminate_supervised_child", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(supervisor_loop_module, "wait_for_child_exit", lambda _child: 0)

    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=("python", "worker.py"),
            log_prefix="child",
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            watchdog_startup_grace_seconds=0,
            max_restarts=1,
        ),
        watchdog_hook=lambda _loop, child, _status: SupervisorLoopDecision.stop(
            f"adopted {child.pid}",
            status="adopted_stop",
        ),
        sleep=lambda _seconds: None,
    )

    result = loop.run()

    assert result.status == "adopted_stop"
    assert result.last_recycle_reason == "adopted 2468"


def test_supervisor_runtime_run_process_group_capture_completes(tmp_path: Path) -> None:
    result = run_process_group_capture(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        timeout_seconds=5.0,
    )

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "ok"
    assert result["stderr"] == ""


def test_supervisor_runtime_run_process_group_capture_passes_stdin(
    tmp_path: Path,
) -> None:
    result = run_process_group_capture(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        cwd=tmp_path,
        input_text="hello",
        timeout_seconds=5.0,
    )

    assert result["status"] == "completed"
    assert result["stdout"].strip() == "HELLO"


def test_supervisor_runtime_run_process_group_capture_times_out_and_kills_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 5432
        returncode = None

        def __init__(self):
            self.calls = 0

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            signals.append((self.pid, signum))

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            if self.calls < 3:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = -signal.SIGKILL
            return "late stdout", "late stderr"

    process = FakeProcess()
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_runtime.os,
        "killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )

    result = run_process_group_capture(
        ["fake"],
        cwd=tmp_path,
        timeout_seconds=1.0,
        kill_wait_seconds=0.0,
    )

    assert result["status"] == "timeout"
    assert result["exit_code"] == -signal.SIGKILL
    assert result["stdout"] == "late stdout"
    assert result["stderr"] == "late stderr"
    assert signals == [(5432, signal.SIGTERM), (5432, signal.SIGKILL)]


def test_supervisor_runtime_run_process_group_capture_reports_launch_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_launch(command, **kwargs):
        raise OSError("missing command")

    monkeypatch.setattr(supervisor_runtime, "launch_process_child", fail_launch)

    result = run_process_group_capture(
        ["missing"],
        cwd=tmp_path,
        timeout_seconds=1.0,
    )

    assert result["status"] == "failed"
    assert result["exit_code"] is None
    assert "missing command" in result["stderr"]


def test_supervisor_runtime_stop_signal_handlers_record_and_restore(
    monkeypatch,
) -> None:
    handlers: dict[int, object] = {
        signal.SIGINT: "previous-int",
        signal.SIGTERM: "previous-term",
    }
    callbacks: list[tuple[int, object]] = []

    def fake_getsignal(signum: int):
        return handlers[signum]

    def fake_signal(signum: int, handler):
        handlers[signum] = handler

    monkeypatch.setattr(supervisor_runtime.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(supervisor_runtime.signal, "signal", fake_signal)

    state = install_stop_signal_handlers(
        on_signal=lambda signum, frame: callbacks.append((signum, frame))
    )

    installed_handler = handlers[signal.SIGTERM]
    assert callable(installed_handler)
    installed_handler(signal.SIGTERM, "frame")

    assert state.stop_requested is True
    assert state.stop_signal == signal.SIGTERM
    assert state.signal_count == 1
    assert state.received_at
    assert callbacks == [(signal.SIGTERM, "frame")]

    state.restore()

    assert handlers == {
        signal.SIGINT: "previous-int",
        signal.SIGTERM: "previous-term",
    }


def test_supervisor_runtime_terminate_process_group_falls_back_to_child(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    class FakeProcess:
        pid = 1234

        def poll(self):
            return None

        def send_signal(self, signum):
            calls.append(("send_signal", self.pid, signum))

    def fake_killpg(pid, signum):
        calls.append(("killpg", pid, signum))
        raise OSError("no process group")

    monkeypatch.setattr(supervisor_runtime.os, "killpg", fake_killpg)

    assert terminate_process_group(FakeProcess(), signal.SIGTERM)
    assert calls == [
        ("killpg", 1234, signal.SIGTERM),
        ("send_signal", 1234, signal.SIGTERM),
    ]


def test_supervisor_runtime_terminate_process_with_grace_escalates(
    monkeypatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4321

        def __init__(self):
            self._exit_code = None
            self._wait_calls = 0

        def poll(self):
            return self._exit_code

        def wait(self, timeout=None):
            self._wait_calls += 1
            if self._wait_calls == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            self._exit_code = -signal.SIGKILL
            return self._exit_code

    def fake_killpg(pid, signum):
        signals.append((pid, signum))

    process = FakeProcess()
    monkeypatch.setattr(supervisor_runtime.os, "killpg", fake_killpg)

    result = terminate_process_with_grace(
        process,
        grace_seconds=0.0,
        kill_wait_seconds=0.0,
    )

    assert result.pid == 4321
    assert result.initial_exit_code is None
    assert result.final_exit_code == -signal.SIGKILL
    assert result.terminate_sent is True
    assert result.kill_sent is True
    assert result.timed_out is False
    assert signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]


def test_supervisor_runtime_terminate_processes_with_grace_signals_all_first(
    monkeypatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(self, pid: int, *, exit_code: int | None = None, waits_to_exit: int = 1):
            self.pid = pid
            self._exit_code = exit_code
            self._wait_calls = 0
            self._waits_to_exit = waits_to_exit

        def poll(self):
            return self._exit_code

        def wait(self, timeout=None):
            self._wait_calls += 1
            if self._wait_calls < self._waits_to_exit:
                raise subprocess.TimeoutExpired("fake", timeout)
            if self._exit_code is None:
                self._exit_code = -signal.SIGTERM
            return self._exit_code

    def fake_killpg(pid, signum):
        signals.append((pid, signum))

    monkeypatch.setattr(supervisor_runtime.os, "killpg", fake_killpg)

    results = terminate_processes_with_grace(
        [
            ("first", FakeProcess(1001)),
            ("second", FakeProcess(1002, waits_to_exit=2)),
            ("done", FakeProcess(1003, exit_code=0)),
        ],
        grace_seconds=0.0,
        kill_wait_seconds=0.0,
    )

    assert signals[:2] == [
        (1001, signal.SIGTERM),
        (1002, signal.SIGTERM),
    ]
    assert signals[2:] == [(1002, signal.SIGKILL)]
    assert results["first"].terminate_sent is True
    assert results["second"].kill_sent is True
    assert results["done"].terminate_sent is False


def test_supervisor_runtime_terminate_processes_with_grace_uses_shared_deadline(
    monkeypatch,
) -> None:
    clock = {"now": 100.0}
    signals: list[tuple[int, int]] = []
    waits: list[tuple[int, float | None]] = []

    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append((self.pid, timeout))
            if self.pid == 2001 and len(waits) == 1:
                clock["now"] += float(timeout or 0.0)
            raise subprocess.TimeoutExpired("fake", timeout)

    def fake_time():
        return clock["now"]

    def fake_killpg(pid, signum):
        signals.append((pid, signum))

    monkeypatch.setattr(supervisor_runtime.time, "time", fake_time)
    monkeypatch.setattr(supervisor_runtime.os, "killpg", fake_killpg)

    results = terminate_processes_with_grace(
        [
            ("first", FakeProcess(2001)),
            ("second", FakeProcess(2002)),
        ],
        grace_seconds=5.0,
        kill_wait_seconds=0.0,
    )

    assert waits[0] == (2001, pytest.approx(5.0))
    assert waits[1] == (2002, pytest.approx(0.0))
    assert signals == [
        (2001, signal.SIGTERM),
        (2002, signal.SIGTERM),
        (2001, signal.SIGKILL),
        (2002, signal.SIGKILL),
    ]
    assert results["first"].timed_out is True
    assert results["second"].timed_out is True


def test_supervisor_runtime_summarizes_child_status_files(tmp_path: Path) -> None:
    active = tmp_path / "active.summary"
    waiting = tmp_path / "waiting.summary"
    active.write_text(
        json.dumps(
            {
                "active_packet_claimed_todo_ids": ["todo-1"],
                "active_packet_phase": "executing_codex_packet",
                "codex_claimed_total": 2,
                "codex_execution_count": 1,
                "codex_scope": "bridge",
                "heartbeat_at": "2026-06-05T00:00:00+00:00",
                "latest_stop_reason": "waiting_for_program_synthesis_todos",
                "worker_id": "worker-active",
            }
        ),
        encoding="utf-8",
    )
    waiting.write_text(
        json.dumps(
            {
                "active_packet_claimed_todo_ids": [],
                "active_packet_phase": "idle",
                "codex_claimed_total": 3,
                "codex_execution_count": 4,
                "codex_scope": "bridge",
                "heartbeat_at": "2026-06-05T00:01:00+00:00",
                "latest_stop_reason": "waiting_for_program_synthesis_todos",
                "worker_id": "worker-waiting",
            }
        ),
        encoding="utf-8",
    )

    health = summarize_child_summary_files(
        [active, waiting, tmp_path / "missing.summary"],
        spec=ChildSummaryHealthSpec(
            numeric_total_fields=("codex_claimed_total", "codex_execution_count"),
            scope_field="codex_scope",
            waiting_reasons=frozenset({"waiting_for_program_synthesis_todos"}),
        ),
        stale_seconds=90.0,
        now=datetime(2026, 6, 5, 0, 2, 0, tzinfo=timezone.utc).timestamp(),
    )

    assert health["summary_count"] == 2
    assert health["active_count"] == 1
    assert health["waiting_count"] == 1
    assert health["stale_count"] == 1
    assert health["stale_child_ids"] == ["worker-active"]
    assert health["scope_counts"] == {"bridge": 2}
    assert health["latest_stop_reasons"] == {"waiting_for_program_synthesis_todos": 2}
    assert health["numeric_totals"] == {
        "codex_claimed_total": 5,
        "codex_execution_count": 5,
    }
    assert health["summary_age_seconds"] == {
        "worker-active": 120.0,
        "worker-waiting": 60.0,
    }


def test_supervisor_runtime_child_health_uses_updated_at_timestamp(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.summary"
    worker.write_text(
        json.dumps(
            {
                "active_packet_claimed_todo_ids": [],
                "active_packet_phase": "idle",
                "heartbeat_at": "2026-06-05T00:00:00+00:00",
                "latest_stop_reason": "waiting_for_program_synthesis_todos",
                "updated_at": "2026-06-05T00:01:45+00:00",
                "worker_id": "worker-updated",
            }
        ),
        encoding="utf-8",
    )

    health = summarize_child_summary_files(
        [worker],
        spec=ChildSummaryHealthSpec(
            waiting_reasons=frozenset({"waiting_for_program_synthesis_todos"}),
        ),
        stale_seconds=90.0,
        now=datetime(2026, 6, 5, 0, 2, 0, tzinfo=timezone.utc).timestamp(),
    )

    assert health["summary_age_seconds"] == {"worker-updated": 15.0}
    assert health["stale_count"] == 0


def test_supervisor_runtime_timestamp_age_seconds_accepts_iso_z() -> None:
    now = datetime(2026, 6, 5, 0, 2, 0, tzinfo=timezone.utc).timestamp()

    assert timestamp_age_seconds("2026-06-05T00:01:30Z", now=now) == pytest.approx(30.0)
    assert timestamp_age_seconds("not-a-timestamp", now=now) is None
    assert timestamp_age_seconds("", now=now) is None


def test_build_configured_merge_resolver_runner_reuses_binding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    events_path = repo / "events.jsonl"
    repo.mkdir()
    captured: dict[str, object] = {}

    def fake_run_configured_merge_resolver_cli(config, argv=None):
        captured["config"] = config
        captured["argv"] = tuple(argv or ())
        return 0

    monkeypatch.setattr(
        merge_resolver,
        "run_configured_merge_resolver_cli",
        fake_run_configured_merge_resolver_cli,
    )

    runner = build_configured_merge_resolver_runner(
        default_events_path=str(events_path),
        default_repo_root=str(repo),
        prompt_heading="Resolve test conflicts.",
        completion_rule="Keep the task blocked until validation passes.",
        extra_rules=("Preserve both sides.",),
        primary_command_env_var="TEST_PRIMARY_RESOLVER",
        description="Test merge resolver",
        missing_event_exit_code=3,
        apply_failed_exit_code=4,
    )

    parsed = runner.parse_args(["--task-id", "AUTO-001"])
    result = runner.run(["--apply", "--task-id", "AUTO-001"])

    assert isinstance(runner, ConfiguredMergeResolverRunner)
    assert parsed.task_id == "AUTO-001"
    assert parsed.events_path == events_path
    assert parsed.repo_root == repo
    assert result == 0
    assert captured["argv"] == ("--apply", "--task-id", "AUTO-001")
    config = captured["config"]
    assert config.default_events_path == events_path
    assert config.default_repo_root == repo
    assert config.prompt_heading == "Resolve test conflicts."
    assert config.completion_rule == "Keep the task blocked until validation passes."
    assert config.extra_rules == ("Preserve both sides.",)
    assert config.primary_command_env_var == "TEST_PRIMARY_RESOLVER"
    assert config.description == "Test merge resolver"
    assert config.missing_event_exit_code == 3
    assert config.apply_failed_exit_code == 4

    event = {
        "type": "merge_finished",
        "task_id": "AUTO-001",
        "attempt": 2,
        "merge_result": {
            "attempted": True,
            "merged": False,
            "branch": "implementation/auto-001",
            "target_branch": "main",
            "command": ["git", "merge", "main"],
            "reason": "conflict",
            "stdout": "merge stdout",
            "stderr": "merge stderr",
            "dirty_paths": ["src/conflicted.py"],
        },
    }
    prompt = runner.build_merge_prompt()(event=event, repo_root=repo)
    assert "Resolve test conflicts." in prompt
    assert "Keep the task blocked until validation passes." in prompt
    assert "Preserve both sides." in prompt
    assert "AUTO-001" in prompt

    events_path.write_text(json.dumps(event) + "\n")
    payload = runner.resolver_payload()(events_path=events_path, repo_root=repo, task_id="AUTO-001")
    assert payload["found"] is True
    assert payload["task_id"] == "AUTO-001"
    assert payload["prompt"].startswith("Resolve test conflicts.")
    assert "Preserve both sides." in payload["prompt"]

    monkeypatch.delenv("TEST_PRIMARY_RESOLVER", raising=False)
    monkeypatch.delenv(merge_resolver.LLM_MERGE_RESOLVER_COMMAND_ENV, raising=False)
    missing = runner.llm_resolver_invoker()({"repo_root": str(repo), "prompt": "prompt"})
    assert missing["applied"] is False
    assert "TEST_PRIMARY_RESOLVER" in missing["apply_error"]

    def fake_invoke_llm_resolver(payload, *, command_template=None, timeout_seconds=None):
        captured["command_template"] = command_template
        captured["timeout_seconds"] = timeout_seconds
        return {**payload, "applied": True}

    monkeypatch.setattr(merge_resolver, "invoke_llm_resolver", fake_invoke_llm_resolver)
    monkeypatch.setenv("TEST_PRIMARY_RESOLVER", "resolver --apply")
    applied = runner.llm_resolver_invoker()({"repo_root": str(repo), "prompt": "prompt"}, timeout_seconds=12.0)
    assert applied["applied"] is True
    assert captured["command_template"] == "resolver --apply"
    assert captured["timeout_seconds"] == 12.0


def test_build_namespace_merge_resolver_runner_from_spec_uses_namespace_defaults(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    resolver_spec = MergeResolverNamespaceSpec(
        namespace="agent_supervisor",
        env_prefix="AGENT",
        prompt_heading="Resolve agent conflict.",
        completion_rule="Keep blocked until validation passes.",
        extra_rules=("Preserve both branches.",),
        missing_event_exit_code=5,
        apply_failed_exit_code=6,
    )

    runner = build_namespace_merge_resolver_runner_from_spec(
        repo_root=repo,
        resolver_spec=resolver_spec,
    )

    assert isinstance(runner, ConfiguredMergeResolverRunner)
    config = runner.config
    assert config.default_events_path == repo / "data" / "agent_supervisor" / "state" / "agent_supervisor_events.jsonl"
    assert config.default_repo_root == repo
    assert config.prompt_heading == "Resolve agent conflict."
    assert config.completion_rule == "Keep blocked until validation passes."
    assert config.extra_rules == ("Preserve both branches.",)
    assert config.primary_command_env_var == "AGENT_LLM_MERGE_RESOLVER_COMMAND"
    assert config.missing_event_exit_code == 5
    assert config.apply_failed_exit_code == 6


def test_build_implementation_supervisor_defaults_from_paths(tmp_path):
    paths = {
        "task_board_path": tmp_path / "tasks.md",
        "state_dir": tmp_path / "state",
        "worktree_root": tmp_path / "worktrees",
    }
    daemon_script = tmp_path / "daemon.py"
    supervisor_script = tmp_path / "supervisor.py"

    defaults = build_implementation_supervisor_defaults_from_paths(
        paths,
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        daemon_script_path=daemon_script,
        supervisor_script_path=supervisor_script,
        todo_path_flag="--task-board",
        max_restarts=3,
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    )

    assert defaults == ImplementationSupervisorDefaults(
        todo_path=tmp_path / "tasks.md",
        state_dir=tmp_path / "state",
        task_prefix="## AUTO-",
        state_prefix="agent",
        worktree_root=tmp_path / "worktrees",
        daemon_script_path=daemon_script,
        supervisor_script_path=supervisor_script,
        todo_path_flag="--task-board",
        max_restarts=3,
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    )
    assert apply_portal_implementation_supervisor_defaults_from_paths(
        ["--once"],
        paths,
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        daemon_script_path=daemon_script,
        supervisor_script_path=supervisor_script,
        todo_path_flag="--task-board",
        max_restarts=3,
        llm_merge_resolver_command="resolve-conflict",
        worktree_submodule_paths=("packages/app", "external/lib"),
    ) == apply_portal_implementation_supervisor_defaults(["--once"], defaults=defaults)


def test_build_refill_defaults_from_paths(tmp_path):
    paths = {
        "objective_path": tmp_path / "objective-heap.md",
        "objective_graph_path": tmp_path / "objective-graph.json",
        "objective_bundle_dir": tmp_path / "objective-bundles",
        "objective_dataset_dir": tmp_path / "objective-datasets",
        "discovery_dir": tmp_path / "discovery",
        "objective_todo_vector_index_path": tmp_path / "todo-vector-index.json",
    }

    objective = build_objective_refill_defaults_from_paths(
        paths,
        objective_path_key="objective_path",
        objective_graph_path_key="objective_graph_path",
        objective_bundle_dir_key="objective_bundle_dir",
        objective_dataset_dir_key="objective_dataset_dir",
        objective_discovery_dir_key="discovery_dir",
        objective_discovery_output_path="data/discovery",
        objective_scan_min_open_tasks=9,
        objective_scan_max_findings=4,
        objective_scan_cooldown_seconds=60,
        objective_todo_vector_index_path_key="objective_todo_vector_index_path",
        objective_surplus_findings_per_goal=7,
        objective_surplus_min_terms_per_todo=5,
        objective_interoperability_focus=("hallucinate_app",),
        objective_interoperability_component_paths=("hallucinate_app", "swissknife"),
        seed_interoperability_goals=True,
    )
    codebase = build_codebase_refill_defaults_from_paths(
        paths,
        codebase_scan_discovery_dir_key="discovery_dir",
        codebase_scan_discovery_output_path="data/discovery",
        codebase_scan_min_open_tasks=3,
        codebase_scan_max_findings=8,
        codebase_scan_cooldown_seconds=120,
        codebase_refill_timeout_seconds=601,
        codebase_scan_skip_prefixes=("data/", "scripts/"),
    )

    assert objective == ObjectiveRefillDefaults(
        objective_path=tmp_path / "objective-heap.md",
        objective_graph_path=tmp_path / "objective-graph.json",
        objective_bundle_dir=tmp_path / "objective-bundles",
        objective_dataset_dir=tmp_path / "objective-datasets",
        objective_discovery_dir=tmp_path / "discovery",
        objective_discovery_output_path="data/discovery",
        objective_scan_min_open_tasks=9,
        objective_scan_max_findings=4,
        objective_scan_cooldown_seconds=60,
        objective_todo_vector_index_path=tmp_path / "todo-vector-index.json",
        objective_surplus_findings_per_goal=7,
        objective_surplus_min_terms_per_todo=5,
        objective_interoperability_focus=("hallucinate_app",),
        objective_interoperability_component_paths=("hallucinate_app", "swissknife"),
        seed_interoperability_goals=True,
    )
    assert codebase == CodebaseRefillDefaults(
        codebase_scan_discovery_dir=tmp_path / "discovery",
        codebase_scan_discovery_output_path="data/discovery",
        codebase_scan_min_open_tasks=3,
        codebase_scan_max_findings=8,
        codebase_scan_cooldown_seconds=120,
        codebase_refill_timeout_seconds=601,
        codebase_scan_skip_prefixes=("data/", "scripts/"),
    )


def test_build_namespace_refill_defaults_factories(tmp_path):
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")

    objective_factory = build_namespace_objective_refill_defaults_factory(
        namespace_paths,
        objective_path=tmp_path / "objective-heap.md",
        objective_discovery_output_path="data/agent_supervisor/discovery",
        objective_scan_min_open_tasks=7,
        objective_scan_max_findings=3,
        objective_scan_cooldown_seconds=90,
        objective_surplus_findings_per_goal=6,
        objective_surplus_min_terms_per_todo=4,
        objective_interoperability_focus=("hallucinate_app",),
        objective_interoperability_component_paths=("hallucinate_app", "swissknife"),
        seed_interoperability_goals=True,
    )
    codebase_factory = build_namespace_codebase_refill_defaults_factory(
        namespace_paths,
        codebase_scan_discovery_output_path="data/agent_supervisor/discovery",
        codebase_scan_min_open_tasks=2,
        codebase_scan_max_findings=5,
        codebase_scan_cooldown_seconds=180,
        codebase_refill_timeout_seconds=602,
        codebase_scan_skip_prefixes=("data/agent_supervisor/state/",),
    )

    assert objective_factory({}) == ObjectiveRefillDefaults(
        objective_path=tmp_path / "objective-heap.md",
        objective_graph_path=namespace_paths.objective_graph_path,
        objective_bundle_dir=namespace_paths.objective_bundle_dir,
        objective_dataset_dir=namespace_paths.objective_dataset_dir,
        objective_discovery_dir=namespace_paths.discovery_dir,
        objective_discovery_output_path="data/agent_supervisor/discovery",
        objective_scan_min_open_tasks=7,
        objective_scan_max_findings=3,
        objective_scan_cooldown_seconds=90,
        objective_todo_vector_index_path=namespace_paths.objective_todo_vector_index_path,
        objective_surplus_findings_per_goal=6,
        objective_surplus_min_terms_per_todo=4,
        objective_interoperability_focus=("hallucinate_app",),
        objective_interoperability_component_paths=("hallucinate_app", "swissknife"),
        seed_interoperability_goals=True,
    )
    assert codebase_factory({}) == CodebaseRefillDefaults(
        codebase_scan_discovery_dir=namespace_paths.discovery_dir,
        codebase_scan_discovery_output_path="data/agent_supervisor/discovery",
        codebase_scan_min_open_tasks=2,
        codebase_scan_max_findings=5,
        codebase_scan_cooldown_seconds=180,
        codebase_refill_timeout_seconds=602,
        codebase_scan_skip_prefixes=("data/agent_supervisor/state/",),
    )

    resolved_paths = {
        "objective_heap_path": tmp_path / "custom-objective.md",
        "objective_graph_path": tmp_path / "custom-graph.json",
        "objective_bundle_dir": tmp_path / "custom-bundles",
        "objective_dataset_dir": tmp_path / "custom-datasets",
        "objective_todo_vector_index_path": tmp_path / "custom-vector.json",
        "discovery_dir": tmp_path / "custom-discovery",
    }
    keyed_objective_factory = build_namespace_objective_refill_defaults_factory(
        namespace_paths,
        objective_path_key="objective_heap_path",
        use_bootstrap_keys=True,
        objective_discovery_output_path_factory=lambda _paths: "custom/discovery",
    )
    keyed_codebase_factory = build_namespace_codebase_refill_defaults_factory(
        namespace_paths,
        use_bootstrap_keys=True,
        codebase_scan_discovery_output_path_factory=lambda _paths: "custom/discovery",
    )

    assert keyed_objective_factory(resolved_paths) == ObjectiveRefillDefaults(
        objective_path=resolved_paths["objective_heap_path"],
        objective_graph_path=resolved_paths["objective_graph_path"],
        objective_bundle_dir=resolved_paths["objective_bundle_dir"],
        objective_dataset_dir=resolved_paths["objective_dataset_dir"],
        objective_discovery_dir=resolved_paths["discovery_dir"],
        objective_discovery_output_path="custom/discovery",
        objective_todo_vector_index_path=resolved_paths["objective_todo_vector_index_path"],
    )
    assert keyed_codebase_factory(resolved_paths) == CodebaseRefillDefaults(
        codebase_scan_discovery_dir=resolved_paths["discovery_dir"],
        codebase_scan_discovery_output_path="custom/discovery",
    )


def test_standard_task_proposal_requested_outputs_builds_shared_checklist():
    assert standard_task_proposal_requested_outputs("runtime contracts to add") == (
        "exact files to edit",
        "runtime contracts to add",
        "tests and fixtures needed",
        "validation commands",
        "risks or blockers",
    )
    assert standard_task_proposal_requested_outputs(
        "data contracts or APIs to add",
        test_output="mocks/fixtures/tests needed to run without hardware",
    ) == (
        "exact files to edit",
        "data contracts or APIs to add",
        "mocks/fixtures/tests needed to run without hardware",
        "validation commands",
        "risks or blockers",
    )


def test_run_configured_task_proposal_router_cli_dry_run(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    task_board = repo / "tasks.md"
    plan_path = repo / "plan.md"
    artifact_dir = repo / "artifacts"
    task_board.write_text(
        """# Tasks

## AUTO-001 Build reusable router

- Status: todo
- Priority: P1
- Track: ops
- Outputs: src/router.py
- Validation: pytest tests/test_router.py
- Acceptance: Router preflight returns JSON.
""",
        encoding="utf-8",
    )
    plan_path.write_text("Build reusable routing helpers.\n", encoding="utf-8")

    result = run_configured_task_proposal_router_cli(
        ["--task-id", "AUTO-001"],
        repo_root=repo,
        task_board_path=task_board,
        task_header_prefix="## AUTO-",
        plan_path=plan_path,
        artifact_dir=artifact_dir,
        prompt_intro="Help implement the test roadmap.",
        requested_outputs=("files", "tests"),
        description="Test proposal router",
        task_id_help="Task id",
        bootstrap=lambda: print("bootstrap noise"),
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == "bootstrap noise\n"
    payload = json.loads(captured.out)
    assert payload["task_id"] == "AUTO-001"
    assert payload["generate"] is False
    assert payload["llm_router_importable"] is True


def test_build_configured_task_proposal_router_runner_reuses_binding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    task_board = repo / "tasks.md"
    plan_path = repo / "plan.md"
    artifact_dir = repo / "artifacts"
    repo.mkdir()
    captured: dict[str, object] = {}
    bootstrapped: list[str] = []

    def fake_run_task_proposal_router_cli(config, argv=None):
        captured["config"] = config
        captured["argv"] = tuple(argv or ())
        return 0

    monkeypatch.setattr(
        task_proposal_router,
        "run_task_proposal_router_cli",
        fake_run_task_proposal_router_cli,
    )

    runner = build_configured_task_proposal_router_runner(
        repo_root=repo,
        task_board_path=task_board,
        task_header_prefix="## AUTO-",
        plan_path=plan_path,
        artifact_dir=artifact_dir,
        prompt_intro="Help implement the test roadmap.",
        requested_outputs=("files", "tests"),
        no_open_task_message="No open tasks.",
        description="Test proposal router",
        task_id_help="Task id",
        include_dry_run_flag=True,
        bootstrap=lambda: bootstrapped.append("bootstrapped"),
    )

    result = runner.run(["--task-id", "AUTO-001"])

    assert isinstance(runner, ConfiguredTaskProposalRouterRunner)
    assert result == 0
    assert captured["argv"] == ("--task-id", "AUTO-001")
    config = captured["config"]
    assert config.router_config.repo_root == repo
    assert config.router_config.task_board_path == task_board
    assert config.router_config.task_header_prefix == "## AUTO-"
    assert config.router_config.plan_path == plan_path
    assert config.router_config.artifact_dir == artifact_dir
    assert config.router_config.no_open_task_message == "No open tasks."
    assert config.description == "Test proposal router"
    assert config.task_id_help == "Task id"
    assert config.include_dry_run_flag is True
    config.bootstrap()
    assert bootstrapped == ["bootstrapped"]


def test_build_repo_task_proposal_router_runner_uses_repo_runtime_bootstrap(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    task_board = repo / "tasks.md"
    plan_path = repo / "plan.md"
    artifact_dir = repo / "artifacts"
    repo.mkdir()
    captured: dict[str, object] = {}
    bootstrapped: list[str] = []

    class RuntimeCallbacks:
        def enter(self):
            bootstrapped.append("entered")

    def fake_build_repo_runtime_environment_callbacks(repo_root, package_names, **kwargs):
        captured["repo_root"] = repo_root
        captured["package_names"] = tuple(package_names)
        captured["kwargs"] = kwargs
        return RuntimeCallbacks()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.wrapper_utils.build_repo_runtime_environment_callbacks",
        fake_build_repo_runtime_environment_callbacks,
    )

    runner = build_repo_task_proposal_router_runner(
        repo_root=repo,
        task_board_path=task_board,
        task_header_prefix="## AUTO-",
        plan_path=plan_path,
        artifact_dir=artifact_dir,
        prompt_intro="Help implement the test roadmap.",
        requested_outputs=("files", "tests"),
        description="Test repo proposal router",
        task_id_help="Task id",
        runtime_package_names=("pkg_a", "pkg_b"),
        runtime_external_dir="vendor",
        runtime_primary_package_names=("pkg_a",),
        runtime_env_var="TEST_PYTHONPATH",
    )

    assert isinstance(runner, ConfiguredTaskProposalRouterRunner)
    assert captured["repo_root"] == repo
    assert captured["package_names"] == ("pkg_a", "pkg_b")
    assert captured["kwargs"] == {
        "external_dir": "vendor",
        "primary_package_names": ("pkg_a",),
        "env_var": "TEST_PYTHONPATH",
    }
    assert runner.config.bootstrap is not None
    runner.config.bootstrap()
    assert bootstrapped == ["entered"]


def test_build_repo_task_proposal_route_runner_derives_paths_and_outputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    runner = build_repo_task_proposal_route_runner(
        repo_root=repo,
        task_board_stem="ROADMAP",
        task_board_dir="docs",
        artifact_namespace="agent_supervisor",
        task_header_prefix="## AUTO-",
        prompt_intro="Help implement the test roadmap.",
        domain_outputs=("runtime contracts to add",),
        description="Test routed proposal router",
        task_id_help="Task id",
        task_board_option="--task-board-path",
        hidden_standard_task_board_option=True,
        include_dry_run_flag=True,
    )

    assert isinstance(runner, ConfiguredTaskProposalRouterRunner)
    config = runner.config
    assert config.router_config.repo_root == repo
    assert config.router_config.task_board_path == repo / "docs" / "ROADMAP.todo.md"
    assert config.router_config.plan_path == repo / "docs" / "ROADMAP.md"
    assert config.router_config.artifact_dir == repo / "data" / "agent_supervisor" / "llm_router"
    assert config.router_config.task_header_prefix == "## AUTO-"
    assert config.task_board_option == "--task-board-path"
    assert config.hidden_task_board_options == ("--todo-path",)
    assert config.include_dry_run_flag is True
    prompt = config.router_config.prompt_builder(type("Task", (), {})(), "Plan text")
    assert "runtime contracts to add" in prompt
    assert "tests and fixtures needed" in prompt


def test_build_repo_task_proposal_route_runner_from_spec_reuses_route_values(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    bootstrapped: list[str] = []
    route_spec = TaskProposalRouteSpec(
        task_board_stem="ROADMAP",
        task_board_dir="docs",
        artifact_namespace="agent_supervisor",
        task_header_prefix="## AUTO-",
        prompt_intro="Help implement the test roadmap.",
        domain_outputs=("runtime contracts to add",),
        description="Test routed proposal router",
        task_id_help="Task id",
        task_board_option="--task-board-path",
        hidden_standard_task_board_option=True,
        include_dry_run_flag=True,
        bootstrap=lambda: bootstrapped.append("from-spec"),
    )

    runner = build_repo_task_proposal_route_runner_from_spec(
        repo_root=repo,
        route_spec=route_spec,
    )

    assert isinstance(runner, ConfiguredTaskProposalRouterRunner)
    config = runner.config
    assert config.router_config.repo_root == repo
    assert config.router_config.task_board_path == repo / "docs" / "ROADMAP.todo.md"
    assert config.router_config.plan_path == repo / "docs" / "ROADMAP.md"
    assert config.router_config.artifact_dir == repo / "data" / "agent_supervisor" / "llm_router"
    assert config.task_board_option == "--task-board-path"
    assert config.hidden_task_board_options == ("--todo-path",)
    assert config.include_dry_run_flag is True
    assert config.bootstrap is not None
    config.bootstrap()
    assert bootstrapped == ["from-spec"]
    prompt = config.router_config.prompt_builder(type("Task", (), {})(), "Plan text")
    assert "runtime contracts to add" in prompt


def test_build_supervisor_runtime_operations_binds_project_wrapper(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    script_path = repo / "custom_supervisor.py"
    repo.mkdir()
    state_dir.mkdir()
    script_path.write_text("# test wrapper\n", encoding="utf-8")
    captured: dict[str, object] = {}
    prepared: list[str] = []

    class FakeProcess:
        pid = 424242

    def fake_popen(command, *, cwd, env, stdin, stdout, stderr, start_new_session):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["stdin"] = stdin
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        return FakeProcess()

    monkeypatch.setattr(supervisor_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda pid: int(pid) == FakeProcess.pid)
    monkeypatch.setattr(supervisor_runtime.time, "sleep", lambda seconds: captured.setdefault("sleep", seconds))

    operations = build_supervisor_runtime_operations(
        repo_root=repo,
        script_path=script_path,
        process_predicate=lambda pid: int(pid) == FakeProcess.pid,
        prepare_environment=lambda: prepared.append("prepared"),
        startup_delay_seconds=0.25,
    )

    result = operations.ensure_running(["--once", "--flag"], state_dir=state_dir, state_prefix="agent")

    assert result["started"] is True
    assert result["pid"] == FakeProcess.pid
    assert captured["command"] == [sys.executable, str(script_path), "--implement", "--flag"]
    assert captured["cwd"] == repo
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["start_new_session"] is True
    assert captured["sleep"] == 0.25
    assert prepared == ["prepared"]
    assert (state_dir / "agent_supervisor_wrapper.pid").read_text(encoding="utf-8") == "424242\n"
    assert operations.is_running(state_dir, "agent") is True
    assert operations.repair_runtime(state_dir, "agent") == {"removed": [], "updated_status": False}


def test_build_configured_supervisor_runtime_binds_project_wrapper(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    script_path = repo / "custom_supervisor.py"
    repo.mkdir()
    state_dir.mkdir()
    script_path.write_text("# test wrapper\n", encoding="utf-8")
    captured: dict[str, object] = {}
    prepared: list[str] = []

    class FakeProcess:
        pid = 525252

    def fake_popen(command, *, cwd, env, stdin, stdout, stderr, start_new_session):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["stdin"] = stdin
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        return FakeProcess()

    monkeypatch.setattr(supervisor_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda pid: int(pid) == FakeProcess.pid)
    monkeypatch.setattr(supervisor_runtime.time, "sleep", lambda seconds: captured.setdefault("sleep", seconds))

    runtime = build_configured_supervisor_runtime(
        repo_root=repo,
        script_path=script_path,
        process_match_any=("custom_supervisor.py",),
        process_predicate=lambda pid: int(pid) == FakeProcess.pid,
        prepare_environment=lambda: prepared.append("prepared"),
        implementation_lock_name="custom.lock",
        startup_delay_seconds=0.25,
    )

    result = runtime.ensure_running(["--once", "--flag"], state_dir=state_dir, state_prefix="agent")

    assert runtime.repo_root == repo
    assert runtime.script_path == script_path
    assert runtime.process_match_any == ("custom_supervisor.py",)
    assert runtime.implementation_lock_name == "custom.lock"
    assert result["started"] is True
    assert result["pid"] == FakeProcess.pid
    assert captured["command"] == [sys.executable, str(script_path), "--implement", "--flag"]
    assert captured["cwd"] == repo
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["start_new_session"] is True
    assert captured["sleep"] == 0.25
    assert prepared == ["prepared"]
    assert (state_dir / "agent_supervisor_wrapper.pid").read_text(encoding="utf-8") == "525252\n"
    assert runtime.is_running(state_dir, "agent") is True
    assert runtime.repair_runtime(state_dir, "agent") == {"removed": [], "updated_status": False}


def test_run_configured_supervisor_with_runtime_composes_callbacks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    script_path = repo / "custom_supervisor.py"
    daemon_path = repo / "custom_daemon.py"
    repo.mkdir()
    script_path.write_text("# supervisor\n", encoding="utf-8")
    daemon_path.write_text("# daemon\n", encoding="utf-8")
    captured: dict[str, object] = {}
    prepared: list[str] = []

    class FakeRuntimeCallbacks:
        def ensure_running(self, context):
            return {"ensured": context}

        def repair_runtime(self, context):
            return {"repaired": context}

    runtime_callbacks = FakeRuntimeCallbacks()

    def fake_build_supervisor_runtime_callbacks(argv, **kwargs):
        captured["runtime_argv"] = tuple(argv)
        captured["runtime_kwargs"] = kwargs
        return runtime_callbacks

    def fake_run_configured_portal_implementation_supervisor(argv, **kwargs):
        captured["run_argv"] = tuple(argv)
        captured["run_kwargs"] = kwargs
        return kwargs["ensure_running_callback"]("ctx")

    monkeypatch.setattr(
        implementation_supervisor_runner,
        "build_supervisor_runtime_callbacks",
        fake_build_supervisor_runtime_callbacks,
    )
    monkeypatch.setattr(
        implementation_supervisor_runner,
        "run_configured_portal_implementation_supervisor",
        fake_run_configured_portal_implementation_supervisor,
    )

    result = implementation_supervisor_runner.run_configured_portal_implementation_supervisor_with_runtime(
        ["--once"],
        repo_root=repo,
        logger=logging.getLogger("test-runtime-helper"),
        script_path=script_path,
        process_match_any=("custom_supervisor.py",),
        prepare_environment=lambda: prepared.append("prepared"),
        implementation_lock_name="custom.lock",
        startup_delay_seconds=0.25,
        daemon_script_path=daemon_path,
        worktree_submodule_paths=("submodule",),
        ensure_running=True,
        repair_runtime=False,
    )

    assert result == {"ensured": "ctx"}
    assert captured["runtime_argv"] == ("--once",)
    assert captured["runtime_kwargs"]["repo_root"] == repo
    assert captured["runtime_kwargs"]["script_path"] == script_path
    assert captured["runtime_kwargs"]["process_match_any"] == ("custom_supervisor.py",)
    assert captured["runtime_kwargs"]["implementation_lock_name"] == "custom.lock"
    assert captured["runtime_kwargs"]["startup_delay_seconds"] == 0.25
    captured["runtime_kwargs"]["prepare_environment"]()
    assert prepared == ["prepared"]
    assert captured["run_argv"] == ("--once",)
    assert captured["run_kwargs"]["repo_root"] == repo
    assert captured["run_kwargs"]["daemon_script_path"] == daemon_path
    assert captured["run_kwargs"]["worktree_submodule_paths"] == ("submodule",)
    assert captured["run_kwargs"]["ensure_running"] is True
    assert captured["run_kwargs"]["ensure_running_callback"] == runtime_callbacks.ensure_running
    assert captured["run_kwargs"]["repair_runtime_callback"] is None


def test_configured_supervisor_runtime_run_configured_reuses_binding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    script_path = repo / "custom_supervisor.py"
    daemon_path = repo / "custom_daemon.py"
    repo.mkdir()
    script_path.write_text("# supervisor\n", encoding="utf-8")
    daemon_path.write_text("# daemon\n", encoding="utf-8")
    captured: dict[str, object] = {}
    prepared: list[str] = []

    def prepare_environment():
        prepared.append("prepared")

    def fake_run_configured_portal_implementation_supervisor_with_runtime(argv, **kwargs):
        captured["argv"] = tuple(argv)
        captured["kwargs"] = kwargs
        return "configured"

    monkeypatch.setattr(
        implementation_supervisor_runner,
        "run_configured_portal_implementation_supervisor_with_runtime",
        fake_run_configured_portal_implementation_supervisor_with_runtime,
    )

    runtime = build_configured_supervisor_runtime(
        repo_root=repo,
        script_path=script_path,
        process_match_any=("custom_supervisor.py",),
        prepare_environment=prepare_environment,
        implementation_lock_name="custom.lock",
        startup_delay_seconds=0.25,
    )
    hook = implementation_supervisor_runner.SupervisorRunHook("before", "hook: %s", lambda context: None)

    result = runtime.run_configured(
        ["--once"],
        logger=logging.getLogger("test-configured-runtime"),
        daemon_script_path=daemon_path,
        worktree_submodule_paths=("external/custom",),
        hooks=(hook,),
        once_complete_message="complete: %s",
        ensure_running=True,
        ensure_running_message="ensure: %s",
        repair_runtime=False,
        repair_runtime_message="repair: %s",
    )

    assert result == "configured"
    assert captured["argv"] == ("--once",)
    assert captured["kwargs"]["repo_root"] == repo
    assert captured["kwargs"]["script_path"] == script_path
    assert captured["kwargs"]["process_match_any"] == ("custom_supervisor.py",)
    assert captured["kwargs"]["prepare_environment"] is prepare_environment
    assert captured["kwargs"]["implementation_lock_name"] == "custom.lock"
    assert captured["kwargs"]["startup_delay_seconds"] == 0.25
    assert captured["kwargs"]["daemon_script_path"] == daemon_path
    assert captured["kwargs"]["worktree_submodule_paths"] == ("external/custom",)
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["once_complete_message"] == "complete: %s"
    assert captured["kwargs"]["ensure_running"] is True
    assert captured["kwargs"]["ensure_running_message"] == "ensure: %s"
    assert captured["kwargs"]["repair_runtime"] is False
    assert captured["kwargs"]["repair_runtime_message"] == "repair: %s"
    captured["kwargs"]["prepare_environment"]()
    assert prepared == ["prepared"]

    paths = {
        "task_board_path": repo / "tasks.md",
        "state_dir": repo / "state",
        "worktree_root": repo / "worktrees",
        "objective_heap_path": repo / "objective.md",
        "objective_graph_path": repo / "objective.json",
        "discovery_dir": repo / "discovery",
    }

    result = runtime.run_configured_from_paths(
        ["--once"],
        paths,
        logger=logging.getLogger("test-configured-runtime-from-paths"),
        todo_path_key="task_board_path",
        task_prefix="## AUTO-",
        state_prefix="agent",
        daemon_script_path=daemon_path,
        todo_path_flag="--task-board",
        llm_merge_resolver_command="bash resolve.sh",
        worktree_submodule_paths=("external/from-paths",),
        objective=ObjectiveRefillDefaults(
            objective_path=paths["objective_heap_path"],
            objective_graph_path=paths["objective_graph_path"],
            objective_interoperability_focus=("hallucinate_app",),
            seed_interoperability_goals=True,
        ),
        codebase=CodebaseRefillDefaults(
            codebase_scan_discovery_dir=paths["discovery_dir"],
            codebase_scan_min_open_tasks=4,
            codebase_scan_skip_prefixes=("data/state/",),
        ),
        hooks=(hook,),
        once_complete_message="paths complete: %s",
        ensure_running=True,
        ensure_running_message="paths ensure: %s",
        repair_runtime=False,
    )

    forwarded = captured["argv"]
    assert result == "configured"
    assert forwarded[forwarded.index("--task-board") + 1] == str(repo / "tasks.md")
    assert forwarded[forwarded.index("--state-dir") + 1] == str(repo / "state")
    assert forwarded[forwarded.index("--task-prefix") + 1] == "## AUTO-"
    assert forwarded[forwarded.index("--state-prefix") + 1] == "agent"
    assert forwarded[forwarded.index("--worktree-root") + 1] == str(repo / "worktrees")
    assert forwarded[forwarded.index("--daemon-script-path") + 1] == str(daemon_path)
    assert forwarded[forwarded.index("--supervisor-script-path") + 1] == str(script_path)
    assert forwarded[forwarded.index("--llm-merge-resolver-command") + 1] == "bash resolve.sh"
    assert forwarded[forwarded.index("--objective-path") + 1] == str(repo / "objective.md")
    assert forwarded[forwarded.index("--objective-graph-path") + 1] == str(repo / "objective.json")
    assert forwarded[forwarded.index("--objective-interoperability-focus") + 1] == "hallucinate_app"
    assert "--objective-seed-interoperability-goals" in forwarded
    assert forwarded[forwarded.index("--codebase-scan-discovery-dir") + 1] == str(repo / "discovery")
    assert forwarded[forwarded.index("--codebase-scan-min-open-tasks") + 1] == "4"
    assert forwarded[forwarded.index("--codebase-scan-skip-prefix") + 1] == "data/state/"
    assert captured["kwargs"]["daemon_script_path"] == daemon_path
    assert captured["kwargs"]["worktree_submodule_paths"] == ("external/from-paths",)
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["once_complete_message"] == "paths complete: %s"
    assert captured["kwargs"]["ensure_running"] is True
    assert captured["kwargs"]["ensure_running_message"] == "paths ensure: %s"
    assert captured["kwargs"]["repair_runtime"] is False


def test_supervisor_config_from_args_applies_embedding_overrides(tmp_path):
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "agent",
            "--worktree-submodule-path",
            "ignored",
            "--objective-interoperability-focus",
            "hallucinate_app,swissknife",
            "--objective-interoperability-component-path",
            "hallucinate_app, external/ipfs_accelerate",
            "--no-worktree-reconciliation",
            "--worktree-reconciliation-max-merges",
            "3",
            "--worktree-reconciliation-dry-run",
            "--no-worktree-reconciliation-preflight",
            "--no-worktree-scan-cache",
            "--worktree-scan-cache-ttl-seconds",
            "12",
            "--worktree-scan-cache-path",
            str(tmp_path / "scan-cache.json"),
            "--no-reconciliation-guardrail",
            "--reconciliation-guardrail-max-findings",
            "2",
            "--reconciliation-guardrail-discovery-output-path",
            "data/reconciliation",
        ]
    )
    config = supervisor_config_from_args(
        args,
        repo_root=tmp_path,
        daemon_script_path=tmp_path / "daemon.py",
        supervisor_script_path=tmp_path / "supervisor.py",
        worktree_submodule_paths=("submodule", "nested/path"),
    )

    assert config.todo_path == tmp_path / "board.md"
    assert config.state_path == tmp_path / "state" / "agent_task_state.json"
    assert config.strategy_path == tmp_path / "state" / "agent_strategy.json"
    assert config.events_path == tmp_path / "state" / "agent_supervisor_events.jsonl"
    assert config.repo_root == tmp_path
    assert config.daemon_script_path == tmp_path / "daemon.py"
    assert config.supervisor_script_path == tmp_path / "supervisor.py"
    assert config.worktree_submodule_paths == ("submodule", "nested/path")
    assert config.worktree_reconciliation_enabled is False
    assert config.worktree_reconciliation_max_merges == 3
    assert config.worktree_reconciliation_dry_run is True
    assert config.worktree_reconciliation_preflight_enabled is False
    assert config.worktree_scan_cache_enabled is False
    assert config.worktree_scan_cache_ttl_seconds == 12
    assert config.worktree_scan_cache_path == tmp_path / "scan-cache.json"
    assert config.reconciliation_guardrail_enabled is False
    assert config.reconciliation_guardrail_max_findings == 2
    assert config.reconciliation_guardrail_discovery_output_path == "data/reconciliation"
    assert config.objective_interoperability_focus == ("hallucinate_app", "swissknife")
    assert config.objective_interoperability_component_paths == (
        "hallucinate_app",
        "external/ipfs_accelerate",
    )


def test_supervisor_reconciliation_only_disables_producers(tmp_path):
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--reconciliation-only",
            "--llm-merge-resolver-command",
            "codex exec -",
            "--codebase-refill-scan",
            "--objective-refill-scan",
        ]
    )
    config = supervisor_config_from_args(args, repo_root=tmp_path)

    assert config.reconciliation_only is True
    assert config.implement is False
    assert config.use_ephemeral_worktree is False
    assert config.worktree_reconciliation_enabled is True
    assert config.retry_budget_guardrail_enabled is False
    assert config.dependency_guardrail_enabled is False
    assert config.reconciliation_guardrail_enabled is True
    assert config.codebase_refill_enabled is False
    assert config.objective_refill_enabled is False
    assert config.llm_merge_resolver_command == ""


def test_supervisor_reconciliation_only_can_keep_resolver_when_allowed(tmp_path):
    args = parse_implementation_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--reconciliation-only",
            "--allow-reconciliation-only-llm-resolver",
            "--llm-merge-resolver-command",
            "codex exec -",
        ]
    )
    config = supervisor_config_from_args(args, repo_root=tmp_path)

    assert config.reconciliation_only is True
    assert config.llm_merge_resolver_command == "codex exec -"


def test_multi_supervisor_runner_parses_and_runs_short_track(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import signal",
                "import sys",
                "import time",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "print('worker started', flush=True)",
                "while True:",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/supervisor.pid|state/daemon.pid",
        stamp="RUN",
    )

    assert supervisor_track_payload(track)["log_path"] == "logs/RUN.log"

    output: list[str] = []
    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.15,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        master_pid_path=tmp_path / "state" / "master.pid",
        label="test runner",
        output=output.append,
    )

    pid = int((tmp_path / "state" / "supervisor.pid").read_text(encoding="utf-8").strip())
    assert result["completed"] is True
    assert result["track_count"] == 1
    assert (tmp_path / "state" / "master.pid").read_text(encoding="utf-8").strip() == str(os.getpid())
    assert "worker started" in (tmp_path / "logs" / "RUN.log").read_text(encoding="utf-8")
    assert any("started T supervisor" in line for line in output)
    assert not pid_alive(pid)


def test_pid_alive_treats_an_unreaped_zombie_as_stopped():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 5
        proc_stat = Path(f"/proc/{process.pid}/stat")
        while time.monotonic() < deadline:
            try:
                state = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].strip()[0]
            except (OSError, IndexError):
                state = ""
            if state == "Z":
                break
            time.sleep(0.01)
        assert state == "Z"
        assert not pid_alive(process.pid)
    finally:
        process.wait(timeout=5)


def test_multi_supervisor_runner_cleans_stale_daemon_pid_marker(tmp_path):
    worker = tmp_path / "worker.py"
    stale_pid = 999_999_999
    worker.write_text(
        "\n".join(
            [
                "import signal",
                "import sys",
                "import time",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "Path('state').mkdir(exist_ok=True)",
                f"Path('state/daemon.pid').write_text('{stale_pid}\\n', encoding='utf-8')",
                "while True:",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/supervisor.pid|state/daemon.pid",
        stamp="RUN",
    )

    output: list[str] = []
    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.15,
        heartbeat_interval_seconds=0.05,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    heartbeat_lines = [line for line in output if "heartbeat T" in line]
    assert result["completed"] is True
    assert heartbeat_lines
    assert any("daemon_pid=unknown" in line for line in heartbeat_lines)
    assert any(f"stale_daemon_pid={stale_pid}" in line for line in heartbeat_lines)
    assert any("daemon_status=stale" in line for line in heartbeat_lines)
    assert any("removed_stale_daemon_pid_file=true" in line for line in heartbeat_lines)
    assert not (tmp_path / "state" / "daemon.pid").exists()


def test_multi_supervisor_runner_restarts_stale_idle_supervisor_status(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json",
                "import signal",
                "import sys",
                "import time",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "Path('state').mkdir(exist_ok=True)",
                "Path('state/task_state.json').write_text(",
                "    json.dumps({'active_task_id': '', 'implementation_in_progress': False}),",
                "    encoding='utf-8',",
                ")",
                "Path('state/example_supervisor_status.json').write_text(",
                "    json.dumps({'updated_at': '2000-01-01T00:00:00+00:00', 'current_status_path': 'state/task_state.json'}),",
                "    encoding='utf-8',",
                ")",
                "while True:",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    track = parse_track_spec(
        "T|worker.py|logs/{stamp}.log|state/example_supervisor.pid|state/example_managed_daemon.pid",
        stamp="RUN",
    )

    output: list[str] = []
    result = run_supervisor_tracks(
        [track],
        repo_root=tmp_path,
        common_args=[],
        duration_seconds=0.25,
        heartbeat_interval_seconds=0.05,
        supervisor_status_stale_seconds=0.01,
        stop_grace_seconds=0.2,
        python_executable=sys.executable,
        label="test runner",
        output=output.append,
    )

    assert result["completed"] is True
    assert sum("started T supervisor" in line for line in output) >= 2
    assert any("supervisor_status=stale" in line for line in output)
    assert any("restart_supervisor=true" in line for line in output)
    assert any("restarting stale T supervisor" in line for line in output)


def test_implementation_supervisor_track_spec_uses_standard_state_layout():
    namespace_paths = agent_supervisor_namespace_paths(Path("/repo"), "virtual_ai_os")
    config = ImplementationSupervisorTrackConfig(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        state_dir="data/virtual_ai_os/state",
        state_prefix="virtual_ai_os",
    )
    namespace_config = implementation_supervisor_namespace_track_config(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        namespace_paths=namespace_paths,
    )
    overridden_namespace_config = implementation_supervisor_namespace_track_config(
        name="MGW",
        script_path="scripts/meta_glasses_display_todo_supervisor.py",
        namespace_paths=agent_supervisor_namespace_paths(Path("/repo"), "meta_glasses_display_widgets"),
        state_prefix="meta_glasses_display",
    )
    compact_spec = implementation_supervisor_compact_track_spec(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        state_dir="data/virtual_ai_os/state",
        state_prefix="virtual_ai_os",
    )
    compact_specs = implementation_supervisor_compact_track_specs(
        (
            config,
            (
                "MGW",
                "scripts/meta_glasses_display_todo_supervisor.py",
                "data/meta_glasses_display_widgets/state",
                "meta_glasses_display",
            ),
        )
    )
    spec = implementation_supervisor_track_spec(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        state_dir="data/virtual_ai_os/state",
        state_prefix="virtual_ai_os",
    )
    track = parse_implementation_track_spec(
        compact_spec,
        stamp="RUN",
    )

    assert compact_spec == "VAI|scripts/virtual_ai_os_todo_supervisor.py|data/virtual_ai_os/state|virtual_ai_os"
    assert config.compact_spec() == compact_spec
    assert namespace_config == ImplementationSupervisorTrackConfig(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        state_dir=Path("/repo/data/virtual_ai_os/state"),
        state_prefix="virtual_ai_os",
    )
    assert overridden_namespace_config.compact_spec() == (
        "MGW|scripts/meta_glasses_display_todo_supervisor.py|"
        "/repo/data/meta_glasses_display_widgets/state|meta_glasses_display"
    )
    assert config.track_spec() == spec
    assert compact_specs == (
        compact_spec,
        "MGW|scripts/meta_glasses_display_todo_supervisor.py|data/meta_glasses_display_widgets/state|"
        "meta_glasses_display",
    )
    assert spec == (
        "VAI|scripts/virtual_ai_os_todo_supervisor.py|"
        "data/virtual_ai_os/state/virtual_ai_os_8h_run_{stamp}.log|"
        "data/virtual_ai_os/state/virtual_ai_os_supervisor.pid|"
        "data/virtual_ai_os/state/virtual_ai_os_managed_daemon.pid"
    )
    assert supervisor_track_payload(track) == {
        "name": "VAI",
        "script_path": "scripts/virtual_ai_os_todo_supervisor.py",
        "log_path": "data/virtual_ai_os/state/virtual_ai_os_8h_run_RUN.log",
        "supervisor_pid_path": "data/virtual_ai_os/state/virtual_ai_os_supervisor.pid",
        "daemon_pid_path": "data/virtual_ai_os/state/virtual_ai_os_managed_daemon.pid",
    }
    lanes = multi_supervisor_runner.expand_implementation_track_lanes(
        compact_spec,
        stamp="RUN",
        lanes_per_track=2,
    )
    assert [lane.name for lane in lanes] == ["VAI-0", "VAI-1"]
    assert lanes[0].log_path == Path("data/virtual_ai_os/state/lane-0/virtual_ai_os_lane_0_8h_run_RUN.log")
    assert lanes[1].supervisor_pid_path == Path(
        "data/virtual_ai_os/state/lane-1/virtual_ai_os_lane_1_supervisor.pid"
    )
    assert lanes[0].extra_args == (
        "--state-dir",
        "data/virtual_ai_os/state/lane-0",
        "--state-prefix",
        "virtual_ai_os_lane_0",
        "--task-shard-count",
        "2",
        "--task-shard-index",
        "0",
    )
    assert lanes[1].extra_args == (
        "--state-dir",
        "data/virtual_ai_os/state/lane-1",
        "--state-prefix",
        "virtual_ai_os_lane_1",
        "--task-shard-count",
        "2",
        "--task-shard-index",
        "1",
    )


def test_implementation_supervisor_namespace_track_configs_builds_multiple_repo_tracks():
    configs = implementation_supervisor_namespace_track_configs(
        repo_root=Path("/repo"),
        track_specs=(
            ("VAI", "scripts/virtual_ai_os_todo_supervisor.py", "virtual_ai_os"),
            (
                "MGW",
                "scripts/meta_glasses_display_todo_supervisor.py",
                "meta_glasses_display_widgets",
                "meta_glasses_display",
            ),
            ImplementationSupervisorNamespaceTrackSpec(
                name="HAO",
                script_path="scripts/hallucinate_multimodal_control_todo_supervisor.py",
                namespace="hallucinate_multimodal_control",
            ),
        ),
    )

    assert configs == (
        ImplementationSupervisorTrackConfig(
            name="VAI",
            script_path="scripts/virtual_ai_os_todo_supervisor.py",
            state_dir=Path("/repo/data/virtual_ai_os/state"),
            state_prefix="virtual_ai_os",
        ),
        ImplementationSupervisorTrackConfig(
            name="MGW",
            script_path="scripts/meta_glasses_display_todo_supervisor.py",
            state_dir=Path("/repo/data/meta_glasses_display_widgets/state"),
            state_prefix="meta_glasses_display",
        ),
        ImplementationSupervisorTrackConfig(
            name="HAO",
            script_path="scripts/hallucinate_multimodal_control_todo_supervisor.py",
            state_dir=Path("/repo/data/hallucinate_multimodal_control/state"),
            state_prefix="hallucinate_multimodal_control",
        ),
    )


def test_implementation_supervisor_common_args_include_long_run_defaults():
    args = implementation_supervisor_common_args(
        implementation_command="bash resolve.sh",
        objective_scan_min_open_tasks=21,
        objective_scan_max_findings=13,
        objective_refill_timeout_seconds=602,
        objective_surplus_findings_per_goal=8,
        objective_surplus_min_terms_per_todo=5,
        codebase_refill_timeout_seconds=603,
    )

    assert args[:3] == ["--implement", "--objective-refill-scan", "--codebase-refill-scan"]
    assert args[args.index("--implementation-command") + 1] == "bash resolve.sh"
    assert args[args.index("--llm-merge-resolver-command") + 1] == "bash resolve.sh"
    assert args[args.index("--objective-scan-min-open-tasks") + 1] == "21"
    assert args[args.index("--objective-scan-max-findings") + 1] == "13"
    assert args[args.index("--objective-refill-timeout-seconds") + 1] == "602"
    assert args[args.index("--objective-surplus-findings-per-goal") + 1] == "8"
    assert args[args.index("--objective-surplus-min-terms-per-todo") + 1] == "5"
    assert args[args.index("--codebase-scan-cooldown-seconds") + 1] == "900"
    assert args[args.index("--codebase-refill-timeout-seconds") + 1] == "603"


def test_implementation_multi_supervisor_env_defaults_are_reusable():
    assert implementation_multi_supervisor_env_defaults() == {
        "PYTHONUNBUFFERED": "1",
        "CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS": "60",
        "COPILOT_MERGE_RESOLVER_TIMEOUT_SECONDS": "60",
    }
    assert implementation_multi_supervisor_env_defaults(
        python_unbuffered=False,
        codex_merge_resolver_timeout_seconds=0,
        prefer_copilot_merge_resolver=True,
    ) == {
        "PYTHONUNBUFFERED": "0",
        "CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS": "0",
        "COPILOT_MERGE_RESOLVER_TIMEOUT_SECONDS": "60",
        "PREFER_COPILOT_MERGE_RESOLVER": "1",
    }


def test_multi_supervisor_common_args_from_parsed_defaults(monkeypatch):
    monkeypatch.setenv("OBJECTIVE_SCAN_MIN_OPEN_TASKS", "22")
    parsed = build_multi_supervisor_arg_parser().parse_args(
        [
            "--implementation-track",
            "T|worker.py|state|agent",
            "--implementation-supervisor-defaults",
            "--implementation-supervisor-command",
            "bash resolve.sh",
            "--common-arg=--objective-scan-min-open-tasks",
            "--common-arg=99",
        ]
    )

    args = common_args_from_parsed_args(parsed)
    tracks = tracks_from_parsed_args(parsed)

    assert "--objective-refill-scan" in args
    assert "--codebase-refill-scan" in args
    assert args[args.index("--implementation-command") + 1] == "bash resolve.sh"
    assert args[args.index("--llm-merge-resolver-command") + 1] == "bash resolve.sh"
    assert args[args.index("--objective-scan-min-open-tasks") + 1] == "22"
    assert args[-2:] == ["--objective-scan-min-open-tasks", "99"]
    assert supervisor_track_payload(tracks[0])["log_path"].endswith("/agent_8h_run_" + parsed.stamp + ".log")


def test_configured_multi_supervisor_cli_runner_builds_base_args_and_dispatches(tmp_path, monkeypatch):
    captured: dict[str, tuple[str, ...]] = {}

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(multi_supervisor_runner, "main", fake_main)
    implementation_track = "VAI|scripts/vai.py|data/vai/state|vai"
    implementation_track_config = ImplementationSupervisorTrackConfig(
        name="MGW",
        script_path="scripts/mgw.py",
        state_dir="data/mgw/state",
        state_prefix="mgw",
    )
    raw_track = "RAW|scripts/raw.py|logs/raw.log|state/raw_supervisor.pid|state/raw_daemon.pid"

    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        duration_seconds=12,
        heartbeat_interval_seconds=0.5,
        stop_grace_seconds=0.25,
        stamp="RUN",
        master_dir="data/agent_supervisor",
        master_log="logs/master.log",
        master_pid_path="state/master.pid",
        label="test multi",
        python_executable=sys.executable,
        implementation_supervisor_defaults=True,
        implementation_supervisor_command="bash resolve.sh",
        implementation_supervisor_llm_merge_resolver_command="bash merge-resolve.sh",
        implementation_tracks=(implementation_track,),
        implementation_track_configs=(implementation_track_config,),
        tracks=(raw_track,),
        common_args=("--extra", "value"),
        detach=True,
    )

    args = runner.args()
    assert isinstance(runner, ConfiguredMultiSupervisorCliRunner)
    assert args[args.index("--repo-root") + 1] == str(tmp_path)
    assert args[args.index("--heartbeat-interval-seconds") + 1] == "0.5"
    assert args[args.index("--stop-grace-seconds") + 1] == "0.25"
    assert args[args.index("--master-log") + 1] == "logs/master.log"
    assert args[args.index("--master-pid-path") + 1] == "state/master.pid"
    assert args[args.index("--python-executable") + 1] == sys.executable
    assert args[args.index("--implementation-supervisor-command") + 1] == "bash resolve.sh"
    assert (
        args[args.index("--implementation-supervisor-llm-merge-resolver-command") + 1]
        == "bash merge-resolve.sh"
    )
    implementation_track_values = [
        args[index + 1] for index, value in enumerate(args) if value == "--implementation-track"
    ]
    assert implementation_track in implementation_track_values
    assert "MGW|scripts/mgw.py|data/mgw/state|mgw" in implementation_track_values
    assert args[args.index("--track") + 1] == raw_track
    assert "--implementation-supervisor-defaults" in args
    assert args[-1] == "--detach"

    assert runner.run(["--duration-seconds", "0.01"]) == 0
    assert captured["argv"][-2:] == ("--duration-seconds", "0.01")
    monkeypatch.setattr(sys, "argv", ["multi-supervisor", "--duration-seconds", "0.02"])
    assert runner.run_cli() == 0
    assert captured["argv"][-2:] == ("--duration-seconds", "0.02")


def test_configured_multi_supervisor_cli_runner_can_read_launch_settings_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_SUPERVISOR_TEST_DURATION", "34")
    monkeypatch.setenv("MULTI_SUPERVISOR_TEST_STAMP", "ENVSTAMP")

    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=tmp_path,
        duration_seconds=12,
        duration_seconds_env_var="MULTI_SUPERVISOR_TEST_DURATION",
        stamp="DEFAULTSTAMP",
        stamp_env_var="MULTI_SUPERVISOR_TEST_STAMP",
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="VAI",
                script_path="scripts/vai.py",
                state_dir="data/vai/state",
                state_prefix="vai",
            ),
        ),
    )

    args = runner.args()
    assert args[args.index("--duration-seconds") + 1] == "34"
    assert args[args.index("--stamp") + 1] == "ENVSTAMP"


def test_configured_multi_supervisor_launcher_prepares_environment(tmp_path, monkeypatch):
    captured: dict[str, tuple[str, ...]] = {}
    prepared: list[str] = []

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(multi_supervisor_runner, "main", fake_main)
    monkeypatch.delenv("MULTI_SUPERVISOR_TEST_DEFAULT", raising=False)

    launcher = build_configured_multi_supervisor_launcher(
        repo_root=tmp_path,
        duration_seconds=12,
        stamp="RUN",
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="VAI",
                script_path="scripts/vai.py",
                state_dir="data/vai/state",
                state_prefix="vai",
            ),
        ),
        env_defaults={"MULTI_SUPERVISOR_TEST_DEFAULT": "1"},
        prepare_environment=lambda: prepared.append("called"),
    )

    assert isinstance(launcher, ConfiguredMultiSupervisorLauncher)
    assert "--implementation-track" in launcher.args()
    assert launcher.run(["--duration-seconds", "0.01"]) == 0
    assert os.environ["MULTI_SUPERVISOR_TEST_DEFAULT"] == "1"
    assert prepared == ["called"]
    assert captured["argv"][-2:] == ("--duration-seconds", "0.01")
    monkeypatch.setattr(sys, "argv", ["multi-supervisor", "--duration-seconds", "0.02"])
    assert launcher.run_cli() == 0
    assert prepared == ["called", "called"]
    assert captured["argv"][-2:] == ("--duration-seconds", "0.02")


def test_repo_implementation_multi_supervisor_launcher_uses_repo_defaults(tmp_path, monkeypatch):
    captured: dict[str, tuple[str, ...]] = {}
    prepared: list[str] = []

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(multi_supervisor_runner, "main", fake_main)
    monkeypatch.delenv("MULTI_SUPERVISOR_REPO_DEFAULT", raising=False)

    launcher = build_repo_implementation_multi_supervisor_launcher(
        repo_root=tmp_path,
        duration_seconds=12,
        stamp="RUN",
        resolver_script_path="scripts/resolve.sh",
        label="repo implementation run",
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="VAI",
                script_path="scripts/vai.py",
                state_dir="data/vai/state",
                state_prefix="vai",
            ),
        ),
        common_args=("--flag", "value"),
        env_defaults={"MULTI_SUPERVISOR_REPO_DEFAULT": "1"},
        prepare_environment=lambda: prepared.append("called"),
    )

    args = launcher.args()
    assert isinstance(launcher, ConfiguredMultiSupervisorLauncher)
    assert args[args.index("--label") + 1] == "repo implementation run"
    assert "--implementation-supervisor-command" not in args
    assert args[args.index("--implementation-supervisor-llm-merge-resolver-command") + 1] == (
        f"bash {tmp_path / 'scripts' / 'resolve.sh'}"
    )
    assert "--implementation-supervisor-defaults" in args
    assert args[args.index("--implementation-track") + 1] == "VAI|scripts/vai.py|data/vai/state|vai"
    assert "--common-arg=--flag" in args
    assert "--common-arg=value" in args

    assert launcher.run(["--duration-seconds", "0.01"]) == 0
    assert os.environ["MULTI_SUPERVISOR_REPO_DEFAULT"] == "1"
    assert prepared == ["called"]
    assert captured["argv"][-2:] == ("--duration-seconds", "0.01")


def test_repo_implementation_multi_supervisor_launcher_uses_packaged_resolver_default(tmp_path, monkeypatch):
    captured: dict[str, tuple[str, ...]] = {}

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(multi_supervisor_runner, "main", fake_main)

    launcher = build_repo_implementation_multi_supervisor_launcher(
        repo_root=tmp_path,
        duration_seconds=12,
        python_executable="python-test",
        label="repo implementation run",
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="VAI",
                script_path="scripts/vai.py",
                state_dir="data/vai/state",
                state_prefix="vai",
            ),
        ),
    )

    args = launcher.args()
    assert "--implementation-supervisor-command" not in args
    assert args[args.index("--implementation-supervisor-llm-merge-resolver-command") + 1] == (
        "python-test -m ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback"
    )
    assert llm_merge_resolver_fallback_command(python_executable="python-test") == (
        "python-test -m ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback"
    )


def test_llm_merge_resolver_fallback_module_uses_codex_first(tmp_path):
    codex_log = tmp_path / "codex.prompt"
    codex_bin = tmp_path / "codex"
    codex_bin.write_text(
        "#!/usr/bin/env bash\n"
        "while (($#)); do\n"
        "  if [[ \"$1\" == \"-C\" ]]; then shift; workspace=\"$1\"; fi\n"
        "  shift || true\n"
        "done\n"
        "cat > \"$workspace/codex.prompt\"\n",
        encoding="utf-8",
    )
    codex_bin.chmod(0o755)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "CODEX_BIN": str(codex_bin),
        "COPILOT_BIN": "",
        "AGENT_RESOLVER_LOCK_BYPASS": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback",
            str(tmp_path),
        ],
        input="resolve this conflict",
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert codex_log.read_text(encoding="utf-8") == "resolve this conflict"


def test_llm_merge_resolver_fallback_skips_unauthenticated_copilot(tmp_path):
    codex_bin = tmp_path / "codex"
    copilot_bin = tmp_path / "copilot"
    copilot_log = tmp_path / "copilot.log"
    codex_bin.write_text("#!/bin/bash\nexit 42\n", encoding="utf-8")
    copilot_bin.write_text(f"#!/bin/bash\nprintf invoked > {shlex.quote(str(copilot_log))}\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    copilot_bin.chmod(0o755)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "CODEX_BIN": str(codex_bin),
        "COPILOT_BIN": str(copilot_bin),
        "COPILOT_GITHUB_TOKEN": "",
        "GH_TOKEN": "",
        "GITHUB_TOKEN": "",
        "PATH": str(tmp_path),
        "AGENT_RESOLVER_LOCK_BYPASS": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.llm_merge_resolver_fallback",
            str(tmp_path),
        ],
        input="resolve this conflict",
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 42
    assert "copilot fallback is not authenticated" in completed.stderr
    assert not copilot_log.exists()


def _seed_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    (child_source / "child.txt").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "child.txt")
    _git(child_source, "commit", "-m", "child base")

    repo = tmp_path / "parent"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(child_source), "libs/child")
    _git(repo, "add", ".gitmodules", "libs/child")
    _git(repo, "commit", "-m", "add child submodule")

    submodule = repo / "libs" / "child"
    _git(submodule, "checkout", "main")
    _git(submodule, "config", "user.name", "Test User")
    _git(submodule, "config", "user.email", "test@example.invalid")
    return repo, submodule


def test_implementation_daemon_recreates_missing_registered_submodule_worktree(
    tmp_path: Path,
):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    branch_name = "implementation/auto-registered"
    worktree = repo / "worktrees" / "auto-registered"
    _git(repo, "worktree", "add", "-b", branch_name, str(worktree), "main")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    assert daemon._create_local_submodule_worktree(
        worktree,
        "libs/child",
        branch_name=branch_name,
    ) is True
    target = worktree / "libs" / "child"
    moved_target = worktree / "libs" / "orphaned-child"
    target.rename(moved_target)
    assert not target.exists()
    assert str(target) in _git(submodule, "worktree", "list", "--porcelain")

    assert daemon._create_local_submodule_worktree(
        worktree,
        "libs/child",
        branch_name=branch_name,
    ) is True
    assert daemon._is_git_worktree(target)
    worktree_listing = _git(submodule, "worktree", "list", "--porcelain")
    assert worktree_listing.count(f"worktree {target}") == 1


def test_implementation_daemon_creates_parent_handoff_for_submodule_only_commit(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    baseline = _git(repo, "rev-parse", "HEAD")

    state_dir = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["outer/inner"],
    )
    submodule_results = [
        {
            "path": "outer/inner",
            "committed": True,
            "commit": "a" * 40,
        }
    ]
    monkeypatch.setattr(
        daemon,
        "_commit_worktree_submodule_changes",
        lambda *_args, **_kwargs: submodule_results,
    )

    result = daemon._commit_worktree_changes(
        repo,
        PortalTask(
            task_id="AUTO-120",
            title="Commit nested implementation",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        1,
    )

    assert result["committed"] is True
    assert result["reason"] == "submodule_only"
    assert result["submodule_results"] == submodule_results
    assert daemon._committed_submodule_paths(submodule_results) == ["outer/inner"]
    assert result["commit"] != baseline
    assert _git(repo, "diff", "--quiet", baseline, result["commit"]) == ""


def test_implementation_daemon_rehydrates_cleaned_merge_queue_branch(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    candidate = _git(repo, "rev-parse", "HEAD")
    branch_name = "implementation/ref-040-recovery"

    state_dir = tmp_path / "state"
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "## REF-040 Recover merge handoff\n\n- Status: todo\n- Completion: manual\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="REF-",
        worktree_submodule_paths=[],
    )
    observed: dict[str, object] = {}

    def merge_branch(selected_branch, *_args, **_kwargs):
        observed["branch"] = selected_branch
        observed["commit"] = _git(repo, "rev-parse", selected_branch)
        return {"merged": True, "returncode": 0}

    monkeypatch.setattr(daemon, "_merge_branch_to_main", merge_branch)
    request = SimpleNamespace(
        branch_name=branch_name,
        commit_sha=candidate,
        task_id="REF-040",
        priority="P0",
        attempt=2,
        metadata={
            "task": {
                "task_id": "REF-040",
                "title": "Recover merge handoff",
                "status": "todo",
                "completion": "manual",
                "priority": "P0",
                "track": "ops",
            }
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["branch_rehydration"]["rehydrated"] is True
    assert observed == {"branch": branch_name, "commit": candidate}
    assert _git(repo, "rev-parse", branch_name) == candidate
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")

    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-m", "later")
    later = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", f"refs/heads/{branch_name}", later, candidate)
    mismatch = daemon._rehydrate_merge_request_branch(
        branch_name=branch_name,
        commit_sha=candidate,
        task=daemon._portal_task_from_merge_request(request),
        attempt=3,
    )
    assert mismatch["ready"] is False
    assert mismatch["reason"] == "merge_branch_candidate_mismatch"
    assert mismatch["branch_commit"] == later


def test_merge_train_accepts_commit_integrated_by_merge_resolver(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/ref-041-resolver"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "resolved.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "resolved.txt")
    _git(repo, "commit", "-m", "REF-041: resolved implementation")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    state_dir = tmp_path / "state"
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "## REF-041 Accept resolver merge\n\n- Status: todo\n- Completion: manual\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="REF-",
        worktree_submodule_paths=[],
    )

    def resolver_integrates_branch(selected_branch, *_args, **_kwargs):
        _git(repo, "merge", "--no-ff", "--no-edit", selected_branch)
        _git(repo, "branch", "-D", selected_branch)
        return {
            "attempted": True,
            "merged": False,
            "returncode": 1,
            "reason": "merge_branch_missing_after_resolver",
            "submodule_merge_results": [],
        }

    monkeypatch.setattr(daemon, "_merge_branch_to_main", resolver_integrates_branch)
    request = SimpleNamespace(
        branch_name=branch_name,
        commit_sha=candidate,
        task_id="REF-041",
        priority="P0",
        attempt=1,
        metadata={
            "task": {
                "task_id": "REF-041",
                "title": "Accept resolver merge",
                "status": "todo",
                "completion": "manual",
                "priority": "P0",
                "track": "ops",
            }
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is True
    assert result["returncode"] == 0
    assert result["reason"] == "implementation_commit_already_merged"
    assert result["post_callback_ancestry_reconciliation"]["previous_reason"] == (
        "merge_branch_missing_after_resolver"
    )
    assert _git(repo, "merge-base", "--is-ancestor", candidate, "main") == ""
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")


def test_merge_train_rejects_resolver_merge_with_unverified_changed_submodule(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/ref-042-resolver"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "resolved.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "resolved.txt")
    _git(repo, "commit", "-m", "REF-042: resolved implementation")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    state_dir = tmp_path / "state"
    todo_path = repo / "todo.md"
    todo_path.write_text(
        "## REF-042 Verify nested merge\n\n- Status: todo\n- Completion: manual\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="REF-",
        worktree_submodule_paths=["libs/child"],
    )

    def resolver_integrates_only_root(selected_branch, *_args, **_kwargs):
        _git(repo, "merge", "--no-ff", "--no-edit", selected_branch)
        return {
            "attempted": True,
            "merged": False,
            "returncode": 1,
            "reason": "resolver_committed_merge",
            "submodule_merge_results": [],
        }

    monkeypatch.setattr(daemon, "_merge_branch_to_main", resolver_integrates_only_root)
    request = SimpleNamespace(
        branch_name=branch_name,
        commit_sha=candidate,
        task_id="REF-042",
        priority="P0",
        attempt=1,
        metadata={
            "changed_submodule_paths": ["libs/child"],
            "task": {
                "task_id": "REF-042",
                "title": "Verify nested merge",
                "status": "todo",
                "completion": "manual",
                "priority": "P0",
                "track": "ops",
            },
        },
    )

    result = daemon._merge_train_callback(request)

    assert result["merged"] is False
    assert result["returncode"] == 2
    assert result["reason"] == "changed_submodule_merge_unverified"
    assert result["missing_changed_submodule_paths"] == ["libs/child"]
    assert result["submodule_verification"] == {
        "verified": False,
        "expected_paths": ["libs/child"],
        "reported_paths": [],
        "previous_reason": "resolver_committed_merge",
    }
    assert _git(repo, "merge-base", "--is-ancestor", candidate, "main") == ""
    assert "- Status: todo" in todo_path.read_text(encoding="utf-8")


def _seed_parent_with_divergent_gitlinks(
    tmp_path: Path,
    *,
    content_conflict: bool = False,
) -> tuple[Path, Path, str, str, str]:
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    base_submodule = _git(submodule, "rev-parse", "HEAD")
    base_parent = _git(repo, "rev-parse", "HEAD")

    if content_conflict:
        (submodule / "child.txt").write_text("ours\n", encoding="utf-8")
        _git(submodule, "commit", "-am", "ours child change")
    else:
        (submodule / "ours.txt").write_text("ours\n", encoding="utf-8")
        _git(submodule, "add", "ours.txt")
        _git(submodule, "commit", "-m", "ours child change")
    ours = _git(submodule, "rev-parse", "HEAD")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "advance child on main")

    _git(submodule, "checkout", "-b", "child-side", base_submodule)
    if content_conflict:
        (submodule / "child.txt").write_text("theirs\n", encoding="utf-8")
        _git(submodule, "commit", "-am", "theirs child change")
    else:
        (submodule / "theirs.txt").write_text("theirs\n", encoding="utf-8")
        _git(submodule, "add", "theirs.txt")
        _git(submodule, "commit", "-m", "theirs child change")
    theirs = _git(submodule, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "implementation/auto-116", base_parent)
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "AUTO-116: advance child on implementation")
    _git(repo, "checkout", "main")
    _git(submodule, "checkout", "main")
    assert _git(submodule, "rev-parse", "HEAD") == ours
    assert _git(repo, "status", "--porcelain") == ""
    return repo, submodule, base_submodule, ours, theirs


def test_implementation_daemon_submodule_gitlink_reconciliation_uses_verified_recovery_ref(tmp_path):
    repo, submodule, base, ours, theirs = _seed_parent_with_divergent_gitlinks(tmp_path)
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-116",
        PortalTask(
            task_id="AUTO-116",
            title="Reconcile divergent gitlinks",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    repair = result["submodule_conflict_repair"]["repairs"][0]
    selected = repair["selected_commit"]
    assert repair["path"] == "libs/child"
    assert repair["ours_candidate"] == ours
    assert repair["theirs_candidate"] == theirs
    assert repair["reachable_merge_bases"] == [base]
    assert repair["selection_reason"] == "created_deterministic_recovery_ref"
    assert repair["recovery_ref"].startswith("refs/agent-supervisor/submodule-merge-recovery/")
    assert _git(submodule, "merge-base", "--is-ancestor", ours, selected) == ""
    assert _git(submodule, "merge-base", "--is-ancestor", theirs, selected) == ""
    assert _git(submodule, "rev-parse", "HEAD") == ours
    assert _git(repo, "rev-parse", "HEAD:libs/child") == selected

    diagnostic = json.loads((state_dir / "submodule-merge-diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostic["schema_version"] == 1
    assert diagnostic["latest"]["repaired"] is True
    assert diagnostic["latest"]["retryable"] is False
    assert diagnostic["latest"]["conflicts"][0]["selected_commit"] == selected


def test_implementation_daemon_anchors_relative_recovery_worktrees_to_repo_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=Path("todo.md"),
        state_path=Path("tmp/supervisor/state/task_state.json"),
        strategy_path=Path("tmp/supervisor/state/strategy.json"),
        events_path=Path("tmp/supervisor/state/events.jsonl"),
        repo_root=repo,
    )

    assert daemon._submodule_recovery_worktree_root() == (
        repo / "tmp/supervisor/state/submodule-merge-recovery-worktrees"
    ).resolve()


def test_implementation_daemon_reports_missing_submodule_recovery_source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    result = daemon._create_submodule_recovery_ref(
        source=repo / "missing-submodule",
        relative="missing-submodule",
        ours="a" * 40,
        theirs="b" * 40,
        recovery_ref="refs/agent-supervisor/submodule-merge-recovery/missing",
        task=PortalTask(
            task_id="AUTO-RECOVERY",
            title="Handle missing recovery source",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
    )

    assert result == {
        "created": False,
        "reason": "recovery_source_unavailable",
        "source": str(repo / "missing-submodule"),
    }


def test_implementation_daemon_records_full_nested_submodule_gitlink_path(tmp_path):
    repo, submodule, base, ours, theirs = _seed_parent_with_divergent_gitlinks(tmp_path)
    merge = subprocess.run(
        ["git", "merge", "--no-ff", "--no-edit", "implementation/auto-116"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=tmp_path,
    )

    result = daemon._repair_submodule_gitlink_merge_conflicts(
        repo,
        task=PortalTask(
            task_id="AUTO-116",
            title="Reconcile nested gitlink",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        attempt=3,
        parent_relative="parent",
    )

    assert result["repaired"] is True
    conflict = result["repairs"][0]
    assert conflict["path"] == "parent/libs/child"
    assert conflict["ours_candidate"] == ours
    assert conflict["theirs_candidate"] == theirs
    assert conflict["reachable_merge_bases"] == [base]
    assert _git(submodule, "merge-base", "--is-ancestor", ours, conflict["selected_commit"]) == ""
    assert _git(submodule, "merge-base", "--is-ancestor", theirs, conflict["selected_commit"]) == ""


def test_implementation_daemon_submodule_gitlink_conflict_stays_retryable_without_side_selection(tmp_path):
    repo, submodule, base, ours, theirs = _seed_parent_with_divergent_gitlinks(
        tmp_path,
        content_conflict=True,
    )
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-116",
        PortalTask(
            task_id="AUTO-116",
            title="Keep unsafe gitlink conflict retryable",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        2,
    )

    assert result["merged"] is False
    assert result["merge_abort_result"]["aborted"] is True
    conflict = result["submodule_conflict_repair"]["repairs"][0]
    assert conflict["ours_candidate"] == ours
    assert conflict["theirs_candidate"] == theirs
    assert conflict["reachable_merge_bases"] == [base]
    assert conflict["selected_commit"] == ""
    assert conflict["repaired"] is False
    assert _git(submodule, "rev-parse", "HEAD") == ours
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "implementation/auto-116", "main"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert ancestor_check.returncode != 0

    diagnostic = json.loads((state_dir / "submodule-merge-diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostic["latest"]["repaired"] is False
    assert diagnostic["latest"]["retryable"] is True
    assert diagnostic["latest"]["conflicts"][0]["selected_commit"] == ""


def test_implementation_daemon_reconciles_nested_submodule_gitlink_inside_recovery_ref(tmp_path):
    grand_source = tmp_path / "grand-source"
    grand_source.mkdir()
    _git(grand_source, "init")
    _git(grand_source, "checkout", "-b", "main")
    _git(grand_source, "config", "user.name", "Test User")
    _git(grand_source, "config", "user.email", "test@example.invalid")
    (grand_source / "base.txt").write_text("base\n", encoding="utf-8")
    _git(grand_source, "add", "base.txt")
    _git(grand_source, "commit", "-m", "grand base")

    child_source = tmp_path / "child-source-nested"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    _git(
        child_source,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(grand_source),
        "nested/grand",
    )
    _git(child_source, "commit", "-am", "add grandchild")

    repo = tmp_path / "nested-parent"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "libs/child",
    )
    _git(repo, "commit", "-am", "add nested child")
    base_parent = _git(repo, "rev-parse", "HEAD")

    child = repo / "libs" / "child"
    _git(child, "config", "user.name", "Test User")
    _git(child, "config", "user.email", "test@example.invalid")
    _git(child, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    grand = child / "nested" / "grand"
    _git(grand, "checkout", "main")
    _git(grand, "config", "user.name", "Test User")
    _git(grand, "config", "user.email", "test@example.invalid")
    base_grand = _git(grand, "rev-parse", "HEAD")
    base_child = _git(child, "rev-parse", "HEAD")

    (grand / "ours.txt").write_text("ours\n", encoding="utf-8")
    _git(grand, "add", "ours.txt")
    _git(grand, "commit", "-m", "grand ours")
    grand_ours = _git(grand, "rev-parse", "HEAD")
    _git(child, "add", "nested/grand")
    _git(child, "commit", "-m", "child points to grand ours")
    child_ours = _git(child, "rev-parse", "HEAD")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "parent points to child ours")

    _git(child, "checkout", "-b", "child-side", base_child)
    _git(grand, "checkout", "-b", "grand-side", base_grand)
    (grand / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _git(grand, "add", "theirs.txt")
    _git(grand, "commit", "-m", "grand theirs")
    grand_theirs = _git(grand, "rev-parse", "HEAD")
    _git(child, "add", "nested/grand")
    _git(child, "commit", "-m", "child points to grand theirs")
    child_theirs = _git(child, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "implementation/auto-116", base_parent)
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "AUTO-116: nested implementation pointer")
    _git(repo, "checkout", "main")
    _git(child, "checkout", "main")
    _git(grand, "checkout", "main")
    assert _git(child, "rev-parse", "HEAD") == child_ours
    assert _git(grand, "rev-parse", "HEAD") == grand_ours
    assert _git(repo, "status", "--porcelain") == ""

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    result = daemon._merge_branch_to_main(
        "implementation/auto-116",
        PortalTask(
            task_id="AUTO-116",
            title="Recursively reconcile nested gitlinks",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        4,
    )

    assert result["merged"] is True
    parent_repair = result["submodule_conflict_repair"]["repairs"][0]
    child_recovery = parent_repair["recovery"]
    nested_repair = child_recovery["nested_gitlink_repair"]["repairs"][0]
    assert parent_repair["ours_candidate"] == child_ours
    assert parent_repair["theirs_candidate"] == child_theirs
    assert nested_repair["path"] == "libs/child/nested/grand"
    assert nested_repair["ours_candidate"] == grand_ours
    assert nested_repair["theirs_candidate"] == grand_theirs
    assert nested_repair["reachable_merge_bases"] == [base_grand]
    assert nested_repair["selection_reason"] == "created_deterministic_recovery_ref"
    assert _git(grand, "merge-base", "--is-ancestor", grand_ours, nested_repair["selected_commit"]) == ""
    assert _git(grand, "merge-base", "--is-ancestor", grand_theirs, nested_repair["selected_commit"]) == ""

    diagnostic = json.loads((state_dir / "submodule-merge-diagnostics.json").read_text(encoding="utf-8"))
    recorded_paths = {
        conflict["path"]
        for entry in diagnostic["attempts"]
        for conflict in entry["conflicts"]
    }
    assert recorded_paths == {"libs/child", "libs/child/nested/grand"}

    # An unchanged/missing branch at the child level must not hide ready work
    # in its grandchild repository.
    nested_parent_branch = "implementation/auto-117"
    nested_branch = daemon._submodule_worktree_branch_name(
        nested_parent_branch,
        "libs/child/nested/grand",
    )
    _git(grand, "checkout", "-b", nested_branch)
    (grand / "later.txt").write_text("later nested work\n", encoding="utf-8")
    _git(grand, "add", "later.txt")
    _git(grand, "commit", "-m", "AUTO-117: later nested work")
    later_commit = _git(grand, "rev-parse", "HEAD")
    _git(grand, "checkout", "main")

    nested_results = daemon._merge_submodule_branches_to_main(
        nested_parent_branch,
        task=PortalTask(
            task_id="AUTO-117",
            title="Merge nested work through unchanged parent",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        attempt=1,
    )

    assert [item["path"] for item in nested_results] == ["libs/child/nested/grand"]
    assert nested_results[0]["merged"] is True
    assert _git(grand, "merge-base", "--is-ancestor", later_commit, "main") == ""

    # Ephemeral setup creates branches in every recursive submodule. Only
    # repositories that produced task-owned commits may enter merge handling.
    filtered_parent_branch = "implementation/auto-118"
    filtered_child_branch = daemon._submodule_worktree_branch_name(
        filtered_parent_branch,
        "libs/child",
    )
    _git(child, "checkout", "-b", filtered_child_branch)
    (child / "filtered.txt").write_text("selected child work\n", encoding="utf-8")
    _git(child, "add", "filtered.txt")
    _git(child, "commit", "-m", "AUTO-118: selected child work")
    filtered_child_commit = _git(child, "rev-parse", "HEAD")
    _git(child, "checkout", "main")

    unrelated_grand_branch = daemon._submodule_worktree_branch_name(
        filtered_parent_branch,
        "libs/child/nested/grand",
    )
    _git(grand, "checkout", "-b", unrelated_grand_branch)
    (grand / "unrelated.txt").write_text("unrelated grandchild work\n", encoding="utf-8")
    _git(grand, "add", "unrelated.txt")
    _git(grand, "commit", "-m", "unrelated grandchild work")
    unrelated_grand_commit = _git(grand, "rev-parse", "HEAD")
    _git(grand, "checkout", "main")
    grand_main_before = _git(grand, "rev-parse", "main")

    filtered_results = daemon._merge_submodule_branches_to_main(
        filtered_parent_branch,
        task=PortalTask(
            task_id="AUTO-118",
            title="Merge only repositories changed by this task",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        attempt=1,
        changed_submodule_paths={"libs/child"},
    )

    assert [item["path"] for item in filtered_results] == ["libs/child"]
    assert _git(child, "merge-base", "--is-ancestor", filtered_child_commit, "main") == ""
    assert _git(grand, "rev-parse", "main") == grand_main_before
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", unrelated_grand_commit, "main"],
        cwd=grand,
        capture_output=True,
        check=False,
    ).returncode != 0


def test_implementation_daemon_skips_unrelated_submodule_branch_when_gitlink_is_unchanged(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    baseline_ref = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "implementation/auto-117")
    (repo / "README.md").write_text("parent-only task\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "AUTO-117: update parent only")
    _git(repo, "checkout", "main")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    submodule_branch = daemon._submodule_worktree_branch_name(
        "implementation/auto-117",
        "libs/child",
    )
    _git(submodule, "checkout", "-b", submodule_branch)
    (submodule / "unrelated.txt").write_text("unrelated task work\n", encoding="utf-8")
    _git(submodule, "add", "unrelated.txt")
    _git(submodule, "commit", "-m", "unrelated child task")
    unrelated_commit = _git(submodule, "rev-parse", "HEAD")
    _git(submodule, "checkout", "main")

    result = daemon._merge_branch_to_main(
        "implementation/auto-117",
        PortalTask(
            task_id="AUTO-117",
            title="Merge a parent-only task",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        1,
        baseline_ref=baseline_ref,
    )

    assert result["merged"] is True
    assert result["submodule_merge_results"] == [
        {
            "path": "libs/child",
            "branch": submodule_branch,
            "default_branch": "main",
            "merged": True,
            "reason": "unchanged_gitlink_in_task",
        }
    ]
    assert _git(submodule, "rev-parse", "main") != unrelated_commit


def test_implementation_daemon_detects_changed_submodule_gitlink_without_error(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    baseline_ref = _git(repo, "rev-parse", "HEAD")
    (submodule / "task-owned.txt").write_text("task-owned child work\n", encoding="utf-8")
    _git(submodule, "add", "task-owned.txt")
    _git(submodule, "commit", "-m", "child work for AUTO-118")
    _git(repo, "checkout", "-b", "implementation/auto-118")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "AUTO-118: update child gitlink")
    _git(repo, "checkout", "main")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    assert daemon._root_submodule_changed_in_task(
        "implementation/auto-118",
        baseline_ref,
        "libs/child",
    ) is True


def test_implementation_daemon_treats_nested_configured_path_as_changed(tmp_path):
    repo, _submodule = _seed_parent_with_submodule(tmp_path)
    baseline_ref = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "implementation/auto-119")
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child/nested/target"],
    )

    assert daemon._root_submodule_changed_in_task(
        "implementation/auto-119",
        baseline_ref,
        "libs/child/nested/target",
    ) is True


def test_implementation_daemon_merges_submodule_with_nonoverlapping_dirty_paths(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    parent_branch = "implementation/auto-121"
    submodule_branch = daemon._submodule_worktree_branch_name(parent_branch, "libs/child")
    _git(submodule, "checkout", "-b", submodule_branch)
    (submodule / "task-owned.txt").write_text("task work\n", encoding="utf-8")
    _git(submodule, "add", "task-owned.txt")
    _git(submodule, "commit", "-m", "AUTO-121: task work")
    task_commit = _git(submodule, "rev-parse", "HEAD")
    _git(submodule, "checkout", "main")
    (submodule / "child.txt").write_text("preserved local dirt\n", encoding="utf-8")

    results = daemon._merge_submodule_branches_to_main(
        parent_branch,
        task=PortalTask(
            task_id="AUTO-121",
            title="Merge around local dirt",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        attempt=1,
    )

    assert results[0]["merged"] is True
    assert results[0]["preserved_dirty_paths"] == ["child.txt"]
    assert _git(submodule, "merge-base", "--is-ancestor", task_commit, "main") == ""
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "preserved local dirt\n"


def test_implementation_daemon_records_merged_root_submodule_gitlink(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    (submodule / "merged.txt").write_text("merged child work\n", encoding="utf-8")
    _git(submodule, "add", "merged.txt")
    _git(submodule, "commit", "-m", "merged child work")
    merged_commit = _git(submodule, "rev-parse", "HEAD")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    task = PortalTask(
        task_id="AUTO-119",
        title="Record merged child revision",
        status="todo",
        completion="manual",
        priority="P0",
        track="ops",
    )

    result = daemon._record_merged_submodule_gitlinks(
        repo,
        [{"path": "libs/child", "merged": True, "commit": merged_commit}],
        task=task,
    )

    assert result["committed"] is True
    assert _git(repo, "rev-parse", "HEAD:libs/child") == merged_commit
    assert _git(repo, "status", "--porcelain") == ""


def test_implementation_daemon_records_nested_gitlink_chain_and_preserves_local_dirt(
    tmp_path,
):
    repo, parent = _seed_parent_with_submodule(tmp_path)
    leaf_source = tmp_path / "leaf-source"
    leaf_source.mkdir()
    _git(leaf_source, "init")
    _git(leaf_source, "checkout", "-b", "main")
    _git(leaf_source, "config", "user.name", "Test User")
    _git(leaf_source, "config", "user.email", "test@example.invalid")
    (leaf_source / "leaf.txt").write_text("base\n", encoding="utf-8")
    _git(leaf_source, "add", "leaf.txt")
    _git(leaf_source, "commit", "-m", "leaf base")

    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf_source),
        "nested/leaf",
    )
    _git(parent, "commit", "-am", "add leaf submodule")
    _git(repo, "add", "libs/child")
    (repo / "local.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "local.txt")
    _git(repo, "commit", "-m", "record nested baseline")
    parent_commit_before_leaf_merge = _git(parent, "rev-parse", "HEAD")

    leaf = parent / "nested" / "leaf"
    _git(leaf, "checkout", "main")
    _git(leaf, "config", "user.name", "Test User")
    _git(leaf, "config", "user.email", "test@example.invalid")
    (leaf / "merged.txt").write_text("merged leaf work\n", encoding="utf-8")
    _git(leaf, "add", "merged.txt")
    _git(leaf, "commit", "-m", "merged leaf work")
    merged_commit = _git(leaf, "rev-parse", "HEAD")
    (repo / "local.txt").write_text("preserved local work\n", encoding="utf-8")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    task = PortalTask(
        task_id="AUTO-122",
        title="Record nested merged revision",
        status="todo",
        completion="manual",
        priority="P0",
        track="ops",
    )

    result = daemon._record_merged_submodule_gitlinks(
        repo,
        [
            {
                "path": "libs/child",
                "merged": True,
                "commit": parent_commit_before_leaf_merge,
            },
            {
                "path": "libs/child/nested/leaf",
                "merged": True,
                "commit": merged_commit,
            }
        ],
        task=task,
    )

    parent_commit = _git(parent, "rev-parse", "HEAD")
    assert result["ok"] is True
    assert result["committed"] is True
    assert [entry["path"] for entry in result["chain"]] == [
        "libs/child/nested/leaf",
        "libs/child",
    ]
    assert _git(parent, "rev-parse", "HEAD:nested/leaf") == merged_commit
    assert _git(repo, "rev-parse", "HEAD:libs/child") == parent_commit
    assert (repo / "local.txt").read_text(encoding="utf-8") == "preserved local work\n"
    assert _git(repo, "status", "--porcelain") == "M local.txt"


def test_implementation_daemon_rolls_back_parent_when_root_submodule_merge_fails(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    base_parent = _git(repo, "rev-parse", "HEAD")
    base_child = _git(submodule, "rev-parse", "HEAD")
    _git(submodule, "checkout", "-b", "implementation/auto-120-submodule-libs-child")
    (submodule / "conflicting.txt").write_text("task child work\n", encoding="utf-8")
    _git(submodule, "add", "conflicting.txt")
    _git(submodule, "commit", "-m", "AUTO-120: child task work")

    _git(repo, "checkout", "-b", "implementation/auto-120")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "AUTO-120: advance child gitlink")
    _git(repo, "checkout", "main")
    _git(submodule, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/auto-120")
    assert _git(repo, "status", "--porcelain").endswith("libs/child")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    result = daemon._rollback_parent_merge_after_submodule_failure(
        repo,
        pre_merge_commit=base_parent,
        failed_submodules=[{"path": "libs/child", "merged": False}],
    )

    assert result["rolled_back"] is True
    assert _git(repo, "rev-parse", "HEAD") == base_parent
    assert _git(repo, "rev-parse", "HEAD:libs/child") == base_child
    assert _git(submodule, "rev-parse", "HEAD") == base_child
    assert _git(repo, "status", "--porcelain") == ""


def test_implementation_daemon_preserves_nested_tmp_and_allows_unchanged_dirty_submodule(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "add parent readme")
    _git(repo, "checkout", "-b", "implementation/auto-116")
    (repo / "README.md").write_text("implementation\n", encoding="utf-8")
    _git(repo, "commit", "-am", "AUTO-116: update parent only")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    (submodule / "child.txt").write_text("unrelated user dirt\n", encoding="utf-8")
    nested_tmp = submodule / "tmp"
    nested_tmp.mkdir()
    (nested_tmp / "worker.log").write_text("preserve me\n", encoding="utf-8")
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    candidates = [
        {
            "task_id": "AUTO-116",
            "branch": "implementation/auto-116",
            "implementation_commit": implementation_commit,
        }
    ]

    preservation = daemon._preserve_generated_nested_worktree_directories()
    blocking, nonblocking = daemon._reconciliation_blocking_dirty_paths(
        candidates,
        target_branch="main",
    )

    assert preservation[0]["path"] == "libs/child/tmp"
    assert preservation[0]["preserved"] is True
    preserved_path = Path(preservation[0]["destination"])
    assert (preserved_path / "worker.log").read_text(encoding="utf-8") == "preserve me\n"
    assert not nested_tmp.exists()
    assert blocking == []
    assert nonblocking == ["libs/child"]

    merge = daemon._merge_branch_to_main(
        "implementation/auto-116",
        PortalTask(
            task_id="AUTO-116",
            title="Merge with unrelated dirty submodule",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
        ),
        1,
    )
    assert merge["merged"] is True
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "unrelated user dirt\n"
    maintenance = daemon._reset_persistently_dirty_submodules()
    assert maintenance["dirty_count"] == 1
    assert maintenance["reset"][0]["preserved"] is True
    assert maintenance["reset"][0]["reset_ok"] is False
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "unrelated user dirt\n"


def test_implementation_daemon_failed_merge_reconciliation_remains_retryable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-116")
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "AUTO-116: feature")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "missing.todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": "AUTO-116",
            "attempt": 1,
            "branch": "implementation/auto-116",
            "implementation_commit": implementation_commit,
            "merge_result": {"attempted": True, "merged": False, "reason": "submodule_merge_failed"},
        },
    )
    daemon._record_event(
        "merge_reconciled",
        {
            "task_id": "AUTO-116",
            "attempt": 1,
            "branch": "implementation/auto-116",
            "implementation_commit": implementation_commit,
            "resolved": False,
            "reason": "merge_retry_failed",
            "merge_result": {"attempted": True, "merged": False, "reason": "submodule_merge_failed"},
        },
    )

    candidates = daemon._failed_merge_candidates()

    assert len(candidates) == 1
    assert candidates[0]["implementation_commit"] == implementation_commit


def test_implementation_daemon_retries_submodule_after_parent_commit_already_landed(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "add parent readme")
    _git(repo, "checkout", "-b", "implementation/auto-116")
    (repo / "README.md").write_text("implementation\n", encoding="utf-8")
    _git(repo, "commit", "-am", "AUTO-116: parent change")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")

    submodule_branch = "implementation/auto-116-submodule-libs-child"
    _git(submodule, "checkout", "-b", submodule_branch)
    (submodule / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(submodule, "add", "feature.txt")
    _git(submodule, "commit", "-m", "AUTO-116: child change")
    submodule_commit = _git(submodule, "rev-parse", "HEAD")
    _git(submodule, "checkout", "main")
    (submodule / "feature.txt").write_text("temporarily dirty\n", encoding="utf-8")

    state_dir = tmp_path / "supervisor-state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "missing.todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )
    task = PortalTask(
        task_id="AUTO-116",
        title="Retry child after parent landed",
        status="todo",
        completion="manual",
        priority="P0",
        track="ops",
    )
    first = daemon._merge_branch_to_main("implementation/auto-116", task, 1)

    assert first["merged"] is False
    assert first["reason"] == "submodule_merge_failed"
    assert first["submodule_merge_results"][0]["reason"] == "submodule_checkout_dirty"
    assert first["submodule_failure_rollback"]["reason"] == "parent_gitlinks_unchanged"
    assert _git(repo, "merge-base", "--is-ancestor", implementation_commit, "main") == ""
    checkpoint_path = state_dir / "merge_checkpoints" / "implementation-auto-116.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "libs/child" in checkpoint["failed_submodules"]

    (submodule / "feature.txt").unlink()
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": task.task_id,
            "attempt": 1,
            "branch": "implementation/auto-116",
            "implementation_commit": implementation_commit,
            "merge_result": first,
        },
    )
    reconciliation = daemon._reconcile_failed_merges()

    assert reconciliation[-1]["resolved"] is True
    assert reconciliation[-1]["reason"] == "implementation_commit_already_merged"
    assert reconciliation[-1]["submodule_merge_results"][0]["merged"] is True
    assert _git(submodule, "merge-base", "--is-ancestor", submodule_commit, "main") == ""
    assert not checkpoint_path.exists()


def test_todo_daemon_runtime_is_ported_to_accelerate_package():
    assert TodoDaemonRunner.__module__ == "ipfs_accelerate_py.agent_supervisor.todo_daemon.runner"
    assert TodoImplementationDaemon.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert TodoImplementationSupervisor.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )
    assert TodoSupervisorConfig.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )


def test_supervisor_runtime_repairs_marker_directories_before_launch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    log_path = state_dir / "child.log"
    pid_path = state_dir / "child.pid"
    latest_log_path = state_dir / "latest.log"
    for marker_path in (log_path, pid_path, latest_log_path):
        marker_path.mkdir()
        (marker_path / "fragment").write_text("old marker directory\n", encoding="utf-8")

    child = None
    try:
        child = launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=(sys.executable, "-c", "import time; time.sleep(30)"),
                log_path=log_path,
                child_pid_path=pid_path,
                latest_log_path=latest_log_path,
            )
        )

        assert log_path.is_file()
        assert pid_path.read_text(encoding="utf-8").strip() == str(child.pid)
        assert latest_log_path.is_symlink()
        assert latest_log_path.readlink() == Path(log_path.name)
        for marker_name in ("child.log", "child.pid", "latest.log"):
            backups = list(state_dir.glob(f"{marker_name}.directory-backup-*"))
            assert len(backups) == 1
            assert (backups[0] / "fragment").exists()
    finally:
        if child is not None:
            terminate_supervised_child(child, grace_seconds=1)


def test_supervisor_status_write_repairs_directory_status_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor_status_path = state_dir / "supervisor_status.json"
    supervisor_status_path.mkdir(parents=True)
    (supervisor_status_path / "fragment").write_text("old status directory\n", encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=supervisor_status_path,
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )

    payload = SupervisorStatusContext(spec).write("running", run_id="run-1")

    assert supervisor_status_path.is_file()
    assert json.loads(supervisor_status_path.read_text(encoding="utf-8"))["status"] == "running"
    assert payload["run_id"] == "run-1"
    backups = list(state_dir.glob("supervisor_status.json.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()


def test_stop_daemon_moves_directory_pid_markers(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor_pid_path = state_dir / "supervisor.pid"
    child_pid_path = state_dir / "child.pid"
    for marker_path in (supervisor_pid_path, child_pid_path):
        marker_path.mkdir(parents=True, exist_ok=True)
        (marker_path / "fragment").write_text("old pid directory\n", encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=supervisor_pid_path,
        child_pid_path=child_pid_path,
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )

    result = stop_daemon(spec, cleanup_tmux=False)

    assert result.exit_code == 0
    assert result.payload["status"] == "not_running"
    for marker_name in ("supervisor.pid", "child.pid"):
        assert not (state_dir / marker_name).exists()
        backups = list(state_dir.glob(f"{marker_name}.directory-backup-*"))
        assert len(backups) == 1
        assert (backups[0] / "fragment").exists()


def test_restart_policy_applies_backoff_and_resets_after_a_healthy_run():
    policy = RestartPolicy(
        restart_backoff_seconds=10,
        fast_restart_backoff_seconds=2,
        backoff_factor=2,
        max_backoff_seconds=60,
        healthy_run_seconds=120,
    )

    assert policy.delay_for_status("stale_heartbeat", run_duration=0) == 10
    assert policy.delay_for_status("stale_heartbeat", run_duration=0) == 20
    assert policy.delay_for_status("stale_heartbeat", run_duration=120) == 10
    assert policy._consecutive_failures == 0

    policy.reset()

    assert policy._consecutive_failures == 0


def test_supervisor_loop_retries_child_launch_failures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=("/definitely/missing/daemon",),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
        latest_log_path=state_dir / "latest.log",
    )
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=("/definitely/missing/daemon",),
            log_prefix="child",
            restart_policy=RestartPolicy(restart_backoff_seconds=0, fast_restart_backoff_seconds=0),
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            max_restarts=2,
        ),
        sleep=lambda _seconds: None,
    )

    result = loop.run()

    assert result.status == "max_restarts_reached"
    assert result.restart_count == 2
    assert result.last_exit_code == 127
    assert result.last_recycle_reason == "launch_failed"
    status = json.loads((state_dir / "supervisor_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "max_restarts_reached"
    assert status["last_recycle_reason"] == "launch_failed"
    assert status["last_exit_code"] == 127


def test_implementation_supervisor_recovers_after_child_loop_restart_exhaustion(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()

    class RecoveringLoop:
        calls = 0

        def __init__(self, config, *, watchdog_hook=None):
            self.config = config
            self.watchdog_hook = watchdog_hook

        def run(self):
            RecoveringLoop.calls += 1
            if RecoveringLoop.calls == 1:
                return SupervisorLoopResult(
                    status="max_restarts_reached",
                    restart_count=2,
                    last_exit_code=127,
                    last_recycle_reason="launch_failed",
                    last_run_id="first",
                    last_log_path="first.log",
                )
            return SupervisorLoopResult(status="stopped", restart_count=0, last_run_id="second")

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            check_interval=0.01,
            max_restarts=1,
            repo_root=repo,
        )
    )
    supervisor.shared_supervisor_loop_class = RecoveringLoop
    supervisor.ensure_event_log_file = lambda: {"repaired": False, "reason": "valid"}
    supervisor.repair_main_checkout_merge_state = lambda: {"attempted": False, "repaired": False, "reason": "clean"}
    supervisor.ensure_managed_daemon_pid_file = lambda: {"adopted": False, "reason": "not_running"}
    run_once_calls: list[bool] = []

    def fake_run_once(*, include_refill=True):
        run_once_calls.append(include_refill)
        return {"stuck": False, "recovered": True}

    supervisor.run_once = fake_run_once
    supervisor._supervisor_loop_recovery_delay_seconds = lambda: 0.0

    supervisor.run_forever()

    assert RecoveringLoop.calls == 2
    assert run_once_calls == [False, True]
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["type"] for event in events] == [
        "supervisor_preflight_maintenance_pass",
        "supervisor_loop_finished",
        "supervisor_loop_recovery_pass",
        "supervisor_loop_restarting_after_recovery",
        "supervisor_loop_finished",
    ]
    assert events[1]["status"] == "max_restarts_reached"
    assert events[2]["recovery"]["recovered"] is True
    assert events[-1]["status"] == "stopped"


def test_implementation_supervisor_signal_cleans_managed_daemon_before_exit(
    tmp_path, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as supervisor_module,
    )

    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    handlers = {
        signal.SIGTERM: "previous-term",
        signal.SIGINT: "previous-int",
    }
    transitions = []

    def fake_getsignal(signum):
        return handlers[signum]

    def fake_signal(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        transitions.append((signum, handler))
        return previous

    cleanup_calls = []
    recorded = []

    def fake_loop():
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(supervisor_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(supervisor_module.signal, "signal", fake_signal)
    monkeypatch.setattr(supervisor, "_run_forever_loop", fake_loop)
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda: cleanup_calls.append(True) or {"pid": 4321, "terminated": True},
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda event_type, payload: recorded.append((event_type, payload)),
    )

    with pytest.raises(SystemExit) as exc_info:
        supervisor.run_forever()

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert cleanup_calls == [True]
    assert recorded == [
        (
            "supervisor_signal_shutdown",
            {
                "signal": signal.SIGTERM,
                "managed_daemon_cleanup": {"pid": 4321, "terminated": True},
            },
        )
    ]
    assert handlers == {
        signal.SIGTERM: "previous-term",
        signal.SIGINT: "previous-int",
    }
    assert len(transitions) == 4


def test_implementation_daemon_accepts_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command="true",
        llm_merge_resolver_timeout_seconds=5,
        worktree_submodule_paths=["packages/app,external/lib", "vendor/tools"],
        objective_path=repo / "objective-heap.md",
        objective_bundle_dir=repo / "objective_bundles",
        merge_reconciliation_max_merges=7,
        merged_worktree_cleanup_max=11,
        task_shard_count=3,
        task_shard_index=1,
    )

    assert daemon.worktree_submodule_paths == ("packages/app", "external/lib", "vendor/tools")
    assert daemon.llm_merge_resolver_command == "true"
    assert daemon.llm_merge_resolver_timeout_seconds == 5
    assert daemon.objective_path == repo / "objective-heap.md"
    assert daemon.objective_bundle_dir == repo / "objective_bundles"
    assert daemon.merge_reconciliation_max_merges == 7
    assert daemon.merged_worktree_cleanup_max == 11
    assert daemon.task_shard_count == 3
    assert daemon.task_shard_index == 1

    args = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(repo / "todo.md"),
            "--llm-merge-resolver-command",
            "true",
            "--llm-merge-resolver-timeout-seconds",
            "5",
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
            "--objective-path",
            str(repo / "objective-heap.md"),
            "--objective-bundle-dir",
            str(repo / "objective_bundles"),
            "--merge-reconciliation-max-merges",
            "9",
            "--merged-worktree-cleanup-max",
            "13",
            "--objective-scan-min-open-tasks",
            "20",
            "--objective-scan-max-findings",
            "6",
            "--objective-scan-cooldown-seconds",
            "900",
            "--objective-surplus-findings-per-goal",
            "2",
            "--objective-surplus-min-terms-per-todo",
            "4",
            "--codebase-scan-min-open-tasks",
            "20",
            "--codebase-scan-max-findings",
            "0",
            "--codebase-scan-cooldown-seconds",
            "900",
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "2",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.llm_merge_resolver_command == "true"
    assert args.llm_merge_resolver_timeout_seconds == 5
    assert args.objective_path == repo / "objective-heap.md"
    assert args.objective_bundle_dir == repo / "objective_bundles"
    assert args.merge_reconciliation_max_merges == 9
    assert args.merged_worktree_cleanup_max == 13
    assert args.objective_scan_min_open_tasks == 20
    assert args.objective_scan_max_findings == 6
    assert args.objective_scan_cooldown_seconds == 900
    assert args.objective_surplus_findings_per_goal == 2
    assert args.objective_surplus_min_terms_per_todo == 4
    assert args.codebase_scan_min_open_tasks == 20
    assert args.codebase_scan_max_findings == 0
    assert args.codebase_scan_cooldown_seconds == 900
    assert args.task_shard_count == 4
    assert args.task_shard_index == 2


def test_configured_daemon_builder_preserves_shard_and_cleanup_args(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    args = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(repo / "todo.md"),
            "--state-dir",
            str(repo / "state"),
            "--state-prefix",
            "agent_lane_1",
            "--merged-worktree-cleanup-max",
            "13",
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "2",
        ]
    )

    daemon, context = build_portal_implementation_daemon_from_args(args, repo_root=repo)

    assert context.parsed is args
    assert daemon.merged_worktree_cleanup_max == 13
    assert daemon.task_shard_count == 4
    assert daemon.task_shard_index == 2


def test_daemon_refill_callbacks_honor_cli_scan_overrides(tmp_path):
    parsed = argparse.Namespace(
        todo_path=tmp_path / "tasks.todo.md",
        task_prefix="## EX-",
        objective_path=None,
        objective_scan_min_open_tasks=20,
        objective_scan_max_findings=6,
        objective_scan_cooldown_seconds=900,
        objective_surplus_findings_per_goal=2,
        objective_surplus_min_terms_per_todo=4,
        codebase_scan_min_open_tasks=20,
        codebase_scan_max_findings=0,
        codebase_scan_cooldown_seconds=900,
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

    assert objective_hook(context) == ["objective"]
    assert codebase_hook(context) == ["codebase"]
    assert captured["objective"]["min_open_tasks"] == 20
    assert captured["objective"]["max_findings"] == 6
    assert captured["objective"]["cooldown_seconds"] == 900
    assert captured["objective"]["surplus_findings_per_goal"] == 2
    assert captured["objective"]["surplus_min_terms_per_todo"] == 4
    assert captured["codebase"]["min_open_tasks"] == 20
    assert captured["codebase"]["max_findings"] == 0
    assert captured["codebase"]["cooldown_seconds"] == 900


def test_implementation_daemon_run_once_cleans_already_merged_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-001-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.txt").write_text("merged worktree payload\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "ACCEL-001: add feature")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", branch_name)
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-001-attempt-1-123"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Completed task

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: feature.txt
- Validation: test -f feature.txt
- Acceptance: Feature has already merged.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        worktree_root=worktree_root,
        merged_worktree_cleanup_max=5,
    )

    result = daemon.run_once()

    assert result["merged_worktree_cleanup"]["removed_count"] == 1
    assert not worktree_path.exists()
    branch_exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch_name],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert branch_exists.returncode != 0
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "merged_worktree_cleanup" for event in events)


def test_implementation_daemon_runs_validation_non_interactively(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    log_path = repo / "validation.log"
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        implementation_timeout=1,
    )
    task = PortalTask(
        task_id="AUTO-001",
        title="Validate stdin handling",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        validation=["python3 -c 'import sys; sys.stdin.read()'"],
    )

    result = daemon._run_validation_commands(repo, task, log_path)

    assert result["passed"] is True
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["timeout"] == 1


def test_implementation_daemon_selects_only_configured_task_shard(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-000 Even task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-001 Odd task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-002 Another even task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        task_shard_count=2,
        task_shard_index=1,
    )

    result = daemon.run_once()
    state = TodoTaskState.load(repo / "state.json")

    assert result["active_task_id"] == "ACCEL-001"
    assert state.recommended_task_id == "ACCEL-001"
    assert state.ready_count == 3


def test_implementation_daemon_borrows_ready_work_when_shard_drained(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-000 Blocked even task

- Status: blocked
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-001 Cross-shard ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-002 Another blocked even task

- Status: blocked
- Completion: manual
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        task_shard_count=2,
        task_shard_index=0,
    )

    result = daemon.run_once()
    state = TodoTaskState.load(repo / "state.json")

    assert result["active_task_id"] == "ACCEL-001"
    assert state.recommended_task_id == "ACCEL-001"
    assert state.ready_task_ids == ["ACCEL-001"]
    assert state.selectable_ready_task_ids == ["ACCEL-001"]
    events = (repo / "events.jsonl").read_text(encoding="utf-8")
    assert "task_shard_ready_fallback" in events


def test_implementation_daemon_filters_repo_wide_task_claims(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Claimed task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-003 Unclaimed task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )
    claim_path = daemon._implementation_task_claim_path("ACCEL-001")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(
        json.dumps(
            {
                "kind": "implementation_task_claim",
                "pid": os.getpid(),
                "repo_root": str(repo.resolve()),
                "task_id": "ACCEL-001",
            }
        ),
        encoding="utf-8",
    )

    result = daemon.run_once()
    state = TodoTaskState.load(repo / "state.json")

    assert result["active_task_id"] == "ACCEL-003"
    assert result["active_task_claims"] == ["ACCEL-001"]
    assert state.recommended_task_id == "ACCEL-003"
    assert state.ready_count == 2


def test_implementation_daemon_uses_shared_merge_receipts_across_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Completed in another lane

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-002 Waiting for another lane merge

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops

## ACCEL-003 Locally ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    queue = MergeQueue(repo / "merge-queue")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        merge_queue=queue,
    )
    tasks = {task.task_id: task for task in parse_task_file(todo_path, "## ACCEL-")}
    completed_request = queue.enqueue(
        branch_name="implementation/accel-001",
        task_id="OTHER-001",
        canonical_task_id=daemon._canonical_ref(tasks["ACCEL-001"]),
        commit_sha="a" * 40,
    )
    claimed = queue.dequeue(consumer_id="merge-train:test")
    assert claimed is not None and claimed.request_id == completed_request.request_id
    queue.complete(claimed)
    queue.enqueue(
        branch_name="implementation/accel-002",
        task_id="OTHER-002",
        canonical_task_id=daemon._canonical_ref(tasks["ACCEL-002"]),
        commit_sha="b" * 40,
    )
    daemon._consume_one_merge_candidate = lambda: None  # type: ignore[method-assign]

    result = daemon.run_once()
    state = TodoTaskState.load(daemon.state_path)

    assert result["active_task_id"] == "ACCEL-003"
    assert result["shared_completed_task_ids"] == ["ACCEL-001"]
    assert result["shared_active_merge_task_ids"] == ["ACCEL-002"]
    assert state.task_statuses["ACCEL-001"] == "completed"
    assert state.task_statuses["ACCEL-002"] == "waiting"
    assert state.task_statuses["ACCEL-003"] == "ready"


def test_implementation_daemon_skips_repo_wide_task_claim_collision(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
    )
    task = PortalTask(
        task_id="ACCEL-001",
        title="Claimed task",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
    )
    claim_path = daemon._implementation_task_claim_path(task.task_id)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(
        json.dumps(
            {
                "kind": "implementation_task_claim",
                "pid": os.getpid(),
                "repo_root": str(repo.resolve()),
                "state_dir": str((repo / "other-lane").resolve()),
                "task_id": task.task_id,
            }
        ),
        encoding="utf-8",
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["skipped"] is True
    assert result["reason"] == "task_claim_lock_exists"
    assert result["lock_owner_pid"] == os.getpid()
    assert result["lock_owner_task_id"] == task.task_id
    assert result["lock_owner_state_dir"] == str((repo / "other-lane").resolve())


def test_validation_command_splitter_preserves_quoted_semicolons():
    inline_python = (
        "python3 -c 'import pathlib, sys; "
        'p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        "' external/ipfs_kit/.github/workflows/auto-doc-maintenance.yml"
    )

    assert split_validation_commands(f"{inline_python}; test -f proof.txt") == [
        inline_python,
        "test -f proof.txt",
    ]


def test_parse_task_file_preserves_quoted_validation_semicolons(tmp_path):
    todo_path = tmp_path / "todo.md"
    inline_python = (
        "python3 -c 'import pathlib, sys; "
        'p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        "' src/config.yml"
    )
    todo_path.write_text(
        f"""# Todos

## ACCEL-001 Validate YAML

- Status: todo
- Validation: {inline_python}; test -f src/config.yml
""",
        encoding="utf-8",
    )

    task = parse_task_file(todo_path, task_header_prefix="## ACCEL-")[0]

    assert task.validation == [inline_python, "test -f src/config.yml"]


def test_implementation_daemon_clears_active_task_when_finished(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    state = TodoTaskState(
        active_task_id="ACCEL-001",
        active_task_title="Complete stale active task cleanup",
        active_task_track="runtime",
        active_task_started_at="2026-06-07T00:00:00+00:00",
        active_attempt=2,
        active_phase="validating",
        active_phase_started_at="2026-06-07T00:01:00+00:00",
        active_phase_detail="pytest",
        active_log_path=str(repo / "state" / "implementation.log"),
        active_worktree_path=str(repo / "worktrees" / "accel-001"),
        active_branch="implementation/accel-001",
        implementation_in_progress=True,
        recommended_task_id="ACCEL-001",
        recommended_actions=["Run validation"],
        last_implementation_task_id="ACCEL-001",
        last_implementation_started_at="2026-06-07T00:00:00+00:00",
        last_implementation_log_path=str(repo / "state" / "implementation.log"),
    )

    daemon._mark_implementation_finished(state, finished_at=finished_at)

    assert state.active_task_id == ""
    assert state.active_task_title == ""
    assert state.active_task_track == ""
    assert state.active_task_started_at == ""
    assert state.active_attempt == 0
    assert state.active_phase == ""
    assert state.active_phase_detail == ""
    assert state.active_log_path == ""
    assert state.active_worktree_path == ""
    assert state.active_branch == ""
    assert state.implementation_in_progress is False
    assert state.recommended_task_id == ""
    assert state.recommended_actions == []
    assert state.last_implementation_task_id == "ACCEL-001"
    assert state.last_implementation_log_path.endswith("implementation.log")
    assert state.heartbeat_at == finished_at
    assert state.last_progress_at == finished_at


def test_implementation_daemon_repairs_invalid_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    strategy_path = repo / "strategy.json"
    strategy_path.write_text("{not json", encoding="utf-8")
    events_path = repo / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["ready_count"] == 1
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_daemon_repairs_directory_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.mkdir(parents=True)
    (strategy_path / "fragment").write_text("old strategy directory\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon.run_once()

    assert result["reason"] == "no_tasks_found"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    backups = list(state_dir.glob("strategy.json.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_daemon_repairs_malformed_state_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_path = repo / "state.json"
    state_path.write_text(json.dumps({"active_attempt": "not-an-int"}), encoding="utf-8")
    events_path = repo / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=repo / "strategy.json",
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["state_file_repair"]["repaired"] is True
    assert result["state_file_repair"]["reason"] == "malformed_state_metadata"
    assert result["ready_count"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_attempt"] == 0
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "state_file_repaired" for event in events)


def test_implementation_daemon_repairs_malformed_event_log(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    events_path = repo / "events.jsonl"
    events_path.write_text(
        json.dumps({"type": "seed", "task_id": "AUTO-000"}) + "\nnot json\n[]\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["event_log_repair"]["repaired"] is True
    assert result["event_log_repair"]["reason"] == "malformed_jsonl"
    assert result["event_log_repair"]["invalid_count"] == 2
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "seed"
    assert any(event["type"] == "event_log_repaired" for event in events)
    quarantines = list(repo.glob("events.jsonl.invalid-jsonl-*"))
    assert len(quarantines) == 1


def test_implementation_daemon_moves_directory_lock_and_acquires_lock(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    lock_path = state_dir / "implementation.lock"
    lock_path.mkdir(parents=True)
    (lock_path / "fragment").write_text("not lock json\n", encoding="utf-8")
    events_path = state_dir / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=events_path,
        repo_root=repo,
    )

    fd, reason, existing = daemon._try_acquire_lock(
        lock_path,
        lock_kind="implementation",
        owner_active=lambda _metadata: False,
    )
    if fd is not None:
        import os

        os.close(fd)

    assert reason == "acquired"
    assert existing is None
    assert lock_path.is_file()
    backups = list(state_dir.glob("implementation.lock.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    lock_events = [event for event in events if event["type"] == "implementation_lock_cleared"]
    assert len(lock_events) == 1
    assert lock_events[0]["moved_directory_path"] == str(backups[0])


def test_implementation_supervisor_repairs_invalid_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text("{not json", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["strategy_file_repair"]["repaired"] is True
    assert result["strategy_file_repair"]["reason"] == "invalid_or_unreadable_strategy_file"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_supervisor_repairs_malformed_state_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"active_attempt": "not-an-int"}), encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["state_file_repair"]["repaired"] is True
    assert result["state_file_repair"]["reason"] == "malformed_state_metadata"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_attempt"] == 0
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "state_file_repaired" for event in events)


def test_implementation_supervisor_run_once_refreshes_recovery_status(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed task

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f todo.md
- Acceptance: Completed seed task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            objective_refill_enabled=True,
        )
    )
    status_path = state_dir / "portal_supervisor_status.json"
    observed_statuses = []

    def fake_objective_refill():
        observed_statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
        return {}

    monkeypatch.setattr(supervisor, "refill_objective_backlog", fake_objective_refill)

    supervisor.run_once()

    assert observed_statuses
    assert observed_statuses[0]["status"] == "agentic_maintenance_started"
    assert observed_statuses[0]["last_agentic_maintenance_status"] == "running"
    assert observed_statuses[0]["last_agentic_maintenance_phase"] == "objective_refill"
    assert observed_statuses[0]["supervisor_pid_alive"] is True
    assert observed_statuses[0]["daemon_pid_alive"] is False
    assert observed_statuses[0]["active_agentic_maintenance_timeout_seconds"] >= 300.0
    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert final_status["status"] == "agentic_maintenance_completed"
    assert final_status["last_agentic_maintenance_status"] == "completed"


def test_implementation_supervisor_watchdog_refreshes_child_maintenance_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    update_phase, finish = supervisor._begin_supervisor_maintenance_heartbeat(
        "watchdog",
        daemon_pid=os.getpid(),
    )
    update_phase("objective_refill")
    finish("completed")

    status = json.loads((state_dir / "portal_supervisor_status.json").read_text(encoding="utf-8"))
    assert status["daemon_pid"] == os.getpid()
    assert status["daemon_pid_alive"] is True
    assert status["active_agentic_maintenance_has_daemon"] is True
    assert status["last_agentic_maintenance_phase"] == "objective_refill"
    assert status["last_agentic_maintenance_status"] == "completed"


def test_implementation_supervisor_watchdog_throttles_maintenance(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    state_dir.mkdir()
    TodoTaskState(ready_count=0, completed_count=1).save(state_dir / "task_state.json")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            check_interval=60,
        )
    )
    calls = []

    def fake_maintenance(_update_phase):
        calls.append("maintenance")
        return {"stuck": False, "main_checkout_repair": {"repaired": False}}

    class Child:
        pid = os.getpid()

    monkeypatch.setattr(supervisor, "_run_once_with_maintenance", fake_maintenance)

    first = supervisor._supervisor_loop_watchdog_decision(None, Child(), {})
    second = supervisor._supervisor_loop_watchdog_decision(None, Child(), {})

    assert first.action == "continue"
    assert second.action == "continue"
    assert calls == ["maintenance"]


def test_implementation_supervisor_watchdog_skips_active_progress_maintenance(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    now = datetime.now(timezone.utc)
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Active task",
        active_task_track="ops",
        active_task_started_at=now.isoformat(),
        active_attempt=1,
        active_phase="implementing",
        active_phase_started_at=now.isoformat(),
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-001",
        last_implementation_started_at=now.isoformat(),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            check_interval=60,
        )
    )
    calls = []

    class Child:
        pid = os.getpid()

    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda _update_phase: calls.append("maintenance") or {"stuck": False},
    )

    decision = supervisor._supervisor_loop_watchdog_decision(None, Child(), {})

    assert decision.action == "continue"
    assert calls == []


def test_implementation_supervisor_check_records_worktree_summary_counts(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            retry_budget_guardrail_enabled=False,
            dependency_guardrail_enabled=False,
            reconciliation_guardrail_enabled=False,
        )
    )
    monkeypatch.setattr(
        supervisor,
        "reconcile_backlogged_worktrees",
        lambda: {
            "candidate_count": 5,
            "processed_count": 3,
            "reconciled_count": 1,
            "preflight_blocked_count": 2,
        },
    )
    monkeypatch.setattr(
        supervisor,
        "cleanup_backlogged_worktrees",
        lambda: {
            "removed_count": 4,
            "dirty_worktree_groups": {"content_not_in_target": {"count": 7}},
        },
    )

    supervisor.run_once()

    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    check = [event for event in events if event["type"] == "supervisor_check"][-1]
    assert check["worktree_reconciliation_candidate_count"] == 5
    assert check["worktree_reconciliation_processed_count"] == 3
    assert check["worktree_reconciliation_reconciled_count"] == 1
    assert check["worktree_reconciliation_preflight_blocked_count"] == 2
    assert check["worktree_cleanup_removed_count"] == 4
    assert check["worktree_cleanup_dirty_group_count"] == 1


def test_implementation_supervisor_repairs_event_log_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    events_path = state_dir / "supervisor_events.jsonl"
    events_path.mkdir(parents=True)
    (events_path / "old-event-fragment").write_text("not json\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=events_path,
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["event_log_repair"]["repaired"] is True
    assert result["event_log_repair"]["reason"] == "event_path_was_directory"
    assert events_path.is_file()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "event_log_repaired" for event in events)
    backup_path = Path(result["event_log_repair"]["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "old-event-fragment").exists()


def test_implementation_supervisor_repairs_directory_managed_pid_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    pid_path = state_dir / "portal_managed_daemon.pid"
    pid_path.mkdir(parents=True)
    (pid_path / "fragment").write_text("not a pid\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == "managed_pid_path_was_directory"
    assert not pid_path.exists()
    backup_path = Path(result["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "fragment").exists()
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "managed_daemon_pid_file_repaired"


def test_implementation_supervisor_repoints_mismatched_managed_pid_to_matching_daemon(
    tmp_path,
    monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as supervisor_module

    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    todo_path = repo / "todo.md"
    daemon_script = repo / "daemon.py"
    pid_path = state_dir / "portal_managed_daemon.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("111\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            daemon_script_path=daemon_script,
            implement=True,
        )
    )
    matching_command = (
        f"python {daemon_script} --state-dir {state_dir} --state-prefix portal "
        f"--todo-path {todo_path} --implement"
    )

    monkeypatch.setattr(supervisor_module, "process_is_running", lambda pid: int(pid) in {111, 222})
    monkeypatch.setattr(
        supervisor_module,
        "process_command_line",
        lambda pid: (
            f"python {daemon_script} --state-dir {state_dir} --state-prefix portal "
            f"--todo-path {todo_path}"
        ),
    )
    monkeypatch.setattr(supervisor, "_list_process_details", lambda: [(111, "wrong daemon"), (222, matching_command)])

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == "managed_pid_command_mismatch_replaced_with_matching_daemon"
    assert result["replacement_pid"] == 222
    assert pid_path.read_text(encoding="utf-8").strip() == "222"


def test_implementation_supervisor_repairs_stale_managed_pid_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    pid_path = state_dir / "portal_managed_daemon.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("999999999\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == "stale_managed_pid"
    assert not pid_path.exists()
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "managed_daemon_pid_file_repaired"
    assert events[-1]["pid"] == 999999999


def test_implementation_supervisor_passes_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon_script = repo / "custom_daemon.py"
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        state_dir=repo / "state",
        implement=True,
        llm_merge_resolver_command="true",
        llm_merge_resolver_timeout_seconds=5,
        worktree_submodule_paths=("packages/app", "external/lib"),
        objective_path=repo / "objective-heap.md",
        objective_bundle_dir=repo / "objective_bundles",
        objective_refill_enabled=True,
        objective_scan_min_open_tasks=20,
        objective_scan_max_findings=6,
        objective_scan_cooldown_seconds=900,
        objective_surplus_findings_per_goal=2,
        objective_surplus_min_terms_per_todo=4,
        codebase_refill_enabled=True,
        codebase_scan_min_open_tasks=20,
        codebase_scan_max_findings=0,
        codebase_scan_cooldown_seconds=900,
        merge_reconciliation_max_merges=0,
        daemon_merged_worktree_cleanup_max=17,
        task_shard_count=2,
        task_shard_index=1,
        daemon_script_path=daemon_script,
    )
    supervisor = TodoImplementationSupervisor(config)

    command = supervisor._build_daemon_command()

    assert command[:2] == [sys.executable, str(daemon_script)]
    assert command.count("--worktree-submodule-path") == 2
    assert "packages/app" in command
    assert "external/lib" in command
    assert command[command.index("--llm-merge-resolver-command") + 1] == "true"
    assert command[command.index("--llm-merge-resolver-timeout-seconds") + 1] == "5"
    assert command[command.index("--objective-path") + 1] == str(repo / "objective-heap.md")
    assert command[command.index("--objective-bundle-dir") + 1] == str(repo / "objective_bundles")
    assert command[command.index("--objective-scan-min-open-tasks") + 1] == "20"
    assert command[command.index("--objective-scan-max-findings") + 1] == "6"
    assert command[command.index("--objective-scan-cooldown-seconds") + 1] == "900"
    assert command[command.index("--objective-surplus-findings-per-goal") + 1] == "2"
    assert command[command.index("--objective-surplus-min-terms-per-todo") + 1] == "4"
    assert command[command.index("--codebase-scan-min-open-tasks") + 1] == "20"
    assert command[command.index("--codebase-scan-max-findings") + 1] == "0"
    assert command[command.index("--codebase-scan-cooldown-seconds") + 1] == "900"
    assert command[command.index("--merge-reconciliation-max-merges") + 1] == "0"
    assert command[command.index("--merged-worktree-cleanup-max") + 1] == "17"
    assert command[command.index("--task-shard-count") + 1] == "2"
    assert command[command.index("--task-shard-index") + 1] == "1"

    args = parse_implementation_supervisor_args(
        [
            "--implement",
            "--todo-path",
            str(repo / "todo.md"),
            "--llm-merge-resolver-command",
            "resolver-command",
            "--llm-merge-resolver-timeout-seconds",
            "7",
            "--daemon-script-path",
            str(repo / "daemon.py"),
            "--supervisor-script-path",
            str(repo / "supervisor.py"),
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
            "--codebase-refill-scan",
            "--dependency-guardrail-discovery-dir",
            str(repo / "dependency-discovery"),
            "--dependency-guardrail-discovery-output-path",
            "dependency-discovery",
            "--dependency-guardrail-max-findings",
            "2",
            "--codebase-scan-min-open-tasks",
            "0",
            "--codebase-scan-depends-on",
            "AUTO-001,AUTO-002",
            "--objective-path",
            str(repo / "objective-heap.md"),
            "--objective-bundle-dir",
            str(repo / "objective_bundles"),
            "--merge-reconciliation-max-merges",
            "0",
            "--daemon-merged-worktree-cleanup-max",
            "19",
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "1",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.llm_merge_resolver_command == "resolver-command"
    assert args.llm_merge_resolver_timeout_seconds == 7
    assert args.daemon_script_path == repo / "daemon.py"
    assert args.supervisor_script_path == repo / "supervisor.py"
    assert args.codebase_refill_scan is True
    assert args.codebase_scan_depends_on == ["AUTO-001,AUTO-002"]
    assert args.dependency_guardrail_discovery_dir == repo / "dependency-discovery"
    assert args.dependency_guardrail_discovery_output_path == "dependency-discovery"
    assert args.dependency_guardrail_max_findings == 2
    assert args.objective_path == repo / "objective-heap.md"
    assert args.objective_bundle_dir == repo / "objective_bundles"
    assert args.merge_reconciliation_max_merges == 0
    assert args.daemon_merged_worktree_cleanup_max == 19
    assert args.task_shard_count == 2
    assert args.task_shard_index == 1


def test_implementation_supervisor_does_not_recycle_active_merge_resolver(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        state_dir=repo / "state",
        stale_seconds=60,
    )
    supervisor = TodoImplementationSupervisor(config)
    now = datetime.now(timezone.utc)
    state = TodoTaskState(
        active_task_id="AUTO-001",
        active_phase="merge_resolver",
        active_phase_detail="merge_conflict",
        heartbeat_at=now.isoformat(),
        last_progress_at="2000-01-01T00:00:00+00:00",
        ready_count=1,
    )

    stuck, reason = supervisor.is_stuck(state, now_ts=now.timestamp())

    assert stuck is False
    assert reason == ""


def test_implementation_supervisor_repairs_stale_merge_resolver_without_worker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    now = datetime.now(timezone.utc)
    old = now.replace(year=2000)
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Stale merge resolver",
        active_task_track="ops",
        active_task_started_at=old.isoformat(),
        active_attempt=1,
        active_phase="merge_resolver",
        active_phase_started_at=old.isoformat(),
        active_phase_detail="merge_conflict",
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-001",
        last_implementation_started_at=old.isoformat(),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        check_interval=1,
        stale_seconds=3600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert "merge_resolver stalled" in result["reason"]
    assert result["state_repair"]["repaired"] is True
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "worktree_phase_without_worker" for event in events)


def test_implementation_supervisor_repairs_generated_dirty_after_stuck_recovery(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    now = datetime.now(timezone.utc)
    old = now.replace(year=2000)
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Stale merge resolver",
        active_task_track="ops",
        active_task_started_at=old.isoformat(),
        active_attempt=1,
        active_phase="merge_resolver",
        active_phase_started_at=old.isoformat(),
        active_phase_detail="merge_conflict",
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-001",
        last_implementation_started_at=old.isoformat(),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            check_interval=1,
            stale_seconds=3600,
            generated_dirty_repair_enabled=True,
        )
    )
    repair_calls: list[int] = []

    def fake_generated_repair() -> dict[str, object]:
        repair_calls.append(len(repair_calls) + 1)
        return {
            "attempted": True,
            "selected_path_count": 1,
            "committed_count": 1,
            "call_index": repair_calls[-1],
        }

    monkeypatch.setattr(supervisor, "repair_generated_dirty_checkouts", fake_generated_repair)

    result = supervisor.run_once()

    assert result["stuck"] is True
    assert repair_calls == [1, 2]
    assert result["generated_dirty_repair"]["call_index"] == 1
    assert result["post_stuck_generated_dirty_repair"]["call_index"] == 2


def test_implementation_supervisor_repairs_implementation_without_worker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    now = datetime.now(timezone.utc)
    started = now - timedelta(minutes=2)
    TodoTaskState(
        active_task_id="AUTO-002",
        active_task_title="Stale implementation worker",
        active_task_track="ops",
        active_task_started_at=started.isoformat(),
        active_attempt=1,
        active_phase="implementing",
        active_phase_started_at=started.isoformat(),
        active_phase_detail="agent worker",
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-002",
        last_implementation_started_at=started.isoformat(),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        check_interval=1,
        stale_seconds=3600,
        implementation_timeout=3600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert "implementing stalled" in result["reason"]
    assert result["state_repair"]["repaired"] is True
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    assert repaired_state.implementation_in_progress is False


def test_implementation_supervisor_prefers_worker_stall_over_log_stall(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    log_dir = state_dir / "implementation_logs"
    log_dir.mkdir()
    log_path = log_dir / "auto-003-attempt-1.log"
    log_path.write_text("stale output\n", encoding="utf-8")
    now = datetime.now(timezone.utc)
    started = now - timedelta(minutes=2)
    os.utime(log_path, (started.timestamp(), started.timestamp()))
    TodoTaskState(
        active_task_id="AUTO-003",
        active_task_title="Stale implementation worker with stale log",
        active_task_track="ops",
        active_task_started_at=started.isoformat(),
        active_attempt=1,
        active_phase="implementing",
        active_phase_started_at=started.isoformat(),
        active_phase_detail="agent worker",
        active_log_path=str(log_path),
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-003",
        last_implementation_started_at=started.isoformat(),
        last_implementation_log_path=str(log_path),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        check_interval=1,
        stale_seconds=3600,
        implementation_timeout=3600,
        implementation_log_stall_seconds=1,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert "implementing stalled" in result["reason"]
    assert "implementation log stalled" not in result["reason"]
    assert result["state_repair"]["repaired"] is True
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    assert repaired_state.implementation_in_progress is False


def test_supervisor_worker_watchdog_detects_active_merge_resolver_without_worker(tmp_path):
    now = datetime.now(timezone.utc)
    old = now.replace(year=2000)

    status = worktree_phase_worker_status(
        {
            "active_phase": "merge_resolver",
            "active_phase_started_at": old.isoformat(),
        },
        daemon_pid=999999999,
        threshold_seconds=60,
        now=now,
    )

    assert status["required"] is True
    assert status["phase"] == "merge_resolver"
    assert status["active_worker_count"] == 0
    assert status["stalled_without_active_worker"] is True


def test_implementation_supervisor_configures_worker_stall_watchdog(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        implementation_log_stall_seconds=42,
    )

    loop_config = TodoImplementationSupervisor(config).build_supervisor_loop_config()

    assert loop_config.status_static_fields["worktree_no_child_stall_seconds"] == 42


def test_implementation_supervisor_repairs_stale_active_state_after_rewrite(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    stale_timestamp = "2000-01-01T00:00:00+00:00"
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Stale active task",
        active_task_track="ops",
        active_task_started_at=stale_timestamp,
        active_attempt=2,
        active_phase="implementing",
        active_phase_started_at=stale_timestamp,
        active_phase_detail="stale worker",
        active_log_path=str(state_dir / "implementation.log"),
        active_worktree_path=str(repo / "worktrees" / "auto-001"),
        active_branch="implementation/auto-001",
        implementation_in_progress=False,
        heartbeat_at=stale_timestamp,
        last_progress_at=stale_timestamp,
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        stale_seconds=60,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert result["state_repair"]["repaired"] is True
    assert result["state_repair"]["active_task_id"] == "AUTO-001"
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    assert repaired_state.implementation_in_progress is False
    assert repaired_state.active_worktree_path == ""
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["deprioritized_tasks"] == ["AUTO-001"]
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "blocked_progress_state_repaired" for event in events)


def test_implementation_daemon_keeps_alive_on_empty_todo_board(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon.run_once()

    assert result["reason"] == "no_tasks_found"
    assert result["task_count"] == 0
    state = TodoTaskState.load(state_dir / "task_state.json")
    assert state.active_task_id == ""
    assert state.ready_count == 0
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "daemon_no_tasks"


def test_implementation_daemon_records_unreadable_todo_text(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_bytes(b"\xff\xfe\x00")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon.run_once()

    assert result["reason"] == "todo_read_failed"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "daemon_no_tasks"
    assert events[-1]["reason"] == "todo_read_failed"


def test_implementation_daemon_records_non_ephemeral_setup_exception(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Record command failure

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs:
- Validation:
- Acceptance: The daemon records setup failures as durable events.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
        implementation_command="missing-implementation-command-for-test",
        use_ephemeral_worktree=False,
    )

    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["returncode"] == 1
    assert implementation["exception_result"]["exception_type"] == "FileNotFoundError"
    assert implementation["exception_result"]["phase"] == "implementing"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "implementation_exception" for event in events)
    assert events[-1]["type"] == "daemon_pass"


def test_implementation_supervisor_creates_missing_todo_before_refill(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime_bridge.py"
    source.parent.mkdir()
    source.write_text(
        """class RuntimeBridge:
    def dispatch(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime bridge proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that runtime bridge dispatch supports virtual AI OS execution.
- Evidence: RuntimeBridge.dispatch, missing_runtime_contract
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime contract proof.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective heap")
    todo_path = repo / "missing-todo.md"
    state_dir = repo / "state"

    result = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_persist_ast_dataset=False,
        )
    ).run_once()

    assert result["todo_board_repair"]["created"] is True
    assert result["objective_refill_count"] == 1
    assert todo_path.exists()
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "todo_board_created" for event in events)


def test_implementation_supervisor_repairs_directory_todo_before_refill(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Directory todo repair

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that todo board directory repair bootstraps refill.
- Evidence: missing_directory_todo_repair
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the directory todo repair proof.
""",
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "runtime_bridge.py").write_text("class RuntimeBridge: pass\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective heap")
    todo_path = repo / "todo.md"
    todo_path.mkdir()
    (todo_path / "fragment").write_text("not a todo board\n", encoding="utf-8")
    state_dir = repo / "state"

    result = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_persist_ast_dataset=False,
        )
    ).run_once()

    assert result["todo_board_repair"]["repaired"] is True
    assert result["todo_board_repair"]["reason"] == "todo_path_was_directory"
    assert todo_path.is_file()
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    backup_path = Path(result["todo_board_repair"]["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "fragment").exists()
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "todo_board_repaired" for event in events)


def test_implementation_supervisor_records_dependency_guardrail(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Waiting on missing dependency

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-999
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task should become ready when dependencies are valid.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
            dependency_guardrail_discovery_dir=repo / "dependency-discovery",
            dependency_guardrail_max_findings=1,
        )
    )

    result = supervisor.run_once()

    assert result["dependency_guardrail_count"] == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve dependency guardrail for AUTO-001" in todo_text
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["dependency_guardrail_findings"][0]["missing_dependencies"] == ["AUTO-999"]
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "dependency_guardrail" for event in supervisor_events)


def test_implementation_supervisor_releases_completed_guardrail_block(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original blocked task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Original task should resume after repair.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Repair task completed.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001"],
                "retry_budget_findings": [
                    {
                        "source_task_id": "AUTO-001",
                        "follow_up_task_id": "AUTO-002",
                        "failure_kind": "implementation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
        )
    )

    result = supervisor.run_once()

    assert result["guardrail_unblock_count"] == 1
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["guardrail_unblock_releases"][0]["source_task_id"] == "AUTO-001"
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "guardrail_blocks_released" for event in supervisor_events)


def test_implementation_supervisor_releases_stale_strategy_blocks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed source

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Already complete.

## AUTO-002 Still blocked source

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Remains blocked.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps({"blocked_tasks": ["AUTO-001", "AUTO-999", "AUTO-002", "AUTO-002"]}),
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
        )
    )

    result = supervisor.run_once()

    assert result["guardrail_unblock_count"] == 3
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-002"]
    reasons = {release["reason"] for release in strategy["guardrail_unblock_releases"]}
    assert reasons == {"duplicate_strategy_block", "missing_task", "source_completed"}
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "guardrail_blocks_released" for event in supervisor_events)


def test_implementation_daemon_records_worktree_setup_exception(tmp_path):
    repo = tmp_path / "not-a-git-repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Record setup failure

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs:
- Validation:
- Acceptance: The daemon records setup failures as durable events.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
        implementation_command="true",
        use_ephemeral_worktree=True,
        worktree_root=repo / "worktrees",
    )

    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["returncode"] == 1
    assert implementation["exception_result"]["exception_type"] == "RuntimeError"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "implementation_exception" for event in events)
    assert events[-1]["type"] == "daemon_pass"


def test_implementation_daemon_records_merge_reconcile_exception(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "task_id": "ACCEL-002",
        "attempt": 3,
        "branch": "implementation/accel-002",
        "implementation_commit": "abc123",
        "title": "Recover failed merge",
    }

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: True  # type: ignore[method-assign]

    def raise_merge_error(*args, **kwargs):
        raise RuntimeError("merge workspace unavailable")

    daemon._merge_branch_to_main = raise_merge_error  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert result == [
        {
            "task_id": "ACCEL-002",
            "attempt": 3,
            "branch": "implementation/accel-002",
            "implementation_commit": "abc123",
            "merge_ref": "implementation/accel-002",
            "merge_ref_source": "branch",
            "resolved": False,
            "reason": "merge_reconcile_exception",
            "exception_type": "RuntimeError",
            "error": "merge workspace unavailable",
        }
    ]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconcile_exception"


def test_implementation_daemon_defers_merge_reconciliation_when_main_checkout_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, text=True, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "agent@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo, check=True)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo, text=True, capture_output=True, check=True)
    (repo / "dirty.txt").write_text("dirty checkout\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": "ACCEL-002",
        "attempt": 1,
        "branch": "implementation/accel-002",
        "implementation_commit": "abc123",
        "title": "Recover failed merge",
    }
    merge_attempts: list[str] = []

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._merge_branch_to_main = lambda branch, task, attempt, baseline_ref="": merge_attempts.append(branch)  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert merge_attempts == []
    assert result == [
        {
            "resolved": False,
            "reason": "main_checkout_dirty",
            "candidate_count": 1,
            "processed_count": 0,
            "dirty_paths": ["dirty.txt"],
        }
    ]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconciliation_deferred"
    assert events[-1]["reason"] == "main_checkout_dirty"


def test_implementation_daemon_abandons_stale_failed_merge_candidates(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        merge_reconciliation_max_age_seconds=60,
    )
    event = {
        "type": "implementation_finished",
        "timestamp": "2000-01-01T00:00:00+00:00",
        "task_id": "ACCEL-002",
        "attempt": 1,
        "branch": "implementation/accel-002",
        "implementation_commit": "abc123",
        "title": "Recover stale merge",
    }
    merge_attempts: list[str] = []

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._merge_branch_to_main = lambda branch, task, attempt, baseline_ref="": merge_attempts.append(branch)  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert merge_attempts == []
    assert result[0]["reason"] == "stale_failed_merge_candidate"
    assert result[0]["task_id"] == "ACCEL-002"
    assert result[0]["implementation_commit"] == "abc123"
    assert result[0]["merge_result"]["attempted"] is False
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconciled"
    assert events[-1]["reason"] == "stale_failed_merge_candidate"


def test_implementation_daemon_reconciles_missing_branch_from_commit_ref(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "task_id": "ACCEL-005",
        "attempt": 1,
        "branch": "implementation/accel-005",
        "implementation_commit": "abc123",
        "title": "Recover missing branch merge",
    }
    merged_refs: list[str] = []

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: ref == "abc123"  # type: ignore[method-assign]

    def fake_merge(ref, task, attempt, baseline_ref=""):
        merged_refs.append(ref)
        return {"merged": True, "merge_commit": "merge456"}

    daemon._merge_branch_to_main = fake_merge  # type: ignore[method-assign]
    daemon._cleanup_merged_worktree = lambda worktree_path, branch: {  # type: ignore[method-assign]
        "cleaned": True,
        "branch": branch,
        "worktree_path": str(worktree_path or ""),
    }

    result = daemon._reconcile_failed_merges()

    assert merged_refs == ["abc123"]
    assert result[0]["resolved"] is True
    assert result[0]["merge_ref"] == "abc123"
    assert result[0]["merge_ref_source"] == "implementation_commit"
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "merge_reconcile_ref_recovered" for event in events)
    assert events[-1]["type"] == "merge_reconciled"


def test_implementation_daemon_reconciled_merge_requires_cleanup_success(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "task_id": "ACCEL-006",
        "attempt": 1,
        "branch": "implementation/accel-006",
        "implementation_commit": "abc123",
        "worktree_path": str(repo / "worktrees" / "accel-006"),
        "title": "Retry merge cleanup",
    }

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: ref == "implementation/accel-006"  # type: ignore[method-assign]
    daemon._merge_branch_to_main = lambda branch, task, attempt, baseline_ref="": {  # type: ignore[method-assign]
        "merged": True,
        "merge_commit": "merge456",
    }
    daemon._cleanup_merged_worktree = lambda worktree_path, branch: {  # type: ignore[method-assign]
        "cleaned": False,
        "branch": branch,
        "worktree_path": str(worktree_path or ""),
        "error": "submodule cleanup failed",
    }

    result = daemon._reconcile_failed_merges()

    assert result[0]["resolved"] is False
    assert result[0]["reason"] == "cleanup_retry_failed"
    assert result[0]["cleanup_result"]["error"] == "submodule cleanup failed"
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconciled"
    assert events[-1]["resolved"] is False


def test_implementation_daemon_recovers_missing_inflight_before_merge_reconciliation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_path = repo / "state" / "task_state.json"
    TodoTaskState(
        active_task_id="ACCEL-999",
        active_task_title="Stale validating task",
        active_task_track="ops",
        active_task_started_at="2026-06-07T00:00:00+00:00",
        active_attempt=2,
        active_phase="validating",
        active_phase_started_at="2026-06-07T00:01:00+00:00",
        active_phase_detail="pytest",
        implementation_in_progress=True,
        last_implementation_task_id="ACCEL-999",
        last_implementation_started_at="2026-06-07T00:00:00+00:00",
    ).save(state_path)
    strategy_path = repo / "state" / "strategy.json"
    strategy_path.write_text(
        json.dumps({"deprioritized_tasks": ["ACCEL-001"]}),
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )
    daemon._find_live_inflight_implementation = lambda: None  # type: ignore[method-assign]

    def assert_state_recovered_before_reconcile(*, skip_task_ids=None, deprioritized_task_ids=None):
        recovered = TodoTaskState.load(state_path)
        assert recovered.implementation_in_progress is False
        assert recovered.active_phase == ""
        assert deprioritized_task_ids == set()
        return []

    daemon._reconcile_failed_merges = assert_state_recovered_before_reconcile  # type: ignore[method-assign]

    result = daemon.run_once()

    assert result["ready_count"] == 1
    recovered = TodoTaskState.load(state_path)
    assert recovered.implementation_in_progress is False
    assert recovered.active_task_id == "ACCEL-001"
    assert recovered.active_phase == ""
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "implementation_state_recovered" for event in events)


def test_implementation_daemon_ignores_task_local_service_processes_as_inflight(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    worktree_path = repo / "worktrees" / "accel-999-attempt-1"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {"worktree_path": str(worktree_path), "command": ["bash", "-lc", "runner"]}
    monkeypatch.setattr(
        daemon,
        "_list_process_commands",
        lambda: [
            f"node {worktree_path}/swissknife/scripts/start-ipfs-kit-mcp-compat.cjs --port 8014",
            f"python {worktree_path}/swissknife/scripts/ipfs_mcp_libp2p_bridge.py --port 9114",
        ],
    )

    assert daemon._implementation_process_active(event) is False

    monkeypatch.setattr(
        daemon,
        "_list_process_commands",
        lambda: [f"node /usr/local/bin/codex exec -C {worktree_path} -"],
    )

    assert daemon._implementation_process_active(event) is True


def test_implementation_daemon_recognizes_shared_checkout_runner_without_worktree_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "worktree_path": "",
        "command": ["bash", "-lc", "serialized wrapper command"],
    }
    monkeypatch.setattr(
        daemon,
        "_list_process_commands",
        lambda: [
            f"node /usr/local/bin/codex exec -C {repo} -",
            f"python {repo}/swissknife/scripts/ipfs_mcp_libp2p_bridge.py --port 9114",
        ],
    )

    assert daemon._implementation_process_active(event) is True


def test_implementation_supervisor_recovers_missing_inflight_before_worktree_reconciliation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    active_worktree = repo / "worktrees" / "accel-999-attempt-1"
    TodoTaskState(
        active_task_id="ACCEL-999",
        active_task_title="Stale implementation task",
        active_task_track="ops",
        active_task_started_at="2026-06-07T00:00:00+00:00",
        active_attempt=1,
        active_phase="implementing",
        active_phase_started_at="2026-06-07T00:00:00+00:00",
        active_worktree_path=str(active_worktree),
        active_branch="implementation/accel-999-attempt-1",
        implementation_in_progress=True,
        last_implementation_task_id="ACCEL-999",
        last_implementation_started_at="2026-06-07T00:00:00+00:00",
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=repo / "worktrees",
        )
    )
    supervisor._list_process_commands = lambda: []  # type: ignore[method-assign]

    def assert_state_recovered_before_reconcile():
        recovered = TodoTaskState.load(state_path)
        assert recovered.implementation_in_progress is False
        assert recovered.active_worktree_path == ""
        assert recovered.active_branch == ""
        return {"attempted": True, "candidate_count": 0, "processed_count": 0, "reconciled_count": 0}

    supervisor.reconcile_backlogged_worktrees = assert_state_recovered_before_reconcile  # type: ignore[method-assign]
    supervisor.cleanup_backlogged_worktrees = lambda: {"attempted": True, "removed_count": 0}  # type: ignore[method-assign]

    result = supervisor.run_once()

    assert result["stale_active_state_repair"]["repaired"] is True
    assert result["stale_active_state_repair"]["active_task_id"] == "ACCEL-999"
    recovered = TodoTaskState.load(state_path)
    assert recovered.active_task_id == "ACCEL-999"
    assert recovered.implementation_in_progress is False
    assert recovered.active_worktree_path == ""
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "stale_active_execution_state_repaired" for event in events)


def test_implementation_supervisor_ignores_task_local_service_processes_when_repairing_stale_state(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    active_worktree = repo / "worktrees" / "accel-999-attempt-1"
    TodoTaskState(
        active_task_id="ACCEL-999",
        active_task_title="Stale implementation task",
        active_task_track="ops",
        active_attempt=1,
        active_phase="validating",
        active_worktree_path=str(active_worktree),
        active_branch="implementation/accel-999-attempt-1",
        implementation_in_progress=True,
    ).save(state_path)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=repo / "worktrees",
        )
    )
    monkeypatch.setattr(supervisor, "_read_managed_daemon_pid", lambda: 0)
    monkeypatch.setattr(
        supervisor,
        "_list_process_commands",
        lambda: [
            f"node {active_worktree}/swissknife/scripts/start-ipfs-accelerate-mcp-compat.cjs --port 3003",
            f"python {active_worktree}/swissknife/scripts/ipfs_mcp_libp2p_bridge.py --port 9114",
        ],
    )

    result = supervisor.repair_stale_active_execution_state()

    assert result["repaired"] is True
    recovered = TodoTaskState.load(state_path)
    assert recovered.implementation_in_progress is False
    assert recovered.active_worktree_path == ""


def test_implementation_daemon_limits_merge_reconciliation_per_pass(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        merge_reconciliation_max_merges=2,
    )
    events = [
        {
            "task_id": f"ACCEL-{index:03d}",
            "attempt": 1,
            "branch": f"implementation/accel-{index:03d}",
            "implementation_commit": f"commit{index}",
            "title": "Recover failed merge",
        }
        for index in range(1, 5)
    ]
    merged_branches: list[str] = []

    daemon._failed_merge_candidates = lambda skip_task_ids=None: events  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: True  # type: ignore[method-assign]

    def fake_merge(branch, task, attempt, baseline_ref=""):
        merged_branches.append(branch)
        return {"merged": False, "reason": "test_conflict"}

    daemon._merge_branch_to_main = fake_merge  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert [item["task_id"] for item in result] == ["ACCEL-001", "ACCEL-002"]
    assert merged_branches == ["implementation/accel-001", "implementation/accel-002"]
    recorded_events = [
        json.loads(line)
        for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    deferred = [event for event in recorded_events if event["type"] == "merge_reconciliation_deferred"][-1]
    assert deferred["candidate_count"] == 4
    assert deferred["processed_count"] == 2
    assert deferred["deferred_count"] == 2


def test_implementation_daemon_zero_merge_reconciliation_disables_failed_merge_retry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        merge_reconciliation_max_merges=0,
    )
    event = {
        "task_id": "ACCEL-001",
        "attempt": 1,
        "branch": "implementation/accel-001",
        "implementation_commit": "commit1",
        "title": "Recover failed merge",
    }
    merge_attempts: list[str] = []

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: True  # type: ignore[method-assign]

    def fake_merge(branch, task, attempt, baseline_ref=""):
        merge_attempts.append(branch)
        return {"merged": False, "reason": "should_not_run"}

    daemon._merge_branch_to_main = fake_merge  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert result == []
    assert merge_attempts == []


def test_implementation_daemon_reconciles_merge_lock_deferrals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-003",
        "attempt": 2,
        "branch": "implementation/accel-003",
        "implementation_commit": "def456",
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": False,
            "merged": False,
            "reason": "lock_exists",
            "branch": "implementation/accel-003",
            "lock_owner_pid": 12345,
        },
    }

    daemon._iter_events = lambda: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]

    assert daemon._failed_merge_candidates() == [event]
    assert daemon._transient_merge_deferrals_by_task()["ACCEL-003"] == event
    assert "ACCEL-003" not in daemon._unresolved_merge_failures_by_task()


def test_implementation_daemon_blocks_unresolved_merge_failures_instead_of_retry_loop(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Stale merge task

- Status: todo
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Do not retry unresolved failed merges forever.
""",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-001",
        "attempt": 1,
        "branch": "implementation/accel-001",
        "implementation_commit": "abc123",
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "reason": "conflict",
            "branch": "implementation/accel-001",
        },
    }

    daemon._reconcile_failed_merges = lambda **_kwargs: []  # type: ignore[method-assign]
    daemon._cleanup_already_merged_worktrees = lambda: []  # type: ignore[method-assign]
    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._latest_implementation_finished_by_task = lambda: {}  # type: ignore[method-assign]
    daemon._successfully_merged_task_ids = lambda: set()  # type: ignore[method-assign]
    daemon._run_implementation = lambda *args, **kwargs: pytest.fail(  # type: ignore[method-assign]
        "unresolved merge task should not run implementation"
    )

    result = daemon.run_once()
    state = TodoTaskState.load(daemon.state_path)
    events = daemon._iter_events()

    assert result["active_task_id"] == ""
    assert result["ready_count"] == 0
    assert result["blocked_count"] == 1
    assert result["implementation_result"] is None
    assert state.task_statuses["ACCEL-001"] == "blocked"
    assert state.blocked_task_ids == ["ACCEL-001"]
    assert not [event for event in events if event["type"] == "implementation_skipped"]


def test_implementation_daemon_retries_cleanup_failures_for_already_merged_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-004",
        "attempt": 1,
        "branch": "implementation/accel-004",
        "implementation_commit": "abc123",
        "worktree_path": str(repo / "worktrees" / "accel-004"),
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "reason": "cleanup_failed",
            "branch": "implementation/accel-004",
        },
    }

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: True  # type: ignore[method-assign]
    daemon._cleanup_merged_worktree = lambda worktree_path, branch: {  # type: ignore[method-assign]
        "cleaned": False,
        "reason": "worktree_remove_failed",
        "branch": branch,
        "worktree_path": str(worktree_path),
    }

    result = daemon._reconcile_failed_merges()

    assert result[0]["resolved"] is False
    assert result[0]["reason"] == "cleanup_retry_failed"
    assert result[0]["cleanup_result"]["reason"] == "worktree_remove_failed"
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconciled"
    assert events[-1]["resolved"] is False


def test_implementation_daemon_discovers_cleanup_failed_successful_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-007",
        "attempt": 1,
        "branch": "implementation/accel-007",
        "implementation_commit": "abc123",
        "worktree_path": str(repo / "worktrees" / "accel-007"),
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": True,
            "branch": "implementation/accel-007",
            "merge_commit": "merge456",
        },
        "cleanup_result": {
            "cleaned": False,
            "error": "branch still checked out in submodule worktree",
        },
    }

    daemon._iter_events = lambda: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: True  # type: ignore[method-assign]

    assert daemon._failed_merge_candidates() == [event]


def test_implementation_daemon_parent_cleanup_fails_when_submodule_cleanup_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    daemon._cleanup_worktree_submodules = lambda worktree_path, branch_name: [  # type: ignore[method-assign]
        {
            "path": "external/lib",
            "branch": "implementation/accel-008-submodule-external-lib",
            "removed_worktree": False,
            "deleted_branch": False,
            "cleaned": False,
            "errors": ["cannot delete branch checked out by worktree"],
            "nested_submodule_cleanup": [],
        }
    ]
    daemon._worktree_path_registered_in_repo = lambda cwd, worktree_path: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: False  # type: ignore[method-assign]

    result = daemon._cleanup_merged_worktree(repo / "worktrees" / "accel-008", "implementation/accel-008")

    assert result["cleaned"] is False
    assert "external/lib" in result["error"]
    assert "cannot delete branch" in result["error"]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "cleanup_finished"
    assert events[-1]["cleaned"] is False


def test_implementation_daemon_removes_registered_submodule_worktree_even_when_detection_fails(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "external" / "lib"
    source.mkdir(parents=True)
    _git(source, "init")
    _git(source, "checkout", "-b", "main")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "file.txt").write_text("base\n", encoding="utf-8")
    _git(source, "add", "file.txt")
    _git(source, "commit", "-m", "base")
    branch_name = "implementation/accel-009"
    submodule_branch = f"{branch_name}-submodule-external-lib"
    _git(source, "branch", submodule_branch, "main")
    target = repo / "worktrees" / "accel-009" / "external" / "lib"
    target.parent.mkdir(parents=True)
    _git(source, "worktree", "add", str(target), submodule_branch)
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["external/lib"],
    )
    original_is_git_worktree = daemon._is_git_worktree

    def fake_is_git_worktree(path):
        if path == target:
            return False
        return original_is_git_worktree(path)

    monkeypatch.setattr(daemon, "_is_git_worktree", fake_is_git_worktree)

    result = daemon._cleanup_worktree_submodules(repo / "worktrees" / "accel-009", branch_name)

    assert result[0]["cleaned"] is True
    assert result[0]["removed_worktree"] is True
    assert result[0]["deleted_branch"] is True
    assert not daemon._worktree_path_registered_in_repo(source, target)
    ref_check = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", submodule_branch],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ref_check.returncode != 0


def test_implementation_supervisor_refills_drained_codebase_backlog(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: inspect drained supervisor refill
    return request
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed drained backlog")
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        dependency_guardrail_enabled=False,
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 1
    assert "## AUTO-002 Resolve code annotation in src/runtime.py:2" in todo_path.read_text(encoding="utf-8")
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_mode"] == "drained_exhaustive"
    assert strategy["last_drained_codebase_scan_task_count"] == 1
    assert list((repo / "discovery").glob("*-auto-002-codebase-scan-*.md"))


def test_implementation_supervisor_refills_no_ready_completed_queue(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: inspect no-ready supervisor refill
    return request
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.

## AUTO-002 Waiting on unavailable prerequisite

- Status: todo
- Completion: manual
- Priority: P2
- Track: ops
- Depends on: AUTO-999
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Waiting task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed no-ready backlog")
    state_dir = repo / "state"
    state_dir.mkdir()
    (state_dir / "task_state.json").write_text(
        json.dumps(
            {
                "task_statuses": {"AUTO-001": "completed", "AUTO-002": "waiting"},
                "completed_count": 1,
                "ready_count": 0,
                "waiting_count": 1,
                "blocked_count": 0,
                "task_count": 2,
            }
        ),
        encoding="utf-8",
    )
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        dependency_guardrail_enabled=False,
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 1
    assert "## AUTO-003 Resolve code annotation in src/runtime.py:2" in todo_path.read_text(encoding="utf-8")
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_mode"] == "runnable_drained_exhaustive"
    assert strategy["last_drained_codebase_scan_task_count"] == 2


def test_implementation_supervisor_defers_codebase_scan_when_objective_refills(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: this would be found by the codebase scanner
    return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime objective

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove runtime integration before low-level scan work.
- Evidence: missing_runtime_integration_contract
- Outputs: src/runtime.py, tests
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "objective-heap.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed objective and scan finding")
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        objective_refill_enabled=True,
        objective_path=objective_path,
        objective_graph_path=repo / "objective-graph.json",
        objective_scan_min_open_tasks=0,
        objective_scan_max_findings=1,
        objective_persist_ast_dataset=False,
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
    )

    result = TodoImplementationSupervisor(config).run_once()

    todo_text = todo_path.read_text(encoding="utf-8")
    assert result["objective_refill_count"] == 1
    assert result["codebase_refill_count"] == 0
    assert result["codebase_deferred_reason"] == "objective_refill_generated_todos"
    assert "Close objective gap" in todo_text
    assert "Resolve code annotation" not in todo_text


def test_implementation_supervisor_runs_codebase_scan_after_goal_only_objective_refill(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed completed board")
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        objective_refill_enabled=True,
        codebase_refill_enabled=True,
    )
    supervisor = TodoImplementationSupervisor(config)
    monkeypatch.setattr(
        supervisor,
        "refill_objective_backlog",
        lambda: {"generated_count": 0, "seeded_interoperability_goal_ids": ["OBJ-G001"]},
    )
    monkeypatch.setattr(supervisor, "refill_codebase_backlog", lambda: [{"fingerprint": "scan"}])

    result = supervisor.run_once()

    assert result["objective_refill_count"] == 0
    assert result["objective_seeded_interoperability_goal_count"] == 1
    assert result["codebase_refill_count"] == 1
    assert result["codebase_deferred_reason"] == ""


def test_implementation_supervisor_runs_codebase_scan_after_objective_refill_timeout(
    tmp_path,
    monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor import objective_daemon

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: this would be found after objective timeout
    return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime objective

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove runtime integration before low-level scan work.
- Evidence: missing_runtime_integration_contract
- Outputs: src/runtime.py, tests
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "todo.md", "objective-heap.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed objective and scan finding")
    state_dir = repo / "state"

    def slow_objective_daemon(_args):
        time.sleep(1.0)
        return {"generated_count": 1, "task_ids": ["AUTO-999"]}

    monkeypatch.setattr(objective_daemon, "run_objective_daemon", slow_objective_daemon)

    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        objective_refill_enabled=True,
        objective_path=objective_path,
        objective_refill_timeout_seconds=0.05,
        objective_scan_min_open_tasks=0,
        objective_scan_max_findings=1,
        objective_persist_ast_dataset=False,
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
    )

    result = TodoImplementationSupervisor(config).run_once()

    todo_text = todo_path.read_text(encoding="utf-8")
    assert result["objective_refill_count"] == 0
    assert result["codebase_refill_count"] == 1
    assert "AUTO-999" not in todo_text
    assert "Resolve code annotation" in todo_text
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_objective_goal_scan_mode"].endswith("_timeout")
    assert strategy["last_objective_refill_timeout_seconds"] == 0.05
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "objective_refill_timeout" for event in events)


def test_implementation_supervisor_records_codebase_refill_failures(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import backlog_refinery

    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"

    def fail_refill(**_kwargs):
        raise FileNotFoundError("vanished worktree")

    monkeypatch.setattr(backlog_refinery, "record_codebase_scan_findings", fail_refill)
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 0
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    failure = [event for event in events if event["type"] == "codebase_refill_failed"][-1]
    assert failure["error_type"] == "FileNotFoundError"
    assert "vanished worktree" in failure["error"]


def test_implementation_supervisor_records_codebase_refill_timeout(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import backlog_refinery

    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"

    def slow_refill(**_kwargs):
        time.sleep(1.0)
        return [{"follow_up_task_id": "AUTO-999"}]

    monkeypatch.setattr(backlog_refinery, "record_codebase_scan_findings", slow_refill)
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
        codebase_refill_timeout_seconds=0.05,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 0
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_mode"].endswith("_timeout")
    assert strategy["last_codebase_refill_timeout_seconds"] == 0.05
    assert strategy["last_drained_codebase_scan_task_count"] == 1
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    timeout = [event for event in events if event["type"] == "codebase_refill_timeout"][-1]
    assert timeout["mode"] == "drained_exhaustive"
    assert timeout["timeout_seconds"] == 0.05


def test_implementation_supervisor_records_retry_budget_guardrail(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix implementation runtime

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Fix the repeated implementation blocker.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    events_path = state_dir / "auto_events.jsonl"
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "returncode": 124,
        "validation_result": {"attempted": False, "passed": True},
        "merge_result": {"attempted": False, "merged": False, "reason": "not_attempted"},
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "auto_task_state.json",
        strategy_path=state_dir / "auto_strategy.json",
        events_path=state_dir / "auto_supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        state_prefix="auto",
        implementation_retry_budget=2,
        validation_retry_budget=0,
        merge_retry_budget=0,
        retry_budget_discovery_dir=repo / "discovery",
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["retry_budget_count"] == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve implementation retry-budget failure for AUTO-001" in todo_text
    strategy = json.loads((state_dir / "auto_strategy.json").read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "auto_supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "retry_budget_guardrail" for event in supervisor_events)


def test_validation_retry_budget_uses_safe_validation_when_failed_command_is_malformed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix validation command

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.yml
- Validation: python3 -c 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); assert p.read_text()' src/runtime.yml
- Acceptance: Fix the repeated validation blocker.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    events_path = state_dir / "auto_events.jsonl"
    events_path.parent.mkdir(parents=True)
    malformed_command = "python3 -c 'import pathlib, sys"
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "returncode": 2,
        "validation_result": {
            "attempted": True,
            "passed": False,
            "failed_command": malformed_command,
        },
        "merge_result": {"attempted": False, "merged": False, "reason": "not_attempted"},
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")
    discovery_dir = repo / "discovery"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "auto_task_state.json",
        strategy_path=state_dir / "auto_strategy.json",
        events_path=state_dir / "auto_supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        state_prefix="auto",
        implementation_retry_budget=0,
        validation_retry_budget=2,
        merge_retry_budget=0,
        retry_budget_discovery_dir=discovery_dir,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["retry_budget_count"] == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve validation retry-budget failure for AUTO-001" in todo_text
    assert f"- Validation: test -f {discovery_dir}" in todo_text
    assert malformed_command in next(discovery_dir.glob("*retry-budget.md")).read_text(encoding="utf-8")


def test_implementation_supervisor_refines_objective_goals_before_generating_todos(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime_bridge.py"
    source.parent.mkdir()
    source.write_text(
        """class RuntimeBridge:
    def dispatch(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime bridge proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that runtime bridge dispatch supports virtual AI OS execution.
- Evidence: RuntimeBridge.dispatch, missing_runtime_contract
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime contract proof.
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f objective-heap.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective refill")
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_max_refinement_children=1,
            objective_max_refinement_depth=3,
            objective_persist_ast_dataset=False,
        )
    )

    result = supervisor.run_once()

    assert result["objective_refined_goal_count"] == 1
    assert result["objective_refill_count"] == 1
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "## VAIOS-G002 Prove missing_runtime_contract for Runtime bridge proof" in objective_text
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close objective gap" in todo_text
    assert "Graph parents: VAIOS-G001" in todo_text
    assert (repo / "objective-graph.json").exists()
    assert (repo / "bundles" / "index.json").exists()
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_objective_goal_scan_mode"] == "drained_exhaustive"
    assert strategy["last_drained_objective_goal_scan_task_count"] == 1
    assert strategy["last_objective_refined_goal_ids"] == ["VAIOS-G002"]
    assert strategy["last_objective_generated_task_ids"] == ["ACCEL-002"]


def test_objective_daemon_generates_todos_bundles_and_dataset(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    dataset_dir = repo / "data" / "agent_supervisor" / "objective_datasets"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(discovery_dir),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["schema"] == "ipfs_accelerate_py.agent_supervisor.objective_daemon"
    assert payload["generated_count"] == 1
    assert payload["task_ids"] == ["ACCEL-001"]
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    assert (bundle_dir / "objective-mobile-meta-display.todo.md").exists()
    assert (bundle_dir / "index.json").exists()
    manifest = dataset_dir / "accel-objective-ast.manifest.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["row_count"] >= 2
    todo_index = bundle_dir / "todo_vector_index.json"
    assert todo_index.exists()
    index_payload = json.loads(todo_index.read_text(encoding="utf-8"))
    assert index_payload["schema"] == "ipfs_accelerate_py.agent_supervisor.todo_vector_index"
    assert index_payload["task_count"] == 1
    assert index_payload["records"][0]["goal_id"] == "VAIOS-G010"
    assert index_payload["records"][0]["merge_key"]
    assert discovery_fingerprints(discovery_dir)


def test_objective_daemon_generates_surplus_vector_indexed_todos(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text(
        """class Bridge:
    def route(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime bridge surplus proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove the runtime bridge has scheduler, timeout, retry, and fallback semantics.
- Evidence: missing_scheduler_policy, missing_timeout_policy, missing_retry_policy, missing_fallback_policy
- Outputs: src/bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime bridge proof.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/bridge.py")
    _git(repo, "commit", "-m", "seed surplus objective heap")

    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(repo / "discovery"),
            "--bundle-dir",
            str(repo / "bundles"),
            "--dataset-dir",
            str(repo / "datasets"),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "3",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["generated_count"] == 3
    assert payload["surplus_findings_per_goal"] == 3
    assert payload["surplus_min_terms_per_todo"] == 3
    todo_text = todo_path.read_text(encoding="utf-8")
    assert todo_text.count("Surplus group: objective/VAIOS-G001") == 3
    assert todo_text.count("Merge family: objective/VAIOS-G001") == 3
    assert "Merge role: aggregate" in todo_text
    assert "Merge role: evidence_cluster" in todo_text
    assert "Work item count: 4" in todo_text
    assert todo_text.count("Work item count: 3") == 2
    assert todo_text.count("Goal packet: goal_packet/runtime/src/") == 3
    assert "Goal packet task count: 3" in todo_text
    assert "Goal packet work item count: 10" in todo_text
    assert "Candidate kind: aggregate" in todo_text
    assert "Candidate kind: evidence_cluster" in todo_text
    index_payload = json.loads((repo / "bundles" / "todo_vector_index.json").read_text(encoding="utf-8"))
    assert index_payload["task_count"] == 3
    assert min(record["work_item_count"] for record in index_payload["records"]) >= 3
    assert {record["goal_packet_task_count"] for record in index_payload["records"]} == {3}
    assert {record["goal_packet_work_item_count"] for record in index_payload["records"]} == {10}
    surplus_groups = {record["surplus_group"] for record in index_payload["records"]}
    assert surplus_groups == {"objective/VAIOS-G001"}
    merge_families = {record["merge_family"] for record in index_payload["records"]}
    assert merge_families == {"objective/VAIOS-G001"}
    assert any(record["related_task_ids"] for record in index_payload["records"])
    assert index_payload["bundle_contexts"]
    assert index_payload["bundle_contexts"][0]["merge_ready"] is True
    assert index_payload["bundle_contexts"][0]["merge_families"] == ["objective/VAIOS-G001"]
    assert index_payload["estimated_compact_context_tokens"] < index_payload["estimated_raw_prompt_tokens"]
    assert index_payload["execution_packets"]
    assert index_payload["execution_packets"][0]["work_item_count_total"] >= 9
    assert index_payload["execution_packets"][0]["compact_packet_tokens"] < index_payload["execution_packets"][0]["raw_prompt_tokens"]
    bundle_index = json.loads((repo / "bundles" / "index.json").read_text(encoding="utf-8"))
    bundle_tasks = next(iter(bundle_index["bundles"].values()))["tasks"]
    assert all(task["merge_key"] for task in bundle_tasks)
    assert all(task["merge_family"] for task in bundle_tasks)
    assert min(task["work_item_count"] for task in bundle_tasks) >= 3
    assert all(task["todo_bundle_context_keys"] for task in bundle_tasks)
    assert all(task["todo_execution_packet_keys"] for task in bundle_tasks)
    assert all(task["todo_vector_key"] for task in bundle_tasks)
    bundle_summary = next(iter(bundle_index["bundles"].values()))["todo_vector_summary"]
    assert bundle_summary["merge_candidate_keys"]
    assert bundle_summary["bundle_context_keys"]
    assert bundle_summary["execution_packet_keys"]
    assert bundle_summary["execution_packet_tokens"] > 0
    assert bundle_summary["goal_packet_keys"]
    assert bundle_summary["goal_packet_work_item_count_max"] == 10
    assert bundle_summary["merge_ready_task_ids"]


def test_objective_daemon_packs_sibling_subgoals_for_vector_bundling(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    pass\n", encoding="utf-8")
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G100 Runtime bridge parent

- Status: completed
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Parent goal for runtime bridge readiness.
- Evidence: parent_runtime_bridge_packet
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md

## VAIOS-G101 Scheduler subgoal

- Status: active
- Parent: VAIOS-G100
- Fib priority: 2
- Track: runtime
- Priority: P1
- Goal: Prove scheduler behavior.
- Evidence: scheduler_policy, scheduler_backpressure, scheduler_metrics
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md

## VAIOS-G102 Fallback subgoal

- Status: active
- Parent: VAIOS-G100
- Fib priority: 3
- Track: runtime
- Priority: P1
- Goal: Prove fallback behavior.
- Evidence: fallback_route, fallback_retry, fallback_metrics
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/bridge.py")
    _git(repo, "commit", "-m", "seed sibling objective heap")

    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(repo / "discovery"),
            "--bundle-dir",
            str(repo / "bundles"),
            "--dataset-dir",
            str(repo / "datasets"),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "2",
            "--surplus-findings-per-goal",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["generated_count"] == 2
    todo_text = todo_path.read_text(encoding="utf-8")
    assert todo_text.count("Goal packet: goal_packet/runtime/src/") == 2
    assert "Candidate kind: goal_packet_aggregate" in todo_text
    assert "Goal packet role: packet_aggregate" in todo_text
    assert todo_text.count("Goal packet task count: 2") == 1
    assert todo_text.count("Goal packet task count: 3") == 1
    assert todo_text.count("Goal packet work item count: 6") == 2
    assert "Goal packet goals: VAIOS-G101, VAIOS-G102" in todo_text
    assert "Merge family: goal_packet/runtime/src/" in todo_text
    index_payload = json.loads((repo / "bundles" / "todo_vector_index.json").read_text(encoding="utf-8"))
    packet_keys = {record["goal_packet_key"] for record in index_payload["records"]}
    assert len(packet_keys) == 1
    assert {tuple(record["goal_packet_goal_ids"]) for record in index_payload["records"]} == {
        ("VAIOS-G101", "VAIOS-G102")
    }
    assert "goal_packet_aggregate" in index_payload["execution_packets"][0]["candidate_kinds"]
    assert index_payload["execution_packets"][0]["goal_packet_work_item_count_max"] == 6
    bundle_index = json.loads((repo / "bundles" / "index.json").read_text(encoding="utf-8"))
    bundle_summary = next(iter(bundle_index["bundles"].values()))["todo_vector_summary"]
    assert bundle_summary["goal_packet_work_item_count_max"] == 6
    assert bundle_summary["goal_packet_goal_ids"] == ["VAIOS-G101", "VAIOS-G102"]


def test_objective_daemon_adds_goal_packet_aggregate_when_capacity_allows(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    pass\n", encoding="utf-8")
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G100 Runtime bridge parent

- Status: completed
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Parent goal for runtime bridge readiness.
- Evidence: parent_runtime_bridge_packet
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md

## VAIOS-G101 Scheduler subgoal

- Status: active
- Parent: VAIOS-G100
- Fib priority: 2
- Track: runtime
- Priority: P1
- Goal: Prove scheduler behavior.
- Evidence: scheduler_policy, scheduler_backpressure, scheduler_metrics
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md

## VAIOS-G102 Fallback subgoal

- Status: active
- Parent: VAIOS-G100
- Fib priority: 3
- Track: runtime
- Priority: P1
- Goal: Prove fallback behavior.
- Evidence: fallback_route, fallback_retry, fallback_metrics
- Outputs: src/bridge.py
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/bridge.py")
    _git(repo, "commit", "-m", "seed packet aggregate objective heap")

    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(repo / "discovery"),
            "--bundle-dir",
            str(repo / "bundles"),
            "--dataset-dir",
            str(repo / "datasets"),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "3",
            "--surplus-findings-per-goal",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["generated_count"] == 3
    todo_text = todo_path.read_text(encoding="utf-8")
    assert todo_text.count("Goal packet: goal_packet/runtime/src/") == 3
    assert "Candidate kind: goal_packet_aggregate" in todo_text
    assert "Merge role: packet_aggregate" in todo_text
    assert "Work item count: 6" in todo_text
    assert "Goal packet role: packet_aggregate" in todo_text
    assert "Goal packet task count: 3" in todo_text
    assert "Goal packet goals: VAIOS-G101, VAIOS-G102" in todo_text

    index_payload = json.loads((repo / "bundles" / "todo_vector_index.json").read_text(encoding="utf-8"))
    assert index_payload["task_count"] == 3
    aggregate_records = [
        record for record in index_payload["records"] if record["candidate_kind"] == "goal_packet_aggregate"
    ]
    assert len(aggregate_records) == 1
    aggregate = aggregate_records[0]
    assert aggregate["goal_packet_role"] == "packet_aggregate"
    assert aggregate["work_item_count"] == 6
    assert aggregate["merge_family"] == aggregate["goal_packet_key"]
    assert aggregate["related_task_ids"]
    assert index_payload["execution_packets"][0]["goal_packet_work_item_count_max"] == 6
    assert "goal_packet_aggregate" in index_payload["execution_packets"][0]["candidate_kinds"]


def test_write_todo_vector_index_clusters_related_goal_tasks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    def route(self):\n        return None\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Close scheduler gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G021
- Graph parents: VAIOS-G020
- Graph depth: 2
- Missing evidence: scheduler policy
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add scheduler policy proof.

## ACCEL-002 Close fallback gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G022
- Graph parents: VAIOS-G020
- Graph depth: 2
- Missing evidence: fallback route
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add fallback route proof.
""",
        encoding="utf-8",
    )

    payload = write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=repo / "todo_vector_index.json",
        task_header_prefix="## ACCEL-",
    )
    records = parse_todo_vector_records(repo_root=repo, todo_path=todo_path, task_header_prefix="## ACCEL-")

    assert payload["task_count"] == 2
    assert len(payload["clusters"]) == 1
    assert payload["clusters"][0]["task_ids"] == ["ACCEL-001", "ACCEL-002"]
    assert payload["merge_candidates"][0]["confidence"] == "high"
    assert payload["merge_candidates"][0]["active_task_ids"] == ["ACCEL-001", "ACCEL-002"]
    assert payload["merge_candidates"][0]["shared_outputs"] == ["src/bridge.py"]
    assert payload["merge_candidates"][0]["merge_families"] == ["objective/VAIOS-G020"]
    assert payload["bundle_contexts"][0]["merge_ready"] is True
    assert payload["bundle_contexts"][0]["merge_ready_task_ids"] == ["ACCEL-001", "ACCEL-002"]
    assert payload["bundle_contexts"][0]["graph_parent_ids"] == ["VAIOS-G020"]
    assert payload["bundle_contexts"][0]["graph_depth_min"] == 2
    assert payload["bundle_contexts"][0]["graph_depth_max"] == 2
    assert payload["bundle_contexts"][0]["compact_context_tokens"] < payload["bundle_contexts"][0]["raw_prompt_tokens"]
    assert payload["execution_packets"][0]["active_task_ids"] == ["ACCEL-001", "ACCEL-002"]
    assert payload["execution_packets"][0]["work_item_count_total"] == 2
    assert payload["execution_packets"][0]["compact_packet_tokens"] < payload["execution_packets"][0]["raw_prompt_tokens"]
    assert records[0].related_task_ids == ["ACCEL-002"]


def test_todo_vector_index_preserves_quoted_validation_semicolons(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "config.yml"
    source.parent.mkdir()
    source.write_text("enabled: true\n", encoding="utf-8")
    inline_python = (
        "python3 -c 'import pathlib, sys; "
        'p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        "' src/config.yml"
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        f"""# Todos

## ACCEL-001 Validate YAML config

- Status: todo
- Priority: P1
- Track: ops
- Outputs: src/config.yml
- Validation: {inline_python}; test -f src/config.yml
- Acceptance: Keep shell validation intact.
""",
        encoding="utf-8",
    )

    records = parse_todo_vector_records(repo_root=repo, todo_path=todo_path, task_header_prefix="## ACCEL-")

    assert records[0].validation == [inline_python, "test -f src/config.yml"]


def test_implementation_prompt_uses_compact_todo_vector_context(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    def route(self):\n        return None\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Close scheduler gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G021
- Missing evidence: scheduler policy
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add scheduler policy proof.

## ACCEL-002 Close fallback gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G022
- Missing evidence: fallback route
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add fallback route proof.
""",
        encoding="utf-8",
    )
    index_path = repo / "objective_bundles" / "todo_vector_index.json"
    write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=index_path,
        task_header_prefix="## ACCEL-",
    )
    state_dir = repo / "state"
    state_dir.mkdir()
    strategy_path = state_dir / "strategy.json"
    strategy_path.write_text(
        json.dumps({"last_objective_todo_vector_index_path": "objective_bundles/todo_vector_index.json"}),
        encoding="utf-8",
    )
    task = parse_task_file(todo_path, task_header_prefix="## ACCEL-")[0]
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    prompt = daemon._build_implementation_prompt(task, attempt=1)

    assert task.metadata["merge key"] == "bridge-runtime"
    assert "Compact todo vector context:" in prompt
    assert "Execution packets: execution_packet/" in prompt
    assert "w=2" in prompt
    assert "- Merge key: bridge-runtime" in prompt
    assert "- Merge family: objective/VAIOS-G020" in prompt
    assert "- Goal id: VAIOS-G021" in prompt
    assert "Bundle contexts: bundle_context/" in prompt
    assert "merge_ready=true" in prompt
    assert "Merge candidates: merge_key/" in prompt
    assert "active=ACCEL-001, ACCEL-002" in prompt
    assert "Related tasks: ACCEL-002" in prompt
    assert '"embedding"' not in prompt


def test_implementation_daemon_budgets_todo_vector_context_packet_first(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    state_dir.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    rendered = daemon._budgeted_todo_vector_context(
        [
            "- Index: objective_bundles/todo_vector_index.json",
            "- Execution packets: execution_packet/runtime/src/abc ids=ACCEL-001,ACCEL-002 w=12 pw=12",
            "- Goal packet: goal_packet/runtime/src/abc",
            "- Goal packet work item count: 12",
        ],
        [
            "- AST symbols: " + ", ".join(f"symbol_{index}" for index in range(80)),
            "- Related tasks: " + " | ".join(f"ACCEL-{index:03d} noisy related task" for index in range(2, 25)),
            "- Merge candidates: " + " | ".join(f"candidate_{index} active=ACCEL-{index:03d}" for index in range(25)),
        ],
        token_budget=36,
    )

    assert "Execution packets: execution_packet/runtime/src/abc" in rendered
    assert "Goal packet: goal_packet/runtime/src/abc" in rendered
    assert "Goal packet work item count: 12" in rendered
    assert "AST symbols:" not in rendered
    assert "Related tasks:" not in rendered
    assert "Context budget:" in rendered


def test_implementation_daemon_prefers_ready_task_from_last_vector_cluster(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    def route(self):\n        return None\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Unrelated runtime cleanup

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/other
- Goal id: VAIOS-G099
- Missing evidence: unrelated cleanup
- Surplus group: objective/VAIOS-G099
- Merge key: unrelated-runtime
- Acceptance: Add unrelated proof.

## ACCEL-010 Completed scheduler proof

- Status: completed
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G021
- Missing evidence: scheduler policy
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add scheduler policy proof.

## ACCEL-011 Related fallback proof

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G022
- Missing evidence: fallback route
- Surplus group: objective/VAIOS-G020
- Merge key: bridge-runtime
- Acceptance: Add fallback route proof.
""",
        encoding="utf-8",
    )
    index_path = repo / "objective_bundles" / "todo_vector_index.json"
    write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=index_path,
        task_header_prefix="## ACCEL-",
    )
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    TodoTaskState(last_implementation_task_id="ACCEL-010").save(state_path)
    strategy_path = state_dir / "strategy.json"
    strategy_path.write_text(
        json.dumps({"last_objective_todo_vector_index_path": "objective_bundles/todo_vector_index.json"}),
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )
    tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")
    statuses = {
        "ACCEL-001": "ready",
        "ACCEL-010": "completed",
        "ACCEL-011": "ready",
    }

    selected = daemon._select_next_task(tasks, statuses, {}, {}, {})

    assert selected is not None
    assert selected.task_id == "ACCEL-011"


def test_implementation_daemon_prefers_larger_goal_work_without_vector_index(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Small objective follow-up

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Work item count: 1
- Candidate kind: evidence_cluster
- Acceptance: Add one proof.

## ACCEL-002 Larger objective aggregate

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Work item count: 6
- Goal packet work item count: 6
- Candidate kind: aggregate
- Acceptance: Add the larger integration proof.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )
    tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")

    selected = daemon._select_next_task(
        tasks,
        {"ACCEL-001": "ready", "ACCEL-002": "ready"},
        {},
        {},
        {},
    )

    assert selected is not None
    assert selected.task_id == "ACCEL-002"


def test_implementation_daemon_prefers_retry_repair_for_blocked_source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Original blocked task

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Original blocked task.

## ACCEL-002 Resolve implementation retry-budget failure for ACCEL-001

- Status: todo
- Priority: P1
- Track: ops
- Outputs: discovery
- Validation: test -f data/discovery/accel-002.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in ACCEL-001. Use evidence in data/discovery/accel-002.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release ACCEL-001 from strategy blocked_tasks.

## ACCEL-010 Unrelated runtime cleanup

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/cleanup.py
- Validation: test -f src/cleanup.py
- Acceptance: Unrelated ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )
    tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")

    selected = daemon._select_next_task(
        tasks,
        {
            "ACCEL-001": "blocked",
            "ACCEL-002": "ready",
            "ACCEL-010": "ready",
        },
        {"blocked_tasks": ["ACCEL-001"]},
        {},
        {},
    )

    assert selected is not None
    assert selected.task_id == "ACCEL-002"


def test_implementation_daemon_marks_bundle_work_order_children_completed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Close scheduler gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Work item count: 3
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_anchor
- Goal packet work item count: 6
- Candidate kind: aggregate
- Acceptance: Add scheduler proof.

## ACCEL-002 Close fallback gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Work item count: 3
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_member
- Goal packet work item count: 6
- Candidate kind: aggregate
- Acceptance: Add fallback proof.

## ACCEL-003 Close packet aggregate

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Work item count: 6
- Merge family: goal_packet/runtime/src/abc
- Merge role: packet_aggregate
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_aggregate
- Goal packet goals: VAIOS-G101, VAIOS-G102
- Goal packet work item count: 6
- Candidate kind: goal_packet_aggregate
- Acceptance: Add the shared packet proof.
""",
        encoding="utf-8",
    )
    index_path = repo / "objective_bundles" / "todo_vector_index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "task_id": "ACCEL-001",
                        "title": "Close scheduler gap",
                        "candidate_kind": "aggregate",
                        "goal_packet_key": "goal_packet/runtime/src/abc",
                        "goal_packet_role": "packet_anchor",
                        "goal_packet_goal_ids": ["VAIOS-G101", "VAIOS-G102"],
                        "goal_packet_work_item_count": 6,
                        "work_item_count": 3,
                    },
                    {
                        "task_id": "ACCEL-002",
                        "title": "Close fallback gap",
                        "candidate_kind": "aggregate",
                        "goal_packet_key": "goal_packet/runtime/src/abc",
                        "goal_packet_role": "packet_member",
                        "goal_packet_goal_ids": ["VAIOS-G101", "VAIOS-G102"],
                        "goal_packet_work_item_count": 6,
                        "work_item_count": 3,
                    },
                    {
                        "task_id": "ACCEL-003",
                        "title": "Close packet aggregate",
                        "candidate_kind": "goal_packet_aggregate",
                        "merge_role": "packet_aggregate",
                        "merge_family": "goal_packet/runtime/src/abc",
                        "goal_packet_key": "goal_packet/runtime/src/abc",
                        "goal_packet_role": "packet_aggregate",
                        "goal_packet_goal_ids": ["VAIOS-G101", "VAIOS-G102"],
                        "goal_packet_work_item_count": 6,
                        "work_item_count": 6,
                    },
                ],
                "execution_packets": [
                    {
                        "packet_key": "execution_packet/runtime/src/abc",
                        "primary_task_id": "ACCEL-003",
                        "active_task_ids": ["ACCEL-001", "ACCEL-002", "ACCEL-003"],
                        "work_item_count_total": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_dir = repo / "state"
    state_dir.mkdir()
    strategy_path = state_dir / "strategy.json"
    strategy_path.write_text(
        json.dumps({"last_objective_todo_vector_index_path": "objective_bundles/todo_vector_index.json"}),
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )
    tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")
    aggregate_task = next(task for task in tasks if task.task_id == "ACCEL-003")

    result = daemon._mark_task_or_bundle_completed_in_todo(aggregate_task)

    assert result["updated"] is True
    assert result["completion_reason"] == "bundle_work_order"
    assert result["updated_task_ids"] == ["ACCEL-003", "ACCEL-001", "ACCEL-002"]
    assert result["bundle_work_order"]["covered_task_ids"] == ["ACCEL-001", "ACCEL-002"]
    updated_tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")
    assert {task.task_id: task.status for task in updated_tasks} == {
        "ACCEL-001": "completed",
        "ACCEL-002": "completed",
        "ACCEL-003": "completed",
    }
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "todo_status_updated"
    assert events[-1]["completion_reason"] == "bundle_work_order"


def test_implementation_daemon_commits_dirty_already_completed_todo_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Close generated status

- Status: todo
- Priority: P1
- Track: ops
- Outputs: data/evidence.md
- Validation: test -f data/evidence.md
- Acceptance: Commit the generated completion status.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed todo")
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").replace("- Status: todo", "- Status: completed"),
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon._mark_task_completed_in_todo("ACCEL-001")

    assert result["updated"] is False
    assert result["reason"] == "already_completed"
    assert result["commit_result"]["committed"] is True
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "todo_status_reconciled"
    assert events[-1]["commit_result"]["committed"] is True


def test_implementation_daemon_prefers_goal_packet_aggregate_as_primary_work(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "bridge.py"
    source.parent.mkdir()
    source.write_text("class Bridge:\n    def route(self):\n        return None\n", encoding="utf-8")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Close scheduler gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G101
- Graph parents: VAIOS-G100
- Missing evidence: scheduler_policy, scheduler_backpressure, scheduler_metrics
- Surplus group: objective/VAIOS-G101
- Merge key: scheduler-runtime
- Merge family: goal_packet/runtime/src/abc
- Merge role: aggregate
- Work item count: 3
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_anchor
- Goal packet goals: VAIOS-G101, VAIOS-G102
- Goal packet task count: 3
- Goal packet work item count: 6
- Candidate kind: aggregate
- Acceptance: Add scheduler proof.

## ACCEL-002 Close fallback gap

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G102
- Graph parents: VAIOS-G100
- Missing evidence: fallback_route, fallback_retry, fallback_metrics
- Surplus group: objective/VAIOS-G102
- Merge key: fallback-runtime
- Merge family: goal_packet/runtime/src/abc
- Merge role: aggregate
- Work item count: 3
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_member
- Goal packet goals: VAIOS-G101, VAIOS-G102
- Goal packet task count: 3
- Goal packet work item count: 6
- Candidate kind: aggregate
- Acceptance: Add fallback proof.

## ACCEL-003 Close objective gap packet: VAIOS-G101, VAIOS-G102

- Status: todo
- Priority: P1
- Track: runtime
- Outputs: src/bridge.py
- Validation: test -f src/bridge.py
- Bundle: objective/runtime/bridge
- Goal id: VAIOS-G101
- Graph parents: VAIOS-G100
- Missing evidence: scheduler_policy, scheduler_backpressure, scheduler_metrics, fallback_route, fallback_retry, fallback_metrics
- Surplus group: goal_packet/runtime/src/abc
- Merge key: packet-runtime
- Merge family: goal_packet/runtime/src/abc
- Merge role: packet_aggregate
- Work item count: 6
- Work scope: goal_subgoal_packet_aggregate; vector_ast_bundle
- Goal packet: goal_packet/runtime/src/abc
- Goal packet role: packet_aggregate
- Goal packet goals: VAIOS-G101, VAIOS-G102
- Goal packet task count: 3
- Goal packet work item count: 6
- Candidate kind: goal_packet_aggregate
- Acceptance: Close the packet-level scheduler and fallback proof in one cohesive change.
""",
        encoding="utf-8",
    )
    index_path = repo / "objective_bundles" / "todo_vector_index.json"
    payload = write_todo_vector_index(
        repo_root=repo,
        todo_path=todo_path,
        index_path=index_path,
        task_header_prefix="## ACCEL-",
    )
    assert payload["execution_packets"][0]["primary_task_id"] == "ACCEL-003"
    assert payload["execution_packets"][0]["active_task_ids"][0] == "ACCEL-003"

    state_dir = repo / "state"
    state_dir.mkdir()
    strategy_path = state_dir / "strategy.json"
    strategy_path.write_text(
        json.dumps({"last_objective_todo_vector_index_path": "objective_bundles/todo_vector_index.json"}),
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )
    tasks = parse_task_file(todo_path, task_header_prefix="## ACCEL-")
    statuses = {task.task_id: "ready" for task in tasks}

    selected = daemon._select_next_task(tasks, statuses, {}, {}, {})

    assert selected is not None
    assert selected.task_id == "ACCEL-003"
    prompt = daemon._build_implementation_prompt(selected, attempt=1)
    assert "primary=ACCEL-003" in prompt
    assert "Packet sibling tasks covered by primary: ACCEL-001, ACCEL-002" in prompt
    assert "Related tasks:" not in prompt


def test_objective_daemon_suppresses_existing_discovery_fingerprint(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    args = argparse.Namespace(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=repo / "data" / "agent_supervisor" / "objective_datasets",
        task_prefix="ACCEL-",
        depends_on=[],
        seen_fingerprint=[],
        repeat_existing=False,
        max_findings=1,
        no_persist_ast_dataset=True,
        submit_bundles=False,
        queue_path=None,
        queue_task_type="codex.todo_bundle",
        queue_model_name="codex",
        log_level="INFO",
    )

    first = run_objective_daemon(args)
    second = run_objective_daemon(args)

    assert first["generated_count"] == 1
    assert second["generated_count"] == 0
    assert todo_path.read_text(encoding="utf-8").count("## ACCEL-001 ") == 1


def test_objective_gap_force_goal_ids_bypasses_seen_fingerprint(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)

    first = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)
    suppressed = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        seen_fingerprints=[first[0].fingerprint],
    )
    forced = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        seen_fingerprints=[first[0].fingerprint],
        force_goal_ids=[first[0].goal_id],
    )

    assert first[0].goal_id == "VAIOS-G010"
    assert suppressed == []
    assert [finding.goal_id for finding in forced] == ["VAIOS-G010"]


def test_objective_daemon_creates_tracking_document_and_graph(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    readme = repo / "README.md"
    readme.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    objective_path = repo / "docs" / "objective-heap.md"
    todo_path = repo / "docs" / "todo.md"
    graph_path = repo / "data" / "agent_supervisor" / "objective_graph.json"
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--graph-path",
            str(graph_path),
            "--ensure-tracking-document",
            "--ultimate-goal",
            "Operate as a virtual AI OS for a Meta glasses remote display.",
            "--root-evidence",
            "missing_meta_display_bridge",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["tracking_document_created"] is True
    assert payload["ensured_goal_ids"] == ["OBJ-G000"]
    assert payload["objective_goal_count"] == 1
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["graph"]["roots"] == ["OBJ-G000"]
    assert graph["thought_graph"]["node_count"] >= 3
    assert {node["kind"] for node in graph["thought_graph"]["nodes"]} >= {
        "goal",
        "evidence_requirement",
        "validation_strategy",
    }
    assert "missing_meta_display_bridge" in objective_path.read_text(encoding="utf-8")
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")


def test_run_goal_validation_preserves_quoted_validation_semicolons(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src" / "config.yml"
    source.parent.mkdir()
    source.write_text("enabled: true\n", encoding="utf-8")
    inline_python = (
        "python3 -c 'import pathlib, sys; "
        'p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        "' src/config.yml"
    )
    goal = ObjectiveGoal(
        goal_id="VAIOS-G001",
        title="Validate config evidence",
        fields={"validation": f"{inline_python}; test -f src/config.yml"},
    )

    result = run_goal_validation(repo_root=repo, goal=goal)

    assert result["passed"] is True
    assert [item["command"] for item in result["results"]] == [inline_python, "test -f src/config.yml"]


def test_objective_daemon_reconciles_completed_goals_from_evidence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "proof.py"
    source.parent.mkdir()
    source.write_text("COMPLETE_RUNTIME_PROOF = True\n", encoding="utf-8")
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Completed runtime proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove runtime evidence reconciliation.
- Evidence: COMPLETE_RUNTIME_PROOF
- Outputs: src/proof.py
- Validation: test -f src/proof.py
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/proof.py")
    _git(repo, "commit", "-m", "seed completed objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["completed_goal_ids"] == ["VAIOS-G001"]
    assert payload["objective_active_goal_count"] == 0
    assert payload["objective_completed_goal_count"] == 1
    assert payload["generated_count"] == 0
    assert payload["objective_completion_validation_results"]["VAIOS-G001"]["passed"] is True
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "- Status: completed" in objective_text
    assert "- Completion evidence: COMPLETE_RUNTIME_PROOF => src/proof.py (exact)" in objective_text
    assert "- Completion validation: 0" in objective_text


def test_objective_daemon_does_not_complete_goal_when_validation_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "proof.py"
    source.parent.mkdir()
    source.write_text("COMPLETE_RUNTIME_PROOF = True\n", encoding="utf-8")
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime proof with failing validation

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove runtime evidence reconciliation.
- Evidence: COMPLETE_RUNTIME_PROOF
- Outputs: src/proof.py
- Validation: test -f missing-validation-proof.txt
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/proof.py")
    _git(repo, "commit", "-m", "seed incomplete validated objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["completed_goal_ids"] == []
    assert payload["objective_active_goal_count"] == 1
    assert payload["objective_completed_goal_count"] == 0
    assert payload["objective_completion_validation_results"]["VAIOS-G001"]["passed"] is False
    assert payload["generated_count"] == 0
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "- Status: active" in objective_text
    assert "Completion evidence:" not in objective_text


def test_objective_daemon_seeds_interoperability_goals_from_submodules(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / ".gitmodules").write_text(
        """[submodule "hallucinate_app"]
    path = hallucinate_app
    url = https://github.com/endomorphosis/hallucinate_app
[submodule "swissknife"]
    path = swissknife
    url = https://github.com/endomorphosis/swissknife
[submodule "mcp_plus_plus"]
    path = mcp_plus_plus
    url = https://github.com/endomorphosis/mcp_plus_plus
""",
        encoding="utf-8",
    )
    (repo / "hallucinate_app" / "interfaces").mkdir(parents=True)
    (repo / "hallucinate_app" / "interfaces" / "control_surface.idl").write_text(
        "interface ControlSurface { void dispatch(); }\n",
        encoding="utf-8",
    )
    (repo / "hallucinate_app" / "pyproject.toml").write_text(
        "[project]\nname = \"hallucinate-app\"\n",
        encoding="utf-8",
    )
    (repo / "swissknife" / "mcp").mkdir(parents=True)
    (repo / "swissknife" / "package.json").write_text(
        "{\"name\":\"swissknife\"}\n",
        encoding="utf-8",
    )
    (repo / "swissknife" / "mcp" / "orb_descriptor.json").write_text(
        "{\"name\":\"orb\"}\n",
        encoding="utf-8",
    )
    (repo / "mcp_plus_plus").mkdir(parents=True)
    (repo / "mcp_plus_plus" / "mcp_descriptor.json").write_text(
        "{\"name\":\"mcp-plus-plus\"}\n",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G000 Virtual AI OS root

- Status: active
- Parent:
- Fib priority: 1
- Track: ops
- Priority: P0
- Goal: Make the virtual AI OS interoperable.
- Evidence: root_virtual_ai_os_proof
- Outputs: docs
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed interop objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--seed-interoperability-goals",
            "--interoperability-focus",
            "hallucinate_app",
            "--max-interoperability-goals",
            "2",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "2",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["seeded_interoperability_goal_ids"] == ["VAIOS-G001", "VAIOS-G002"]
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "Interoperate hallucinate_app with swissknife" in objective_text
    assert "Interoperate hallucinate_app with mcp_plus_plus" in objective_text
    assert "- Goal kind: interoperability" in objective_text
    assert "- Package manifests:" in objective_text
    assert "- Interface descriptors:" in objective_text
    assert "- MCP descriptors:" in objective_text
    assert "tests/integration/test_hallucinate_app_swissknife_interop.py" in objective_text
    assert "hallucinate_app/interfaces/control_surface.idl" in objective_text
    assert "swissknife/mcp/orb_descriptor.json" in objective_text
    assert payload["objective_heap_schedule_count"] >= 1
    assert payload["generated_count"] == 2
    graph = json.loads((repo / "data" / "agent_supervisor" / "objective_graph.json").read_text(encoding="utf-8"))
    thought_kinds = {node["kind"] for node in graph["thought_graph"]["nodes"]}
    assert "interoperability_pair" in thought_kinds
    assert "test_strategy" in thought_kinds
    assert "interface_descriptor" in thought_kinds
    assert "mcp_descriptor" in thought_kinds
    assert "package_manifest" in thought_kinds

    second = run_objective_daemon(args)
    objective_text = objective_path.read_text(encoding="utf-8")

    assert second["seeded_interoperability_goal_ids"] == []
    assert objective_text.count("Interoperate hallucinate_app with swissknife") == 1
    assert objective_text.count("Interoperate hallucinate_app with mcp_plus_plus") == 1


def test_objective_daemon_compacts_duplicate_interoperability_goals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G000 Virtual AI OS root

- Status: completed
- Parent:
- Fib priority: 1
- Track: ops
- Priority: P0
- Goal: Make the virtual AI OS interoperable.
- Evidence: root_virtual_ai_os_proof
- Outputs: docs
- Validation: test -f objective-heap.md

## VAIOS-G001 Interoperate hallucinate_app with external/ipfs_datasets

- Status: active
- Parent: VAIOS-G000
- Fib priority: 3000
- Track: interoperability
- Priority: P1
- Goal kind: interoperability
- Interoperability pair: hallucinate_app, external/ipfs_datasets
- Evidence: datasets_interop
- Outputs: tests
- Validation: test -f objective-heap.md

## VAIOS-G002 Interoperate hallucinate_app with ipfs_datasets_py

- Status: active
- Parent: VAIOS-G000
- Fib priority: 3001
- Track: interoperability
- Priority: P1
- Goal kind: interoperability
- Interoperability pair: hallucinate_app, ipfs_datasets_py
- Evidence: datasets_py_interop
- Outputs: tests
- Validation: test -f objective-heap.md

## VAIOS-G003 Interoperate hallucinate_app with external/ipfs_datasets

- Status: active
- Parent: VAIOS-G000
- Fib priority: 3002
- Track: interoperability
- Priority: P1
- Goal kind: interoperability
- Interoperability pair: hallucinate_app, external/ipfs_datasets
- Evidence: duplicate_datasets_interop
- Outputs: tests
- Validation: test -f objective-heap.md

## VAIOS-G004 Interoperate hallucinate_app with swissknife

- Status: active
- Parent: VAIOS-G000
- Fib priority: 3003
- Track: interoperability
- Priority: P1
- Goal kind: interoperability
- Interoperability pair: hallucinate_app, swissknife
- Evidence: swissknife_interop
- Outputs: tests
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed duplicate interop objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "0",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["deduplicated_interoperability_goal_ids"] == ["VAIOS-G002", "VAIOS-G003"]
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "## VAIOS-G001 Interoperate hallucinate_app with external/ipfs_datasets" in objective_text
    assert "## VAIOS-G002 Interoperate hallucinate_app with ipfs_datasets_py" not in objective_text
    assert "## VAIOS-G003 Interoperate hallucinate_app with external/ipfs_datasets" not in objective_text
    assert "## VAIOS-G004 Interoperate hallucinate_app with swissknife" in objective_text
    schedule = objective_heap_schedule(parse_goal_heap(objective_text))
    assert [record.goal_id for record in schedule] == ["VAIOS-G001", "VAIOS-G004"]


def test_objective_daemon_seeds_all_interoperability_pairs_without_focus(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / ".gitmodules").write_text(
        """[submodule "component_a"]
    path = component_a
    url = https://example.invalid/component_a
[submodule "component_b"]
    path = component_b
    url = https://example.invalid/component_b
[submodule "component_c"]
    path = component_c
    url = https://example.invalid/component_c
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## OBJ-G000 Reusable integration root

- Status: active
- Parent:
- Fib priority: 1
- Track: ops
- Priority: P0
- Goal: Make the package interoperable with configured components.
- Evidence: root_integration_proof
- Outputs: docs
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", ".gitmodules", "objective-heap.md", "todo.md")
    _git(repo, "commit", "-m", "seed generic interop objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--seed-interoperability-goals",
            "--max-interoperability-goals",
            "3",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "3",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["seeded_interoperability_goal_ids"] == ["OBJ-G001", "OBJ-G002", "OBJ-G003"]
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "Interoperate component_a with component_b" in objective_text
    assert "Interoperate component_a with component_c" in objective_text
    assert "Interoperate component_b with component_c" in objective_text


def test_objective_daemon_seeds_interoperability_goals_from_gitlinks_without_gitmodules_mapping(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / ".gitmodules").write_text(
        """[submodule "external/ipfs_accelerate"]
    path = external/ipfs_accelerate
    url = https://github.com/endomorphosis/ipfs_accelerate
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G000 Virtual AI OS root

- Status: active
- Parent:
- Fib priority: 1
- Track: ops
- Priority: P0
- Goal: Make the virtual AI OS interoperable.
- Evidence: root_virtual_ai_os_proof
- Outputs: docs
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", ".gitmodules", "objective-heap.md", "todo.md")
    _git(repo, "update-index", "--add", "--cacheinfo", "160000", "a" * 40, "hallucinate_app")
    _git(repo, "update-index", "--add", "--cacheinfo", "160000", "b" * 40, "swissknife")
    _git(repo, "update-index", "--add", "--cacheinfo", "160000", "c" * 40, "Mcp-Plus-Plus")
    _git(repo, "commit", "-m", "seed gitlink interop objective")
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--seed-interoperability-goals",
            "--interoperability-focus",
            "hallucinate_app",
            "--max-interoperability-goals",
            "3",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "3",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["seeded_interoperability_goal_ids"] == ["VAIOS-G001", "VAIOS-G002", "VAIOS-G003"]
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "Interoperate hallucinate_app with" in objective_text
    assert "hallucinate_app, swissknife" in objective_text
    assert "hallucinate_app, Mcp-Plus-Plus" in objective_text


def test_objective_daemon_refines_missing_evidence_into_child_goals(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--refine-objective-heap",
            "--max-refinement-children",
            "1",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "2",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)
    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    refined = [goal for goal in goals if goal.goal_id in payload["refined_goal_ids"]]

    assert payload["refined_goal_ids"] == ["VAIOS-G011"]
    assert payload["generated_count"] == 1
    assert len(refined) == 1
    assert refined[0].parent_goal_ids == ["VAIOS-G010"]
    assert refined[0].required_evidence == ["missing_gesture_policy"]
    assert refined[0].fields["fib_priority"] == str(fibonacci_priority(1, 0))
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "Prove missing_gesture_policy for Meta display control bridge" in todo_text


def test_bundle_supervisor_plans_isolated_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "data" / "agent_supervisor" / "objective_bundles" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/runtime/kernel": {
                        "shard_path": "data/agent_supervisor/objective_bundles/runtime.todo.md",
                        "parallel_lane": "objective/runtime/kernel",
                        "conflict_policy": "bundle-local edits",
                        "tasks": [{"task_id": "ACCEL-001"}],
                    },
                    "objective/mobile/meta-display": {
                        "shard_path": "data/agent_supervisor/objective_bundles/mobile.todo.md",
                        "parallel_lane": "objective/mobile/meta-display",
                        "conflict_policy": "invoke resolver",
                        "tasks": [{"task_id": "ACCEL-002"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="ACCEL-",
        implement=True,
        implementation_command="codex exec --full-auto",
        llm_merge_resolver_command="python resolver.py",
        generated_dirty_repair_enabled=True,
        worktree_submodule_paths=("ipfs_datasets_py/ipfs_accelerate_py",),
        log_level="DEBUG",
        max_lanes=None,
    )

    assert [lane.bundle_key for lane in lanes] == [
        "objective/mobile/meta-display",
        "objective/runtime/kernel",
    ]
    assert lanes[0].todo_path == repo / "data/agent_supervisor/objective_bundles/mobile.todo.md"
    assert lanes[0].state_dir != lanes[1].state_dir
    assert lanes[0].worktree_root != lanes[1].worktree_root
    assert lanes[0].task_ids == ["ACCEL-002"]
    assert "--implement" in lanes[0].command
    assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor" in lanes[0].command
    assert "--implementation-command" in lanes[0].command
    assert lanes[0].command[lanes[0].command.index("--log-level") + 1] == "DEBUG"
    assert lanes[0].command[lanes[0].command.index("--llm-merge-resolver-command") + 1] == "python resolver.py"
    assert "--auto-commit-generated-dirty" in lanes[0].command
    assert lanes[0].command.count("--worktree-submodule-path") == 1
    assert "ipfs_datasets_py/ipfs_accelerate_py" in lanes[0].command


def test_bundle_supervisor_writes_manifest_without_starting_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "objective_bundles" / "index.json"
    index_path.parent.mkdir()
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/ops/root": {
                        "shard_path": "objective_bundles/root.todo.md",
                        "parallel_lane": "objective/ops/root",
                        "conflict_policy": "use merge resolver",
                        "tasks": [{"task_id": "ACCEL-009"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = repo / "manifest.json"
    args = build_bundle_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--bundle-index-path",
            str(index_path),
            "--manifest-path",
            str(manifest_path),
            "--task-prefix",
            "ACCEL-",
            "--no-implement",
        ]
    )

    payload = run_bundle_supervisor(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["planned_count"] == 1
    assert payload["started_count"] == 0
    assert manifest["lanes"][0]["bundle_key"] == "objective/ops/root"
    assert manifest["lanes"][0]["todo_path"] == "objective_bundles/root.todo.md"
    assert "--no-implement" in manifest["lanes"][0]["command"]


def test_implementation_daemon_invokes_configured_llm_merge_resolver(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    capture_path = tmp_path / "resolver-prompt.txt"
    resolver_script = tmp_path / "resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )
    result = daemon._invoke_llm_merge_resolver_for_failed_merge(
        workspace=repo,
        task=PortalTask(
            task_id="ACCEL-999",
            title="Resolve semantic merge",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=2,
        branch_name="implementation/accel-999",
        target_branch="main",
        merge_command=["git", "merge", "implementation/accel-999"],
        merge_stdout="",
        merge_stderr="CONFLICT (content): Merge conflict",
    )

    assert result["applied"] is True
    assert result["llm_returncode"] == 0
    prompt = capture_path.read_text(encoding="utf-8")
    assert "ACCEL-999" in prompt
    assert "implementation/accel-999" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "llm_merge_resolver_invoked"
    assert events[-1]["prompt_chars"] == len(prompt)


def test_llm_merge_resolver_times_out_hung_command(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge_resolver import invoke_llm_resolver

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    result = invoke_llm_resolver(
        {"found": True, "repo_root": str(tmp_path), "prompt": "resolve this merge"},
        command_template=f"{shlex.quote(sys.executable)} {shlex.quote(str(sleeper))}",
        timeout_seconds=0.01,
    )

    assert result["applied"] is False
    assert result["llm_timeout"] is True
    assert result["llm_returncode"] is None
    assert "timed out" in result["apply_error"]


def test_implementation_supervisor_aborts_interrupted_main_checkout_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "merge_aborted_without_resolver"
    assert result["abort_result"]["aborted"] is True
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    assert target.read_text(encoding="utf-8") == "main\n"


def test_implementation_supervisor_aborts_interrupted_main_checkout_merge_with_reset_fallback(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as implementation_supervisor_module,
    )

    real_run = implementation_supervisor_module.subprocess.run

    def fail_merge_abort(command, *args, **kwargs):
        if list(command) == ["git", "merge", "--abort"]:
            return subprocess.CompletedProcess(command, 1, "", "simulated abort failure")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(implementation_supervisor_module.subprocess, "run", fail_merge_abort)
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["abort_result"]["aborted"] is True
    assert result["abort_result"]["returncode"] == 1
    assert result["abort_result"]["reset_merge_fallback"]["reset"] is True
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    assert target.read_text(encoding="utf-8") == "main\n"


def test_implementation_supervisor_invokes_llm_for_interrupted_main_checkout_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    capture_path = tmp_path / "supervisor-resolver-prompt.txt"
    resolver_script = tmp_path / "supervisor_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "pathlib.Path('conflict.txt').write_text('resolved by supervisor\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'conflict.txt'])\n",
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
            llm_merge_resolver_command=(
                f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} "
                f"{shlex.quote(str(capture_path))}"
            ),
            llm_merge_resolver_timeout_seconds=5,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "llm_resolved_merge"
    assert result["llm_merge_resolver"]["applied"] is True
    assert result["commit_result"]["completed"] is True
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    assert target.read_text(encoding="utf-8") == "resolved by supervisor\n"
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/conflict", "HEAD") == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "supervisor_main_checkout_merge_in_progress" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "main_checkout_merge_state_repair"


def test_implementation_supervisor_defers_merge_repair_when_checkout_lock_is_live(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
        )
    )
    supervisor._repo_merge_lock_path().write_text(
        json.dumps(
            {
                "kind": "merge",
                "pid": os.getpid(),
                "owner_script": "",
                "repo_root": str(repo.resolve()),
                "task_id": "OTHER-1",
                "branch": "implementation/other",
            }
        ),
        encoding="utf-8",
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is False
    assert result["reason"] == "checkout_mutation_lock_exists"
    assert result["lock_owner_task_id"] == "OTHER-1"
    assert supervisor._git_merge_head(repo) != ""
    assert supervisor._git_unmerged_paths(repo) == ["conflict.txt"]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "main_checkout_merge_state_repair_deferred"


def test_implementation_supervisor_clears_stale_same_state_checkout_lock(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )
    supervisor._repo_merge_lock_path().write_text(
        json.dumps(
            {
                "kind": "merge",
                "pid": os.getpid(),
                "owner_script": Path(sys.argv[0]).name,
                "repo_root": str(repo.resolve()),
                "state_dir": str(state_dir.resolve()),
                "state_path": str((state_dir / "task_state.json").resolve()),
                "task_id": "AUTO-404",
                "branch": "implementation/stale",
            }
        ),
        encoding="utf-8",
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "merge_aborted_without_resolver"
    assert supervisor._git_merge_head(repo) == ""
    assert not supervisor._repo_merge_lock_path().exists()
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "checkout_mutation_lock_cleared" for event in events)


def test_implementation_supervisor_deterministically_repairs_objective_heap_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "implementation_plan" / "docs" / "objective-goal-heap.md"
    objective_path.parent.mkdir(parents=True)
    objective_path.write_text(
        "# Objective Heap\n\n"
        "## VAIOS-G001 Root\n\n"
        "- Status: active\n"
        "- Goal: Keep the root goal.\n",
        encoding="utf-8",
    )
    _git(repo, "add", "implementation_plan/docs/objective-goal-heap.md")
    _git(repo, "commit", "-m", "seed objective heap")

    _git(repo, "checkout", "-b", "implementation/objective-feature")
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + "\n## VAIOS-G002 Feature goal\n\n- Status: active\n- Goal: Keep the feature goal.\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "feature objective")

    _git(repo, "checkout", "main")
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + "\n## VAIOS-G003 Main goal\n\n- Status: active\n- Goal: Keep the main goal.\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "main objective")
    merge = subprocess.run(
        ["git", "merge", "--no-ff", "--no-edit", "implementation/objective-feature"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0
    assert "implementation_plan/docs/objective-goal-heap.md" in _git(repo, "diff", "--name-only", "--diff-filter=U")

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            objective_path=objective_path,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "deterministic_generated_markdown_conflict_repair"
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    text = objective_path.read_text(encoding="utf-8")
    assert "## VAIOS-G001 Root" in text
    assert "## VAIOS-G002 Feature goal" in text
    assert "## VAIOS-G003 Main goal" in text
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/objective-feature", "HEAD") == ""


def test_implementation_daemon_refuses_path_commit_during_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    result = daemon._commit_specific_path(repo, "conflict.txt", subject="should not partially commit")

    assert result["committed"] is False
    assert result["reason"] == "repo_merge_in_progress"
    assert result["unmerged_paths"] == ["conflict.txt"]
    assert _git(repo, "diff", "--name-only", "--diff-filter=U") == "conflict.txt"


def test_implementation_daemon_defers_generated_commit_when_checkout_lock_is_live(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    generated = repo / "generated.md"
    generated.write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "generated.md")
    _git(repo, "commit", "-m", "seed")
    generated.write_text("changed\n", encoding="utf-8")

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    daemon._repo_merge_lock_path().write_text(
        json.dumps(
            {
                "kind": "merge",
                "pid": os.getpid(),
                "owner_script": "",
                "repo_root": str(repo.resolve()),
                "task_id": "OTHER-2",
                "branch": "implementation/other",
            }
        ),
        encoding="utf-8",
    )

    result = daemon._commit_generated_file_update(generated, task_id="ACCEL-2", subject="generated update")

    assert result["committed"] is False
    assert result["reason"] == "checkout_mutation_lock_exists"
    assert result["lock_owner_task_id"] == "OTHER-2"
    assert _git(repo, "status", "--porcelain", "--", "generated.md").startswith("M ")


def test_implementation_daemon_clears_stale_same_state_merge_lock_for_generated_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    generated = repo / "generated.md"
    generated.write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "generated.md")
    _git(repo, "commit", "-m", "seed")
    generated.write_text("changed\n", encoding="utf-8")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )
    daemon._repo_merge_lock_path().write_text(
        json.dumps(
            {
                "kind": "merge",
                "pid": os.getpid(),
                "owner_script": Path(sys.argv[0]).name,
                "repo_root": str(repo.resolve()),
                "state_dir": str(state_dir.resolve()),
                "state_path": str((state_dir / "task_state.json").resolve()),
                "task_id": "ACCEL-404",
                "branch": "implementation/stale",
            }
        ),
        encoding="utf-8",
    )

    result = daemon._commit_generated_file_update(generated, task_id="ACCEL-2", subject="generated update")

    assert result["committed"] is True
    assert result["commit"]
    assert not daemon._repo_merge_lock_path().exists()
    assert _git(repo, "status", "--porcelain", "--", "generated.md") == ""
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "merge_lock_cleared" for event in events)


def test_implementation_daemon_classifies_signal_and_timeout_returncodes(tmp_path):
    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
    )

    signal_result = daemon._implementation_returncode_detail(-15)
    timeout_result = daemon._implementation_returncode_detail(124)

    assert signal_result["termination_reason"] == "signal"
    assert signal_result["signal"] == 15
    assert signal_result["signal_name"] == "SIGTERM"
    assert timeout_result == {"termination_reason": "timeout", "timed_out": True}
    assert daemon._implementation_returncode_detail(1) == {}


def test_implementation_supervisor_cleans_merged_backlogged_worktrees(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-clean")
    marker.write_text("merged\n", encoding="utf-8")
    _git(repo, "commit", "-am", "merged branch")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/auto-clean")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "auto-clean"
    _git(repo, "worktree", "add", str(worktree_path), "implementation/auto-clean")

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 1
    assert not worktree_path.exists()
    branch_exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "implementation/auto-clean"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert branch_exists.returncode != 0


def test_implementation_supervisor_cleans_redundant_dirty_merged_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/redundant-clean")
    marker.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch change")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/redundant-clean")
    (repo / "shared.py").write_text("already on main\n", encoding="utf-8")
    _git(repo, "add", "shared.py")
    _git(repo, "commit", "-m", "add shared file")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "redundant-clean"
    _git(repo, "worktree", "add", str(worktree_path), "implementation/redundant-clean")
    (worktree_path / "shared.py").write_text("already on main\n", encoding="utf-8")

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 1
    assert result["removed"][0]["dirty_redundancy"]["redundant"] is True
    assert not worktree_path.exists()


def test_implementation_supervisor_cleans_merged_worktree_with_deleted_configured_submodule(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/submodule-clean")
    marker.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch change")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", "implementation/submodule-clean")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "submodule-clean"
    _git(repo, "worktree", "add", str(worktree_path), "implementation/submodule-clean")

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            worktree_submodule_paths=("external/ipfs_datasets",),
        )
    )
    monkeypatch.setattr(
        supervisor,
        "_git_status_short",
        lambda path: [" D external/ipfs_datasets"] if path == worktree_path else [],
    )
    monkeypatch.setattr(
        supervisor,
        "_target_ref_has_path",
        lambda relative, target_ref: relative == "external/ipfs_datasets" and target_ref == "main",
    )

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 1
    assert result["removed"][0]["dirty_redundancy"]["redundant"] is True
    assert result["removed"][0]["dirty_redundancy"]["reason"] == (
        "configured_submodule_deletions_match_target"
    )
    assert not worktree_path.exists()


def _merged_cleanup_worktree_fixture(
    tmp_path: Path,
    branch_name: str,
) -> tuple[Path, Path, TodoImplementationSupervisor]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    files = {
        "README.md": "base\n",
        "docs/tasks.todo.md": "# Agent Todos\n",
        "docs/objective.md": "# Objective Heap\n",
        "state/objective_graph.json": "{}\n",
        "src/app.py": "VALUE = 'base'\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", *files.keys())
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", branch_name)
    (repo / "README.md").write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch change")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-edit", branch_name)

    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / branch_name.rsplit("/", 1)[-1]
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    supervisor_state = repo / "supervisor_state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "docs" / "tasks.todo.md",
            state_path=supervisor_state / "task_state.json",
            strategy_path=supervisor_state / "strategy.json",
            events_path=supervisor_state / "events.jsonl",
            state_dir=supervisor_state,
            repo_root=repo,
            worktree_root=worktree_root,
            reconciliation_guardrail_discovery_dir=repo / "data" / "ns" / "discovery",
            codebase_scan_discovery_dir=repo / "data" / "ns" / "codebase-discovery",
            objective_path=repo / "docs" / "objective.md",
            objective_graph_path=repo / "state" / "objective_graph.json",
            objective_bundle_dir=repo / "data" / "ns" / "bundles",
            objective_dataset_dir=repo / "data" / "ns" / "datasets",
            objective_discovery_dir=repo / "data" / "ns" / "objective-discovery",
            objective_todo_vector_index_path=repo / "data" / "ns" / "bundles" / "todo_vector_index.json",
        )
    )
    return repo, worktree_path, supervisor


def test_implementation_supervisor_detects_stale_worktree_from_git_and_dirty_signals(tmp_path):
    repo, worktree_path, supervisor = _merged_cleanup_worktree_fixture(
        tmp_path,
        "implementation/stale-signal-dirty",
    )
    (worktree_path / "src" / "app.py").write_text("VALUE = 'dirty stale signal'\n", encoding="utf-8")

    result = supervisor.detect_stale_worktrees()

    assert result["stale_count"] == 1
    stale = result["stale"][0]
    assert stale["path"] == str(worktree_path)
    assert stale["remedy"] == "rescue_dirty_worktree_then_reconcile"
    assert "branch_already_merged" in stale["reasons"]
    assert "dirty_inactive_worktree" in stale["reasons"]
    assert stale["dirty"] is True
    events = [
        json.loads(line)
        for line in (repo / "supervisor_state" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "stale_worktree_detection"
    assert events[-1]["reason_counts"]["dirty_inactive_worktree"] == 1


def test_implementation_supervisor_does_not_mark_active_worktree_stale_from_calendar_only(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/calendar-only"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "active.txt").write_text("active branch work\n", encoding="utf-8")
    _git(repo, "add", "active.txt")
    _git(repo, "commit", "-m", "active branch work")
    _git(repo, "checkout", "main")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "calendar-only"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)
    state_dir = repo / "state"
    log_path = state_dir / "implementation.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("still moving\n", encoding="utf-8")
    old_timestamp = "2000-01-01T00:00:00+00:00"
    TodoTaskState(
        active_task_id="ACCEL-123",
        active_task_title="Calendar age is not enough",
        active_task_track="ops",
        active_task_started_at=old_timestamp,
        active_attempt=1,
        active_phase="implementing",
        active_phase_started_at=old_timestamp,
        active_phase_detail="running",
        active_log_path=str(log_path),
        active_worktree_path=str(worktree_path),
        active_branch=branch_name,
        implementation_in_progress=True,
        heartbeat_at=old_timestamp,
        last_progress_at=old_timestamp,
    ).save(state_dir / "task_state.json")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            stale_seconds=60,
            implementation_log_stall_seconds=60,
        )
    )
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [f"worker --worktree {worktree_path}"])
    monkeypatch.setattr(supervisor, "_read_managed_daemon_pid", lambda: 0)

    result = supervisor.detect_stale_worktrees(now_ts=time.time())

    assert result["stale_count"] == 0
    assert any(item["reason"] == "active_state_worktree" for item in result["skipped"])


def test_implementation_supervisor_cleans_merged_worktree_with_generated_only_dirty_outputs(tmp_path):
    _repo, worktree_path, supervisor = _merged_cleanup_worktree_fixture(
        tmp_path,
        "implementation/generated-cleanup",
    )
    (worktree_path / "docs" / "tasks.todo.md").write_text("# Agent Todos\n\n- generated\n", encoding="utf-8")
    (worktree_path / "docs" / "objective.md").write_text("# Objective Heap\n\n## Generated\n", encoding="utf-8")
    (worktree_path / "state" / "objective_graph.json").unlink()
    discovery_output = worktree_path / "data" / "ns" / "discovery" / "scan.md"
    discovery_output.parent.mkdir(parents=True, exist_ok=True)
    discovery_output.write_text("# Generated discovery\n", encoding="utf-8")
    vector_output = worktree_path / "data" / "ns" / "bundles" / "todo_vector_index.json"
    vector_output.parent.mkdir(parents=True, exist_ok=True)
    vector_output.write_text("{}\n", encoding="utf-8")

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 1
    dirty_redundancy = result["removed"][0]["dirty_redundancy"]
    assert dirty_redundancy["redundant"] is True
    assert dirty_redundancy["reason"] == "generated_only_status_paths_dropped"
    checked_paths = {item["path"] for item in dirty_redundancy["checked"]}
    assert "docs/tasks.todo.md" in checked_paths
    assert "docs/objective.md" in checked_paths
    assert "state/objective_graph.json" in checked_paths
    expanded_paths = {
        path
        for item in dirty_redundancy["checked"]
        for path in item.get("expanded_untracked_paths", [])
    }
    assert "data" in checked_paths
    assert "data/ns/discovery/scan.md" in expanded_paths
    assert "data/ns/bundles/todo_vector_index.json" in expanded_paths
    assert not worktree_path.exists()


def test_implementation_supervisor_keeps_merged_worktree_with_generated_and_source_dirty_paths(tmp_path):
    _repo, worktree_path, supervisor = _merged_cleanup_worktree_fixture(
        tmp_path,
        "implementation/generated-mixed",
    )
    (worktree_path / "docs" / "tasks.todo.md").write_text("# Agent Todos\n\n- generated\n", encoding="utf-8")
    (worktree_path / "src" / "app.py").write_text("VALUE = 'local source change'\n", encoding="utf-8")

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 0
    assert result["skipped_reason_counts"]["dirty_worktree_rescued"] == 1
    assert result["skipped"][0]["rescue_result"]["preserved"] is True
    assert worktree_path.exists()


def test_implementation_supervisor_rescues_dirty_merged_worktree(tmp_path):
    repo, worktree_path, supervisor = _merged_cleanup_worktree_fixture(
        tmp_path,
        "implementation/rescue-dirty-cleanup",
    )
    (worktree_path / "src" / "app.py").write_text("VALUE = 'rescued dirty content'\n", encoding="utf-8")

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["removed_count"] == 0
    assert result["skipped_reason_counts"]["dirty_worktree_rescued"] == 1
    rescue_result = result["skipped"][0]["rescue_result"]
    assert rescue_result["preserved"] is True
    assert rescue_result["rescue_branch"].startswith("rescue/worktree/")
    assert worktree_path.exists()
    current_branch = _git(worktree_path, "branch", "--show-current")
    assert current_branch == rescue_result["rescue_branch"]
    assert _git(repo, "show", f"{current_branch}:src/app.py") == "VALUE = 'rescued dirty content'"


def test_implementation_supervisor_merges_rescued_worktree_and_deletes_it(tmp_path):
    repo, worktree_path, supervisor = _merged_cleanup_worktree_fixture(
        tmp_path,
        "implementation/rescue-dirty-merge",
    )
    (worktree_path / "src" / "app.py").write_text("VALUE = 'rescued and merged'\n", encoding="utf-8")

    cleanup_result = supervisor.cleanup_backlogged_worktrees()
    rescue_branch = cleanup_result["skipped"][0]["rescue_result"]["rescue_branch"]
    supervisor._main_status_for_worktree_reconciliation = lambda repo_root, worktree_root: []  # type: ignore[method-assign]

    reconcile_result = supervisor.reconcile_backlogged_worktrees()

    assert reconcile_result["reconciled_count"] == 1
    assert reconcile_result["cleanup_count"] == 1
    assert reconcile_result["processed"][0]["branch"] == rescue_branch
    assert reconcile_result["processed"][0]["cleanup_result"]["cleaned"] is True
    assert not worktree_path.exists()
    assert (repo / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 'rescued and merged'\n"
    rescue_branch_exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", rescue_branch],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rescue_branch_exists.returncode != 0


def test_implementation_supervisor_caps_dirty_worktree_evidence_samples(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    worktree_root = repo / "worktrees"
    worktree_root.mkdir()
    records = []
    for index in range(25):
        path = worktree_root / f"content-{index:02d}"
        path.mkdir()
        records.append({"worktree": str(path), "branch": f"refs/heads/implementation/content-{index:02d}", "HEAD": f"c{index}"})
    for index in range(3):
        path = worktree_root / f"unsupported-{index:02d}"
        path.mkdir()
        records.append({"worktree": str(path), "branch": f"refs/heads/implementation/unsupported-{index:02d}", "HEAD": f"u{index}"})

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )
    evidence_calls: list[str] = []
    monkeypatch.setattr(supervisor, "_git_worktree_records", lambda _repo: records)
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    monkeypatch.setattr(supervisor, "_git_ref_is_ancestor", lambda _repo, _ancestor, _descendant: True)
    monkeypatch.setattr(supervisor, "_git_status_short", lambda path: [" M generated.txt"])

    def fake_redundancy(path: Path, _dirty: list[str], _target_ref: str) -> dict[str, str]:
        reason = "unsupported_status" if "unsupported" in path.name else "content_not_in_target"
        return {"redundant": False, "reason": reason}

    def fake_evidence(path: Path, _dirty: list[str]) -> dict[str, str]:
        evidence_calls.append(path.name)
        return {"diff_stat": f"{path.name} | 1 +"}

    monkeypatch.setattr(supervisor, "_redundant_dirty_worktree_status", fake_redundancy)
    monkeypatch.setattr(supervisor, "_dirty_worktree_evidence", fake_evidence)
    monkeypatch.setattr(
        supervisor,
        "_rescue_dirty_worktree",
        lambda _path, **_kwargs: {"attempted": True, "preserved": False, "reason": "simulated_rescue_failure"},
    )

    result = supervisor.cleanup_backlogged_worktrees()

    assert result["dirty_worktree_groups"]["content_not_in_target"]["count"] == 25
    assert result["dirty_worktree_groups"]["unsupported_status"]["count"] == 3
    assert sum(1 for item in evidence_calls if item.startswith("content-")) == 20
    assert sum(1 for item in evidence_calls if item.startswith("unsupported-")) == 3
    assert result["dirty_worktree_groups"]["content_not_in_target"]["samples"][-1]["dirty_evidence"]
    events = [
        json.loads(line)
        for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "merged_worktree_cleanup"
    assert events[-1]["removed_count"] == 0
    assert events[-1]["dirty_worktree_groups"]["content_not_in_target"]["count"] == 25


def test_implementation_supervisor_rechecks_cleanup_scan_cache_for_dirty_blockers(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "cached-dirty"
    worktree_path.mkdir(parents=True)
    records = [
        {
            "worktree": str(worktree_path),
            "branch": "refs/heads/implementation/cached-dirty",
            "HEAD": "abc123",
        }
    ]

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
            worktree_root=worktree_root,
            worktree_scan_cache_ttl_seconds=60,
        )
    )
    ancestry_calls = {"count": 0}
    status_calls = {"count": 0}
    monkeypatch.setattr(supervisor, "_git_worktree_records", lambda _repo: records)
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    monkeypatch.setattr(supervisor, "_git_current_branch", lambda _repo: "main")
    monkeypatch.setattr(supervisor, "_git_ref_commit", lambda _repo, _ref: "target-commit")

    def fake_ancestor(_repo: Path, _ancestor: str, _descendant: str) -> bool:
        ancestry_calls["count"] += 1
        return True

    def fake_status(_path: Path) -> list[str]:
        status_calls["count"] += 1
        return [" M generated.txt"]

    monkeypatch.setattr(supervisor, "_git_ref_is_ancestor", fake_ancestor)
    monkeypatch.setattr(supervisor, "_git_status_short", fake_status)
    monkeypatch.setattr(
        supervisor,
        "_redundant_dirty_worktree_status",
        lambda _path, _dirty, _target: {"redundant": False, "reason": "content_not_in_target"},
    )
    monkeypatch.setattr(supervisor, "_dirty_worktree_evidence", lambda _path, _dirty: {"diff_stat": "generated.txt | 1 +"})
    monkeypatch.setattr(
        supervisor,
        "_rescue_dirty_worktree",
        lambda _path, **_kwargs: {"attempted": True, "preserved": False, "reason": "simulated_rescue_failure"},
    )

    first = supervisor.cleanup_backlogged_worktrees()
    second = supervisor.cleanup_backlogged_worktrees()

    assert first["scan_cache_hit_count"] == 0
    assert first["scan_cache_written"] is True
    assert second["scan_cache_hit_count"] == 0
    assert second["dirty_worktree_groups"]["content_not_in_target"]["count"] == 1
    assert second["dirty_worktree_groups"]["content_not_in_target"]["samples"][0]["dirty_evidence"]
    assert ancestry_calls["count"] == 4
    assert status_calls["count"] == 2


def test_implementation_supervisor_reconciles_clean_backlogged_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-010-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    feature = repo / "feature.txt"
    feature.write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-010-attempt-1-123"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["candidate_count"] == 1
    assert result["processed_count"] == 1
    assert result["reconciled_count"] == 1
    assert result["cleanup_count"] == 1
    assert (repo / "feature.txt").read_text(encoding="utf-8") == "feature\n"
    assert not worktree_path.exists()
    branch_exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch_name],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert branch_exists.returncode != 0


def test_implementation_supervisor_preflights_conflicting_backlogged_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-010-conflict-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    marker.write_text("branch change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    marker.write_text("main change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "main change")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-010-conflict-attempt-1-123"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["candidate_count"] == 1
    assert result["processed_count"] == 1
    assert result["reconciled_count"] == 0
    assert result["preflight_blocked_count"] == 1
    assert result["processed"][0]["merge_result"]["reason"] == "preflight_merge_conflict"
    assert result["processed"][0]["preflight_result"]["mergeable"] is False
    assert "README.md" in result["processed"][0]["preflight_result"]["conflict_paths"]
    assert marker.read_text(encoding="utf-8") == "main change\n"
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert worktree_path.exists()


def test_implementation_supervisor_escalates_preflight_conflict_to_configured_resolver(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-010-conflict-attempt-1-456"
    _git(repo, "checkout", "-b", branch_name)
    marker.write_text("branch change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    marker.write_text("main change\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "main change")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-010-conflict-attempt-1-456"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            llm_merge_resolver_command="configured-resolver",
        )
    )
    merge_calls: list[str] = []

    class ResolverBackedDaemon:
        def _merge_branch_to_main(self, branch, task, attempt):
            merge_calls.append(branch)
            return {"attempted": True, "merged": True, "returncode": 0}

        def _cleanup_merged_worktree(self, path, branch):
            return {"cleaned": True, "path": str(path), "branch": branch}

    supervisor._build_worktree_reconciliation_daemon = lambda: ResolverBackedDaemon()  # type: ignore[method-assign]

    result = supervisor.reconcile_backlogged_worktrees()

    assert merge_calls == [branch_name]
    assert result["candidate_count"] == 1
    assert result["reconciled_count"] == 1
    assert result["preflight_blocked_count"] == 0
    assert result["preflight_resolver_escalation_count"] == 1
    assert result["processed"][0]["preflight_result"]["mergeable"] is False
    assert result["processed"][0]["preflight_resolver_escalated"] is True


def test_implementation_supervisor_defers_worktree_reconciliation_when_main_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-011-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    (repo / "dirty.txt").write_text("dirty checkout\n", encoding="utf-8")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-011-attempt-1-123"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["candidate_count"] == 1
    assert result["processed_count"] == 0
    assert result["reconciled_count"] == 0
    assert result["main_checkout_dirty"] is True
    assert "dirty.txt" in result["main_dirty_evidence"]["status_paths"]
    assert result["main_dirty_evidence"]["path_categories"]["untracked"] == 1
    assert any(item["reason"] == "main_checkout_dirty" for item in result["skipped"])
    assert worktree_path.exists()
    assert not (repo / "feature.txt").exists()


def test_implementation_supervisor_ignores_generated_objective_heap_dirty_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "implementation_plan" / "docs" / "objective-heap.md"
    objective_path.parent.mkdir(parents=True)
    objective_path.write_text("# Objective\n", encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md", "implementation_plan/docs/objective-heap.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-011-attempt-1-456"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    objective_path.write_text("# Objective\n\n## Generated goal\n", encoding="utf-8")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-011-attempt-1-456"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
            objective_path=objective_path,
        )
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["candidate_count"] == 1
    assert result["processed_count"] == 1
    assert result["main_checkout_dirty"] is False
    assert result["main_status_short"] == []
    assert result["raw_main_checkout_dirty"] is True
    assert "implementation_plan/docs/objective-heap.md" in result["raw_main_dirty_evidence"]["status_paths"]
    assert "implementation_plan/docs/objective-heap.md" in result["main_dirty_evidence"]["filtered_generated_status_paths"]
    assert (repo / "feature.txt").exists()
    assert objective_path.read_text(encoding="utf-8") == "# Objective\n\n## Generated goal\n"


def test_implementation_supervisor_ignores_generated_state_directory_dirty_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    state_dir = repo / "tmp" / "supervisor" / "state"
    diagnostics_path = state_dir / "submodule-merge-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    diagnostics_path.write_text('{"attempts": []}\n', encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md", "tmp/supervisor/state/submodule-merge-diagnostics.json")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-011-attempt-1-789"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    diagnostics_path.write_text('{"attempts": [{"task_id": "ACCEL-011"}]}\n', encoding="utf-8")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "accel-011-attempt-1-789"
    _git(repo, "worktree", "add", str(worktree_path), branch_name)

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    result = supervisor.reconcile_backlogged_worktrees()

    assert result["candidate_count"] == 1
    assert result["processed_count"] == 1
    assert result["main_checkout_dirty"] is False
    assert result["main_status_short"] == []
    assert result["raw_main_checkout_dirty"] is True
    assert "tmp/supervisor/state/submodule-merge-diagnostics.json" in result["raw_main_dirty_evidence"]["status_paths"]
    assert "tmp/supervisor/state/submodule-merge-diagnostics.json" in result["main_dirty_evidence"]["filtered_generated_status_paths"]
    assert (repo / "feature.txt").exists()
    assert diagnostics_path.read_text(encoding="utf-8") == '{"attempts": [{"task_id": "ACCEL-011"}]}\n'


def test_implementation_daemon_recognizes_configured_state_directory_as_generated_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "tmp" / "supervisor" / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )

    assert daemon._path_is_generated_status_output("tmp/supervisor/state/submodule-merge-diagnostics.json") is True
    assert daemon._path_is_generated_status_output("tmp/supervisor/state/submodule-merge-recovery-worktrees/attempt") is True
    assert daemon._path_is_generated_status_output("tmp/supervisor/unrelated.json") is False


def test_implementation_supervisor_reuses_reconciliation_scan_cache_when_main_dirty(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    worktree_root = repo / "worktrees"
    worktree_path = worktree_root / "cached-candidate"
    worktree_path.mkdir(parents=True)
    records = [
        {
            "worktree": str(worktree_path),
            "branch": "refs/heads/implementation/cached-candidate",
            "HEAD": "def456",
        }
    ]

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
            worktree_root=worktree_root,
            worktree_scan_cache_ttl_seconds=60,
        )
    )
    ref_exists_calls = {"count": 0}
    ancestry_calls = {"count": 0}
    status_calls = {"count": 0}
    monkeypatch.setattr(supervisor, "_git_worktree_records", lambda _repo: records)
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    monkeypatch.setattr(supervisor, "_git_current_branch", lambda _repo: "main")
    monkeypatch.setattr(supervisor, "_git_ref_commit", lambda _repo, _ref: "target-commit")
    monkeypatch.setattr(
        supervisor,
        "_main_status_for_worktree_reconciliation",
        lambda _repo, _root: [" M dirty-main.txt"],
    )

    def fake_ref_exists(_repo: Path, _ref: str) -> bool:
        ref_exists_calls["count"] += 1
        return True

    def fake_ancestor(_repo: Path, _ancestor: str, _descendant: str) -> bool:
        ancestry_calls["count"] += 1
        return False

    def fake_status(_path: Path) -> list[str]:
        status_calls["count"] += 1
        return []

    monkeypatch.setattr(supervisor, "_git_ref_exists", fake_ref_exists)
    monkeypatch.setattr(supervisor, "_git_ref_is_ancestor", fake_ancestor)
    monkeypatch.setattr(supervisor, "_git_status_short", fake_status)

    first = supervisor.reconcile_backlogged_worktrees()
    second = supervisor.reconcile_backlogged_worktrees()

    assert first["candidate_count"] == 1
    assert first["scan_cache_hit_count"] == 0
    assert first["scan_cache_written"] is True
    assert second["candidate_count"] == 1
    assert second["scan_cache_hit_count"] == 1
    assert any(item["reason"] == "main_checkout_dirty" and item.get("cached") for item in second["skipped"])
    assert ref_exists_calls["count"] == 1
    assert ancestry_calls["count"] == 2
    assert status_calls["count"] == 1


def test_implementation_supervisor_records_reconciliation_guardrail_for_dirty_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    branch_name = "implementation/accel-012-attempt-1-123"
    _git(repo, "checkout", "-b", branch_name)
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature branch")
    _git(repo, "checkout", "main")
    (repo / "dirty.txt").write_text("dirty checkout\n", encoding="utf-8")
    worktree_root = repo / "worktrees"
    _git(repo, "worktree", "add", str(worktree_root / "accel-012-attempt-1-123"), branch_name)
    (repo / "todo.md").write_text("# Agent Todos\n", encoding="utf-8")

    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )
    reconciliation = supervisor.reconcile_backlogged_worktrees()

    findings = supervisor.record_reconciliation_guardrails(
        reconciliation,
        {"attempted": True, "skipped": []},
    )

    assert len(findings) == 1
    assert findings[0]["kind"] == "main_checkout_dirty"
    todo_text = (repo / "todo.md").read_text(encoding="utf-8")
    assert "Resolve dirty main checkout blocking 1 worktree merges" in todo_text
    discovery_path = Path(findings[0]["discovery_path"])
    assert discovery_path.exists()
    discovery_text = discovery_path.read_text(encoding="utf-8")
    assert "dirty.txt" in discovery_text
    assert "## Main Checkout Evidence" in discovery_text
    assert "Path categories: `untracked=" in discovery_text
    manifest = json.loads(discovery_text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert "dirty.txt" in manifest["main_dirty_evidence"]["status_paths"]
    assert manifest["main_dirty_evidence"]["path_categories"]["untracked"] >= 1
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["reconciliation_guardrail_findings"][0]["follow_up_task_id"] == "PORTAL-001"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "reconciliation_guardrail"


def test_reconciliation_guardrail_ignores_generated_dirty_main_evidence(tmp_path):
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "state" / "strategy.json"
    discovery_dir = tmp_path / "discovery"
    stale_discovery = discovery_dir / "stale.md"
    stale_discovery.parent.mkdir(parents=True)
    stale_discovery_text = "# ACCEL-001 Reconciliation Guardrail\n\nCandidate count: 2\n"
    stale_discovery.write_text(stale_discovery_text, encoding="utf-8")
    todo_text = (
        "# Agent Todos\n\n"
        "## ACCEL-001 Resolve dirty main checkout blocking 2 worktree merges\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P1\n"
        "- Track: ops\n"
        "- Fingerprint: original\n"
        "- Dedupe key: reconciliation_guardrail:main_checkout_dirty\n"
        "- Validation: test -f "
        f"{stale_discovery}\n"
        "- Acceptance: Existing guardrail.\n"
    )
    todo_path.write_text(todo_text, encoding="utf-8")

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        reconciliation_result={
            "attempted": True,
            "main_checkout_dirty": True,
            "candidate_count": 2,
            "main_status_short": [" M discovery/stale.md", " M todo.md"],
            "main_dirty_evidence": {
                "status_short": [" M discovery/stale.md", " M todo.md"],
                "status_paths": ["discovery/stale.md", "todo.md"],
                "path_categories": {"modified": 2},
                "name_status": "M\tdiscovery/stale.md\nM\ttodo.md",
            },
            "candidates": [{"branch": "implementation/example", "path": "/tmp/example"}],
        },
        task_prefix="ACCEL-",
        repo_root=tmp_path,
    )

    assert findings == []
    assert todo_path.read_text(encoding="utf-8") == todo_text
    assert stale_discovery.read_text(encoding="utf-8") == stale_discovery_text


def test_reconciliation_guardrail_ignores_generated_objective_heap_status(tmp_path):
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "state" / "strategy.json"
    discovery_dir = tmp_path / "discovery"
    objective_path = tmp_path / "implementation_plan" / "docs" / "objective-heap.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    objective_path.parent.mkdir(parents=True)
    objective_path.write_text("# Objective\n", encoding="utf-8")

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        reconciliation_result={
            "attempted": True,
            "main_checkout_dirty": True,
            "candidate_count": 2,
            "main_status_short": [" M implementation_plan/docs/objective-heap.md"],
            "main_dirty_evidence": {
                "status_short": [" M implementation_plan/docs/objective-heap.md"],
                "status_paths": ["implementation_plan/docs/objective-heap.md"],
                "path_categories": {"modified": 1},
                "name_status": "M\timplementation_plan/docs/objective-heap.md",
            },
            "candidates": [{"branch": "implementation/example", "path": "/tmp/example"}],
        },
        task_prefix="ACCEL-",
        repo_root=tmp_path,
        additional_generated_status_paths=(objective_path,),
    )

    assert findings == []
    assert todo_path.read_text(encoding="utf-8") == "# Agent Todos\n"


def test_reconciliation_guardrail_filters_generated_submodule_todo_status(tmp_path):
    repo = tmp_path / "repo"
    todo_path = repo / "hallucinate_app" / "docs" / "todo.md"
    todo_path.parent.mkdir(parents=True)
    (repo / "hallucinate_app" / ".git").write_text("gitdir: ../.git/modules/hallucinate_app\n", encoding="utf-8")
    (repo / "src").mkdir(parents=True)
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    discovery_dir = repo / "data" / "discovery"

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=discovery_dir,
        reconciliation_result={
            "attempted": True,
            "main_checkout_dirty": True,
            "candidate_count": 3,
            "main_status_short": [
                " m hallucinate_app",
                " M data/discovery/existing.md",
                " M src/runtime.py",
            ],
            "main_dirty_evidence": {
                "status_short": [
                    " m hallucinate_app",
                    " M data/discovery/existing.md",
                    " M src/runtime.py",
                ],
                "status_paths": ["hallucinate_app", "data/discovery/existing.md", "src/runtime.py"],
                "path_categories": {"modified": 2, "other_dirty": 1},
                "name_status": "M\thallucinate_app\nM\tdata/discovery/existing.md\nM\tsrc/runtime.py",
            },
            "candidates": [{"branch": "implementation/example", "path": "/tmp/example"}],
        },
        task_prefix="ACCEL-",
        repo_root=repo,
    )

    assert len(findings) == 1
    discovery_path = Path(findings[0]["discovery_path"])
    manifest = json.loads(discovery_path.read_text(encoding="utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])
    evidence = manifest["main_dirty_evidence"]
    assert evidence["status_paths"] == ["src/runtime.py"]
    assert evidence["path_categories"] == {"modified": 1}
    assert "hallucinate_app" in evidence["filtered_generated_status_paths"]
    assert "data/discovery/existing.md" in evidence["filtered_generated_status_paths"]


def test_reconciliation_guardrail_records_use_full_dirty_group_counts():
    records = reconciliation_guardrail_records(
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {
                "content_not_in_target": {
                    "count": 77,
                    "samples": [
                        {
                            "branch": "implementation/example",
                            "path": "/tmp/worktrees/example",
                            "status_short": [" M generated.txt"],
                            "dirty_evidence": {
                                "name_status": "M\tgenerated.txt",
                                "diff_stat": "generated.txt | 2 ++",
                            },
                        }
                    ],
                }
            },
        }
    )

    assert len(records) == 1
    assert records[0]["candidate_count"] == 77
    assert records[0]["summary"] == "Resolve 77 dirty backlogged worktrees blocked by content_not_in_target"
    assert records[0]["dedupe_key"] == "reconciliation_guardrail:dirty_backlogged_worktree:content_not_in_target"
    assert records[0]["samples"][0]["dirty_evidence"]["name_status"] == "M\tgenerated.txt"
    plan = reconciliation_guardrail_plan(records[0])
    assert plan["candidate_count"] == 77
    assert "generated.txt" in plan["sample_status_paths"]
    assert any(item["action"] == "compare_dirty_content_to_target" for item in plan["actions"])


def test_reconciliation_guardrail_records_preflight_merge_conflict_bundle():
    records = reconciliation_guardrail_records(
        reconciliation_result={
            "attempted": True,
            "processed": [
                {
                    "branch": "implementation/alpha",
                    "path": "/tmp/worktrees/alpha",
                    "target_ref": "main",
                    "preflight_result": {
                        "mergeable": False,
                        "reason": "preflight_merge_conflict",
                        "conflict_paths": ["docs/todo.md", "hallucinate_app"],
                    },
                },
                {
                    "branch": "implementation/beta",
                    "path": "/tmp/worktrees/beta",
                    "target_ref": "main",
                    "preflight_result": {
                        "mergeable": False,
                        "reason": "preflight_merge_conflict",
                        "conflict_paths": ["docs/todo.md"],
                    },
                },
            ],
        }
    )

    assert len(records) == 1
    assert records[0]["kind"] == "preflight_merge_conflict"
    assert records[0]["candidate_count"] == 2
    assert records[0]["summary"] == "Resolve 2 preflight-conflicting backlogged worktree merges"
    assert records[0]["dedupe_key"] == "reconciliation_guardrail:preflight_merge_conflict"
    assert records[0]["conflict_path_counts"] == {"docs/todo.md": 2, "hallucinate_app": 1}
    plan = reconciliation_guardrail_plan(records[0])
    assert plan["top_conflict_paths"] == ["docs/todo.md", "hallucinate_app"]
    assert "docs/todo.md" in plan["sample_status_paths"]
    assert any(item["action"] == "bundle_preflight_conflicts_by_path" for item in plan["actions"])


def test_reconciliation_guardrail_dedupes_dirty_group_when_count_changes(tmp_path):
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "state" / "strategy.json"
    discovery_dir = tmp_path / "discovery"
    stale_discovery = discovery_dir / "stale.md"
    stale_discovery.parent.mkdir(parents=True)
    stale_discovery.write_text(
        "# ACCEL-001 Reconciliation Guardrail\n\n"
        "Candidate count: 40\n",
        encoding="utf-8",
    )
    todo_path.write_text(
        "# Agent Todos\n\n"
        "## ACCEL-001 Resolve 40 dirty backlogged worktrees blocked by content_not_in_target\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P2\n"
        "- Track: ops\n"
        f"- Validation: test -f {stale_discovery}\n"
        "- Acceptance: Reconciliation guardrail filed this because 40 branch or worktree cleanup candidates are blocked by content_not_in_target.\n",
        encoding="utf-8",
    )

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {
                "content_not_in_target": {
                    "count": 77,
                    "samples": [{"branch": "implementation/example", "path": "/tmp/example"}],
                }
            },
        },
        task_prefix="ACCEL-",
    )

    assert len(findings) == 1
    assert findings[0]["refreshed"] is True
    updated_todo = todo_path.read_text(encoding="utf-8")
    assert "ACCEL-002" not in updated_todo
    assert "Resolve 77 dirty backlogged worktrees blocked by content_not_in_target" in updated_todo
    assert "Dedupe key: reconciliation_guardrail:dirty_backlogged_worktree:content_not_in_target" in updated_todo
    assert "machine-readable reconciliation plan" in updated_todo
    discovery_text = stale_discovery.read_text(encoding="utf-8")
    assert "Candidate count: 77" in discovery_text
    assert "## Machine Readable Manifest" in discovery_text
    manifest = json.loads(discovery_text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert manifest["candidate_count"] == 77
    assert manifest["dedupe_key"] == "reconciliation_guardrail:dirty_backlogged_worktree:content_not_in_target"
    assert any(item["action"] == "compare_dirty_content_to_target" for item in manifest["actions"])

    stale_discovery.write_text("# stale discovery without manifest\nCandidate count: 77\n", encoding="utf-8")
    discovery_only_refresh = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {
                "content_not_in_target": {
                    "count": 77,
                    "samples": [{"branch": "implementation/example", "path": "/tmp/example"}],
                }
            },
        },
        task_prefix="ACCEL-",
    )
    assert len(discovery_only_refresh) == 1
    assert discovery_only_refresh[0]["refreshed"] is True
    assert todo_path.read_text(encoding="utf-8") == updated_todo
    assert "## Machine Readable Manifest" in stale_discovery.read_text(encoding="utf-8")

    assert record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        cleanup_result={
            "attempted": True,
            "dirty_worktree_groups": {
                "content_not_in_target": {
                    "count": 77,
                    "samples": [{"branch": "implementation/example", "path": "/tmp/example"}],
                }
            },
        },
        task_prefix="ACCEL-",
    ) == []


def test_reconciliation_guardrail_dedupes_preflight_conflict_when_count_changes(tmp_path):
    todo_path = tmp_path / "todo.md"
    strategy_path = tmp_path / "state" / "strategy.json"
    discovery_dir = tmp_path / "discovery"
    stale_discovery = discovery_dir / "stale.md"
    stale_discovery.parent.mkdir(parents=True)
    stale_discovery.write_text(
        "# ACCEL-001 Reconciliation Guardrail\n\n"
        "Candidate count: 1\n",
        encoding="utf-8",
    )
    todo_path.write_text(
        "# Agent Todos\n\n"
        "## ACCEL-001 Resolve 1 preflight-conflicting backlogged worktree merges\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P1\n"
        "- Track: ops\n"
        "- Dedupe key: reconciliation_guardrail:preflight_merge_conflict\n"
        "- Depends on:\n"
        f"- Validation: test -f {stale_discovery}\n"
        "- Acceptance: old\n",
        encoding="utf-8",
    )

    findings = record_reconciliation_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        reconciliation_result={
            "attempted": True,
            "processed": [
                {
                    "branch": "implementation/alpha",
                    "path": "/tmp/alpha",
                    "target_ref": "main",
                    "preflight_result": {
                        "mergeable": False,
                        "reason": "preflight_merge_conflict",
                        "conflict_paths": ["docs/todo.md"],
                    },
                },
                {
                    "branch": "implementation/beta",
                    "path": "/tmp/beta",
                    "target_ref": "main",
                    "preflight_result": {
                        "mergeable": False,
                        "reason": "preflight_merge_conflict",
                        "conflict_paths": ["docs/todo.md", "hallucinate_app"],
                    },
                },
            ],
        },
        task_prefix="ACCEL-",
    )

    assert len(findings) == 1
    assert findings[0]["refreshed"] is True
    updated_todo = todo_path.read_text(encoding="utf-8")
    assert "ACCEL-002" not in updated_todo
    assert "Resolve 2 preflight-conflicting backlogged worktree merges" in updated_todo
    discovery_text = stale_discovery.read_text(encoding="utf-8")
    assert "Candidate count: 2" in discovery_text
    assert "`hallucinate_app`" in discovery_text
    manifest = json.loads(discovery_text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert manifest["conflict_path_counts"] == {"docs/todo.md": 2, "hallucinate_app": 1}
    assert any(item["action"] == "resolve_code_or_submodule_conflicts_in_isolated_worktree" for item in manifest["actions"])


def test_implementation_daemon_deterministically_repairs_objective_heap_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        "# Objective Heap\n\n"
        "## VAIOS-G001 Root\n\n"
        "- Status: active\n"
        "- Goal: Keep the root goal.\n",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md")
    _git(repo, "commit", "-m", "seed objective heap")

    _git(repo, "checkout", "-b", "implementation/auto-objective")
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + "\n## VAIOS-G002 Feature goal\n\n- Status: active\n- Goal: Keep the feature goal.\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "feature objective")
    _git(repo, "checkout", "main")
    objective_path.write_text(
        objective_path.read_text(encoding="utf-8")
        + "\n## VAIOS-G003 Main goal\n\n- Status: active\n- Goal: Keep the main goal.\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "main objective")

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        objective_path=objective_path,
    )
    result = daemon._merge_branch_to_main(
        "implementation/auto-objective",
        PortalTask(
            task_id="AUTO-OBJECTIVE",
            title="Merge objective heap",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["deterministic_conflict_repair"][0]["resolved"] is True
    assert daemon._unmerged_worktree_paths(repo) == set()
    text = objective_path.read_text(encoding="utf-8")
    assert "## VAIOS-G002 Feature goal" in text
    assert "## VAIOS-G003 Main goal" in text
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/auto-objective", "HEAD") == ""


def test_implementation_daemon_deterministically_repairs_launch_readiness_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    doc_path = repo / "docs" / "launch" / "phone_desktop_glasses_readiness.md"
    test_path = repo / "tests" / "test_virtual_ai_os_launch_readiness_gate.py"
    doc_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    doc_path.write_text(
        """# Phone, Desktop, and Meta Glasses Launch Readiness

## Gate Contract

- Receipt packet: `data/virtual_ai_os/discovery/2026-06-23-vai-340-launch-readiness-gate.md`
- Backlog bridge: `VAI-340` for `VAIOS-G697`
- Python guard: `tests/test_virtual_ai_os_launch_readiness_gate.py`
""",
        encoding="utf-8",
    )
    test_path.write_text(
        """from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VAI_340_RECEIPT_PATH = (
    REPO_ROOT / "data" / "virtual_ai_os" / "discovery" / "2026-06-23-vai-340-launch-readiness-gate.md"
)
SWISSKNIFE_PACKAGE_PATH = REPO_ROOT / "swissknife" / "package.json"


def test_readiness_doc_and_heap_name_the_same_launch_validation_gate():
    doc_source = "docs/launch/phone_desktop_glasses_readiness.md"
    heap_source = "implementation_plan/docs/23-virtual-ai-os-objective-goal-heap.md"
    vai_source = VAI_340_RECEIPT_PATH.read_text(encoding="utf-8")

    for term in ["LaunchReadinessGate"]:
        assert term in doc_source
        assert term in heap_source
        assert term in vai_source

    assert "2026-06-23-vai-340-launch-readiness-gate.md" in doc_source
    assert "VAI-340" in doc_source
    assert "VAI-340" in vai_source
""",
        encoding="utf-8",
    )
    _git(repo, "add", "docs/launch/phone_desktop_glasses_readiness.md", "tests/test_virtual_ai_os_launch_readiness_gate.py")
    _git(repo, "commit", "-m", "seed launch readiness gate")

    _git(repo, "checkout", "-b", "implementation/mgw-launch")
    doc_path.write_text(
        doc_path.read_text(encoding="utf-8").replace(
            "- Backlog bridge: `VAI-340` for `VAIOS-G697`",
            "- MGW supervisor packet: `data/meta_glasses_display_widgets/discovery/2026-06-23-mgw-274-launch-readiness-gate.md`\n"
            "- Backlog bridge: `MGW-274` / `VAI-340` for `VAIOS-G697`",
        ),
        encoding="utf-8",
    )
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        .replace(
            "SWISSKNIFE_PACKAGE_PATH = REPO_ROOT / \"swissknife\" / \"package.json\"",
            "MGW_274_RECEIPT_PATH = (\n"
            "    REPO_ROOT\n"
            "    / \"data\"\n"
            "    / \"meta_glasses_display_widgets\"\n"
            "    / \"discovery\"\n"
            "    / \"2026-06-23-mgw-274-launch-readiness-gate.md\"\n"
            ")\n"
            "SWISSKNIFE_PACKAGE_PATH = REPO_ROOT / \"swissknife\" / \"package.json\"",
        )
        .replace(
            "    vai_source = VAI_340_RECEIPT_PATH.read_text(encoding=\"utf-8\")",
            "    mgw_source = MGW_274_RECEIPT_PATH.read_text(encoding=\"utf-8\")\n"
            "    vai_source = VAI_340_RECEIPT_PATH.read_text(encoding=\"utf-8\")",
        )
        .replace(
            "        assert term in vai_source",
            "        assert term in mgw_source\n        assert term in vai_source",
        )
        .replace(
            '    assert "2026-06-23-vai-340-launch-readiness-gate.md" in doc_source',
            '    assert "2026-06-23-mgw-274-launch-readiness-gate.md" in doc_source\n'
            '    assert "2026-06-23-vai-340-launch-readiness-gate.md" in doc_source',
        )
        .replace(
            '    assert "VAI-340" in doc_source',
            '    assert "MGW-274" in doc_source\n    assert "VAI-340" in doc_source',
        )
        .replace(
            '    assert "VAI-340" in vai_source',
            '    assert "MGW-274" in mgw_source\n    assert "VAI-340" in vai_source',
        ),
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "add mgw launch gate")

    _git(repo, "checkout", "main")
    doc_path.write_text(
        doc_path.read_text(encoding="utf-8").replace(
            "- Backlog bridge: `VAI-340` for `VAIOS-G697`",
            "- HAO backlog packet: `data/hallucinate_multimodal_control/discovery/2026-06-23-hao-436-launch-readiness-gate.md`\n"
            "- Backlog bridge: `HAO-436` / `VAI-340` for `VAIOS-G697`",
        ),
        encoding="utf-8",
    )
    test_path.write_text(
        test_path.read_text(encoding="utf-8")
        .replace(
            "SWISSKNIFE_PACKAGE_PATH = REPO_ROOT / \"swissknife\" / \"package.json\"",
            "HAO_436_RECEIPT_PATH = (\n"
            "    REPO_ROOT\n"
            "    / \"data\"\n"
            "    / \"hallucinate_multimodal_control\"\n"
            "    / \"discovery\"\n"
            "    / \"2026-06-23-hao-436-launch-readiness-gate.md\"\n"
            ")\n"
            "SWISSKNIFE_PACKAGE_PATH = REPO_ROOT / \"swissknife\" / \"package.json\"",
        )
        .replace(
            "    vai_source = VAI_340_RECEIPT_PATH.read_text(encoding=\"utf-8\")",
            "    hao_source = HAO_436_RECEIPT_PATH.read_text(encoding=\"utf-8\")\n"
            "    vai_source = VAI_340_RECEIPT_PATH.read_text(encoding=\"utf-8\")",
        )
        .replace(
            "        assert term in vai_source",
            "        assert term in hao_source\n        assert term in vai_source",
        )
        .replace(
            '    assert "2026-06-23-vai-340-launch-readiness-gate.md" in doc_source',
            '    assert "2026-06-23-hao-436-launch-readiness-gate.md" in doc_source\n'
            '    assert "2026-06-23-vai-340-launch-readiness-gate.md" in doc_source',
        )
        .replace(
            '    assert "VAI-340" in doc_source',
            '    assert "HAO-436" in doc_source\n    assert "VAI-340" in doc_source',
        )
        .replace(
            '    assert "VAI-340" in vai_source',
            '    assert "HAO-436" in hao_source\n    assert "VAI-340" in vai_source',
        ),
        encoding="utf-8",
    )
    _git(repo, "commit", "-am", "add hao launch gate")

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    result = daemon._merge_branch_to_main(
        "implementation/mgw-launch",
        PortalTask(
            task_id="MGW-274",
            title="Merge launch readiness gate",
            status="todo",
            completion="manual",
            priority="P0",
            track="launch",
        ),
        1,
    )

    assert result["merged"] is True
    repaired_paths = {item["path"] for item in result["deterministic_conflict_repair"]}
    assert repaired_paths == {
        "docs/launch/phone_desktop_glasses_readiness.md",
        "tests/test_virtual_ai_os_launch_readiness_gate.py",
    }
    assert daemon._unmerged_worktree_paths(repo) == set()
    doc_text = doc_path.read_text(encoding="utf-8")
    test_text = test_path.read_text(encoding="utf-8")
    assert "`HAO-436` / `MGW-274` / `VAI-340`" in doc_text
    assert "2026-06-23-hao-436-launch-readiness-gate.md" in doc_text
    assert "2026-06-23-mgw-274-launch-readiness-gate.md" in doc_text
    assert "HAO_436_RECEIPT_PATH" in test_text
    assert "MGW_274_RECEIPT_PATH" in test_text
    assert "assert term in hao_source" in test_text
    assert "assert term in mgw_source" in test_text
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/mgw-launch", "HEAD") == ""


def test_implementation_daemon_invokes_llm_resolver_for_dirty_checkout_blocker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "blocked.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "blocked.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-001")
    target.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch")
    _git(repo, "checkout", "main")
    target.write_text("local dirty\n", encoding="utf-8")

    capture_path = tmp_path / "dirty-resolver-prompt.txt"
    resolver_script = tmp_path / "dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-001",
        PortalTask(
            task_id="AUTO-001",
            title="Resolve dirty checkout blocker",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is False
    assert result["reason"] == "main_checkout_dirty_conflict"
    assert result["dirty_paths"] == ["blocked.txt"]
    assert result["llm_merge_resolver"]["applied"] is True
    prompt = capture_path.read_text(encoding="utf-8")
    assert "main_checkout_dirty_conflict" in prompt
    assert "Dirty paths: blocked.txt" in prompt


def test_implementation_daemon_restores_generated_dirty_checkout_overlap_without_llm(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    discovery = repo / "data" / "track" / "discovery" / "generated.md"
    todo = repo / "data" / "track" / "todo.md"
    discovery.parent.mkdir(parents=True, exist_ok=True)
    todo.write_text("# Agent Todos\n", encoding="utf-8")
    discovery.write_text("base generated\n", encoding="utf-8")
    _git(repo, "add", "data/track/todo.md", "data/track/discovery/generated.md")
    _git(repo, "commit", "-m", "base generated output")
    _git(repo, "checkout", "-b", "implementation/auto-generated")
    discovery.write_text("branch generated\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch generated output")
    _git(repo, "checkout", "main")
    discovery.write_text("local generated\n", encoding="utf-8")

    capture_path = tmp_path / "generated-dirty-resolver-prompt.txt"
    resolver_script = tmp_path / "generated_dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo,
        state_path=repo / "data" / "track" / "state" / "task_state.json",
        strategy_path=repo / "data" / "track" / "state" / "strategy.json",
        events_path=repo / "data" / "track" / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-generated",
        PortalTask(
            task_id="AUTO-GENERATED",
            title="Merge generated dirty output",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["restored_generated_dirty_overlap"][0]["path"] == "data/track/discovery/generated.md"
    assert result["restored_generated_dirty_overlap"][0]["restored"] is True
    assert discovery.read_text(encoding="utf-8") == "branch generated\n"
    assert not capture_path.exists()


def test_implementation_daemon_reconciles_generated_dirty_submodule_overlap_without_llm(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    submodule_todo = submodule / "docs" / "child.todo.md"
    submodule_todo.parent.mkdir()
    submodule_todo.write_text(
        """# Child Todos

## CHILD-001 Generated status

- Status: todo
- Fingerprint: base
""",
        encoding="utf-8",
    )
    _git(submodule, "add", "docs/child.todo.md")
    _git(submodule, "commit", "-m", "seed generated child todo")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "record child todo")

    _git(submodule, "checkout", "-b", "implementation/auto-001-submodule-libs-child")
    (submodule / "child.txt").write_text("branch\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "AUTO-001: update child")
    _git(repo, "checkout", "-b", "implementation/auto-001")
    _git(repo, "add", "libs/child")
    _git(repo, "commit", "-m", "AUTO-001: update child pointer")

    _git(repo, "checkout", "main")
    _git(submodule, "checkout", "main")
    submodule_todo.write_text(submodule_todo.read_text(encoding="utf-8").replace("base", "main-dirty"), encoding="utf-8")
    assert "libs/child" in _git(repo, "status", "--porcelain")

    capture_path = tmp_path / "unexpected-generated-submodule-resolver-prompt.txt"
    resolver_script = tmp_path / "unexpected_generated_submodule_resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=submodule_todo,
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-001",
        PortalTask(
            task_id="AUTO-001",
            title="Reconcile generated submodule dirt",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["generated_submodule_reconciliation"][0]["reconciled"] is True
    assert result["generated_submodule_reconciliation"][0]["generated_commit"]["committed"] is True
    assert result["generated_submodule_reconciliation"][0]["submodule_merge"]["merged"] is True
    assert not capture_path.exists()
    assert _git(repo, "status", "--porcelain", "--", "libs/child") == ""
    assert _git(submodule, "status", "--porcelain") == ""


def test_implementation_daemon_repairs_dirty_managed_main_merge_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "blocked.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "blocked.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-004")
    target.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch")
    _git(repo, "checkout", "-b", "driver", "main")

    capture_path = tmp_path / "managed-main-resolver-prompt.txt"
    resolver_script = tmp_path / "managed_main_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "subprocess.check_call(['git', 'checkout', '--', 'blocked.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_root=repo / "worktrees",
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )
    workspace_result = daemon._prepare_main_merge_workspace("main", "implementation/auto-004")
    workspace = Path(str(workspace_result["path"]))
    (workspace / "blocked.txt").write_text("dirty managed worktree\n", encoding="utf-8")

    result = daemon._merge_branch_to_main(
        "implementation/auto-004",
        PortalTask(
            task_id="AUTO-004",
            title="Repair dirty managed main merge worktree",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["llm_workspace_resolver"]["applied"] is True
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/auto-004", "main") == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "main_merge_worktree_dirty" in prompt
    assert "Dirty paths: blocked.txt" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "main_merge_workspace_blocker_resolved" for event in events)


def test_implementation_daemon_invokes_llm_resolver_for_dirty_submodule_checkout(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    _git(submodule, "checkout", "-b", "implementation/auto-001-submodule-libs-child")
    (submodule / "child.txt").write_text("branch\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "submodule branch")
    _git(submodule, "checkout", "main")
    (submodule / "child.txt").write_text("dirty\n", encoding="utf-8")

    capture_path = tmp_path / "submodule-dirty-prompt.txt"
    resolver_script = tmp_path / "submodule_dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "subprocess.check_call(['git', 'checkout', '--', 'child.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    results = daemon._merge_submodule_branches_to_main(
        "implementation/auto-001",
        task=PortalTask(
            task_id="AUTO-001",
            title="Resolve dirty submodule checkout",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=1,
    )

    assert results[0]["merged"] is True
    assert results[0]["path"] == "libs/child"
    assert _git(submodule, "rev-parse", "HEAD") == _git(
        submodule,
        "rev-parse",
        "implementation/auto-001-submodule-libs-child",
    )
    prompt = capture_path.read_text(encoding="utf-8")
    assert "submodule_checkout_dirty" in prompt
    assert "Dirty paths: child.txt" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "submodule_checkout_blocker_resolved" for event in events)


def test_implementation_daemon_skips_dirty_submodule_resolver_when_branch_already_merged(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    _git(submodule, "branch", "implementation/auto-001-submodule-libs-child")
    (submodule / "child.txt").write_text("dirty but unrelated\n", encoding="utf-8")

    capture_path = tmp_path / "unexpected-submodule-dirty-prompt.txt"
    resolver_script = tmp_path / "unexpected_submodule_dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    results = daemon._merge_submodule_branches_to_main(
        "implementation/auto-001",
        task=PortalTask(
            task_id="AUTO-001",
            title="Skip already merged dirty submodule",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=1,
    )

    assert results == [
        {
            "path": "libs/child",
            "branch": "implementation/auto-001-submodule-libs-child",
            "default_branch": "main",
            "merged": True,
            "reason": "already_merged",
        }
    ]
    assert not capture_path.exists()
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "dirty but unrelated\n"


def test_implementation_daemon_repairs_stale_submodule_worktree_config(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    submodule_git_dir = Path(_git(submodule, "rev-parse", "--absolute-git-dir"))
    stale_worktree = "../../../../../missing/worktree/libs/child"
    _git(repo, "config", "--file", str(submodule_git_dir / "config"), "core.worktree", stale_worktree)

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
    )

    result = daemon._repair_stale_submodule_worktree_configs(repo)

    assert result["repaired_count"] == 1
    assert result["repairs"][0]["module_path"] == "libs/child"
    assert result["repairs"][0]["old_worktree"] == stale_worktree
    assert _git(submodule, "status", "--short") == ""


def test_implementation_daemon_commits_llm_resolved_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(
        ["git", "merge", "feature"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0
    target.write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    result = daemon._commit_llm_resolved_merge(repo)

    assert result["completed"] is True
    assert target.read_text(encoding="utf-8") == "resolved\n"
    no_merge_head = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_merge_head.returncode != 0


def test_implementation_daemon_accepts_resolver_committed_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-merge")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")

    resolver_script = tmp_path / "resolver_commits.py"
    resolver_script.write_text(
        "import pathlib, subprocess\n"
        "pathlib.Path('conflict.txt').write_text('resolved by resolver\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'conflict.txt'])\n"
        "subprocess.check_call(['git', 'commit', '--no-edit'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))}",
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-merge",
        PortalTask(
            task_id="AUTO-002",
            title="Accept resolver committed merge",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["llm_merge_commit_result"]["reason"] == "resolver_committed_merge"
    assert target.read_text(encoding="utf-8") == "resolved by resolver\n"
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/auto-merge", "HEAD") == ""


def test_implementation_daemon_invokes_llm_resolver_for_submodule_merge_conflict(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    _git(submodule, "checkout", "-b", "implementation/auto-003-submodule-libs-child")
    (submodule / "child.txt").write_text("branch\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "submodule branch")
    _git(submodule, "checkout", "main")
    (submodule / "child.txt").write_text("main\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "main change")

    capture_path = tmp_path / "submodule-conflict-prompt.txt"
    resolver_script = tmp_path / "submodule_conflict_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "pathlib.Path('child.txt').write_text('resolved\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'child.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    results = daemon._merge_submodule_branches_to_main(
        "implementation/auto-003",
        task=PortalTask(
            task_id="AUTO-003",
            title="Resolve submodule merge conflict",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=1,
    )

    assert results[0]["merged"] is True
    assert results[0]["ff_only_result"]["returncode"] != 0
    assert results[0]["llm_merge_resolver"]["applied"] is True
    assert results[0]["llm_merge_commit_result"]["completed"] is True
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "resolved\n"
    assert _git(
        submodule,
        "merge-base",
        "--is-ancestor",
        "implementation/auto-003-submodule-libs-child",
        "HEAD",
    ) == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "submodule_merge_conflict" in prompt
    assert "Unmerged paths: child.txt" in prompt
