from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from ipfs_accelerate_py.agent_supervisor.objective_graph import parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.todo_vector_index import parse_todo_vector_records, write_todo_vector_index
from ipfs_accelerate_py.agent_supervisor.objective_tracker import fibonacci_priority
from ipfs_accelerate_py.agent_supervisor import merge_resolver
from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    ConfiguredMergeResolverRunner,
    build_configured_merge_resolver_runner,
)
from ipfs_accelerate_py.agent_supervisor import task_proposal_router
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    ConfiguredTaskProposalRouterRunner,
    build_configured_task_proposal_router_runner,
    build_repo_task_proposal_router_runner,
    run_configured_task_proposal_router_cli,
    standard_task_proposal_requested_outputs,
)
from ipfs_accelerate_py.agent_supervisor import multi_supervisor_runner
from ipfs_accelerate_py.agent_supervisor.multi_supervisor_runner import (
    ConfiguredMultiSupervisorCliRunner,
    ConfiguredMultiSupervisorLauncher,
    ImplementationSupervisorTrackConfig,
    build_arg_parser as build_multi_supervisor_arg_parser,
    build_configured_multi_supervisor_launcher,
    build_configured_multi_supervisor_cli_runner,
    common_args_from_parsed_args,
    implementation_supervisor_compact_track_spec,
    implementation_supervisor_compact_track_specs,
    implementation_supervisor_common_args,
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
    apply_portal_implementation_daemon_defaults,
    apply_portal_implementation_daemon_defaults_from_paths,
    build_configured_implementation_daemon_runner,
    build_implementation_daemon_defaults_from_paths,
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
    build_objective_refill_defaults_from_paths,
)
from ipfs_accelerate_py.agent_supervisor import implementation_supervisor_runner
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
    SupervisorLoopResult,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    ConfiguredSupervisorEntrypoint,
    RestartPolicy,
    SupervisedChildSpec,
    background_supervisor_args,
    build_configured_implementation_supervisor_entrypoint,
    build_supervisor_runtime_operations,
    implementation_supervisor_args,
    launch_supervised_child,
    repair_supervisor_runtime,
    supervisor_is_running,
    supervisor_runtime_paths,
    terminate_supervised_child,
)
from ipfs_accelerate_py.agent_supervisor.wrapper_utils import (
    BootstrapPathCallbacks,
    BootstrapPathSpec,
    CodebaseScanEnvSettings,
    RuntimeEnvironmentCallbacks,
    android_validation_command_needs_environment,
    android_validation_environment_contract,
    apply_env_defaults,
    apply_environment_contract,
    bootstrap_runtime_environment,
    build_android_validation_callbacks,
    build_bootstrap_path_ensurer,
    build_bootstrap_path_resolver,
    build_default_llm_merge_resolver_command_callback,
    build_prefixed_bootstrap_path_callbacks,
    build_prefixed_default_llm_merge_resolver_command_callback,
    build_repo_runtime_environment_callbacks,
    build_runtime_environment_callback,
    build_runtime_environment_callbacks,
    csv_tuple,
    default_llm_merge_resolver_command,
    enforce_android_validation_environment,
    ensure_named_directories,
    ensure_runtime_pythonpath,
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
    repo_external_package_root,
    repo_external_package_roots,
    repo_root_from_env,
    repo_relative_or_default,
    repo_script_command,
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
    assert repo_external_package_root(tmp_path, "ipfs_accelerate") == tmp_path / "external" / "ipfs_accelerate"
    assert repo_external_package_root(tmp_path, "custom", external_dir="vendor") == tmp_path / "vendor" / "custom"
    absolute_package = tmp_path / "absolute-package"
    assert repo_external_package_root(tmp_path, absolute_package) == absolute_package
    assert repo_external_package_roots(tmp_path, ("ipfs_accelerate", "ipfs_datasets")) == (
        tmp_path / "external" / "ipfs_accelerate",
        tmp_path / "external" / "ipfs_datasets",
    )
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
    assert prefixed_codebase_scan_env_settings("WRAPPER_UTILS") == CodebaseScanEnvSettings(
        min_open_tasks=6,
        max_findings=8,
        cooldown_seconds=120,
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
    }
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_MIN_OPEN_TASKS", "21")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_MAX_FINDINGS", "13")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SCAN_COOLDOWN_SECONDS", "901")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "7")
    monkeypatch.setenv("WRAPPER_UTILS_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "5")
    assert prefixed_objective_refill_env_settings("WRAPPER_UTILS") == ObjectiveRefillEnvSettings(
        min_open_tasks=21,
        max_findings=13,
        cooldown_seconds=901,
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
        "objective_surplus_findings_per_goal": 7,
        "objective_surplus_min_terms_per_todo": 5,
    }
    assert task_board_env_var("WRAPPER_UTILS") == "WRAPPER_UTILS_TODO_PATH"
    assert task_board_env_var("WRAPPER_UTILS_") == "WRAPPER_UTILS_TODO_PATH"
    assert task_board_filename("roadmap") == "roadmap.todo.md"
    assert task_board_filename("roadmap", ".markdown") == "roadmap.todo.markdown"
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

    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(second), "existing"]))
    original_sys_path = list(sys.path)
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
    try:
        callbacks.enter()
        assert Path.cwd() == repo
    finally:
        os.chdir(cwd)


def test_default_llm_merge_resolver_command_prefers_env(monkeypatch):
    monkeypatch.setenv("PRIMARY_RESOLVER_COMMAND", "primary-command")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", "fallback-command")
    assert default_llm_merge_resolver_command(primary_env_var="PRIMARY_RESOLVER_COMMAND") == "primary-command"

    monkeypatch.delenv("PRIMARY_RESOLVER_COMMAND")
    assert default_llm_merge_resolver_command(primary_env_var="PRIMARY_RESOLVER_COMMAND") == "fallback-command"

    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    assert default_llm_merge_resolver_command(codex_args=("exec", "-")) == "/usr/bin/codex exec -"


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


def test_configured_implementation_supervisor_entrypoint_defaults_and_dispatches():
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
        worktree_submodule_paths=("packages/app", "external/lib"),
    ) == apply_portal_implementation_daemon_defaults(["--once"], defaults=defaults)


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
    assert captured["kwargs"]["hooks"] == (hook,)
    assert captured["kwargs"]["pass_complete_message"] == "paths complete: %s"


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
        seed_interoperability_goals=True,
    )
    codebase = build_codebase_refill_defaults_from_paths(
        paths,
        codebase_scan_discovery_dir_key="discovery_dir",
        codebase_scan_discovery_output_path="data/discovery",
        codebase_scan_min_open_tasks=3,
        codebase_scan_max_findings=8,
        codebase_scan_cooldown_seconds=120,
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
        seed_interoperability_goals=True,
    )
    assert codebase == CodebaseRefillDefaults(
        codebase_scan_discovery_dir=tmp_path / "discovery",
        codebase_scan_discovery_output_path="data/discovery",
        codebase_scan_min_open_tasks=3,
        codebase_scan_max_findings=8,
        codebase_scan_cooldown_seconds=120,
        codebase_scan_skip_prefixes=("data/", "scripts/"),
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
    assert config.objective_interoperability_focus == ("hallucinate_app", "swissknife")
    assert config.objective_interoperability_component_paths == (
        "hallucinate_app",
        "external/ipfs_accelerate",
    )


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


def test_implementation_supervisor_track_spec_uses_standard_state_layout():
    config = ImplementationSupervisorTrackConfig(
        name="VAI",
        script_path="scripts/virtual_ai_os_todo_supervisor.py",
        state_dir="data/virtual_ai_os/state",
        state_prefix="virtual_ai_os",
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


def test_implementation_supervisor_common_args_include_long_run_defaults():
    args = implementation_supervisor_common_args(
        implementation_command="bash resolve.sh",
        objective_scan_min_open_tasks=21,
        objective_scan_max_findings=13,
        objective_surplus_findings_per_goal=8,
        objective_surplus_min_terms_per_todo=5,
    )

    assert args[:3] == ["--implement", "--stale-seconds", "1800"]
    assert args[args.index("--implementation-command") + 1] == "bash resolve.sh"
    assert args[args.index("--llm-merge-resolver-command") + 1] == "bash resolve.sh"
    assert args[args.index("--objective-scan-min-open-tasks") + 1] == "21"
    assert args[args.index("--objective-scan-max-findings") + 1] == "13"
    assert args[args.index("--objective-surplus-findings-per-goal") + 1] == "8"
    assert args[args.index("--objective-surplus-min-terms-per-todo") + 1] == "5"
    assert args[args.index("--codebase-scan-cooldown-seconds") + 1] == "900"


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
    supervisor.run_once = lambda: {"stuck": False, "recovered": True}
    supervisor._supervisor_loop_recovery_delay_seconds = lambda: 0.0

    supervisor.run_forever()

    assert RecoveringLoop.calls == 2
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["type"] for event in events] == [
        "supervisor_loop_finished",
        "supervisor_loop_recovery_pass",
        "supervisor_loop_restarting_after_recovery",
        "supervisor_loop_finished",
    ]
    assert events[0]["status"] == "max_restarts_reached"
    assert events[1]["recovery"]["recovered"] is True
    assert events[-1]["status"] == "stopped"


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
    )

    assert daemon.worktree_submodule_paths == ("packages/app", "external/lib", "vendor/tools")
    assert daemon.llm_merge_resolver_command == "true"
    assert daemon.llm_merge_resolver_timeout_seconds == 5
    assert daemon.objective_path == repo / "objective-heap.md"
    assert daemon.objective_bundle_dir == repo / "objective_bundles"

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
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.llm_merge_resolver_command == "true"
    assert args.llm_merge_resolver_timeout_seconds == 5
    assert args.objective_path == repo / "objective-heap.md"
    assert args.objective_bundle_dir == repo / "objective_bundles"


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
            "resolved": False,
            "reason": "merge_reconcile_exception",
            "exception_type": "RuntimeError",
            "error": "merge workspace unavailable",
        }
    ]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconcile_exception"


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
    assert daemon._unresolved_merge_failures_by_task()["ACCEL-003"] == event


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
    assert result["codebase_deferred_reason"] == "objective_refill_produced_goal_work"
    assert "Close objective gap" in todo_text
    assert "Resolve code annotation" not in todo_text


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
