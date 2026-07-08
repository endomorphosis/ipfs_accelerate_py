"""Reusable runner helpers for configured implementation supervisors."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .wrapper_utils import (
    AgentSupervisorNamespacePaths,
    with_default,
    with_flag_default,
    with_repeated_default,
)


@dataclass(frozen=True)
class ImplementationSupervisorRunContext:
    """Parsed supervisor args and resolved config shared with runner hooks."""

    parsed: argparse.Namespace
    config: object
    state_path: Path
    strategy_path: Path
    events_path: Path
    daemon_events_path: Path


SupervisorRunHookCallback = Callable[[ImplementationSupervisorRunContext], Any]
SupervisorRefillRecordCallback = Callable[..., Any]


@dataclass(frozen=True)
class SupervisorRunHook:
    """One before/after hook around a configured supervisor pass."""

    phase: str
    message: str
    callback: SupervisorRunHookCallback
    log_level: int = logging.WARNING


RefillHookEntry = tuple[str, SupervisorRunHookCallback]
SupervisorBootstrapPathCallback = Callable[[Mapping[str, Path | str]], Any]
SupervisorBootstrapFactory = Callable[[Mapping[str, Path | str]], Any]
SupervisorBootstrapHookFactory = Callable[[Mapping[str, Path | str]], Sequence[SupervisorRunHook]]
SupervisorBootstrapOutputPathFactory = Callable[[Mapping[str, Path | str]], str | None]
SupervisorBootstrapExtraKwargsFactory = Callable[[Mapping[str, Path | str]], Mapping[str, Any] | None]
SupervisorMergeResolverCommand = str | Callable[[], str]


def _with_extra_kwargs(
    kwargs: dict[str, Any],
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _set_present(kwargs: dict[str, Any], **values: Any) -> None:
    for key, value in values.items():
        if value is not None:
            kwargs[key] = value


@dataclass(frozen=True)
class SupervisorRuntimeCallbacks:
    """Runtime repair and ensure callbacks bound to a supervisor wrapper."""

    ensure_running: SupervisorRunHookCallback
    repair_runtime: SupervisorRunHookCallback


@dataclass(frozen=True)
class ConfiguredSupervisorRuntime:
    """Project-bound runtime operations and runner wiring for a supervisor wrapper."""

    repo_root: Path
    script_path: Path
    process_match_any: Sequence[str]
    process_predicate: Callable[[int], bool] | None
    prepare_environment: Callable[[], None] | None
    implementation_lock_name: str
    startup_delay_seconds: float
    operations: Any

    def repair_runtime(self, state_dir: Path, state_prefix: str) -> dict[str, Any]:
        """Repair stale supervisor runtime markers for this wrapper."""

        return self.operations.repair_runtime(state_dir, state_prefix)

    def is_running(self, state_dir: Path, state_prefix: str) -> bool:
        """Return whether this supervisor wrapper appears to be running."""

        return self.operations.is_running(state_dir, state_prefix)

    def ensure_running(self, argv: Sequence[str], *, state_dir: Path, state_prefix: str) -> dict[str, Any]:
        """Ensure this supervisor wrapper is running in the background."""

        return self.operations.ensure_running(argv, state_dir=state_dir, state_prefix=state_prefix)

    def run_configured(
        self,
        argv: Sequence[str],
        *,
        logger: logging.Logger,
        daemon_script_path: Path | None = None,
        worktree_submodule_paths: Sequence[str] | None = None,
        hooks: Sequence[SupervisorRunHook] = (),
        once_complete_message: str = "Portal implementation supervisor check complete: %s",
        ensure_running: bool = False,
        ensure_running_message: str = "Supervisor ensure complete: %s",
        repair_runtime: bool = True,
        repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
    ) -> Any:
        """Run a configured supervisor using this runtime binding."""

        return run_configured_portal_implementation_supervisor_with_runtime(
            argv,
            repo_root=self.repo_root,
            logger=logger,
            script_path=self.script_path,
            process_match_any=self.process_match_any,
            process_predicate=self.process_predicate,
            prepare_environment=self.prepare_environment,
            implementation_lock_name=self.implementation_lock_name,
            startup_delay_seconds=self.startup_delay_seconds,
            daemon_script_path=daemon_script_path,
            worktree_submodule_paths=worktree_submodule_paths,
            hooks=hooks,
            once_complete_message=once_complete_message,
            ensure_running=ensure_running,
            ensure_running_message=ensure_running_message,
            repair_runtime=repair_runtime,
            repair_runtime_message=repair_runtime_message,
        )

    def run_configured_from_paths(
        self,
        argv: Sequence[str],
        paths: Mapping[str, Path | str],
        *,
        logger: logging.Logger,
        task_prefix: str,
        state_prefix: str,
        daemon_script_path: Path | str,
        supervisor_script_path: Path | str | None = None,
        todo_path_key: str = "todo_path",
        state_dir_key: str = "state_dir",
        worktree_root_key: str = "worktree_root",
        todo_path_flag: str = "--todo-path",
        max_restarts: int | str = 0,
        llm_merge_resolver_command: str = "",
        worktree_submodule_paths: Sequence[str] = (),
        generated_dirty_repair_enabled: bool = False,
        generated_dirty_repair_commit_subject: str | None = None,
        generated_dirty_repair_include_submodule_gitlinks: bool = False,
        generated_dirty_repair_max_paths: int | None = None,
        generated_dirty_repair_stale_lock_seconds: float | None = None,
        objective: "ObjectiveRefillDefaults | None" = None,
        codebase: "CodebaseRefillDefaults | None" = None,
        hooks: Sequence[SupervisorRunHook] = (),
        once_complete_message: str = "Portal implementation supervisor check complete: %s",
        ensure_running: bool = False,
        ensure_running_message: str = "Supervisor ensure complete: %s",
        repair_runtime: bool = True,
        repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
    ) -> Any:
        """Apply path-derived defaults and run this configured supervisor."""

        resolved_daemon_script_path = Path(daemon_script_path)
        args = apply_portal_implementation_supervisor_defaults_from_paths(
            argv,
            paths,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            daemon_script_path=resolved_daemon_script_path,
            supervisor_script_path=supervisor_script_path or self.script_path,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            max_restarts=max_restarts,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
            generated_dirty_repair_enabled=generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=(
                generated_dirty_repair_include_submodule_gitlinks
            ),
            generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
            objective=objective,
            codebase=codebase,
        )
        return self.run_configured(
            args,
            logger=logger,
            daemon_script_path=resolved_daemon_script_path,
            worktree_submodule_paths=worktree_submodule_paths,
            hooks=hooks,
            once_complete_message=once_complete_message,
            ensure_running=ensure_running,
            ensure_running_message=ensure_running_message,
            repair_runtime=repair_runtime,
            repair_runtime_message=repair_runtime_message,
        )

    def run_configured_from_bootstrap(
        self,
        argv: Sequence[str],
        *,
        logger: logging.Logger,
        ensure_paths: Callable[[], Mapping[str, Path | str]],
        task_prefix: str,
        state_prefix: str,
        daemon_script_path: Path | str,
        supervisor_script_path: Path | str | None = None,
        enter_runtime_environment: Callable[[], Any] | None = None,
        enter_runtime_before_paths: bool = False,
        path_callbacks: Sequence[SupervisorBootstrapPathCallback] = (),
        objective_factory: SupervisorBootstrapFactory | None = None,
        codebase_factory: SupervisorBootstrapFactory | None = None,
        hooks_factory: SupervisorBootstrapHookFactory | None = None,
        hooks: Sequence[SupervisorRunHook] = (),
        ensure_running_flag: str = "--ensure-running",
        ensure_running: bool | None = None,
        todo_path_key: str = "todo_path",
        state_dir_key: str = "state_dir",
        worktree_root_key: str = "worktree_root",
        todo_path_flag: str = "--todo-path",
        max_restarts: int | str = 0,
        llm_merge_resolver_command: str = "",
        worktree_submodule_paths: Sequence[str] = (),
        generated_dirty_repair_enabled: bool = False,
        generated_dirty_repair_commit_subject: str | None = None,
        generated_dirty_repair_include_submodule_gitlinks: bool = False,
        generated_dirty_repair_max_paths: int | None = None,
        generated_dirty_repair_stale_lock_seconds: float | None = None,
        once_complete_message: str = "Portal implementation supervisor check complete: %s",
        ensure_running_message: str = "Supervisor ensure complete: %s",
        repair_runtime: bool = True,
        repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
    ) -> Any:
        """Resolve bootstrap paths, build refill defaults, and start the supervisor."""

        args = list(argv)
        effective_ensure_running = ensure_running
        if effective_ensure_running is None:
            if ensure_running_flag:
                from .todo_daemon.supervisor_runtime import pop_bool_flag

                effective_ensure_running = pop_bool_flag(args, ensure_running_flag)
            else:
                effective_ensure_running = False
        if enter_runtime_environment is not None and enter_runtime_before_paths:
            enter_runtime_environment()
        paths = ensure_paths()
        if enter_runtime_environment is not None and not enter_runtime_before_paths:
            enter_runtime_environment()
        for callback in path_callbacks:
            callback(paths)
        objective = objective_factory(paths) if objective_factory is not None else None
        codebase = codebase_factory(paths) if codebase_factory is not None else None
        effective_hooks = hooks_factory(paths) if hooks_factory is not None else hooks
        return self.run_configured_from_paths(
            args,
            paths,
            logger=logger,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            daemon_script_path=daemon_script_path,
            supervisor_script_path=supervisor_script_path,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            max_restarts=max_restarts,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
            generated_dirty_repair_enabled=generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=(
                generated_dirty_repair_include_submodule_gitlinks
            ),
            generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
            objective=objective,
            codebase=codebase,
            hooks=effective_hooks,
            once_complete_message=once_complete_message,
            ensure_running=effective_ensure_running,
            ensure_running_message=ensure_running_message,
            repair_runtime=repair_runtime,
            repair_runtime_message=repair_runtime_message,
        )


@dataclass(frozen=True)
class ConfiguredSupervisorRuntimeExports:
    """Stable public bindings derived from a configured supervisor runtime."""

    runtime: ConfiguredSupervisorRuntime
    process_match_any: tuple[str, ...]
    repair_runtime: Callable[[Path, str], dict[str, Any]]
    is_running: Callable[[Path, str], bool]
    ensure_running: Callable[..., dict[str, Any]]


def build_configured_supervisor_runtime_exports(
    runtime: ConfiguredSupervisorRuntime,
) -> ConfiguredSupervisorRuntimeExports:
    """Return reusable wrapper exports for a configured supervisor runtime."""

    return ConfiguredSupervisorRuntimeExports(
        runtime=runtime,
        process_match_any=tuple(runtime.process_match_any),
        repair_runtime=runtime.repair_runtime,
        is_running=runtime.is_running,
        ensure_running=runtime.ensure_running,
    )


@dataclass(frozen=True)
class ConfiguredSupervisorBootstrapRunner:
    """Project-bound supervisor bootstrap wiring with reusable run dispatch."""

    runtime: ConfiguredSupervisorRuntime
    logger: logging.Logger
    ensure_paths: Callable[[], Mapping[str, Path | str]]
    task_prefix: str
    state_prefix: str
    daemon_script_path: Path | str
    supervisor_script_path: Path | str | None = None
    enter_runtime_environment: Callable[[], Any] | None = None
    enter_runtime_before_paths: bool = False
    path_callbacks: Sequence[SupervisorBootstrapPathCallback] = ()
    objective_factory: SupervisorBootstrapFactory | None = None
    codebase_factory: SupervisorBootstrapFactory | None = None
    hooks_factory: SupervisorBootstrapHookFactory | None = None
    hooks: Sequence[SupervisorRunHook] = ()
    ensure_running_flag: str = "--ensure-running"
    ensure_running: bool | None = None
    todo_path_key: str = "todo_path"
    state_dir_key: str = "state_dir"
    worktree_root_key: str = "worktree_root"
    todo_path_flag: str = "--todo-path"
    max_restarts: int | str = 0
    llm_merge_resolver_command: SupervisorMergeResolverCommand = ""
    worktree_submodule_paths: Sequence[str] = ()
    generated_dirty_repair_enabled: bool = False
    generated_dirty_repair_commit_subject: str | None = None
    generated_dirty_repair_include_submodule_gitlinks: bool = False
    generated_dirty_repair_max_paths: int | None = None
    generated_dirty_repair_stale_lock_seconds: float | None = None
    once_complete_message: str = "Portal implementation supervisor check complete: %s"
    ensure_running_message: str = "Supervisor ensure complete: %s"
    repair_runtime: bool = True
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s"

    def _resolved_llm_merge_resolver_command(self) -> str:
        command = self.llm_merge_resolver_command
        return str(command()) if callable(command) else str(command or "")

    def run(self, argv: Sequence[str] | None = None) -> Any:
        """Run the configured supervisor from bootstrap paths."""

        args = list(sys.argv[1:] if argv is None else argv)
        return self.runtime.run_configured_from_bootstrap(
            args,
            logger=self.logger,
            ensure_paths=self.ensure_paths,
            enter_runtime_environment=self.enter_runtime_environment,
            enter_runtime_before_paths=self.enter_runtime_before_paths,
            path_callbacks=self.path_callbacks,
            objective_factory=self.objective_factory,
            codebase_factory=self.codebase_factory,
            hooks_factory=self.hooks_factory,
            hooks=self.hooks,
            ensure_running_flag=self.ensure_running_flag,
            ensure_running=self.ensure_running,
            todo_path_key=self.todo_path_key,
            state_dir_key=self.state_dir_key,
            worktree_root_key=self.worktree_root_key,
            todo_path_flag=self.todo_path_flag,
            task_prefix=self.task_prefix,
            state_prefix=self.state_prefix,
            daemon_script_path=self.daemon_script_path,
            supervisor_script_path=self.supervisor_script_path,
            max_restarts=self.max_restarts,
            llm_merge_resolver_command=self._resolved_llm_merge_resolver_command(),
            worktree_submodule_paths=self.worktree_submodule_paths,
            generated_dirty_repair_enabled=self.generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=self.generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=(
                self.generated_dirty_repair_include_submodule_gitlinks
            ),
            generated_dirty_repair_max_paths=self.generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=self.generated_dirty_repair_stale_lock_seconds,
            once_complete_message=self.once_complete_message,
            ensure_running_message=self.ensure_running_message,
            repair_runtime=self.repair_runtime,
            repair_runtime_message=self.repair_runtime_message,
        )


def build_configured_supervisor_bootstrap_runner(
    *,
    runtime: ConfiguredSupervisorRuntime,
    logger: logging.Logger,
    ensure_paths: Callable[[], Mapping[str, Path | str]],
    task_prefix: str,
    state_prefix: str,
    daemon_script_path: Path | str,
    supervisor_script_path: Path | str | None = None,
    enter_runtime_environment: Callable[[], Any] | None = None,
    enter_runtime_before_paths: bool = False,
    path_callbacks: Sequence[SupervisorBootstrapPathCallback] = (),
    objective_factory: SupervisorBootstrapFactory | None = None,
    codebase_factory: SupervisorBootstrapFactory | None = None,
    hooks_factory: SupervisorBootstrapHookFactory | None = None,
    hooks: Sequence[SupervisorRunHook] = (),
    ensure_running_flag: str = "--ensure-running",
    ensure_running: bool | None = None,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    max_restarts: int | str = 0,
    llm_merge_resolver_command: SupervisorMergeResolverCommand = "",
    worktree_submodule_paths: Sequence[str] = (),
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str | None = None,
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    once_complete_message: str = "Portal implementation supervisor check complete: %s",
    ensure_running_message: str = "Supervisor ensure complete: %s",
    repair_runtime: bool = True,
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
) -> ConfiguredSupervisorBootstrapRunner:
    """Build reusable supervisor bootstrap/run wiring for a project wrapper."""

    return ConfiguredSupervisorBootstrapRunner(
        runtime=runtime,
        logger=logger,
        ensure_paths=ensure_paths,
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        daemon_script_path=daemon_script_path,
        supervisor_script_path=supervisor_script_path,
        enter_runtime_environment=enter_runtime_environment,
        enter_runtime_before_paths=enter_runtime_before_paths,
        path_callbacks=tuple(path_callbacks),
        objective_factory=objective_factory,
        codebase_factory=codebase_factory,
        hooks_factory=hooks_factory,
        hooks=tuple(hooks),
        ensure_running_flag=ensure_running_flag,
        ensure_running=ensure_running,
        todo_path_key=todo_path_key,
        state_dir_key=state_dir_key,
        worktree_root_key=worktree_root_key,
        todo_path_flag=todo_path_flag,
        max_restarts=max_restarts,
        llm_merge_resolver_command=llm_merge_resolver_command,
        worktree_submodule_paths=tuple(worktree_submodule_paths),
        generated_dirty_repair_enabled=generated_dirty_repair_enabled,
        generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
        generated_dirty_repair_include_submodule_gitlinks=generated_dirty_repair_include_submodule_gitlinks,
        generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
        generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
        once_complete_message=once_complete_message,
        ensure_running_message=ensure_running_message,
        repair_runtime=repair_runtime,
        repair_runtime_message=repair_runtime_message,
    )


def build_configured_supervisor_runtime(
    *,
    repo_root: Path | str,
    script_path: Path | str,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
) -> ConfiguredSupervisorRuntime:
    """Build reusable runtime operations bound to a project supervisor wrapper."""

    from .todo_daemon.supervisor_runtime import build_supervisor_runtime_operations

    resolved_repo_root = Path(repo_root)
    resolved_script_path = Path(script_path)
    process_markers = tuple(process_match_any)
    operations = build_supervisor_runtime_operations(
        repo_root=resolved_repo_root,
        script_path=resolved_script_path,
        process_match_any=process_markers,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
    )
    return ConfiguredSupervisorRuntime(
        repo_root=resolved_repo_root,
        script_path=resolved_script_path,
        process_match_any=process_markers,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
        operations=operations,
    )


def build_script_supervisor_runtime(
    *,
    repo_root: Path | str,
    script_path: Path | str,
    process_match_any: Sequence[str] | None = None,
    extra_process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
) -> ConfiguredSupervisorRuntime:
    """Build supervisor runtime wiring for one wrapper script."""

    resolved_script_path = Path(script_path).resolve()
    if process_match_any is None:
        markers = (resolved_script_path.name, *tuple(extra_process_match_any))
    else:
        markers = tuple(process_match_any)
    return build_configured_supervisor_runtime(
        repo_root=repo_root,
        script_path=resolved_script_path,
        process_match_any=markers,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
    )


def build_script_supervisor_bootstrap_runner(
    *,
    repo_root: Path | str,
    script_path: Path | str,
    logger: logging.Logger,
    ensure_paths: Callable[[], Mapping[str, Path | str]],
    task_prefix: str,
    state_prefix: str,
    daemon_script_path: Path | str,
    supervisor_script_path: Path | str | None = None,
    process_match_any: Sequence[str] | None = None,
    extra_process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
    enter_runtime_environment: Callable[[], Any] | None = None,
    enter_runtime_before_paths: bool = False,
    path_callbacks: Sequence[SupervisorBootstrapPathCallback] = (),
    objective_factory: SupervisorBootstrapFactory | None = None,
    codebase_factory: SupervisorBootstrapFactory | None = None,
    hooks_factory: SupervisorBootstrapHookFactory | None = None,
    hooks: Sequence[SupervisorRunHook] = (),
    ensure_running_flag: str = "--ensure-running",
    ensure_running: bool | None = None,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    max_restarts: int | str = 0,
    llm_merge_resolver_command: SupervisorMergeResolverCommand = "",
    worktree_submodule_paths: Sequence[str] = (),
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str | None = None,
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    once_complete_message: str = "Portal implementation supervisor check complete: %s",
    ensure_running_message: str = "Supervisor ensure complete: %s",
    repair_runtime: bool = True,
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
) -> ConfiguredSupervisorBootstrapRunner:
    """Build script-bound supervisor runtime and bootstrap/run wiring."""

    runtime = build_script_supervisor_runtime(
        repo_root=repo_root,
        script_path=script_path,
        process_match_any=process_match_any,
        extra_process_match_any=extra_process_match_any,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
    )
    return build_configured_supervisor_bootstrap_runner(
        runtime=runtime,
        logger=logger,
        ensure_paths=ensure_paths,
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        daemon_script_path=daemon_script_path,
        supervisor_script_path=supervisor_script_path,
        enter_runtime_environment=enter_runtime_environment,
        enter_runtime_before_paths=enter_runtime_before_paths,
        path_callbacks=path_callbacks,
        objective_factory=objective_factory,
        codebase_factory=codebase_factory,
        hooks_factory=hooks_factory,
        hooks=hooks,
        ensure_running_flag=ensure_running_flag,
        ensure_running=ensure_running,
        todo_path_key=todo_path_key,
        state_dir_key=state_dir_key,
        worktree_root_key=worktree_root_key,
        todo_path_flag=todo_path_flag,
        max_restarts=max_restarts,
        llm_merge_resolver_command=llm_merge_resolver_command,
        worktree_submodule_paths=worktree_submodule_paths,
        generated_dirty_repair_enabled=generated_dirty_repair_enabled,
        generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
        generated_dirty_repair_include_submodule_gitlinks=(
            generated_dirty_repair_include_submodule_gitlinks
        ),
        generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
        generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
        once_complete_message=once_complete_message,
        ensure_running_message=ensure_running_message,
        repair_runtime=repair_runtime,
        repair_runtime_message=repair_runtime_message,
    )


@dataclass(frozen=True)
class ImplementationSupervisorDefaults:
    """Default CLI values for a project-specific implementation supervisor wrapper."""

    todo_path: Path
    state_dir: Path
    task_prefix: str
    state_prefix: str
    worktree_root: Path
    daemon_script_path: Path
    supervisor_script_path: Path
    todo_path_flag: str = "--todo-path"
    max_restarts: int | str = 0
    llm_merge_resolver_command: str = ""
    worktree_submodule_paths: Sequence[str] = ()
    generated_dirty_repair_enabled: bool = False
    generated_dirty_repair_commit_subject: str | None = None
    generated_dirty_repair_include_submodule_gitlinks: bool = False
    generated_dirty_repair_max_paths: int | None = None
    generated_dirty_repair_stale_lock_seconds: float | None = None


def _path_from_mapping(paths: Mapping[str, Path | str], key: str) -> Path:
    return Path(paths[key])


def _optional_path_from_mapping(
    paths: Mapping[str, Path | str],
    *,
    key: str | None = None,
    value: Path | str | None = None,
) -> Path | None:
    if value is not None:
        return Path(value)
    if key is None:
        return None
    return _path_from_mapping(paths, key)


def build_implementation_supervisor_defaults_from_paths(
    paths: Mapping[str, Path | str],
    *,
    task_prefix: str,
    state_prefix: str,
    daemon_script_path: Path | str,
    supervisor_script_path: Path | str,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    max_restarts: int | str = 0,
    llm_merge_resolver_command: str = "",
    worktree_submodule_paths: Sequence[str] = (),
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str | None = None,
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
) -> ImplementationSupervisorDefaults:
    """Build reusable implementation-supervisor defaults from resolved wrapper paths."""

    return ImplementationSupervisorDefaults(
        todo_path=_path_from_mapping(paths, todo_path_key),
        state_dir=_path_from_mapping(paths, state_dir_key),
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        worktree_root=_path_from_mapping(paths, worktree_root_key),
        daemon_script_path=Path(daemon_script_path),
        supervisor_script_path=Path(supervisor_script_path),
        todo_path_flag=todo_path_flag,
        max_restarts=max_restarts,
        llm_merge_resolver_command=llm_merge_resolver_command,
        worktree_submodule_paths=worktree_submodule_paths,
        generated_dirty_repair_enabled=generated_dirty_repair_enabled,
        generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
        generated_dirty_repair_include_submodule_gitlinks=generated_dirty_repair_include_submodule_gitlinks,
        generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
        generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
    )


@dataclass(frozen=True)
class ObjectiveRefillDefaults:
    """Default objective-refill CLI values for an implementation supervisor."""

    objective_path: Path | None = None
    objective_graph_path: Path | None = None
    objective_bundle_dir: Path | None = None
    objective_dataset_dir: Path | None = None
    objective_discovery_dir: Path | None = None
    objective_discovery_output_path: str | None = None
    objective_scan_min_open_tasks: int | None = None
    objective_scan_max_findings: int | None = None
    objective_scan_cooldown_seconds: int | None = None
    objective_refill_timeout_seconds: int | None = None
    objective_todo_vector_index_path: Path | None = None
    objective_surplus_findings_per_goal: int | None = None
    objective_surplus_min_terms_per_todo: int | None = None
    objective_goal_completion_todo_boards: Sequence[str] = ()
    objective_interoperability_focus: Sequence[str] = ()
    objective_interoperability_component_paths: Sequence[str] = ()
    objective_max_interoperability_goals: int | None = None
    objective_max_launch_readiness_goals: int | None = None
    refill_scan: bool = True
    seed_interoperability_goals: bool = False
    seed_launch_readiness_goals: bool = False


@dataclass(frozen=True)
class CodebaseRefillDefaults:
    """Default codebase-refill CLI values for an implementation supervisor."""

    codebase_scan_discovery_dir: Path | None = None
    codebase_scan_discovery_output_path: str | None = None
    codebase_scan_min_open_tasks: int | None = None
    codebase_scan_max_findings: int | None = None
    codebase_scan_cooldown_seconds: int | None = None
    codebase_refill_timeout_seconds: int | None = None
    codebase_scan_skip_prefixes: Sequence[str] = ()
    refill_scan: bool = True


def build_objective_refill_defaults_from_paths(
    paths: Mapping[str, Path | str],
    *,
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_graph_path_key: str | None = None,
    objective_graph_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    objective_dataset_dir_key: str | None = None,
    objective_dataset_dir: Path | str | None = None,
    objective_discovery_dir_key: str | None = None,
    objective_discovery_dir: Path | str | None = None,
    objective_discovery_output_path: str | None = None,
    objective_scan_min_open_tasks: int | None = None,
    objective_scan_max_findings: int | None = None,
    objective_scan_cooldown_seconds: int | None = None,
    objective_refill_timeout_seconds: int | None = None,
    objective_todo_vector_index_path_key: str | None = None,
    objective_todo_vector_index_path: Path | str | None = None,
    objective_surplus_findings_per_goal: int | None = None,
    objective_surplus_min_terms_per_todo: int | None = None,
    objective_goal_completion_todo_boards: Sequence[str] = (),
    objective_interoperability_focus: Sequence[str] = (),
    objective_interoperability_component_paths: Sequence[str] = (),
    objective_max_interoperability_goals: int | None = None,
    objective_max_launch_readiness_goals: int | None = None,
    refill_scan: bool = True,
    seed_interoperability_goals: bool = False,
    seed_launch_readiness_goals: bool = False,
) -> ObjectiveRefillDefaults:
    """Build reusable objective-refill defaults from resolved wrapper paths."""

    return ObjectiveRefillDefaults(
        objective_path=_optional_path_from_mapping(paths, key=objective_path_key, value=objective_path),
        objective_graph_path=_optional_path_from_mapping(
            paths,
            key=objective_graph_path_key,
            value=objective_graph_path,
        ),
        objective_bundle_dir=_optional_path_from_mapping(
            paths,
            key=objective_bundle_dir_key,
            value=objective_bundle_dir,
        ),
        objective_dataset_dir=_optional_path_from_mapping(
            paths,
            key=objective_dataset_dir_key,
            value=objective_dataset_dir,
        ),
        objective_discovery_dir=_optional_path_from_mapping(
            paths,
            key=objective_discovery_dir_key,
            value=objective_discovery_dir,
        ),
        objective_discovery_output_path=objective_discovery_output_path,
        objective_scan_min_open_tasks=objective_scan_min_open_tasks,
        objective_scan_max_findings=objective_scan_max_findings,
        objective_scan_cooldown_seconds=objective_scan_cooldown_seconds,
        objective_refill_timeout_seconds=objective_refill_timeout_seconds,
        objective_todo_vector_index_path=_optional_path_from_mapping(
            paths,
            key=objective_todo_vector_index_path_key,
            value=objective_todo_vector_index_path,
        ),
        objective_surplus_findings_per_goal=objective_surplus_findings_per_goal,
        objective_surplus_min_terms_per_todo=objective_surplus_min_terms_per_todo,
        objective_goal_completion_todo_boards=objective_goal_completion_todo_boards,
        objective_interoperability_focus=objective_interoperability_focus,
        objective_interoperability_component_paths=objective_interoperability_component_paths,
        objective_max_interoperability_goals=objective_max_interoperability_goals,
        objective_max_launch_readiness_goals=objective_max_launch_readiness_goals,
        refill_scan=refill_scan,
        seed_interoperability_goals=seed_interoperability_goals,
        seed_launch_readiness_goals=seed_launch_readiness_goals,
    )


def build_codebase_refill_defaults_from_paths(
    paths: Mapping[str, Path | str],
    *,
    codebase_scan_discovery_dir_key: str | None = None,
    codebase_scan_discovery_dir: Path | str | None = None,
    codebase_scan_discovery_output_path: str | None = None,
    codebase_scan_min_open_tasks: int | None = None,
    codebase_scan_max_findings: int | None = None,
    codebase_scan_cooldown_seconds: int | None = None,
    codebase_refill_timeout_seconds: int | None = None,
    codebase_scan_skip_prefixes: Sequence[str] = (),
    refill_scan: bool = True,
) -> CodebaseRefillDefaults:
    """Build reusable codebase-refill defaults from resolved wrapper paths."""

    return CodebaseRefillDefaults(
        codebase_scan_discovery_dir=_optional_path_from_mapping(
            paths,
            key=codebase_scan_discovery_dir_key,
            value=codebase_scan_discovery_dir,
        ),
        codebase_scan_discovery_output_path=codebase_scan_discovery_output_path,
        codebase_scan_min_open_tasks=codebase_scan_min_open_tasks,
        codebase_scan_max_findings=codebase_scan_max_findings,
        codebase_scan_cooldown_seconds=codebase_scan_cooldown_seconds,
        codebase_refill_timeout_seconds=codebase_refill_timeout_seconds,
        codebase_scan_skip_prefixes=codebase_scan_skip_prefixes,
        refill_scan=refill_scan,
    )


def _output_path_from_factory(
    paths: Mapping[str, Path | str],
    *,
    value: str | None = None,
    factory: SupervisorBootstrapOutputPathFactory | None = None,
) -> str | None:
    if factory is not None:
        return factory(paths)
    return value


def _extra_kwargs_from_factory(
    paths: Mapping[str, Path | str],
    *,
    values: Mapping[str, Any] | None = None,
    factory: SupervisorBootstrapExtraKwargsFactory | None = None,
) -> dict[str, Any] | None:
    kwargs = dict(values or {})
    if factory is not None:
        kwargs.update(factory(paths) or {})
    return kwargs or None


def build_objective_refill_defaults_factory(
    *,
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_graph_path_key: str | None = None,
    objective_graph_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    objective_dataset_dir_key: str | None = None,
    objective_dataset_dir: Path | str | None = None,
    objective_discovery_dir_key: str | None = None,
    objective_discovery_dir: Path | str | None = None,
    objective_discovery_output_path: str | None = None,
    objective_discovery_output_path_factory: SupervisorBootstrapOutputPathFactory | None = None,
    objective_scan_min_open_tasks: int | None = None,
    objective_scan_max_findings: int | None = None,
    objective_scan_cooldown_seconds: int | None = None,
    objective_refill_timeout_seconds: int | None = None,
    objective_todo_vector_index_path_key: str | None = None,
    objective_todo_vector_index_path: Path | str | None = None,
    objective_surplus_findings_per_goal: int | None = None,
    objective_surplus_min_terms_per_todo: int | None = None,
    objective_goal_completion_todo_boards: Sequence[str] = (),
    objective_interoperability_focus: Sequence[str] = (),
    objective_interoperability_component_paths: Sequence[str] = (),
    objective_max_interoperability_goals: int | None = None,
    objective_max_launch_readiness_goals: int | None = None,
    refill_scan: bool = True,
    seed_interoperability_goals: bool = False,
    seed_launch_readiness_goals: bool = False,
) -> SupervisorBootstrapFactory:
    """Build a reusable bootstrap factory for objective-refill defaults."""

    def factory(paths: Mapping[str, Path | str]) -> ObjectiveRefillDefaults:
        return build_objective_refill_defaults_from_paths(
            paths,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            objective_graph_path_key=objective_graph_path_key,
            objective_graph_path=objective_graph_path,
            objective_bundle_dir_key=objective_bundle_dir_key,
            objective_bundle_dir=objective_bundle_dir,
            objective_dataset_dir_key=objective_dataset_dir_key,
            objective_dataset_dir=objective_dataset_dir,
            objective_discovery_dir_key=objective_discovery_dir_key,
            objective_discovery_dir=objective_discovery_dir,
            objective_discovery_output_path=_output_path_from_factory(
                paths,
                value=objective_discovery_output_path,
                factory=objective_discovery_output_path_factory,
            ),
            objective_scan_min_open_tasks=objective_scan_min_open_tasks,
            objective_scan_max_findings=objective_scan_max_findings,
            objective_scan_cooldown_seconds=objective_scan_cooldown_seconds,
            objective_refill_timeout_seconds=objective_refill_timeout_seconds,
            objective_todo_vector_index_path_key=objective_todo_vector_index_path_key,
            objective_todo_vector_index_path=objective_todo_vector_index_path,
            objective_surplus_findings_per_goal=objective_surplus_findings_per_goal,
            objective_surplus_min_terms_per_todo=objective_surplus_min_terms_per_todo,
            objective_goal_completion_todo_boards=objective_goal_completion_todo_boards,
            objective_interoperability_focus=objective_interoperability_focus,
            objective_interoperability_component_paths=objective_interoperability_component_paths,
            objective_max_interoperability_goals=objective_max_interoperability_goals,
            objective_max_launch_readiness_goals=objective_max_launch_readiness_goals,
            refill_scan=refill_scan,
            seed_interoperability_goals=seed_interoperability_goals,
            seed_launch_readiness_goals=seed_launch_readiness_goals,
        )

    return factory


def build_codebase_refill_defaults_factory(
    *,
    codebase_scan_discovery_dir_key: str | None = None,
    codebase_scan_discovery_dir: Path | str | None = None,
    codebase_scan_discovery_output_path: str | None = None,
    codebase_scan_discovery_output_path_factory: SupervisorBootstrapOutputPathFactory | None = None,
    codebase_scan_min_open_tasks: int | None = None,
    codebase_scan_max_findings: int | None = None,
    codebase_scan_cooldown_seconds: int | None = None,
    codebase_refill_timeout_seconds: int | None = None,
    codebase_scan_skip_prefixes: Sequence[str] = (),
    refill_scan: bool = True,
) -> SupervisorBootstrapFactory:
    """Build a reusable bootstrap factory for codebase-refill defaults."""

    def factory(paths: Mapping[str, Path | str]) -> CodebaseRefillDefaults:
        return build_codebase_refill_defaults_from_paths(
            paths,
            codebase_scan_discovery_dir_key=codebase_scan_discovery_dir_key,
            codebase_scan_discovery_dir=codebase_scan_discovery_dir,
            codebase_scan_discovery_output_path=_output_path_from_factory(
                paths,
                value=codebase_scan_discovery_output_path,
                factory=codebase_scan_discovery_output_path_factory,
            ),
            codebase_scan_min_open_tasks=codebase_scan_min_open_tasks,
            codebase_scan_max_findings=codebase_scan_max_findings,
            codebase_scan_cooldown_seconds=codebase_scan_cooldown_seconds,
            codebase_refill_timeout_seconds=codebase_refill_timeout_seconds,
            codebase_scan_skip_prefixes=codebase_scan_skip_prefixes,
            refill_scan=refill_scan,
        )

    return factory


def build_namespace_objective_refill_defaults_factory(
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    use_bootstrap_keys: bool = False,
    objective_discovery_output_path: str | None = None,
    objective_discovery_output_path_factory: SupervisorBootstrapOutputPathFactory | None = None,
    objective_scan_min_open_tasks: int | None = None,
    objective_scan_max_findings: int | None = None,
    objective_scan_cooldown_seconds: int | None = None,
    objective_refill_timeout_seconds: int | None = None,
    objective_surplus_findings_per_goal: int | None = None,
    objective_surplus_min_terms_per_todo: int | None = None,
    objective_goal_completion_todo_boards: Sequence[str] = (),
    objective_interoperability_focus: Sequence[str] = (),
    objective_interoperability_component_paths: Sequence[str] = (),
    objective_max_interoperability_goals: int | None = None,
    objective_max_launch_readiness_goals: int | None = None,
    refill_scan: bool = True,
    seed_interoperability_goals: bool = False,
    seed_launch_readiness_goals: bool = False,
) -> SupervisorBootstrapFactory:
    """Build objective-refill defaults from a standard namespace path bundle."""

    path_kwargs: dict[str, Any]
    if use_bootstrap_keys:
        path_kwargs = {
            "objective_graph_path_key": "objective_graph_path",
            "objective_bundle_dir_key": "objective_bundle_dir",
            "objective_dataset_dir_key": "objective_dataset_dir",
            "objective_discovery_dir_key": "discovery_dir",
            "objective_todo_vector_index_path_key": "objective_todo_vector_index_path",
        }
    else:
        path_kwargs = {
            "objective_graph_path": namespace_paths.objective_graph_path,
            "objective_bundle_dir": namespace_paths.objective_bundle_dir,
            "objective_dataset_dir": namespace_paths.objective_dataset_dir,
            "objective_discovery_dir": namespace_paths.discovery_dir,
            "objective_todo_vector_index_path": namespace_paths.objective_todo_vector_index_path,
        }
    return build_objective_refill_defaults_factory(
        objective_path_key=objective_path_key,
        objective_path=objective_path,
        objective_discovery_output_path=objective_discovery_output_path,
        objective_discovery_output_path_factory=objective_discovery_output_path_factory,
        objective_scan_min_open_tasks=objective_scan_min_open_tasks,
        objective_scan_max_findings=objective_scan_max_findings,
        objective_scan_cooldown_seconds=objective_scan_cooldown_seconds,
        objective_refill_timeout_seconds=objective_refill_timeout_seconds,
        objective_surplus_findings_per_goal=objective_surplus_findings_per_goal,
        objective_surplus_min_terms_per_todo=objective_surplus_min_terms_per_todo,
        objective_goal_completion_todo_boards=objective_goal_completion_todo_boards,
        objective_interoperability_focus=objective_interoperability_focus,
        objective_interoperability_component_paths=objective_interoperability_component_paths,
        objective_max_interoperability_goals=objective_max_interoperability_goals,
        objective_max_launch_readiness_goals=objective_max_launch_readiness_goals,
        refill_scan=refill_scan,
        seed_interoperability_goals=seed_interoperability_goals,
        seed_launch_readiness_goals=seed_launch_readiness_goals,
        **path_kwargs,
    )


def build_namespace_codebase_refill_defaults_factory(
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    use_bootstrap_keys: bool = False,
    codebase_scan_discovery_output_path: str | None = None,
    codebase_scan_discovery_output_path_factory: SupervisorBootstrapOutputPathFactory | None = None,
    codebase_scan_min_open_tasks: int | None = None,
    codebase_scan_max_findings: int | None = None,
    codebase_scan_cooldown_seconds: int | None = None,
    codebase_refill_timeout_seconds: int | None = None,
    codebase_scan_skip_prefixes: Sequence[str] = (),
    refill_scan: bool = True,
) -> SupervisorBootstrapFactory:
    """Build codebase-refill defaults from a standard namespace path bundle."""

    return build_codebase_refill_defaults_factory(
        codebase_scan_discovery_dir_key="discovery_dir" if use_bootstrap_keys else None,
        codebase_scan_discovery_dir=None if use_bootstrap_keys else namespace_paths.discovery_dir,
        codebase_scan_discovery_output_path=codebase_scan_discovery_output_path,
        codebase_scan_discovery_output_path_factory=codebase_scan_discovery_output_path_factory,
        codebase_scan_min_open_tasks=codebase_scan_min_open_tasks,
        codebase_scan_max_findings=codebase_scan_max_findings,
        codebase_scan_cooldown_seconds=codebase_scan_cooldown_seconds,
        codebase_refill_timeout_seconds=codebase_refill_timeout_seconds,
        codebase_scan_skip_prefixes=codebase_scan_skip_prefixes,
        refill_scan=refill_scan,
    )


def _with_optional_default(args: Sequence[str], flag: str, value: object | None) -> list[str]:
    if value is None:
        return list(args)
    return with_default(args, flag, str(value))


def apply_portal_implementation_supervisor_defaults(
    argv: Sequence[str],
    *,
    defaults: ImplementationSupervisorDefaults,
    objective: ObjectiveRefillDefaults | None = None,
    codebase: CodebaseRefillDefaults | None = None,
) -> list[str]:
    """Apply reusable implementation-supervisor CLI defaults to ``argv``."""

    args = list(argv)
    args = with_default(args, defaults.todo_path_flag, str(defaults.todo_path))
    args = with_default(args, "--state-dir", str(defaults.state_dir))
    args = with_default(args, "--task-prefix", defaults.task_prefix)
    args = with_default(args, "--state-prefix", defaults.state_prefix)
    args = with_default(args, "--worktree-root", str(defaults.worktree_root))
    args = with_default(args, "--daemon-script-path", str(defaults.daemon_script_path))
    args = with_default(args, "--supervisor-script-path", str(defaults.supervisor_script_path))
    args = with_default(args, "--max-restarts", str(defaults.max_restarts))
    if defaults.llm_merge_resolver_command:
        args = with_default(args, "--llm-merge-resolver-command", defaults.llm_merge_resolver_command)
    if defaults.worktree_submodule_paths:
        args = with_repeated_default(args, "--worktree-submodule-path", defaults.worktree_submodule_paths)
    if defaults.generated_dirty_repair_enabled:
        args = with_flag_default(args, "--auto-commit-generated-dirty")
    if defaults.generated_dirty_repair_commit_subject:
        args = with_default(
            args,
            "--generated-dirty-commit-subject",
            defaults.generated_dirty_repair_commit_subject,
        )
    if not defaults.generated_dirty_repair_include_submodule_gitlinks:
        args = with_flag_default(args, "--no-generated-dirty-submodule-gitlinks")
    args = _with_optional_default(
        args,
        "--generated-dirty-max-paths",
        defaults.generated_dirty_repair_max_paths,
    )
    args = _with_optional_default(
        args,
        "--generated-dirty-stale-lock-seconds",
        defaults.generated_dirty_repair_stale_lock_seconds,
    )

    if objective is not None:
        if objective.refill_scan:
            args = with_flag_default(args, "--objective-refill-scan")
        if objective.seed_interoperability_goals:
            args = with_flag_default(args, "--objective-seed-interoperability-goals")
        if objective.seed_launch_readiness_goals:
            args = with_flag_default(args, "--objective-seed-launch-readiness-goals")
        if objective.objective_interoperability_focus:
            args = with_repeated_default(
                args,
                "--objective-interoperability-focus",
                objective.objective_interoperability_focus,
            )
        if objective.objective_interoperability_component_paths:
            args = with_repeated_default(
                args,
                "--objective-interoperability-component-path",
                objective.objective_interoperability_component_paths,
            )
        args = _with_optional_default(
            args,
            "--objective-max-interoperability-goals",
            objective.objective_max_interoperability_goals,
        )
        args = _with_optional_default(
            args,
            "--objective-max-launch-readiness-goals",
            objective.objective_max_launch_readiness_goals,
        )
        args = _with_optional_default(args, "--objective-path", objective.objective_path)
        args = _with_optional_default(args, "--objective-graph-path", objective.objective_graph_path)
        args = _with_optional_default(args, "--objective-bundle-dir", objective.objective_bundle_dir)
        args = _with_optional_default(args, "--objective-dataset-dir", objective.objective_dataset_dir)
        args = _with_optional_default(args, "--objective-discovery-dir", objective.objective_discovery_dir)
        args = _with_optional_default(
            args,
            "--objective-discovery-output-path",
            objective.objective_discovery_output_path,
        )
        args = _with_optional_default(
            args,
            "--objective-scan-min-open-tasks",
            objective.objective_scan_min_open_tasks,
        )
        args = _with_optional_default(
            args,
            "--objective-scan-max-findings",
            objective.objective_scan_max_findings,
        )
        args = _with_optional_default(
            args,
            "--objective-scan-cooldown-seconds",
            objective.objective_scan_cooldown_seconds,
        )
        args = _with_optional_default(
            args,
            "--objective-refill-timeout-seconds",
            objective.objective_refill_timeout_seconds,
        )
        args = _with_optional_default(
            args,
            "--objective-todo-vector-index-path",
            objective.objective_todo_vector_index_path,
        )
        args = _with_optional_default(
            args,
            "--objective-surplus-findings-per-goal",
            objective.objective_surplus_findings_per_goal,
        )
        args = _with_optional_default(
            args,
            "--objective-surplus-min-terms-per-todo",
            objective.objective_surplus_min_terms_per_todo,
        )
        if objective.objective_goal_completion_todo_boards:
            args = with_repeated_default(
                args,
                "--objective-goal-completion-todo-board",
                objective.objective_goal_completion_todo_boards,
            )

    if codebase is not None:
        if codebase.refill_scan:
            args = with_flag_default(args, "--codebase-refill-scan")
        args = _with_optional_default(
            args,
            "--codebase-scan-discovery-dir",
            codebase.codebase_scan_discovery_dir,
        )
        args = _with_optional_default(
            args,
            "--codebase-scan-discovery-output-path",
            codebase.codebase_scan_discovery_output_path,
        )
        args = _with_optional_default(
            args,
            "--codebase-scan-min-open-tasks",
            codebase.codebase_scan_min_open_tasks,
        )
        args = _with_optional_default(
            args,
            "--codebase-scan-max-findings",
            codebase.codebase_scan_max_findings,
        )
        args = _with_optional_default(
            args,
            "--codebase-scan-cooldown-seconds",
            codebase.codebase_scan_cooldown_seconds,
        )
        args = _with_optional_default(
            args,
            "--codebase-refill-timeout-seconds",
            codebase.codebase_refill_timeout_seconds,
        )
        if codebase.codebase_scan_skip_prefixes:
            args = with_repeated_default(
                args,
                "--codebase-scan-skip-prefix",
                codebase.codebase_scan_skip_prefixes,
            )
    return args


def apply_portal_implementation_supervisor_defaults_from_paths(
    argv: Sequence[str],
    paths: Mapping[str, Path | str],
    *,
    task_prefix: str,
    state_prefix: str,
    daemon_script_path: Path | str,
    supervisor_script_path: Path | str,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    max_restarts: int | str = 0,
    llm_merge_resolver_command: str = "",
    worktree_submodule_paths: Sequence[str] = (),
    generated_dirty_repair_enabled: bool = False,
    generated_dirty_repair_commit_subject: str | None = None,
    generated_dirty_repair_include_submodule_gitlinks: bool = False,
    generated_dirty_repair_max_paths: int | None = None,
    generated_dirty_repair_stale_lock_seconds: float | None = None,
    objective: ObjectiveRefillDefaults | None = None,
    codebase: CodebaseRefillDefaults | None = None,
) -> list[str]:
    """Apply implementation-supervisor CLI defaults directly from resolved wrapper paths."""

    return apply_portal_implementation_supervisor_defaults(
        argv,
        defaults=build_implementation_supervisor_defaults_from_paths(
            paths,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            daemon_script_path=daemon_script_path,
            supervisor_script_path=supervisor_script_path,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            max_restarts=max_restarts,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
            generated_dirty_repair_enabled=generated_dirty_repair_enabled,
            generated_dirty_repair_commit_subject=generated_dirty_repair_commit_subject,
            generated_dirty_repair_include_submodule_gitlinks=(
                generated_dirty_repair_include_submodule_gitlinks
            ),
            generated_dirty_repair_max_paths=generated_dirty_repair_max_paths,
            generated_dirty_repair_stale_lock_seconds=generated_dirty_repair_stale_lock_seconds,
        ),
        objective=objective,
        codebase=codebase,
    )


def _ordered_refill_entries(
    entries: Sequence[RefillHookEntry],
    order: Sequence[str] | None,
) -> list[RefillHookEntry]:
    if order is None:
        return list(entries)
    by_name = {name: callback for name, callback in entries}
    ordered: list[RefillHookEntry] = [
        (name, by_name[name])
        for name in order
        if name in by_name
    ]
    ordered_names = {name for name, _callback in ordered}
    ordered.extend((name, callback) for name, callback in entries if name not in ordered_names)
    return ordered


def _refill_hook_message(
    *,
    scope_label: str,
    finding_label: str,
    phase_label: str,
    runner_label: str,
) -> str:
    label = " ".join(part for part in (scope_label.strip(), finding_label.strip()) if part)
    return f"Recorded {label} findings {phase_label} {runner_label} pass: %s"


def build_supervisor_refill_hooks(
    entries: Sequence[RefillHookEntry],
    *,
    scope_label: str = "",
    before: bool = True,
    after_once: bool = True,
    after_once_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> tuple[SupervisorRunHook, ...]:
    """Build standard before/after-once refill hooks for a supervisor wrapper."""

    hooks: list[SupervisorRunHook] = []
    if before:
        hooks.extend(
            SupervisorRunHook(
                "before",
                _refill_hook_message(
                    scope_label=scope_label,
                    finding_label=finding_label,
                    phase_label="before",
                    runner_label="supervisor",
                ),
                callback,
                log_level=log_level,
            )
            for finding_label, callback in entries
        )
    if after_once:
        hooks.extend(
            SupervisorRunHook(
                "after_once",
                _refill_hook_message(
                    scope_label=scope_label,
                    finding_label=finding_label,
                    phase_label="after",
                    runner_label="supervisor",
                ),
                callback,
                log_level=log_level,
            )
            for finding_label, callback in _ordered_refill_entries(entries, after_once_order)
        )
    return tuple(hooks)


def build_supervisor_objective_refill_callback(
    callback: SupervisorRefillRecordCallback,
    *,
    discovery_dir: Path,
    objective_path: Path | None = None,
    repo_root: Path | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> SupervisorRunHookCallback:
    """Build a supervisor hook that records objective-refill findings."""

    def hook(ctx: ImplementationSupervisorRunContext) -> Any:
        kwargs: dict[str, Any] = {
            "todo_path": ctx.parsed.todo_path,
            "state_path": ctx.state_path,
            "strategy_path": ctx.strategy_path,
            "discovery_dir": discovery_dir,
            "task_header_prefix": ctx.parsed.task_prefix,
        }
        _set_present(
            kwargs,
            objective_path=getattr(ctx.parsed, "objective_path", None) or objective_path,
            bundle_dir=getattr(ctx.parsed, "objective_bundle_dir", None),
            dataset_dir=getattr(ctx.parsed, "objective_dataset_dir", None),
            todo_vector_index_path=getattr(ctx.parsed, "objective_todo_vector_index_path", None),
            repo_root=repo_root,
            min_open_tasks=getattr(ctx.parsed, "objective_scan_min_open_tasks", None),
            max_findings=getattr(ctx.parsed, "objective_scan_max_findings", None),
            cooldown_seconds=getattr(ctx.parsed, "objective_scan_cooldown_seconds", None),
            surplus_findings_per_goal=getattr(ctx.parsed, "objective_surplus_findings_per_goal", None),
            surplus_min_terms_per_todo=getattr(ctx.parsed, "objective_surplus_min_terms_per_todo", None),
        )
        return callback(**_with_extra_kwargs(kwargs, extra_kwargs))

    return hook


def build_supervisor_codebase_scan_refill_callback(
    callback: SupervisorRefillRecordCallback,
    *,
    discovery_dir: Path,
    repo_root: Path | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> SupervisorRunHookCallback:
    """Build a supervisor hook that records codebase-scan findings."""

    def hook(ctx: ImplementationSupervisorRunContext) -> Any:
        kwargs: dict[str, Any] = {
            "todo_path": ctx.parsed.todo_path,
            "state_path": ctx.state_path,
            "strategy_path": ctx.strategy_path,
            "discovery_dir": discovery_dir,
            "task_header_prefix": ctx.parsed.task_prefix,
        }
        _set_present(
            kwargs,
            repo_root=repo_root,
            min_open_tasks=getattr(ctx.parsed, "codebase_scan_min_open_tasks", None),
            max_findings=getattr(ctx.parsed, "codebase_scan_max_findings", None),
            cooldown_seconds=getattr(ctx.parsed, "codebase_scan_cooldown_seconds", None),
        )
        return callback(**_with_extra_kwargs(kwargs, extra_kwargs))

    return hook


def build_supervisor_retry_budget_refill_callback(
    callback: SupervisorRefillRecordCallback,
    *,
    discovery_dir: Path,
    extra_kwargs: dict[str, Any] | None = None,
) -> SupervisorRunHookCallback:
    """Build a supervisor hook that records retry-budget findings."""

    def hook(ctx: ImplementationSupervisorRunContext) -> Any:
        return callback(
            **_with_extra_kwargs(
                {
                    "todo_path": ctx.parsed.todo_path,
                    "events_path": ctx.daemon_events_path,
                    "strategy_path": ctx.strategy_path,
                    "discovery_dir": discovery_dir,
                    "task_header_prefix": ctx.parsed.task_prefix,
                },
                extra_kwargs,
            )
        )

    return hook


def build_supervisor_refill_hooks_from_recorders(
    *,
    discovery_dir: Path,
    objective_recorder: SupervisorRefillRecordCallback | None = None,
    codebase_scan_recorder: SupervisorRefillRecordCallback | None = None,
    retry_budget_recorder: SupervisorRefillRecordCallback | None = None,
    objective_path: Path | None = None,
    repo_root: Path | None = None,
    objective_extra_kwargs: dict[str, Any] | None = None,
    codebase_scan_extra_kwargs: dict[str, Any] | None = None,
    retry_budget_extra_kwargs: dict[str, Any] | None = None,
    scope_label: str = "",
    before: bool = True,
    after_once: bool = True,
    after_once_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> tuple[SupervisorRunHook, ...]:
    """Build standard supervisor refill hooks from configured recorder callbacks."""

    entries: list[RefillHookEntry] = []
    if objective_recorder is not None:
        entries.append(
            (
                "objective-goal",
                build_supervisor_objective_refill_callback(
                    objective_recorder,
                    discovery_dir=discovery_dir,
                    objective_path=objective_path,
                    repo_root=repo_root,
                    extra_kwargs=objective_extra_kwargs,
                ),
            )
        )
    if codebase_scan_recorder is not None:
        entries.append(
            (
                "codebase-scan",
                build_supervisor_codebase_scan_refill_callback(
                    codebase_scan_recorder,
                    discovery_dir=discovery_dir,
                    repo_root=repo_root,
                    extra_kwargs=codebase_scan_extra_kwargs,
                ),
            )
        )
    if retry_budget_recorder is not None:
        entries.append(
            (
                "retry-budget",
                build_supervisor_retry_budget_refill_callback(
                    retry_budget_recorder,
                    discovery_dir=discovery_dir,
                    extra_kwargs=retry_budget_extra_kwargs,
                ),
            )
        )
    return build_supervisor_refill_hooks(
        tuple(entries),
        scope_label=scope_label,
        before=before,
        after_once=after_once,
        after_once_order=after_once_order,
        log_level=log_level,
    )


def build_supervisor_refill_hooks_factory_from_recorders(
    *,
    discovery_dir_key: str | None = None,
    discovery_dir: Path | str | None = None,
    objective_recorder: SupervisorRefillRecordCallback | None = None,
    codebase_scan_recorder: SupervisorRefillRecordCallback | None = None,
    retry_budget_recorder: SupervisorRefillRecordCallback | None = None,
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    repo_root: Path | None = None,
    objective_extra_kwargs: Mapping[str, Any] | None = None,
    objective_extra_kwargs_factory: SupervisorBootstrapExtraKwargsFactory | None = None,
    codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
    codebase_scan_extra_kwargs_factory: SupervisorBootstrapExtraKwargsFactory | None = None,
    retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
    retry_budget_extra_kwargs_factory: SupervisorBootstrapExtraKwargsFactory | None = None,
    scope_label: str = "",
    before: bool = True,
    after_once: bool = True,
    after_once_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> SupervisorBootstrapHookFactory:
    """Build a reusable bootstrap factory for supervisor refill hooks."""

    def factory(paths: Mapping[str, Path | str]) -> tuple[SupervisorRunHook, ...]:
        resolved_discovery_dir = _optional_path_from_mapping(
            paths,
            key=discovery_dir_key,
            value=discovery_dir,
        )
        if resolved_discovery_dir is None:
            raise ValueError("discovery_dir or discovery_dir_key is required")
        return build_supervisor_refill_hooks_from_recorders(
            objective_recorder=objective_recorder,
            codebase_scan_recorder=codebase_scan_recorder,
            retry_budget_recorder=retry_budget_recorder,
            discovery_dir=resolved_discovery_dir,
            objective_path=_optional_path_from_mapping(
                paths,
                key=objective_path_key,
                value=objective_path,
            ),
            repo_root=repo_root,
            objective_extra_kwargs=_extra_kwargs_from_factory(
                paths,
                values=objective_extra_kwargs,
                factory=objective_extra_kwargs_factory,
            ),
            codebase_scan_extra_kwargs=_extra_kwargs_from_factory(
                paths,
                values=codebase_scan_extra_kwargs,
                factory=codebase_scan_extra_kwargs_factory,
            ),
            retry_budget_extra_kwargs=_extra_kwargs_from_factory(
                paths,
                values=retry_budget_extra_kwargs,
                factory=retry_budget_extra_kwargs_factory,
            ),
            scope_label=scope_label,
            before=before,
            after_once=after_once,
            after_once_order=after_once_order,
            log_level=log_level,
        )

    return factory


def build_supervisor_runtime_callbacks(
    argv: Sequence[str],
    *,
    repo_root: Path,
    script_path: Path,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
) -> SupervisorRuntimeCallbacks:
    """Build standard runtime repair/ensure callbacks for a supervisor wrapper."""

    from .todo_daemon.supervisor_runtime import build_supervisor_runtime_operations

    args = tuple(argv)
    operations = build_supervisor_runtime_operations(
        repo_root=repo_root,
        script_path=script_path,
        process_match_any=process_match_any,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
    )

    def ensure_running(ctx: ImplementationSupervisorRunContext) -> dict[str, Any]:
        return operations.ensure_running(
            args,
            state_dir=ctx.parsed.state_dir,
            state_prefix=ctx.parsed.state_prefix,
        )

    def repair_runtime(ctx: ImplementationSupervisorRunContext) -> dict[str, Any]:
        return operations.repair_runtime(
            ctx.parsed.state_dir,
            ctx.parsed.state_prefix,
        )

    return SupervisorRuntimeCallbacks(
        ensure_running=ensure_running,
        repair_runtime=repair_runtime,
    )


def configure_supervisor_logging(
    parsed: argparse.Namespace,
    *,
    log_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
) -> None:
    """Configure standard supervisor logging from parsed supervisor args."""

    level_name = str(getattr(parsed, "log_level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=log_format)


def build_portal_implementation_supervisor_from_args(
    parsed: argparse.Namespace,
    *,
    repo_root: Path,
    daemon_script_path: Path | None = None,
    worktree_submodule_paths: Sequence[str] | None = None,
) -> tuple[object, ImplementationSupervisorRunContext]:
    """Build a ``PortalImplementationSupervisor`` and context from parsed args."""

    from .todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
        supervisor_config_from_args,
    )

    config = supervisor_config_from_args(
        parsed,
        repo_root=repo_root,
        daemon_script_path=daemon_script_path,
        worktree_submodule_paths=worktree_submodule_paths,
    )
    from .implementation_daemon_runner import implementation_state_artifact_paths

    state_paths = implementation_state_artifact_paths(
        parsed.state_dir,
        parsed.state_prefix,
        supervisor_events=True,
    )
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=config,
        state_path=config.state_path,
        strategy_path=config.strategy_path,
        events_path=config.events_path,
        daemon_events_path=state_paths["daemon_events_path"],
    )
    return PortalImplementationSupervisor(config), context


def _run_hooks(
    hooks: Sequence[SupervisorRunHook],
    *,
    phase: str,
    context: ImplementationSupervisorRunContext,
    logger: logging.Logger,
) -> None:
    for hook in hooks:
        if hook.phase != phase:
            continue
        result = hook.callback(context)
        if result:
            logger.log(hook.log_level, hook.message, result)


def run_portal_implementation_supervisor(
    supervisor: object,
    context: ImplementationSupervisorRunContext,
    *,
    logger: logging.Logger,
    hooks: Sequence[SupervisorRunHook] = (),
    once_complete_message: str = "Portal implementation supervisor check complete: %s",
    ensure_running: bool = False,
    ensure_running_callback: SupervisorRunHookCallback | None = None,
    ensure_running_message: str = "Supervisor ensure complete: %s",
    repair_runtime_callback: SupervisorRunHookCallback | None = None,
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
) -> Any:
    """Run a configured supervisor with optional local hooks and runtime repair."""

    if bool(getattr(context.parsed, "once", False)):
        _run_hooks(hooks, phase="before", context=context, logger=logger)
    elif hooks:
        logger.debug(
            "Skipping supervisor before hooks for long-running startup; "
            "managed watchdog maintenance will run refill hooks after daemon launch."
        )
    if ensure_running:
        if ensure_running_callback is None:
            return None
        result = ensure_running_callback(context)
        logger.info(ensure_running_message, result)
        return result

    if repair_runtime_callback is not None:
        repairs = repair_runtime_callback(context)
        if isinstance(repairs, dict) and (repairs.get("removed") or repairs.get("updated_status")):
            logger.info(repair_runtime_message, repairs)

    if context.parsed.once:
        result = supervisor.run_once()
        _run_hooks(hooks, phase="after_once", context=context, logger=logger)
        logger.info(once_complete_message, result)
        return result
    return supervisor.run_forever()


def run_configured_portal_implementation_supervisor(
    argv: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
    daemon_script_path: Path | None = None,
    worktree_submodule_paths: Sequence[str] | None = None,
    hooks: Sequence[SupervisorRunHook] = (),
    once_complete_message: str = "Portal implementation supervisor check complete: %s",
    ensure_running: bool = False,
    ensure_running_callback: SupervisorRunHookCallback | None = None,
    ensure_running_message: str = "Supervisor ensure complete: %s",
    repair_runtime_callback: SupervisorRunHookCallback | None = None,
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
) -> Any:
    """Parse, build, and run a configured portal implementation supervisor."""

    from .todo_daemon.implementation_supervisor import parse_args

    parsed = parse_args(list(argv))
    configure_supervisor_logging(parsed)
    supervisor, context = build_portal_implementation_supervisor_from_args(
        parsed,
        repo_root=repo_root,
        daemon_script_path=getattr(parsed, "daemon_script_path", None) or daemon_script_path,
        worktree_submodule_paths=getattr(parsed, "worktree_submodule_path", None) or worktree_submodule_paths,
    )
    return run_portal_implementation_supervisor(
        supervisor,
        context,
        logger=logger,
        hooks=hooks,
        once_complete_message=once_complete_message,
        ensure_running=ensure_running,
        ensure_running_callback=ensure_running_callback,
        ensure_running_message=ensure_running_message,
        repair_runtime_callback=repair_runtime_callback,
        repair_runtime_message=repair_runtime_message,
    )


def run_configured_portal_implementation_supervisor_with_runtime(
    argv: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
    script_path: Path,
    process_match_any: Sequence[str] = (),
    process_predicate: Callable[[int], bool] | None = None,
    prepare_environment: Callable[[], None] | None = None,
    implementation_lock_name: str = "implementation.lock",
    startup_delay_seconds: float = 1.0,
    daemon_script_path: Path | None = None,
    worktree_submodule_paths: Sequence[str] | None = None,
    hooks: Sequence[SupervisorRunHook] = (),
    once_complete_message: str = "Portal implementation supervisor check complete: %s",
    ensure_running: bool = False,
    ensure_running_message: str = "Supervisor ensure complete: %s",
    repair_runtime: bool = True,
    repair_runtime_message: str = "Repaired stale supervisor runtime markers: %s",
) -> Any:
    """Run a configured supervisor with standard runtime repair/ensure wiring."""

    runtime_callbacks = build_supervisor_runtime_callbacks(
        argv,
        repo_root=repo_root,
        script_path=script_path,
        process_match_any=process_match_any,
        process_predicate=process_predicate,
        prepare_environment=prepare_environment,
        implementation_lock_name=implementation_lock_name,
        startup_delay_seconds=startup_delay_seconds,
    )
    return run_configured_portal_implementation_supervisor(
        argv,
        repo_root=repo_root,
        logger=logger,
        daemon_script_path=daemon_script_path,
        worktree_submodule_paths=worktree_submodule_paths,
        hooks=hooks,
        once_complete_message=once_complete_message,
        ensure_running=ensure_running,
        ensure_running_callback=runtime_callbacks.ensure_running,
        ensure_running_message=ensure_running_message,
        repair_runtime_callback=runtime_callbacks.repair_runtime if repair_runtime else None,
        repair_runtime_message=repair_runtime_message,
    )
