"""Reusable runner helpers for configured implementation daemons."""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..objectives.scan_receipts import RefillScanResult
from ..core.wrapper_utils import (
    AgentSupervisorNamespacePaths,
    with_default,
    with_repeated_default,
)
from ..runtime.event_log import append_jsonl_event


DAEMON_HOOK_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_HOOK_TIMEOUT_SECONDS"
DEFAULT_DAEMON_HOOK_TIMEOUT_SECONDS = 60.0
IDLE_DAEMON_PASS_LOG_INTERVAL_SECONDS = 300.0


def daemon_pass_is_idle(result: Mapping[str, Any]) -> bool:
    """Return whether a pass made no durable or implementation progress."""

    return (
        result.get("unchanged") is True
        and int(result.get("write_count", 0) or 0) == 0
        and not result.get("implementation_result")
        and not result.get("merge_reconciliation")
        and not result.get("completion_receipt_writes")
        and not result.get("retry_budget_resets")
    )


def compact_daemon_pass_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bounded operator heartbeat used for an idle daemon pass."""

    keys = (
        "task_count",
        "completed_count",
        "ready_count",
        "blocked_count",
        "active_task_id",
        "selection_idle_reason",
        "source_digest",
        "wake_kinds",
        "requirement_id",
    )
    return {key: result[key] for key in keys if key in result}


def log_daemon_pass_result(
    logger: logging.Logger,
    pass_complete_message: str,
    result: Mapping[str, Any],
    *,
    emit_idle_info: bool,
) -> None:
    """Log changed passes in full and throttle bounded summaries for idle passes."""

    if not daemon_pass_is_idle(result):
        logger.info(pass_complete_message, result)
        return
    logger.debug("Full idle daemon pass result: %s", result)
    if emit_idle_info:
        logger.info(pass_complete_message, compact_daemon_pass_result(result))


def bounded_daemon_wait_timeout(
    result: Mapping[str, Any],
    *,
    default_timeout: float,
) -> float:
    """Return the configured wait bounded by a daemon's durable retry schedule."""

    timeout = max(0.0, float(default_timeout))
    retry_after = result.get("next_wake_after_seconds")
    if isinstance(retry_after, bool) or not isinstance(retry_after, (int, float)):
        return timeout
    retry_after = float(retry_after)
    if not math.isfinite(retry_after):
        return timeout
    return min(timeout, max(0.0, retry_after))


class DaemonHookTimeoutError(TimeoutError):
    """Raised when a daemon before/after hook exceeds its bounded runtime."""


@dataclass(frozen=True)
class ImplementationDaemonRunContext:
    """Runtime paths and parsed arguments shared with daemon loop hooks."""

    parsed: argparse.Namespace
    state_path: Path
    strategy_path: Path
    events_path: Path
    pass_index: int = 0

    def for_pass(self, pass_index: int) -> "ImplementationDaemonRunContext":
        return ImplementationDaemonRunContext(
            parsed=self.parsed,
            state_path=self.state_path,
            strategy_path=self.strategy_path,
            events_path=self.events_path,
            pass_index=pass_index,
        )


DaemonLoopHookCallback = Callable[[ImplementationDaemonRunContext], Any]
DaemonRefillRecordCallback = Callable[..., Any]
DaemonBootstrapPathCallback = Callable[[Mapping[str, Path | str]], Any]
DaemonBootstrapHookFactory = Callable[[Mapping[str, Path | str]], Sequence["DaemonLoopHook"]]
DaemonBootstrapExtraKwargsFactory = Callable[[Mapping[str, Path | str]], Mapping[str, Any] | None]
DaemonMergeResolverCommand = str | Callable[[], str]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class DaemonLoopHook:
    """One before/after hook for a configured implementation daemon loop."""

    phase: str
    message: str
    callback: DaemonLoopHookCallback
    log_level: int = logging.WARNING


RefillHookEntry = tuple[str, DaemonLoopHookCallback]


@dataclass(frozen=True)
class ConfiguredImplementationDaemonRunner:
    """Project-bound runner wiring for a configured implementation daemon."""

    repo_root: Path
    logger: logging.Logger
    default_worktree_submodule_paths: Sequence[str] | None = None
    default_implementation_protected_paths: Sequence[str] | None = None
    default_objective_path: Path | None = None
    default_objective_bundle_dir: Path | None = None
    pass_complete_message: str = "Portal implementation daemon pass complete: %s"

    def run_configured(
        self,
        argv: Sequence[str],
        *,
        hooks: Sequence[DaemonLoopHook] = (),
        pass_complete_message: str | None = None,
    ) -> Any:
        """Run a configured implementation daemon using this project binding."""

        return run_configured_portal_implementation_daemon(
            argv,
            repo_root=self.repo_root,
            logger=self.logger,
            default_worktree_submodule_paths=self.default_worktree_submodule_paths,
            default_implementation_protected_paths=self.default_implementation_protected_paths,
            default_objective_path=self.default_objective_path,
            default_objective_bundle_dir=self.default_objective_bundle_dir,
            hooks=hooks,
            pass_complete_message=pass_complete_message or self.pass_complete_message,
        )

    def run_configured_from_paths(
        self,
        argv: Sequence[str],
        paths: Mapping[str, Path | str],
        *,
        task_prefix: str,
        state_prefix: str,
        todo_path_key: str = "todo_path",
        state_dir_key: str = "state_dir",
        worktree_root_key: str = "worktree_root",
        todo_path_flag: str = "--todo-path",
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        objective_bundle_dir_key: str | None = None,
        objective_bundle_dir: Path | str | None = None,
        llm_merge_resolver_command: str = "",
        worktree_submodule_paths: Sequence[str] = (),
        hooks: Sequence[DaemonLoopHook] = (),
        pass_complete_message: str | None = None,
    ) -> Any:
        """Apply path-derived defaults and run this configured daemon."""

        args = apply_portal_implementation_daemon_defaults_from_paths(
            argv,
            paths,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            objective_bundle_dir_key=objective_bundle_dir_key,
            objective_bundle_dir=objective_bundle_dir,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
        )
        return self.run_configured(
            args,
            hooks=hooks,
            pass_complete_message=pass_complete_message,
        )

    def run_configured_from_bootstrap(
        self,
        argv: Sequence[str],
        *,
        ensure_paths: Callable[[], Mapping[str, Path | str]],
        task_prefix: str,
        state_prefix: str,
        enter_runtime_environment: Callable[[], Any] | None = None,
        enter_runtime_before_paths: bool = False,
        path_callbacks: Sequence[DaemonBootstrapPathCallback] = (),
        hooks_factory: DaemonBootstrapHookFactory | None = None,
        hooks: Sequence[DaemonLoopHook] = (),
        todo_path_key: str = "todo_path",
        state_dir_key: str = "state_dir",
        worktree_root_key: str = "worktree_root",
        todo_path_flag: str = "--todo-path",
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        objective_bundle_dir_key: str | None = None,
        objective_bundle_dir: Path | str | None = None,
        llm_merge_resolver_command: DaemonMergeResolverCommand = "",
        worktree_submodule_paths: Sequence[str] = (),
        pass_complete_message: str | None = None,
    ) -> Any:
        """Resolve bootstrap paths, run project callbacks, and start the daemon."""

        if enter_runtime_environment is not None and enter_runtime_before_paths:
            enter_runtime_environment()
        paths = ensure_paths()
        if enter_runtime_environment is not None and not enter_runtime_before_paths:
            enter_runtime_environment()
        for callback in path_callbacks:
            callback(paths)
        effective_hooks = hooks_factory(paths) if hooks_factory is not None else hooks
        return self.run_configured_from_paths(
            argv,
            paths,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            objective_bundle_dir_key=objective_bundle_dir_key,
            objective_bundle_dir=objective_bundle_dir,
            llm_merge_resolver_command=_resolved_daemon_merge_resolver_command(llm_merge_resolver_command),
            worktree_submodule_paths=worktree_submodule_paths,
            hooks=effective_hooks,
            pass_complete_message=pass_complete_message,
        )

    def run_namespace_configured_from_bootstrap(
        self,
        argv: Sequence[str],
        *,
        ensure_paths: Callable[[], Mapping[str, Path | str]],
        namespace_paths: AgentSupervisorNamespacePaths,
        task_prefix: str,
        state_prefix: str,
        use_bootstrap_keys: bool = False,
        enter_runtime_environment: Callable[[], Any] | None = None,
        enter_runtime_before_paths: bool = False,
        path_callbacks: Sequence[DaemonBootstrapPathCallback] = (),
        hooks_factory: DaemonBootstrapHookFactory | None = None,
        hooks: Sequence[DaemonLoopHook] = (),
        todo_path_key: str = "todo_path",
        state_dir_key: str = "state_dir",
        worktree_root_key: str = "worktree_root",
        todo_path_flag: str = "--todo-path",
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        objective_bundle_dir_key: str | None = None,
        objective_bundle_dir: Path | str | None = None,
        llm_merge_resolver_command: DaemonMergeResolverCommand = "",
        worktree_submodule_paths: Sequence[str] = (),
        pass_complete_message: str | None = None,
    ) -> Any:
        """Run a configured daemon using standard namespace path defaults."""

        resolved_objective_bundle_dir_key = objective_bundle_dir_key
        resolved_objective_bundle_dir = objective_bundle_dir
        if resolved_objective_bundle_dir_key is None and resolved_objective_bundle_dir is None:
            if use_bootstrap_keys:
                resolved_objective_bundle_dir_key = "objective_bundle_dir"
            else:
                resolved_objective_bundle_dir = namespace_paths.objective_bundle_dir

        return self.run_configured_from_bootstrap(
            argv,
            ensure_paths=ensure_paths,
            enter_runtime_environment=enter_runtime_environment,
            enter_runtime_before_paths=enter_runtime_before_paths,
            path_callbacks=path_callbacks,
            hooks_factory=hooks_factory,
            hooks=hooks,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            objective_bundle_dir_key=resolved_objective_bundle_dir_key,
            objective_bundle_dir=resolved_objective_bundle_dir,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
            pass_complete_message=pass_complete_message,
        )


@dataclass(frozen=True)
class ConfiguredDaemonBootstrapRunner:
    """Reusable bootstrap/run wiring for a project implementation daemon wrapper."""

    runner: ConfiguredImplementationDaemonRunner
    ensure_paths: Callable[[], Mapping[str, Path | str]]
    task_prefix: str
    state_prefix: str
    namespace_paths: AgentSupervisorNamespacePaths | None = None
    use_bootstrap_keys: bool = False
    enter_runtime_environment: Callable[[], Any] | None = None
    enter_runtime_before_paths: bool = False
    path_callbacks: Sequence[DaemonBootstrapPathCallback] = ()
    hooks_factory: DaemonBootstrapHookFactory | None = None
    hooks: Sequence[DaemonLoopHook] = ()
    todo_path_key: str = "todo_path"
    state_dir_key: str = "state_dir"
    worktree_root_key: str = "worktree_root"
    todo_path_flag: str = "--todo-path"
    objective_path_key: str | None = None
    objective_path: Path | str | None = None
    objective_bundle_dir_key: str | None = None
    objective_bundle_dir: Path | str | None = None
    llm_merge_resolver_command: DaemonMergeResolverCommand = ""
    worktree_submodule_paths: Sequence[str] = ()
    pass_complete_message: str | None = None

    def run(self, argv: Sequence[str] | None = None) -> Any:
        """Run the configured implementation daemon from bootstrap paths."""

        args = list(sys.argv[1:] if argv is None else argv)
        kwargs: dict[str, Any] = {
            "ensure_paths": self.ensure_paths,
            "enter_runtime_environment": self.enter_runtime_environment,
            "enter_runtime_before_paths": self.enter_runtime_before_paths,
            "path_callbacks": self.path_callbacks,
            "hooks_factory": self.hooks_factory,
            "hooks": self.hooks,
            "todo_path_key": self.todo_path_key,
            "state_dir_key": self.state_dir_key,
            "worktree_root_key": self.worktree_root_key,
            "todo_path_flag": self.todo_path_flag,
            "task_prefix": self.task_prefix,
            "state_prefix": self.state_prefix,
            "objective_path_key": self.objective_path_key,
            "objective_path": self.objective_path,
            "objective_bundle_dir_key": self.objective_bundle_dir_key,
            "objective_bundle_dir": self.objective_bundle_dir,
            "llm_merge_resolver_command": self.llm_merge_resolver_command,
            "worktree_submodule_paths": self.worktree_submodule_paths,
            "pass_complete_message": self.pass_complete_message,
        }
        if self.namespace_paths is not None:
            kwargs["namespace_paths"] = self.namespace_paths
            kwargs["use_bootstrap_keys"] = self.use_bootstrap_keys
            return self.runner.run_namespace_configured_from_bootstrap(args, **kwargs)
        return self.runner.run_configured_from_bootstrap(args, **kwargs)


def build_configured_daemon_bootstrap_runner(
    *,
    runner: ConfiguredImplementationDaemonRunner,
    ensure_paths: Callable[[], Mapping[str, Path | str]],
    task_prefix: str,
    state_prefix: str,
    namespace_paths: AgentSupervisorNamespacePaths | None = None,
    use_bootstrap_keys: bool = False,
    enter_runtime_environment: Callable[[], Any] | None = None,
    enter_runtime_before_paths: bool = False,
    path_callbacks: Sequence[DaemonBootstrapPathCallback] = (),
    hooks_factory: DaemonBootstrapHookFactory | None = None,
    hooks: Sequence[DaemonLoopHook] = (),
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    llm_merge_resolver_command: DaemonMergeResolverCommand = "",
    worktree_submodule_paths: Sequence[str] = (),
    pass_complete_message: str | None = None,
) -> ConfiguredDaemonBootstrapRunner:
    """Build reusable daemon bootstrap/run wiring for a project wrapper."""

    return ConfiguredDaemonBootstrapRunner(
        runner=runner,
        ensure_paths=ensure_paths,
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        namespace_paths=namespace_paths,
        use_bootstrap_keys=use_bootstrap_keys,
        enter_runtime_environment=enter_runtime_environment,
        enter_runtime_before_paths=enter_runtime_before_paths,
        path_callbacks=tuple(path_callbacks),
        hooks_factory=hooks_factory,
        hooks=tuple(hooks),
        todo_path_key=todo_path_key,
        state_dir_key=state_dir_key,
        worktree_root_key=worktree_root_key,
        todo_path_flag=todo_path_flag,
        objective_path_key=objective_path_key,
        objective_path=objective_path,
        objective_bundle_dir_key=objective_bundle_dir_key,
        objective_bundle_dir=objective_bundle_dir,
        llm_merge_resolver_command=llm_merge_resolver_command,
        worktree_submodule_paths=tuple(worktree_submodule_paths),
        pass_complete_message=pass_complete_message,
    )


def build_configured_implementation_daemon_runner(
    *,
    repo_root: Path | str,
    logger: logging.Logger,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_implementation_protected_paths: Sequence[str] | None = None,
    default_objective_path: Path | str | None = None,
    default_objective_bundle_dir: Path | str | None = None,
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> ConfiguredImplementationDaemonRunner:
    """Build reusable daemon runner wiring bound to a project repository."""

    return ConfiguredImplementationDaemonRunner(
        repo_root=Path(repo_root),
        logger=logger,
        default_worktree_submodule_paths=(
            tuple(default_worktree_submodule_paths)
            if default_worktree_submodule_paths is not None
            else None
        ),
        default_implementation_protected_paths=(
            tuple(default_implementation_protected_paths)
            if default_implementation_protected_paths is not None
            else None
        ),
        default_objective_path=(
            Path(default_objective_path)
            if default_objective_path is not None
            else None
        ),
        default_objective_bundle_dir=(
            Path(default_objective_bundle_dir)
            if default_objective_bundle_dir is not None
            else None
        ),
        pass_complete_message=pass_complete_message,
    )


def build_namespace_configured_implementation_daemon_runner(
    *,
    repo_root: Path | str,
    logger: logging.Logger,
    namespace_paths: AgentSupervisorNamespacePaths,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_implementation_protected_paths: Sequence[str] | None = None,
    default_objective_path: Path | str | None = None,
    default_objective_bundle_dir: Path | str | None = None,
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> ConfiguredImplementationDaemonRunner:
    """Build a configured daemon runner using conventional namespace defaults."""

    return build_configured_implementation_daemon_runner(
        repo_root=repo_root,
        logger=logger,
        default_worktree_submodule_paths=default_worktree_submodule_paths,
        default_implementation_protected_paths=default_implementation_protected_paths,
        default_objective_path=default_objective_path,
        default_objective_bundle_dir=(
            default_objective_bundle_dir
            if default_objective_bundle_dir is not None
            else namespace_paths.objective_bundle_dir
        ),
        pass_complete_message=pass_complete_message,
    )


def build_namespace_daemon_bootstrap_runner(
    *,
    repo_root: Path | str,
    logger: logging.Logger,
    namespace_paths: AgentSupervisorNamespacePaths,
    ensure_paths: Callable[[], Mapping[str, Path | str]],
    task_prefix: str,
    state_prefix: str,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_implementation_protected_paths: Sequence[str] | None = None,
    default_objective_path: Path | str | None = None,
    default_objective_bundle_dir: Path | str | None = None,
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
    use_bootstrap_keys: bool = False,
    enter_runtime_environment: Callable[[], Any] | None = None,
    enter_runtime_before_paths: bool = False,
    path_callbacks: Sequence[DaemonBootstrapPathCallback] = (),
    hooks_factory: DaemonBootstrapHookFactory | None = None,
    hooks: Sequence[DaemonLoopHook] = (),
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    llm_merge_resolver_command: DaemonMergeResolverCommand = "",
    worktree_submodule_paths: Sequence[str] | None = None,
    run_pass_complete_message: str | None = None,
) -> ConfiguredDaemonBootstrapRunner:
    """Build a namespace-scoped daemon bootstrap runner with reusable defaults."""

    runner = build_namespace_configured_implementation_daemon_runner(
        repo_root=repo_root,
        logger=logger,
        namespace_paths=namespace_paths,
        default_worktree_submodule_paths=default_worktree_submodule_paths,
        default_implementation_protected_paths=default_implementation_protected_paths,
        default_objective_path=default_objective_path,
        default_objective_bundle_dir=default_objective_bundle_dir,
        pass_complete_message=pass_complete_message,
    )
    effective_worktree_submodule_paths = (
        tuple(worktree_submodule_paths)
        if worktree_submodule_paths is not None
        else tuple(default_worktree_submodule_paths or ())
    )
    return build_configured_daemon_bootstrap_runner(
        runner=runner,
        ensure_paths=ensure_paths,
        namespace_paths=namespace_paths,
        use_bootstrap_keys=use_bootstrap_keys,
        enter_runtime_environment=enter_runtime_environment,
        enter_runtime_before_paths=enter_runtime_before_paths,
        path_callbacks=path_callbacks,
        hooks_factory=hooks_factory,
        hooks=hooks,
        todo_path_key=todo_path_key,
        state_dir_key=state_dir_key,
        worktree_root_key=worktree_root_key,
        todo_path_flag=todo_path_flag,
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        objective_path_key=objective_path_key,
        objective_path=objective_path,
        objective_bundle_dir_key=objective_bundle_dir_key,
        objective_bundle_dir=objective_bundle_dir,
        llm_merge_resolver_command=llm_merge_resolver_command,
        worktree_submodule_paths=effective_worktree_submodule_paths,
        pass_complete_message=run_pass_complete_message,
    )


def _with_extra_kwargs(
    kwargs: dict[str, Any],
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _resolved_daemon_merge_resolver_command(command: DaemonMergeResolverCommand) -> str:
    if callable(command):
        command = command()
    return str(command or "").strip()


def _extra_kwargs_from_factory(
    paths: Mapping[str, Path | str],
    *,
    values: Mapping[str, Any] | None = None,
    factory: DaemonBootstrapExtraKwargsFactory | None = None,
) -> dict[str, Any] | None:
    kwargs = dict(values or {})
    if factory is not None:
        kwargs.update(factory(paths) or {})
    return kwargs or None


@dataclass(frozen=True)
class ImplementationDaemonDefaults:
    """Default CLI values for a project-specific implementation daemon wrapper."""

    todo_path: Path
    state_dir: Path
    task_prefix: str
    state_prefix: str
    worktree_root: Path
    todo_path_flag: str = "--todo-path"
    objective_path: Path | None = None
    objective_bundle_dir: Path | None = None
    llm_merge_resolver_command: str = ""
    worktree_submodule_paths: Sequence[str] = ()


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


def build_implementation_daemon_defaults_from_paths(
    paths: Mapping[str, Path | str],
    *,
    task_prefix: str,
    state_prefix: str,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    llm_merge_resolver_command: str = "",
    worktree_submodule_paths: Sequence[str] = (),
) -> ImplementationDaemonDefaults:
    """Build reusable implementation-daemon defaults from resolved wrapper paths."""

    return ImplementationDaemonDefaults(
        todo_path=_path_from_mapping(paths, todo_path_key),
        state_dir=_path_from_mapping(paths, state_dir_key),
        task_prefix=task_prefix,
        state_prefix=state_prefix,
        worktree_root=_path_from_mapping(paths, worktree_root_key),
        todo_path_flag=todo_path_flag,
        objective_path=_optional_path_from_mapping(paths, key=objective_path_key, value=objective_path),
        objective_bundle_dir=_optional_path_from_mapping(
            paths,
            key=objective_bundle_dir_key,
            value=objective_bundle_dir,
        ),
        llm_merge_resolver_command=str(llm_merge_resolver_command or "").strip(),
        worktree_submodule_paths=worktree_submodule_paths,
    )


def _with_optional_default(args: Sequence[str], flag: str, value: object | None) -> list[str]:
    if value is None:
        return list(args)
    return with_default(args, flag, str(value))


def apply_portal_implementation_daemon_defaults(
    argv: Sequence[str],
    *,
    defaults: ImplementationDaemonDefaults,
) -> list[str]:
    """Apply reusable implementation-daemon CLI defaults to ``argv``."""

    args = list(argv)
    args = with_default(args, defaults.todo_path_flag, str(defaults.todo_path))
    args = with_default(args, "--state-dir", str(defaults.state_dir))
    args = with_default(args, "--task-prefix", defaults.task_prefix)
    args = with_default(args, "--state-prefix", defaults.state_prefix)
    args = with_default(args, "--worktree-root", str(defaults.worktree_root))
    args = _with_optional_default(args, "--objective-path", defaults.objective_path)
    args = _with_optional_default(args, "--objective-bundle-dir", defaults.objective_bundle_dir)
    if defaults.llm_merge_resolver_command:
        args = with_default(args, "--llm-merge-resolver-command", defaults.llm_merge_resolver_command)
    if defaults.worktree_submodule_paths:
        args = with_repeated_default(args, "--worktree-submodule-path", defaults.worktree_submodule_paths)
    return args


def apply_portal_implementation_daemon_defaults_from_paths(
    argv: Sequence[str],
    paths: Mapping[str, Path | str],
    *,
    task_prefix: str,
    state_prefix: str,
    todo_path_key: str = "todo_path",
    state_dir_key: str = "state_dir",
    worktree_root_key: str = "worktree_root",
    todo_path_flag: str = "--todo-path",
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    objective_bundle_dir_key: str | None = None,
    objective_bundle_dir: Path | str | None = None,
    llm_merge_resolver_command: str = "",
    worktree_submodule_paths: Sequence[str] = (),
) -> list[str]:
    """Apply implementation-daemon CLI defaults directly from resolved wrapper paths."""

    return apply_portal_implementation_daemon_defaults(
        argv,
        defaults=build_implementation_daemon_defaults_from_paths(
            paths,
            task_prefix=task_prefix,
            state_prefix=state_prefix,
            todo_path_key=todo_path_key,
            state_dir_key=state_dir_key,
            worktree_root_key=worktree_root_key,
            todo_path_flag=todo_path_flag,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            objective_bundle_dir_key=objective_bundle_dir_key,
            objective_bundle_dir=objective_bundle_dir,
            llm_merge_resolver_command=llm_merge_resolver_command,
            worktree_submodule_paths=worktree_submodule_paths,
        ),
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


def build_daemon_refill_hooks(
    entries: Sequence[RefillHookEntry],
    *,
    scope_label: str = "",
    before: bool = True,
    after: bool = True,
    after_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> tuple[DaemonLoopHook, ...]:
    """Build standard before/after refill hooks for a daemon wrapper."""

    hooks: list[DaemonLoopHook] = []
    if before:
        hooks.extend(
            DaemonLoopHook(
                "before",
                _refill_hook_message(
                    scope_label=scope_label,
                    finding_label=finding_label,
                    phase_label="before",
                    runner_label="daemon",
                ),
                callback,
                log_level=log_level,
            )
            for finding_label, callback in entries
        )
    if after:
        hooks.extend(
            DaemonLoopHook(
                "after",
                _refill_hook_message(
                    scope_label=scope_label,
                    finding_label=finding_label,
                    phase_label="after",
                    runner_label="daemon",
                ),
                callback,
                log_level=log_level,
            )
            for finding_label, callback in _ordered_refill_entries(entries, after_order)
        )
    return tuple(hooks)


def build_daemon_objective_refill_callback(
    callback: DaemonRefillRecordCallback,
    *,
    discovery_dir: Path,
    objective_path: Path | None = None,
    repo_root: Path | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> DaemonLoopHookCallback:
    """Build a daemon hook that records objective-refill findings."""

    def hook(ctx: ImplementationDaemonRunContext) -> Any:
        kwargs: dict[str, Any] = {
            "todo_path": ctx.parsed.todo_path,
            "state_path": ctx.state_path,
            "strategy_path": ctx.strategy_path,
            "discovery_dir": discovery_dir,
            "task_header_prefix": ctx.parsed.task_prefix,
        }
        resolved_objective_path = getattr(ctx.parsed, "objective_path", None) or objective_path
        if resolved_objective_path is not None:
            kwargs["objective_path"] = resolved_objective_path
        if repo_root is not None:
            kwargs["repo_root"] = repo_root
        bundle_dir = getattr(ctx.parsed, "objective_bundle_dir", None)
        if bundle_dir is not None:
            kwargs["bundle_dir"] = bundle_dir
        for attr, key in (
            ("objective_scan_min_open_tasks", "min_open_tasks"),
            ("objective_scan_max_findings", "max_findings"),
            ("objective_scan_cooldown_seconds", "cooldown_seconds"),
            ("objective_surplus_findings_per_goal", "surplus_findings_per_goal"),
            ("objective_surplus_min_terms_per_todo", "surplus_min_terms_per_todo"),
        ):
            value = getattr(ctx.parsed, attr, None)
            if value is not None:
                kwargs[key] = value
        return callback(**_with_extra_kwargs(kwargs, extra_kwargs))

    return hook


def build_daemon_codebase_scan_refill_callback(
    callback: DaemonRefillRecordCallback,
    *,
    discovery_dir: Path,
    objective_path: Path | None = None,
    repo_root: Path | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> DaemonLoopHookCallback:
    """Build a daemon hook that records codebase-scan findings."""

    def hook(ctx: ImplementationDaemonRunContext) -> Any:
        kwargs: dict[str, Any] = {
            "todo_path": ctx.parsed.todo_path,
            "state_path": ctx.state_path,
            "strategy_path": ctx.strategy_path,
            "discovery_dir": discovery_dir,
            "task_header_prefix": ctx.parsed.task_prefix,
        }
        if repo_root is not None:
            kwargs["repo_root"] = repo_root
        resolved_objective_path = getattr(ctx.parsed, "objective_path", None) or objective_path
        if resolved_objective_path is not None:
            kwargs["objective_path"] = resolved_objective_path
        bundle_dir = getattr(ctx.parsed, "objective_bundle_dir", None)
        if bundle_dir is not None:
            kwargs["bundle_dir"] = bundle_dir
        for attr, key in (
            ("codebase_scan_min_open_tasks", "min_open_tasks"),
            ("codebase_scan_max_findings", "max_findings"),
            ("codebase_scan_cooldown_seconds", "cooldown_seconds"),
        ):
            value = getattr(ctx.parsed, attr, None)
            if value is not None:
                kwargs[key] = value
        return callback(**_with_extra_kwargs(kwargs, extra_kwargs))

    return hook


def build_daemon_retry_budget_refill_callback(
    callback: DaemonRefillRecordCallback,
    *,
    discovery_dir: Path,
    extra_kwargs: dict[str, Any] | None = None,
) -> DaemonLoopHookCallback:
    """Build a daemon hook that records retry-budget findings."""

    def hook(ctx: ImplementationDaemonRunContext) -> Any:
        return callback(
            **_with_extra_kwargs(
                {
                    "todo_path": ctx.parsed.todo_path,
                    "events_path": ctx.events_path,
                    "strategy_path": ctx.strategy_path,
                    "discovery_dir": discovery_dir,
                    "task_header_prefix": ctx.parsed.task_prefix,
                },
                extra_kwargs,
            )
        )

    return hook


def build_daemon_refill_hooks_from_recorders(
    *,
    discovery_dir: Path,
    objective_recorder: DaemonRefillRecordCallback | None = None,
    codebase_scan_recorder: DaemonRefillRecordCallback | None = None,
    retry_budget_recorder: DaemonRefillRecordCallback | None = None,
    objective_path: Path | None = None,
    repo_root: Path | None = None,
    objective_extra_kwargs: dict[str, Any] | None = None,
    codebase_scan_extra_kwargs: dict[str, Any] | None = None,
    retry_budget_extra_kwargs: dict[str, Any] | None = None,
    scope_label: str = "",
    before: bool = True,
    after: bool = True,
    after_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> tuple[DaemonLoopHook, ...]:
    """Build standard daemon refill hooks from configured recorder callbacks."""

    entries: list[RefillHookEntry] = []
    if objective_recorder is not None:
        entries.append(
            (
                "objective-goal",
                build_daemon_objective_refill_callback(
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
                build_daemon_codebase_scan_refill_callback(
                    codebase_scan_recorder,
                    discovery_dir=discovery_dir,
                    objective_path=objective_path,
                    repo_root=repo_root,
                    extra_kwargs=codebase_scan_extra_kwargs,
                ),
            )
        )
    if retry_budget_recorder is not None:
        entries.append(
            (
                "retry-budget",
                build_daemon_retry_budget_refill_callback(
                    retry_budget_recorder,
                    discovery_dir=discovery_dir,
                    extra_kwargs=retry_budget_extra_kwargs,
                ),
            )
        )
    return build_daemon_refill_hooks(
        tuple(entries),
        scope_label=scope_label,
        before=before,
        after=after,
        after_order=after_order,
        log_level=log_level,
    )


def build_daemon_refill_hooks_factory_from_recorders(
    *,
    discovery_dir_key: str | None = None,
    discovery_dir: Path | str | None = None,
    objective_recorder: DaemonRefillRecordCallback | None = None,
    codebase_scan_recorder: DaemonRefillRecordCallback | None = None,
    retry_budget_recorder: DaemonRefillRecordCallback | None = None,
    objective_path_key: str | None = None,
    objective_path: Path | str | None = None,
    repo_root: Path | None = None,
    objective_extra_kwargs: Mapping[str, Any] | None = None,
    objective_extra_kwargs_factory: DaemonBootstrapExtraKwargsFactory | None = None,
    codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
    codebase_scan_extra_kwargs_factory: DaemonBootstrapExtraKwargsFactory | None = None,
    retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
    retry_budget_extra_kwargs_factory: DaemonBootstrapExtraKwargsFactory | None = None,
    scope_label: str = "",
    before: bool = True,
    after: bool = True,
    after_order: Sequence[str] | None = None,
    log_level: int = logging.WARNING,
) -> DaemonBootstrapHookFactory:
    """Build a reusable bootstrap factory for daemon refill hooks."""

    def factory(paths: Mapping[str, Path | str]) -> tuple[DaemonLoopHook, ...]:
        resolved_discovery_dir = _optional_path_from_mapping(
            paths,
            key=discovery_dir_key,
            value=discovery_dir,
        )
        if resolved_discovery_dir is None:
            raise ValueError("discovery_dir or discovery_dir_key is required")
        return build_daemon_refill_hooks_from_recorders(
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
            after=after,
            after_order=after_order,
            log_level=log_level,
        )

    return factory


def implementation_state_artifact_paths(
    state_dir: Path | str,
    state_prefix: str,
    *,
    supervisor_events: bool = False,
) -> dict[str, Path]:
    """Return standard task-state, strategy, and event-log artifact paths."""

    resolved_state_dir = Path(state_dir)
    resolved_state_prefix = str(state_prefix)
    paths = {
        "state_path": resolved_state_dir / f"{resolved_state_prefix}_task_state.json",
        "strategy_path": resolved_state_dir / f"{resolved_state_prefix}_strategy.json",
        "events_path": resolved_state_dir
        / f"{resolved_state_prefix}_{'supervisor_' if supervisor_events else ''}events.jsonl",
    }
    if supervisor_events:
        paths["daemon_events_path"] = resolved_state_dir / f"{resolved_state_prefix}_events.jsonl"
    return paths


def namespace_implementation_state_artifact_paths(
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    state_prefix: str | None = None,
    state_dir: Path | str | None = None,
    supervisor_events: bool = False,
) -> dict[str, Path]:
    """Return standard state artifacts for a supervisor namespace."""

    return implementation_state_artifact_paths(
        state_dir or namespace_paths.state_dir,
        state_prefix or namespace_paths.namespace,
        supervisor_events=supervisor_events,
    )


def implementation_state_paths(parsed: argparse.Namespace) -> dict[str, Path]:
    """Return standard task-state, strategy, and event-log paths for parsed daemon args."""

    return implementation_state_artifact_paths(
        Path(parsed.state_dir),
        str(parsed.state_prefix),
    )


def configure_daemon_logging(
    parsed: argparse.Namespace,
    *,
    log_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
) -> None:
    """Configure standard daemon logging from parsed implementation-daemon args."""

    level_name = str(getattr(parsed, "log_level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=log_format)


def apply_merge_resolver_environment(parsed: argparse.Namespace) -> None:
    """Apply parsed LLM merge-resolver settings to the shared daemon environment."""

    from .implementation_daemon import (
        LLM_MERGE_RESOLVER_COMMAND_ENV,
        LLM_MERGE_RESOLVER_TIMEOUT_ENV,
    )

    command = str(getattr(parsed, "llm_merge_resolver_command", "") or "").strip()
    if command:
        os.environ[LLM_MERGE_RESOLVER_COMMAND_ENV] = command
    timeout_seconds = getattr(parsed, "llm_merge_resolver_timeout_seconds", None)
    if timeout_seconds is not None:
        os.environ[LLM_MERGE_RESOLVER_TIMEOUT_ENV] = str(timeout_seconds)


def resolve_database_implementation_paths(
    parsed: argparse.Namespace,
) -> dict[str, Path | None]:
    """Resolve control-plane database paths for database-authoritative execution.

    JSON queue/status/events/PID projections are intentionally optional and may
    be absent under database authority.
    """

    database_path = getattr(parsed, "database_path", None)
    if database_path is not None:
        database_path = Path(database_path)
    todo_path = getattr(parsed, "todo_path", None)
    if database_path is None and todo_path is not None:
        candidate = Path(todo_path)
        if candidate.suffix.lower() in {".duckdb", ".ddb"}:
            database_path = candidate
    coordination_path = getattr(parsed, "coordination_path", None)
    if coordination_path is not None:
        coordination_path = Path(coordination_path)
    return {
        "database_path": database_path,
        "coordination_path": coordination_path,
    }


def build_portal_implementation_daemon_from_args(
    parsed: argparse.Namespace,
    *,
    repo_root: Path,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_implementation_protected_paths: Sequence[str] | None = None,
    default_objective_path: Path | None = None,
    default_objective_bundle_dir: Path | None = None,
) -> tuple[object, ImplementationDaemonRunContext]:
    """Build a portal or database implementation daemon from parsed CLI args."""

    from .implementation_daemon import (
        DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS,
        DatabaseImplementationDaemon,
        PortalImplementationDaemon,
        database_program_from_daemon_namespace,
        is_database_authority_mode,
    )

    apply_merge_resolver_environment(parsed)
    state_paths = implementation_state_paths(parsed)
    program = database_program_from_daemon_namespace(parsed)
    db_paths = resolve_database_implementation_paths(parsed)
    database_path = db_paths["database_path"]
    if database_path is None and program is not None and program.store_id:
        candidate = Path(program.store_id)
        if candidate.suffix.lower() in {".duckdb", ".ddb"}:
            database_path = candidate

    authority_mode = (
        program.authority_mode
        if program is not None
        else str(getattr(parsed, "authority_mode", "") or "")
    )
    task_source_kind = (
        program.task_source_kind
        if program is not None
        else str(getattr(parsed, "task_source_kind", "") or "")
    )
    if (
        database_path is not None
        and is_database_authority_mode(
            authority_mode=authority_mode,
            task_source_kind=task_source_kind,
        )
    ):
        # Database-authoritative cutover: JSON projections may be absent.
        optional_state = state_paths["state_path"]
        optional_strategy = state_paths["strategy_path"]
        optional_events = state_paths["events_path"]
        # Prefer absent projections when state_dir was not explicitly needed.
        use_projections = bool(getattr(parsed, "require_json_projections", False))
        daemon: object = DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=db_paths["coordination_path"],
            owner_session_id=str(getattr(parsed, "owner_session_id", "") or ""),
            authority_mode=authority_mode or "embedded",
            task_source_kind=task_source_kind or "duckdb",
            markdown_path=(
                parsed.todo_path
                if str(getattr(parsed, "todo_path", "") or "").endswith(".md")
                else None
            ),
            state_path=optional_state if use_projections else None,
            strategy_path=optional_strategy if use_projections else None,
            events_path=optional_events if use_projections else None,
            pid_path=None,
            queue_path=None,
        )
        return daemon, ImplementationDaemonRunContext(
            parsed=parsed,
            state_path=optional_state,
            strategy_path=optional_strategy,
            events_path=optional_events,
        )

    worktree_submodule_paths = (
        getattr(parsed, "worktree_submodule_path", None)
        or default_worktree_submodule_paths
        or None
    )
    implementation_protected_paths = (
        getattr(parsed, "implementation_protected_path", None)
        or default_implementation_protected_paths
        or None
    )
    daemon = PortalImplementationDaemon(
        todo_path=parsed.todo_path,
        task_source=(
            parsed.todo_path
            if str(getattr(parsed, "task_source_kind", "") or "")
            in {"markdown", "duckdb"}
            else None
        ),
        task_source_kind=(
            str(getattr(parsed, "task_source_kind", "") or "")
            if str(getattr(parsed, "task_source_kind", "") or "")
            in {"markdown", "duckdb"}
            else ""
        ),
        state_path=state_paths["state_path"],
        strategy_path=state_paths["strategy_path"],
        events_path=state_paths["events_path"],
        repo_root=repo_root,
        task_header_prefix=parsed.task_prefix,
        implement=parsed.implement,
        implementation_command=parsed.implementation_command or None,
        implementation_timeout=parsed.implementation_timeout or DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS,
        use_ephemeral_worktree=parsed.implement and not parsed.no_ephemeral_worktree,
        worktree_root=parsed.worktree_root,
        merge_target_branch=getattr(parsed, "merge_target_branch", "") or None,
        merge_queue_dir=getattr(parsed, "merge_queue_dir", None),
        worktree_submodule_paths=worktree_submodule_paths,
        implementation_protected_paths=implementation_protected_paths,
        manual_completion_authority_task_ids=getattr(
            parsed,
            "manual_completion_authority_task_id",
            (),
        ),
        manual_completion_authority_required_task_ids=getattr(
            parsed,
            "manual_completion_authority_required_task_id",
            (),
        ),
        manual_completion_authority_epoch_id=getattr(
            parsed,
            "manual_completion_authority_epoch_id",
            "",
        ),
        manual_completion_authority_revalidation_only=bool(
            getattr(
                parsed,
                "manual_completion_authority_revalidation_only",
                False,
            )
        ),
        objective_path=parsed.objective_path or default_objective_path,
        objective_bundle_dir=parsed.objective_bundle_dir or default_objective_bundle_dir,
        execution_slice_task_ids=getattr(parsed, "execution_slice_task_id", ()),
        execution_slice_task_cids=getattr(parsed, "execution_slice_task_cid", ()),
        llm_merge_resolver_command=parsed.llm_merge_resolver_command or None,
        llm_merge_resolver_timeout_seconds=parsed.llm_merge_resolver_timeout_seconds,
        merge_reconciliation_max_merges=parsed.merge_reconciliation_max_merges,
        merged_worktree_cleanup_max=parsed.merged_worktree_cleanup_max,
        task_shard_count=parsed.task_shard_count,
        task_shard_index=parsed.task_shard_index,
        strict_task_sharding=bool(getattr(parsed, "strict_task_sharding", False)),
        maintenance_interval_seconds=getattr(parsed, "maintenance_interval_seconds", None),
    )
    return daemon, ImplementationDaemonRunContext(parsed=parsed, **state_paths)


def build_database_implementation_daemon_from_args(
    parsed: argparse.Namespace,
    *,
    database_path: Path | str | None = None,
    owner_session_id: str = "",
    provider_fn: Callable[..., Any] | None = None,
    effect_fn: Callable[..., Any] | None = None,
    validation_fn: Callable[..., Any] | None = None,
) -> object:
    """Build a DatabaseImplementationDaemon@1 from CLI/env authority bindings."""

    from .implementation_daemon import (
        DatabaseImplementationDaemon,
        database_program_from_daemon_namespace,
    )

    program = database_program_from_daemon_namespace(parsed)
    db_paths = resolve_database_implementation_paths(parsed)
    resolved_db = Path(database_path) if database_path is not None else db_paths["database_path"]
    if resolved_db is None:
        raise ValueError(
            "database_path is required for DatabaseImplementationDaemon "
            "(pass --database-path or a .duckdb --todo-path)"
        )
    authority_mode = (
        program.authority_mode if program is not None else "embedded"
    )
    task_source_kind = (
        program.task_source_kind if program is not None else "duckdb"
    )
    return DatabaseImplementationDaemon(
        database_path=resolved_db,
        coordination_path=db_paths["coordination_path"],
        owner_session_id=owner_session_id
        or str(getattr(parsed, "owner_session_id", "") or ""),
        authority_mode=authority_mode,
        task_source_kind=task_source_kind,
        provider_fn=provider_fn,
        effect_fn=effect_fn,
        validation_fn=validation_fn,
        state_path=None,
        strategy_path=None,
        events_path=None,
        pid_path=None,
        queue_path=None,
    )


def _run_hooks(
    hooks: Sequence[DaemonLoopHook],
    *,
    phase: str,
    context: ImplementationDaemonRunContext,
    logger: logging.Logger,
) -> None:
    timeout_seconds = getattr(context.parsed, "daemon_hook_timeout_seconds", None)
    if timeout_seconds is None:
        timeout_seconds = _env_float(DAEMON_HOOK_TIMEOUT_ENV, DEFAULT_DAEMON_HOOK_TIMEOUT_SECONDS)
    for hook in hooks:
        if hook.phase != phase:
            continue
        try:
            result = _run_hook_callback_with_timeout(
                hook.callback,
                context,
                timeout_seconds=float(timeout_seconds or 0.0),
            )
        except DaemonHookTimeoutError as exc:
            payload = {
                "phase": hook.phase,
                "message": hook.message,
                "timeout_seconds": float(timeout_seconds or 0.0),
                "error": str(exc),
            }
            append_jsonl_event(context.events_path, "daemon_hook_timeout", payload)
            logger.warning("Daemon hook timed out: %s", payload)
            continue
        should_log = (
            result.generated_count > 0
            if isinstance(result, RefillScanResult)
            else bool(result)
        )
        if should_log:
            logger.log(hook.log_level, hook.message, result)


def _run_hook_callback_with_timeout(
    callback: DaemonLoopHookCallback,
    context: ImplementationDaemonRunContext,
    *,
    timeout_seconds: float,
) -> Any:
    if timeout_seconds <= 0.0:
        return callback(context)

    def _handle_timeout(_signum, _frame):
        raise DaemonHookTimeoutError(f"daemon hook exceeded {timeout_seconds:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        return callback(context)
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def run_portal_implementation_daemon_loop(
    daemon: object,
    context: ImplementationDaemonRunContext,
    *,
    logger: logging.Logger,
    hooks: Sequence[DaemonLoopHook] = (),
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> None:
    """Run a configured daemon with optional before/after pass hooks."""

    parsed = context.parsed
    authority_revalidation_only = bool(
        getattr(
            parsed,
            "manual_completion_authority_revalidation_only",
            False,
        )
    )
    effective_hooks = () if authority_revalidation_only else hooks
    pass_index = 0
    last_idle_info_at: float | None = None
    try:
        while True:
            pass_context = context.for_pass(pass_index)
            _run_hooks(
                effective_hooks,
                phase="before",
                context=pass_context,
                logger=logger,
            )
            result = daemon.run_once()
            _run_hooks(
                effective_hooks,
                phase="after",
                context=pass_context,
                logger=logger,
            )
            now = time.monotonic()
            emit_idle_info = (
                bool(parsed.once)
                or last_idle_info_at is None
                or now - last_idle_info_at >= IDLE_DAEMON_PASS_LOG_INTERVAL_SECONDS
            )
            log_daemon_pass_result(
                logger,
                pass_complete_message,
                result,
                emit_idle_info=emit_idle_info,
            )
            if daemon_pass_is_idle(result) and emit_idle_info:
                last_idle_info_at = now
            if parsed.once:
                break
            pass_index += 1
            wait_for_wake = getattr(daemon, "wait_for_wake", None)
            if callable(wait_for_wake):
                wait_for_wake(
                    timeout=bounded_daemon_wait_timeout(
                        result,
                        default_timeout=parsed.interval,
                    )
                )
            else:
                # Preserve compatibility with daemon implementations which
                # have not adopted the event-driven wake contract.
                time.sleep(
                    bounded_daemon_wait_timeout(
                        result,
                        default_timeout=parsed.interval,
                    )
                )
    finally:
        close_event_runtime = getattr(daemon, "close_event_runtime", None)
        if callable(close_event_runtime):
            close_event_runtime()


def run_configured_portal_implementation_daemon(
    argv: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_implementation_protected_paths: Sequence[str] | None = None,
    default_objective_path: Path | None = None,
    default_objective_bundle_dir: Path | None = None,
    hooks: Sequence[DaemonLoopHook] = (),
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> None:
    """Parse, build, and run a configured portal implementation daemon."""

    from .implementation_daemon import parse_args

    parsed = parse_args(list(argv))
    configure_daemon_logging(parsed)
    daemon, context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=repo_root,
        default_worktree_submodule_paths=default_worktree_submodule_paths,
        default_implementation_protected_paths=default_implementation_protected_paths,
        default_objective_path=default_objective_path,
        default_objective_bundle_dir=default_objective_bundle_dir,
    )
    run_portal_implementation_daemon_loop(
        daemon,
        context,
        logger=logger,
        hooks=hooks,
        pass_complete_message=pass_complete_message,
    )
