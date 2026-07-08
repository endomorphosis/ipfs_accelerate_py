"""Reusable runner helpers for configured implementation daemons."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .wrapper_utils import (
    AgentSupervisorNamespacePaths,
    with_default,
    with_repeated_default,
)
from .event_log import append_jsonl_event


DAEMON_HOOK_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_HOOK_TIMEOUT_SECONDS"
DEFAULT_DAEMON_HOOK_TIMEOUT_SECONDS = 60.0


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
    default_objective_path: Path | str | None = None,
    default_objective_bundle_dir: Path | str | None = None,
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> ConfiguredImplementationDaemonRunner:
    """Build a configured daemon runner using conventional namespace defaults."""

    return build_configured_implementation_daemon_runner(
        repo_root=repo_root,
        logger=logger,
        default_worktree_submodule_paths=default_worktree_submodule_paths,
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

    from .todo_daemon.implementation_daemon import (
        LLM_MERGE_RESOLVER_COMMAND_ENV,
        LLM_MERGE_RESOLVER_TIMEOUT_ENV,
    )

    command = str(getattr(parsed, "llm_merge_resolver_command", "") or "").strip()
    if command:
        os.environ[LLM_MERGE_RESOLVER_COMMAND_ENV] = command
    timeout_seconds = getattr(parsed, "llm_merge_resolver_timeout_seconds", None)
    if timeout_seconds is not None:
        os.environ[LLM_MERGE_RESOLVER_TIMEOUT_ENV] = str(timeout_seconds)


def build_portal_implementation_daemon_from_args(
    parsed: argparse.Namespace,
    *,
    repo_root: Path,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_objective_path: Path | None = None,
    default_objective_bundle_dir: Path | None = None,
) -> tuple[object, ImplementationDaemonRunContext]:
    """Build a ``PortalImplementationDaemon`` from parsed CLI args and local defaults."""

    from .todo_daemon.implementation_daemon import (
        DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS,
        PortalImplementationDaemon,
    )

    apply_merge_resolver_environment(parsed)
    state_paths = implementation_state_paths(parsed)
    worktree_submodule_paths = (
        getattr(parsed, "worktree_submodule_path", None)
        or default_worktree_submodule_paths
        or None
    )
    daemon = PortalImplementationDaemon(
        todo_path=parsed.todo_path,
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
        worktree_submodule_paths=worktree_submodule_paths,
        objective_path=parsed.objective_path or default_objective_path,
        objective_bundle_dir=parsed.objective_bundle_dir or default_objective_bundle_dir,
        llm_merge_resolver_command=parsed.llm_merge_resolver_command or None,
        llm_merge_resolver_timeout_seconds=parsed.llm_merge_resolver_timeout_seconds,
        merge_reconciliation_max_merges=parsed.merge_reconciliation_max_merges,
        merged_worktree_cleanup_max=parsed.merged_worktree_cleanup_max,
        task_shard_count=parsed.task_shard_count,
        task_shard_index=parsed.task_shard_index,
    )
    return daemon, ImplementationDaemonRunContext(parsed=parsed, **state_paths)


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
        if result:
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
    pass_index = 0
    while True:
        pass_context = context.for_pass(pass_index)
        _run_hooks(hooks, phase="before", context=pass_context, logger=logger)
        result = daemon.run_once()
        _run_hooks(hooks, phase="after", context=pass_context, logger=logger)
        logger.info(pass_complete_message, result)
        if parsed.once:
            break
        pass_index += 1
        time.sleep(parsed.interval)


def run_configured_portal_implementation_daemon(
    argv: Sequence[str],
    *,
    repo_root: Path,
    logger: logging.Logger,
    default_worktree_submodule_paths: Sequence[str] | None = None,
    default_objective_path: Path | None = None,
    default_objective_bundle_dir: Path | None = None,
    hooks: Sequence[DaemonLoopHook] = (),
    pass_complete_message: str = "Portal implementation daemon pass complete: %s",
) -> None:
    """Parse, build, and run a configured portal implementation daemon."""

    from .todo_daemon.implementation_daemon import parse_args

    parsed = parse_args(list(argv))
    configure_daemon_logging(parsed)
    daemon, context = build_portal_implementation_daemon_from_args(
        parsed,
        repo_root=repo_root,
        default_worktree_submodule_paths=default_worktree_submodule_paths,
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
