"""Reusable runner helpers for configured implementation daemons."""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .wrapper_utils import with_default, with_repeated_default


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


@dataclass(frozen=True)
class DaemonLoopHook:
    """One before/after hook for a configured implementation daemon loop."""

    phase: str
    message: str
    callback: DaemonLoopHookCallback
    log_level: int = logging.WARNING


RefillHookEntry = tuple[str, DaemonLoopHookCallback]


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
    worktree_submodule_paths: Sequence[str] = ()


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
    if defaults.worktree_submodule_paths:
        args = with_repeated_default(args, "--worktree-submodule-path", defaults.worktree_submodule_paths)
    return args


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


def implementation_state_paths(parsed: argparse.Namespace) -> dict[str, Path]:
    """Return standard task-state, strategy, and event-log paths for parsed daemon args."""

    state_dir = Path(parsed.state_dir)
    state_prefix = str(parsed.state_prefix)
    return {
        "state_path": state_dir / f"{state_prefix}_task_state.json",
        "strategy_path": state_dir / f"{state_prefix}_strategy.json",
        "events_path": state_dir / f"{state_prefix}_events.jsonl",
    }


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
    )
    return daemon, ImplementationDaemonRunContext(parsed=parsed, **state_paths)


def _run_hooks(
    hooks: Sequence[DaemonLoopHook],
    *,
    phase: str,
    context: ImplementationDaemonRunContext,
    logger: logging.Logger,
) -> None:
    for hook in hooks:
        if hook.phase != phase:
            continue
        result = hook.callback(context)
        if result:
            logger.log(hook.log_level, hook.message, result)


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
