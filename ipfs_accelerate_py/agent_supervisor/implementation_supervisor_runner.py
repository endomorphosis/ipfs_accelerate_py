"""Reusable runner helpers for configured implementation supervisors."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


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


@dataclass(frozen=True)
class SupervisorRunHook:
    """One before/after hook around a configured supervisor pass."""

    phase: str
    message: str
    callback: SupervisorRunHookCallback
    log_level: int = logging.WARNING


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
    context = ImplementationSupervisorRunContext(
        parsed=parsed,
        config=config,
        state_path=config.state_path,
        strategy_path=config.strategy_path,
        events_path=config.events_path,
        daemon_events_path=parsed.state_dir / f"{parsed.state_prefix}_events.jsonl",
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

    _run_hooks(hooks, phase="before", context=context, logger=logger)
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
