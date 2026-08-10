"""Fail-closed launcher for sealed agent-supervisor scheduler configurations.

The implementation supervisor already owns worker lifecycle, deterministic
task sharding, worktree isolation, and merge serialization.  This module is a
small configuration boundary that turns a reviewed ``scheduler_config@1``
JSON document into arguments for that existing runtime.

No provider is imported or probed while loading, preflighting, or rendering a
launch plan.  In particular, dry runs do not read credentials or install
optional tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .configured_board_bootstrap_seal import (
    BootstrapSealError,
    content_id,
    verify_bootstrap_seal,
)
from .multi_supervisor_runner import (
    ImplementationSupervisorTrackConfig,
    build_configured_multi_supervisor_cli_runner,
    utc_run_stamp,
)
from .ordered_provider_authoring import (
    AUTHORING_LAUNCH_SCHEMA,
    BOOTSTRAP_AUTHORING_BOARD_ID_ENV,
    BOOTSTRAP_BASELINE_ID_ENV,
    BOOTSTRAP_FOREST_ID_ENV,
    BOOTSTRAP_INVENTORY_ID_ENV,
    BOOTSTRAP_SEAL_ID_ENV,
    BOOTSTRAP_SEAL_PATH_ENV,
    CONFIGURED_BOARD_CONFIG_PATH_ENV,
    CONFIGURED_BOARD_LAUNCH_HEAD_ENV,
    CONFIGURED_BOARD_LAUNCH_ID_ENV,
    CONFIGURED_BOARD_LAUNCH_TREE_ENV,
    CONFIGURED_BOARD_NAMESPACE_ENV,
    GROK_MAX_TURNS_ENV,
    ORDERED_PROVIDER_ENV_NAMES,
)

SCHEDULER_SCHEMA_PATTERN = re.compile(
    r"^ipfs_accelerate_py\.agent_supervisor\."
    r"[a-z0-9_.-]+\.scheduler_config@1$"
)
IMPLEMENTATION_ENTRY_PATH = Path(
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)
PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
FALLBACK_PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
FALLBACK_TRIGGER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
GROK_MODEL_ENV = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV = "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
SCHEDULER_PROVIDER_ENV_NAMES = (
    PROVIDER_ENV,
    FALLBACK_PROVIDER_ENV,
    FALLBACK_TRIGGER_ENV,
    GROK_MODEL_ENV,
    CODEX_MODEL_ENV,
    CODEX_REASONING_EFFORT_ENV,
    *ORDERED_PROVIDER_ENV_NAMES,
)
ORDERED_PROVIDER_FIELDS = (
    "primary_provider_id",
    "primary_model_id",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_trigger",
    "fallback_reasoning_effort",
)
ORDERED_PRIMARY_PROVIDER_ID = "grok_cli"
ORDERED_PRIMARY_MODEL_ID = "grok-4.5"
ORDERED_FALLBACK_PROVIDER_ID = "codex"
ORDERED_FALLBACK_MODEL_ID = "gpt-5.6-terra"
# ``primary_quota_exhausted`` is the reviewed legacy contract used by the
# existing KITA boards and generic launcher default.  New sealed DCR
# authoring can opt into the broader, still-closed pre-dispatch-unavailability
# route.  Keep the values distinct: accepting the new trigger must not
# silently grant it to an old board.
ORDERED_LEGACY_FALLBACK_TRIGGER = "primary_quota_exhausted"
ORDERED_PRIMARY_UNAVAILABLE_FALLBACK_TRIGGER = (
    "primary_unavailable_or_quota_exhausted"
)
ORDERED_FALLBACK_TRIGGER = ORDERED_LEGACY_FALLBACK_TRIGGER
ORDERED_FALLBACK_TRIGGERS = frozenset(
    {
        ORDERED_LEGACY_FALLBACK_TRIGGER,
        ORDERED_PRIMARY_UNAVAILABLE_FALLBACK_TRIGGER,
    }
)
ORDERED_FALLBACK_REASONING_EFFORTS = frozenset({"medium", "high"})


class ConfiguredBoardError(ValueError):
    """The scheduler document or its repository binding is inadmissible."""


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfiguredBoardError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ConfiguredBoardError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(f"{field} must be a positive integer") from exc
    if parsed < 1:
        raise ConfiguredBoardError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    return parsed


def _required_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfiguredBoardError(f"{field} must be a nonempty string")
    return value.strip()


def _provider_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = _required_string(payload, field)
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ConfiguredBoardError(f"{field} must be a single-line nonempty string")
    return value


def _optional_provider_string(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = payload.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfiguredBoardError(f"{field} must be a string")
    normalized = value.strip()
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ConfiguredBoardError(f"{field} must be a single-line string")
    return normalized


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ConfiguredBoardError(f"{field} must be a relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or path.as_posix() in {".", ".."}
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ConfiguredBoardError(f"{field} contains unsafe relative path {value!r}")
    return path.as_posix()


def _safe_relative_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfiguredBoardError(f"{field} must be a list")
    paths = tuple(
        _safe_relative(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(paths) != len(set(paths)):
        raise ConfiguredBoardError(f"{field} contains duplicate paths")
    return paths


def _contained_path(repo_root: Path, relative: str) -> Path:
    candidate = repo_root / relative
    try:
        candidate.resolve(strict=False).relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfiguredBoardError(f"path escapes repository: {relative}") from exc
    return candidate


def _task_header_prefix(task_prefix: str) -> str:
    stripped = task_prefix.strip()
    return stripped if stripped.startswith("## ") else f"## {stripped}"


def _slug(value: str) -> str:
    return (
        re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
        or "configured-board"
    )


@dataclass(frozen=True)
class ConfiguredBoard:
    """Validated scheduler JSON and its exact checkout binding."""

    config_path: Path
    repo_root: Path
    payload: Mapping[str, Any]
    taskboard_path: str
    objectives_path: str
    plan_path: str
    validator_path: str
    task_prefix: str
    board_namespace: str
    merge_target_branch: str
    max_lanes: int
    strict_task_sharding: bool
    worktree_submodule_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    runtime_paths: Mapping[str, str]
    bootstrap_seal_path: str

    @property
    def task_header_prefix(self) -> str:
        return _task_header_prefix(self.task_prefix)

    def path(self, relative: str) -> Path:
        return _contained_path(self.repo_root, relative)


def load_configured_board(
    config_path: Path | str,
    *,
    repo_root: Path | str,
) -> ConfiguredBoard:
    """Load and structurally validate one sealed scheduler document."""

    root = Path(repo_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfiguredBoardError(
            "scheduler config must be inside the repository"
        ) from exc
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ConfiguredBoardError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardError(
            f"scheduler config is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfiguredBoardError("scheduler config root must be an object")
    schema = _required_string(payload, "schema")
    if SCHEDULER_SCHEMA_PATTERN.fullmatch(schema) is None:
        raise ConfiguredBoardError(f"unsupported scheduler schema: {schema!r}")

    taskboard_path = _safe_relative(
        _required_string(payload, "taskboard_path"),
        field="taskboard_path",
    )
    objectives_path = _safe_relative(
        _required_string(payload, "objectives_path"),
        field="objectives_path",
    )
    plan_path = _safe_relative(
        _required_string(payload, "plan_path"),
        field="plan_path",
    )
    validator_path = _safe_relative(
        _required_string(payload, "validator_path"),
        field="validator_path",
    )
    task_prefix = _required_string(payload, "task_prefix")
    if re.fullmatch(r"(?:## )?[A-Z][A-Z0-9_-]*-", task_prefix) is None:
        raise ConfiguredBoardError("task_prefix is not a supported task prefix")
    board_namespace = _required_string(payload, "board_namespace")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", board_namespace) is None:
        raise ConfiguredBoardError("board_namespace is unsafe")
    merge_target_branch = _required_string(payload, "merge_target_branch")
    if (
        merge_target_branch.startswith("-")
        or "\x00" in merge_target_branch
        or any(character.isspace() for character in merge_target_branch)
    ):
        raise ConfiguredBoardError("merge_target_branch is unsafe")

    max_lanes = _positive_int(payload.get("max_lanes"), field="max_lanes")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != max_lanes:
        raise ConfiguredBoardError("lanes must contain exactly max_lanes entries")
    expected_indices = list(range(max_lanes))
    actual_indices: list[int] = []
    for position, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise ConfiguredBoardError(f"lanes[{position}] must be an object")
        raw_index = lane.get("index")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise ConfiguredBoardError(
                f"lanes[{position}].index must be a nonnegative integer"
            )
        index = raw_index
        if lane.get("strict_shard_remainder") != index:
            raise ConfiguredBoardError(
                f"lanes[{position}] strict shard remainder mismatch"
            )
        name = _required_string(lane, "name")
        if _slug(name) != name:
            raise ConfiguredBoardError(f"lanes[{position}].name is unsafe")
        actual_indices.append(index)
    if actual_indices != expected_indices:
        raise ConfiguredBoardError("lane indices must be contiguous and ordered")

    strict_task_sharding = payload.get("strict_task_sharding")
    if not isinstance(strict_task_sharding, bool):
        raise ConfiguredBoardError("strict_task_sharding must be boolean")
    submodules = _safe_relative_list(
        payload.get("worktree_submodule_paths"),
        field="worktree_submodule_paths",
    )
    protected = _safe_relative_list(
        payload.get("protected_paths"),
        field="protected_paths",
    )
    config_relative = path.relative_to(root).as_posix()
    if config_relative not in protected:
        raise ConfiguredBoardError("scheduler config must protect its own source path")
    source_binding = payload.get("source_binding")
    if not isinstance(source_binding, dict):
        raise ConfiguredBoardError("source_binding must be an object")
    raw_seal_path = source_binding.get("bootstrap_seal_path")
    bootstrap_seal_path = (
        ""
        if raw_seal_path is None
        else _safe_relative(
            raw_seal_path,
            field="source_binding.bootstrap_seal_path",
        )
    )
    if bootstrap_seal_path:
        if bootstrap_seal_path not in protected:
            raise ConfiguredBoardError("bootstrap seal path must be protected")
        for field in (
            "record_recursive_repository_forest_at_launch",
            "changed_revision_requires_fresh_inventory_and_baseline",
        ):
            if source_binding.get(field) is not True:
                raise ConfiguredBoardError(
                    f"source_binding.{field} must be true when a bootstrap "
                    "seal is configured"
                )

    runtime_raw = payload.get("runtime_paths")
    if not isinstance(runtime_raw, dict):
        raise ConfiguredBoardError("runtime_paths must be an object")
    runtime_paths = {
        field: _safe_relative(
            _required_string(runtime_raw, field),
            field=f"runtime_paths.{field}",
        )
        for field in (
            "root",
            "state",
            "worktrees",
            "merge_queue",
            "logs",
        )
    }
    runtime_root_parts = PurePosixPath(runtime_paths["root"]).parts
    for field, relative in runtime_paths.items():
        if field == "root":
            continue
        if PurePosixPath(relative).parts[: len(runtime_root_parts)] != (
            runtime_root_parts
        ):
            raise ConfiguredBoardError(
                f"runtime_paths.{field} must be under runtime_paths.root"
            )

    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ConfiguredBoardError("provider must be an object")
    ordered_provider = any(field in provider for field in ORDERED_PROVIDER_FIELDS)
    if ordered_provider:
        primary_provider_id = _provider_string(
            provider,
            "primary_provider_id",
        )
        primary_model_id = _provider_string(provider, "primary_model_id")
        fallback_provider_id = _provider_string(
            provider,
            "fallback_provider_id",
        )
        fallback_model_id = _provider_string(provider, "fallback_model_id")
        fallback_trigger = _provider_string(
            provider,
            "fallback_trigger",
        )
        fallback_reasoning_effort = _provider_string(
            provider,
            "fallback_reasoning_effort",
        )
        if primary_provider_id != ORDERED_PRIMARY_PROVIDER_ID:
            raise ConfiguredBoardError(
                "provider.primary_provider_id must be 'grok_cli' for "
                "the ordered provider contract"
            )
        if primary_model_id != ORDERED_PRIMARY_MODEL_ID:
            raise ConfiguredBoardError(
                "provider.primary_model_id must be 'grok-4.5' for "
                "the ordered provider contract"
            )
        if fallback_provider_id != ORDERED_FALLBACK_PROVIDER_ID:
            raise ConfiguredBoardError(
                "provider.fallback_provider_id must be 'codex' for "
                "the ordered provider contract"
            )
        if fallback_model_id != ORDERED_FALLBACK_MODEL_ID:
            raise ConfiguredBoardError(
                "provider.fallback_model_id must be 'gpt-5.6-terra' for "
                "the ordered provider contract"
            )
        if fallback_trigger not in ORDERED_FALLBACK_TRIGGERS:
            raise ConfiguredBoardError(
                "provider.fallback_trigger must be one of "
                "'primary_quota_exhausted', "
                "'primary_unavailable_or_quota_exhausted' for the ordered "
                "provider contract"
            )
        if fallback_reasoning_effort not in ORDERED_FALLBACK_REASONING_EFFORTS:
            raise ConfiguredBoardError(
                "provider.fallback_reasoning_effort must be one of "
                "'medium', 'high' for "
                "the ordered provider contract"
            )
        if (
            fallback_trigger == ORDERED_PRIMARY_UNAVAILABLE_FALLBACK_TRIGGER
            and fallback_reasoning_effort != "high"
        ):
            raise ConfiguredBoardError(
                "provider.fallback_reasoning_effort must be 'high' when "
                "provider.fallback_trigger is "
                "'primary_unavailable_or_quota_exhausted'"
            )
        if provider.get("provider_fallback_for_other_failures", False) is not False:
            raise ConfiguredBoardError(
                "provider.provider_fallback_for_other_failures must be false "
                "for the ordered provider contract"
            )
        if "provider_id" in provider or "model_id" in provider:
            raise ConfiguredBoardError(
                "ordered provider fields cannot be mixed with legacy "
                "provider_id/model_id"
            )
    else:
        provider_id = _optional_provider_string(
            provider,
            "provider_id",
        ).lower()
        _optional_provider_string(provider, "model_id")
        if (
            provider_id
            and re.fullmatch(
                r"[a-z0-9][a-z0-9_-]*",
                provider_id,
            )
            is None
        ):
            raise ConfiguredBoardError(
                "provider.provider_id is not a supported identifier"
            )
    concurrency = _positive_int(
        provider.get("max_concurrency"),
        field="provider.max_concurrency",
    )
    if concurrency < max_lanes:
        raise ConfiguredBoardError("provider.max_concurrency is lower than max_lanes")
    for field in (
        "strict_task_sharding",
        "exit_when_all_tracks_terminal",
        "objective_refill_enabled",
        "codebase_refill_enabled",
    ):
        if not isinstance(payload.get(field), bool):
            raise ConfiguredBoardError(f"{field} must be boolean")

    for field in (
        "poll_interval_seconds",
        "daemon_interval_seconds",
        "check_interval_seconds",
        "stale_seconds",
        "watchdog_startup_grace_seconds",
        "implementation_timeout_seconds",
        "implementation_max_timeout_seconds",
        "implementation_log_stall_seconds",
    ):
        _nonnegative_number(payload.get(field), field=field)
    for field in (
        "max_restarts",
        "max_task_attempts",
        "implementation_retry_budget",
        "validation_retry_budget",
        "merge_retry_budget",
    ):
        _positive_int(payload.get(field), field=field)

    return ConfiguredBoard(
        config_path=path,
        repo_root=root,
        payload=payload,
        taskboard_path=taskboard_path,
        objectives_path=objectives_path,
        plan_path=plan_path,
        validator_path=validator_path,
        task_prefix=task_prefix,
        board_namespace=board_namespace,
        merge_target_branch=merge_target_branch,
        max_lanes=max_lanes,
        strict_task_sharding=strict_task_sharding,
        worktree_submodule_paths=submodules,
        protected_paths=protected,
        runtime_paths=runtime_paths,
        bootstrap_seal_path=bootstrap_seal_path,
    )


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = list(argv)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def _git(
    board: ConfiguredBoard,
    *args: str,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ("git", *args),
        cwd=board.repo_root,
        timeout=timeout,
    )


def _append_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _gitlink_commit(
    board: ConfiguredBoard,
    relative: str,
) -> str:
    result = _git(board, "ls-tree", "HEAD", "--", relative)
    if result.returncode != 0:
        return ""
    match = re.fullmatch(
        rf"160000 commit ([0-9a-f]{{40}})\t{re.escape(relative)}\n?",
        result.stdout,
    )
    return match.group(1) if match else ""


def preflight_configured_board(board: ConfiguredBoard) -> dict[str, Any]:
    """Prove that a scheduler document can safely launch from this checkout."""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    top = _git(board, "rev-parse", "--show-toplevel")
    _append_check(
        checks,
        errors,
        name="repository_root",
        passed=(
            top.returncode == 0
            and Path(top.stdout.strip()).resolve() == board.repo_root
        ),
        detail=top.stderr.strip() or top.stdout.strip(),
    )

    source_binding = board.payload.get("source_binding")
    if not isinstance(source_binding, dict):
        errors.append("source_binding must be an object")
        source_binding = {}
    required_branch = str(
        source_binding.get("accelerator_required_branch") or ""
    ).strip()
    current_branch = _git(board, "branch", "--show-current")
    _append_check(
        checks,
        errors,
        name="required_branch",
        passed=(
            current_branch.returncode == 0
            and current_branch.stdout.strip() == required_branch
            and required_branch == board.merge_target_branch
        ),
        detail={
            "expected": required_branch,
            "merge_target": board.merge_target_branch,
            "actual": current_branch.stdout.strip(),
        },
    )
    branch_format = _git(
        board,
        "check-ref-format",
        "--branch",
        board.merge_target_branch,
    )
    target_ref = _git(
        board,
        "rev-parse",
        "--verify",
        f"{board.merge_target_branch}^{{commit}}",
    )
    _append_check(
        checks,
        errors,
        name="merge_target",
        passed=branch_format.returncode == 0 and target_ref.returncode == 0,
        detail=target_ref.stderr.strip() or target_ref.stdout.strip(),
    )
    required_ancestor = str(
        source_binding.get("accelerator_required_ancestor") or ""
    ).strip()
    ancestor = _git(
        board,
        "merge-base",
        "--is-ancestor",
        required_ancestor,
        "HEAD",
    )
    _append_check(
        checks,
        errors,
        name="required_ancestor",
        passed=bool(re.fullmatch(r"[0-9a-f]{40}", required_ancestor))
        and ancestor.returncode == 0,
        detail=required_ancestor,
    )

    required_files = {
        board.config_path.relative_to(board.repo_root).as_posix(),
        board.taskboard_path,
        board.objectives_path,
        board.plan_path,
        board.validator_path,
        *board.protected_paths,
    }
    missing_files = sorted(
        relative for relative in required_files if not board.path(relative).is_file()
    )
    _append_check(
        checks,
        errors,
        name="control_files_present",
        passed=not missing_files,
        detail=missing_files,
    )
    tracked = [
        relative
        for relative in sorted(required_files)
        if _git(
            board,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ).returncode
        == 0
    ]
    untracked_control = sorted(required_files - set(tracked))
    _append_check(
        checks,
        errors,
        name="control_files_tracked",
        passed=not untracked_control,
        detail=untracked_control,
    )
    status = _git(
        board,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    dirty_lines = [line for line in status.stdout.splitlines() if line]
    _append_check(
        checks,
        errors,
        name="checkout_clean",
        passed=status.returncode == 0 and not dirty_lines,
        detail=dirty_lines[:100],
    )

    validator_report: dict[str, Any] = {}
    if board.path(board.validator_path).is_file():
        validator = _run(
            (
                sys.executable,
                str(board.path(board.validator_path)),
                "--check-all",
            ),
            cwd=board.repo_root,
        )
        try:
            parsed = json.loads(validator.stdout)
            if isinstance(parsed, dict):
                validator_report = parsed
        except json.JSONDecodeError:
            validator_report = {}
        _append_check(
            checks,
            errors,
            name="declared_validator",
            passed=(
                validator.returncode == 0 and validator_report.get("valid") is True
            ),
            detail={
                "returncode": validator.returncode,
                "stderr": validator.stderr[-2000:],
                "errors": validator_report.get("errors"),
            },
        )

    planning_revisions: dict[str, str] = {}
    for key, value in source_binding.items():
        if not key.endswith("_submodule_path") or not isinstance(value, str):
            continue
        prefix = key[: -len("_submodule_path")]
        revision = source_binding.get(f"{prefix}_planning_revision")
        if isinstance(revision, str) and revision.strip():
            planning_revisions[value.strip()] = revision.strip()

    submodule_checks: list[dict[str, Any]] = []
    for relative in board.worktree_submodule_paths:
        gitlink = _gitlink_commit(board, relative)
        target = board.path(relative)
        top_level = (
            _run(
                ("git", "rev-parse", "--show-toplevel"),
                cwd=target,
                timeout=60,
            )
            if target.is_dir()
            else None
        )
        exact_worktree = bool(
            top_level is not None
            and top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == target.resolve()
        )
        head = (
            _run(
                ("git", "rev-parse", "HEAD"),
                cwd=target,
                timeout=60,
            )
            if exact_worktree
            else None
        )
        clean = (
            _run(
                ("git", "status", "--porcelain=v1", "--untracked-files=all"),
                cwd=target,
                timeout=60,
            )
            if head is not None and head.returncode == 0
            else None
        )
        actual_head = head.stdout.strip() if head is not None else ""
        expected_planning = planning_revisions.get(relative, "")
        planning_ancestor = (
            _run(
                (
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    expected_planning,
                    actual_head,
                ),
                cwd=target,
                timeout=60,
            )
            if (
                exact_worktree
                and re.fullmatch(r"[0-9a-f]{40}", expected_planning)
                and re.fullmatch(r"[0-9a-f]{40}", actual_head)
            )
            else None
        )
        valid = bool(
            gitlink
            and exact_worktree
            and head is not None
            and head.returncode == 0
            and actual_head == gitlink
            and planning_ancestor is not None
            and planning_ancestor.returncode == 0
            and clean is not None
            and clean.returncode == 0
            and not clean.stdout.strip()
        )
        submodule_checks.append(
            {
                "path": relative,
                "valid": valid,
                "gitlink": gitlink,
                "head": actual_head,
                "exact_worktree": exact_worktree,
                "planning_revision": expected_planning,
                "planning_revision_is_ancestor": bool(
                    planning_ancestor is not None and planning_ancestor.returncode == 0
                ),
                "dirty": (clean.stdout.splitlines()[:50] if clean is not None else []),
            }
        )
    _append_check(
        checks,
        errors,
        name="configured_submodules",
        passed=all(item["valid"] for item in submodule_checks),
        detail=submodule_checks,
    )

    if board.bootstrap_seal_path:
        try:
            seal_detail: Any = verify_bootstrap_seal(
                repo_root=board.repo_root,
                board_namespace=board.board_namespace,
                source_binding=source_binding,
                worktree_submodule_paths=board.worktree_submodule_paths,
                protected_paths=board.protected_paths,
                seal_path=board.bootstrap_seal_path,
                taskboard_path=board.taskboard_path,
                task_header_prefix=board.task_prefix,
                validator_report=validator_report,
            )
            seal_valid = True
        except BootstrapSealError as exc:
            seal_detail = str(exc)
            seal_valid = False
        _append_check(
            checks,
            errors,
            name="bootstrap_seal",
            passed=seal_valid,
            detail=seal_detail,
        )

    implementation_entry = board.path(IMPLEMENTATION_ENTRY_PATH.as_posix())
    _append_check(
        checks,
        errors,
        name="implementation_entry",
        passed=implementation_entry.is_file(),
        detail=str(implementation_entry),
    )

    return {
        "schema": ("ipfs_accelerate_py/agent-supervisor/configured-board-preflight@1"),
        "valid": not errors,
        "config_path": str(board.config_path),
        "repo_root": str(board.repo_root),
        "board_namespace": board.board_namespace,
        "taskboard_path": str(board.path(board.taskboard_path)),
        "max_lanes": board.max_lanes,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "validator_report": validator_report,
    }


def configured_board_common_args(
    board: ConfiguredBoard,
    *,
    implement: bool,
) -> tuple[str, ...]:
    """Map scheduler policy to existing implementation-supervisor CLI args."""

    payload = board.payload
    args: list[str] = [
        "--todo-path",
        str(board.path(board.taskboard_path)),
        "--task-prefix",
        board.task_header_prefix,
        "--worktree-root",
        str(board.path(board.runtime_paths["worktrees"])),
        "--merge-target-branch",
        board.merge_target_branch,
        "--merge-queue-dir",
        str(board.path(board.runtime_paths["merge_queue"])),
        "--stale-seconds",
        str(payload["stale_seconds"]),
        "--check-interval",
        str(payload["check_interval_seconds"]),
        "--watchdog-startup-grace-seconds",
        str(payload["watchdog_startup_grace_seconds"]),
        "--max-restarts",
        str(payload["max_restarts"]),
        "--max-task-attempts",
        str(payload["max_task_attempts"]),
        "--daemon-interval",
        str(payload["daemon_interval_seconds"]),
        "--implementation-timeout",
        str(payload["implementation_timeout_seconds"]),
        "--implementation-max-timeout",
        str(payload["implementation_max_timeout_seconds"]),
        "--implementation-log-stall-seconds",
        str(payload["implementation_log_stall_seconds"]),
        "--implementation-retry-budget",
        str(payload["implementation_retry_budget"]),
        "--validation-retry-budget",
        str(payload["validation_retry_budget"]),
        "--merge-retry-budget",
        str(payload["merge_retry_budget"]),
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
        # DCR boards use an exact finite task population; reconciliation-guardrail
        # ops rows (DCR-10x) break validator wave equality and authoring seals.
        "--no-reconciliation-guardrail",
        "--log-level",
        "INFO",
    ]
    args.append("--implement" if implement else "--no-implement")
    if board.strict_task_sharding:
        args.append("--strict-task-sharding")
    for relative in board.worktree_submodule_paths:
        args.extend(["--worktree-submodule-path", relative])
    for relative in board.protected_paths:
        args.extend(["--implementation-protected-path", relative])
    if payload.get("objective_refill_enabled") is True:
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(board.path(board.objectives_path)),
            ]
        )
    if payload.get("codebase_refill_enabled") is True:
        args.append("--codebase-refill-scan")
    return tuple(args)


def configured_board_launch_plan(
    board: ConfiguredBoard,
    *,
    implement: bool,
    detach: bool,
    duration_seconds: float = float("inf"),
    stamp: str | None = None,
    preflight_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the exact existing multi-supervisor runner invocation."""

    run_stamp = stamp or utc_run_stamp()
    runtime_root = board.path(board.runtime_paths["root"])
    state_dir = board.path(board.runtime_paths["state"])
    log_dir = board.path(board.runtime_paths["logs"])
    entry = board.path(IMPLEMENTATION_ENTRY_PATH.as_posix())
    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=board.repo_root,
        duration_seconds=duration_seconds,
        heartbeat_interval_seconds=max(
            1.0,
            float(board.payload["poll_interval_seconds"]),
        ),
        supervisor_status_stale_seconds=max(
            60.0,
            float(board.payload["stale_seconds"]),
        ),
        stop_grace_seconds=max(
            30.0,
            float(board.payload["check_interval_seconds"]) * 2.0,
        ),
        stamp=run_stamp,
        master_dir=runtime_root,
        master_log=log_dir / f"configured-board-{run_stamp}.log",
        master_pid_path=state_dir / "configured-board-master.pid",
        label=board.board_namespace,
        python_executable=sys.executable,
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name=board.board_namespace,
                script_path=entry,
                state_dir=state_dir,
                state_prefix=_slug(board.task_prefix),
            ),
        ),
        common_args=configured_board_common_args(
            board,
            implement=implement,
        ),
        detach=detach,
    )
    runner_args = [
        *runner.args(),
        "--implementation-supervisor-lanes-per-track",
        str(board.max_lanes),
    ]
    if board.strict_task_sharding:
        runner_args.append("--implementation-supervisor-strict-task-sharding")
    if board.payload.get("exit_when_all_tracks_terminal") is True:
        runner_args.append("--exit-when-all-tracks-terminal")

    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    ordered_provider = any(field in provider for field in ORDERED_PROVIDER_FIELDS)
    if ordered_provider:
        environment = {
            PROVIDER_ENV: str(provider["primary_provider_id"]).strip(),
            FALLBACK_PROVIDER_ENV: str(provider["fallback_provider_id"]).strip(),
            FALLBACK_TRIGGER_ENV: str(provider["fallback_trigger"]).strip(),
            GROK_MODEL_ENV: str(provider["primary_model_id"]).strip(),
            CODEX_MODEL_ENV: str(provider["fallback_model_id"]).strip(),
            CODEX_REASONING_EFFORT_ENV: str(
                provider["fallback_reasoning_effort"]
            ).strip(),
        }
    else:
        provider_id = str(provider.get("provider_id") or "").strip()
        model_id = str(provider.get("model_id") or "").strip()
        environment = {}
        if provider_id and provider_id != "auto":
            environment[PROVIDER_ENV] = provider_id
        if model_id and provider_id in {"", "auto", "codex", "openai"}:
            environment[CODEX_MODEL_ENV] = model_id
    bootstrap_seal: dict[str, Any] = {}
    authoring_launch: dict[str, Any] = {}
    if board.bootstrap_seal_path:
        report = dict(preflight_report or {})
        if (
            report.get("valid") is not True
            or report.get("board_namespace") != board.board_namespace
            or Path(str(report.get("config_path") or "")).resolve() != board.config_path
            or Path(str(report.get("repo_root") or "")).resolve() != board.repo_root
        ):
            raise ConfiguredBoardError(
                "a matching valid preflight report is required for sealed launch"
            )
        seal_checks = [
            item
            for item in report.get("checks", ())
            if isinstance(item, Mapping) and item.get("name") == "bootstrap_seal"
        ]
        if (
            len(seal_checks) != 1
            or seal_checks[0].get("passed") is not True
            or not isinstance(seal_checks[0].get("detail"), Mapping)
        ):
            raise ConfiguredBoardError(
                "preflight report has no unique verified bootstrap seal"
            )
        source_binding = board.payload.get("source_binding")
        validator_report = report.get("validator_report")
        try:
            current_seal = verify_bootstrap_seal(
                repo_root=board.repo_root,
                board_namespace=board.board_namespace,
                source_binding=(
                    source_binding if isinstance(source_binding, Mapping) else {}
                ),
                worktree_submodule_paths=board.worktree_submodule_paths,
                protected_paths=board.protected_paths,
                seal_path=board.bootstrap_seal_path,
                taskboard_path=board.taskboard_path,
                task_header_prefix=board.task_prefix,
                validator_report=(
                    validator_report if isinstance(validator_report, Mapping) else {}
                ),
            )
        except BootstrapSealError as exc:
            raise ConfiguredBoardError(
                f"bootstrap seal changed after preflight: {exc}"
            ) from exc
        verified_detail = dict(seal_checks[0]["detail"])
        if current_seal != verified_detail:
            raise ConfiguredBoardError(
                "bootstrap seal identity changed after preflight"
            )
        bootstrap_seal = dict(current_seal)
        if implement and ordered_provider:
            launch_head_result = _git(board, "rev-parse", "HEAD^{commit}")
            launch_tree_result = _git(board, "rev-parse", "HEAD^{tree}")
            launch_head = launch_head_result.stdout.strip()
            launch_tree = launch_tree_result.stdout.strip()
            if (
                launch_head_result.returncode != 0
                or launch_tree_result.returncode != 0
                or re.fullmatch(r"[0-9a-f]{40}", launch_head) is None
                or re.fullmatch(r"[0-9a-f]{40}", launch_tree) is None
            ):
                raise ConfiguredBoardError(
                    "cannot bind the configured-board launch Git identity"
                )
            config_relative = board.config_path.relative_to(board.repo_root).as_posix()
            config_sha256 = hashlib.sha256(board.config_path.read_bytes()).hexdigest()
            launch_body = {
                "schema": AUTHORING_LAUNCH_SCHEMA,
                "board_namespace": board.board_namespace,
                "scheduler_config_path": config_relative,
                "scheduler_config_sha256": config_sha256,
                "seal_id": str(bootstrap_seal.get("seal_id") or ""),
                "forest_id": str(bootstrap_seal.get("forest_id") or ""),
                "inventory_id": str(bootstrap_seal.get("inventory_id") or ""),
                "baseline_id": str(bootstrap_seal.get("baseline_id") or ""),
                "authoring_board_id": str(
                    bootstrap_seal.get("authoring_board_id") or ""
                ),
                "launch_head": launch_head,
                "launch_tree": launch_tree,
            }
            authoring_launch = {
                **launch_body,
                "launch_id": content_id(launch_body),
            }
            environment.update(
                {
                    CONFIGURED_BOARD_NAMESPACE_ENV: board.board_namespace,
                    CONFIGURED_BOARD_CONFIG_PATH_ENV: config_relative,
                    BOOTSTRAP_SEAL_PATH_ENV: board.bootstrap_seal_path,
                    BOOTSTRAP_SEAL_ID_ENV: str(bootstrap_seal.get("seal_id") or ""),
                    BOOTSTRAP_FOREST_ID_ENV: str(bootstrap_seal.get("forest_id") or ""),
                    BOOTSTRAP_INVENTORY_ID_ENV: str(
                        bootstrap_seal.get("inventory_id") or ""
                    ),
                    BOOTSTRAP_BASELINE_ID_ENV: str(
                        bootstrap_seal.get("baseline_id") or ""
                    ),
                    BOOTSTRAP_AUTHORING_BOARD_ID_ENV: str(
                        bootstrap_seal.get("authoring_board_id") or ""
                    ),
                    CONFIGURED_BOARD_LAUNCH_HEAD_ENV: launch_head,
                    CONFIGURED_BOARD_LAUNCH_TREE_ENV: launch_tree,
                    CONFIGURED_BOARD_LAUNCH_ID_ENV: str(authoring_launch["launch_id"]),
                    GROK_MAX_TURNS_ENV: "100000",
                }
            )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/configured-board-launch-plan@1"
        ),
        "board_namespace": board.board_namespace,
        "implement": bool(implement),
        "detach": bool(detach),
        "lanes": board.max_lanes,
        "strict_task_sharding": board.strict_task_sharding,
        "argv": runner_args,
        "environment": environment,
        "runtime_root": str(runtime_root),
        "master_pid_path": str(state_dir / "configured-board-master.pid"),
        "master_log": str(log_dir / f"configured-board-{run_stamp}.log"),
        "bootstrap_seal": bootstrap_seal,
        "authoring_launch": authoring_launch,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and launch a sealed supervisor scheduler config"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "preflight",
        help="Validate control files, Git bindings, submodules, and board",
    )
    launch = subparsers.add_parser(
        "launch",
        help="Render or run the configured multi-lane supervisor",
    )
    launch.add_argument(
        "--implement",
        action="store_true",
        help="Authorize implementation-provider dispatch",
    )
    launch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact launch plan without starting processes",
    )
    launch.add_argument(
        "--foreground",
        action="store_true",
        help="Keep the multi-supervisor runner in the foreground",
    )
    launch.add_argument(
        "--duration-seconds",
        type=float,
        default=float("inf"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        board = load_configured_board(
            args.config,
            repo_root=args.repo_root,
        )
        preflight = preflight_configured_board(board)
    except ConfiguredBoardError as exc:
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/configured-board-error@1"
                    ),
                    "valid": False,
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if args.command == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["valid"] else 2
    if not preflight["valid"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2

    detach = not bool(args.foreground)
    try:
        plan = configured_board_launch_plan(
            board,
            implement=bool(args.implement),
            detach=detach,
            duration_seconds=float(args.duration_seconds),
            preflight_report=preflight,
        )
    except ConfiguredBoardError as exc:
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/configured-board-error@1"
                    ),
                    "valid": False,
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    for name in SCHEDULER_PROVIDER_ENV_NAMES:
        if name not in plan["environment"]:
            os.environ.pop(name, None)
    for name, value in plan["environment"].items():
        os.environ[name] = value
    from .multi_supervisor_runner import main as multi_supervisor_main

    return int(multi_supervisor_main(plan["argv"]))


__all__ = (
    "ConfiguredBoard",
    "ConfiguredBoardError",
    "configured_board_common_args",
    "configured_board_launch_plan",
    "load_configured_board",
    "main",
    "preflight_configured_board",
)
