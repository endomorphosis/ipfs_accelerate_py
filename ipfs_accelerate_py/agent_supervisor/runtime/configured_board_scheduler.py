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
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ...agent_implementation_route import (
    AgentSupervisorNativeDependencyLaunch,
    AgentSupervisorNativeDependencyPin,
    parse_agent_supervisor_native_dependency_pin,
    seal_agent_supervisor_native_dependency,
    verify_agent_supervisor_native_dependency_sealed_fd,
)
from ...llm_router import (
    AgentImplementationControlPlanePin,
    AgentImplementationRoutePlan,
    AgentImplementationSealedControlPlane,
    load_agent_implementation_route_authorization,
    materialize_agent_implementation_control_plane_capsule,
    project_agent_implementation_route_capacity,
    resolve_agent_implementation_route,
    seal_agent_implementation_control_plane_capsule,
    verify_agent_implementation_sealed_control_plane,
)
from ..contracts.execution import InvocationBudget
from ..control.plan_execution_store import (
    ConfiguredBoardExecutionSlices,
    ExecutionPlanError,
    ParallelismDecisionReceipt,
    ProductionParallelPlanAdapter,
    _load_plan_bound_execution_lease_locked,
    _load_plan_bound_proposal_disposition_locked,
    _load_plan_bound_wave_diff_barrier_locked,
    _secure_store_active,
    _secure_store_cas,
)
from ..merge.checkout_lock import serialized_lock_update
from ..planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
)
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.plan_revision_store import PlanRevisionStore
from ..task_sources.task_identity import canonical_task_identity
from ..task_sources.task_source import recompute_readiness_statuses
from ..task_sources.todo_vector_index import parse_todo_blocks, split_csv
from ..validation.validation_commands import split_validation_commands
from .configured_board_extension_projection import (
    CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
    CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
    ConfiguredBoardExtensionPin,
    ConfiguredBoardExtensionSetPin,
    build_configured_board_extension_set_pin,
    inspect_configured_board_extension_sources,
    parse_configured_board_extension_pin,
    parse_configured_board_extension_set_pin,
    project_configured_board_extension_set_home,
    verify_configured_board_extension_set_home,
)
from .configured_board_live_capsule import (
    ConfiguredBoardLiveCapsuleAdmission,
    ConfiguredBoardLiveCapsuleError,
    build_configured_board_live_capsule_admission,
    parse_configured_board_live_capsule_policy,
    verify_configured_board_live_capsule,
)
from .multi_supervisor_runner import (
    AUTHORITY_MODE_LEGACY_MARKDOWN,
    AUTHORITY_MODE_QUACK,
    DATABASE_PROGRAM_CONFIG_INTERFACE,
    DATABASE_PROGRAM_ENV_NAMES,
    DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
    ImplementationSupervisorTrackConfig,
    PlanBoundSupervisorChild,
    _read_stable_regular_bytes,
    _read_stable_regular_json,
    _StableArtifactReadError,
    accepted_control_plane_pin_json,
    build_configured_multi_supervisor_cli_runner,
    build_sealed_control_plane_module_command,
    parse_accepted_control_plane_pin,
    parse_database_program_config,
    parse_native_dependency_launch_json,
    utc_run_stamp,
)
from .provider_capacity_monitor import (
    DEFAULT_RESPONSE_TOKENS_PER_REQUEST,
    ProviderCapacityMonitor,
    ProviderCapacityMonitorConfig,
)
from .resource_scheduler import sample_host_resources

SCHEDULER_SCHEMA_PATTERN = re.compile(
    r"^ipfs_accelerate_py\.agent_supervisor\."
    r"[a-z0-9_.-]+\.scheduler_config@1$"
)
IMPLEMENTATION_ENTRY_PATH = Path(
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)
CONFIGURED_SCHEDULER_ENTRY_PATH = Path(
    "scripts/ops/agent_supervisor/configured_board_scheduler.py"
)
PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
FALLBACK_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
)
FALLBACK_TRIGGER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
)
GROK_MODEL_ENV = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV = (
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
)
GROK_BIN_ENV = "IPFS_ACCELERATE_AGENT_GROK_BIN"
ROUTE_BOARD_NAMESPACE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE"
)
ROUTE_AUTHORIZATION_PATH_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH"
)
ROUTE_AUTHORIZATION_SHA256_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256"
)
ROUTE_AUTHORIZATION_ID_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID"
)
ROUTE_AUTHORIZATION_KIND_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND"
)
ROUTE_SOURCE_HEAD_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD"
)
ROUTE_SOURCE_TREE_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE"
)
ROUTE_ID_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID"
MAX_COORDINATOR_WAVES = 4096
SCHEDULER_PROVIDER_ENV_NAMES = (
    PROVIDER_ENV,
    FALLBACK_PROVIDER_ENV,
    FALLBACK_TRIGGER_ENV,
    GROK_MODEL_ENV,
    CODEX_MODEL_ENV,
    CODEX_REASONING_EFFORT_ENV,
    GROK_BIN_ENV,
    ROUTE_BOARD_NAMESPACE_ENV,
    ROUTE_AUTHORIZATION_PATH_ENV,
    ROUTE_AUTHORIZATION_SHA256_ENV,
    ROUTE_AUTHORIZATION_ID_ENV,
    ROUTE_AUTHORIZATION_KIND_ENV,
    ROUTE_SOURCE_HEAD_ENV,
    ROUTE_SOURCE_TREE_ENV,
    ROUTE_ID_ENV,
)
ORDERED_PROVIDER_FIELDS = (
    "primary_provider_id",
    "primary_model_id",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_trigger",
    "fallback_reasoning_effort",
)
ORDERED_PRIMARY_EXECUTABLE_FIELD = "primary_executable"
ORDERED_PROVIDER_DETECTION_FIELDS = (
    *ORDERED_PROVIDER_FIELDS,
    ORDERED_PRIMARY_EXECUTABLE_FIELD,
)
ORDERED_PRIMARY_PROVIDER_ID = "grok_cli"
ORDERED_PRIMARY_MODEL_ID = "grok-4.6"
ORDERED_FALLBACK_PROVIDER_ID = "codex"
ORDERED_FALLBACK_MODEL_ID = "gpt-5.6-terra"
ORDERED_FALLBACK_TRIGGER = "primary_quota_exhausted"
ORDERED_FALLBACK_TRIGGERS = frozenset(
    {
        "primary_quota_exhausted",
        "primary_quota_or_auth_unavailable",
    }
)
ORDERED_FALLBACK_REASONING_EFFORTS = frozenset({"medium", "high"})
ROUTE_AUTHORIZATION_PATH_FIELD = "route_authorization_path"


class ConfiguredBoardError(ValueError):
    """The scheduler document or its repository binding is inadmissible."""


@dataclass(frozen=True)
class _ConfiguredBoardTaskPopulation:
    all_records: tuple[dict[str, Any], ...]
    ready_records: tuple[dict[str, Any], ...]
    completed_task_ids: tuple[str, ...]
    attempt_limited_task_ids: tuple[str, ...]
    state_snapshot_id: str


@dataclass(frozen=True)
class _ConfiguredBoardDependencySealSnapshot:
    """One exact HEAD-bound dependency-seal read reused for live launch."""

    payload: Mapping[str, Any]
    artifact: Mapping[str, object]


def _plan_bound_profile(board: "ConfiguredBoard") -> bool:
    """Whether this is the sealed v3 profile, rather than a legacy board."""

    return board.board_namespace == "agent-supervisor-prompt-only-self-improvement-v3"


def _sealed_configured_control_plane_required(board: "ConfiguredBoard") -> bool:
    """Whether live launch must re-enter through the accepted source capsule.

    The plan-bound v3 profile already has this property.  Quack-backed boards
    using the datasets-authoritative operational schema need the same source
    closure without changing their task authority to ``PlanRevisionStore``.
    """

    program = board.database_program
    return bool(
        _plan_bound_profile(board)
        or (
            program is not None
            and program.authority_mode == AUTHORITY_MODE_QUACK
            and program.schema_revision
            == DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
        )
    )


def _sanitized_git_environment() -> dict[str, str]:
    """Return a Git environment without ambient repository/config authority."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _sealed_coordinator_environment(
    board: "ConfiguredBoard",
    *,
    extension_directory: Path | None = None,
    extension_set_pin: ConfiguredBoardExtensionSetPin | None = None,
) -> dict[str, str]:
    """Build the positive environment for an accepted scheduler capsule.

    Provider routing is reconstructed from the sealed scheduler config inside
    the capsule.  A Quack coordinator additionally needs its already-resolved
    state credential and closed mutation binding; those values remain in the
    trusted process environment and never enter argv, the capsule, or receipts.
    """

    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    primary_executable = str(
        provider.get(ORDERED_PRIMARY_EXECUTABLE_FIELD) or ""
    ).strip()
    provider_path_entries: list[str] = []
    if primary_executable:
        primary_path = Path(primary_executable)
        if (
            not primary_path.is_absolute()
            or not primary_path.is_file()
            or not os.access(primary_path, os.X_OK)
        ):
            raise ConfiguredBoardError(
                "configured-board primary provider executable is unavailable"
            )
        provider_path_entries.append(str(primary_path.parent))
    for system_entry in ("/usr/local/bin", "/usr/bin", "/bin"):
        if system_entry not in provider_path_entries:
            provider_path_entries.append(system_entry)
    environment = {
        "IPFS_DATASETS_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY": "disabled",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.pathsep.join(provider_path_entries),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.environ.get("TZ"):
        environment["TZ"] = os.environ["TZ"]
    if (extension_directory is None) is not (extension_set_pin is None):
        raise ConfiguredBoardError(
            "configured-board extension projection and exact set pin must "
            "be supplied together"
        )
    if extension_directory is not None and extension_set_pin is not None:
        parsed_extension_set = parse_configured_board_extension_set_pin(
            extension_set_pin.as_dict()
        )
        resolved_extension_directory = extension_directory.resolve(strict=True)
        if (
            not resolved_extension_directory.is_dir()
            or resolved_extension_directory.name != "extensions"
            or resolved_extension_directory.parent.name != ".duckdb"
        ):
            raise ConfiguredBoardError(
                "configured-board extension projection directory is invalid"
            )
        try:
            verified_home = verify_configured_board_extension_set_home(
                parsed_extension_set.pins,
                resolved_extension_directory.parent.parent,
            )
        except (OSError, ValueError) as exc:
            raise ConfiguredBoardError(
                "configured-board exact extension set projection is invalid"
            ) from exc
        if verified_home / ".duckdb/extensions" != resolved_extension_directory:
            raise ConfiguredBoardError(
                "configured-board exact extension set directory drifted"
            )
        environment[CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV] = str(
            resolved_extension_directory
        )
        environment[CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV] = (
            parsed_extension_set.to_json()
        )
    program = board.database_program
    if program is None or program.authority_mode != AUTHORITY_MODE_QUACK:
        return environment
    permitted = set(DATABASE_PROGRAM_ENV_NAMES) - {
        CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
    }
    permitted.add("IPFS_ACCELERATE_AGENT_QUACK_TOKEN")
    handle = str(program.endpoint_secret_handle or "").strip()
    if handle.startswith("env://"):
        target = handle.removeprefix("env://").strip()
        if target:
            permitted.add(target)
    environment.update(
        {
            name: value
            for name, value in os.environ.items()
            if name in permitted
        }
    )
    return environment


def _git_run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    command = ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *argv]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=_sanitized_git_environment(),
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


def _canonical_no_symlink_root(path: Path) -> Path:
    """Validate a lexical absolute repository root without following links."""

    raw = Path(path)
    if not raw.is_absolute() or Path(os.path.abspath(raw)) != raw:
        raise ConfiguredBoardError("repository root is not lexical absolute")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot lstat repository root component: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"repository root component is not a real directory: {current}"
            )
    if raw.resolve(strict=True) != raw:
        raise ConfiguredBoardError("repository root is not canonical")
    return raw


def _lexical_repo_artifact(repo_root: Path, path: Path) -> tuple[Path, str]:
    """Return an exact contained artifact after rejecting linked parents."""

    root = _canonical_no_symlink_root(repo_root)
    artifact = Path(path)
    if not artifact.is_absolute() or Path(os.path.abspath(artifact)) != artifact:
        raise ConfiguredBoardError(f"authority file is not lexical absolute: {artifact}")
    try:
        relative_path = artifact.relative_to(root)
    except ValueError as exc:
        raise ConfiguredBoardError(
            f"authority file escapes repository: {artifact}"
        ) from exc
    current = root
    for part in relative_path.parts[:-1]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot lstat authority parent: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"authority parent is not a real directory: {current}"
            )
    return artifact, relative_path.as_posix()


def _git_identity(repo_root: Path) -> tuple[str, str]:
    root = _canonical_no_symlink_root(repo_root)
    head = _git_run(("rev-parse", "HEAD"), cwd=root).stdout.strip()
    tree = _git_run(("rev-parse", "HEAD^{tree}"), cwd=root).stdout.strip()
    if not head or not tree:
        raise ConfiguredBoardError("cannot bind adaptive execution plan to HEAD and tree")
    return head, tree


def _identity(value: Any) -> str:
    return content_identity(value)


def _tracked_head_snapshot(
    *,
    repo_root: Path,
    path: Path,
    source_head: str,
    max_bytes: int = 4_194_304,
) -> tuple[bytes, str]:
    """Read one stable regular file whose exact bytes equal ``source_head``.

    This joins the filesystem read and Git authority without parsing or
    hashing a second pathname read.  A symlink, hardlink, untracked file,
    staged/unstaged change, or HEAD replacement fails closed.
    """

    root = _canonical_no_symlink_root(repo_root)
    artifact, relative = _lexical_repo_artifact(root, Path(path))
    try:
        payload, _evidence = _read_stable_regular_bytes(
            artifact,
            max_bytes=max_bytes,
        )
    except _StableArtifactReadError as exc:
        raise ConfiguredBoardError(str(exc)) from exc
    if payload is None:
        raise ConfiguredBoardError(f"authority file is absent: {relative}")
    expected = _git_run(
        ("rev-parse", f"{source_head}:{relative}"),
        cwd=root,
    )
    if expected.returncode != 0 or not expected.stdout.strip():
        raise ConfiguredBoardError(
            f"authority file is not tracked at current HEAD: {relative}"
        )
    try:
        actual = subprocess.run(
            ("/usr/bin/git", "-c", "core.hooksPath=/dev/null", "hash-object", "--stdin"),
            cwd=root,
            env=_sanitized_git_environment(),
            input=payload,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfiguredBoardError(
            f"cannot hash authority file: {relative}"
        ) from exc
    try:
        actual_oid = actual.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ConfiguredBoardError(
            f"Git returned an invalid blob identity for {relative}"
        ) from exc
    if actual.returncode != 0 or actual_oid != expected.stdout.strip():
        raise ConfiguredBoardError(
            f"authority file differs from current HEAD: {relative}"
        )
    clean = _git_run(("diff", "--quiet", source_head, "--", relative), cwd=root)
    if clean.returncode != 0:
        raise ConfiguredBoardError(
            f"authority file has staged or unstaged changes: {relative}"
        )
    current_head, _current_tree = _git_identity(root)
    if current_head != source_head:
        raise ConfiguredBoardError("repository HEAD changed during authority snapshot")
    revision = _identity(
        {
            "path": relative,
            "git_blob_oid": actual_oid,
            "bytes_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return payload, revision


def _configured_board_task_records(
    board: "ConfiguredBoard",
    *,
    source_head: str,
    taskboard_bytes: bytes | None = None,
    provider_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Project every current board row into canonical readiness records."""

    path = board.path(board.taskboard_path)
    if taskboard_bytes is None:
        taskboard_bytes, _revision = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=path,
            source_head=source_head,
        )
    try:
        text = taskboard_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfiguredBoardError("taskboard is not valid UTF-8") from exc
    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    if provider_id is None:
        provider_id = str(
            provider.get("primary_provider_id")
            or provider.get("provider_id")
            or ""
        ).strip()
    records: list[dict[str, Any]] = []
    for task_id, title, _line, fields in parse_todo_blocks(
        text,
        task_header_prefix=board.task_header_prefix,
    ):
        outputs = tuple(split_csv(fields.get("outputs", "")))
        predicted = tuple(
            split_csv(fields.get("predicted_files", "") or fields.get("files", ""))
        )
        task_identity = canonical_task_identity(
            {
                "task_id": task_id,
                "title": title,
                "outputs": outputs,
                "acceptance": str(fields.get("acceptance") or ""),
                "metadata": fields,
            },
            board_namespace=board.board_namespace,
            source_path=path,
        )
        records.append(
            {
                "task_id": task_id,
                "canonical_task_cid": task_identity.canonical_task_cid,
                "status": (
                    str(fields.get("status") or "todo").strip().lower()
                    if str(fields.get("is_schedulable") or "true").strip().lower()
                    in {"1", "true", "yes"}
                    else "blocked"
                ),
                "depends_on": tuple(split_csv(fields.get("depends_on", ""))),
                "outputs": outputs,
                "predicted_files": predicted,
                "validation_commands": tuple(
                    split_validation_commands(str(fields.get("validation") or ""))
                ),
                "priority": str(fields.get("priority") or "P2"),
                "resource_class": str(fields.get("resource_class") or "cpu-small"),
                "provider_id": provider_id,
                "exclusive_group": str(fields.get("exclusive_group") or ""),
                "interfaces": tuple(split_csv(fields.get("interfaces", ""))),
                "submodules": tuple(split_csv(fields.get("submodules", ""))),
                "expected_base_revision": source_head,
                "expected_merge_target": board.merge_target_branch,
                "lease_duration_ms": max(
                    60_000,
                    int(float(board.payload["implementation_timeout_seconds"]) * 1000),
                ),
            }
        )
    return tuple(records)


def _configured_board_task_state_snapshots(
    board: "ConfiguredBoard",
) -> tuple[Mapping[str, Any], ...]:
    """Load bounded canonical daemon state projections for attempt fencing."""

    state_root = board.path(board.runtime_paths["state"])
    try:
        root_stat = os.lstat(state_root)
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ConfiguredBoardError(
            f"task-state projection root is unreadable: {state_root}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ConfiguredBoardError(
            f"task-state projection root is not a real directory: {state_root}"
        )
    pending = [state_root]
    discovered: list[Path] = []
    scanned_entries = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = tuple(sorted(iterator, key=lambda item: item.name))
        except OSError as exc:
            raise ConfiguredBoardError(
                f"task-state projection directory is unreadable: {directory}"
            ) from exc
        for entry in entries:
            scanned_entries += 1
            if scanned_entries > 1_024:
                raise ConfiguredBoardError(
                    "task-state projection tree exceeds traversal bound"
                )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"task-state projection entry is unreadable: {entry.path}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ConfiguredBoardError(
                    f"task-state projection entry is a symbolic link: {entry.path}"
                )
            entry_path = Path(entry.path)
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(entry_path)
            elif (
                stat.S_ISREG(metadata.st_mode)
                and entry.name.endswith("_task_state.json")
            ):
                discovered.append(entry_path)
    paths = tuple(sorted(discovered))
    if len(paths) > 128:
        raise ConfiguredBoardError("task-state projection population exceeds bound")
    snapshots: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            path.relative_to(state_root)
            _lexical_repo_artifact(board.repo_root, path)
            payload, _identity = _read_stable_regular_json(path)
            if payload is None:
                raise _StableArtifactReadError(
                    f"task-state projection disappeared: {path}"
                )
        except (OSError, ValueError, _StableArtifactReadError) as exc:
            raise ConfiguredBoardError(
                f"task-state projection is unreadable: {path}"
            ) from exc
        snapshots.append(dict(payload))
    return tuple(snapshots)


def _configured_board_task_population(
    board: "ConfiguredBoard",
    *,
    source_head: str,
    taskboard_bytes: bytes | None = None,
    provider_id: str | None = None,
    task_state_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> _ConfiguredBoardTaskPopulation:
    """Return the exact dependency-ready, retry-admissible current population."""

    records = _configured_board_task_records(
        board,
        source_head=source_head,
        taskboard_bytes=taskboard_bytes,
        provider_id=provider_id,
    )
    task_ids = [str(item["task_id"]) for item in records]
    if len(task_ids) != len(set(task_ids)):
        raise ConfiguredBoardError("taskboard contains duplicate task IDs")
    completed = tuple(
        sorted(
            str(item["task_id"])
            for item in records
            if str(item.get("status") or "").lower()
            in {"complete", "completed", "done"}
        )
    )
    statuses = recompute_readiness_statuses(
        records,
        completed_ids=completed,
    )
    snapshots = tuple(
        _configured_board_task_state_snapshots(board)
        if task_state_snapshots is None
        else (dict(item) for item in task_state_snapshots)
    )
    current_cid_by_id = {
        str(item["task_id"]): str(item["canonical_task_cid"])
        for item in records
    }
    legacy_attempts_by_id: dict[str, int] = {}
    attempts_by_task_revision: dict[tuple[str, str], int] = {}
    attempts_by_cid: dict[str, int] = {}

    def attempt_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfiguredBoardError("task-state attempt count is invalid")
        return value

    for snapshot in snapshots:
        raw_by_id = snapshot.get("implementation_attempts")
        raw_by_cid = snapshot.get("implementation_attempts_by_cid")
        raw_task_identities = snapshot.get("task_identities")
        if raw_by_id not in (None, {}) and not isinstance(raw_by_id, Mapping):
            raise ConfiguredBoardError("task-state implementation_attempts is invalid")
        if raw_by_cid not in (None, {}) and not isinstance(raw_by_cid, Mapping):
            raise ConfiguredBoardError(
                "task-state implementation_attempts_by_cid is invalid"
            )
        if raw_task_identities not in (None, {}) and not isinstance(
            raw_task_identities, Mapping
        ):
            raise ConfiguredBoardError("task-state task_identities is invalid")

        snapshot_attempts_by_cid = {
            str(key): attempt_count(value)
            for key, value in dict(raw_by_cid or {}).items()
        }
        for canonical_task_cid, count in snapshot_attempts_by_cid.items():
            attempts_by_cid[canonical_task_cid] = max(
                attempts_by_cid.get(canonical_task_cid, 0), count
            )

        identity_cid_by_id: dict[str, str] = {}
        for key, value in dict(raw_task_identities or {}).items():
            display_task_id = str(key)
            if not isinstance(value, Mapping):
                raise ConfiguredBoardError("task-state task identity is invalid")
            identity_display_task_id = value.get("display_task_id")
            if identity_display_task_id not in (None, "") and (
                not isinstance(identity_display_task_id, str)
                or identity_display_task_id.strip() != display_task_id
            ):
                raise ConfiguredBoardError(
                    "task-state task identity display ID is invalid"
                )
            identity_cid = value.get("canonical_task_cid")
            if identity_cid in (None, ""):
                # Older projections carried provenance without a canonical
                # identity.  Their display-ID counter remains a conservative
                # retry limit for every later revision of the same task ID.
                continue
            if (
                not isinstance(identity_cid, str)
                or not identity_cid.strip()
                or identity_cid != identity_cid.strip()
            ):
                raise ConfiguredBoardError(
                    "task-state canonical task identity is invalid"
                )
            identity_cid_by_id[display_task_id] = identity_cid

        for key, value in dict(raw_by_id or {}).items():
            display_task_id = str(key)
            count = attempt_count(value)
            identity_cid = identity_cid_by_id.get(display_task_id)
            if not identity_cid:
                legacy_attempts_by_id[display_task_id] = max(
                    legacy_attempts_by_id.get(display_task_id, 0), count
                )
                continue
            current_cid = current_cid_by_id.get(display_task_id)
            if (
                current_cid
                and identity_cid != current_cid
                and snapshot_attempts_by_cid.get(identity_cid, 0) < count
            ):
                raise ConfiguredBoardError(
                    "task-state mismatched task identity is not backed by "
                    "its canonical attempt ledger"
                )
            revision_key = (display_task_id, identity_cid)
            attempts_by_task_revision[revision_key] = max(
                attempts_by_task_revision.get(revision_key, 0), count
            )
    max_attempts = int(board.payload["max_task_attempts"])
    attempt_limited: set[str] = set()
    ready: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record["task_id"])
        task_cid = str(record["canonical_task_cid"])
        if statuses.get(task_id) != "ready":
            continue
        attempt_count = max(
            legacy_attempts_by_id.get(task_id, 0),
            attempts_by_task_revision.get((task_id, task_cid), 0),
            attempts_by_cid.get(task_cid, 0),
        )
        if max_attempts > 0 and attempt_count >= max_attempts:
            attempt_limited.add(task_id)
            continue
        ready.append(record)
    state_snapshot_id = _identity(
        {
            "statuses": statuses,
            "implementation_attempts": legacy_attempts_by_id,
            "implementation_attempts_by_task_revision": [
                {
                    "task_id": task_id,
                    "canonical_task_cid": canonical_task_cid,
                    "attempts": count,
                }
                for (task_id, canonical_task_cid), count in sorted(
                    attempts_by_task_revision.items()
                )
            ],
            "implementation_attempts_by_cid": attempts_by_cid,
        }
    )
    return _ConfiguredBoardTaskPopulation(
        all_records=records,
        ready_records=tuple(ready),
        completed_task_ids=completed,
        attempt_limited_task_ids=tuple(sorted(attempt_limited)),
        state_snapshot_id=state_snapshot_id,
    )


def _plan_authority_roots(
    board: "ConfiguredBoard",
    *,
    head: str,
    tree: str,
    task_source_revision: str,
    task_population: _ConfiguredBoardTaskPopulation,
    route_capacity_profile_id: str = "",
) -> PlanAuthorityRoots:
    source = {
        "board_namespace": board.board_namespace,
        "taskboard_path": board.taskboard_path,
        "task_source_revision": task_source_revision,
    }
    return PlanAuthorityRoots(
        repository_id=_slug(board.board_namespace),
        repository_root_cid=_identity({"head": head, "tree": tree}),
        dirty_worktree_root=_identity({"tree": tree}),
        task_source_id=_identity(source),
        task_source_revision=task_source_revision,
        policy_root=_identity({"protected_paths": board.protected_paths}),
        intent_ir_root=_identity({"plan_path": board.plan_path}),
        legal_ir_root=_identity({"board_namespace": board.board_namespace}),
        security_ir_root=_identity({"protected_paths": board.protected_paths}),
        program_root=_identity(
            {
                "task_ids": [
                    item["task_id"] for item in task_population.all_records
                ],
                "ready_task_ids": [
                    item["task_id"] for item in task_population.ready_records
                ],
                "attempt_limited_task_ids": list(
                    task_population.attempt_limited_task_ids
                ),
                "state_snapshot_id": task_population.state_snapshot_id,
            }
        ),
        capability_catalog_root=_identity({"submodules": board.worktree_submodule_paths}),
        provider_catalog_root=_identity(
            {
                "provider": dict(board.payload.get("provider") or {}),
                "route_capacity_profile_id": route_capacity_profile_id,
            }
        ),
        usage_policy_root=_identity({"max_lanes": board.max_lanes}),
        configuration_root=board.configuration_root,
    )


def configured_board_capacity_observation(
    board: "ConfiguredBoard",
    *,
    now_ms: int | None = None,
    host_capacity_snapshot: Mapping[str, Any] | None = None,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], int]:
    """Return fresh host/provider evidence for compile and pre-claim gates.

    Configuration limits remain ceilings only.  In production the provider
    records always come from the authenticated readiness/process monitor; the
    optional records exist for deterministic contract tests.
    """

    if now_ms is None:
        # Freshness is measured only against this process's trusted local
        # clock.  Provider observations are evidence, never clock authority;
        # in particular a future-dated record must not advance its own
        # freshness boundary.
        current_ms = int(time.time() * 1000)
    elif isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms <= 0:
        raise ConfiguredBoardError("capacity observation time is invalid")
    else:
        current_ms = now_ms
    host = dict(
        host_capacity_snapshot
        or sample_host_resources(
            board.repo_root,
            worker_limit=board.max_lanes,
            active_phase="execution",
        ).to_dict()
    )
    provider_payload = board.payload.get("provider")
    provider_payload = (
        provider_payload if isinstance(provider_payload, Mapping) else {}
    )
    provider_max_age_ms = max(
        5_000,
        int(float(board.payload["poll_interval_seconds"]) * 3_000),
    )
    if provider_capacity_snapshots is None:
        configured_concurrency = int(
            provider_payload.get("max_concurrency") or 1
        )
        monitor = ProviderCapacityMonitor(
            ProviderCapacityMonitorConfig(
                snapshot_path=(
                    board.path(board.runtime_paths["state"])
                    / "provider-capacity.json"
                ),
                max_age_ms=provider_max_age_ms,
                interval_seconds=min(
                    float(board.payload["poll_interval_seconds"]),
                    provider_max_age_ms / 2_000,
                ),
                grok_max_concurrency=configured_concurrency,
                codex_max_concurrency=configured_concurrency,
                grok_request_budget=configured_concurrency,
                codex_request_budget=configured_concurrency,
                grok_token_budget=(
                    configured_concurrency
                    * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
                ),
                codex_token_budget=(
                    configured_concurrency
                    * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
                ),
            )
        )
        sampled, _diagnostics = monitor.sample()
        providers = tuple(dict(item.to_dict()) for item in sampled)
    else:
        providers = tuple(dict(item) for item in provider_capacity_snapshots)
    if not providers:
        raise ConfiguredBoardError("fresh provider capacity evidence is required")
    return host, providers, current_ms


def configured_board_route_capacity_projection(
    board: "ConfiguredBoard",
    *,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]],
    now_ms: int,
) -> tuple[dict[str, Any], AgentImplementationRoutePlan]:
    """Return the router-owned logical provider snapshot for the sealed route.

    The scheduler deliberately supplies unclassified monitor observations and
    retains the router DTO unchanged.  In particular, the fallback lane's
    capacity is never interpreted here as dispatch authority.
    """

    if not _plan_bound_profile(board):
        raise ConfiguredBoardError(
            "logical route capacity projection requires the sealed v3 profile"
        )
    provider = board.payload.get("provider")
    if not isinstance(provider, Mapping):
        raise ConfiguredBoardError("sealed v3 provider configuration is absent")
    route = _resolved_ordered_provider_route(
        provider,
        repo_root=board.repo_root,
        board_namespace=board.board_namespace,
    )
    max_age_ms = max(
        5_000,
        int(float(board.payload["poll_interval_seconds"]) * 3_000),
    )
    try:
        profile = project_agent_implementation_route_capacity(
            route,
            observations=[dict(item) for item in provider_capacity_snapshots],
            now_ms=now_ms,
            max_age_ms=max_age_ms,
        )
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "router rejected provider capacity observations"
        ) from exc
    snapshot = profile.as_compiler_snapshot()
    if (
        not isinstance(snapshot, dict)
        or snapshot != profile.as_dict()
        or snapshot.get("provider_id") != route.route_id
        or snapshot.get("route_id") != route.route_id
        or not isinstance(snapshot.get("profile_id"), str)
        or not snapshot["profile_id"]
    ):
        raise ConfiguredBoardError(
            "router returned a noncanonical logical capacity snapshot"
        )
    return dict(snapshot), route


def materialize_configured_board_execution_plan(
    board: "ConfiguredBoard",
    *,
    now_ms: int | None = None,
    host_capacity_snapshot: Mapping[str, Any] | None = None,
    provider_capacity_snapshots: Sequence[Mapping[str, Any]] | None = None,
    task_state_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> ParallelismDecisionReceipt | None:
    """Compile and atomically publish one exact v3 wave before child launch."""

    if not _plan_bound_profile(board):
        raise ConfiguredBoardError("adaptive plan materialization requires the sealed v3 profile")
    head, tree = _git_identity(board.repo_root)
    current_board = load_configured_board(
        board.config_path,
        repo_root=board.repo_root,
    )
    if (
        current_board.configuration_root != board.configuration_root
        or current_board.board_namespace != board.board_namespace
    ):
        raise ConfiguredBoardError(
            "scheduler configuration changed before wave materialization"
        )
    board = current_board
    config_bytes, _config_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.config_path,
        source_head=head,
    )
    if _identity(
        {"bytes_sha256": hashlib.sha256(config_bytes).hexdigest()}
    ) != board.configuration_root:
        raise ConfiguredBoardError(
            "tracked scheduler config root differs from parsed configuration"
        )
    taskboard_bytes, task_source_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.path(board.taskboard_path),
        source_head=head,
    )
    host, provider_observations, current_ms = configured_board_capacity_observation(
        board,
        now_ms=now_ms,
        host_capacity_snapshot=host_capacity_snapshot,
        provider_capacity_snapshots=provider_capacity_snapshots,
    )
    route_capacity, route = configured_board_route_capacity_projection(
        board,
        provider_capacity_snapshots=provider_observations,
        now_ms=current_ms,
    )
    task_population = _configured_board_task_population(
        board,
        source_head=head,
        taskboard_bytes=taskboard_bytes,
        provider_id=route.route_id,
        task_state_snapshots=task_state_snapshots,
    )
    records = task_population.ready_records
    if not records:
        return None
    roots = _plan_authority_roots(
        board,
        head=head,
        tree=tree,
        task_source_revision=task_source_revision,
        task_population=task_population,
        route_capacity_profile_id=str(route_capacity["profile_id"]),
    )
    providers = (route_capacity,)
    capacity = {
        **host,
        "host": host,
        "providers": list(providers),
        "provider_observations": [
            dict(item) for item in provider_observations
        ],
        "route_capacity_profile_id": route_capacity["profile_id"],
    }
    store_root = board.path(board.runtime_paths["state"]) / "plan-revision-store"
    store = PlanRevisionStore(store_root)
    adapter = ProductionParallelPlanAdapter(store)
    active = None
    prior_revision = None
    scope_drift_leases: list[tuple[str, Any]] = []
    denied_wave_barrier: tuple[str, Any] | None = None
    denied_wave_dispositions: list[tuple[str, Any]] = []
    try:
        with store._thread_lock:  # noqa: SLF001
            with store._guard():  # noqa: SLF001
                active = _secure_store_active(store)
                if active is not None:
                    stored_revision = _secure_store_cas(
                        store, active.revision_cid
                    )
                    prior_revision = PlanRevision.from_dict(stored_revision)
                    if prior_revision.to_dict() != stored_revision:
                        raise ExecutionPlanError(
                            "active plan revision changed during typed decode"
                        )
                    manifest_payload = _secure_store_cas(
                        store,
                        prior_revision.materialization_transaction_cid,
                    )
                    prior_manifest = ConfiguredBoardExecutionSlices.from_dict(
                        manifest_payload
                    )
                    if prior_manifest.to_dict() != manifest_payload:
                        raise ExecutionPlanError(
                            "active slice manifest changed during typed decode"
                        )
                    observed_barrier = _load_plan_bound_wave_diff_barrier_locked(
                        store,
                        revision_cid=active.revision_cid,
                        slice_manifest_cid=(
                            prior_revision.materialization_transaction_cid
                        ),
                    )
                    if (
                        observed_barrier is not None
                        and observed_barrier[1].decision != "released"
                    ):
                        denied_wave_barrier = observed_barrier
                        for row in observed_barrier[1].dispositions:
                            disposition = (
                                _load_plan_bound_proposal_disposition_locked(
                                    store,
                                    revision_cid=active.revision_cid,
                                    slice_id=row["slice_id"],
                                )
                            )
                            if (
                                disposition is None
                                or disposition[0] != row["disposition_cid"]
                            ):
                                raise ExecutionPlanError(
                                    "denied wave lost proposal disposition evidence"
                                )
                            denied_wave_dispositions.append(disposition)
                    for execution_slice in prior_manifest.slices:
                        reassignment = adapter._load_slice_reassignment_locked(  # noqa: SLF001
                            revision_cid=active.revision_cid,
                            slice_id=execution_slice.slice_id,
                        )
                        owner_lane_id = (
                            reassignment[1].recipient_lane_id
                            if reassignment is not None
                            else execution_slice.lane_id
                        )
                        execution_lease = (
                            _load_plan_bound_execution_lease_locked(
                                store,
                                revision_cid=active.revision_cid,
                                slice_id=execution_slice.slice_id,
                                lane_id=owner_lane_id,
                            )
                        )
                        if (
                            execution_lease is not None
                            and execution_lease[1].phase == "scope_drift"
                        ):
                            scope_drift_leases.append(execution_lease)
    except ExecutionPlanError as exc:
        raise ConfiguredBoardError(
            "cannot securely adopt the active plan revision"
        ) from exc

    prior_conflict_cid = (
        prior_revision.conflict_contract.conflict_surface_cid
        if prior_revision is not None
        else ""
    )
    observed_scope_paths = {
        path
        for _lease_cid, lease in scope_drift_leases
        for path in lease.actual_changed_paths
    }
    observed_scope_paths.update(
        path
        for _disposition_cid, disposition in denied_wave_dispositions
        for path in disposition.actual_changed_paths
    )
    if prior_conflict_cid and prior_revision is not None:
        observed_scope_paths.update(
            prior_revision.conflict_contract.predicted_files
        )
    scope_drift_evidence_cid = prior_conflict_cid
    if scope_drift_leases or denied_wave_barrier is not None:
        scope_drift_evidence_cid = _identity(
            {
                "kind": "plan-bound-actual-scope-drift",
                "prior_conflict_surface_cid": prior_conflict_cid,
                "wave_barrier_cid": (
                    denied_wave_barrier[0]
                    if denied_wave_barrier is not None
                    else ""
                ),
                "wave_barrier_decision": (
                    denied_wave_barrier[1].decision
                    if denied_wave_barrier is not None
                    else ""
                ),
                "wave_barrier_reason_codes": (
                    list(denied_wave_barrier[1].reason_codes)
                    if denied_wave_barrier is not None
                    else []
                ),
                "proposal_disposition_cids": [
                    disposition_cid
                    for disposition_cid, _disposition in denied_wave_dispositions
                ],
                "execution_lease_cids": [
                    lease_cid for lease_cid, _lease in scope_drift_leases
                ],
                "proposal_receipt_ids": [
                    lease.proposal_receipt_id
                    for _lease_cid, lease in scope_drift_leases
                ],
                "changed_paths": sorted(observed_scope_paths),
                "merge_enqueue_reached": False,
            }
        )
    budget = InvocationBudget(
        max_lanes=1 if scope_drift_evidence_cid else board.max_lanes
    )
    plan_root_cid = _identity(
        {
            "roots": roots.to_dict(),
            "task_cids": [
                record["canonical_task_cid"]
                for record in task_population.all_records
            ],
            "ready_task_cids": [
                record["canonical_task_cid"] for record in records
            ],
            "state_snapshot_id": task_population.state_snapshot_id,
            "capacity_snapshot": capacity,
            "invocation_budget": budget.to_dict(),
            "scope_drift_evidence_cid": scope_drift_evidence_cid,
        }
    )
    observed_board = load_configured_board(
        board.config_path,
        repo_root=board.repo_root,
    )
    observed_taskboard, observed_task_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.path(board.taskboard_path),
        source_head=head,
    )
    observed_config, _observed_config_revision = _tracked_head_snapshot(
        repo_root=board.repo_root,
        path=board.config_path,
        source_head=head,
    )
    if (
        observed_board.configuration_root != board.configuration_root
        or observed_config != config_bytes
        or observed_task_revision != task_source_revision
        or observed_taskboard != taskboard_bytes
        or _git_identity(board.repo_root) != (head, tree)
    ):
        raise ConfiguredBoardError(
            "repository/configuration/task authority changed before publish"
        )
    plan, slices = adapter.compile_wave(
        board_namespace=board.board_namespace,
        plan_root_cid=plan_root_cid,
        tasks=records,
        budget=budget,
        repository_snapshot={
            "tree_id": tree,
            "merge_target": board.merge_target_branch,
            "protected_paths": list(board.protected_paths),
        },
        capacity_snapshot=capacity,
        provider_snapshots=providers,
        completed_task_ids=task_population.completed_task_ids,
        protected_paths=board.protected_paths,
        submodule_paths=board.worktree_submodule_paths,
        post_merge_validation=(
            f"{sys.executable} {board.path(board.validator_path)}",
        ),
        source_head=head,
        task_source_revision=task_source_revision,
        configuration_root=board.configuration_root,
        current_time_ms=current_ms,
    )
    task_cids = tuple(
        record["canonical_task_cid"]
        for record in task_population.all_records
    )
    cid_by_id = {
        str(record["task_id"]): str(record["canonical_task_cid"])
        for record in task_population.all_records
    }
    goal_cid = _identity({"board_namespace": board.board_namespace, "kind": "goal-population"})
    same_wave_adoption = bool(active and active.plan_root_cid == plan_root_cid)
    semantic_revision = (
        active.semantic_revision if same_wave_adoption and active is not None
        else ((active.semantic_revision + 1) if active is not None else 1)
    )
    delta: PlanDelta | None = None
    if active is not None and not same_wave_adoption:
        delta_request_cid = _identity(
            {
                "base_revision_cid": active.revision_cid,
                "next_plan_root_cid": plan_root_cid,
                "capacity_snapshot_id": plan.capacity_snapshot_id,
                "task_source_revision": task_source_revision,
            }
        )
        delta = PlanDelta(
            base_plan_root=active.plan_root_cid,
            base_plan_revision=active.semantic_revision,
            request_cid=delta_request_cid,
            roots=roots,
            items=(
                PlanDeltaItem(
                    item_key="configured-wave-replan",
                    operation=PlanDeltaOperation.ATTACH_EVIDENCE,
                    target_cid=active.plan_root_cid,
                    expected_target_lifecycle=LifecycleState.PROPOSED,
                    expected_target_spec_revision=active.revision_cid,
                    before_digest=active.plan_root_cid,
                    after_record_cid=plan_root_cid,
                    effect_class=DeltaEffectClass.EVIDENCE_ONLY,
                    rationale=(
                        "Recompile from a fresh repository, task-source, "
                        "attempt, and capacity observation."
                    ),
                    provenance={
                        "source_head": head,
                        "task_source_revision": task_source_revision,
                        "capacity_snapshot_id": plan.capacity_snapshot_id,
                        "scope_drift_evidence_cid": scope_drift_evidence_cid,
                    },
                    resource_impact=(plan.capacity_snapshot_id,),
                ),
            ),
        )

    current_completed_cids = {
        cid_by_id[task_id]
        for task_id in task_population.completed_task_ids
        if task_id in cid_by_id
    }
    prior_task_cids = set(
        prior_revision.task_population.member_cids if prior_revision else ()
    )
    prior_completed_cids = set(
        prior_revision.completed_population.member_cids if prior_revision else ()
    )
    prior_claimed_cids = set(
        prior_revision.claimed_population.member_cids if prior_revision else ()
    )
    blocked_cids = {
        str(record["canonical_task_cid"])
        for record in task_population.all_records
        if str(record.get("status") or "").lower()
        in {"blocked", "failed", "quarantined"}
    }
    blocked_cids.update(
        cid_by_id[task_id]
        for task_id in task_population.attempt_limited_task_ids
        if task_id in cid_by_id
    )

    def revision_factory(execution_plan_cid: str, slice_manifest_cid: str) -> PlanRevision:
        if prior_revision is not None and same_wave_adoption:
            return prior_revision
        origin = PlanOrigin.STEER if active is not None else PlanOrigin.CREATE
        return PlanRevision(
            plan_root_cid=plan_root_cid,
            semantic_revision=semantic_revision,
            parent_plan_root=(active.plan_root_cid if active else ""),
            origin=origin,
            roots=roots,
            request_cid=_identity(
                {
                    "budget": budget.to_dict(),
                    "tree": tree,
                    "active_revision_cid": active.revision_cid if active else "",
                }
            ),
            delta_cid=(delta.delta_cid if delta is not None else ""),
            scan_receipt_cid=_identity({"task_source_revision": task_source_revision}),
            query_plan_cid=_identity({"task_ids": [record["task_id"] for record in records]}),
            evidence_bundle_cid=_identity(
                {
                    "config": roots.configuration_root,
                    "scope_drift_evidence_cid": scope_drift_evidence_cid,
                }
            ),
            admission_receipt_cid=_identity({"plan_root_cid": plan_root_cid, "admitted": True}),
            execution_plan_cid=execution_plan_cid,
            goal_population=PlanPopulationDigest(PopulationKind.RETAINED, (goal_cid,)),
            task_population=PlanPopulationDigest(PopulationKind.RETAINED, task_cids),
            added_population=PlanPopulationDigest(
                PopulationKind.ADDED,
                (
                    (goal_cid, *task_cids)
                    if prior_revision is None
                    else tuple(sorted(set(task_cids) - prior_task_cids))
                ),
            ),
            superseded_population=PlanPopulationDigest(PopulationKind.SUPERSEDED),
            retained_population=PlanPopulationDigest(
                PopulationKind.RETAINED,
                tuple(sorted(set(task_cids) & prior_task_cids)),
            ),
            deferred_population=PlanPopulationDigest(PopulationKind.DEFERRED),
            claimed_population=PlanPopulationDigest(
                PopulationKind.CLAIMED,
                tuple(sorted(prior_claimed_cids)),
            ),
            completed_population=PlanPopulationDigest(
                PopulationKind.COMPLETED,
                tuple(sorted(prior_completed_cids | current_completed_cids)),
            ),
            blocked_population=PlanPopulationDigest(
                PopulationKind.BLOCKED,
                tuple(sorted(blocked_cids)),
            ),
            resource_contract=PlanResourceContract(resource_class="process-control"),
            provider_contract=PlanProviderContract(
                provider_requirement=route.route_id
            ),
            lease_contract=PlanLeaseContract(
                lease_duration_ms=max(60_000, int(float(board.payload["implementation_timeout_seconds"]) * 1000)),
                fencing_epoch=semantic_revision,
                heartbeat_interval_ms=max(1, int(float(board.payload["poll_interval_seconds"]) * 1000)),
            ),
            retry_contract=PlanRetryContract(max_retries=int(board.payload["max_task_attempts"])),
            worktree_contract=PlanWorktreeContract(
                policy="isolated",
                expected_base_revision=head,
                expected_merge_target=board.merge_target_branch,
                isolation_required=True,
            ),
            merge_strategy=PlanMergeStrategy(
                kind=MergeStrategyKind.REBASE_THEN_MERGE,
                merge_train_id=f"merge-train:{_slug(board.board_namespace)}",
                post_merge_validation_cids=(_identity({"validator": board.validator_path}),),
            ),
            conflict_contract=PlanConflictContract(
                predicted_files=tuple(
                    sorted(
                        {
                            *observed_scope_paths,
                            *(
                                path
                                for record in records
                                for path in record.get("predicted_files", ())
                            ),
                        }
                    )
                ),
                protected_paths=board.protected_paths,
                conflict_surface_cid=scope_drift_evidence_cid,
            ),
            completion_rule=PlanCompletionRule(authority=CompletionAuthority.VALIDATION_GATE),
            validation_dag=(
                PlanValidationNode(
                    validation_key="configured-board-post-merge",
                    argv=(sys.executable, str(board.path(board.validator_path))),
                ),
            ),
            materialization_transaction_cid=slice_manifest_cid,
            rollback_ref=(
                prior_revision.rollback_ref if prior_revision is not None else head
            ),
            event_cursor=task_source_revision,
        )

    return adapter.publish_wave(
        plan=plan,
        slice_manifest=slices,
        revision_factory=revision_factory,
        observed_roots=roots,
        idempotency_key=f"configured-wave:{plan_root_cid}:{plan.plan_id}",
        delta=delta,
        expected_active_plan_root=(active.plan_root_cid if active else ""),
        expected_active_revision_cid=(active.revision_cid if active else ""),
        base_event_cursor=(active.event_cursor if active else ""),
        fencing_token=semantic_revision,
        lease_id=f"configured-wave:{semantic_revision}:{plan.plan_id}",
    )


def _resolved_ordered_provider_route(
    provider: Mapping[str, Any],
    *,
    repo_root: Path,
    board_namespace: str,
) -> AgentImplementationRoutePlan:
    """Resolve scheduler profile input through the canonical router policy."""

    values = {
        field: _provider_string(provider, field)
        for field in ORDERED_PROVIDER_FIELDS
    }
    authorization = None
    authorization_path = str(
        provider.get(ROUTE_AUTHORIZATION_PATH_FIELD) or ""
    ).strip()
    if authorization_path:
        try:
            authorization = load_agent_implementation_route_authorization(
                repo_root=repo_root,
                artifact_path=authorization_path,
                board_namespace=board_namespace,
            )
        except (OSError, ValueError) as exc:
            raise ConfiguredBoardError(str(exc)) from exc
    try:
        return resolve_agent_implementation_route(
            **values,
            authorization=authorization,
        )
    except ValueError as exc:
        raise ConfiguredBoardError(str(exc)) from exc


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
        raise ConfiguredBoardError(
            f"{field} must be a positive integer"
        ) from exc
    if parsed < 1:
        raise ConfiguredBoardError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            f"{field} must be finite and nonnegative"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ConfiguredBoardError(f"{field} must be finite and nonnegative")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfiguredBoardError(f"{field} must be a nonnegative integer")
    return value


def _objective_refill_controls(
    payload: Mapping[str, Any],
) -> tuple[int, int, int] | None:
    """Return the sealed low-watermark, epoch bound, and cooldown controls."""

    if payload.get("objective_refill_enabled") is not True:
        return None
    refill_policy = payload.get("refill_policy")
    if not isinstance(refill_policy, dict):
        raise ConfiguredBoardError(
            "refill_policy must be an object when objective refill is enabled"
        )
    derived = refill_policy.get("derived_refill")
    if not isinstance(derived, dict):
        raise ConfiguredBoardError(
            "refill_policy.derived_refill must be an object when objective "
            "refill is enabled"
        )
    min_open_tasks = _nonnegative_int(
        derived.get("min_open_tasks"),
        field="refill_policy.derived_refill.min_open_tasks",
    )
    max_findings = _positive_int(
        derived.get("max_tasks_per_epoch"),
        field="refill_policy.derived_refill.max_tasks_per_epoch",
    )
    max_open_tasks = _positive_int(
        derived.get("max_open_tasks"),
        field="refill_policy.derived_refill.max_open_tasks",
    )
    cooldown_seconds = _nonnegative_int(
        derived.get("cooldown_seconds"),
        field="refill_policy.derived_refill.cooldown_seconds",
    )
    if min_open_tasks >= max_open_tasks:
        raise ConfiguredBoardError(
            "refill_policy.derived_refill.min_open_tasks must be below "
            "max_open_tasks"
        )
    return min_open_tasks, max_findings, cooldown_seconds


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
        raise ConfiguredBoardError(
            f"{field} must be a single-line nonempty string"
        )
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
        raise ConfiguredBoardError(
            f"{field} contains unsafe relative path {value!r}"
        )
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
        raise ConfiguredBoardError(
            f"path escapes repository: {relative}"
        ) from exc
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
    configuration_root: str
    configuration_revision: str
    taskboard_path: str
    objectives_path: str
    plan_path: str
    validator_path: str
    dependency_validator_path: str
    dependency_seal_path: str
    task_prefix: str
    board_namespace: str
    merge_target_branch: str
    max_lanes: int
    strict_task_sharding: bool
    idle_lane_work_stealing: str
    worktree_submodule_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    live_capsule_control_paths: tuple[str, ...]
    runtime_paths: Mapping[str, str]
    database_program: DatabaseProgramConfig | None = None

    @property
    def task_header_prefix(self) -> str:
        return _task_header_prefix(self.task_prefix)

    def path(self, relative: str) -> Path:
        return _contained_path(self.repo_root, relative)

    def resolved_database_program(self) -> DatabaseProgramConfig:
        """Return the explicit database/task-source selection for this board.

        Implicit legacy-Markdown defaults are deprecated. When no
        ``database_program`` section is present the board still launches, but
        only after constructing an *explicit* legacy selection from
        ``source_binding.bootstrap_task_source`` or a labeled explicit-legacy
        fallback.
        """

        if self.database_program is not None:
            return self.database_program
        source_binding = self.payload.get("source_binding")
        bootstrap = ""
        if isinstance(source_binding, Mapping):
            bootstrap = str(
                source_binding.get("bootstrap_task_source") or ""
            ).strip().lower()
        if bootstrap in {"", "legacy-markdown", "legacy_markdown", "markdown-legacy"}:
            return DatabaseProgramConfig.explicit_legacy_markdown()
        if bootstrap in {"markdown"}:
            return DatabaseProgramConfig(
                authority_mode=AUTHORITY_MODE_LEGACY_MARKDOWN,
                task_source_kind="markdown",
                explicit_legacy=True,
            )
        if bootstrap in {"duckdb", "quack"}:
            raise ConfiguredBoardError(
                "bootstrap_task_source requires a full database_program "
                f"section when set to {bootstrap!r}"
            )
        raise ConfiguredBoardError(
            f"unsupported bootstrap_task_source: {bootstrap!r}"
        )


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
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "scheduler config must be inside the repository"
        ) from exc
    try:
        config_bytes, _config_evidence = _read_stable_regular_bytes(
            path,
            max_bytes=4_194_304,
        )
        if config_bytes is None:
            raise ConfiguredBoardError("scheduler config is absent")
        configuration_revision = _identity(
            {
                "path": path.resolve(strict=False).relative_to(root).as_posix(),
                "bytes_sha256": hashlib.sha256(config_bytes).hexdigest(),
            }
        )
        payload = json.loads(
            config_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ConfiguredBoardError:
        raise
    except _StableArtifactReadError as exc:
        raise ConfiguredBoardError(str(exc)) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardError(
            f"scheduler config is unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfiguredBoardError("scheduler config root must be an object")
    schema = _required_string(payload, "schema")
    if SCHEDULER_SCHEMA_PATTERN.fullmatch(schema) is None:
        raise ConfiguredBoardError(
            f"unsupported scheduler schema: {schema!r}"
        )

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
    dependency_validator_path = ""
    dependency_seal_path = ""
    if "dependency_validator_path" in payload or "dependency_seal_path" in payload:
        dependency_validator_path = _safe_relative(
            _required_string(payload, "dependency_validator_path"),
            field="dependency_validator_path",
        )
        dependency_seal_path = _safe_relative(
            _required_string(payload, "dependency_seal_path"),
            field="dependency_seal_path",
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
        raise ConfiguredBoardError(
            "lanes must contain exactly max_lanes entries"
        )
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
    idle_lane_work_stealing = str(
        payload.get("idle_lane_work_stealing") or ""
    ).strip().lower()
    if idle_lane_work_stealing not in {"", "virgin-transfer"}:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing must be empty or 'virgin-transfer'"
        )
    if idle_lane_work_stealing and not strict_task_sharding:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing requires strict_task_sharding"
        )
    if idle_lane_work_stealing and max_lanes <= 1:
        raise ConfiguredBoardError(
            "idle_lane_work_stealing requires at least two lanes"
        )
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
        raise ConfiguredBoardError(
            "scheduler config must protect its own source path"
        )
    live_capsule_control_paths: tuple[str, ...] = ()
    if "configured_board_live_capsule" in payload:
        try:
            live_capsule_control_paths = (
                parse_configured_board_live_capsule_policy(
                    payload.get("configured_board_live_capsule")
                )
            )
        except ConfiguredBoardLiveCapsuleError as exc:
            raise ConfiguredBoardError(str(exc)) from exc
        if set(live_capsule_control_paths) != set(protected):
            raise ConfiguredBoardError(
                "configured-board live capsule must bind every protected path"
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
    ordered_provider = any(
        field in provider for field in ORDERED_PROVIDER_DETECTION_FIELDS
    )
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
                "provider.primary_model_id must be 'grok-4.6' for "
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
                "provider.fallback_trigger must be "
                "'primary_quota_exhausted' or "
                "'primary_quota_or_auth_unavailable' for the ordered "
                "provider contract"
            )
        if fallback_reasoning_effort not in ORDERED_FALLBACK_REASONING_EFFORTS:
            raise ConfiguredBoardError(
                "provider.fallback_reasoning_effort must be one of "
                "'medium', 'high' for "
                "the ordered provider contract"
            )
        if "provider_id" in provider or "model_id" in provider:
            raise ConfiguredBoardError(
                "ordered provider fields cannot be mixed with legacy "
                "provider_id/model_id"
            )
        _resolved_ordered_provider_route(
            provider,
            repo_root=root,
            board_namespace=board_namespace,
        )
        primary_executable = _optional_provider_string(
            provider,
            ORDERED_PRIMARY_EXECUTABLE_FIELD,
        )
        if primary_executable:
            executable_path = Path(primary_executable)
            if (
                not executable_path.is_absolute()
                or os.path.abspath(primary_executable) != primary_executable
            ):
                raise ConfiguredBoardError(
                    "provider.primary_executable must be a normalized "
                    "absolute path"
                )
            if not executable_path.is_file() or not os.access(
                executable_path,
                os.X_OK,
            ):
                raise ConfiguredBoardError(
                    "provider.primary_executable must name an executable file"
                )
    else:
        provider_id = _optional_provider_string(
            provider,
            "provider_id",
        ).lower()
        _optional_provider_string(provider, "model_id")
        if provider_id and re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*",
            provider_id,
        ) is None:
            raise ConfiguredBoardError(
                "provider.provider_id is not a supported identifier"
            )
    concurrency = _positive_int(
        provider.get("max_concurrency"),
        field="provider.max_concurrency",
    )
    if concurrency < max_lanes:
        raise ConfiguredBoardError(
            "provider.max_concurrency is lower than max_lanes"
        )
    for field in (
        "strict_task_sharding",
        "exit_when_all_tracks_terminal",
        "objective_refill_enabled",
        "codebase_refill_enabled",
    ):
        if not isinstance(payload.get(field), bool):
            raise ConfiguredBoardError(f"{field} must be boolean")
    if (
        "objective_goal_refinement_enabled" in payload
        and not isinstance(payload.get("objective_goal_refinement_enabled"), bool)
    ):
        raise ConfiguredBoardError(
            "objective_goal_refinement_enabled must be boolean"
        )
    for field in (
        "retry_budget_guardrail_enabled",
        "dependency_guardrail_enabled",
        "reconciliation_guardrail_enabled",
    ):
        if field in payload and not isinstance(payload.get(field), bool):
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

    database_program: DatabaseProgramConfig | None = None
    if "database_program" in payload:
        raw_program = payload.get("database_program")
        if not isinstance(raw_program, dict):
            raise ConfiguredBoardError("database_program must be an object")
        try:
            program_payload = dict(raw_program)
            if not program_payload.get("worktree_root"):
                program_payload["worktree_root"] = runtime_paths["worktrees"]
            database_program = parse_database_program_config(program_payload)
        except DatabaseProgramConfigError as exc:
            raise ConfiguredBoardError(str(exc)) from exc

    _objective_refill_controls(payload)

    return ConfiguredBoard(
        config_path=path,
        repo_root=root,
        payload=payload,
        configuration_root=_identity(
            {"bytes_sha256": hashlib.sha256(config_bytes).hexdigest()}
        ),
        configuration_revision=configuration_revision,
        taskboard_path=taskboard_path,
        objectives_path=objectives_path,
        plan_path=plan_path,
        validator_path=validator_path,
        dependency_validator_path=dependency_validator_path,
        dependency_seal_path=dependency_seal_path,
        task_prefix=task_prefix,
        board_namespace=board_namespace,
        merge_target_branch=merge_target_branch,
        max_lanes=max_lanes,
        strict_task_sharding=strict_task_sharding,
        idle_lane_work_stealing=idle_lane_work_stealing,
        worktree_submodule_paths=submodules,
        protected_paths=protected,
        live_capsule_control_paths=live_capsule_control_paths,
        runtime_paths=runtime_paths,
        database_program=database_program,
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
    return _git_run(
        args,
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


def _control_file_is_tracked(
    board: ConfiguredBoard,
    relative: str,
) -> bool:
    """Recognize control files tracked by the outer or an owned submodule."""

    if (
        _git(
            board,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ).returncode
        == 0
    ):
        return True
    relative_path = PurePosixPath(relative)
    for submodule in sorted(
        board.worktree_submodule_paths,
        key=lambda value: len(PurePosixPath(value).parts),
        reverse=True,
    ):
        submodule_path = PurePosixPath(submodule)
        prefix = submodule_path.parts
        if (
            relative_path.parts[: len(prefix)] != prefix
            or len(relative_path.parts) == len(prefix)
        ):
            continue
        if (
            _git(
                board,
                "ls-files",
                "--error-unmatch",
                "--",
                submodule,
            ).returncode
            != 0
        ):
            return False
        nested_root = board.path(submodule)
        inner = PurePosixPath(*relative_path.parts[len(prefix) :]).as_posix()
        return (
            _run(
                ("git", "ls-files", "--error-unmatch", "--", inner),
                cwd=nested_root,
                timeout=60.0,
            ).returncode
            == 0
        )
    return False


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
    if board.dependency_validator_path:
        required_files.update(
            {board.dependency_validator_path, board.dependency_seal_path}
        )
    missing_files = sorted(
        relative
        for relative in required_files
        if not board.path(relative).is_file()
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
        if _control_file_is_tracked(board, relative)
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
                validator.returncode == 0
                and validator_report.get("valid") is True
            ),
            detail={
                "returncode": validator.returncode,
                "stderr": validator.stderr[-2000:],
                "errors": validator_report.get("errors"),
            },
        )

    if (
        board.dependency_validator_path
        and board.path(board.dependency_validator_path).is_file()
    ):
        dependency_validator = _run(
            (
                sys.executable,
                str(board.path(board.dependency_validator_path)),
                "--check-all",
            ),
            cwd=board.repo_root,
        )
        dependency_report: dict[str, Any] = {}
        try:
            parsed_dependency = json.loads(dependency_validator.stdout)
            if isinstance(parsed_dependency, dict):
                dependency_report = parsed_dependency
        except json.JSONDecodeError:
            dependency_report = {}
        _append_check(
            checks,
            errors,
            name="dependency_seal_validator",
            passed=(
                dependency_validator.returncode == 0
                and dependency_report.get("valid") is True
            ),
            detail={
                "returncode": dependency_validator.returncode,
                "stderr": dependency_validator.stderr[-2000:],
                "errors": dependency_report.get("errors"),
                "seal_path": board.dependency_seal_path,
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
        top_level = _run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=target,
            timeout=60,
        ) if target.is_dir() else None
        exact_worktree = bool(
            top_level is not None
            and top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == target.resolve()
        )
        head = _run(
            ("git", "rev-parse", "HEAD"),
            cwd=target,
            timeout=60,
        ) if exact_worktree else None
        clean = _run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=target,
            timeout=60,
        ) if head is not None and head.returncode == 0 else None
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
                    planning_ancestor is not None
                    and planning_ancestor.returncode == 0
                ),
                "dirty": (
                    clean.stdout.splitlines()[:50]
                    if clean is not None
                    else []
                ),
            }
        )
    _append_check(
        checks,
        errors,
        name="configured_submodules",
        passed=all(item["valid"] for item in submodule_checks),
        detail=submodule_checks,
    )

    implementation_entry = board.path(
        IMPLEMENTATION_ENTRY_PATH.as_posix()
    )
    _append_check(
        checks,
        errors,
        name="implementation_entry",
        passed=implementation_entry.is_file(),
        detail=str(implementation_entry),
    )

    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "configured-board-preflight@1"
        ),
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
    objective_refill_controls = _objective_refill_controls(payload)
    program_for_paths = board.resolved_database_program()
    worktree_root = (
        str(board.path(program_for_paths.worktree_root))
        if program_for_paths.worktree_root
        else str(board.path(board.runtime_paths["worktrees"]))
    )
    args: list[str] = [
        "--todo-path",
        str(board.path(board.taskboard_path)),
        "--task-prefix",
        board.task_header_prefix,
        "--worktree-root",
        worktree_root,
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
        "--log-level",
        "INFO",
    ]
    # Explicit database-program selections are supervisor inputs.  The
    # fallback legacy-Markdown program, however, is a daemon-only compatibility
    # projection: implementation_supervisor does not accept the database CLI
    # flags and already launches the daemon with its closed legacy-Markdown
    # default.  Passing those daemon-only flags through the supervisor creates
    # an immediate argparse/restart loop before any task can run.
    program = board.resolved_database_program()
    program_args = program.cli_args() if board.database_program is not None else []
    skip_next = False
    for item in program_args:
        if skip_next:
            skip_next = False
            continue
        if item == "--worktree-root":
            skip_next = True
            continue
        args.append(item)
    if _plan_bound_profile(board):
        # The bounded child re-opens the sealed profile solely to sample live
        # host/provider capacity before the daemon's canonical claim gate.
        args.extend(["--scheduler-config", str(board.config_path)])
    args.append("--implement" if implement else "--no-implement")
    # Legacy profiles retain their configured hash-sharding behavior.  A v3
    # child receives one exact compiler slice below, so hash sharding and its
    # strict fallback policy must both be disabled for that child.
    if board.strict_task_sharding and not _plan_bound_profile(board):
        args.append("--strict-task-sharding")
    if board.idle_lane_work_stealing and not _plan_bound_profile(board):
        args.extend(
            ["--idle-lane-work-stealing", board.idle_lane_work_stealing]
        )
    for relative in board.worktree_submodule_paths:
        args.extend(["--worktree-submodule-path", relative])
    for relative in board.protected_paths:
        args.extend(["--implementation-protected-path", relative])
    if objective_refill_controls is not None:
        min_open_tasks, max_findings, cooldown_seconds = (
            objective_refill_controls
        )
        args.extend(
            [
                "--objective-refill-scan",
                "--objective-path",
                str(board.path(board.objectives_path)),
                "--objective-scan-min-open-tasks",
                str(min_open_tasks),
                "--objective-scan-max-findings",
                str(max_findings),
                "--objective-scan-cooldown-seconds",
                str(cooldown_seconds),
            ]
        )
        if payload.get("objective_goal_refinement_enabled") is False:
            args.append("--no-objective-goal-refinement")
    if payload.get("codebase_refill_enabled") is True:
        args.append("--codebase-refill-scan")
    if payload.get("retry_budget_guardrail_enabled") is False:
        args.append("--no-retry-budget-guardrail")
    if payload.get("dependency_guardrail_enabled") is False:
        args.append("--no-dependency-guardrail")
    if payload.get("reconciliation_guardrail_enabled") is False:
        args.append("--no-reconciliation-guardrail")
    return tuple(args)


def configured_board_launch_plan(
    board: ConfiguredBoard,
    *,
    implement: bool,
    detach: bool,
    duration_seconds: float = float("inf"),
    stamp: str | None = None,
    parallelism_receipt: ParallelismDecisionReceipt | None = None,
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
    configured_board_live_admission: (
        ConfiguredBoardLiveCapsuleAdmission | None
    ) = None,
) -> dict[str, Any]:
    """Render the exact existing multi-supervisor runner invocation."""

    run_stamp = stamp or utc_run_stamp()
    runtime_root = board.path(board.runtime_paths["root"])
    state_dir = board.path(board.runtime_paths["state"])
    state_relative = Path(board.runtime_paths["state"])
    log_dir = board.path(board.runtime_paths["logs"])
    entry = board.path(IMPLEMENTATION_ENTRY_PATH.as_posix())
    program = board.resolved_database_program()
    plan_bound = _plan_bound_profile(board)
    live_admission = configured_board_live_admission
    if live_admission is not None:
        if not board.live_capsule_control_paths:
            raise ConfiguredBoardError(
                "configured-board live admission lacks a required board policy"
            )
        if accepted_control_plane_pin is None:
            raise ConfiguredBoardError(
                "configured-board live admission lacks its control-plane capsule"
            )
        if native_dependency_launch is None:
            raise ConfiguredBoardError(
                "configured-board live admission lacks its native dependency"
            )
        try:
            live_admission = verify_configured_board_live_capsule(
                live_admission,
                control_plane_pin=accepted_control_plane_pin,
                control_plane_descriptor=accepted_control_plane_descriptor,
                native_dependency_launch=native_dependency_launch,
                repo_root=board.repo_root,
                expected_board_namespace=board.board_namespace,
                expected_config_path=(
                    board.config_path.relative_to(board.repo_root).as_posix()
                ),
            )
        except (OSError, ConfiguredBoardLiveCapsuleError, ValueError) as exc:
            raise ConfiguredBoardError(
                "configured-board live admission is invalid"
            ) from exc
        expected_database_authority = {
            "authority_mode": program.authority_mode,
            "task_source_kind": program.task_source_kind,
            "schema_revision": program.schema_revision,
            "failover_policy": program.failover_policy,
            "store_id": program.store_id,
            "store_generation": int(program.store_generation),
            "endpoint_secret_handle": program.endpoint_secret_handle,
        }
        if (
            live_admission.configuration_root != board.configuration_root
            or live_admission.plan_revision
            != str(board.payload.get("plan_revision") or "")
            or live_admission.task_prefix != board.task_prefix
            or live_admission.max_lanes != board.max_lanes
            or live_admission.strict_task_sharding
            is not board.strict_task_sharding
            or dict(live_admission.database_authority)
            != expected_database_authority
            or tuple(
                str(item["path"])
                for item in live_admission.control_artifacts
            )
            != board.live_capsule_control_paths
        ):
            raise ConfiguredBoardError(
                "configured-board live admission differs from board authority"
            )
    plan_bound_children: tuple[PlanBoundSupervisorChild, ...] = ()
    implementation_tracks: tuple[ImplementationSupervisorTrackConfig, ...] = ()
    if plan_bound and parallelism_receipt is not None:
        binding = parallelism_receipt.binding
        manifest = parallelism_receipt.slice_manifest
        plan_bound_children = tuple(
            PlanBoundSupervisorChild(
                name=f"{board.board_namespace}-lane-{execution_slice.lane_index}",
                accepted_tree_root=board.repo_root,
                script_path=IMPLEMENTATION_ENTRY_PATH.as_posix(),
                state_dir=(state_relative / f"lane-{execution_slice.lane_index}"),
                state_prefix=(
                    f"{_slug(board.task_prefix)}_lane_"
                    f"{execution_slice.lane_index}"
                ),
                plan_revision_store_path=(state_relative / "plan-revision-store"),
                revision_cid=binding.revision_cid,
                plan_root_cid=binding.plan_root_cid,
                execution_plan_cid=binding.execution_plan_cid,
                capacity_snapshot_id=binding.capacity_snapshot_id,
                slice_manifest_cid=parallelism_receipt.slice_manifest_cid,
                slice_id=execution_slice.slice_id,
                source_head=manifest.source_head,
                source_tree=manifest.repository_tree_id,
                task_source_revision=manifest.task_source_revision,
                configuration_root=manifest.configuration_root,
                lane_id=execution_slice.lane_id,
                task_ids=execution_slice.task_ids,
                task_cids=execution_slice.task_cids,
            )
            for execution_slice in manifest.nonempty
        )
    elif not plan_bound:
        implementation_tracks = (
            ImplementationSupervisorTrackConfig(
                name=board.board_namespace,
                script_path=entry,
                state_dir=state_dir,
                state_prefix=_slug(board.task_prefix),
                database_program=board.database_program,
            ),
        )
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
        master_pid_path=(
            state_dir / "configured-board-wave.pid"
            if plan_bound
            else state_dir / "configured-board-master.pid"
        ),
        label=board.board_namespace,
        python_executable=sys.executable,
        implementation_track_configs=implementation_tracks,
        plan_bound_tracks=plan_bound_children,
        common_args=configured_board_common_args(
            board,
            implement=implement,
        ),
        detach=(detach and not plan_bound),
        database_program=board.database_program,
    )
    runner_args = runner.args()
    if plan_bound:
        # An empty first wave is an explicit bounded success.  The reusable
        # runner accepts this marker without constructing or starting a child.
        if "--plan-bound-wave" not in runner_args:
            runner_args.append("--plan-bound-wave")
    else:
        runner_args.extend(
            [
                "--implementation-supervisor-lanes-per-track",
                str(board.max_lanes),
            ]
        )
        if live_admission is not None:
            runner_args.extend(
                [
                    "--require-configured-board-live-capsule",
                    "--configured-board-live-admission-json",
                    live_admission.to_json(),
                    "--configured-board-live-native-launch-json",
                    native_dependency_launch.to_json(),
                    "--configured-board-live-native-fd",
                    str(native_dependency_launch.descriptor.descriptor),
                ]
            )
    if accepted_control_plane_pin is not None:
        verify_agent_implementation_sealed_control_plane(
            accepted_control_plane_pin,
            accepted_control_plane_descriptor,
        )
        expected_generation = (
            (
                parallelism_receipt.slice_manifest.source_head,
                parallelism_receipt.slice_manifest.repository_tree_id,
            )
            if parallelism_receipt is not None
            else _git_identity(board.repo_root)
        )
        if (
            accepted_control_plane_pin.source_head,
            accepted_control_plane_pin.source_tree,
        ) != expected_generation:
            raise ConfiguredBoardError(
                "accepted control-plane generation differs from the launch"
            )
        runner_args.extend(
            [
                "--accepted-control-plane-pin-json",
                accepted_control_plane_pin_json(accepted_control_plane_pin),
                "--accepted-control-plane-fd",
                str(accepted_control_plane_descriptor),
            ]
        )
    if board.strict_task_sharding and not plan_bound:
        runner_args.append(
            "--implementation-supervisor-strict-task-sharding"
        )
    if board.idle_lane_work_stealing and not plan_bound:
        runner_args.extend(
            [
                "--implementation-supervisor-idle-lane-work-stealing",
                board.idle_lane_work_stealing,
            ]
        )
    if plan_bound or board.payload.get("exit_when_all_tracks_terminal") is True:
        runner_args.append("--exit-when-all-tracks-terminal")

    provider = board.payload.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    ordered_provider = any(
        field in provider for field in ORDERED_PROVIDER_DETECTION_FIELDS
    )
    if ordered_provider:
        route_plan = _resolved_ordered_provider_route(
            provider,
            repo_root=board.repo_root,
            board_namespace=board.board_namespace,
        )
        environment = route_plan.as_environment()
        primary_executable = _optional_provider_string(
            provider,
            ORDERED_PRIMARY_EXECUTABLE_FIELD,
        )
        if primary_executable:
            environment[GROK_BIN_ENV] = primary_executable
    else:
        provider_id = str(provider.get("provider_id") or "").strip()
        model_id = str(provider.get("model_id") or "").strip()
        environment = {}
        if provider_id and provider_id != "auto":
            environment[PROVIDER_ENV] = provider_id
        if model_id and provider_id in {"", "auto", "codex", "openai"}:
            environment[CODEX_MODEL_ENV] = model_id
    # Database authority is explicit and non-secret. The endpoint field is an
    # opaque secret handle; raw credentials are never copied into this plan.
    if board.database_program is not None:
        environment.update(program.environment())
    if live_admission is not None:
        extension_directory = str(
            os.environ.get(CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV, "") or ""
        )
        if not extension_directory:
            raise ConfiguredBoardError(
                "configured-board live launch lacks its extension projection"
            )
        environment[CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV] = (
            extension_directory
        )
        environment[CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV] = (
            live_admission.extension_set_pin.to_json()
        )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "configured-board-launch-plan@1"
        ),
        "board_namespace": board.board_namespace,
        "implement": bool(implement),
        "detach": bool(detach),
        "lanes": board.max_lanes,
        "admitted_lanes": len(plan_bound_children) if plan_bound else board.max_lanes,
        "strict_task_sharding": board.strict_task_sharding,
        "idle_lane_work_stealing": board.idle_lane_work_stealing,
        "effective_strict_task_sharding": (
            board.strict_task_sharding if not plan_bound else False
        ),
        "effective_idle_lane_work_stealing": (
            board.idle_lane_work_stealing if not plan_bound else ""
        ),
        "plan_bound_dispatch": plan_bound,
        "configured_board_live_capsule": {
            "required": bool(board.live_capsule_control_paths),
            "admitted": live_admission is not None,
            "admission_cid": (
                live_admission.admission_cid
                if live_admission is not None
                else ""
            ),
            "control_plane_capsule_id": (
                live_admission.control_plane_capsule_id
                if live_admission is not None
                else ""
            ),
        },
        "accepted_control_plane_required": (
            _sealed_configured_control_plane_required(board)
        ),
        "accepted_control_plane_bound": accepted_control_plane_pin is not None,
        "active_plan_revision_cid": (
            parallelism_receipt.binding.revision_cid
            if parallelism_receipt is not None
            else ""
        ),
        "slice_manifest_cid": (
            parallelism_receipt.slice_manifest_cid
            if parallelism_receipt is not None
            else ""
        ),
        "argv": runner_args,
        "environment": environment,
        "database_program": program.redacted_dict(),
        "database_program_interface": DATABASE_PROGRAM_CONFIG_INTERFACE,
        "runtime_root": str(runtime_root),
        "master_pid_path": str(
            state_dir / "configured-board-master.pid"
        ),
        "master_log": str(
            log_dir / f"configured-board-{run_stamp}.log"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and launch a sealed supervisor scheduler config"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-tree-root", type=Path, default=None)
    parser.add_argument(
        "--accepted-control-plane-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-capsule-parent",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-launch-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
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


def _apply_configured_board_environment(plan: Mapping[str, Any]) -> None:
    environment = plan.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    for name in SCHEDULER_PROVIDER_ENV_NAMES:
        if name not in environment:
            os.environ.pop(name, None)
    for name, value in environment.items():
        os.environ[str(name)] = str(value)


def _ensure_plan_bound_runtime_directory(repo_root: Path, path: Path) -> Path:
    """Create a contained runtime directory one no-symlink component at a time."""

    root = _canonical_no_symlink_root(repo_root)
    directory = Path(path)
    if not directory.is_absolute() or Path(os.path.abspath(directory)) != directory:
        raise ConfiguredBoardError("runtime directory is not lexical absolute")
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ConfiguredBoardError("runtime directory escapes repository") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"cannot create runtime directory: {current}"
                ) from exc
            try:
                observed = os.lstat(current)
            except OSError as exc:
                raise ConfiguredBoardError(
                    f"cannot revalidate runtime directory: {current}"
                ) from exc
        except OSError as exc:
            raise ConfiguredBoardError(
                f"cannot inspect runtime directory: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ConfiguredBoardError(
                f"runtime path component is not a real directory: {current}"
            )
    return directory


def _open_plan_bound_coordinator_log(log_path: Path):
    """Open one append-only log without following or accepting hardlinks."""

    path = Path(log_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConfiguredBoardError(
            "cannot open detached coordinator log safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        observed = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(observed.st_dev), int(observed.st_ino))
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(observed.st_uid) != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise ConfiguredBoardError(
                "detached coordinator log is not a stable single-link file"
            )
        return os.fdopen(descriptor, "ab", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _reserve_coordinator_pid_projection(pid_path: Path) -> tuple[int, tuple[int, int]]:
    """Exclusively reserve a no-follow PID artifact before process creation."""

    path = Path(pid_path)
    with serialized_lock_update(path):
        try:
            existing = os.lstat(path)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot inspect detached coordinator PID projection"
            ) from exc
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                reason = "symbolic link"
            elif not stat.S_ISREG(existing.st_mode):
                reason = "non-regular file"
            elif int(existing.st_nlink) != 1:
                reason = "hardlinked file"
            else:
                reason = "existing owned file"
            raise ConfiguredBoardError(
                "detached coordinator PID projection is an unsafe " + reason
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ConfiguredBoardError(
                "cannot exclusively reserve detached coordinator PID projection"
            ) from exc
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise ConfiguredBoardError(
                "detached coordinator PID reservation is not a single-link file"
            )
        return descriptor, (int(opened.st_dev), int(opened.st_ino))


def _publish_reserved_coordinator_pid(
    pid_path: Path,
    descriptor: int,
    reserved_identity: tuple[int, int],
    pid: int,
) -> None:
    """Publish an exact PID only while the reserved pathname still owns the fd."""

    payload = f"{int(pid)}\n".encode("ascii")
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short PID projection write")
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        observed = os.lstat(pid_path)
        if (
            (int(opened.st_dev), int(opened.st_ino)) != reserved_identity
            or (int(observed.st_dev), int(observed.st_ino))
            != reserved_identity
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(opened.st_uid) != os.geteuid()
            or int(observed.st_uid) != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_IMODE(observed.st_mode) != 0o600
            or int(observed.st_size) != len(payload)
        ):
            raise ConfiguredBoardError(
                "detached coordinator PID projection changed during publication"
            )
    except OSError as exc:
        raise ConfiguredBoardError(
            "cannot publish detached coordinator PID projection"
        ) from exc


def _remove_reserved_coordinator_pid(
    pid_path: Path,
    reserved_identity: tuple[int, int],
) -> None:
    """Remove only the still-identical empty reservation after launch failure."""

    with serialized_lock_update(pid_path):
        try:
            observed = os.lstat(pid_path)
        except FileNotFoundError:
            return
        if (
            (int(observed.st_dev), int(observed.st_ino)) == reserved_identity
            and stat.S_ISREG(observed.st_mode)
            and int(observed.st_nlink) == 1
            and int(observed.st_uid) == os.geteuid()
            and stat.S_IMODE(observed.st_mode) == 0o600
        ):
            pid_path.unlink()


def _materialize_plan_bound_control_plane(
    board: ConfiguredBoard,
) -> tuple[
    AgentImplementationControlPlanePin,
    AgentImplementationSealedControlPlane,
    Path,
]:
    """Seal one clean accepted HEAD outside the candidate repository."""

    accepted_tree_root = Path(__file__).absolute().parents[3]
    if board.repo_root != accepted_tree_root:
        raise ConfiguredBoardError(
            "plan-bound coordinator repo root is not the accepted module tree"
        )
    source_head, source_tree = _git_identity(accepted_tree_root)
    capsule_parent = Path(
        tempfile.mkdtemp(prefix="asref-configured-control-plane-")
    )
    try:
        pin = materialize_agent_implementation_control_plane_capsule(
            source_root=accepted_tree_root,
            capsule_parent=capsule_parent,
            source_head=source_head,
            source_tree=source_tree,
        )
        sealed = seal_agent_implementation_control_plane_capsule(pin)
        if (
            pin.source_head != source_head
            or pin.source_tree != source_tree
            or verify_agent_implementation_sealed_control_plane(
                pin,
                sealed.descriptor,
            )
            != sealed.executable_path
        ):
            raise ConfiguredBoardError(
                "accepted control-plane capsule identity drifted"
            )
        return pin, sealed, capsule_parent
    except BaseException:
        try:
            shutil.rmtree(capsule_parent)
        except OSError:
            pass
        raise


def _build_live_capsule_admission(
    board: ConfiguredBoard,
    *,
    pin: AgentImplementationControlPlanePin,
    descriptor: int,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
) -> ConfiguredBoardLiveCapsuleAdmission:
    """Bind the existing accepted source capsule to this configured board."""

    if not board.live_capsule_control_paths:
        raise ConfiguredBoardError(
            "configured-board live control capsule policy is absent"
        )
    try:
        verify_agent_implementation_sealed_control_plane(pin, descriptor)
        native_executable = verify_agent_supervisor_native_dependency_sealed_fd(
            native_dependency_launch
        )
        native_descriptor = native_dependency_launch.descriptor.descriptor
        if (
            native_descriptor == descriptor
            or native_executable != f"/proc/self/fd/{native_descriptor}"
        ):
            raise ConfiguredBoardError(
                "configured-board native dependency descriptor drifted"
            )
        program = board.resolved_database_program()
        extension_set_pin, extension_pins, _extension_sources = (
            _configured_board_extension_set_projection(
                board,
                dependency_seal_snapshot=dependency_seal_snapshot,
            )
        )
        admission = build_configured_board_live_capsule_admission(
            repo_root=board.repo_root,
            board_namespace=board.board_namespace,
            plan_revision=str(board.payload.get("plan_revision") or ""),
            task_prefix=board.task_prefix,
            config_path=board.config_path.relative_to(
                board.repo_root
            ).as_posix(),
            configuration_root=board.configuration_root,
            control_paths=board.live_capsule_control_paths,
            control_plane_pin=pin,
            native_authorization_id=(
                native_dependency_launch.accepted_authorization_id
            ),
            native_dependency_id=native_dependency_launch.pin.dependency_id,
            native_python_executable_sha256=(
                native_dependency_launch.pin.python_executable_sha256
            ),
            quack_extension_projection=extension_pins["quack"],
            extension_set_pin=extension_set_pin,
            database_authority={
                "authority_mode": program.authority_mode,
                "task_source_kind": program.task_source_kind,
                "schema_revision": program.schema_revision,
                "failover_policy": program.failover_policy,
                "store_id": program.store_id,
                "store_generation": int(program.store_generation),
                "endpoint_secret_handle": program.endpoint_secret_handle,
            },
            max_lanes=board.max_lanes,
            strict_task_sharding=board.strict_task_sharding,
        )
        dependency_artifacts = tuple(
            artifact
            for artifact in admission.control_artifacts
            if artifact.get("path") == board.dependency_seal_path
        )
        if dependency_artifacts != (dependency_seal_snapshot.artifact,):
            raise ConfiguredBoardError(
                "configured-board live admission observed a different dependency seal"
            )
    except (OSError, ConfiguredBoardLiveCapsuleError, ValueError) as exc:
        raise ConfiguredBoardError(
            "configured-board live control capsule admission failed"
        ) from exc
    return admission


def _configured_board_dependency_seal_snapshot(
    board: ConfiguredBoard,
    *,
    expected_artifact: Mapping[str, object] | None = None,
) -> _ConfiguredBoardDependencySealSnapshot:
    """Read the protected dependency seal once from the accepted Git generation."""

    if not board.dependency_seal_path:
        raise ConfiguredBoardError(
            "configured-board live launch lacks a dependency seal"
        )
    if (
        board.dependency_seal_path not in board.protected_paths
        or board.dependency_seal_path not in board.live_capsule_control_paths
    ):
        raise ConfiguredBoardError(
            "configured-board dependency seal is not a protected live control"
        )
    try:
        source_head, _source_tree = _git_identity(board.repo_root)
        raw, _revision = _tracked_head_snapshot(
            repo_root=board.repo_root,
            path=board.path(board.dependency_seal_path),
            source_head=source_head,
            max_bytes=4_194_304,
        )
        def reject_duplicate_keys(
            pairs: Sequence[tuple[str, Any]],
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload:
                    raise ConfiguredBoardError(
                        "configured-board dependency seal repeats a JSON key"
                    )
                payload[key] = value
            return payload

        seal = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
        if type(seal) is not dict:
            raise ConfiguredBoardError(
                "configured-board dependency seal is not a JSON object"
            )
        artifact: Mapping[str, object] = {
            "path": board.dependency_seal_path,
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }
        if expected_artifact is not None and dict(expected_artifact) != artifact:
            raise ConfiguredBoardError(
                "configured-board dependency seal differs from live admission"
            )
        return _ConfiguredBoardDependencySealSnapshot(
            payload=dict(seal),
            artifact=artifact,
        )
    except ConfiguredBoardError:
        raise
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "configured-board dependency seal snapshot failed"
        ) from exc


def _configured_board_quack_projection(
    board: ConfiguredBoard,
    *,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
) -> tuple[ConfiguredBoardExtensionPin, Path, Path]:
    """Resolve Quack only from the exact seal snapshot used for this launch."""

    try:
        seal = dependency_seal_snapshot.payload
        projection = (
            seal.get("configured_board_quack_projection")
            if type(seal) is dict
            else None
        )
        if type(projection) is not dict or set(projection) != {
            "schema",
            "source_path",
            "info_path",
            "pin",
            "load_policy",
            "network_install_allowed",
            "unsigned_extension_allowed",
        }:
            raise ConfiguredBoardError(
                "configured-board Quack projection seal is noncanonical"
            )
        if (
            projection.get("schema")
            != "semantic-addressed-world-model/configured-board-quack-projection@1"
            or projection.get("load_policy") != "local_load_only"
            or projection.get("network_install_allowed") is not False
            or projection.get("unsigned_extension_allowed") is not False
        ):
            raise ConfiguredBoardError(
                "configured-board Quack projection policy is invalid"
            )
        pin = parse_configured_board_extension_pin(projection.get("pin"))
        source = Path(str(projection.get("source_path") or ""))
        info = Path(str(projection.get("info_path") or ""))
        if not source.is_absolute() or not info.is_absolute():
            raise ConfiguredBoardError(
                "configured-board Quack projection sources are not absolute"
            )
        return pin, source, info
    except ConfiguredBoardError:
        raise
    except (OSError, TypeError, ValueError, _StableArtifactReadError) as exc:
        raise ConfiguredBoardError(
            "configured-board Quack projection admission failed"
        ) from exc


def _configured_board_extension_set_projection(
    board: ConfiguredBoard,
    *,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
) -> tuple[
    ConfiguredBoardExtensionSetPin,
    dict[str, ConfiguredBoardExtensionPin],
    dict[str, tuple[Path, Path]],
]:
    """Resolve one exact co-versioned HTTPFS+Quack load set.

    The protected dependency-seal snapshot supplies byte authority.  The
    protected scheduler configuration supplies the expected DuckDB extension
    versions used to verify the native loader's post-LOAD rows.  Neither
    source alone is sufficient.
    """

    quack_pin, quack_source, quack_info = _configured_board_quack_projection(
        board,
        dependency_seal_snapshot=dependency_seal_snapshot,
    )
    seal = dependency_seal_snapshot.payload
    httpfs = seal.get("httpfs_extension_pin")
    expected_httpfs_fields = {
        "path",
        "sha256",
        "size",
        "info_path",
        "info_sha256",
        "info_size",
        "version",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }
    if type(httpfs) is not dict or set(httpfs) != expected_httpfs_fields:
        raise ConfiguredBoardError(
            "configured-board HTTPFS dependency pin is noncanonical"
        )
    owner = board.payload.get("quack_owner")
    if type(owner) is not dict:
        raise ConfiguredBoardError(
            "configured-board Quack owner authority is unavailable"
        )
    owner_pins = {
        "httpfs": owner.get("pinned_httpfs_extension"),
        "quack": owner.get("pinned_extension"),
    }
    if any(type(value) is not dict for value in owner_pins.values()):
        raise ConfiguredBoardError(
            "configured-board extension owner pins are noncanonical"
        )
    httpfs_source = Path(str(httpfs.get("path") or ""))
    httpfs_info = Path(str(httpfs.get("info_path") or ""))
    if not httpfs_source.is_absolute() or not httpfs_info.is_absolute():
        raise ConfiguredBoardError(
            "configured-board HTTPFS projection sources are not absolute"
        )
    try:
        httpfs_pin = inspect_configured_board_extension_sources(
            httpfs_source,
            httpfs_info,
            name="httpfs",
            engine_version=quack_pin.engine_version,
            platform=quack_pin.platform,
        )
        observed_quack_pin = inspect_configured_board_extension_sources(
            quack_source,
            quack_info,
            name="quack",
            engine_version=quack_pin.engine_version,
            platform=quack_pin.platform,
        )
    except (OSError, ValueError) as exc:
        raise ConfiguredBoardError(
            "configured-board extension source inspection failed"
        ) from exc
    if observed_quack_pin != quack_pin:
        raise ConfiguredBoardError(
            "configured-board Quack source differs from its protected pin"
        )
    if (
        httpfs.get("network_install_allowed") is not False
        or httpfs.get("unsigned_extension_allowed") is not False
        or httpfs_pin.payload_sha256
        != f"sha256:{str(httpfs.get('sha256') or '')}"
        or httpfs_pin.payload_size != httpfs.get("size")
        or httpfs_pin.info_sha256
        != f"sha256:{str(httpfs.get('info_sha256') or '')}"
        or httpfs_pin.info_size != httpfs.get("info_size")
    ):
        raise ConfiguredBoardError(
            "configured-board HTTPFS source differs from its protected pin"
        )
    pins = {"httpfs": httpfs_pin, "quack": quack_pin}
    sources = {
        "httpfs": (httpfs_source, httpfs_info),
        "quack": (quack_source, quack_info),
    }
    for name, protected in (
        ("httpfs", httpfs),
        (
            "quack",
            {
                "path": str(quack_source),
                "info_path": str(quack_info),
                "sha256": quack_pin.payload_sha256.removeprefix("sha256:"),
                "size": quack_pin.payload_size,
                "info_sha256": quack_pin.info_sha256.removeprefix("sha256:"),
                "info_size": quack_pin.info_size,
                "network_install_allowed": False,
                "unsigned_extension_allowed": False,
            },
        ),
    ):
        configured = owner_pins[name]
        assert isinstance(configured, dict)
        for field in (
            "path",
            "info_path",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
        ):
            if configured.get(field) != protected.get(field):
                raise ConfiguredBoardError(
                    f"configured-board {name} owner pin differs from "
                    "the dependency seal"
                )
    versions = {
        name: str(owner_pin.get("version") or "")
        for name, owner_pin in owner_pins.items()
        if isinstance(owner_pin, dict)
    }
    try:
        extension_set_pin = build_configured_board_extension_set_pin(
            pins,
            versions=versions,
        )
    except ValueError as exc:
        raise ConfiguredBoardError(
            "configured-board exact extension set pin is invalid"
        ) from exc
    return extension_set_pin, pins, sources


def _configured_board_native_dependency_authority(
    board: ConfiguredBoard,
    *,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
) -> tuple[AgentSupervisorNativeDependencyPin, str, Path]:
    """Authenticate the protected authorization without creating a launch."""

    if (
        not board.dependency_seal_path
        or board.dependency_seal_path not in board.protected_paths
        or board.dependency_seal_path not in board.live_capsule_control_paths
    ):
        raise ConfiguredBoardError(
            "configured-board native dependency lacks a protected seal"
        )
    try:
        seal = dependency_seal_snapshot.payload
        if (
            type(seal) is not dict
            or seal.get("schema")
            != "semantic-addressed-world-model/dependency-seal@1"
            or seal.get("board_namespace") != board.board_namespace
            or seal.get("plan_revision")
            != str(board.payload.get("plan_revision") or "")
            or seal.get("status") != "sealed"
        ):
            raise ConfiguredBoardError(
                "configured-board dependency seal identity is invalid"
            )
        native = seal.get("configured_board_native_dependency")
        if type(native) is not dict or set(native) != {
            "schema",
            "source_path",
            "acceptance",
            "pin",
            "sealed_memfd_required",
            "ambient_site_import_allowed",
            "ambient_loader_environment_allowed",
        }:
            raise ConfiguredBoardError(
                "configured-board native dependency seal is noncanonical"
            )
        if (
            native.get("schema")
            != "semantic-addressed-world-model/configured-board-native-dependency@1"
            or native.get("sealed_memfd_required") is not True
            or native.get("ambient_site_import_allowed") is not False
            or native.get("ambient_loader_environment_allowed") is not False
        ):
            raise ConfiguredBoardError(
                "configured-board native dependency policy is invalid"
            )
        pin = parse_agent_supervisor_native_dependency_pin(native.get("pin"))
        reference = native.get("acceptance")
        if type(reference) is not dict or set(reference) != {
            "schema",
            "path",
            "sha256",
            "size",
            "authorization_id",
        }:
            raise ConfiguredBoardError(
                "configured-board native authorization reference is noncanonical"
            )
        if reference.get("schema") != (
            "semantic-addressed-world-model/"
            "native-dependency-authorization-reference@1"
        ):
            raise ConfiguredBoardError(
                "configured-board native authorization reference is invalid"
            )
        authorization_relative = _safe_relative(
            str(reference.get("path") or ""),
            field="native authorization path",
        )
        if (
            authorization_relative not in board.protected_paths
            or authorization_relative not in board.live_capsule_control_paths
        ):
            raise ConfiguredBoardError(
                "configured-board native authorization is not protected"
            )
        authorization, authorization_evidence = _read_stable_regular_json(
            board.path(authorization_relative),
            max_bytes=65_536,
        )
        if (
            type(authorization) is not dict
            or set(authorization) != {
                "schema",
                "board_namespace",
                "plan_revision",
                "status",
                "scope",
                "dependency_id",
                "payload_sha256",
                "python_executable_sha256",
                "authority_basis",
                "inspection_is_authority",
                "authorization_may_claim_task_completion",
                "authorization_id",
            }
            or authorization_evidence.get("content_sha256")
            != reference.get("sha256")
            or authorization_evidence.get("size") != reference.get("size")
        ):
            raise ConfiguredBoardError(
                "configured-board native authorization artifact differs"
            )
        unsigned_authorization = dict(authorization)
        authorization_id = str(
            unsigned_authorization.pop("authorization_id", "") or ""
        )
        expected_authorization_id = "sha256:" + hashlib.sha256(
            json.dumps(
                unsigned_authorization,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            authorization_id != expected_authorization_id
            or authorization_id != reference.get("authorization_id")
            or authorization.get("schema") != (
                "semantic-addressed-world-model/"
                "native-dependency-launch-authorization@1"
            )
            or authorization.get("board_namespace") != board.board_namespace
            or authorization.get("plan_revision")
            != str(board.payload.get("plan_revision") or "")
            or authorization.get("status") != "accepted"
            or authorization.get("scope")
            != "configured-board-live-control-plane"
            or authorization.get("dependency_id") != pin.dependency_id
            or authorization.get("payload_sha256") != pin.payload_sha256
            or authorization.get("python_executable_sha256")
            != pin.python_executable_sha256
            or authorization.get("authority_basis")
            != (
                "operator-owned protected control inside the accepted "
                "immutable source capsule"
            )
            or authorization.get("inspection_is_authority") is not False
            or authorization.get("authorization_may_claim_task_completion")
            is not False
        ):
            raise ConfiguredBoardError(
                "configured-board native authorization was not admitted"
            )
        source = Path(str(native.get("source_path") or ""))
        if not source.is_absolute():
            raise ConfiguredBoardError(
                "configured-board native dependency source is not absolute"
            )
        return pin, authorization_id, source
    except ConfiguredBoardError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        _StableArtifactReadError,
    ) as exc:
        raise ConfiguredBoardError(
            "configured-board native dependency admission failed"
        ) from exc


def _seal_configured_board_native_dependency(
    board: ConfiguredBoard,
    *,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
) -> AgentSupervisorNativeDependencyLaunch:
    """Authenticate the protected authorization, then seal exact DuckDB bytes."""

    pin, authorization_id, source = _configured_board_native_dependency_authority(
        board,
        dependency_seal_snapshot=dependency_seal_snapshot,
    )
    try:
        launch = seal_agent_supervisor_native_dependency(
            source,
            expected_pin=pin,
            accepted_authorization_id=authorization_id,
        )
        verify_agent_supervisor_native_dependency_sealed_fd(launch)
        return launch
    except (OSError, TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "configured-board native dependency sealing failed"
        ) from exc


def _authenticate_configured_board_native_dependency_launch(
    board: ConfiguredBoard,
    *,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot,
    launch: AgentSupervisorNativeDependencyLaunch,
) -> None:
    """Re-authenticate one propagated launch at an accepted inner birth."""

    pin, authorization_id, _source = (
        _configured_board_native_dependency_authority(
            board,
            dependency_seal_snapshot=dependency_seal_snapshot,
        )
    )
    try:
        launch_pin = parse_agent_supervisor_native_dependency_pin(
            launch.pin.as_dict()
        )
        verify_agent_supervisor_native_dependency_sealed_fd(launch)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ConfiguredBoardError(
            "configured-board propagated native dependency is invalid"
        ) from exc
    if launch_pin != pin or launch.accepted_authorization_id != authorization_id:
        raise ConfiguredBoardError(
            "configured-board propagated native dependency is unauthorized"
        )


def _plan_bound_coordinator_module_argv(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    pin: AgentImplementationControlPlanePin,
    sealed: AgentImplementationSealedControlPlane,
    capsule_parent: Path,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
) -> list[str]:
    argv = [
        "--repo-root",
        str(board.repo_root),
        "--config",
        str(board.config_path),
        "--accepted-tree-root",
        str(board.repo_root),
        "--accepted-control-plane-pin-json",
        accepted_control_plane_pin_json(pin),
        "--accepted-control-plane-fd",
        str(sealed.descriptor),
        "--accepted-control-plane-capsule-parent",
        str(capsule_parent),
        "launch",
        "--foreground",
        "--duration-seconds",
        str(duration_seconds),
    ]
    if native_dependency_launch is not None:
        argv[argv.index("launch"):argv.index("launch")] = [
            "--configured-board-live-native-launch-json",
            native_dependency_launch.to_json(),
            "--configured-board-live-native-fd",
            str(native_dependency_launch.descriptor.descriptor),
        ]
    if implement:
        argv.append("--implement")
    return argv


def _cleanup_plan_bound_control_plane(
    pin: AgentImplementationControlPlanePin,
    capsule_parent: Path,
) -> None:
    """Remove only the uniquely-created private capsule parent after fencing."""

    parent = Path(capsule_parent)
    capsule = Path(pin.capsule_root)
    if (
        not parent.is_absolute()
        or parent.parent != Path(tempfile.gettempdir())
        or not parent.name.startswith("asref-configured-control-plane-")
        or capsule.parent != parent
    ):
        return
    try:
        for entry in parent.rglob("*"):
            observed = os.lstat(entry)
            if stat.S_ISLNK(observed.st_mode) or int(observed.st_uid) != os.geteuid():
                return
        directories = sorted(
            (entry for entry in parent.rglob("*") if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        )
        for directory in directories:
            os.chmod(directory, 0o700)
        os.chmod(parent, 0o700)
        shutil.rmtree(parent)
    except OSError:
        return


def _launch_foreground_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot | None = None,
) -> int:
    pin, sealed, capsule_parent = _materialize_plan_bound_control_plane(board)
    try:
        if dependency_seal_snapshot is None:
            raise ConfiguredBoardError(
                "configured-board coordinator lacks its dependency-seal snapshot"
            )
        extension_set_pin, extension_pins, extension_sources = (
            _configured_board_extension_set_projection(
                board,
                dependency_seal_snapshot=dependency_seal_snapshot,
            )
        )
        extension_home = project_configured_board_extension_set_home(
            extension_pins,
            sources=extension_sources,
            parent=capsule_parent,
        )
        extension_directory = extension_home / ".duckdb/extensions"
        command = build_sealed_control_plane_module_command(
            python_executable=sys.executable,
            pin=pin,
            descriptor=sealed.descriptor,
            native_dependency_launch=native_dependency_launch,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "configured_board_scheduler"
            ),
            argv=_plan_bound_coordinator_module_argv(
                board,
                implement=implement,
                duration_seconds=duration_seconds,
                pin=pin,
                sealed=sealed,
                capsule_parent=capsule_parent,
                native_dependency_launch=native_dependency_launch,
            ),
        )
        process = subprocess.Popen(
            command,
            cwd=board.repo_root,
            env=_sealed_coordinator_environment(
                board,
                extension_directory=extension_directory,
                extension_set_pin=extension_set_pin,
            ),
            stdin=subprocess.DEVNULL,
            start_new_session=False,
            pass_fds=(
                sealed.descriptor,
                *(
                    native_dependency_launch.pass_fds
                    if native_dependency_launch is not None
                    else ()
                ),
            ),
        )
        return int(process.wait())
    finally:
        os.close(sealed.descriptor)
        _cleanup_plan_bound_control_plane(pin, capsule_parent)


def _launch_detached_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot | None = None,
) -> dict[str, Any]:
    """Detach the outer coordinator, never an individual finite wave."""

    state_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["state"]),
    )
    log_dir = _ensure_plan_bound_runtime_directory(
        board.repo_root,
        board.path(board.runtime_paths["logs"]),
    )
    stamp = utc_run_stamp()
    log_path = log_dir / f"configured-board-{stamp}.log"
    pid_path = state_dir / "configured-board-master.pid"
    accepted_tree_root = Path(__file__).absolute().parents[3]
    if board.repo_root != accepted_tree_root:
        raise ConfiguredBoardError(
            "detached coordinator repo root is not the accepted module tree"
        )
    entry = accepted_tree_root / CONFIGURED_SCHEDULER_ENTRY_PATH
    _lexical_repo_artifact(accepted_tree_root, pid_path)
    source_head, _source_tree = _git_identity(accepted_tree_root)
    for authority_path in (
        entry,
        board.config_path,
        board.path(board.taskboard_path),
    ):
        _tracked_head_snapshot(
            repo_root=accepted_tree_root,
            path=authority_path,
            source_head=source_head,
        )
    descriptor, reserved_identity = _reserve_coordinator_pid_projection(
        pid_path
    )
    process: subprocess.Popen[bytes] | None = None
    sealed: AgentImplementationSealedControlPlane | None = None
    capsule_parent: Path | None = None
    try:
        pin, sealed, capsule_parent = _materialize_plan_bound_control_plane(
            board
        )
        if dependency_seal_snapshot is None:
            raise ConfiguredBoardError(
                "configured-board coordinator lacks its dependency-seal snapshot"
            )
        extension_set_pin, extension_pins, extension_sources = (
            _configured_board_extension_set_projection(
                board,
                dependency_seal_snapshot=dependency_seal_snapshot,
            )
        )
        extension_home = project_configured_board_extension_set_home(
            extension_pins,
            sources=extension_sources,
            parent=capsule_parent,
        )
        extension_directory = extension_home / ".duckdb/extensions"
        command = build_sealed_control_plane_module_command(
            python_executable=sys.executable,
            pin=pin,
            descriptor=sealed.descriptor,
            native_dependency_launch=native_dependency_launch,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.runtime."
                "configured_board_scheduler"
            ),
            argv=_plan_bound_coordinator_module_argv(
                board,
                implement=implement,
                duration_seconds=duration_seconds,
                pin=pin,
                sealed=sealed,
                capsule_parent=capsule_parent,
                native_dependency_launch=native_dependency_launch,
            ),
        )
        with _open_plan_bound_coordinator_log(log_path) as stream:
            process = subprocess.Popen(
                command,
                cwd=accepted_tree_root,
                env=_sealed_coordinator_environment(
                    board,
                    extension_directory=extension_directory,
                    extension_set_pin=extension_set_pin,
                ),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(
                    sealed.descriptor,
                    *(
                        native_dependency_launch.pass_fds
                        if native_dependency_launch is not None
                        else ()
                    ),
                ),
            )
        _publish_reserved_coordinator_pid(
            pid_path,
            descriptor,
            reserved_identity,
            process.pid,
        )
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        _remove_reserved_coordinator_pid(pid_path, reserved_identity)
        if capsule_parent is not None:
            try:
                shutil.rmtree(capsule_parent)
            except OSError:
                pass
        raise
    finally:
        os.close(descriptor)
        if sealed is not None:
            os.close(sealed.descriptor)
    assert process is not None
    return {
        "coordinator_pid": process.pid,
        "coordinator_pid_path": str(pid_path),
        "coordinator_log": str(log_path),
    }


def _run_plan_bound_coordinator(
    board: ConfiguredBoard,
    *,
    implement: bool,
    duration_seconds: float,
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
) -> int:
    """Publish and execute fresh exact waves until drain or the run bound."""

    from .multi_supervisor_runner import PLAN_BOUND_REPLAN_RETURN_CODE
    from .multi_supervisor_runner import main as multi_supervisor_main

    started = time.monotonic()
    base_stamp = utc_run_stamp()
    for wave_index in range(MAX_COORDINATOR_WAVES):
        elapsed = time.monotonic() - started
        if math.isfinite(duration_seconds) and elapsed >= duration_seconds:
            return 0
        try:
            current_board = load_configured_board(
                board.config_path,
                repo_root=board.repo_root,
            )
            if current_board.board_namespace != board.board_namespace:
                raise ConfiguredBoardError(
                    "coordinator configuration changed board namespace"
                )
            receipt = materialize_configured_board_execution_plan(current_board)
        except (ConfiguredBoardError, OSError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    {"valid": False, "errors": [f"adaptive_plan: {exc}"]},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if receipt is None:
            print(
                json.dumps(
                    {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "configured-board-coordinator-result@1"
                        ),
                        "board_namespace": board.board_namespace,
                        "waves_completed": wave_index,
                        "reason": "no_dependency_ready_retry_admissible_tasks",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        remaining = (
            max(0.0, duration_seconds - elapsed)
            if math.isfinite(duration_seconds)
            else float("inf")
        )
        plan = configured_board_launch_plan(
            current_board,
            implement=implement,
            detach=False,
            duration_seconds=remaining,
            stamp=f"{base_stamp}-wave-{wave_index}",
            parallelism_receipt=receipt,
            accepted_control_plane_pin=accepted_control_plane_pin,
            accepted_control_plane_descriptor=(
                accepted_control_plane_descriptor
            ),
            native_dependency_launch=native_dependency_launch,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        _apply_configured_board_environment(plan)
        result = int(multi_supervisor_main(plan["argv"]))
        if result == PLAN_BOUND_REPLAN_RETURN_CODE:
            continue
        if result != 0:
            return result
    print(
        json.dumps(
            {
                "valid": False,
                "errors": ["adaptive coordinator exceeded its wave bound"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2


def _remove_owned_coordinator_pid(board: ConfiguredBoard) -> bool:
    """Remove only this coordinator's detached-launch PID projection."""

    pid_path = (
        board.path(board.runtime_paths["state"])
        / "configured-board-master.pid"
    )
    try:
        _lexical_repo_artifact(board.repo_root, pid_path)
        with serialized_lock_update(pid_path):
            payload, evidence = _read_stable_regular_bytes(
                pid_path,
                max_bytes=32,
            )
            if payload is None or not re.fullmatch(rb"[1-9][0-9]*\n", payload):
                return False
            recorded_pid = int(payload[:-1].decode("ascii"))
            if recorded_pid != os.getpid():
                return False
            observed = os.lstat(pid_path)
            if (
                evidence.get("state") != "present"
                or int(evidence.get("device", -1)) != int(observed.st_dev)
                or int(evidence.get("inode", -1)) != int(observed.st_ino)
                or stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or int(observed.st_uid) != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                return False
            pid_path.unlink()
            return True
    except (
        ConfiguredBoardError,
        _StableArtifactReadError,
        OSError,
        UnicodeError,
        ValueError,
    ):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    control_plane_pin: AgentImplementationControlPlanePin | None = None
    control_plane_descriptor = -1
    control_plane_parent: Path | None = None
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None
    dependency_seal_snapshot: _ConfiguredBoardDependencySealSnapshot | None = None
    native_dependency_owned = False
    try:
        board = load_configured_board(
            args.config,
            repo_root=args.repo_root,
        )
        sealed_control_plane_required = (
            _sealed_configured_control_plane_required(board)
        )
        has_control_plane = bool(args.accepted_control_plane_pin_json)
        has_descriptor = args.accepted_control_plane_fd >= 3
        has_parent = args.accepted_control_plane_capsule_parent is not None
        if len({has_control_plane, has_descriptor, has_parent}) != 1:
            raise ConfiguredBoardError(
                "accepted control-plane launch fields are incomplete"
            )
        if has_control_plane and not sealed_control_plane_required:
            raise ConfiguredBoardError(
                "accepted control-plane launch is not declared by this profile"
            )
        has_native_launch = bool(args.configured_board_live_native_launch_json)
        has_native_descriptor = args.configured_board_live_native_fd >= 3
        if has_native_launch != has_native_descriptor:
            raise ConfiguredBoardError(
                "configured-board native launch fields are incomplete"
            )
        if has_native_launch and not sealed_control_plane_required:
            raise ConfiguredBoardError(
                "configured-board native launch is foreign to this profile"
            )
        if has_native_launch:
            try:
                native_dependency_launch = parse_native_dependency_launch_json(
                    args.configured_board_live_native_launch_json
                )
                if (
                    native_dependency_launch.descriptor.descriptor
                    != args.configured_board_live_native_fd
                ):
                    raise ValueError(
                        "native dependency descriptor was substituted"
                    )
                verify_agent_supervisor_native_dependency_sealed_fd(
                    native_dependency_launch
                )
                if not _plan_bound_profile(board):
                    dependency_seal_snapshot = (
                        _configured_board_dependency_seal_snapshot(board)
                    )
                    _authenticate_configured_board_native_dependency_launch(
                        board,
                        dependency_seal_snapshot=dependency_seal_snapshot,
                        launch=native_dependency_launch,
                    )
            except (OSError, ValueError) as exc:
                raise ConfiguredBoardError(
                    "configured-board native launch binding is invalid"
                ) from exc
        if has_control_plane:
            try:
                control_plane_pin = parse_accepted_control_plane_pin(
                    args.accepted_control_plane_pin_json
                )
                control_plane_descriptor = int(
                    args.accepted_control_plane_fd
                )
                verify_agent_implementation_sealed_control_plane(
                    control_plane_pin,
                    control_plane_descriptor,
                )
            except (OSError, ValueError) as exc:
                raise ConfiguredBoardError(
                    "accepted control-plane launch binding is invalid"
                ) from exc
            control_plane_parent = Path(
                args.accepted_control_plane_capsule_parent
            )
            if (
                control_plane_parent.parent != Path(tempfile.gettempdir())
                or not control_plane_parent.name.startswith(
                    "asref-configured-control-plane-"
                )
                or Path(control_plane_pin.capsule_root).parent
                != control_plane_parent
                or (
                    control_plane_pin.source_head,
                    control_plane_pin.source_tree,
                )
                != _git_identity(board.repo_root)
            ):
                raise ConfiguredBoardError(
                    "accepted control-plane launch provenance is foreign"
                )
        if args.accepted_tree_root is not None:
            accepted_tree_root = _canonical_no_symlink_root(
                args.accepted_tree_root
            )
            module_tree_root = (
                board.repo_root
                if control_plane_pin is not None
                else Path(__file__).resolve().parents[3]
            )
            if (
                accepted_tree_root != module_tree_root
                or accepted_tree_root != board.repo_root.resolve()
            ):
                raise ConfiguredBoardError(
                    "configured scheduler accepted-tree root is foreign"
                )
        preflight = preflight_configured_board(board)
        if (
            args.command == "launch"
            and not args.dry_run
            and preflight.get("valid") is True
            and sealed_control_plane_required
            and not _plan_bound_profile(board)
            and control_plane_pin is None
        ):
            dependency_seal_snapshot = (
                _configured_board_dependency_seal_snapshot(board)
            )
            native_dependency_launch = _seal_configured_board_native_dependency(
                board,
                dependency_seal_snapshot=dependency_seal_snapshot,
            )
            native_dependency_owned = True
    except ConfiguredBoardError as exc:
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "configured-board-error@1"
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
    if (
        sealed_control_plane_required
        and not _plan_bound_profile(board)
        and (
            not board.live_capsule_control_paths
            or native_dependency_launch is None
        )
        and not args.dry_run
    ):
        print(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "configured-board-error@1"
                    ),
                    "valid": False,
                    "errors": [
                        "operational live launch requires the closed "
                        "configured_board_live_capsule policy and native "
                        "dependency launch"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if sealed_control_plane_required:
        if args.dry_run:
            plan = configured_board_launch_plan(
                board,
                implement=bool(args.implement),
                detach=detach,
                duration_seconds=float(args.duration_seconds),
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if detach and control_plane_pin is None:
            plan = configured_board_launch_plan(
                board,
                implement=bool(args.implement),
                detach=True,
                duration_seconds=float(args.duration_seconds),
            )
            try:
                plan.update(
                    _launch_detached_plan_bound_coordinator(
                        board,
                        implement=bool(args.implement),
                        duration_seconds=float(args.duration_seconds),
                        native_dependency_launch=native_dependency_launch,
                        dependency_seal_snapshot=dependency_seal_snapshot,
                    )
                )
            except (ConfiguredBoardError, OSError) as exc:
                print(
                    json.dumps(
                        {"valid": False, "errors": [f"coordinator_launch: {exc}"]},
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            finally:
                if native_dependency_owned and native_dependency_launch is not None:
                    os.close(native_dependency_launch.descriptor.descriptor)
                    native_dependency_owned = False
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if control_plane_pin is None:
            try:
                return _launch_foreground_plan_bound_coordinator(
                    board,
                    implement=bool(args.implement),
                    duration_seconds=float(args.duration_seconds),
                    native_dependency_launch=native_dependency_launch,
                    dependency_seal_snapshot=dependency_seal_snapshot,
                )
            except (ConfiguredBoardError, OSError, ValueError) as exc:
                print(
                    json.dumps(
                        {
                            "valid": False,
                            "errors": [f"coordinator_launch: {exc}"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            finally:
                if native_dependency_owned and native_dependency_launch is not None:
                    os.close(native_dependency_launch.descriptor.descriptor)
                    native_dependency_owned = False
        if _plan_bound_profile(board):
            try:
                return _run_plan_bound_coordinator(
                    board,
                    implement=bool(args.implement),
                    duration_seconds=float(args.duration_seconds),
                    accepted_control_plane_pin=control_plane_pin,
                    accepted_control_plane_descriptor=control_plane_descriptor,
                    native_dependency_launch=native_dependency_launch,
                )
            finally:
                _remove_owned_coordinator_pid(board)
                if control_plane_parent is not None:
                    _cleanup_plan_bound_control_plane(
                        control_plane_pin,
                        control_plane_parent,
                    )

    live_admission = (
        _build_live_capsule_admission(
            board,
            pin=control_plane_pin,
            descriptor=control_plane_descriptor,
            native_dependency_launch=native_dependency_launch,
            dependency_seal_snapshot=dependency_seal_snapshot,
        )
        if control_plane_pin is not None
        and board.live_capsule_control_paths
        else None
    )
    plan = configured_board_launch_plan(
        board,
        implement=bool(args.implement),
        detach=detach,
        duration_seconds=float(args.duration_seconds),
        accepted_control_plane_pin=control_plane_pin,
        accepted_control_plane_descriptor=control_plane_descriptor,
        native_dependency_launch=native_dependency_launch,
        configured_board_live_admission=live_admission,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    _apply_configured_board_environment(plan)
    from .multi_supervisor_runner import main as multi_supervisor_main

    try:
        return int(multi_supervisor_main(plan["argv"]))
    finally:
        if control_plane_pin is not None and control_plane_parent is not None:
            _remove_owned_coordinator_pid(board)
            _cleanup_plan_bound_control_plane(
                control_plane_pin,
                control_plane_parent,
            )


__all__ = (
    "ConfiguredBoard",
    "ConfiguredBoardError",
    "configured_board_capacity_observation",
    "configured_board_common_args",
    "configured_board_launch_plan",
    "load_configured_board",
    "materialize_configured_board_execution_plan",
    "main",
    "preflight_configured_board",
)
