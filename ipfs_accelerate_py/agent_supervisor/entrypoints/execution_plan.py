"""Production adapter for one immutable, plan-bound parallel wave.

The compiler, revision store, task-source claim authority, worktree lifecycle,
and merge queue already have canonical owners.  This module only joins those
owners: it compiles one bounded wave, stores the complete plan and exact lane
slices in the canonical :class:`PlanRevisionStore`, and returns an
``ActivePlanBinding`` for launch.  It intentionally owns no claim/effect
database and performs no provider, worktree, validation, or merge effects.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..planning.parallel_plan_compiler import (
    ParallelExecutionPlan,
    ParallelPlanCompiler,
)
from ..planning.plan_revision_contracts import (
    PlanAuthorityRoots,
    PlanDelta,
    PlanRevision,
)
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.plan_revision_store import (
    PLAN_REVISION_ACTIVE_SCHEMA,
    PLAN_REVISION_CONTINUATION_SCHEMA,
    PLAN_REVISION_STORE_SCHEMA,
    PlanRevisionActiveProjection,
    PlanRevisionApplyRequest,
    PlanRevisionStore,
)
from ..task_sources.task_source import ActivePlanBinding, bind_active_plan_revision
from .contracts import InvocationBudget

ADAPTIVE_EXECUTION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-execution-plan@1"
)
EXECUTION_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-slice@1"
)
CONFIGURED_BOARD_EXECUTION_SLICES_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-execution-slices@1"
)
PARALLELISM_DECISION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/parallelism-decision-receipt@1"
)
PLAN_SLICE_REASSIGNMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-slice-reassignment@1"
)
PLAN_BOUND_EXECUTION_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-execution-lease@1"
)
PLAN_BOUND_PROCESS_BIRTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-process-birth@2"
)
PLAN_BOUND_PROCESS_BIRTH_EXHAUSTED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-process-birth-exhausted@1"
)
PLAN_BOUND_PROPOSAL_DISPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-proposal-disposition@1"
)
PLAN_BOUND_TERMINAL_MISSING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-terminal-missing@1"
)
PLAN_BOUND_WAVE_DIFF_BARRIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-wave-diff-barrier@1"
)
PLAN_BOUND_WAVE_DIFF_BARRIER_WINDOW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-wave-diff-barrier-window@1"
)
PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-merge-authorization@1"
)
PLAN_BOUND_MERGE_ENQUEUE_INTENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-merge-enqueue-intent@1"
)
PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-merge-queue-receipt@1"
)
PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-merge-recovery-birth@2"
)
PLAN_BOUND_MERGE_TERMINAL_FAILURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-merge-terminal-failure@1"
)
PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-proposal-handoff@1"
)
PLAN_BOUND_RECOVERY_LAUNCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-recovery-launch@1"
)
MAX_TASKS: Final[int] = 4096
MAX_SLICE_REASSIGNMENTS: Final[int] = 4096
MAX_PLAN_BOUND_WAVE_TRANSFERS: Final[int] = 16
MAX_AUTHORITY_JSON_BYTES: Final[int] = 1_048_576

_PLAN_BOUND_EXECUTION_LEASE_PHASES: Final[tuple[str, ...]] = (
    "reserved",
    "claimed",
    "workspace_prepared",
    "provider_ready",
    "proposal_ready",
    "merge_enqueue_reached",
    "merge_enqueue_prepared",
    "merge_enqueue_confirmed",
    "merge_completed",
    "proposal_rejected",
    "scope_drift",
)
_COMPILED_ASSIGNMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "task_id",
        "shard_id",
        "affinity_key",
        "exclusive_group",
        "exclusive_paths",
        "worktree_id",
        "worktree_path",
        "base_revision",
        "merge_target",
        "lease_id",
        "lease_scope",
        "lease_duration_ms",
        "heartbeat_interval_ms",
        "lease_owner_rule",
        "fence_epoch",
        "fence_token",
        "provider_id",
        "resource_class",
    }
)


class ExecutionPlanError(RuntimeError):
    """Base class for inadmissible adaptive execution operations."""


class ExecutionSliceViolationError(ExecutionPlanError):
    """A lane attempted to select work outside its sealed slice."""


class ExecutionClaimConflictError(ExecutionPlanError):
    """A task/effect is already owned by another live fenced claim."""


class ExecutionReplanRequired(ExecutionPlanError):
    """Observed work invalidated the optimistic conflict plan."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanError("execution-plan data must be canonical JSON") from exc


def _cid(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def _text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result or "\x00" in result:
        raise ExecutionPlanError(f"{field_name} must be a nonempty text value")
    return result


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionPlanError(
                f"plan store authority has duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _stable_authority_json(path: Path) -> dict[str, Any]:
    """Read one bounded, single-link store object without following links."""

    artifact = Path(path)

    def validate_metadata(observed: os.stat_result) -> None:
        if int(observed.st_uid) != os.geteuid():
            raise ExecutionPlanError(
                "plan store authority is not owned by the effective user"
            )
        if stat.S_IMODE(observed.st_mode) != 0o600:
            raise ExecutionPlanError(
                "plan store authority permissions must be exactly 0600"
            )

    try:
        before = os.lstat(artifact)
    except OSError as exc:
        raise ExecutionPlanError(
            f"cannot lstat plan store authority: {artifact}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise ExecutionPlanError(
            f"plan store authority is a symbolic link: {artifact}"
        )
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ExecutionPlanError(
            f"plan store authority is not a single-link regular file: {artifact}"
        )
    validate_metadata(before)
    if not 0 <= int(before.st_size) <= MAX_AUTHORITY_JSON_BYTES:
        raise ExecutionPlanError("plan store authority exceeds its read bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(artifact, flags)
    except OSError as exc:
        reason = "symbolic link" if exc.errno == errno.ELOOP else str(exc)
        raise ExecutionPlanError(
            f"cannot securely open plan store authority: {reason}"
        ) from exc

    def identity(observed: os.stat_result) -> tuple[int, ...]:
        return (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_mode),
            int(observed.st_nlink),
            int(observed.st_uid),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )

    try:
        opened = os.fstat(descriptor)
        validate_metadata(opened)
        if identity(opened) != identity(before):
            raise ExecutionPlanError("plan store authority changed before open")
        chunks: list[bytes] = []
        remaining = MAX_AUTHORITY_JSON_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        validate_metadata(after)
        if len(raw) > MAX_AUTHORITY_JSON_BYTES or identity(after) != identity(opened):
            raise ExecutionPlanError("plan store authority changed while read")
    finally:
        os.close(descriptor)
    try:
        final = os.lstat(artifact)
    except OSError as exc:
        raise ExecutionPlanError("plan store authority disappeared after read") from exc
    validate_metadata(final)
    if identity(final) != identity(before):
        raise ExecutionPlanError("plan store authority pathname changed while read")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionPlanError("plan store authority is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ExecutionPlanError("plan store authority must be a JSON object")
    return payload


def _secure_store_cas(store: PlanRevisionStore, cid: str) -> dict[str, Any]:
    exact_cid = _text(cid, "cid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._@-]{0,511}", exact_cid) is None:
        raise ExecutionPlanError("plan store CID is not a safe artifact name")
    record = _stable_authority_json(store.cas_dir / exact_cid)
    if set(record) != {"schema", "cid", "media_type", "payload"} or (
        record.get("schema") != PLAN_REVISION_STORE_SCHEMA
        or record.get("cid") != exact_cid
        or not isinstance(record.get("media_type"), str)
        or not isinstance(record.get("payload"), Mapping)
    ):
        raise ExecutionPlanError("plan store CAS envelope is malformed")
    payload = dict(record["payload"])
    if content_identity(payload) != exact_cid:
        raise ExecutionPlanError("plan store CAS payload identity is invalid")
    return payload


def _secure_store_active(
    store: PlanRevisionStore,
) -> PlanRevisionActiveProjection | None:
    try:
        os.lstat(store.active_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ExecutionPlanError("cannot inspect active plan pointer") from exc
    payload = _stable_authority_json(store.active_path)
    expected = {
        "schema", "active_cid", "plan_root_cid", "revision_cid",
        "semantic_revision", "event_cursor", "markdown_projection_cid",
        "duckdb_projection_cid", "markdown_path", "duckdb_path", "intent_cid",
        "prior_active_cid", "deferred_item_keys", "quarantined",
    }
    if set(payload) != expected or payload.get("schema") != PLAN_REVISION_ACTIVE_SCHEMA:
        raise ExecutionPlanError("active plan pointer fields are not exact")
    if (
        isinstance(payload["semantic_revision"], bool)
        or not isinstance(payload["semantic_revision"], int)
        or payload["semantic_revision"] < 1
        or not isinstance(payload["quarantined"], bool)
        or not isinstance(payload["deferred_item_keys"], list)
        or any(not isinstance(item, str) for item in payload["deferred_item_keys"])
        or any(
            not isinstance(payload[name], str)
            for name in expected
            - {"semantic_revision", "quarantined", "deferred_item_keys"}
        )
    ):
        raise ExecutionPlanError("active plan pointer types are invalid")
    active = PlanRevisionActiveProjection.from_dict(payload)
    if active.to_dict() != payload:
        raise ExecutionPlanError("active plan pointer changed during typed decode")
    return active


def _safe_continuation_filename(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    cleaned = "".join(
        character if character.isalnum() or character in "._-@" else "_"
        for character in value
    )[:96]
    return f"{cleaned}.{digest[:16]}.json"


def _secure_store_continuation(
    store: PlanRevisionStore,
    key: str,
) -> Mapping[str, Any] | None:
    exact_key = _text(key, "idempotency_key")
    path = store.continuations_dir / _safe_continuation_filename(exact_key)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ExecutionPlanError("cannot inspect plan continuation") from exc
    record = _stable_authority_json(path)
    if set(record) != {
        "schema", "idempotency_key", "payload", "updated_at_ns",
        "continuation_cid",
    } or (
        record.get("schema") != PLAN_REVISION_CONTINUATION_SCHEMA
        or record.get("idempotency_key") != exact_key
        or not isinstance(record.get("payload"), Mapping)
        or isinstance(record.get("updated_at_ns"), bool)
        or not isinstance(record.get("updated_at_ns"), int)
        or record.get("updated_at_ns", 0) < 0
        or not isinstance(record.get("continuation_cid"), str)
        or not record.get("continuation_cid")
    ):
        raise ExecutionPlanError("plan continuation envelope is malformed")
    identity_payload = dict(record)
    observed_cid = str(identity_payload.pop("continuation_cid"))
    if content_identity(identity_payload) != observed_cid:
        raise ExecutionPlanError("plan continuation identity is invalid")
    return dict(record["payload"])


def _paths(value: Any, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = (value,) if isinstance(value, str) else value
    try:
        iterable = tuple(values)
    except TypeError as exc:
        raise ExecutionPlanError(f"{field_name} must be a path sequence") from exc
    result: set[str] = set()
    for item in iterable:
        path = str(item or "").strip().replace("\\", "/").removeprefix("./").rstrip("/")
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts:
            raise ExecutionPlanError(f"{field_name} has non-canonical path {item!r}")
        result.add(pure.as_posix())
    return tuple(sorted(result))


def _overlaps(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _any_path_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    return any(_overlaps(a, b) for a in left for b in right)


def _plan_bound_wave_transfer_budget(
    manifest: "ConfiguredBoardExecutionSlices",
) -> int:
    """Return the small immutable recovery budget for one compiled wave."""

    return min(
        MAX_PLAN_BOUND_WAVE_TRANSFERS,
        max(1, len(manifest.nonempty)),
    )


def _string_set(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = (value,) if isinstance(value, str) else value
    try:
        return tuple(sorted({_text(item, "metadata") for item in values}))
    except TypeError as exc:
        raise ExecutionPlanError("metadata must be a sequence") from exc


@dataclass(frozen=True)
class ExecutionTask:
    """Canonical scheduler fields extracted from an admitted task record."""

    task_cid: str
    dependencies: tuple[str, ...] = ()
    declared_paths: tuple[str, ...] = ()
    scope_paths: tuple[str, ...] = ()
    resource_class: str = ""
    provider_id: str = ""
    validation_keys: tuple[str, ...] = ()
    exclusive_keys: tuple[str, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "dependencies", _string_set(self.dependencies))
        object.__setattr__(self, "declared_paths", _paths(self.declared_paths, "declared_paths"))
        object.__setattr__(self, "scope_paths", _paths(self.scope_paths, "scope_paths"))
        object.__setattr__(self, "validation_keys", _string_set(self.validation_keys))
        object.__setattr__(self, "exclusive_keys", _string_set(self.exclusive_keys))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ExecutionPlanError("priority must be an integer")

    @property
    def mutation_scope(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.declared_paths) | set(self.scope_paths)))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "ExecutionTask":
        resource = value.get("resource_contract") if isinstance(value.get("resource_contract"), Mapping) else {}
        provider = value.get("provider_contract") if isinstance(value.get("provider_contract"), Mapping) else {}
        conflict = value.get("conflict_contract") if isinstance(value.get("conflict_contract"), Mapping) else {}
        task_cid = value.get("task_cid") or value.get("canonical_task_cid") or value.get("task_id") or value.get("id")
        dependencies = value.get("dependency_task_cids") or value.get("depends_on") or value.get("dependencies") or ()
        paths = value.get("outputs") or value.get("predicted_files") or value.get("predicted_paths") or ()
        scopes = value.get("scope_paths") or value.get("affected_paths") or conflict.get("scope_paths") or ()
        raw_priority = value.get("priority_rank", value.get("priority", 0))
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            # Repository boards commonly use P0..P4 while materialized plans
            # use a numeric rank.  Preserve their ordering without making a
            # presentation label a source of non-determinism.
            label = str(raw_priority or "").strip().upper()
            priority = 4 - int(label[1:]) if len(label) == 2 and label[0] == "P" and label[1:].isdigit() else 0
        return cls(
            task_cid=str(task_cid or ""), dependencies=tuple(dependencies),
            declared_paths=paths, scope_paths=scopes,
            resource_class=str(value.get("resource_class") or resource.get("resource_class") or ""),
            provider_id=str(value.get("provider_id") or provider.get("provider_id") or provider.get("provider") or ""),
            validation_keys=value.get("validation_keys") or value.get("validation_commands") or value.get("validations") or (),
            exclusive_keys=value.get("exclusive_keys") or conflict.get("exclusive_keys") or value.get("exclusive_group") or (),
            priority=priority,
        )


@dataclass(frozen=True)
class CapacitySnapshot:
    """Live capacity evidence consumed for exactly one scheduling pass."""

    snapshot_id: str
    host_lanes: int
    provider_lanes: int
    observed_at_ms: int
    fresh_until_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        for name in ("host_lanes", "provider_lanes", "observed_at_ms", "fresh_until_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionPlanError(f"{name} must be a nonnegative integer")
        if self.fresh_until_ms and self.fresh_until_ms < self.observed_at_ms:
            raise ExecutionPlanError("fresh_until_ms precedes observed_at_ms")

    @property
    def lane_cap(self) -> int:
        return min(self.host_lanes, self.provider_lanes)

    def is_current(self, now_ms: int) -> bool:
        return bool(self.observed_at_ms and (not self.fresh_until_ms or now_ms < self.fresh_until_ms))


@dataclass(frozen=True)
class ExecutionSlice:
    """Exact task CIDs one lane may select for one immutable revision."""

    plan_revision: str
    lane_id: str
    task_cids: tuple[str, ...]
    capacity_snapshot_id: str
    slice_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_revision", _text(self.plan_revision, "plan_revision"))
        object.__setattr__(self, "lane_id", _text(self.lane_id, "lane_id"))
        object.__setattr__(self, "capacity_snapshot_id", _text(self.capacity_snapshot_id, "capacity_snapshot_id"))
        cids = _string_set(self.task_cids)
        object.__setattr__(self, "task_cids", cids)
        expected = _cid("execution-slice", self.to_dict(include_id=False))
        if self.slice_id and self.slice_id != expected:
            raise ExecutionPlanError("execution slice identity does not match its immutable contents")
        object.__setattr__(self, "slice_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {"schema": EXECUTION_SLICE_SCHEMA, "plan_revision": self.plan_revision, "lane_id": self.lane_id, "task_cids": list(self.task_cids), "capacity_snapshot_id": self.capacity_snapshot_id}
        if include_id:
            value["slice_id"] = self.slice_id
        return value


@dataclass(frozen=True)
class AdaptiveExecutionPlan:
    plan_revision: str
    capacity_snapshot_id: str
    requested_lanes: int
    admitted_lanes: int
    ready_task_cids: tuple[str, ...]
    selected_task_cids: tuple[str, ...]
    conflict_pairs: tuple[tuple[str, str], ...]
    slices: tuple[ExecutionSlice, ...]
    plan_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_revision", _text(self.plan_revision, "plan_revision"))
        object.__setattr__(self, "capacity_snapshot_id", _text(self.capacity_snapshot_id, "capacity_snapshot_id"))
        for name in ("requested_lanes", "admitted_lanes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionPlanError(f"{name} must be a nonnegative integer")
        if self.admitted_lanes != len(self.selected_task_cids) or len(self.slices) != self.admitted_lanes:
            raise ExecutionPlanError("each admitted task must have one execution slice")
        expected = _cid("adaptive-execution-plan", self.to_dict(include_id=False))
        if self.plan_id and self.plan_id != expected:
            raise ExecutionPlanError("adaptive execution plan identity does not match")
        object.__setattr__(self, "plan_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {"schema": ADAPTIVE_EXECUTION_PLAN_SCHEMA, "plan_revision": self.plan_revision, "capacity_snapshot_id": self.capacity_snapshot_id, "requested_lanes": self.requested_lanes, "admitted_lanes": self.admitted_lanes, "ready_task_cids": list(self.ready_task_cids), "selected_task_cids": list(self.selected_task_cids), "conflict_pairs": [list(pair) for pair in self.conflict_pairs], "slices": [item.to_dict() for item in self.slices]}
        if include_id:
            value["plan_id"] = self.plan_id
        return value


@dataclass(frozen=True)
class ConfiguredBoardExecutionSlice:
    """One exact, non-authoritative projection of a CAS-owned lane slice."""

    lane_index: int
    lane_id: str
    task_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    plan_root_cid: str
    compiler_plan_id: str
    capacity_snapshot_id: str
    repository_tree_id: str
    slice_id: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.lane_index, bool)
            or not isinstance(self.lane_index, int)
            or self.lane_index < 0
        ):
            raise ExecutionPlanError("lane_index must be a nonnegative integer")
        object.__setattr__(self, "lane_id", _text(self.lane_id, "lane_id"))
        raw_ids = tuple(self.task_ids)
        raw_cids = tuple(self.task_cids)
        if len(raw_ids) != len(raw_cids):
            raise ExecutionPlanError("slice task ID/CID populations disagree")
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in (*raw_ids, *raw_cids)
        ):
            raise ExecutionPlanError("slice task ID/CID pairs must be exact text")
        pairs = tuple(sorted(zip(raw_ids, raw_cids, strict=True)))
        if (
            len({task_id for task_id, _task_cid in pairs}) != len(pairs)
            or len({task_cid for _task_id, task_cid in pairs}) != len(pairs)
        ):
            raise ExecutionPlanError("slice task ID/CID pairs must be unique")
        object.__setattr__(self, "task_ids", tuple(pair[0] for pair in pairs))
        object.__setattr__(self, "task_cids", tuple(pair[1] for pair in pairs))
        for field_name in (
            "plan_root_cid", "compiler_plan_id", "capacity_snapshot_id",
            "repository_tree_id",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        expected = _cid("configured-board-execution-slice", self.to_dict(include_id=False))
        if self.slice_id and self.slice_id != expected:
            raise ExecutionPlanError("configured slice identity does not match")
        object.__setattr__(self, "slice_id", expected)

    @property
    def empty(self) -> bool:
        return not self.task_ids

    @property
    def task_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.task_ids, self.task_cids, strict=True))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "lane_index": self.lane_index,
            "lane_id": self.lane_id,
            "task_ids": list(self.task_ids),
            "task_cids": list(self.task_cids),
            "plan_root_cid": self.plan_root_cid,
            "compiler_plan_id": self.compiler_plan_id,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "repository_tree_id": self.repository_tree_id,
        }
        if include_id:
            payload["slice_id"] = self.slice_id
        return payload


@dataclass(frozen=True)
class ConfiguredBoardExecutionSlices:
    """Complete immutable first-wave slice manifest stored in canonical CAS."""

    board_namespace: str
    plan_root_cid: str
    compiler_plan_id: str
    capacity_snapshot_id: str
    repository_tree_id: str
    source_head: str
    task_source_revision: str
    configuration_root: str
    slices: tuple[ConfiguredBoardExecutionSlice, ...]
    wave_index: int = 0
    schema: str = CONFIGURED_BOARD_EXECUTION_SLICES_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "board_namespace", "plan_root_cid", "compiler_plan_id",
            "capacity_snapshot_id", "repository_tree_id", "source_head",
            "task_source_revision",
            "configuration_root",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        if self.schema != CONFIGURED_BOARD_EXECUTION_SLICES_SCHEMA:
            raise ExecutionPlanError("unsupported configured slice manifest schema")
        if (
            isinstance(self.wave_index, bool)
            or not isinstance(self.wave_index, int)
            or self.wave_index < 0
        ):
            raise ExecutionPlanError("wave_index must be a nonnegative integer")
        slices = tuple(self.slices)
        if any(not isinstance(item, ConfiguredBoardExecutionSlice) for item in slices):
            raise ExecutionPlanError("configured manifest slices must be typed slices")
        if len({item.lane_index for item in slices}) != len(slices):
            raise ExecutionPlanError("configured slice lane indices are duplicated")
        if len({item.lane_id for item in slices}) != len(slices):
            raise ExecutionPlanError("configured slice lane IDs are duplicated")
        if len({task for item in slices for task in item.task_ids}) != sum(
            len(item.task_ids) for item in slices
        ):
            raise ExecutionPlanError("configured slice task IDs are duplicated")
        if len({task for item in slices for task in item.task_cids}) != sum(
            len(item.task_cids) for item in slices
        ):
            raise ExecutionPlanError("configured slice task CIDs are duplicated")
        for item in slices:
            if (
                item.plan_root_cid != self.plan_root_cid
                or item.compiler_plan_id != self.compiler_plan_id
                or item.capacity_snapshot_id != self.capacity_snapshot_id
                or item.repository_tree_id != self.repository_tree_id
            ):
                raise ExecutionPlanError(
                    "configured slice carries mixed manifest authority"
                )
        object.__setattr__(self, "slices", tuple(sorted(slices, key=lambda item: item.lane_index)))

    @property
    def nonempty(self) -> tuple[ConfiguredBoardExecutionSlice, ...]:
        return tuple(item for item in self.slices if not item.empty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "board_namespace": self.board_namespace,
            "plan_root_cid": self.plan_root_cid,
            "compiler_plan_id": self.compiler_plan_id,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "repository_tree_id": self.repository_tree_id,
            "source_head": self.source_head,
            "task_source_revision": self.task_source_revision,
            "configuration_root": self.configuration_root,
            "wave_index": self.wave_index,
            "slices": [item.to_dict() for item in self.slices],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfiguredBoardExecutionSlices":
        if not isinstance(payload, Mapping):
            raise ExecutionPlanError("configured slice manifest must be an object")
        expected_fields = {
            "schema",
            "board_namespace",
            "plan_root_cid",
            "compiler_plan_id",
            "capacity_snapshot_id",
            "repository_tree_id",
            "source_head",
            "task_source_revision",
            "configuration_root",
            "wave_index",
            "slices",
        }
        if set(payload) != expected_fields:
            raise ExecutionPlanError(
                "configured slice manifest fields are not exact"
            )
        text_fields = expected_fields - {"wave_index", "slices"}
        if any(not isinstance(payload[name], str) for name in text_fields):
            raise ExecutionPlanError(
                "configured slice manifest text fields have invalid types"
            )
        wave_index = payload["wave_index"]
        if (
            isinstance(wave_index, bool)
            or not isinstance(wave_index, int)
            or wave_index < 0
        ):
            raise ExecutionPlanError("wave_index must be a nonnegative integer")
        raw_slices = payload["slices"]
        if not isinstance(raw_slices, list):
            raise ExecutionPlanError("configured manifest slices must be a list")
        slice_fields = {
            "lane_index",
            "lane_id",
            "task_ids",
            "task_cids",
            "plan_root_cid",
            "compiler_plan_id",
            "capacity_snapshot_id",
            "repository_tree_id",
            "slice_id",
        }
        parsed_slices: list[ConfiguredBoardExecutionSlice] = []
        for item in raw_slices:
            if not isinstance(item, Mapping):
                raise ExecutionPlanError("configured manifest slice must be an object")
            if set(item) != slice_fields:
                raise ExecutionPlanError("configured slice fields are not exact")
            lane_index = item["lane_index"]
            if (
                isinstance(lane_index, bool)
                or not isinstance(lane_index, int)
                or lane_index < 0
            ):
                raise ExecutionPlanError(
                    "lane_index must be a nonnegative integer"
                )
            if any(
                not isinstance(item[name], str)
                for name in slice_fields - {"lane_index", "task_ids", "task_cids"}
            ):
                raise ExecutionPlanError(
                    "configured slice text fields have invalid types"
                )
            for name in ("task_ids", "task_cids"):
                if not isinstance(item[name], list) or any(
                    not isinstance(value, str) for value in item[name]
                ):
                    raise ExecutionPlanError(
                        f"configured slice {name} must be a string list"
                    )
            parsed_slices.append(
                ConfiguredBoardExecutionSlice(
                    lane_index=lane_index,
                    lane_id=item["lane_id"],
                    task_ids=tuple(item["task_ids"]),
                    task_cids=tuple(item["task_cids"]),
                    plan_root_cid=item["plan_root_cid"],
                    compiler_plan_id=item["compiler_plan_id"],
                    capacity_snapshot_id=item["capacity_snapshot_id"],
                    repository_tree_id=item["repository_tree_id"],
                    slice_id=item["slice_id"],
                )
            )
        result = cls(
            board_namespace=payload["board_namespace"],
            plan_root_cid=payload["plan_root_cid"],
            compiler_plan_id=payload["compiler_plan_id"],
            capacity_snapshot_id=payload["capacity_snapshot_id"],
            repository_tree_id=payload["repository_tree_id"],
            source_head=payload["source_head"],
            task_source_revision=payload["task_source_revision"],
            configuration_root=payload["configuration_root"],
            wave_index=wave_index,
            slices=tuple(parsed_slices),
            schema=payload["schema"],
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "configured slice manifest failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanSliceReassignment:
    """One same-revision CAS transfer of a fenced, unclaimed slice."""

    revision_cid: str
    plan_root_cid: str
    slice_manifest_cid: str
    slice_id: str
    donor_lane_id: str
    recipient_lane_id: str
    task_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    generation: int
    prior_reassignment_cid: str
    donor_process_birth_cid: str
    attempt_absence_cid: str
    claim_absence_cid: str
    reassignment_id: str = ""
    schema: str = PLAN_SLICE_REASSIGNMENT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "revision_cid",
            "plan_root_cid",
            "slice_manifest_cid",
            "slice_id",
            "donor_lane_id",
            "recipient_lane_id",
            "donor_process_birth_cid",
            "attempt_absence_cid",
            "claim_absence_cid",
        ):
            object.__setattr__(
                self, field_name, _text(getattr(self, field_name), field_name)
            )
        if self.donor_lane_id == self.recipient_lane_id:
            raise ExecutionPlanError("slice reassignment requires a new lane owner")
        if self.schema != PLAN_SLICE_REASSIGNMENT_SCHEMA:
            raise ExecutionPlanError("unsupported slice reassignment schema")
        if not isinstance(self.prior_reassignment_cid, str):
            raise ExecutionPlanError("prior_reassignment_cid must be text")
        task_ids = tuple(_text(item, "task_id") for item in self.task_ids)
        task_cids = tuple(_text(item, "task_cid") for item in self.task_cids)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "task_cids", task_cids)
        if (
            not task_ids
            or len(task_ids) != len(task_cids)
            or len(set(task_ids)) != len(task_ids)
            or len(set(task_cids)) != len(task_cids)
        ):
            raise ExecutionPlanError(
                "slice reassignment requires one exact nonempty ID/CID population"
            )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 1 <= self.generation <= MAX_SLICE_REASSIGNMENTS
        ):
            raise ExecutionPlanError("slice reassignment generation is invalid")
        expected = _cid("plan-slice-reassignment", self.to_dict(include_id=False))
        if self.reassignment_id and self.reassignment_id != expected:
            raise ExecutionPlanError("slice reassignment identity does not match")
        object.__setattr__(self, "reassignment_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "donor_lane_id": self.donor_lane_id,
            "recipient_lane_id": self.recipient_lane_id,
            "task_ids": list(self.task_ids),
            "task_cids": list(self.task_cids),
            "generation": self.generation,
            "prior_reassignment_cid": self.prior_reassignment_cid,
            "donor_process_birth_cid": self.donor_process_birth_cid,
            "attempt_absence_cid": self.attempt_absence_cid,
            "claim_absence_cid": self.claim_absence_cid,
        }
        if include_id:
            payload["reassignment_id"] = self.reassignment_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSliceReassignment":
        if not isinstance(payload, Mapping):
            raise ExecutionPlanError("slice reassignment must be an object")
        expected_fields = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "slice_manifest_cid",
            "slice_id",
            "donor_lane_id",
            "recipient_lane_id",
            "task_ids",
            "task_cids",
            "generation",
            "prior_reassignment_cid",
            "donor_process_birth_cid",
            "attempt_absence_cid",
            "claim_absence_cid",
            "reassignment_id",
        }
        if set(payload) != expected_fields:
            raise ExecutionPlanError("slice reassignment fields are not exact")
        if payload["schema"] != PLAN_SLICE_REASSIGNMENT_SCHEMA:
            raise ExecutionPlanError("unsupported slice reassignment schema")
        text_fields = expected_fields - {"task_ids", "task_cids", "generation"}
        if any(not isinstance(payload[name], str) for name in text_fields):
            raise ExecutionPlanError(
                "slice reassignment text fields have invalid types"
            )
        for name in ("task_ids", "task_cids"):
            if not isinstance(payload[name], list) or any(
                not isinstance(value, str) for value in payload[name]
            ):
                raise ExecutionPlanError(
                    f"slice reassignment {name} must be a string list"
                )
        generation = payload["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ExecutionPlanError("slice reassignment generation is invalid")
        result = cls(
            revision_cid=payload["revision_cid"],
            plan_root_cid=payload["plan_root_cid"],
            slice_manifest_cid=payload["slice_manifest_cid"],
            slice_id=payload["slice_id"],
            donor_lane_id=payload["donor_lane_id"],
            recipient_lane_id=payload["recipient_lane_id"],
            task_ids=tuple(payload["task_ids"]),
            task_cids=tuple(payload["task_cids"]),
            generation=generation,
            prior_reassignment_cid=payload["prior_reassignment_cid"],
            donor_process_birth_cid=payload["donor_process_birth_cid"],
            attempt_absence_cid=payload["attempt_absence_cid"],
            claim_absence_cid=payload["claim_absence_cid"],
            reassignment_id=payload["reassignment_id"],
            schema=payload["schema"],
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "slice reassignment failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundProcessBirth:
    """One bounded, immutable supervisor process-birth generation."""

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    task_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    configuration_root: str
    accepted_tree_root: str
    profile: Mapping[str, Any]
    process_birth: Mapping[str, Any]
    generation: int
    global_budget: int
    prior_process_birth_cid: str = ""
    schema: str = PLAN_BOUND_PROCESS_BIRTH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLAN_BOUND_PROCESS_BIRTH_SCHEMA:
            raise ExecutionPlanError("unsupported plan-bound process-birth schema")
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "configuration_root",
            "accepted_tree_root",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if len(tuple(self.task_ids)) != len(tuple(self.task_cids)):
            raise ExecutionPlanError(
                "plan-bound process-birth task pairs are partial"
            )
        pairs = tuple(zip(tuple(self.task_ids), tuple(self.task_cids), strict=True))
        if (
            not pairs
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for pair in pairs
                for value in pair
            )
            or len({item[0] for item in pairs}) != len(pairs)
            or len({item[1] for item in pairs}) != len(pairs)
        ):
            raise ExecutionPlanError("plan-bound process-birth task pairs are invalid")
        if not isinstance(self.profile, Mapping) or not self.profile:
            raise ExecutionPlanError("plan-bound process-birth profile is invalid")
        if not isinstance(self.process_birth, Mapping) or not self.process_birth:
            raise ExecutionPlanError("plan-bound process-birth identity is invalid")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 0 <= self.generation <= MAX_PLAN_BOUND_WAVE_TRANSFERS
            or isinstance(self.global_budget, bool)
            or not isinstance(self.global_budget, int)
            or self.global_budget != MAX_PLAN_BOUND_WAVE_TRANSFERS
            or bool(self.prior_process_birth_cid) != (self.generation > 0)
            or not isinstance(self.prior_process_birth_cid, str)
        ):
            raise ExecutionPlanError("plan-bound process-birth generation is invalid")
        object.__setattr__(self, "task_ids", tuple(item[0] for item in pairs))
        object.__setattr__(self, "task_cids", tuple(item[1] for item in pairs))
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "process_birth", dict(self.process_birth))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "task_ids": list(self.task_ids),
            "task_cids": list(self.task_cids),
            "configuration_root": self.configuration_root,
            "accepted_tree_root": self.accepted_tree_root,
            "profile": dict(self.profile),
            "process_birth": dict(self.process_birth),
            "generation": self.generation,
            "global_budget": self.global_budget,
            "prior_process_birth_cid": self.prior_process_birth_cid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanBoundProcessBirth":
        if not isinstance(value, Mapping):
            raise ExecutionPlanError("plan-bound process-birth payload is not an object")
        payload = dict(value)
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "task_ids",
            "task_cids",
            "configuration_root",
            "accepted_tree_root",
            "profile",
            "process_birth",
            "generation",
            "global_budget",
            "prior_process_birth_cid",
        }
        if set(payload) != expected:
            raise ExecutionPlanError("plan-bound process-birth fields are not exact")
        text_fields = expected - {
            "task_ids",
            "task_cids",
            "profile",
            "process_birth",
            "generation",
            "global_budget",
        }
        if (
            any(not isinstance(payload[name], str) for name in text_fields)
            or any(not isinstance(payload[name], list) for name in ("task_ids", "task_cids"))
            or any(
                not isinstance(item, str)
                for name in ("task_ids", "task_cids")
                for item in payload[name]
            )
            or not isinstance(payload["profile"], Mapping)
            or not isinstance(payload["process_birth"], Mapping)
            or any(
                isinstance(payload[name], bool) or not isinstance(payload[name], int)
                for name in ("generation", "global_budget")
            )
        ):
            raise ExecutionPlanError("plan-bound process-birth scalar types are invalid")
        result = cls(
            revision_cid=payload["revision_cid"],
            plan_root_cid=payload["plan_root_cid"],
            execution_plan_cid=payload["execution_plan_cid"],
            capacity_snapshot_id=payload["capacity_snapshot_id"],
            slice_manifest_cid=payload["slice_manifest_cid"],
            slice_id=payload["slice_id"],
            lane_id=payload["lane_id"],
            task_ids=tuple(payload["task_ids"]),
            task_cids=tuple(payload["task_cids"]),
            configuration_root=payload["configuration_root"],
            accepted_tree_root=payload["accepted_tree_root"],
            profile=payload["profile"],
            process_birth=payload["process_birth"],
            generation=payload["generation"],
            global_budget=payload["global_budget"],
            prior_process_birth_cid=payload["prior_process_birth_cid"],
            schema=payload["schema"],
        )
        if result.to_dict() != payload:
            raise ExecutionPlanError("plan-bound process-birth normalized during decode")
        return result


@dataclass(frozen=True)
class PlanBoundExecutionLease:
    """Canonical-store bridge from a compiled slice to its real effects.

    The compiler's lease/worktree/fence values are immutable authority, while
    the implementation daemon's canonical task claim and worktree lifecycle
    records are the real effect guards.  This record joins those owners without
    replacing either one.  Every transition is a new CAS object linked to the
    prior generation and selected through one guarded continuation pointer.
    """

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    reassignment_cid: str
    task_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    compiled_task_bindings: tuple[Mapping[str, Any], ...]
    process_birth_cid: str
    process_birth: Mapping[str, Any]
    generation: int
    phase: str
    prior_execution_lease_cid: str = ""
    active_task_id: str = ""
    active_task_cid: str = ""
    daemon_process_birth: Mapping[str, Any] | None = None
    canonical_claim_path: str = ""
    canonical_claim_cid: str = ""
    canonical_claim_lease_id: str = ""
    workspace_lifecycle_path: str = ""
    workspace_lifecycle_cid: str = ""
    workspace_record_id: str = ""
    workspace_path: str = ""
    workspace_lease_id: str = ""
    workspace_fence: int = 0
    provider_ready: bool = False
    proposal_id: str = ""
    proposal_receipt_id: str = ""
    proposal_reason_codes: tuple[str, ...] = ()
    actual_changed_paths: tuple[str, ...] = ()
    merge_enqueue_reached: bool = False
    proposal_handoff_cid: str = ""
    merge_authorization_cid: str = ""
    merge_enqueue_intent_cid: str = ""
    merge_request_id: str = ""
    merge_queue_receipt_cid: str = ""
    schema: str = PLAN_BOUND_EXECUTION_LEASE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "process_birth_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.schema != PLAN_BOUND_EXECUTION_LEASE_SCHEMA:
            raise ExecutionPlanError("unsupported plan-bound execution lease schema")
        if self.phase not in _PLAN_BOUND_EXECUTION_LEASE_PHASES:
            raise ExecutionPlanError("plan-bound execution lease phase is invalid")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 1 <= self.generation <= MAX_SLICE_REASSIGNMENTS
        ):
            raise ExecutionPlanError("plan-bound execution lease generation is invalid")
        if not isinstance(self.prior_execution_lease_cid, str):
            raise ExecutionPlanError("prior execution lease CID must be text")
        if not isinstance(self.reassignment_cid, str):
            raise ExecutionPlanError("execution lease reassignment CID must be text")
        if (
            isinstance(self.workspace_fence, bool)
            or not isinstance(self.workspace_fence, int)
            or self.workspace_fence < 0
            or not isinstance(self.provider_ready, bool)
            or not isinstance(self.merge_enqueue_reached, bool)
        ):
            raise ExecutionPlanError("plan-bound execution effect fields are invalid")
        if not isinstance(self.process_birth, Mapping) or not self.process_birth:
            raise ExecutionPlanError("plan-bound execution lease requires process birth")
        daemon_birth = self.daemon_process_birth
        if daemon_birth is None:
            daemon_birth = {}
        if not isinstance(daemon_birth, Mapping):
            raise ExecutionPlanError("daemon process birth must be an object")
        object.__setattr__(self, "process_birth", dict(self.process_birth))
        object.__setattr__(self, "daemon_process_birth", dict(daemon_birth))

        raw_ids = tuple(self.task_ids)
        raw_cids = tuple(self.task_cids)
        if (
            not raw_ids
            or len(raw_ids) != len(raw_cids)
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in (*raw_ids, *raw_cids)
            )
            or len(set(raw_ids)) != len(raw_ids)
            or len(set(raw_cids)) != len(raw_cids)
        ):
            raise ExecutionPlanError(
                "plan-bound execution lease task pairs are invalid"
            )
        object.__setattr__(self, "task_ids", raw_ids)
        object.__setattr__(self, "task_cids", raw_cids)

        bindings: list[dict[str, Any]] = []
        for raw, task_id, task_cid in zip(
            self.compiled_task_bindings,
            raw_ids,
            raw_cids,
            strict=True,
        ):
            if not isinstance(raw, Mapping) or set(raw) != {
                "task_id",
                "task_cid",
                "assignment",
            }:
                raise ExecutionPlanError("compiled task binding fields are not exact")
            assignment = raw.get("assignment")
            if (
                raw.get("task_id") != task_id
                or raw.get("task_cid") != task_cid
                or not isinstance(assignment, Mapping)
                or set(assignment) != _COMPILED_ASSIGNMENT_FIELDS
                or assignment.get("task_id") != task_id
            ):
                raise ExecutionPlanError("compiled task binding is mixed")
            string_fields = _COMPILED_ASSIGNMENT_FIELDS - {
                "exclusive_paths",
                "lease_duration_ms",
                "heartbeat_interval_ms",
                "fence_epoch",
            }
            if any(not isinstance(assignment[name], str) for name in string_fields):
                raise ExecutionPlanError("compiled assignment text fields are invalid")
            for name in ("worktree_id", "worktree_path", "lease_id", "fence_token"):
                if not assignment[name]:
                    raise ExecutionPlanError(
                        f"compiled assignment requires {name}"
                    )
            if not isinstance(assignment["exclusive_paths"], list) or any(
                not isinstance(item, str) for item in assignment["exclusive_paths"]
            ):
                raise ExecutionPlanError("compiled exclusive paths are invalid")
            for name in (
                "lease_duration_ms",
                "heartbeat_interval_ms",
                "fence_epoch",
            ):
                value = assignment[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ExecutionPlanError(
                        f"compiled assignment {name} is invalid"
                    )
            bindings.append(
                {
                    "task_id": task_id,
                    "task_cid": task_cid,
                    "assignment": dict(assignment),
                }
            )
        object.__setattr__(self, "compiled_task_bindings", tuple(bindings))

        effect_text_fields = (
            "active_task_id",
            "active_task_cid",
            "canonical_claim_path",
            "canonical_claim_cid",
            "canonical_claim_lease_id",
            "workspace_lifecycle_path",
            "workspace_lifecycle_cid",
            "workspace_record_id",
            "workspace_path",
            "workspace_lease_id",
            "proposal_id",
            "proposal_receipt_id",
            "proposal_handoff_cid",
            "merge_authorization_cid",
            "merge_enqueue_intent_cid",
            "merge_request_id",
            "merge_queue_receipt_cid",
        )
        if any(not isinstance(getattr(self, name), str) for name in effect_text_fields):
            raise ExecutionPlanError("plan-bound execution effect fields must be text")
        task_pair = (self.active_task_id, self.active_task_cid)
        if bool(task_pair[0]) != bool(task_pair[1]):
            raise ExecutionPlanError("active task ID/CID pair is partial")
        if task_pair[0] and task_pair not in set(zip(raw_ids, raw_cids, strict=True)):
            raise ExecutionPlanError("active task ID/CID pair is outside the slice")
        object.__setattr__(
            self,
            "proposal_reason_codes",
            _string_set(self.proposal_reason_codes),
        )
        object.__setattr__(
            self,
            "actual_changed_paths",
            _paths(self.actual_changed_paths, "actual_changed_paths"),
        )

        claim_fields = (
            self.canonical_claim_path,
            self.canonical_claim_cid,
            self.canonical_claim_lease_id,
        )
        workspace_fields = (
            self.workspace_lifecycle_path,
            self.workspace_lifecycle_cid,
            self.workspace_record_id,
            self.workspace_path,
            self.workspace_lease_id,
        )
        merge_handoff_fields = (
            self.merge_authorization_cid,
            self.merge_enqueue_intent_cid,
            self.merge_request_id,
            self.merge_queue_receipt_cid,
        )
        changed_proposal = bool(
            self.proposal_id
            and self.proposal_receipt_id
            and self.actual_changed_paths
        )
        no_change_proposal = bool(
            not self.proposal_id
            and not self.proposal_receipt_id
            and not self.actual_changed_paths
        )
        if self.phase == "reserved":
            if (
                any(task_pair)
                or any(claim_fields)
                or any(workspace_fields)
                or self.daemon_process_birth
                or self.workspace_fence
                or self.provider_ready
                or self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError("reserved execution lease carries effects")
        elif self.phase == "claimed":
            if (
                not all(task_pair)
                or not all(claim_fields)
                or not self.daemon_process_birth
                or any(workspace_fields)
                or self.workspace_fence
                or self.provider_ready
                or self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError("claimed execution lease is partial")
        else:
            if (
                not all(task_pair)
                or not all(claim_fields)
                or not self.daemon_process_birth
                or not all(workspace_fields)
                or self.workspace_fence < 1
            ):
                raise ExecutionPlanError("workspace execution lease is partial")
            if self.provider_ready != (
                self.phase
                in {
                    "provider_ready",
                    "proposal_ready",
                    "merge_enqueue_reached",
                    "merge_enqueue_prepared",
                    "merge_enqueue_confirmed",
                    "merge_completed",
                    "proposal_rejected",
                    "scope_drift",
                }
            ):
                raise ExecutionPlanError("workspace provider-ready phase is mixed")
            if self.phase == "workspace_prepared" and (
                self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError("prepared workspace carries proposal result")
            if self.phase == "provider_ready" and (
                self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError("provider-ready lease carries proposal result")
            if self.phase == "proposal_ready":
                if (
                    not (changed_proposal or no_change_proposal)
                    or self.proposal_reason_codes
                    or self.merge_enqueue_reached
                    or not self.proposal_handoff_cid
                    or any(merge_handoff_fields)
                ):
                    raise ExecutionPlanError(
                        "proposal-ready execution lease is partial"
                    )
            if self.phase == "merge_enqueue_reached" and (
                not (changed_proposal or no_change_proposal)
                or self.proposal_reason_codes
                or not self.merge_enqueue_reached
                or not self.proposal_handoff_cid
                or not self.merge_authorization_cid
                or any(merge_handoff_fields[1:])
            ):
                raise ExecutionPlanError(
                    "merge-enqueue execution lease is partial"
                )
            if self.phase == "merge_enqueue_prepared" and (
                not (changed_proposal or no_change_proposal)
                or self.proposal_reason_codes
                or not self.merge_enqueue_reached
                or not self.proposal_handoff_cid
                or not all(merge_handoff_fields[:2])
                or any(merge_handoff_fields[2:])
            ):
                raise ExecutionPlanError(
                    "merge-enqueue prepared lease is partial"
                )
            if self.phase == "merge_enqueue_confirmed" and (
                not (changed_proposal or no_change_proposal)
                or self.proposal_reason_codes
                or not self.merge_enqueue_reached
                or not self.proposal_handoff_cid
                or not all(merge_handoff_fields)
            ):
                raise ExecutionPlanError(
                    "merge-enqueue confirmed lease is partial"
                )
            if self.phase == "merge_completed" and (
                not (changed_proposal or no_change_proposal)
                or self.proposal_reason_codes
                or not self.merge_enqueue_reached
                or not self.proposal_handoff_cid
                or not all(merge_handoff_fields)
            ):
                raise ExecutionPlanError(
                    "merge-completed execution lease is partial"
                )
            if self.phase == "proposal_rejected" and (
                not self.proposal_id
                or not self.proposal_receipt_id
                or not self.proposal_reason_codes
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError(
                    "proposal-rejected execution lease is partial"
                )
            if self.phase == "scope_drift" and (
                not self.proposal_id
                or not self.proposal_receipt_id
                or "path_outside_scope" not in self.proposal_reason_codes
                or not self.actual_changed_paths
                or self.merge_enqueue_reached
                or self.proposal_handoff_cid
                or any(merge_handoff_fields)
            ):
                raise ExecutionPlanError("scope-drift execution lease is partial")

    def assignment_for(self, task_id: str, task_cid: str) -> Mapping[str, Any]:
        for binding in self.compiled_task_bindings:
            if binding["task_id"] == task_id and binding["task_cid"] == task_cid:
                return dict(binding["assignment"])
        raise ExecutionPlanError("task has no compiled execution lease assignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "reassignment_cid": self.reassignment_cid,
            "task_ids": list(self.task_ids),
            "task_cids": list(self.task_cids),
            "compiled_task_bindings": [
                {
                    "task_id": item["task_id"],
                    "task_cid": item["task_cid"],
                    "assignment": dict(item["assignment"]),
                }
                for item in self.compiled_task_bindings
            ],
            "process_birth_cid": self.process_birth_cid,
            "process_birth": dict(self.process_birth),
            "generation": self.generation,
            "phase": self.phase,
            "prior_execution_lease_cid": self.prior_execution_lease_cid,
            "active_task_id": self.active_task_id,
            "active_task_cid": self.active_task_cid,
            "daemon_process_birth": dict(self.daemon_process_birth or {}),
            "canonical_claim_path": self.canonical_claim_path,
            "canonical_claim_cid": self.canonical_claim_cid,
            "canonical_claim_lease_id": self.canonical_claim_lease_id,
            "workspace_lifecycle_path": self.workspace_lifecycle_path,
            "workspace_lifecycle_cid": self.workspace_lifecycle_cid,
            "workspace_record_id": self.workspace_record_id,
            "workspace_path": self.workspace_path,
            "workspace_lease_id": self.workspace_lease_id,
            "workspace_fence": self.workspace_fence,
            "provider_ready": self.provider_ready,
            "proposal_id": self.proposal_id,
            "proposal_receipt_id": self.proposal_receipt_id,
            "proposal_reason_codes": list(self.proposal_reason_codes),
            "actual_changed_paths": list(self.actual_changed_paths),
            "merge_enqueue_reached": self.merge_enqueue_reached,
            "proposal_handoff_cid": self.proposal_handoff_cid,
            "merge_authorization_cid": self.merge_authorization_cid,
            "merge_enqueue_intent_cid": self.merge_enqueue_intent_cid,
            "merge_request_id": self.merge_request_id,
            "merge_queue_receipt_cid": self.merge_queue_receipt_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBoundExecutionLease":
        if not isinstance(payload, Mapping):
            raise ExecutionPlanError("plan-bound execution lease must be an object")
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "task_ids",
            "task_cids",
            "compiled_task_bindings",
            "process_birth_cid",
            "process_birth",
            "generation",
            "phase",
            "prior_execution_lease_cid",
            "active_task_id",
            "active_task_cid",
            "daemon_process_birth",
            "canonical_claim_path",
            "canonical_claim_cid",
            "canonical_claim_lease_id",
            "workspace_lifecycle_path",
            "workspace_lifecycle_cid",
            "workspace_record_id",
            "workspace_path",
            "workspace_lease_id",
            "workspace_fence",
            "provider_ready",
            "proposal_id",
            "proposal_receipt_id",
            "proposal_reason_codes",
            "actual_changed_paths",
            "merge_enqueue_reached",
            "proposal_handoff_cid",
            "merge_authorization_cid",
            "merge_enqueue_intent_cid",
            "merge_request_id",
            "merge_queue_receipt_cid",
        }
        if set(payload) != expected:
            raise ExecutionPlanError("plan-bound execution lease fields are not exact")
        list_fields = {
            "task_ids",
            "task_cids",
            "compiled_task_bindings",
            "proposal_reason_codes",
            "actual_changed_paths",
        }
        mapping_fields = {"process_birth", "daemon_process_birth"}
        int_fields = {"generation", "workspace_fence"}
        bool_fields = {"provider_ready", "merge_enqueue_reached"}
        text_fields = expected - list_fields - mapping_fields - int_fields - bool_fields
        if any(not isinstance(payload[name], str) for name in text_fields):
            raise ExecutionPlanError("plan-bound execution lease text fields are invalid")
        if any(not isinstance(payload[name], list) for name in list_fields):
            raise ExecutionPlanError("plan-bound execution lease list fields are invalid")
        if any(
            any(not isinstance(item, str) for item in payload[name])
            for name in (
                "task_ids",
                "task_cids",
                "proposal_reason_codes",
                "actual_changed_paths",
            )
        ):
            raise ExecutionPlanError(
                "plan-bound execution lease string-list fields are invalid"
            )
        if any(not isinstance(payload[name], Mapping) for name in mapping_fields):
            raise ExecutionPlanError("plan-bound execution lease object fields are invalid")
        if any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in int_fields
        ) or any(not isinstance(payload[name], bool) for name in bool_fields):
            raise ExecutionPlanError("plan-bound execution lease scalar types are invalid")
        result = cls(
            revision_cid=payload["revision_cid"],
            plan_root_cid=payload["plan_root_cid"],
            execution_plan_cid=payload["execution_plan_cid"],
            capacity_snapshot_id=payload["capacity_snapshot_id"],
            slice_manifest_cid=payload["slice_manifest_cid"],
            slice_id=payload["slice_id"],
            lane_id=payload["lane_id"],
            reassignment_cid=payload["reassignment_cid"],
            task_ids=tuple(payload["task_ids"]),
            task_cids=tuple(payload["task_cids"]),
            compiled_task_bindings=tuple(payload["compiled_task_bindings"]),
            process_birth_cid=payload["process_birth_cid"],
            process_birth=payload["process_birth"],
            generation=payload["generation"],
            phase=payload["phase"],
            prior_execution_lease_cid=payload["prior_execution_lease_cid"],
            active_task_id=payload["active_task_id"],
            active_task_cid=payload["active_task_cid"],
            daemon_process_birth=payload["daemon_process_birth"],
            canonical_claim_path=payload["canonical_claim_path"],
            canonical_claim_cid=payload["canonical_claim_cid"],
            canonical_claim_lease_id=payload["canonical_claim_lease_id"],
            workspace_lifecycle_path=payload["workspace_lifecycle_path"],
            workspace_lifecycle_cid=payload["workspace_lifecycle_cid"],
            workspace_record_id=payload["workspace_record_id"],
            workspace_path=payload["workspace_path"],
            workspace_lease_id=payload["workspace_lease_id"],
            workspace_fence=payload["workspace_fence"],
            provider_ready=payload["provider_ready"],
            proposal_id=payload["proposal_id"],
            proposal_receipt_id=payload["proposal_receipt_id"],
            proposal_reason_codes=tuple(payload["proposal_reason_codes"]),
            actual_changed_paths=tuple(payload["actual_changed_paths"]),
            merge_enqueue_reached=payload["merge_enqueue_reached"],
            proposal_handoff_cid=payload["proposal_handoff_cid"],
            merge_authorization_cid=payload["merge_authorization_cid"],
            merge_enqueue_intent_cid=payload["merge_enqueue_intent_cid"],
            merge_request_id=payload["merge_request_id"],
            merge_queue_receipt_cid=payload["merge_queue_receipt_cid"],
            schema=payload["schema"],
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "plan-bound execution lease failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundRecoveryLaunch:
    """Canonical permission to restart one already-effectful slice.

    This record never authorizes task selection or a provider call.  It only
    lets the sealed launch gate tolerate a clean descendant repository HEAD
    while the exact current execution lease proves that a proposal/merge
    handoff already exists and must be adopted without replay.
    """

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    reassignment_cid: str
    execution_lease_cid: str
    execution_phase: str
    proposal_handoff_cid: str
    merge_authorization_cid: str
    merge_enqueue_intent_cid: str
    merge_request_id: str
    merge_queue_receipt_cid: str
    source_head: str
    source_tree: str
    repository_head: str
    repository_tree: str
    runtime_artifacts: tuple[Mapping[str, Any], ...]
    launch_artifact_paths: tuple[str, ...]
    decision: str = "recover_existing_handoff"
    schema: str = PLAN_BOUND_RECOVERY_LAUNCH_SCHEMA

    def __post_init__(self) -> None:
        text_fields = (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "execution_lease_cid",
            "execution_phase",
            "proposal_handoff_cid",
            "merge_authorization_cid",
            "merge_enqueue_intent_cid",
            "merge_request_id",
            "merge_queue_receipt_cid",
            "source_head",
            "source_tree",
            "repository_head",
            "repository_tree",
            "decision",
            "schema",
        )
        if any(not isinstance(getattr(self, name), str) for name in text_fields):
            raise ExecutionPlanError("recovery-launch fields must be text")
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "execution_lease_cid",
            "proposal_handoff_cid",
        ):
            if not getattr(self, name):
                raise ExecutionPlanError(
                    f"recovery-launch {name} must be nonempty"
                )
        if self.schema != PLAN_BOUND_RECOVERY_LAUNCH_SCHEMA:
            raise ExecutionPlanError("unsupported recovery-launch schema")
        if self.decision != "recover_existing_handoff":
            raise ExecutionPlanError("recovery-launch decision is invalid")
        if self.execution_phase not in {
            "proposal_ready",
            "merge_enqueue_prepared",
            "merge_enqueue_confirmed",
        }:
            raise ExecutionPlanError("recovery-launch phase is not adoptable")
        for name in (
            "source_head",
            "repository_head",
            "source_tree",
            "repository_tree",
        ):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)) is None:
                raise ExecutionPlanError(
                    f"recovery-launch {name} is not a full Git identity"
                )
        merge_fields = (
            self.merge_authorization_cid,
            self.merge_enqueue_intent_cid,
        )
        confirmed_fields = (
            self.merge_request_id,
            self.merge_queue_receipt_cid,
        )
        if self.execution_phase == "proposal_ready" and (
            any(merge_fields) or any(confirmed_fields)
        ):
            raise ExecutionPlanError(
                "proposal-ready recovery carries merge authority"
            )
        if self.execution_phase == "merge_enqueue_prepared" and (
            not all(merge_fields) or any(confirmed_fields)
        ):
            raise ExecutionPlanError(
                "prepared recovery-launch handoff is partial"
            )
        if self.execution_phase == "merge_enqueue_confirmed" and (
            not all(merge_fields) or not all(confirmed_fields)
        ):
            raise ExecutionPlanError(
                "confirmed recovery-launch handoff is partial"
            )
        artifact_fields = {
            "path",
            "kind",
            "sha256",
            "mode",
            "uid",
            "nlink",
            "size",
        }
        artifacts: list[dict[str, Any]] = []
        for raw in self.runtime_artifacts:
            if not isinstance(raw, Mapping) or set(raw) != artifact_fields:
                raise ExecutionPlanError(
                    "recovery-launch runtime artifact fields are not exact"
                )
            artifact = dict(raw)
            if (
                not isinstance(artifact["path"], str)
                or not artifact["path"]
                or artifact["path"] != artifact["path"].strip()
                or PurePosixPath(artifact["path"]).is_absolute()
                or ".." in PurePosixPath(artifact["path"]).parts
                or artifact["kind"] not in {"file", "workspace"}
                or re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["sha256"])
                is None
                or any(
                    isinstance(artifact[name], bool)
                    or not isinstance(artifact[name], int)
                    or int(artifact[name]) < 0
                    for name in ("mode", "uid", "nlink", "size")
                )
                or (
                    artifact["kind"] == "file"
                    and artifact["nlink"] != 1
                )
            ):
                raise ExecutionPlanError(
                    "recovery-launch runtime artifact evidence is invalid"
                )
            artifacts.append(artifact)
        if (
            not artifacts
            or [item["path"] for item in artifacts]
            != sorted(item["path"] for item in artifacts)
            or len({item["path"] for item in artifacts}) != len(artifacts)
        ):
            raise ExecutionPlanError(
                "recovery-launch runtime artifact population is ambiguous"
            )
        object.__setattr__(self, "runtime_artifacts", tuple(artifacts))
        launch_paths = tuple(self.launch_artifact_paths)
        if (
            not launch_paths
            or any(
                not isinstance(path, str)
                or not path
                or path != path.strip()
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or PurePosixPath(path).as_posix() != path
                for path in launch_paths
            )
            or launch_paths != tuple(sorted(launch_paths))
            or len(set(launch_paths)) != len(launch_paths)
            or set(launch_paths) & {item["path"] for item in artifacts}
        ):
            raise ExecutionPlanError(
                "recovery-launch owned artifact paths are ambiguous"
            )
        object.__setattr__(self, "launch_artifact_paths", launch_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision": self.decision,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "reassignment_cid": self.reassignment_cid,
            "execution_lease_cid": self.execution_lease_cid,
            "execution_phase": self.execution_phase,
            "proposal_handoff_cid": self.proposal_handoff_cid,
            "merge_authorization_cid": self.merge_authorization_cid,
            "merge_enqueue_intent_cid": self.merge_enqueue_intent_cid,
            "merge_request_id": self.merge_request_id,
            "merge_queue_receipt_cid": self.merge_queue_receipt_cid,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "repository_head": self.repository_head,
            "repository_tree": self.repository_tree,
            "runtime_artifacts": [dict(item) for item in self.runtime_artifacts],
            "launch_artifact_paths": list(self.launch_artifact_paths),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBoundRecoveryLaunch":
        fields = {
            "schema",
            "decision",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "execution_lease_cid",
            "execution_phase",
            "proposal_handoff_cid",
            "merge_authorization_cid",
            "merge_enqueue_intent_cid",
            "merge_request_id",
            "merge_queue_receipt_cid",
            "source_head",
            "source_tree",
            "repository_head",
            "repository_tree",
            "runtime_artifacts",
            "launch_artifact_paths",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ExecutionPlanError("recovery-launch fields are not exact")
        if any(
            not isinstance(payload[name], str)
            for name in fields - {"runtime_artifacts", "launch_artifact_paths"}
        ) or not isinstance(payload["runtime_artifacts"], list) or not isinstance(
            payload["launch_artifact_paths"], list
        ):
            raise ExecutionPlanError("recovery-launch field types are invalid")
        values = dict(payload)
        values["runtime_artifacts"] = tuple(values["runtime_artifacts"])
        values["launch_artifact_paths"] = tuple(values["launch_artifact_paths"])
        result = cls(**values)
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "recovery-launch failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundProposalDisposition:
    """One current slice owner's immutable final pre-merge proposal result."""

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    reassignment_cid: str
    task_id: str
    task_cid: str
    execution_lease_cid: str
    process_birth_cid: str
    proposal_id: str
    proposal_receipt_id: str
    outcome: str
    reason_codes: tuple[str, ...]
    actual_changed_paths: tuple[str, ...]
    baseline_ref: str = ""
    implementation_commit: str = ""
    schema: str = PLAN_BOUND_PROPOSAL_DISPOSITION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "task_id",
            "task_cid",
            "execution_lease_cid",
            "process_birth_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "reassignment_cid",
            "proposal_id",
            "proposal_receipt_id",
            "baseline_ref",
            "implementation_commit",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or value != value.strip()
                or "\x00" in value
            ):
                raise ExecutionPlanError(f"{name} must be canonical text")
        if self.schema != PLAN_BOUND_PROPOSAL_DISPOSITION_SCHEMA:
            raise ExecutionPlanError("unsupported proposal disposition schema")
        if not isinstance(self.outcome, str) or self.outcome not in {
            "changed",
            "no_change",
            "rejected",
        }:
            raise ExecutionPlanError("proposal disposition outcome is invalid")
        reasons = _string_set(self.reason_codes)
        paths = _paths(self.actual_changed_paths, "actual_changed_paths")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "actual_changed_paths", paths)
        if re.fullmatch(r"[0-9a-f]{40}", self.baseline_ref) is None:
            raise ExecutionPlanError(
                "proposal disposition baseline is not a resolved commit"
            )
        if self.outcome == "changed":
            if (
                reasons
                or not paths
                or not self.proposal_id
                or not self.proposal_receipt_id
                or re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit)
                is None
            ):
                raise ExecutionPlanError("changed proposal disposition is partial")
        elif self.outcome == "no_change":
            if (
                reasons
                or paths
                or self.proposal_id
                or self.proposal_receipt_id
                or self.implementation_commit != self.baseline_ref
            ):
                raise ExecutionPlanError("no-change proposal disposition is partial")
        elif (
            not reasons
            or not self.proposal_id
            or not self.proposal_receipt_id
            or (
                self.implementation_commit
                and re.fullmatch(
                    r"[0-9a-f]{40}",
                    self.implementation_commit,
                )
                is None
            )
        ):
            raise ExecutionPlanError("rejected proposal disposition is partial")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "reassignment_cid": self.reassignment_cid,
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "execution_lease_cid": self.execution_lease_cid,
            "process_birth_cid": self.process_birth_cid,
            "proposal_id": self.proposal_id,
            "proposal_receipt_id": self.proposal_receipt_id,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "actual_changed_paths": list(self.actual_changed_paths),
            "baseline_ref": self.baseline_ref,
            "implementation_commit": self.implementation_commit,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PlanBoundProposalDisposition":
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "task_id",
            "task_cid",
            "execution_lease_cid",
            "process_birth_cid",
            "proposal_id",
            "proposal_receipt_id",
            "outcome",
            "reason_codes",
            "actual_changed_paths",
            "baseline_ref",
            "implementation_commit",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ExecutionPlanError("proposal disposition fields are not exact")
        list_fields = {"reason_codes", "actual_changed_paths"}
        if any(not isinstance(payload[name], list) for name in list_fields):
            raise ExecutionPlanError("proposal disposition list fields are invalid")
        if any(
            any(not isinstance(item, str) for item in payload[name])
            for name in list_fields
        ):
            raise ExecutionPlanError("proposal disposition paths/reasons are invalid")
        if any(
            not isinstance(payload[name], str)
            for name in expected - list_fields
        ):
            raise ExecutionPlanError("proposal disposition text fields are invalid")
        result = cls(
            **{
                **dict(payload),
                "reason_codes": tuple(payload["reason_codes"]),
                "actual_changed_paths": tuple(payload["actual_changed_paths"]),
            }
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "proposal disposition failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundTerminalMissing:
    """Durable proof that one launched current owner cannot publish a result."""

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    reassignment_cid: str
    task_id: str
    task_cid: str
    process_birth_cid: str
    process_fence_cid: str
    exit_code: int
    observed_at_ms: int
    reason_codes: tuple[str, ...]
    schema: str = PLAN_BOUND_TERMINAL_MISSING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "task_id",
            "task_cid",
            "process_birth_cid",
            "process_fence_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.schema != PLAN_BOUND_TERMINAL_MISSING_SCHEMA:
            raise ExecutionPlanError("unsupported terminal-missing schema")
        if not isinstance(self.reassignment_cid, str):
            raise ExecutionPlanError("terminal-missing reassignment CID is invalid")
        if (
            isinstance(self.exit_code, bool)
            or not isinstance(self.exit_code, int)
            or not -255 <= self.exit_code <= 255
            or isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 1
        ):
            raise ExecutionPlanError("terminal-missing scalars are invalid")
        reasons = _string_set(self.reason_codes)
        if not reasons or not set(reasons).issubset(
            {
                "process_exited_without_disposition",
                "safe_reassignment_exhausted",
            }
        ):
            raise ExecutionPlanError("terminal-missing reasons are invalid")
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "reassignment_cid": self.reassignment_cid,
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "process_birth_cid": self.process_birth_cid,
            "process_fence_cid": self.process_fence_cid,
            "exit_code": self.exit_code,
            "observed_at_ms": self.observed_at_ms,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBoundTerminalMissing":
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "task_id",
            "task_cid",
            "process_birth_cid",
            "process_fence_cid",
            "exit_code",
            "observed_at_ms",
            "reason_codes",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ExecutionPlanError("terminal-missing fields are not exact")
        int_fields = {"exit_code", "observed_at_ms"}
        list_fields = {"reason_codes"}
        if any(
            not isinstance(payload[name], str)
            for name in expected - int_fields - list_fields
        ) or any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in int_fields
        ):
            raise ExecutionPlanError("terminal-missing scalar fields are invalid")
        if not isinstance(payload["reason_codes"], list) or any(
            not isinstance(item, str) for item in payload["reason_codes"]
        ):
            raise ExecutionPlanError("terminal-missing reasons are invalid")
        result = cls(
            **{
                **dict(payload),
                "reason_codes": tuple(payload["reason_codes"]),
            }
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "terminal-missing failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundProcessBirthExhausted:
    """Durable proof that a result-bearing slice exhausted gated recovery births."""

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    lane_id: str
    reassignment_cid: str
    task_id: str
    task_cid: str
    execution_lease_cid: str
    disposition_cid: str
    process_birth_cid: str
    process_fence_cid: str
    generation: int
    global_budget: int
    exit_code: int
    observed_at_ms: int
    reason_codes: tuple[str, ...]
    schema: str = PLAN_BOUND_PROCESS_BIRTH_EXHAUSTED_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "task_id",
            "task_cid",
            "execution_lease_cid",
            "disposition_cid",
            "process_birth_cid",
            "process_fence_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.schema != PLAN_BOUND_PROCESS_BIRTH_EXHAUSTED_SCHEMA:
            raise ExecutionPlanError(
                "unsupported process-birth-exhausted schema"
            )
        if not isinstance(self.reassignment_cid, str):
            raise ExecutionPlanError(
                "process-birth-exhausted reassignment CID is invalid"
            )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation != MAX_PLAN_BOUND_WAVE_TRANSFERS
            or isinstance(self.global_budget, bool)
            or not isinstance(self.global_budget, int)
            or self.global_budget != MAX_PLAN_BOUND_WAVE_TRANSFERS
            or isinstance(self.exit_code, bool)
            or not isinstance(self.exit_code, int)
            or not -255 <= self.exit_code <= 255
            or isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 1
        ):
            raise ExecutionPlanError(
                "process-birth-exhausted scalars are invalid"
            )
        reasons = _string_set(self.reason_codes)
        if reasons != ("process_birth_budget_exhausted",):
            raise ExecutionPlanError(
                "process-birth-exhausted reasons are invalid"
            )
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "lane_id": self.lane_id,
            "reassignment_cid": self.reassignment_cid,
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "execution_lease_cid": self.execution_lease_cid,
            "disposition_cid": self.disposition_cid,
            "process_birth_cid": self.process_birth_cid,
            "process_fence_cid": self.process_fence_cid,
            "generation": self.generation,
            "global_budget": self.global_budget,
            "exit_code": self.exit_code,
            "observed_at_ms": self.observed_at_ms,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PlanBoundProcessBirthExhausted":
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "task_id",
            "task_cid",
            "execution_lease_cid",
            "disposition_cid",
            "process_birth_cid",
            "process_fence_cid",
            "generation",
            "global_budget",
            "exit_code",
            "observed_at_ms",
            "reason_codes",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ExecutionPlanError(
                "process-birth-exhausted fields are not exact"
            )
        int_fields = {
            "generation",
            "global_budget",
            "exit_code",
            "observed_at_ms",
        }
        list_fields = {"reason_codes"}
        if any(
            not isinstance(payload[name], str)
            for name in expected - int_fields - list_fields
        ) or any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in int_fields
        ):
            raise ExecutionPlanError(
                "process-birth-exhausted scalar fields are invalid"
            )
        if not isinstance(payload["reason_codes"], list) or any(
            not isinstance(item, str) for item in payload["reason_codes"]
        ):
            raise ExecutionPlanError(
                "process-birth-exhausted reasons are invalid"
            )
        result = cls(
            **{
                **dict(payload),
                "reason_codes": tuple(payload["reason_codes"]),
            }
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "process-birth-exhausted failed exact semantic round trip"
            )
        return result


@dataclass(frozen=True)
class PlanBoundWaveDiffBarrier:
    """Immutable release-or-fence decision for every launched wave slice."""

    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    window_cid: str
    wave_index: int
    expected_members: tuple[Mapping[str, str], ...]
    dispositions: tuple[Mapping[str, str], ...]
    terminal_missing: tuple[Mapping[str, str], ...]
    decision: str
    reason_codes: tuple[str, ...] = ()
    missing_slice_ids: tuple[str, ...] = ()
    deadline_at_ms: int = 0
    decided_at_ms: int = 0
    overlap_witness: Mapping[str, str] | None = None
    schema: str = PLAN_BOUND_WAVE_DIFF_BARRIER_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "window_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.schema != PLAN_BOUND_WAVE_DIFF_BARRIER_SCHEMA:
            raise ExecutionPlanError("unsupported wave diff barrier schema")
        if (
            isinstance(self.wave_index, bool)
            or not isinstance(self.wave_index, int)
            or self.wave_index < 0
        ):
            raise ExecutionPlanError("wave diff barrier index is invalid")
        if self.decision not in {"released", "rejected", "overlap", "missing"}:
            raise ExecutionPlanError("wave diff barrier decision is invalid")
        for name in ("deadline_at_ms", "decided_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ExecutionPlanError(
                    f"wave diff barrier {name} is invalid"
                )
        if (
            self.decision == "missing"
            and not self.terminal_missing
            and self.decided_at_ms < self.deadline_at_ms
        ):
            raise ExecutionPlanError(
                "missing wave decision predates its bounded deadline"
            )

        members: list[dict[str, str]] = []
        for raw in self.expected_members:
            if not isinstance(raw, Mapping) or set(raw) != {
                "slice_id",
                "task_id",
                "task_cid",
            }:
                raise ExecutionPlanError("wave diff barrier member is malformed")
            members.append(
                {
                    name: _text(raw[name], name)
                    for name in ("slice_id", "task_id", "task_cid")
                }
            )
        members = sorted(members, key=lambda item: item["slice_id"])
        if (
            not members
            or len({item["slice_id"] for item in members}) != len(members)
            or len({item["task_id"] for item in members}) != len(members)
            or len({item["task_cid"] for item in members}) != len(members)
        ):
            raise ExecutionPlanError("wave diff barrier membership is ambiguous")
        object.__setattr__(self, "expected_members", tuple(members))

        dispositions: list[dict[str, str]] = []
        for raw in self.dispositions:
            if not isinstance(raw, Mapping) or set(raw) != {
                "slice_id",
                "disposition_cid",
            }:
                raise ExecutionPlanError("wave diff barrier disposition is malformed")
            dispositions.append(
                {
                    "slice_id": _text(raw["slice_id"], "slice_id"),
                    "disposition_cid": _text(
                        raw["disposition_cid"], "disposition_cid"
                    ),
                }
            )
        dispositions = sorted(dispositions, key=lambda item: item["slice_id"])
        if (
            len({item["slice_id"] for item in dispositions})
            != len(dispositions)
            or not {item["slice_id"] for item in dispositions}.issubset(
                {item["slice_id"] for item in members}
            )
        ):
            raise ExecutionPlanError("wave diff barrier dispositions are ambiguous")
        object.__setattr__(self, "dispositions", tuple(dispositions))
        terminal_missing: list[dict[str, str]] = []
        for raw in self.terminal_missing:
            if not isinstance(raw, Mapping) or set(raw) != {
                "slice_id",
                "terminal_missing_cid",
            }:
                raise ExecutionPlanError(
                    "wave diff barrier terminal-missing evidence is malformed"
                )
            terminal_missing.append(
                {
                    "slice_id": _text(raw["slice_id"], "slice_id"),
                    "terminal_missing_cid": _text(
                        raw["terminal_missing_cid"],
                        "terminal_missing_cid",
                    ),
                }
            )
        terminal_missing = sorted(
            terminal_missing,
            key=lambda item: item["slice_id"],
        )
        if (
            len({item["slice_id"] for item in terminal_missing})
            != len(terminal_missing)
            or not {item["slice_id"] for item in terminal_missing}.issubset(
                {item["slice_id"] for item in members}
            )
            or {item["slice_id"] for item in terminal_missing}
            & {item["slice_id"] for item in dispositions}
        ):
            raise ExecutionPlanError(
                "wave diff barrier terminal-missing evidence is ambiguous"
            )
        object.__setattr__(self, "terminal_missing", tuple(terminal_missing))
        object.__setattr__(self, "reason_codes", _string_set(self.reason_codes))
        object.__setattr__(
            self,
            "missing_slice_ids",
            _string_set(self.missing_slice_ids),
        )
        witness = dict(self.overlap_witness or {})
        if witness and set(witness) != {
            "left_slice_id",
            "right_slice_id",
            "left_path",
            "right_path",
        }:
            raise ExecutionPlanError("wave diff barrier overlap witness is malformed")
        if witness:
            witness = {name: _text(value, name) for name, value in witness.items()}
            if not _overlaps(witness["left_path"], witness["right_path"]):
                raise ExecutionPlanError("wave diff barrier overlap witness is false")
        object.__setattr__(self, "overlap_witness", witness)
        expected_slice_ids = {item["slice_id"] for item in members}
        if not set(self.missing_slice_ids).issubset(expected_slice_ids):
            raise ExecutionPlanError("wave diff barrier missing evidence is mixed")
        if self.decision == "released" and (
            len(dispositions) != len(members)
            or terminal_missing
            or self.reason_codes
            or self.missing_slice_ids
            or witness
        ):
            raise ExecutionPlanError("released wave diff barrier is partial")
        if self.decision != "released" and not self.reason_codes:
            raise ExecutionPlanError("denied wave diff barrier lacks reasons")
        if self.decision == "overlap" and not witness:
            raise ExecutionPlanError("overlap barrier lacks a witness")
        if self.decision == "missing" and not self.missing_slice_ids:
            raise ExecutionPlanError("missing barrier lacks missing members")
        if terminal_missing and self.decision != "missing":
            raise ExecutionPlanError(
                "terminal-missing evidence requires a missing decision"
            )
        if not {item["slice_id"] for item in terminal_missing}.issubset(
            set(self.missing_slice_ids)
        ):
            raise ExecutionPlanError(
                "terminal-missing evidence differs from missing membership"
            )
        if self.decision in {"released", "rejected", "overlap"} and len(
            dispositions
        ) != len(members):
            raise ExecutionPlanError(
                "non-missing wave decision requires every slice disposition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "window_cid": self.window_cid,
            "wave_index": self.wave_index,
            "expected_members": [dict(item) for item in self.expected_members],
            "dispositions": [dict(item) for item in self.dispositions],
            "terminal_missing": [dict(item) for item in self.terminal_missing],
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "missing_slice_ids": list(self.missing_slice_ids),
            "deadline_at_ms": self.deadline_at_ms,
            "decided_at_ms": self.decided_at_ms,
            "overlap_witness": dict(self.overlap_witness or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBoundWaveDiffBarrier":
        expected = {
            "schema",
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "window_cid",
            "wave_index",
            "expected_members",
            "dispositions",
            "terminal_missing",
            "decision",
            "reason_codes",
            "missing_slice_ids",
            "deadline_at_ms",
            "decided_at_ms",
            "overlap_witness",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ExecutionPlanError("wave diff barrier fields are not exact")
        if any(
            not isinstance(payload[name], list)
            for name in (
                "expected_members",
                "dispositions",
                "terminal_missing",
                "reason_codes",
                "missing_slice_ids",
            )
        ) or not isinstance(payload["overlap_witness"], Mapping):
            raise ExecutionPlanError("wave diff barrier collections are invalid")
        if any(
            not isinstance(payload[name], str)
            for name in (
                "schema",
                "revision_cid",
                "plan_root_cid",
                "execution_plan_cid",
                "capacity_snapshot_id",
                "slice_manifest_cid",
                "window_cid",
                "decision",
            )
        ) or any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in ("wave_index", "deadline_at_ms", "decided_at_ms")
        ):
            raise ExecutionPlanError("wave diff barrier scalar fields are invalid")
        result = cls(
            **{
                **dict(payload),
                "expected_members": tuple(payload["expected_members"]),
                "dispositions": tuple(payload["dispositions"]),
                "terminal_missing": tuple(payload["terminal_missing"]),
                "reason_codes": tuple(payload["reason_codes"]),
                "missing_slice_ids": tuple(payload["missing_slice_ids"]),
                "overlap_witness": dict(payload["overlap_witness"]),
            }
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "wave diff barrier failed exact semantic round trip"
            )
        return result


def _validate_plan_bound_proposal_handoff_locked(
    store: PlanRevisionStore,
    lease: PlanBoundExecutionLease,
) -> None:
    """Validate the restart-stable pre-barrier candidate handoff."""

    phases = {
        "proposal_ready",
        "merge_enqueue_reached",
        "merge_enqueue_prepared",
        "merge_enqueue_confirmed",
        "merge_completed",
    }
    if lease.phase not in phases:
        if lease.proposal_handoff_cid:
            raise ExecutionPlanError(
                "non-admissible execution phase carries a proposal handoff"
            )
        return
    proposal_lease = lease
    if lease.phase != "proposal_ready":
        authorization = _secure_store_cas(
            store,
            lease.merge_authorization_cid,
        )
        proposal_lease_cid = authorization.get("execution_lease_cid")
        if (
            authorization.get("schema")
            != PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA
            or not isinstance(proposal_lease_cid, str)
            or not proposal_lease_cid
            or authorization.get("proposal_handoff_cid")
            != lease.proposal_handoff_cid
        ):
            raise ExecutionPlanError(
                "proposal handoff merge authorization is malformed or mixed"
            )
        proposal_lease = PlanBoundExecutionLease.from_dict(
            _secure_store_cas(store, proposal_lease_cid)
        )
        if (
            proposal_lease.phase != "proposal_ready"
            or proposal_lease.proposal_handoff_cid
            != lease.proposal_handoff_cid
            or proposal_lease.revision_cid != lease.revision_cid
            or proposal_lease.slice_manifest_cid != lease.slice_manifest_cid
            or proposal_lease.slice_id != lease.slice_id
            or proposal_lease.lane_id != lease.lane_id
            or proposal_lease.active_task_id != lease.active_task_id
            or proposal_lease.active_task_cid != lease.active_task_cid
        ):
            raise ExecutionPlanError(
                "proposal handoff proposal lease is malformed or mixed"
            )
    handoff = _secure_store_cas(store, lease.proposal_handoff_cid)
    fields = {
        "schema",
        "revision_cid",
        "plan_root_cid",
        "execution_plan_cid",
        "capacity_snapshot_id",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "reassignment_cid",
        "task_id",
        "task_cid",
        "source_execution_lease_cid",
        "process_birth_cid",
        "canonical_claim_cid",
        "canonical_claim_lease_id",
        "workspace_lifecycle_cid",
        "workspace_record_id",
        "workspace_path",
        "workspace_lease_id",
        "workspace_fence",
        "attempt",
        "branch_name",
        "baseline_ref",
        "implementation_commit",
        "actual_changed_paths",
        "outcome",
        "enqueue_fields",
        "enqueue_fields_cid",
        "created_at_ms",
    }
    int_fields = {"workspace_fence", "attempt", "created_at_ms"}
    list_fields = {"actual_changed_paths"}
    mapping_fields = {"enqueue_fields"}
    text_fields = fields - int_fields - list_fields - mapping_fields
    if (
        set(handoff) != fields
        or handoff.get("schema") != PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA
        or any(not isinstance(handoff.get(name), str) for name in text_fields)
        or any(
            isinstance(handoff.get(name), bool)
            or not isinstance(handoff.get(name), int)
            or int(handoff[name]) < 1
            for name in int_fields
        )
        or not isinstance(handoff.get("actual_changed_paths"), list)
        or any(
            not isinstance(path, str)
            for path in handoff.get("actual_changed_paths", ())
        )
        or not isinstance(handoff.get("enqueue_fields"), Mapping)
        or content_identity(dict(handoff["enqueue_fields"]))
        != handoff.get("enqueue_fields_cid")
        or re.fullmatch(r"[0-9a-f]{40}", handoff.get("baseline_ref", ""))
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}", handoff.get("implementation_commit", "")
        )
        is None
        or handoff.get("outcome") not in {"changed", "no_change"}
    ):
        raise ExecutionPlanError("proposal handoff CAS is malformed")
    expected = {
        "revision_cid": proposal_lease.revision_cid,
        "plan_root_cid": proposal_lease.plan_root_cid,
        "execution_plan_cid": proposal_lease.execution_plan_cid,
        "capacity_snapshot_id": proposal_lease.capacity_snapshot_id,
        "slice_manifest_cid": proposal_lease.slice_manifest_cid,
        "slice_id": proposal_lease.slice_id,
        "lane_id": proposal_lease.lane_id,
        "reassignment_cid": proposal_lease.reassignment_cid,
        "task_id": proposal_lease.active_task_id,
        "task_cid": proposal_lease.active_task_cid,
        "source_execution_lease_cid": proposal_lease.prior_execution_lease_cid,
        "process_birth_cid": proposal_lease.process_birth_cid,
        "canonical_claim_cid": proposal_lease.canonical_claim_cid,
        "canonical_claim_lease_id": proposal_lease.canonical_claim_lease_id,
        "workspace_lifecycle_cid": proposal_lease.workspace_lifecycle_cid,
        "workspace_record_id": proposal_lease.workspace_record_id,
        "workspace_path": proposal_lease.workspace_path,
        "workspace_lease_id": proposal_lease.workspace_lease_id,
        "workspace_fence": proposal_lease.workspace_fence,
        "actual_changed_paths": list(proposal_lease.actual_changed_paths),
    }
    if any(handoff.get(name) != value for name, value in expected.items()):
        raise ExecutionPlanError("proposal handoff authority is mixed")
    changed = bool(
        proposal_lease.proposal_id
        and proposal_lease.proposal_receipt_id
        and proposal_lease.actual_changed_paths
    )
    if (
        (handoff["outcome"] == "changed") != changed
        or (
            handoff["outcome"] == "no_change"
            and (
                proposal_lease.actual_changed_paths
                or handoff["baseline_ref"] != handoff["implementation_commit"]
            )
        )
    ):
        raise ExecutionPlanError("proposal handoff outcome is mixed")
    enqueue = dict(handoff["enqueue_fields"])
    enqueue_fields = {
        "branch_name",
        "task_id",
        "priority",
        "lane_id",
        "attempt",
        "metadata",
        "commit_sha",
        "canonical_task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "target_repository_id",
        "target_branch",
    }
    if (
        set(enqueue) != enqueue_fields
        or not isinstance(enqueue.get("metadata"), Mapping)
        or isinstance(enqueue.get("attempt"), bool)
        or not isinstance(enqueue.get("attempt"), int)
        or any(
            not isinstance(enqueue.get(name), str)
            for name in enqueue_fields - {"attempt", "metadata"}
        )
        or enqueue.get("task_id") != proposal_lease.active_task_id
        or enqueue.get("canonical_task_id") != proposal_lease.active_task_cid
        or enqueue.get("attempt") != handoff["attempt"]
        or enqueue.get("branch_name") != handoff["branch_name"]
        or enqueue.get("commit_sha") != handoff["implementation_commit"]
        or enqueue["metadata"].get("baseline_ref") != handoff["baseline_ref"]
        or enqueue["metadata"].get("implementation_commit")
        != handoff["implementation_commit"]
    ):
        raise ExecutionPlanError("proposal handoff enqueue fields are mixed")
    source = PlanBoundExecutionLease.from_dict(
        _secure_store_cas(store, proposal_lease.prior_execution_lease_cid)
    )
    if (
        source.phase != "provider_ready"
        or source.revision_cid != proposal_lease.revision_cid
        or source.slice_id != proposal_lease.slice_id
        or source.lane_id != proposal_lease.lane_id
        or source.generation + 1 != proposal_lease.generation
    ):
        raise ExecutionPlanError("proposal handoff predecessor is mixed")


def _validate_plan_bound_merge_handoff_locked(
    store: PlanRevisionStore,
    lease: PlanBoundExecutionLease,
) -> None:
    """Validate every durable merge-handoff reference carried by a lease."""

    merge_phases = {
        "merge_enqueue_reached",
        "merge_enqueue_prepared",
        "merge_enqueue_confirmed",
        "merge_completed",
    }
    if lease.phase not in merge_phases:
        return
    authorization = _secure_store_cas(store, lease.merge_authorization_cid)
    authorization_fields = {
        "schema",
        "revision_cid",
        "plan_root_cid",
        "execution_plan_cid",
        "capacity_snapshot_id",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "reassignment_cid",
        "task_id",
        "task_cid",
        "process_birth_cid",
        "execution_lease_cid",
        "proposal_handoff_cid",
        "recovery_birth_cid",
        "disposition_cid",
        "barrier_cid",
        "outcome",
        "canonical_claim_cid",
        "workspace_lifecycle_cid",
        "workspace_fence",
        "workspace_lease_id",
        "workspace_path",
        "attempt",
        "branch_name",
        "baseline_ref",
        "implementation_commit",
        "actual_changed_paths",
        "authorized_at_ms",
    }
    text_fields = authorization_fields - {
        "workspace_fence",
        "attempt",
        "actual_changed_paths",
        "authorized_at_ms",
    }
    if (
        set(authorization) != authorization_fields
        or authorization.get("schema") != PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA
        or any(not isinstance(authorization.get(name), str) for name in text_fields)
        or not isinstance(authorization.get("actual_changed_paths"), list)
        or any(
            not isinstance(item, str)
            for item in authorization.get("actual_changed_paths", ())
        )
        or any(
            isinstance(authorization.get(name), bool)
            or not isinstance(authorization.get(name), int)
            or int(authorization[name]) < 1
            for name in ("workspace_fence", "attempt", "authorized_at_ms")
        )
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(authorization.get("baseline_ref") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(authorization.get("implementation_commit") or ""),
        )
        is None
    ):
        raise ExecutionPlanError("merge authorization CAS is malformed")
    expected_authority = {
        "revision_cid": lease.revision_cid,
        "plan_root_cid": lease.plan_root_cid,
        "execution_plan_cid": lease.execution_plan_cid,
        "capacity_snapshot_id": lease.capacity_snapshot_id,
        "slice_manifest_cid": lease.slice_manifest_cid,
        "slice_id": lease.slice_id,
        "lane_id": lease.lane_id,
        "reassignment_cid": lease.reassignment_cid,
        "task_id": lease.active_task_id,
        "task_cid": lease.active_task_cid,
        "process_birth_cid": lease.process_birth_cid,
        "canonical_claim_cid": lease.canonical_claim_cid,
        "workspace_lifecycle_cid": lease.workspace_lifecycle_cid,
        "workspace_fence": lease.workspace_fence,
        "workspace_lease_id": lease.workspace_lease_id,
        "workspace_path": lease.workspace_path,
        "actual_changed_paths": list(lease.actual_changed_paths),
        "proposal_handoff_cid": lease.proposal_handoff_cid,
    }
    if any(authorization.get(name) != value for name, value in expected_authority.items()):
        raise ExecutionPlanError("merge authorization authority is mixed")
    recovery_birth_cid = authorization["recovery_birth_cid"]
    if recovery_birth_cid:
        from ..merge.worktree_lifecycle import (
            ProcessBirthIdentity as WorktreeProcessBirthIdentity,
        )
        from ..merge.worktree_lifecycle import WorkspaceLifecycleRecord

        recovery_fields = {
            "schema",
            "revision_cid",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "generation",
            "execution_lease_cid",
            "proposal_handoff_cid",
            "merge_authorization_cid",
            "merge_enqueue_intent_cid",
            "supervisor_process_birth_cid",
            "prior_supervisor_process_birth_cid",
            "canonical_claim_cid",
            "canonical_claim_lease_id",
            "custody_kind",
            "authorized_workspace_lifecycle_cid",
            "lifecycle_owner_process_birth",
            "prior_recovery_daemon_process_birth",
            "daemon_process_birth",
            "workspace_lifecycle_path",
            "workspace_lifecycle_cid",
            "workspace_lifecycle_json",
            "prior_recovery_birth_cid",
            "observed_at_ms",
        }
        birth_fields = {
            "pid",
            "start_time_ticks",
            "boot_id",
            "parent_pid",
        }

        def exact_worktree_birth(raw: Any) -> WorktreeProcessBirthIdentity:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != birth_fields
                or any(
                    isinstance(raw[name], bool) or not isinstance(raw[name], int)
                    for name in ("pid", "start_time_ticks", "parent_pid")
                )
                or not isinstance(raw["boot_id"], str)
            ):
                raise ExecutionPlanError("merge recovery process birth is malformed")
            decoded = WorktreeProcessBirthIdentity.from_dict(raw)
            if _canonical(decoded.to_dict()) != _canonical(dict(raw)):
                raise ExecutionPlanError("merge recovery process birth normalized")
            return decoded

        recovery = _secure_store_cas(store, recovery_birth_cid)
        generation = recovery.get("generation")
        lifecycle_json = recovery.get("workspace_lifecycle_json")
        try:
            if not isinstance(lifecycle_json, str) or not lifecycle_json:
                raise TypeError("lifecycle JSON must be nonempty text")
            lifecycle_raw = json.loads(
                lifecycle_json,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            if not isinstance(lifecycle_raw, Mapping):
                raise TypeError("lifecycle JSON must contain an object")
            lifecycle = WorkspaceLifecycleRecord.from_dict(lifecycle_raw)
        except Exception as exc:
            raise ExecutionPlanError(
                "merge authorization recovery lifecycle is malformed"
            ) from exc
        canonical_lifecycle_bytes = lifecycle_json.encode("utf-8")
        lifecycle_cid = "sha256:" + hashlib.sha256(
            canonical_lifecycle_bytes
        ).hexdigest()
        lifecycle_owner = exact_worktree_birth(
            recovery.get("lifecycle_owner_process_birth")
        )
        prior_recovery_daemon = exact_worktree_birth(
            recovery.get("prior_recovery_daemon_process_birth")
        )
        recovery_daemon = exact_worktree_birth(
            recovery.get("daemon_process_birth")
        )
        supervisor_birth = _secure_store_cas(
            store,
            str(recovery.get("supervisor_process_birth_cid") or ""),
        )
        if (
            set(recovery) != recovery_fields
            or recovery.get("schema") != PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or not 1 <= generation <= MAX_PLAN_BOUND_WAVE_TRANSFERS
            or isinstance(recovery.get("observed_at_ms"), bool)
            or not isinstance(recovery.get("observed_at_ms"), int)
            or int(recovery["observed_at_ms"]) < 1
            or recovery.get("revision_cid") != lease.revision_cid
            or recovery.get("slice_manifest_cid") != lease.slice_manifest_cid
            or recovery.get("slice_id") != lease.slice_id
            or recovery.get("lane_id") != lease.lane_id
            or recovery.get("execution_lease_cid")
            != authorization["execution_lease_cid"]
            or recovery.get("proposal_handoff_cid")
            != lease.proposal_handoff_cid
            or recovery.get("merge_authorization_cid") != ""
            or recovery.get("merge_enqueue_intent_cid") != ""
            or recovery.get("canonical_claim_cid")
            != lease.canonical_claim_cid
            or recovery.get("canonical_claim_lease_id")
            != lease.canonical_claim_lease_id
            or recovery.get("custody_kind") != "settling_candidate"
            or recovery.get("authorized_workspace_lifecycle_cid")
            != lease.workspace_lifecycle_cid
            or recovery.get("workspace_lifecycle_path")
            != lease.workspace_lifecycle_path
            or recovery.get("workspace_lifecycle_cid")
            != lease.workspace_lifecycle_cid
            or lifecycle_cid != lease.workspace_lifecycle_cid
            or lifecycle_json
            != json.dumps(dict(lifecycle_raw), indent=2, sort_keys=True) + "\n"
            or _canonical(lifecycle.to_dict())
            != _canonical(dict(lifecycle_raw))
            or lifecycle.state.value != "settling"
            or lifecycle.task_id != lease.active_task_id
            or lifecycle.canonical_task_cid != lease.active_task_cid
            or lifecycle.owner.to_dict() != lease.daemon_process_birth
            or lifecycle_owner.to_dict() != lease.daemon_process_birth
            or lifecycle.workspace_path != lease.workspace_path
            or lifecycle.record_id != lease.workspace_record_id
            or lifecycle.lease_id != lease.workspace_lease_id
            or lifecycle.fence != lease.workspace_fence
            or recovery_daemon.pid <= 0
            or recovery_daemon.start_time_ticks <= 0
            or prior_recovery_daemon.pid <= 0
            or not isinstance(supervisor_birth, Mapping)
            or supervisor_birth.get("schema")
            != PLAN_BOUND_PROCESS_BIRTH_SCHEMA
            or supervisor_birth.get("revision_cid") != lease.revision_cid
            or supervisor_birth.get("slice_manifest_cid")
            != lease.slice_manifest_cid
            or supervisor_birth.get("slice_id") != lease.slice_id
            or supervisor_birth.get("lane_id") != lease.lane_id
            or supervisor_birth.get("prior_process_birth_cid")
            != recovery.get("prior_supervisor_process_birth_cid")
        ):
            raise ExecutionPlanError("merge authorization recovery birth is mixed")
        if generation == 1:
            if (
                recovery.get("prior_recovery_birth_cid")
                or recovery.get("prior_supervisor_process_birth_cid")
                != lease.process_birth_cid
                or prior_recovery_daemon.to_dict()
                != lease.daemon_process_birth
            ):
                raise ExecutionPlanError(
                    "initial merge recovery predecessor is mixed"
                )
        else:
            prior_recovery_cid = str(
                recovery.get("prior_recovery_birth_cid") or ""
            )
            prior_recovery = _secure_store_cas(store, prior_recovery_cid)
            if (
                set(prior_recovery) != recovery_fields
                or prior_recovery.get("schema")
                != PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA
                or prior_recovery.get("generation") != generation - 1
                or prior_recovery.get("revision_cid") != lease.revision_cid
                or prior_recovery.get("slice_id") != lease.slice_id
                or prior_recovery.get("lane_id") != lease.lane_id
                or prior_recovery.get("supervisor_process_birth_cid")
                != recovery.get("prior_supervisor_process_birth_cid")
                or prior_recovery.get("daemon_process_birth")
                != recovery.get("prior_recovery_daemon_process_birth")
                or prior_recovery.get("custody_kind")
                != "settling_candidate"
                or prior_recovery.get("authorized_workspace_lifecycle_cid")
                != recovery.get("authorized_workspace_lifecycle_cid")
                or prior_recovery.get("workspace_lifecycle_cid")
                != recovery.get("workspace_lifecycle_cid")
                or prior_recovery.get("workspace_lifecycle_json")
                != recovery.get("workspace_lifecycle_json")
            ):
                raise ExecutionPlanError("merge recovery predecessor is mixed")
    proposal_lease = PlanBoundExecutionLease.from_dict(
        _secure_store_cas(store, str(authorization["execution_lease_cid"]))
    )
    disposition = PlanBoundProposalDisposition.from_dict(
        _secure_store_cas(store, str(authorization["disposition_cid"]))
    )
    barrier = PlanBoundWaveDiffBarrier.from_dict(
        _secure_store_cas(store, str(authorization["barrier_cid"]))
    )
    if (
        proposal_lease.phase != "proposal_ready"
        or disposition.execution_lease_cid != authorization["execution_lease_cid"]
        or disposition.slice_id != lease.slice_id
        or disposition.task_id != lease.active_task_id
        or disposition.task_cid != lease.active_task_cid
        or disposition.outcome != authorization["outcome"]
        or disposition.outcome not in {"changed", "no_change"}
        or disposition.actual_changed_paths != lease.actual_changed_paths
        or authorization["proposal_handoff_cid"]
        != lease.proposal_handoff_cid
        or barrier.revision_cid != lease.revision_cid
        or barrier.slice_manifest_cid != lease.slice_manifest_cid
        or barrier.decision != "released"
        or not any(
            row.get("slice_id") == lease.slice_id
            and row.get("disposition_cid") == authorization["disposition_cid"]
            for row in barrier.dispositions
        )
    ):
        raise ExecutionPlanError("merge authorization evidence is mixed")

    if lease.phase in {
        "merge_enqueue_prepared",
        "merge_enqueue_confirmed",
        "merge_completed",
    }:
        intent = _secure_store_cas(store, lease.merge_enqueue_intent_cid)
        if (
            set(intent)
            != {
                "schema",
                "authorization_cid",
                "enqueue_fields",
                "enqueue_fields_cid",
                "prepared_at_ms",
            }
            or intent.get("schema") != PLAN_BOUND_MERGE_ENQUEUE_INTENT_SCHEMA
            or intent.get("authorization_cid") != lease.merge_authorization_cid
            or not isinstance(intent.get("enqueue_fields"), Mapping)
            or content_identity(dict(intent["enqueue_fields"]))
            != intent.get("enqueue_fields_cid")
            or isinstance(intent.get("prepared_at_ms"), bool)
            or not isinstance(intent.get("prepared_at_ms"), int)
            or int(intent["prepared_at_ms"]) < int(authorization["authorized_at_ms"])
        ):
            raise ExecutionPlanError("merge enqueue intent CAS is malformed or mixed")
    if lease.phase in {"merge_enqueue_confirmed", "merge_completed"}:
        receipt = _secure_store_cas(store, lease.merge_queue_receipt_cid)
        if (
            set(receipt)
            != {
                "schema",
                "authorization_cid",
                "intent_cid",
                "enqueue_fields_cid",
                "request_id",
                "dedupe_key",
                "observed_status",
                "confirmed_at_ms",
            }
            or receipt.get("schema") != PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA
            or receipt.get("authorization_cid") != lease.merge_authorization_cid
            or receipt.get("intent_cid") != lease.merge_enqueue_intent_cid
            or receipt.get("enqueue_fields_cid") != intent["enqueue_fields_cid"]
            or receipt.get("request_id") != lease.merge_request_id
            or not isinstance(receipt.get("dedupe_key"), str)
            or not receipt["dedupe_key"]
            or not isinstance(receipt.get("observed_status"), str)
            or not receipt["observed_status"]
            or isinstance(receipt.get("confirmed_at_ms"), bool)
            or not isinstance(receipt.get("confirmed_at_ms"), int)
            or int(receipt["confirmed_at_ms"]) < int(intent["prepared_at_ms"])
            or (
                lease.phase == "merge_completed"
                and receipt.get("observed_status") != "completed"
            )
        ):
            raise ExecutionPlanError("merge queue receipt CAS is malformed or mixed")


def plan_bound_process_birth_key(
    revision_cid: str,
    slice_id: str,
    lane_id: str,
) -> str:
    """Return the one-winner head key for a bounded birth history."""

    return (
        "plan-bound-process-birth:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}:"
        f"{_text(lane_id, 'lane_id')}"
    )


def _load_plan_bound_process_birth_chain_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
    lane_id: str,
) -> tuple[str, PlanBoundProcessBirth, tuple[tuple[str, PlanBoundProcessBirth], ...]] | None:
    """Load and fully revalidate one bounded newest-to-oldest birth chain."""

    key = plan_bound_process_birth_key(revision_cid, slice_id, lane_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    pointer_fields = {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "lane_id",
        "process_birth_cid",
        "generation",
        "global_budget",
    }
    if (
        set(pointer) != pointer_fields
        or pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_process_birth"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_id") != slice_id
        or pointer.get("lane_id") != lane_id
        or not isinstance(pointer.get("process_birth_cid"), str)
        or not pointer["process_birth_cid"]
        or isinstance(pointer.get("generation"), bool)
        or not isinstance(pointer.get("generation"), int)
        or isinstance(pointer.get("global_budget"), bool)
        or pointer.get("global_budget") != MAX_PLAN_BOUND_WAVE_TRANSFERS
        or not 0 <= int(pointer["generation"]) <= MAX_PLAN_BOUND_WAVE_TRANSFERS
    ):
        raise ExecutionPlanError("plan-bound process-birth pointer is malformed")

    head_cid = str(pointer["process_birth_cid"])
    current_cid = head_cid
    expected_generation = int(pointer["generation"])
    seen: set[str] = set()
    chain: list[tuple[str, PlanBoundProcessBirth]] = []
    invariant: tuple[Any, ...] | None = None
    from ..control.lifecycle_orchestrator import (
        LifecycleProfile,
        ProcessIdentity,
    )

    while True:
        if not current_cid or current_cid in seen:
            raise ExecutionPlanError("plan-bound process-birth chain cycles or is missing")
        seen.add(current_cid)
        current = PlanBoundProcessBirth.from_dict(
            _secure_store_cas(store, current_cid)
        )
        try:
            profile = LifecycleProfile.from_dict(current.profile)
            identity = ProcessIdentity.from_dict(current.process_birth)
        except Exception as exc:
            raise ExecutionPlanError(
                "plan-bound process-birth lifecycle identity is malformed"
            ) from exc
        if (
            profile.to_dict() != current.profile
            or identity.to_dict() != current.process_birth
            or identity.profile_id != profile.profile_id
            or identity.run_id != profile.run_id
            or identity.target_id != profile.target_id
            or profile.repository_root != current.accepted_tree_root
        ):
            raise ExecutionPlanError(
                "plan-bound process-birth lifecycle identity drifted"
            )
        current_invariant = (
            current.revision_cid,
            current.plan_root_cid,
            current.execution_plan_cid,
            current.capacity_snapshot_id,
            current.slice_manifest_cid,
            current.slice_id,
            current.lane_id,
            current.task_ids,
            current.task_cids,
            current.configuration_root,
            current.accepted_tree_root,
            current.global_budget,
        )
        if invariant is None:
            invariant = current_invariant
        if (
            current_invariant != invariant
            or current.revision_cid != revision_cid
            or current.slice_id != slice_id
            or current.lane_id != lane_id
            or current.generation != expected_generation
        ):
            raise ExecutionPlanError(
                "plan-bound process-birth chain has identity or generation drift"
            )
        chain.append((current_cid, current))
        if expected_generation == 0:
            if current.prior_process_birth_cid:
                raise ExecutionPlanError(
                    "plan-bound process-birth root has prior authority"
                )
            break
        if not current.prior_process_birth_cid:
            raise ExecutionPlanError("plan-bound process-birth chain is truncated")
        current_cid = current.prior_process_birth_cid
        expected_generation -= 1
    if (
        len(chain) != int(pointer["generation"]) + 1
        or chain[0][0] != head_cid
        or chain[0][1].generation != pointer["generation"]
    ):
        raise ExecutionPlanError("plan-bound process-birth head was rolled back")

    # The continuation is a projection, while each birth record is immutable
    # CAS.  A copied older continuation must not hide a later valid birth, and
    # a crash/concurrent writer must not leave two candidate children at one
    # generation.  Bound the authority scan so a hostile store cannot turn
    # restart validation into an unbounded directory walk.
    matching_cas: dict[int, str] = {}
    try:
        entries = tuple(os.scandir(store.cas_dir))
    except OSError as exc:
        raise ExecutionPlanError(
            "cannot inspect process-birth CAS population"
        ) from exc
    if len(entries) > 100_000:
        raise ExecutionPlanError("process-birth CAS population exceeds its bound")
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            raise ExecutionPlanError("plan store CAS population is not regular")
        envelope = _stable_authority_json(Path(entry.path))
        payload = envelope.get("payload")
        if (
            set(envelope) != {"schema", "cid", "media_type", "payload"}
            or envelope.get("schema") != PLAN_REVISION_STORE_SCHEMA
            or envelope.get("cid") != entry.name
            or not isinstance(payload, Mapping)
        ):
            raise ExecutionPlanError("plan store CAS envelope is malformed")
        if payload.get("schema") != PLAN_BOUND_PROCESS_BIRTH_SCHEMA:
            continue
        candidate = PlanBoundProcessBirth.from_dict(payload)
        if (
            candidate.revision_cid != revision_cid
            or candidate.slice_id != slice_id
            or candidate.lane_id != lane_id
        ):
            continue
        if content_identity(candidate.to_dict()) != entry.name:
            raise ExecutionPlanError("process-birth CAS identity is invalid")
        existing = matching_cas.get(candidate.generation)
        if existing is not None and existing != entry.name:
            raise ExecutionPlanError(
                "process-birth CAS has concurrent generation forks"
            )
        matching_cas[candidate.generation] = entry.name
    chain_by_generation = {birth.generation: cid for cid, birth in chain}
    if matching_cas != chain_by_generation:
        raise ExecutionPlanError(
            "plan-bound process-birth pointer is rolled back or forked"
        )
    return head_cid, chain[0][1], tuple(chain)


def plan_bound_execution_lease_key(
    revision_cid: str,
    slice_id: str,
    lane_id: str,
) -> str:
    return (
        "plan-bound-execution-lease:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}:"
        f"{_text(lane_id, 'lane_id')}"
    )


def _load_plan_bound_execution_lease_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
    lane_id: str,
) -> tuple[str, PlanBoundExecutionLease] | None:
    """Load and cross-check one execution lease under the store guard."""

    key = plan_bound_execution_lease_key(revision_cid, slice_id, lane_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    expected_pointer_fields = {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "lane_id",
        "execution_lease_cid",
        "generation",
        "execution_phase",
    }
    if (
        set(pointer) != expected_pointer_fields
        or pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_execution_lease"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_id") != slice_id
        or pointer.get("lane_id") != lane_id
    ):
        raise ExecutionPlanError("plan-bound execution lease pointer is malformed")
    generation = pointer.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ExecutionPlanError("plan-bound execution lease pointer generation is invalid")
    lease_cid = _text(pointer.get("execution_lease_cid"), "execution_lease_cid")
    record = PlanBoundExecutionLease.from_dict(_secure_store_cas(store, lease_cid))
    if (
        record.revision_cid != revision_cid
        or record.slice_id != slice_id
        or record.lane_id != lane_id
        or record.generation != generation
        or record.phase != pointer.get("execution_phase")
    ):
        raise ExecutionPlanError("plan-bound execution lease pointer is mixed")

    active = _secure_store_active(store)
    if active is None or active.revision_cid != revision_cid:
        raise ExecutionPlanError("plan-bound execution lease lost active revision")
    revision_payload = _secure_store_cas(store, revision_cid)
    revision = PlanRevision.from_dict(revision_payload)
    if revision.to_dict() != revision_payload:
        raise ExecutionPlanError("execution lease revision changed during decode")
    if revision.materialization_transaction_cid != record.slice_manifest_cid:
        raise ExecutionPlanError("execution lease carries a foreign slice manifest")
    adapter = ProductionParallelPlanAdapter(store)
    try:
        execution_slice = adapter._validate_slice_owner_locked(  # noqa: SLF001
            revision_cid=revision_cid,
            slice_manifest_cid=record.slice_manifest_cid,
            slice_id=slice_id,
            lane_id=lane_id,
            reassignment_cid=record.reassignment_cid,
        )
    except ExecutionPlanError as exc:
        raise ExecutionPlanError(
            "execution lease lost canonical slice ownership"
        ) from exc
    if (
        active.plan_root_cid != record.plan_root_cid
        or revision.execution_plan_cid != record.execution_plan_cid
        or execution_slice.task_ids != record.task_ids
        or execution_slice.task_cids != record.task_cids
        or execution_slice.capacity_snapshot_id != record.capacity_snapshot_id
    ):
        raise ExecutionPlanError("plan-bound execution lease authority is mixed")

    birth_binding = _load_plan_bound_process_birth_chain_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
        lane_id=lane_id,
    )
    if birth_binding is None:
        raise ExecutionPlanError(
            "execution lease process birth lost its bounded chain"
        )
    matching_births = tuple(
        birth
        for birth_cid, birth in birth_binding[2]
        if birth_cid == record.process_birth_cid
    )
    if len(matching_births) != 1:
        raise ExecutionPlanError(
            "execution lease process birth is absent or duplicated in its chain"
        )
    typed_birth = matching_births[0]
    if (
        typed_birth.revision_cid != revision_cid
        or typed_birth.slice_manifest_cid != record.slice_manifest_cid
        or typed_birth.slice_id != slice_id
        or typed_birth.lane_id != lane_id
        or typed_birth.task_ids != record.task_ids
        or typed_birth.task_cids != record.task_cids
        or typed_birth.process_birth != record.process_birth
    ):
        raise ExecutionPlanError("execution lease process birth is mixed")
    _validate_plan_bound_proposal_handoff_locked(store, record)
    _validate_plan_bound_merge_handoff_locked(store, record)
    return lease_cid, record


def _publish_plan_bound_execution_lease_locked(
    store: PlanRevisionStore,
    record: PlanBoundExecutionLease,
    *,
    expected_current_cid: str = "",
) -> str:
    """CAS-publish one exact execution-lease generation under the guard."""

    current = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=record.revision_cid,
        slice_id=record.slice_id,
        lane_id=record.lane_id,
    )
    if current is None:
        if expected_current_cid or record.generation != 1 or record.prior_execution_lease_cid:
            raise ExecutionPlanError("execution lease creation lost its CAS precondition")
    else:
        current_cid, current_record = current
        if (
            current_cid != expected_current_cid
            or record.prior_execution_lease_cid != current_cid
            or record.generation != current_record.generation + 1
        ):
            raise ExecutionPlanError("execution lease update lost its CAS precondition")
    lease_cid = store.put_cas(record.to_dict())
    if _secure_store_cas(store, lease_cid) != record.to_dict():
        raise ExecutionPlanError("execution lease failed durable CAS round trip")
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_execution_lease",
        "revision_cid": record.revision_cid,
        "slice_id": record.slice_id,
        "lane_id": record.lane_id,
        "execution_lease_cid": lease_cid,
        "generation": record.generation,
        "execution_phase": record.phase,
    }
    key = plan_bound_execution_lease_key(
        record.revision_cid,
        record.slice_id,
        record.lane_id,
    )
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError("execution lease pointer failed durable round trip")
    observed = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=record.revision_cid,
        slice_id=record.slice_id,
        lane_id=record.lane_id,
    )
    if observed is None or observed[0] != lease_cid or observed[1] != record:
        raise ExecutionPlanError("execution lease publication was not exact")
    return lease_cid


def plan_bound_proposal_disposition_key(
    revision_cid: str,
    slice_id: str,
) -> str:
    """Return the one-winner disposition key for an immutable wave slice."""

    return (
        "plan-bound-proposal-disposition:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}"
    )


def plan_bound_terminal_missing_key(
    revision_cid: str,
    slice_id: str,
) -> str:
    """Return the one-winner terminal-missing key for a launched slice."""

    return (
        "plan-bound-terminal-missing:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}"
    )


def plan_bound_process_birth_exhausted_key(
    revision_cid: str,
    slice_id: str,
) -> str:
    """Return the one-winner key for a recovery-birth budget exhaustion."""

    return (
        "plan-bound-process-birth-exhausted:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}"
    )


def plan_bound_merge_terminal_failure_key(
    revision_cid: str,
    slice_id: str,
) -> str:
    """Return the one-winner terminal merge-recovery failure key."""

    return (
        "plan-bound-merge-terminal-failure:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}"
    )


_PLAN_BOUND_MERGE_TERMINAL_FAILURE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "revision_cid",
        "plan_root_cid",
        "execution_plan_cid",
        "capacity_snapshot_id",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "reassignment_cid",
        "task_id",
        "task_cid",
        "execution_lease_cid",
        "proposal_handoff_cid",
        "merge_authorization_cid",
        "merge_enqueue_intent_cid",
        "enqueue_fields_cid",
        "request_id",
        "queue_status",
        "queue_dedupe_key",
        "queue_request_json",
        "queue_request_sha256",
        "reason_codes",
        "observed_at_ms",
    }
)


def _validate_plan_bound_merge_terminal_failure_locked(
    store: PlanRevisionStore,
    record: Mapping[str, Any],
) -> None:
    """Cross-check a terminal canonical-queue outcome against its intent."""

    if not isinstance(record, Mapping) or set(record) != set(
        _PLAN_BOUND_MERGE_TERMINAL_FAILURE_FIELDS
    ):
        raise ExecutionPlanError("merge terminal failure fields are not exact")
    text_fields = set(_PLAN_BOUND_MERGE_TERMINAL_FAILURE_FIELDS) - {
        "reason_codes",
        "observed_at_ms",
    }
    if (
        record.get("schema") != PLAN_BOUND_MERGE_TERMINAL_FAILURE_SCHEMA
        or any(not isinstance(record.get(name), str) for name in text_fields)
        or not isinstance(record.get("reason_codes"), list)
        or not record["reason_codes"]
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in record["reason_codes"]
        )
        or record["reason_codes"] != sorted(set(record["reason_codes"]))
        or isinstance(record.get("observed_at_ms"), bool)
        or not isinstance(record.get("observed_at_ms"), int)
        or int(record["observed_at_ms"]) < 1
        or record.get("queue_status") not in {"failed", "quarantined"}
        or not str(record.get("request_id") or "")
        or not str(record.get("queue_request_json") or "")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(record.get("queue_request_sha256") or ""),
        )
        is None
    ):
        raise ExecutionPlanError("merge terminal failure scalars are invalid")
    queue_json = str(record["queue_request_json"])
    if (
        "sha256:" + hashlib.sha256(queue_json.encode("utf-8")).hexdigest()
        != record["queue_request_sha256"]
    ):
        raise ExecutionPlanError("merge terminal failure queue digest is mixed")
    try:
        queue_row = json.loads(
            queue_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionPlanError(
            "merge terminal failure queue evidence is malformed"
        ) from exc
    queue_fields = {
        "request_id",
        "branch_name",
        "task_id",
        "priority",
        "lane_id",
        "enqueued_at",
        "attempt",
        "metadata",
        "commit_sha",
        "canonical_task_id",
        "canonical_task_key",
        "status",
        "claimed_at",
        "consumer_id",
        "failure_count",
        "failure_reason",
        "claim_token",
        "claim_generation",
        "retry_not_before",
        "dedupe_key",
    }
    if (
        not isinstance(queue_row, Mapping)
        or set(queue_row) != queue_fields
        or queue_json
        != json.dumps(dict(queue_row), sort_keys=True, separators=(",", ":"))
        or queue_row.get("request_id") != record["request_id"]
        or queue_row.get("status") != record["queue_status"]
        or queue_row.get("dedupe_key") != record["queue_dedupe_key"]
        or not isinstance(queue_row.get("metadata"), Mapping)
        or isinstance(queue_row.get("attempt"), bool)
        or not isinstance(queue_row.get("attempt"), int)
        or isinstance(queue_row.get("failure_count"), bool)
        or not isinstance(queue_row.get("failure_count"), int)
        or int(queue_row["failure_count"]) < 1
    ):
        raise ExecutionPlanError("merge terminal failure queue evidence is mixed")

    lease = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=str(record["revision_cid"]),
        slice_id=str(record["slice_id"]),
        lane_id=str(record["lane_id"]),
    )
    if (
        lease is None
        or lease[0] != record["execution_lease_cid"]
        or lease[1].phase
        not in {"merge_enqueue_prepared", "merge_enqueue_confirmed"}
    ):
        raise ExecutionPlanError("merge terminal failure lost its execution lease")
    execution_lease = lease[1]
    expected = {
        "plan_root_cid": execution_lease.plan_root_cid,
        "execution_plan_cid": execution_lease.execution_plan_cid,
        "capacity_snapshot_id": execution_lease.capacity_snapshot_id,
        "slice_manifest_cid": execution_lease.slice_manifest_cid,
        "reassignment_cid": execution_lease.reassignment_cid,
        "task_id": execution_lease.active_task_id,
        "task_cid": execution_lease.active_task_cid,
        "proposal_handoff_cid": execution_lease.proposal_handoff_cid,
        "merge_authorization_cid": execution_lease.merge_authorization_cid,
        "merge_enqueue_intent_cid": execution_lease.merge_enqueue_intent_cid,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise ExecutionPlanError("merge terminal failure authority is mixed")
    intent = _secure_store_cas(store, execution_lease.merge_enqueue_intent_cid)
    if (
        intent.get("enqueue_fields_cid") != record["enqueue_fields_cid"]
        or content_identity(dict(intent.get("enqueue_fields") or {}))
        != record["enqueue_fields_cid"]
        or queue_row.get("task_id") != execution_lease.active_task_id
        or queue_row.get("canonical_task_id")
        != execution_lease.active_task_cid
        or (
            execution_lease.merge_request_id
            and execution_lease.merge_request_id != record["request_id"]
        )
    ):
        raise ExecutionPlanError("merge terminal failure intent evidence is mixed")


def _load_plan_bound_merge_terminal_failure_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Load the one-winner terminal queue outcome under the store guard."""

    key = plan_bound_merge_terminal_failure_key(revision_cid, slice_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    if set(pointer) != {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "failure_cid",
    } or pointer != {
        "phase": "committed",
        "operation": "plan_bound_merge_terminal_failure",
        "revision_cid": revision_cid,
        "slice_id": slice_id,
        "failure_cid": pointer.get("failure_cid"),
    }:
        raise ExecutionPlanError("merge terminal failure pointer is malformed")
    failure_cid = _text(pointer.get("failure_cid"), "failure_cid")
    record = _secure_store_cas(store, failure_cid)
    _validate_plan_bound_merge_terminal_failure_locked(store, record)
    return failure_cid, dict(record)


def _publish_plan_bound_merge_terminal_failure_locked(
    store: PlanRevisionStore,
    record: Mapping[str, Any],
) -> str:
    """Publish one immutable failed/quarantined queue outcome."""

    _validate_plan_bound_merge_terminal_failure_locked(store, record)
    existing = _load_plan_bound_merge_terminal_failure_locked(
        store,
        revision_cid=str(record["revision_cid"]),
        slice_id=str(record["slice_id"]),
    )
    if existing is not None:
        if existing[1] != dict(record):
            raise ExecutionPlanError(
                "merge terminal failure conflicts with its one-winner record"
            )
        return existing[0]
    failure_cid = store.put_cas(dict(record))
    if _secure_store_cas(store, failure_cid) != dict(record):
        raise ExecutionPlanError(
            "merge terminal failure failed CAS round trip"
        )
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_merge_terminal_failure",
        "revision_cid": record["revision_cid"],
        "slice_id": record["slice_id"],
        "failure_cid": failure_cid,
    }
    store.put_continuation(
        plan_bound_merge_terminal_failure_key(
            str(record["revision_cid"]),
            str(record["slice_id"]),
        ),
        pointer,
    )
    if _secure_store_continuation(
        store,
        plan_bound_merge_terminal_failure_key(
            str(record["revision_cid"]),
            str(record["slice_id"]),
        ),
    ) != pointer:
        raise ExecutionPlanError(
            "merge terminal failure pointer failed durable round trip"
        )
    return failure_cid


def plan_bound_recovery_launch_key(
    revision_cid: str,
    slice_id: str,
    lane_id: str,
) -> str:
    """Return the authoritative pointer for one slice's recovery launch."""

    return (
        "plan-bound-recovery-launch:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_id, 'slice_id')}:"
        f"{_text(lane_id, 'lane_id')}"
    )


def _load_plan_bound_recovery_launch_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
    lane_id: str,
    authorization_cid: str,
) -> PlanBoundRecoveryLaunch:
    """Load and rederive one recovery-only launch decision under the guard."""

    key = plan_bound_recovery_launch_key(revision_cid, slice_id, lane_id)
    pointer = _secure_store_continuation(store, key)
    expected_pointer = {
        "phase": "committed",
        "operation": "plan_bound_recovery_launch",
        "revision_cid": revision_cid,
        "slice_id": slice_id,
        "lane_id": lane_id,
        "authorization_cid": authorization_cid,
    }
    if pointer != expected_pointer:
        raise ExecutionPlanError(
            "recovery-launch pointer is absent, stale, or malformed"
        )
    decision = PlanBoundRecoveryLaunch.from_dict(
        _secure_store_cas(store, authorization_cid)
    )
    current = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
        lane_id=lane_id,
    )
    if current is None:
        raise ExecutionPlanError("recovery launch lost its execution lease")
    execution_lease_cid, lease = current
    if _load_plan_bound_merge_terminal_failure_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
    ) is not None:
        raise ExecutionPlanError(
            "terminal merge failure forbids recovery launch"
        )
    expected = {
        "revision_cid": lease.revision_cid,
        "plan_root_cid": lease.plan_root_cid,
        "execution_plan_cid": lease.execution_plan_cid,
        "capacity_snapshot_id": lease.capacity_snapshot_id,
        "slice_manifest_cid": lease.slice_manifest_cid,
        "slice_id": lease.slice_id,
        "lane_id": lease.lane_id,
        "reassignment_cid": lease.reassignment_cid,
        "execution_lease_cid": execution_lease_cid,
        "execution_phase": lease.phase,
        "proposal_handoff_cid": lease.proposal_handoff_cid,
        "merge_authorization_cid": lease.merge_authorization_cid,
        "merge_enqueue_intent_cid": lease.merge_enqueue_intent_cid,
        "merge_request_id": lease.merge_request_id,
        "merge_queue_receipt_cid": lease.merge_queue_receipt_cid,
    }
    if any(getattr(decision, name) != value for name, value in expected.items()):
        raise ExecutionPlanError(
            "recovery-launch decision lost its exact handoff authority"
        )
    manifest = ConfiguredBoardExecutionSlices.from_dict(
        _secure_store_cas(store, lease.slice_manifest_cid)
    )
    if (
        manifest.source_head != decision.source_head
        or manifest.repository_tree_id != decision.source_tree
    ):
        raise ExecutionPlanError(
            "recovery-launch decision carries a foreign source generation"
        )
    return decision


def _publish_plan_bound_recovery_launch_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
    lane_id: str,
    source_head: str,
    source_tree: str,
    repository_head: str,
    repository_tree: str,
    runtime_artifacts: tuple[Mapping[str, Any], ...],
    launch_artifact_paths: tuple[str, ...],
) -> tuple[str, PlanBoundRecoveryLaunch]:
    """Authorize only an already-persisted proposal/merge handoff restart."""

    current = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
        lane_id=lane_id,
    )
    if current is None:
        raise ExecutionPlanError("recovery launch has no execution lease")
    execution_lease_cid, lease = current
    decision = PlanBoundRecoveryLaunch(
        revision_cid=lease.revision_cid,
        plan_root_cid=lease.plan_root_cid,
        execution_plan_cid=lease.execution_plan_cid,
        capacity_snapshot_id=lease.capacity_snapshot_id,
        slice_manifest_cid=lease.slice_manifest_cid,
        slice_id=lease.slice_id,
        lane_id=lease.lane_id,
        reassignment_cid=lease.reassignment_cid,
        execution_lease_cid=execution_lease_cid,
        execution_phase=lease.phase,
        proposal_handoff_cid=lease.proposal_handoff_cid,
        merge_authorization_cid=lease.merge_authorization_cid,
        merge_enqueue_intent_cid=lease.merge_enqueue_intent_cid,
        merge_request_id=lease.merge_request_id,
        merge_queue_receipt_cid=lease.merge_queue_receipt_cid,
        source_head=source_head,
        source_tree=source_tree,
        repository_head=repository_head,
        repository_tree=repository_tree,
        runtime_artifacts=runtime_artifacts,
        launch_artifact_paths=launch_artifact_paths,
    )
    manifest = ConfiguredBoardExecutionSlices.from_dict(
        _secure_store_cas(store, lease.slice_manifest_cid)
    )
    if (
        manifest.source_head != source_head
        or manifest.repository_tree_id != source_tree
    ):
        raise ExecutionPlanError(
            "recovery launch source differs from the immutable manifest"
        )
    if _load_plan_bound_merge_terminal_failure_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
    ) is not None:
        raise ExecutionPlanError(
            "terminal merge failure forbids recovery launch"
        )
    authorization_cid = store.put_cas(decision.to_dict())
    if _secure_store_cas(store, authorization_cid) != decision.to_dict():
        raise ExecutionPlanError(
            "recovery-launch decision failed CAS round trip"
        )
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_recovery_launch",
        "revision_cid": revision_cid,
        "slice_id": slice_id,
        "lane_id": lane_id,
        "authorization_cid": authorization_cid,
    }
    store.put_continuation(
        plan_bound_recovery_launch_key(revision_cid, slice_id, lane_id),
        pointer,
    )
    if _secure_store_continuation(
        store,
        plan_bound_recovery_launch_key(revision_cid, slice_id, lane_id),
    ) != pointer:
        raise ExecutionPlanError(
            "recovery-launch pointer failed durable round trip"
        )
    observed = _load_plan_bound_recovery_launch_locked(
        store,
        revision_cid=revision_cid,
        slice_id=slice_id,
        lane_id=lane_id,
        authorization_cid=authorization_cid,
    )
    if observed != decision:
        raise ExecutionPlanError(
            "recovery-launch authority changed after publication"
        )
    return authorization_cid, decision


def plan_bound_wave_diff_barrier_key(
    revision_cid: str,
    slice_manifest_cid: str,
) -> str:
    """Return the one-winner whole-wave decision key."""

    return (
        "plan-bound-wave-diff-barrier:"
        f"{_text(revision_cid, 'revision_cid')}:"
        f"{_text(slice_manifest_cid, 'slice_manifest_cid')}"
    )


def _plan_bound_wave_diff_barrier_window_key(
    revision_cid: str,
    slice_manifest_cid: str,
) -> str:
    return f"{plan_bound_wave_diff_barrier_key(revision_cid, slice_manifest_cid)}:window"


def _active_plan_bound_manifest_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_manifest_cid: str,
) -> tuple[PlanRevisionActiveProjection, PlanRevision, ConfiguredBoardExecutionSlices]:
    """Load one active revision and its exact immutable slice manifest."""

    active = _secure_store_active(store)
    if active is None or active.revision_cid != revision_cid or active.quarantined:
        raise ExecutionPlanError("wave barrier requires the active revision")
    revision_payload = _secure_store_cas(store, revision_cid)
    revision = PlanRevision.from_dict(revision_payload)
    if revision.to_dict() != revision_payload:
        raise ExecutionPlanError("wave barrier revision changed during decode")
    if revision.materialization_transaction_cid != slice_manifest_cid:
        raise ExecutionPlanError("wave barrier manifest is not owned by revision")
    manifest_payload = _secure_store_cas(store, slice_manifest_cid)
    manifest = ConfiguredBoardExecutionSlices.from_dict(manifest_payload)
    if manifest.to_dict() != manifest_payload:
        raise ExecutionPlanError("wave barrier manifest changed during decode")
    if (
        active.plan_root_cid != manifest.plan_root_cid
        or revision.execution_plan_cid == ""
    ):
        raise ExecutionPlanError("wave barrier authority is mixed")
    # The configured compiler launches one bounded daemon per selected task.
    # Public slice DTOs remain general, but plan-bound effect execution is
    # deliberately narrowed to one ID/CID pair per launched nonempty slice.
    if not manifest.nonempty or any(
        len(item.task_ids) != 1 or len(item.task_cids) != 1
        for item in manifest.nonempty
    ):
        raise ExecutionPlanError(
            "plan-bound wave barrier requires singleton launched slices"
        )
    return active, revision, manifest


def _validate_plan_bound_proposal_disposition_locked(
    store: PlanRevisionStore,
    disposition: PlanBoundProposalDisposition,
) -> None:
    """Cross-check a disposition before or after publishing its pointer."""

    active, revision, manifest = _active_plan_bound_manifest_locked(
        store,
        revision_cid=disposition.revision_cid,
        slice_manifest_cid=disposition.slice_manifest_cid,
    )
    matches = tuple(
        item
        for item in manifest.nonempty
        if item.slice_id == disposition.slice_id
    )
    if len(matches) != 1:
        raise ExecutionPlanError("proposal disposition slice is absent")
    execution_slice = matches[0]
    adapter = ProductionParallelPlanAdapter(store)
    adapter._validate_slice_owner_locked(  # noqa: SLF001
        revision_cid=disposition.revision_cid,
        slice_manifest_cid=disposition.slice_manifest_cid,
        slice_id=disposition.slice_id,
        lane_id=disposition.lane_id,
        reassignment_cid=disposition.reassignment_cid,
    )
    lease = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=disposition.revision_cid,
        slice_id=disposition.slice_id,
        lane_id=disposition.lane_id,
    )
    disposition_lease_payload = _secure_store_cas(
        store,
        disposition.execution_lease_cid,
    )
    disposition_lease = PlanBoundExecutionLease.from_dict(
        disposition_lease_payload
    )
    if disposition_lease.to_dict() != disposition_lease_payload:
        raise ExecutionPlanError(
            "proposal disposition lease changed during typed decode"
        )
    if lease is None:
        raise ExecutionPlanError("proposal disposition lost its execution lease")
    current_lease_cid, current_lease = lease
    if current_lease_cid != disposition.execution_lease_cid:
        merge_phases = {
            "merge_enqueue_reached",
            "merge_enqueue_prepared",
            "merge_enqueue_confirmed",
            "merge_completed",
        }
        authorization = (
            _secure_store_cas(store, current_lease.merge_authorization_cid)
            if current_lease.phase in merge_phases
            else {}
        )
        immutable_effect_fields = (
            "revision_cid",
            "plan_root_cid",
            "execution_plan_cid",
            "capacity_snapshot_id",
            "slice_manifest_cid",
            "slice_id",
            "lane_id",
            "reassignment_cid",
            "task_ids",
            "task_cids",
            "compiled_task_bindings",
            "process_birth_cid",
            "process_birth",
            "active_task_id",
            "active_task_cid",
            "daemon_process_birth",
            "canonical_claim_path",
            "canonical_claim_cid",
            "canonical_claim_lease_id",
            "workspace_lifecycle_path",
            "workspace_lifecycle_cid",
            "workspace_record_id",
            "workspace_path",
            "workspace_lease_id",
            "workspace_fence",
            "provider_ready",
            "proposal_id",
            "proposal_receipt_id",
            "proposal_reason_codes",
            "actual_changed_paths",
        )
        if not (
            disposition.outcome in {"changed", "no_change"}
            and current_lease.phase in merge_phases
            and authorization.get("execution_lease_cid")
            == disposition.execution_lease_cid
            and all(
                getattr(current_lease, name) == getattr(disposition_lease, name)
                for name in immutable_effect_fields
            )
        ):
            raise ExecutionPlanError(
                "proposal disposition lost its execution lease"
            )
    lease_record = disposition_lease
    expected_phases = (
        {"proposal_ready"}
        if disposition.outcome in {"changed", "no_change"}
        else {"proposal_rejected", "scope_drift"}
    )
    if (
        lease_record.phase not in expected_phases
        or active.plan_root_cid != disposition.plan_root_cid
        or revision.execution_plan_cid != disposition.execution_plan_cid
        or manifest.capacity_snapshot_id != disposition.capacity_snapshot_id
        or disposition.slice_manifest_cid
        != revision.materialization_transaction_cid
        or execution_slice.task_pairs
        != ((disposition.task_id, disposition.task_cid),)
        or lease_record.active_task_id != disposition.task_id
        or lease_record.active_task_cid != disposition.task_cid
        or lease_record.process_birth_cid != disposition.process_birth_cid
        or lease_record.reassignment_cid != disposition.reassignment_cid
        or lease_record.proposal_id != disposition.proposal_id
        or lease_record.proposal_receipt_id != disposition.proposal_receipt_id
        or lease_record.proposal_reason_codes != disposition.reason_codes
        or lease_record.actual_changed_paths != disposition.actual_changed_paths
        or lease_record.merge_enqueue_reached
        or disposition.baseline_ref != manifest.source_head
    ):
        raise ExecutionPlanError("proposal disposition authority is mixed")


def _load_plan_bound_proposal_disposition_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
) -> tuple[str, PlanBoundProposalDisposition] | None:
    """Load and revalidate one current-owner disposition under the store guard."""

    key = plan_bound_proposal_disposition_key(revision_cid, slice_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    expected_pointer_fields = {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "disposition_cid",
        "outcome",
    }
    if (
        set(pointer) != expected_pointer_fields
        or pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_proposal_disposition"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_id") != slice_id
    ):
        raise ExecutionPlanError("proposal disposition pointer is malformed")
    disposition_cid = _text(
        pointer.get("disposition_cid"), "disposition_cid"
    )
    disposition = PlanBoundProposalDisposition.from_dict(
        _secure_store_cas(store, disposition_cid)
    )
    if (
        disposition.revision_cid != revision_cid
        or disposition.slice_id != slice_id
        or disposition.outcome != pointer.get("outcome")
    ):
        raise ExecutionPlanError("proposal disposition pointer is mixed")
    _validate_plan_bound_proposal_disposition_locked(store, disposition)
    return disposition_cid, disposition


def _publish_plan_bound_proposal_disposition_locked(
    store: PlanRevisionStore,
    disposition: PlanBoundProposalDisposition,
) -> str:
    """Publish an immutable, slice-keyed disposition under the store guard."""

    existing = _load_plan_bound_proposal_disposition_locked(
        store,
        revision_cid=disposition.revision_cid,
        slice_id=disposition.slice_id,
    )
    if existing is not None:
        if existing[1] != disposition:
            raise ExecutionPlanError(
                "proposal disposition conflicts with its one-winner slice record"
            )
        return existing[0]
    if _secure_store_continuation(
        store,
        plan_bound_terminal_missing_key(
            disposition.revision_cid,
            disposition.slice_id,
        ),
    ) is not None:
        raise ExecutionPlanError(
            "proposal disposition conflicts with terminal-missing evidence"
        )
    if _secure_store_continuation(
        store,
        plan_bound_wave_diff_barrier_key(
            disposition.revision_cid,
            disposition.slice_manifest_cid,
        ),
    ) is not None:
        raise ExecutionPlanError(
            "proposal disposition arrived after the terminal wave barrier"
        )
    _validate_plan_bound_proposal_disposition_locked(store, disposition)
    disposition_cid = store.put_cas(disposition.to_dict())
    if _secure_store_cas(store, disposition_cid) != disposition.to_dict():
        raise ExecutionPlanError("proposal disposition failed CAS round trip")
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_proposal_disposition",
        "revision_cid": disposition.revision_cid,
        "slice_id": disposition.slice_id,
        "disposition_cid": disposition_cid,
        "outcome": disposition.outcome,
    }
    key = plan_bound_proposal_disposition_key(
        disposition.revision_cid,
        disposition.slice_id,
    )
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError("proposal disposition pointer did not persist")
    observed = _load_plan_bound_proposal_disposition_locked(
        store,
        revision_cid=disposition.revision_cid,
        slice_id=disposition.slice_id,
    )
    if observed != (disposition_cid, disposition):
        raise ExecutionPlanError("proposal disposition publication was not exact")
    return disposition_cid


def _validate_plan_bound_terminal_missing_locked(
    store: PlanRevisionStore,
    terminal: PlanBoundTerminalMissing,
) -> None:
    """Cross-check one terminal absence against current process authority."""

    active, revision, manifest = _active_plan_bound_manifest_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_manifest_cid=terminal.slice_manifest_cid,
    )
    matches = tuple(
        item for item in manifest.nonempty if item.slice_id == terminal.slice_id
    )
    if len(matches) != 1:
        raise ExecutionPlanError("terminal-missing slice is absent")
    execution_slice = matches[0]
    ProductionParallelPlanAdapter(store)._validate_slice_owner_locked(  # noqa: SLF001
        revision_cid=terminal.revision_cid,
        slice_manifest_cid=terminal.slice_manifest_cid,
        slice_id=terminal.slice_id,
        lane_id=terminal.lane_id,
        reassignment_cid=terminal.reassignment_cid,
    )
    lease = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_id=terminal.slice_id,
        lane_id=terminal.lane_id,
    )
    if lease is None or lease[1].process_birth_cid != terminal.process_birth_cid:
        raise ExecutionPlanError(
            "terminal-missing lost its current execution/process birth"
        )
    if lease[1].phase in {
        "merge_enqueue_reached",
        "merge_enqueue_prepared",
        "merge_enqueue_confirmed",
    }:
        raise ExecutionPlanError(
            "terminal-missing cannot follow merge-enqueue admission"
        )
    if (
        active.plan_root_cid != terminal.plan_root_cid
        or revision.execution_plan_cid != terminal.execution_plan_cid
        or manifest.capacity_snapshot_id != terminal.capacity_snapshot_id
        or execution_slice.task_pairs != ((terminal.task_id, terminal.task_cid),)
    ):
        raise ExecutionPlanError("terminal-missing authority is mixed")

    birth_binding = _load_plan_bound_process_birth_chain_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_id=terminal.slice_id,
        lane_id=terminal.lane_id,
    )
    if birth_binding is None:
        raise ExecutionPlanError("terminal-missing launch birth is absent")
    birth_cid, typed_birth, _birth_chain = birth_binding
    birth = typed_birth.to_dict()
    if (
        birth_cid != terminal.process_birth_cid
        or typed_birth.plan_root_cid != terminal.plan_root_cid
        or typed_birth.execution_plan_cid != terminal.execution_plan_cid
        or typed_birth.capacity_snapshot_id != terminal.capacity_snapshot_id
        or typed_birth.slice_manifest_cid != terminal.slice_manifest_cid
        or typed_birth.task_ids != (terminal.task_id,)
        or typed_birth.task_cids != (terminal.task_cid,)
    ):
        raise ExecutionPlanError("terminal-missing launch birth is mixed")

    fence = _secure_store_cas(store, terminal.process_fence_cid)
    fence_fields = {
        "schema",
        "revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "reassignment_cid",
        "process_birth_cid",
        "profile",
        "process_birth",
        "fenced_tree",
        "exit_code",
        "observed_at_ms",
    }
    if set(fence) != fence_fields or fence.get("schema") != (
        "ipfs_accelerate_py/agent-supervisor/plan-bound-terminal-process-fence@1"
    ):
        raise ExecutionPlanError("terminal-missing process fence is malformed")
    try:
        from ..control.lifecycle_orchestrator import (
            LifecycleProfile,
            ProcessIdentity,
            ProcessTreeSnapshot,
        )

        profile = LifecycleProfile.from_dict(fence["profile"])
        identity = ProcessIdentity.from_dict(fence["process_birth"])
        fenced_tree = ProcessTreeSnapshot.from_dict(fence["fenced_tree"])
    except Exception as exc:
        raise ExecutionPlanError(
            "terminal-missing process fence lifecycle evidence is invalid"
        ) from exc
    if (
        profile.to_dict() != fence["profile"]
        or identity.to_dict() != fence["process_birth"]
        or fenced_tree.to_dict() != fence["fenced_tree"]
        or fenced_tree.members
        or fence.get("revision_cid") != terminal.revision_cid
        or fence.get("slice_manifest_cid") != terminal.slice_manifest_cid
        or fence.get("slice_id") != terminal.slice_id
        or fence.get("lane_id") != terminal.lane_id
        or fence.get("reassignment_cid") != terminal.reassignment_cid
        or fence.get("process_birth_cid") != terminal.process_birth_cid
        or fence.get("profile") != birth.get("profile")
        or fence.get("process_birth") != birth.get("process_birth")
        or fence.get("exit_code") != terminal.exit_code
        or fence.get("observed_at_ms") != terminal.observed_at_ms
        or identity.profile_id != profile.profile_id
        or identity.run_id != profile.run_id
        or identity.target_id != profile.target_id
    ):
        raise ExecutionPlanError("terminal-missing process fence is mixed")


def _load_plan_bound_terminal_missing_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
) -> tuple[str, PlanBoundTerminalMissing] | None:
    """Load one current-owner terminal absence under the store guard."""

    key = plan_bound_terminal_missing_key(revision_cid, slice_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    if set(pointer) != {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "terminal_missing_cid",
    } or (
        pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_terminal_missing"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_id") != slice_id
    ):
        raise ExecutionPlanError("terminal-missing pointer is malformed")
    terminal_cid = _text(
        pointer.get("terminal_missing_cid"),
        "terminal_missing_cid",
    )
    terminal = PlanBoundTerminalMissing.from_dict(
        _secure_store_cas(store, terminal_cid)
    )
    if terminal.revision_cid != revision_cid or terminal.slice_id != slice_id:
        raise ExecutionPlanError("terminal-missing pointer is mixed")
    _validate_plan_bound_terminal_missing_locked(store, terminal)
    return terminal_cid, terminal


def _publish_plan_bound_terminal_missing_locked(
    store: PlanRevisionStore,
    terminal: PlanBoundTerminalMissing,
) -> str:
    """Publish one immutable terminal absence without replacing a result."""

    existing = _load_plan_bound_terminal_missing_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_id=terminal.slice_id,
    )
    if existing is not None:
        if existing[1] != terminal:
            raise ExecutionPlanError("terminal-missing one-winner CAS conflicts")
        return existing[0]
    if _load_plan_bound_proposal_disposition_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_id=terminal.slice_id,
    ) is not None:
        raise ExecutionPlanError(
            "terminal-missing conflicts with a published disposition"
        )
    if _secure_store_continuation(
        store,
        plan_bound_wave_diff_barrier_key(
            terminal.revision_cid,
            terminal.slice_manifest_cid,
        ),
    ) is not None:
        raise ExecutionPlanError(
            "terminal-missing arrived after the terminal wave barrier"
        )
    _validate_plan_bound_terminal_missing_locked(store, terminal)
    terminal_cid = store.put_cas(terminal.to_dict())
    if _secure_store_cas(store, terminal_cid) != terminal.to_dict():
        raise ExecutionPlanError("terminal-missing failed CAS round trip")
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_terminal_missing",
        "revision_cid": terminal.revision_cid,
        "slice_id": terminal.slice_id,
        "terminal_missing_cid": terminal_cid,
    }
    key = plan_bound_terminal_missing_key(terminal.revision_cid, terminal.slice_id)
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError("terminal-missing pointer did not persist")
    observed = _load_plan_bound_terminal_missing_locked(
        store,
        revision_cid=terminal.revision_cid,
        slice_id=terminal.slice_id,
    )
    if observed != (terminal_cid, terminal):
        raise ExecutionPlanError("terminal-missing publication was not exact")
    return terminal_cid


def _validate_plan_bound_process_birth_exhausted_locked(
    store: PlanRevisionStore,
    exhausted: PlanBoundProcessBirthExhausted,
) -> None:
    """Cross-check a bounded recovery failure against every durable authority."""

    active, revision, manifest = _active_plan_bound_manifest_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_manifest_cid=exhausted.slice_manifest_cid,
    )
    matches = tuple(
        item for item in manifest.nonempty if item.slice_id == exhausted.slice_id
    )
    if len(matches) != 1:
        raise ExecutionPlanError("process-birth-exhausted slice is absent")
    execution_slice = matches[0]
    ProductionParallelPlanAdapter(store)._validate_slice_owner_locked(  # noqa: SLF001
        revision_cid=exhausted.revision_cid,
        slice_manifest_cid=exhausted.slice_manifest_cid,
        slice_id=exhausted.slice_id,
        lane_id=exhausted.lane_id,
        reassignment_cid=exhausted.reassignment_cid,
    )
    lease = _load_plan_bound_execution_lease_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_id=exhausted.slice_id,
        lane_id=exhausted.lane_id,
    )
    if (
        lease is None
        or lease[0] != exhausted.execution_lease_cid
        or lease[1].phase
        not in {
            "proposal_ready",
            "merge_enqueue_prepared",
            "merge_enqueue_confirmed",
        }
        or lease[1].active_task_id != exhausted.task_id
        or lease[1].active_task_cid != exhausted.task_cid
    ):
        raise ExecutionPlanError(
            "process-birth-exhausted lost its recoverable execution lease"
        )
    disposition = _load_plan_bound_proposal_disposition_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_id=exhausted.slice_id,
    )
    if (
        disposition is None
        or disposition[0] != exhausted.disposition_cid
        or disposition[1].outcome not in {"changed", "no_change"}
        or disposition[1].task_id != exhausted.task_id
        or disposition[1].task_cid != exhausted.task_cid
        or disposition[1].lane_id != exhausted.lane_id
        or disposition[1].reassignment_cid != exhausted.reassignment_cid
    ):
        raise ExecutionPlanError(
            "process-birth-exhausted lost its proposal disposition"
        )
    if (
        active.plan_root_cid != exhausted.plan_root_cid
        or revision.execution_plan_cid != exhausted.execution_plan_cid
        or manifest.capacity_snapshot_id != exhausted.capacity_snapshot_id
        or (exhausted.task_id, exhausted.task_cid)
        not in execution_slice.task_pairs
    ):
        raise ExecutionPlanError("process-birth-exhausted authority is mixed")

    birth_binding = _load_plan_bound_process_birth_chain_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_id=exhausted.slice_id,
        lane_id=exhausted.lane_id,
    )
    if birth_binding is None:
        raise ExecutionPlanError("process-birth-exhausted launch birth is absent")
    birth_cid, typed_birth, _birth_chain = birth_binding
    if (
        birth_cid != exhausted.process_birth_cid
        or typed_birth.generation != exhausted.generation
        or typed_birth.global_budget != exhausted.global_budget
        or typed_birth.generation != MAX_PLAN_BOUND_WAVE_TRANSFERS
        or typed_birth.plan_root_cid != exhausted.plan_root_cid
        or typed_birth.execution_plan_cid != exhausted.execution_plan_cid
        or typed_birth.capacity_snapshot_id != exhausted.capacity_snapshot_id
        or typed_birth.slice_manifest_cid != exhausted.slice_manifest_cid
        or (exhausted.task_id, exhausted.task_cid)
        not in tuple(zip(typed_birth.task_ids, typed_birth.task_cids, strict=True))
    ):
        raise ExecutionPlanError("process-birth-exhausted launch birth is mixed")

    fence = _secure_store_cas(store, exhausted.process_fence_cid)
    fence_fields = {
        "schema",
        "revision_cid",
        "slice_manifest_cid",
        "slice_id",
        "lane_id",
        "reassignment_cid",
        "process_birth_cid",
        "generation",
        "global_budget",
        "profile",
        "process_birth",
        "fenced_tree",
        "exit_code",
        "observed_at_ms",
    }
    if set(fence) != fence_fields or fence.get("schema") != (
        "ipfs_accelerate_py/agent-supervisor/"
        "plan-bound-process-birth-exhausted-fence@1"
    ):
        raise ExecutionPlanError(
            "process-birth-exhausted process fence is malformed"
        )
    try:
        from ..control.lifecycle_orchestrator import (
            LifecycleProfile,
            ProcessIdentity,
            ProcessTreeSnapshot,
        )

        profile = LifecycleProfile.from_dict(fence["profile"])
        identity = ProcessIdentity.from_dict(fence["process_birth"])
        fenced_tree = ProcessTreeSnapshot.from_dict(fence["fenced_tree"])
    except Exception as exc:
        raise ExecutionPlanError(
            "process-birth-exhausted lifecycle evidence is invalid"
        ) from exc
    if (
        profile.to_dict() != fence["profile"]
        or identity.to_dict() != fence["process_birth"]
        or fenced_tree.to_dict() != fence["fenced_tree"]
        or fenced_tree.members
        or fence.get("revision_cid") != exhausted.revision_cid
        or fence.get("slice_manifest_cid") != exhausted.slice_manifest_cid
        or fence.get("slice_id") != exhausted.slice_id
        or fence.get("lane_id") != exhausted.lane_id
        or fence.get("reassignment_cid") != exhausted.reassignment_cid
        or fence.get("process_birth_cid") != exhausted.process_birth_cid
        or fence.get("generation") != exhausted.generation
        or fence.get("global_budget") != exhausted.global_budget
        or fence.get("profile") != typed_birth.profile
        or fence.get("process_birth") != typed_birth.process_birth
        or fence.get("exit_code") != exhausted.exit_code
        or fence.get("observed_at_ms") != exhausted.observed_at_ms
        or identity.profile_id != profile.profile_id
        or identity.run_id != profile.run_id
        or identity.target_id != profile.target_id
    ):
        raise ExecutionPlanError("process-birth-exhausted process fence is mixed")


def _load_plan_bound_process_birth_exhausted_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_id: str,
) -> tuple[str, PlanBoundProcessBirthExhausted] | None:
    """Load the one-winner recovery-birth exhaustion under the store guard."""

    key = plan_bound_process_birth_exhausted_key(revision_cid, slice_id)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    if set(pointer) != {
        "phase",
        "operation",
        "revision_cid",
        "slice_id",
        "exhausted_cid",
    } or pointer != {
        "phase": "committed",
        "operation": "plan_bound_process_birth_exhausted",
        "revision_cid": revision_cid,
        "slice_id": slice_id,
        "exhausted_cid": pointer.get("exhausted_cid"),
    }:
        raise ExecutionPlanError(
            "process-birth-exhausted pointer is malformed"
        )
    exhausted_cid = _text(pointer.get("exhausted_cid"), "exhausted_cid")
    exhausted = PlanBoundProcessBirthExhausted.from_dict(
        _secure_store_cas(store, exhausted_cid)
    )
    if (
        exhausted.revision_cid != revision_cid
        or exhausted.slice_id != slice_id
    ):
        raise ExecutionPlanError("process-birth-exhausted pointer is mixed")
    _validate_plan_bound_process_birth_exhausted_locked(store, exhausted)
    return exhausted_cid, exhausted


def _publish_plan_bound_process_birth_exhausted_locked(
    store: PlanRevisionStore,
    exhausted: PlanBoundProcessBirthExhausted,
) -> str:
    """Publish one immutable recovery-birth exhaustion with one-winner CAS."""

    existing = _load_plan_bound_process_birth_exhausted_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_id=exhausted.slice_id,
    )
    if existing is not None:
        if existing[1] != exhausted:
            raise ExecutionPlanError(
                "process-birth-exhausted one-winner CAS conflicts"
            )
        return existing[0]
    _validate_plan_bound_process_birth_exhausted_locked(store, exhausted)
    exhausted_cid = store.put_cas(exhausted.to_dict())
    if _secure_store_cas(store, exhausted_cid) != exhausted.to_dict():
        raise ExecutionPlanError(
            "process-birth-exhausted failed CAS round trip"
        )
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_process_birth_exhausted",
        "revision_cid": exhausted.revision_cid,
        "slice_id": exhausted.slice_id,
        "exhausted_cid": exhausted_cid,
    }
    key = plan_bound_process_birth_exhausted_key(
        exhausted.revision_cid,
        exhausted.slice_id,
    )
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError(
            "process-birth-exhausted pointer did not persist"
        )
    observed = _load_plan_bound_process_birth_exhausted_locked(
        store,
        revision_cid=exhausted.revision_cid,
        slice_id=exhausted.slice_id,
    )
    if observed != (exhausted_cid, exhausted):
        raise ExecutionPlanError(
            "process-birth-exhausted publication was not exact"
        )
    return exhausted_cid


def _load_plan_bound_wave_diff_barrier_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_manifest_cid: str,
) -> tuple[str, PlanBoundWaveDiffBarrier] | None:
    key = plan_bound_wave_diff_barrier_key(revision_cid, slice_manifest_cid)
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    if set(pointer) != {
        "phase",
        "operation",
        "revision_cid",
        "slice_manifest_cid",
        "barrier_cid",
        "decision",
    } or (
        pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_wave_diff_barrier"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_manifest_cid") != slice_manifest_cid
    ):
        raise ExecutionPlanError("wave diff barrier pointer is malformed")
    barrier_cid = _text(pointer.get("barrier_cid"), "barrier_cid")
    barrier = PlanBoundWaveDiffBarrier.from_dict(
        _secure_store_cas(store, barrier_cid)
    )
    if (
        barrier.revision_cid != revision_cid
        or barrier.slice_manifest_cid != slice_manifest_cid
        or barrier.decision != pointer.get("decision")
    ):
        raise ExecutionPlanError("wave diff barrier pointer is mixed")
    active, revision, manifest = _active_plan_bound_manifest_locked(
        store,
        revision_cid=revision_cid,
        slice_manifest_cid=slice_manifest_cid,
    )
    expected_members = _plan_bound_wave_expected_members(manifest)
    disposition_rows, disposition_records, lease_records = (
        _plan_bound_wave_disposition_evidence_locked(
            store,
            revision_cid=revision_cid,
            expected_members=expected_members,
        )
    )
    terminal_rows, terminal_records = (
        _plan_bound_wave_terminal_evidence_locked(
            store,
            revision_cid=revision_cid,
            expected_members=expected_members,
        )
    )
    window = _load_plan_bound_wave_window_locked(
        store,
        revision_cid=revision_cid,
        slice_manifest_cid=slice_manifest_cid,
        manifest=manifest,
        require_current_reassignments=True,
    )
    if window is None:
        raise ExecutionPlanError("wave diff barrier lost its durable window")
    window_cid, window_record = window
    if (
        barrier.plan_root_cid != active.plan_root_cid
        or barrier.execution_plan_cid != revision.execution_plan_cid
        or barrier.capacity_snapshot_id != manifest.capacity_snapshot_id
        or barrier.wave_index != manifest.wave_index
        or barrier.expected_members != expected_members
        or barrier.dispositions != disposition_rows
        or barrier.terminal_missing != terminal_rows
        or barrier.window_cid != window_cid
        or barrier.deadline_at_ms != window_record["deadline_at_ms"]
    ):
        raise ExecutionPlanError("wave diff barrier authority is mixed")
    semantics = _plan_bound_wave_decision_semantics(
        expected_members=expected_members,
        disposition_records=disposition_records,
        lease_records=lease_records,
        terminal_records=terminal_records,
    )
    if (
        barrier.decision != semantics[0]
        or barrier.reason_codes != semantics[1]
        or barrier.missing_slice_ids != semantics[2]
        or dict(barrier.overlap_witness or {}) != semantics[3]
        or (
            barrier.decision == "missing"
            and not barrier.terminal_missing
            and barrier.decided_at_ms < window_record["deadline_at_ms"]
        )
    ):
        raise ExecutionPlanError("wave diff barrier decision is not reproducible")
    return barrier_cid, barrier


def _plan_bound_wave_expected_members(
    manifest: ConfiguredBoardExecutionSlices,
) -> tuple[dict[str, str], ...]:
    """Derive immutable one-child membership from one validated manifest."""

    return tuple(
        sorted(
            (
                {
                    "slice_id": item.slice_id,
                    "task_id": item.task_ids[0],
                    "task_cid": item.task_cids[0],
                }
                for item in manifest.nonempty
            ),
            key=lambda item: item["slice_id"],
        )
    )


def _plan_bound_wave_disposition_evidence_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    expected_members: Sequence[Mapping[str, str]],
) -> tuple[
    tuple[dict[str, str], ...],
    tuple[PlanBoundProposalDisposition, ...],
    dict[str, PlanBoundExecutionLease],
]:
    """Reload every current immutable lane result and its linked lease."""

    disposition_rows: list[dict[str, str]] = []
    disposition_records: list[PlanBoundProposalDisposition] = []
    lease_records: dict[str, PlanBoundExecutionLease] = {}
    for member in expected_members:
        observed = _load_plan_bound_proposal_disposition_locked(
            store,
            revision_cid=revision_cid,
            slice_id=member["slice_id"],
        )
        if observed is None:
            continue
        disposition_cid, disposition = observed
        disposition_rows.append(
            {
                "slice_id": member["slice_id"],
                "disposition_cid": disposition_cid,
            }
        )
        disposition_records.append(disposition)
        payload = _secure_store_cas(store, disposition.execution_lease_cid)
        disposition_lease = PlanBoundExecutionLease.from_dict(payload)
        if disposition_lease.to_dict() != payload:
            raise ExecutionPlanError(
                "wave diff barrier lease changed during typed decode"
            )
        lease_records[member["slice_id"]] = disposition_lease
    return (
        tuple(sorted(disposition_rows, key=lambda item: item["slice_id"])),
        tuple(sorted(disposition_records, key=lambda item: item.slice_id)),
        lease_records,
    )


def _plan_bound_wave_terminal_evidence_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    expected_members: Sequence[Mapping[str, str]],
) -> tuple[
    tuple[dict[str, str], ...],
    tuple[PlanBoundTerminalMissing, ...],
]:
    """Reload every current-owner process-terminal absence record."""

    rows: list[dict[str, str]] = []
    records: list[PlanBoundTerminalMissing] = []
    for member in expected_members:
        observed = _load_plan_bound_terminal_missing_locked(
            store,
            revision_cid=revision_cid,
            slice_id=member["slice_id"],
        )
        if observed is None:
            continue
        terminal_cid, terminal = observed
        rows.append(
            {
                "slice_id": member["slice_id"],
                "terminal_missing_cid": terminal_cid,
            }
        )
        records.append(terminal)
    return (
        tuple(sorted(rows, key=lambda item: item["slice_id"])),
        tuple(sorted(records, key=lambda item: item.slice_id)),
    )


def _plan_bound_wave_decision_semantics(
    *,
    expected_members: Sequence[Mapping[str, str]],
    disposition_records: Sequence[PlanBoundProposalDisposition],
    lease_records: Mapping[str, PlanBoundExecutionLease],
    terminal_records: Sequence[PlanBoundTerminalMissing] = (),
) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """Recompute the only valid decision from linked, typed lane evidence."""

    expected_slice_ids = {item["slice_id"] for item in expected_members}
    observed_slice_ids = {item.slice_id for item in disposition_records}
    missing = tuple(sorted(expected_slice_ids - observed_slice_ids))
    terminal_slice_ids = {item.slice_id for item in terminal_records}
    if not terminal_slice_ids.issubset(set(missing)):
        raise ExecutionPlanError(
            "wave terminal-missing evidence conflicts with a disposition"
        )
    if terminal_records:
        return (
            "missing",
            tuple(
                sorted(
                    {"wave_slice_terminal_without_disposition"}
                    | {
                        reason
                        for item in terminal_records
                        for reason in item.reason_codes
                    }
                )
            ),
            missing,
            {},
        )
    if missing:
        return (
            "missing",
            ("wave_disposition_deadline_expired",),
            missing,
            {},
        )

    rejected = tuple(
        item for item in disposition_records if item.outcome == "rejected"
    )
    if rejected:
        return (
            "rejected",
            tuple(
                sorted(
                    {"proposal_rejected"}
                    | {
                        reason
                        for item in rejected
                        for reason in item.reason_codes
                    }
                )
            ),
            (),
            {},
        )

    witness: dict[str, str] = {}
    ordered = sorted(disposition_records, key=lambda item: item.slice_id)
    for left_index, left in enumerate(ordered):
        left_lease = lease_records.get(left.slice_id)
        if left_lease is None:
            raise ExecutionPlanError("wave diff barrier lost left lease evidence")
        left_declared = tuple(
            left_lease.assignment_for(left.task_id, left.task_cid).get(
                "exclusive_paths", ()
            )
        )
        for right in ordered[left_index + 1 :]:
            right_lease = lease_records.get(right.slice_id)
            if right_lease is None:
                raise ExecutionPlanError(
                    "wave diff barrier lost right lease evidence"
                )
            right_declared = tuple(
                right_lease.assignment_for(right.task_id, right.task_cid).get(
                    "exclusive_paths", ()
                )
            )
            candidate_pairs = (
                (left.actual_changed_paths, right.actual_changed_paths),
                (left.actual_changed_paths, right_declared),
                (left_declared, right.actual_changed_paths),
            )
            for left_paths, right_paths in candidate_pairs:
                for left_path in left_paths:
                    for right_path in right_paths:
                        if _overlaps(left_path, right_path):
                            witness = {
                                "left_slice_id": left.slice_id,
                                "right_slice_id": right.slice_id,
                                "left_path": left_path,
                                "right_path": right_path,
                            }
                            break
                    if witness:
                        break
                if witness:
                    break
            if witness:
                break
        if witness:
            break
    if witness:
        return (
            "overlap",
            ("cross_lane_actual_diff_overlap",),
            (),
            witness,
        )
    return "released", (), (), {}


def _validate_plan_bound_wave_window_payload(
    payload: Mapping[str, Any],
    *,
    revision_cid: str,
    slice_manifest_cid: str,
) -> None:
    """Strictly validate one immutable barrier-window CAS generation."""

    if set(payload) != {
        "schema",
        "revision_cid",
        "slice_manifest_cid",
        "generation",
        "prior_window_cid",
        "started_at_ms",
        "deadline_at_ms",
        "timeout_ms",
        "reassignment_extensions",
    } or (
        payload.get("schema") != PLAN_BOUND_WAVE_DIFF_BARRIER_WINDOW_SCHEMA
        or payload.get("revision_cid") != revision_cid
        or payload.get("slice_manifest_cid") != slice_manifest_cid
    ):
        raise ExecutionPlanError("wave diff barrier window is malformed")
    for name in (
        "generation",
        "started_at_ms",
        "deadline_at_ms",
        "timeout_ms",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExecutionPlanError(
                "wave diff barrier window scalars are invalid"
            )
    if not 50 <= payload["timeout_ms"] <= 86_400_000:
        raise ExecutionPlanError("wave diff barrier window is inconsistent")
    if (
        not isinstance(payload.get("prior_window_cid"), str)
        or not isinstance(payload.get("reassignment_extensions"), list)
    ):
        raise ExecutionPlanError("wave diff barrier window links are invalid")
    rows: list[dict[str, Any]] = []
    for raw in payload["reassignment_extensions"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "slice_id",
            "reassignment_cid",
            "reassignment_generation",
            "extended_at_ms",
        }:
            raise ExecutionPlanError(
                "wave diff barrier reassignment extension is malformed"
            )
        if (
            not isinstance(raw.get("slice_id"), str)
            or not raw["slice_id"]
            or raw["slice_id"] != raw["slice_id"].strip()
            or not isinstance(raw.get("reassignment_cid"), str)
            or not raw["reassignment_cid"]
        ):
            raise ExecutionPlanError(
                "wave diff barrier reassignment extension text is invalid"
            )
        for name in ("reassignment_generation", "extended_at_ms"):
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ExecutionPlanError(
                    "wave diff barrier reassignment extension scalar is invalid"
                )
        rows.append(dict(raw))
    if rows != sorted(rows, key=lambda item: item["slice_id"]) or len(
        {item["slice_id"] for item in rows}
    ) != len(rows):
        raise ExecutionPlanError(
            "wave diff barrier reassignment extensions are ambiguous"
        )


def _load_plan_bound_wave_window_locked(
    store: PlanRevisionStore,
    *,
    revision_cid: str,
    slice_manifest_cid: str,
    manifest: ConfiguredBoardExecutionSlices,
    require_current_reassignments: bool,
) -> tuple[str, dict[str, Any]] | None:
    """Load one CAS-linked finite window and validate its whole history."""

    key = _plan_bound_wave_diff_barrier_window_key(
        revision_cid,
        slice_manifest_cid,
    )
    pointer = _secure_store_continuation(store, key)
    if pointer is None:
        return None
    if set(pointer) != {
        "phase",
        "operation",
        "revision_cid",
        "slice_manifest_cid",
        "window_cid",
        "generation",
    } or (
        pointer.get("phase") != "committed"
        or pointer.get("operation") != "plan_bound_wave_diff_barrier_window"
        or pointer.get("revision_cid") != revision_cid
        or pointer.get("slice_manifest_cid") != slice_manifest_cid
    ):
        raise ExecutionPlanError("wave diff barrier window pointer is malformed")
    generation = pointer.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ExecutionPlanError("wave diff barrier window generation is invalid")
    window_cid = _text(pointer.get("window_cid"), "window_cid")
    record = _secure_store_cas(store, window_cid)
    _validate_plan_bound_wave_window_payload(
        record,
        revision_cid=revision_cid,
        slice_manifest_cid=slice_manifest_cid,
    )
    if record["generation"] != generation:
        raise ExecutionPlanError("wave diff barrier window pointer is mixed")

    manifest_slice_ids = {item.slice_id for item in manifest.nonempty}
    transfer_budget = _plan_bound_wave_transfer_budget(manifest)
    cursor_cid = window_cid
    cursor = record
    seen: set[str] = set()
    while True:
        if cursor_cid in seen or len(seen) >= transfer_budget + 1:
            raise ExecutionPlanError("wave diff barrier window history cycles")
        seen.add(cursor_cid)
        _validate_plan_bound_wave_window_payload(
            cursor,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
        )
        rows = {
            item["slice_id"]: item
            for item in cursor["reassignment_extensions"]
        }
        if not set(rows).issubset(manifest_slice_ids):
            raise ExecutionPlanError(
                "wave diff barrier window extends a foreign slice"
            )
        if sum(
            int(row["reassignment_generation"])
            for row in rows.values()
        ) > transfer_budget:
            raise ExecutionPlanError(
                "wave diff barrier recovery exceeds its immutable budget"
            )
        for slice_id, row in rows.items():
            reassignment_payload = _secure_store_cas(
                store,
                row["reassignment_cid"],
            )
            reassignment = PlanSliceReassignment.from_dict(
                reassignment_payload
            )
            if (
                reassignment.revision_cid != revision_cid
                or reassignment.slice_manifest_cid != slice_manifest_cid
                or reassignment.slice_id != slice_id
                or reassignment.generation
                != row["reassignment_generation"]
            ):
                raise ExecutionPlanError(
                    "wave diff barrier extension evidence is mixed"
                )
        if cursor["generation"] == 1:
            if (
                cursor["prior_window_cid"]
                or cursor["reassignment_extensions"]
                or cursor["deadline_at_ms"] - cursor["started_at_ms"]
                != cursor["timeout_ms"]
            ):
                raise ExecutionPlanError(
                    "initial wave diff barrier window is inconsistent"
                )
            break
        prior_cid = cursor["prior_window_cid"]
        if not prior_cid:
            raise ExecutionPlanError("wave diff barrier window history is truncated")
        prior = _secure_store_cas(store, prior_cid)
        _validate_plan_bound_wave_window_payload(
            prior,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
        )
        if (
            cursor["generation"] != prior["generation"] + 1
            or cursor["started_at_ms"] != prior["started_at_ms"]
            or cursor["timeout_ms"] != prior["timeout_ms"]
        ):
            raise ExecutionPlanError("wave diff barrier window history is mixed")
        prior_rows = {
            item["slice_id"]: item
            for item in prior["reassignment_extensions"]
        }
        if not set(prior_rows).issubset(rows):
            raise ExecutionPlanError("wave diff barrier extensions regressed")
        changed = [
            row
            for slice_id, row in rows.items()
            if prior_rows.get(slice_id) != row
        ]
        if not changed or any(
            slice_id in prior_rows
            and rows[slice_id]["reassignment_generation"]
            <= prior_rows[slice_id]["reassignment_generation"]
            for slice_id in rows
            if prior_rows.get(slice_id) != rows[slice_id]
        ):
            raise ExecutionPlanError(
                "wave diff barrier extension history did not advance"
            )
        expected_deadline = max(
            prior["deadline_at_ms"],
            *(item["extended_at_ms"] + cursor["timeout_ms"] for item in changed),
        )
        if cursor["deadline_at_ms"] != expected_deadline:
            raise ExecutionPlanError(
                "wave diff barrier extension deadline is inconsistent"
            )
        cursor_cid, cursor = prior_cid, prior

    if require_current_reassignments:
        adapter = ProductionParallelPlanAdapter(store)
        current_rows: dict[str, tuple[str, PlanSliceReassignment]] = {}
        for item in manifest.nonempty:
            current = adapter._load_slice_reassignment_locked(  # noqa: SLF001
                revision_cid=revision_cid,
                slice_id=item.slice_id,
            )
            if current is not None:
                current_rows[item.slice_id] = current
        extension_rows = {
            item["slice_id"]: item
            for item in record["reassignment_extensions"]
        }
        if set(extension_rows) != set(current_rows):
            raise ExecutionPlanError(
                "wave diff barrier window lost current reassignment evidence"
            )
        for slice_id, (reassignment_cid, reassignment) in current_rows.items():
            row = extension_rows[slice_id]
            if (
                row["reassignment_cid"] != reassignment_cid
                or row["reassignment_generation"] != reassignment.generation
            ):
                raise ExecutionPlanError(
                    "wave diff barrier window reassignment evidence is stale"
                )
    return window_cid, record


def _publish_plan_bound_wave_window_locked(
    store: PlanRevisionStore,
    *,
    record: Mapping[str, Any],
    expected_current_cid: str,
    manifest: ConfiguredBoardExecutionSlices,
) -> tuple[str, dict[str, Any]]:
    """Publish one immutable window generation through its guarded pointer."""

    current = _load_plan_bound_wave_window_locked(
        store,
        revision_cid=str(record["revision_cid"]),
        slice_manifest_cid=str(record["slice_manifest_cid"]),
        manifest=manifest,
        require_current_reassignments=False,
    )
    if (current[0] if current is not None else "") != expected_current_cid:
        raise ExecutionPlanError("wave diff barrier window CAS lost")
    window_cid = store.put_cas(dict(record))
    if _secure_store_cas(store, window_cid) != dict(record):
        raise ExecutionPlanError("wave diff barrier window failed CAS round trip")
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_wave_diff_barrier_window",
        "revision_cid": record["revision_cid"],
        "slice_manifest_cid": record["slice_manifest_cid"],
        "window_cid": window_cid,
        "generation": record["generation"],
    }
    key = _plan_bound_wave_diff_barrier_window_key(
        str(record["revision_cid"]),
        str(record["slice_manifest_cid"]),
    )
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError("wave diff barrier window pointer did not persist")
    observed = _load_plan_bound_wave_window_locked(
        store,
        revision_cid=str(record["revision_cid"]),
        slice_manifest_cid=str(record["slice_manifest_cid"]),
        manifest=manifest,
        require_current_reassignments=True,
    )
    if observed != (window_cid, dict(record)):
        raise ExecutionPlanError("wave diff barrier window publication was not exact")
    return observed


def _publish_plan_bound_wave_diff_barrier_locked(
    store: PlanRevisionStore,
    barrier: PlanBoundWaveDiffBarrier,
) -> tuple[str, PlanBoundWaveDiffBarrier]:
    existing = _load_plan_bound_wave_diff_barrier_locked(
        store,
        revision_cid=barrier.revision_cid,
        slice_manifest_cid=barrier.slice_manifest_cid,
    )
    if existing is not None:
        if existing[1] != barrier:
            # The first complete decision is immutable.  A contender must
            # consume it rather than replace it with a later observation.
            return existing
        return existing
    barrier_cid = store.put_cas(barrier.to_dict())
    if _secure_store_cas(store, barrier_cid) != barrier.to_dict():
        raise ExecutionPlanError("wave diff barrier failed CAS round trip")
    pointer = {
        "phase": "committed",
        "operation": "plan_bound_wave_diff_barrier",
        "revision_cid": barrier.revision_cid,
        "slice_manifest_cid": barrier.slice_manifest_cid,
        "barrier_cid": barrier_cid,
        "decision": barrier.decision,
    }
    key = plan_bound_wave_diff_barrier_key(
        barrier.revision_cid,
        barrier.slice_manifest_cid,
    )
    store.put_continuation(key, pointer)
    if _secure_store_continuation(store, key) != pointer:
        raise ExecutionPlanError("wave diff barrier pointer did not persist")
    observed = _load_plan_bound_wave_diff_barrier_locked(
        store,
        revision_cid=barrier.revision_cid,
        slice_manifest_cid=barrier.slice_manifest_cid,
    )
    if observed != (barrier_cid, barrier):
        raise ExecutionPlanError("wave diff barrier publication was not exact")
    return observed


@dataclass(frozen=True)
class ParallelismDecisionReceipt:
    """Durable publication result consumed by the configured launcher."""

    binding: ActivePlanBinding
    slice_manifest: ConfiguredBoardExecutionSlices
    slice_manifest_cid: str
    apply_receipt_cid: str
    schema: str = PARALLELISM_DECISION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": self.binding.to_dict(),
            "slice_manifest": self.slice_manifest.to_dict(),
            "slice_manifest_cid": self.slice_manifest_cid,
            "apply_receipt_cid": self.apply_receipt_cid,
        }


class ProductionParallelPlanAdapter:
    """Thin adapter over the canonical compiler and ``PlanRevisionStore``.

    Claims, stealing, leases/fences, actual-diff checks, validation, worktrees,
    and merge serialization remain owned by their existing production
    services.  In particular this adapter has no private mutable ledger.
    """

    def __init__(
        self,
        plan_revision_store: PlanRevisionStore,
        *,
        compiler: ParallelPlanCompiler | None = None,
    ) -> None:
        if not isinstance(plan_revision_store, PlanRevisionStore):
            raise TypeError("plan_revision_store must be PlanRevisionStore")
        self.plan_revision_store = plan_revision_store
        self.compiler = compiler or ParallelPlanCompiler()

    def compile_wave(
        self,
        *,
        board_namespace: str,
        plan_root_cid: str,
        tasks: Sequence[Mapping[str, Any]],
        budget: InvocationBudget,
        repository_snapshot: Mapping[str, Any],
        capacity_snapshot: Mapping[str, Any],
        provider_snapshots: Sequence[Mapping[str, Any]] = (),
        completed_task_ids: Sequence[str] = (),
        protected_paths: Sequence[str] = (),
        submodule_paths: Sequence[str] = (),
        post_merge_validation: Sequence[str] = (),
        source_head: str,
        task_source_revision: str,
        configuration_root: str,
        current_time_ms: int | None = None,
    ) -> tuple[ParallelExecutionPlan, ConfiguredBoardExecutionSlices]:
        if not isinstance(budget, InvocationBudget):
            raise TypeError("budget must be the canonical InvocationBudget")
        if not tasks or len(tasks) > MAX_TASKS:
            raise ExecutionPlanError("task population must be nonempty and bounded")
        compiler_tasks: list[dict[str, Any]] = []
        cid_by_id: dict[str, str] = {}
        for raw in tasks:
            record = dict(raw)
            task_id = _text(record.get("task_id"), "task_id")
            task_cid = _text(
                record.get("canonical_task_cid") or record.get("task_cid"),
                "canonical_task_cid",
            )
            if task_id in cid_by_id or task_cid in cid_by_id.values():
                raise ExecutionPlanError("task ID/CID population contains duplicates")
            cid_by_id[task_id] = task_cid
            # ParallelPlanCompiler prefers task_cid over task_id.  Remove the
            # CID aliases so its assignment graph remains display-ID keyed;
            # the exact canonical CID stays paired in the slice manifest.
            record.pop("task_cid", None)
            record.pop("canonical_task_cid", None)
            record.pop("canonical_task_key", None)
            record["task_id"] = task_id
            # Markdown ``Outputs`` are repository mutation envelopes, not
            # unique logical-artifact producer IDs.  Feed them to the
            # compiler's path-conflict relation (alongside Predicted files)
            # so exact and parent/child overlap serialize instead of being
            # rejected as a duplicate abstract output producer.
            declared_paths = _paths(
                record.get("outputs")
                or record.get("output_paths")
                or record.get("expected_outputs"),
                "outputs",
            )
            predicted_paths = _paths(
                record.get("predicted_files")
                or record.get("predicted_paths")
                or record.get("files"),
                "predicted_files",
            )
            if any(
                character in path
                for path in (*declared_paths, *predicted_paths)
                for character in "*?["
            ):
                # Proposal admission intentionally supports glob envelopes,
                # while the canonical parallel compiler uses exact
                # repository-path prefix overlap.  Until a shared glob
                # algebra owns both checks, a glob-bearing task cannot be
                # proved conflict-disjoint and must not enter a parallel wave.
                raise ExecutionPlanError(
                    "glob-like mutation paths require serialized replan"
                )
            if declared_paths or predicted_paths:
                record["predicted_files"] = list(
                    sorted(set(declared_paths) | set(predicted_paths))
                )
            for field_name in ("outputs", "output_paths", "expected_outputs"):
                record.pop(field_name, None)
            compiler_tasks.append(record)
        now_ms = int(time.time() * 1000) if current_time_ms is None else int(current_time_ms)
        plan = self.compiler.compile(
            tasks=tuple(compiler_tasks),
            requested_width=budget.max_lanes,
            repository_snapshot=repository_snapshot,
            capacity_snapshot=capacity_snapshot,
            provider_snapshots=provider_snapshots,
            budget={"max_ready_width": budget.max_lanes},
            current_time_ms=now_ms,
            completed_task_ids=tuple(completed_task_ids),
            protected_paths=tuple(protected_paths),
            submodule_paths=tuple(submodule_paths),
            post_merge_validation=tuple(post_merge_validation),
        )
        if not plan.admitted:
            reasons = ", ".join(issue.code.value for issue in plan.issues)
            raise ExecutionPlanError(f"parallel execution plan rejected: {reasons}")
        first_wave = plan.execution_waves[0].task_ids if plan.execution_waves else ()
        unknown = tuple(task_id for task_id in first_wave if task_id not in cid_by_id)
        if unknown:
            raise ExecutionPlanError("compiler selected tasks outside canonical population")
        manifest = ConfiguredBoardExecutionSlices(
            board_namespace=board_namespace,
            plan_root_cid=plan_root_cid,
            compiler_plan_id=plan.plan_id,
            capacity_snapshot_id=plan.capacity_snapshot_id,
            repository_tree_id=plan.repository_tree_id,
            source_head=source_head,
            task_source_revision=task_source_revision,
            configuration_root=configuration_root,
            slices=tuple(
                ConfiguredBoardExecutionSlice(
                    lane_index=index,
                    lane_id=f"lane-{index}",
                    task_ids=(task_id,),
                    task_cids=(cid_by_id[task_id],),
                    plan_root_cid=plan_root_cid,
                    compiler_plan_id=plan.plan_id,
                    capacity_snapshot_id=plan.capacity_snapshot_id,
                    repository_tree_id=plan.repository_tree_id,
                )
                for index, task_id in enumerate(first_wave)
            ),
        )
        return plan, manifest

    def publish_wave(
        self,
        *,
        plan: ParallelExecutionPlan,
        slice_manifest: ConfiguredBoardExecutionSlices,
        revision_factory: Callable[[str, str], PlanRevision],
        observed_roots: PlanAuthorityRoots,
        idempotency_key: str,
        delta: PlanDelta | None = None,
        expected_active_plan_root: str = "",
        expected_active_revision_cid: str = "",
        base_event_cursor: str = "",
        fencing_token: int = 1,
        lease_id: str = "",
    ) -> ParallelismDecisionReceipt:
        slice_manifest_cid = self.plan_revision_store.put_cas(slice_manifest.to_dict())
        plan_payload = plan.to_dict(include_replay_request=True)
        plan_payload["configured_board_execution_slices_cid"] = slice_manifest_cid
        execution_plan_cid = self.plan_revision_store.put_cas(plan_payload)
        revision = revision_factory(execution_plan_cid, slice_manifest_cid)
        if not isinstance(revision, PlanRevision):
            raise TypeError("revision_factory must return PlanRevision")
        if revision.execution_plan_cid != execution_plan_cid:
            raise ExecutionPlanError("revision does not own the stored execution plan")
        if revision.materialization_transaction_cid != slice_manifest_cid:
            raise ExecutionPlanError("revision does not own the stored slice manifest")
        receipt = self.plan_revision_store.apply(
            PlanRevisionApplyRequest(
                revision=revision,
                observed_roots=observed_roots,
                idempotency_key=idempotency_key,
                delta=delta,
                repository_tree_id=plan.repository_tree_id,
                fencing_token=fencing_token,
                lease_id=lease_id,
                base_event_cursor=base_event_cursor,
                expected_active_plan_root=expected_active_plan_root,
                expected_active_revision_cid=expected_active_revision_cid,
                records={
                    "parallel_execution_plan": plan_payload,
                    "configured_board_execution_slices": slice_manifest.to_dict(),
                },
            )
        )
        binding = load_plan_revision_store_binding(self.plan_revision_store)
        if (
            binding.execution_plan_cid != execution_plan_cid
            or binding.plan_root_cid != slice_manifest.plan_root_cid
            or binding.capacity_snapshot_id != slice_manifest.capacity_snapshot_id
            or binding.repository_tree_id != slice_manifest.repository_tree_id
        ):
            raise ExecutionPlanError("published active binding is partial or mixed")
        return ParallelismDecisionReceipt(
            binding=binding,
            slice_manifest=slice_manifest,
            slice_manifest_cid=slice_manifest_cid,
            apply_receipt_cid=receipt.receipt_cid,
        )

    def load_execution_lease(
        self,
        *,
        revision_cid: str,
        slice_id: str,
        lane_id: str,
    ) -> tuple[str, PlanBoundExecutionLease] | None:
        """Load one exact real-effect bridge through the canonical store lock."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _load_plan_bound_execution_lease_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_id=slice_id,
                    lane_id=lane_id,
                )

    def authorize_recovery_launch(
        self,
        *,
        revision_cid: str,
        slice_id: str,
        lane_id: str,
        source_head: str,
        source_tree: str,
        repository_head: str,
        repository_tree: str,
        runtime_artifacts: tuple[Mapping[str, Any], ...],
        launch_artifact_paths: tuple[str, ...],
    ) -> tuple[str, PlanBoundRecoveryLaunch]:
        """Publish one recovery-only sealed-gate decision."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _publish_plan_bound_recovery_launch_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_id=slice_id,
                    lane_id=lane_id,
                    source_head=source_head,
                    source_tree=source_tree,
                    repository_head=repository_head,
                    repository_tree=repository_tree,
                    runtime_artifacts=runtime_artifacts,
                    launch_artifact_paths=launch_artifact_paths,
                )

    def load_recovery_launch(
        self,
        *,
        revision_cid: str,
        slice_id: str,
        lane_id: str,
        authorization_cid: str,
    ) -> PlanBoundRecoveryLaunch:
        """Revalidate a recovery-only sealed-gate decision."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _load_plan_bound_recovery_launch_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_id=slice_id,
                    lane_id=lane_id,
                    authorization_cid=authorization_cid,
                )

    def recovery_workspace_paths(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
    ) -> tuple[str, ...]:
        """Return exact current wave workspaces under the canonical store guard."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                _active, _revision, manifest = _active_plan_bound_manifest_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_manifest_cid=slice_manifest_cid,
                )
                workspaces: list[str] = []
                for execution_slice in manifest.nonempty:
                    reassignment = self._load_slice_reassignment_locked(
                        revision_cid=revision_cid,
                        slice_id=execution_slice.slice_id,
                    )
                    lane_id = (
                        execution_slice.lane_id
                        if reassignment is None
                        else reassignment[1].recipient_lane_id
                    )
                    execution = _load_plan_bound_execution_lease_locked(
                        self.plan_revision_store,
                        revision_cid=revision_cid,
                        slice_id=execution_slice.slice_id,
                        lane_id=lane_id,
                    )
                    if execution is not None and execution[1].workspace_path:
                        workspaces.append(execution[1].workspace_path)
                if len(workspaces) != len(set(workspaces)):
                    raise ExecutionPlanError(
                        "plan-bound recovery workspace paths are ambiguous"
                    )
                return tuple(sorted(workspaces))

    def recovery_runtime_bindings(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return exact manifest/lease names used to admit runtime artifacts."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                _active, _revision, manifest = _active_plan_bound_manifest_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_manifest_cid=slice_manifest_cid,
                )
                result: list[dict[str, Any]] = []
                for execution_slice in manifest.nonempty:
                    reassignment = self._load_slice_reassignment_locked(
                        revision_cid=revision_cid,
                        slice_id=execution_slice.slice_id,
                    )
                    lane_id = (
                        execution_slice.lane_id
                        if reassignment is None
                        else reassignment[1].recipient_lane_id
                    )
                    execution = _load_plan_bound_execution_lease_locked(
                        self.plan_revision_store,
                        revision_cid=revision_cid,
                        slice_id=execution_slice.slice_id,
                        lane_id=lane_id,
                    )
                    request_id = ""
                    dedupe_key = ""
                    workspace_path = ""
                    active_task_id = ""
                    attempt = 0
                    if execution is not None:
                        lease = execution[1]
                        request_id = lease.merge_request_id
                        workspace_path = lease.workspace_path
                        active_task_id = lease.active_task_id
                        if lease.proposal_handoff_cid:
                            _validate_plan_bound_proposal_handoff_locked(
                                self.plan_revision_store,
                                lease,
                            )
                            handoff = _secure_store_cas(
                                self.plan_revision_store,
                                lease.proposal_handoff_cid,
                            )
                            raw_attempt = handoff.get("attempt")
                            if (
                                isinstance(raw_attempt, bool)
                                or not isinstance(raw_attempt, int)
                                or raw_attempt < 1
                            ):
                                raise ExecutionPlanError(
                                    "proposal handoff has no exact attempt"
                                )
                            attempt = raw_attempt
                        if lease.merge_queue_receipt_cid:
                            receipt = _secure_store_cas(
                                self.plan_revision_store,
                                lease.merge_queue_receipt_cid,
                            )
                            raw_dedupe = receipt.get("dedupe_key")
                            if not isinstance(raw_dedupe, str):
                                raise ExecutionPlanError(
                                    "merge receipt has no exact dedupe key"
                                )
                            dedupe_key = raw_dedupe
                    result.append(
                        {
                            "slice_id": execution_slice.slice_id,
                            "lane_index": execution_slice.lane_index,
                            "lane_id": lane_id,
                            "task_ids": list(execution_slice.task_ids),
                            "task_cids": list(execution_slice.task_cids),
                            "active_task_id": active_task_id,
                            "attempt": attempt,
                            "workspace_path": workspace_path,
                            "merge_request_id": request_id,
                            "merge_dedupe_key": dedupe_key,
                        }
                    )
                return tuple(result)

    @staticmethod
    def _wave_diff_barrier_window_locked(
        store: PlanRevisionStore,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
        manifest: ConfiguredBoardExecutionSlices,
        timeout_ms: int,
        now_ms: int,
    ) -> tuple[str, int, int, int]:
        """Create or extend the finite window from verified owner transfers."""

        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 50 <= timeout_ms <= 86_400_000
            or isinstance(now_ms, bool)
            or not isinstance(now_ms, int)
            or now_ms < 1
        ):
            raise ExecutionPlanError("wave diff barrier timing is invalid")
        current = _load_plan_bound_wave_window_locked(
            store,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
            manifest=manifest,
            require_current_reassignments=False,
        )
        if current is None:
            record = {
                "schema": PLAN_BOUND_WAVE_DIFF_BARRIER_WINDOW_SCHEMA,
                "revision_cid": revision_cid,
                "slice_manifest_cid": slice_manifest_cid,
                "generation": 1,
                "prior_window_cid": "",
                "started_at_ms": now_ms,
                "deadline_at_ms": now_ms + timeout_ms,
                "timeout_ms": timeout_ms,
                "reassignment_extensions": [],
            }
            current = _publish_plan_bound_wave_window_locked(
                store,
                record=record,
                expected_current_cid="",
                manifest=manifest,
            )
        current_cid, current_record = current
        if current_record["timeout_ms"] != timeout_ms:
            raise ExecutionPlanError(
                "wave diff barrier timeout differs across contenders"
            )
        adapter = ProductionParallelPlanAdapter(store)
        extension_rows = {
            item["slice_id"]: dict(item)
            for item in current_record["reassignment_extensions"]
        }
        changed = False
        for item in manifest.nonempty:
            reassignment = adapter._load_slice_reassignment_locked(  # noqa: SLF001
                revision_cid=revision_cid,
                slice_id=item.slice_id,
            )
            if reassignment is None:
                if item.slice_id in extension_rows:
                    raise ExecutionPlanError(
                        "wave diff barrier reassignment extension regressed"
                    )
                continue
            reassignment_cid, reassignment_record = reassignment
            prior = extension_rows.get(item.slice_id)
            if prior is not None and (
                prior["reassignment_cid"] == reassignment_cid
                and prior["reassignment_generation"]
                == reassignment_record.generation
            ):
                continue
            if prior is not None and (
                reassignment_record.generation
                <= prior["reassignment_generation"]
            ):
                raise ExecutionPlanError(
                    "wave diff barrier reassignment extension did not advance"
                )
            extension_rows[item.slice_id] = {
                "slice_id": item.slice_id,
                "reassignment_cid": reassignment_cid,
                "reassignment_generation": reassignment_record.generation,
                "extended_at_ms": max(
                    now_ms,
                    int(current_record["started_at_ms"]),
                ),
            }
            changed = True
        if changed:
            next_rows = sorted(
                extension_rows.values(),
                key=lambda item: item["slice_id"],
            )
            changed_rows = [
                row
                for row in next_rows
                if row not in current_record["reassignment_extensions"]
            ]
            record = {
                **dict(current_record),
                "generation": int(current_record["generation"]) + 1,
                "prior_window_cid": current_cid,
                "deadline_at_ms": max(
                    int(current_record["deadline_at_ms"]),
                    *(
                        int(row["extended_at_ms"]) + timeout_ms
                        for row in changed_rows
                    ),
                ),
                "reassignment_extensions": next_rows,
            }
            current_cid, current_record = (
                _publish_plan_bound_wave_window_locked(
                    store,
                    record=record,
                    expected_current_cid=current_cid,
                    manifest=manifest,
                )
            )
        else:
            _load_plan_bound_wave_window_locked(
                store,
                revision_cid=revision_cid,
                slice_manifest_cid=slice_manifest_cid,
                manifest=manifest,
                require_current_reassignments=True,
            )
        extension_count = sum(
            int(item["reassignment_generation"])
            for item in current_record["reassignment_extensions"]
        )
        if extension_count > _plan_bound_wave_transfer_budget(manifest):
            raise ExecutionPlanError(
                "wave diff barrier recovery generations exceed immutable width"
            )
        return (
            current_cid,
            int(current_record["started_at_ms"]),
            int(current_record["deadline_at_ms"]),
            extension_count,
        )

    def _evaluate_wave_diff_barrier_locked(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
        timeout_ms: int,
        now_ms: int,
    ) -> tuple[str, PlanBoundWaveDiffBarrier] | None:
        """Evaluate one complete wave without sleeping under authority locks."""

        existing = _load_plan_bound_wave_diff_barrier_locked(
            self.plan_revision_store,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
        )
        if existing is not None:
            return existing
        active, revision, manifest = _active_plan_bound_manifest_locked(
            self.plan_revision_store,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
        )
        (
            window_cid,
            _started_at_ms,
            deadline_at_ms,
            _extension_count,
        ) = self._wave_diff_barrier_window_locked(
            self.plan_revision_store,
            revision_cid=revision_cid,
            slice_manifest_cid=slice_manifest_cid,
            manifest=manifest,
            timeout_ms=timeout_ms,
            now_ms=now_ms,
        )
        expected_members = _plan_bound_wave_expected_members(manifest)
        disposition_rows, disposition_records, lease_records = (
            _plan_bound_wave_disposition_evidence_locked(
                self.plan_revision_store,
                revision_cid=revision_cid,
                expected_members=expected_members,
            )
        )
        terminal_rows, terminal_records = (
            _plan_bound_wave_terminal_evidence_locked(
                self.plan_revision_store,
                revision_cid=revision_cid,
                expected_members=expected_members,
            )
        )
        semantics = _plan_bound_wave_decision_semantics(
            expected_members=expected_members,
            disposition_records=disposition_records,
            lease_records=lease_records,
            terminal_records=terminal_records,
        )
        missing = semantics[2]
        if missing and not terminal_records and now_ms < deadline_at_ms:
            return None

        barrier = PlanBoundWaveDiffBarrier(
            revision_cid=revision_cid,
            plan_root_cid=active.plan_root_cid,
            execution_plan_cid=revision.execution_plan_cid,
            capacity_snapshot_id=manifest.capacity_snapshot_id,
            slice_manifest_cid=slice_manifest_cid,
            window_cid=window_cid,
            wave_index=manifest.wave_index,
            expected_members=expected_members,
            dispositions=tuple(disposition_rows),
            terminal_missing=terminal_rows,
            decision=semantics[0],
            reason_codes=semantics[1],
            missing_slice_ids=missing,
            deadline_at_ms=deadline_at_ms,
            decided_at_ms=now_ms,
            overlap_witness=semantics[3],
        )
        return _publish_plan_bound_wave_diff_barrier_locked(
            self.plan_revision_store,
            barrier,
        )

    def publish_proposal_disposition(
        self,
        disposition: PlanBoundProposalDisposition,
    ) -> str:
        """Publish one exact proposal result through canonical store authority."""

        if not isinstance(disposition, PlanBoundProposalDisposition):
            raise TypeError("disposition must be PlanBoundProposalDisposition")
        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _publish_plan_bound_proposal_disposition_locked(
                    self.plan_revision_store,
                    disposition,
                )

    def publish_terminal_missing(
        self,
        terminal: PlanBoundTerminalMissing,
    ) -> str:
        """Publish one process-fenced missing member through store authority."""

        if not isinstance(terminal, PlanBoundTerminalMissing):
            raise TypeError("terminal must be PlanBoundTerminalMissing")
        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _publish_plan_bound_terminal_missing_locked(
                    self.plan_revision_store,
                    terminal,
                )

    def load_terminal_missing(
        self,
        *,
        revision_cid: str,
        slice_id: str,
    ) -> tuple[str, PlanBoundTerminalMissing] | None:
        """Load one strict terminal-missing record through store authority."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _load_plan_bound_terminal_missing_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_id=slice_id,
                )

    def await_wave_diff_barrier(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
        timeout_ms: int,
    ) -> tuple[str, PlanBoundWaveDiffBarrier]:
        """Wait outside locks for a finite whole-wave release-or-fence record."""

        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 50 <= timeout_ms <= 86_400_000
        ):
            raise ExecutionPlanError("wave diff barrier timeout is invalid")
        started_wall_ms = int(time.time() * 1000)
        started_monotonic = time.monotonic()
        local_deadline_monotonic = started_monotonic + timeout_ms / 1000.0
        last_wall_ms = started_wall_ms
        durable_deadline_at_ms = 0
        while True:
            monotonic_now = time.monotonic()
            elapsed_ms = max(
                0,
                int((monotonic_now - started_monotonic) * 1000),
            )
            observed_wall_ms = int(time.time() * 1000)
            last_wall_ms = max(last_wall_ms, observed_wall_ms)
            now_ms = max(last_wall_ms, started_wall_ms + elapsed_ms)
            if (
                durable_deadline_at_ms
                and monotonic_now >= local_deadline_monotonic
            ):
                # Wall-clock rollback cannot turn one finite durable window
                # into an unbounded wait.  Verified transfer generations extend
                # the local cap below; no other observation can do so.
                now_ms = max(now_ms, durable_deadline_at_ms)
            with self.plan_revision_store._thread_lock:  # noqa: SLF001
                with self.plan_revision_store._guard():  # noqa: SLF001
                    observed = self._evaluate_wave_diff_barrier_locked(
                        revision_cid=revision_cid,
                        slice_manifest_cid=slice_manifest_cid,
                        timeout_ms=timeout_ms,
                        now_ms=now_ms,
                    )
                    if observed is None:
                        _active, _revision, manifest = (
                            _active_plan_bound_manifest_locked(
                                self.plan_revision_store,
                                revision_cid=revision_cid,
                                slice_manifest_cid=slice_manifest_cid,
                            )
                        )
                        window = _load_plan_bound_wave_window_locked(
                            self.plan_revision_store,
                            revision_cid=revision_cid,
                            slice_manifest_cid=slice_manifest_cid,
                            manifest=manifest,
                            require_current_reassignments=True,
                        )
                        if window is None:
                            raise ExecutionPlanError(
                                "wave diff barrier lost its durable window"
                            )
                        window_record = window[1]
                        durable_deadline_at_ms = int(
                            window_record["deadline_at_ms"]
                        )
                        extension_count = sum(
                            int(item["reassignment_generation"])
                            for item in window_record[
                                "reassignment_extensions"
                            ]
                        )
                        if extension_count > _plan_bound_wave_transfer_budget(
                            manifest
                        ):
                            raise ExecutionPlanError(
                                "wave recovery exceeds immutable width"
                            )
                        hard_local_cap = started_monotonic + (
                            timeout_ms * (1 + extension_count) / 1000.0
                        )
                        remaining_seconds = max(
                            0.0,
                            (durable_deadline_at_ms - now_ms) / 1000.0,
                        )
                        local_deadline_monotonic = min(
                            hard_local_cap,
                            max(
                                local_deadline_monotonic,
                                monotonic_now + remaining_seconds,
                            ),
                        )
            if observed is not None:
                return observed
            time.sleep(0.05)

    def load_wave_diff_barrier(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
    ) -> tuple[str, PlanBoundWaveDiffBarrier] | None:
        """Load a terminal whole-wave decision through strict store readers."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return _load_plan_bound_wave_diff_barrier_locked(
                    self.plan_revision_store,
                    revision_cid=revision_cid,
                    slice_manifest_cid=slice_manifest_cid,
                )

    @staticmethod
    def _reassignment_key(revision_cid: str, slice_id: str) -> str:
        return f"plan-slice-reassignment:{revision_cid}:{slice_id}"

    def load_slice_reassignment(
        self,
        *,
        revision_cid: str,
        slice_id: str,
    ) -> tuple[str, PlanSliceReassignment] | None:
        """Load the canonical current owner transfer for one immutable slice."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return self._load_slice_reassignment_locked(
                    revision_cid=revision_cid,
                    slice_id=slice_id,
                )

    def _load_slice_reassignment_locked(
        self,
        *,
        revision_cid: str,
        slice_id: str,
    ) -> tuple[str, PlanSliceReassignment] | None:
        """Load one transfer while the canonical cross-process lock is held."""

        revision_cid = _text(revision_cid, "revision_cid")
        slice_id = _text(slice_id, "slice_id")
        continuation = _secure_store_continuation(
            self.plan_revision_store,
            self._reassignment_key(revision_cid, slice_id)
        )
        if continuation is None:
            return None
        if set(continuation) != {
            "phase", "operation", "revision_cid", "plan_root_cid",
            "slice_id", "reassignment_cid", "generation",
        } or (
            continuation.get("phase") != "committed"
            or continuation.get("operation") != "slice_reassignment"
            or continuation.get("revision_cid") != revision_cid
            or continuation.get("slice_id") != slice_id
        ):
            raise ExecutionPlanError("slice reassignment pointer is malformed")
        pointer_generation = continuation.get("generation")
        if (
            isinstance(pointer_generation, bool)
            or not isinstance(pointer_generation, int)
            or pointer_generation < 1
        ):
            raise ExecutionPlanError("slice reassignment pointer generation is invalid")
        reassignment_cid = _text(
            continuation.get("reassignment_cid"), "reassignment_cid"
        )
        active = _secure_store_active(self.plan_revision_store)
        if active is None or active.revision_cid != revision_cid:
            raise ExecutionPlanError("slice reassignment requires the active revision")
        revision_payload = _secure_store_cas(
            self.plan_revision_store,
            revision_cid,
        )
        revision = PlanRevision.from_dict(revision_payload)
        if revision.to_dict() != revision_payload:
            raise ExecutionPlanError("plan revision changed during typed decode")
        slice_manifest_cid = revision.materialization_transaction_cid
        manifest_payload = _secure_store_cas(
            self.plan_revision_store,
            slice_manifest_cid,
        )
        manifest = ConfiguredBoardExecutionSlices.from_dict(manifest_payload)
        matches = tuple(item for item in manifest.slices if item.slice_id == slice_id)
        if len(matches) != 1:
            raise ExecutionPlanError("slice reassignment immutable slice is absent")
        execution_slice = matches[0]

        def cas_payload(cid: str, label: str) -> dict[str, Any]:
            try:
                return _secure_store_cas(self.plan_revision_store, cid)
            except ExecutionPlanError as exc:
                raise ExecutionPlanError(f"{label} CAS identity is invalid") from exc

        record = PlanSliceReassignment.from_dict(
            cas_payload(reassignment_cid, "slice reassignment")
        )
        if (
            record.revision_cid != revision_cid
            or record.plan_root_cid != active.plan_root_cid
            or continuation.get("plan_root_cid") != active.plan_root_cid
            or record.slice_manifest_cid != slice_manifest_cid
            or record.slice_id != slice_id
            or record.generation != pointer_generation
            or record.task_ids != execution_slice.task_ids
            or record.task_cids != execution_slice.task_cids
        ):
            raise ExecutionPlanError("slice reassignment pointer is mixed")

        chain: list[tuple[str, PlanSliceReassignment]] = []
        current_cid = reassignment_cid
        current = record
        while True:
            chain.append((current_cid, current))
            if len(chain) > _plan_bound_wave_transfer_budget(manifest):
                raise ExecutionPlanError("slice reassignment chain exceeds its bound")
            if current.generation == 1:
                if current.prior_reassignment_cid:
                    raise ExecutionPlanError("first slice reassignment has a prior CID")
                if current.donor_lane_id != execution_slice.lane_id:
                    raise ExecutionPlanError("first reassignment donor is not slice owner")
                break
            if not current.prior_reassignment_cid:
                raise ExecutionPlanError("slice reassignment chain is truncated")
            prior_cid = current.prior_reassignment_cid
            if any(cid == prior_cid for cid, _item in chain):
                raise ExecutionPlanError("slice reassignment chain contains a cycle")
            prior = PlanSliceReassignment.from_dict(
                cas_payload(prior_cid, "prior slice reassignment")
            )
            if (
                prior.revision_cid != revision_cid
                or prior.plan_root_cid != active.plan_root_cid
                or prior.slice_manifest_cid != slice_manifest_cid
                or prior.slice_id != slice_id
                or prior.task_ids != execution_slice.task_ids
                or prior.task_cids != execution_slice.task_cids
                or prior.generation + 1 != current.generation
                or prior.recipient_lane_id != current.donor_lane_id
            ):
                raise ExecutionPlanError("slice reassignment prior chain is mixed")
            current_cid, current = prior_cid, prior

        owner_history = [execution_slice.lane_id]
        owner_history.extend(
            item.recipient_lane_id
            for _item_cid, item in reversed(chain)
        )
        if len(owner_history) != len(set(owner_history)):
            raise ExecutionPlanError(
                "slice reassignment revisits a prior lane owner"
            )

        from ..control.lifecycle_orchestrator import (
            LifecycleProfile,
            ProcessIdentity,
            ProcessTreeSnapshot,
        )

        for _chain_cid, item in chain:
            process = cas_payload(
                item.donor_process_birth_cid,
                "donor process fence",
            )
            process_fields = {
                "schema", "revision_cid", "slice_manifest_cid", "slice_id",
                "donor_lane_id", "donor_track_name", "profile",
                "process_birth", "fenced_tree", "launch_process_birth_cid",
            }
            if set(process) != process_fields or process.get("schema") != (
                "ipfs_accelerate_py/agent-supervisor/plan-slice-donor-fence@1"
            ):
                raise ExecutionPlanError("donor process fence evidence is malformed")
            try:
                profile = LifecycleProfile.from_dict(process["profile"])
                identity = ProcessIdentity.from_dict(process["process_birth"])
                fenced_tree = ProcessTreeSnapshot.from_dict(process["fenced_tree"])
            except Exception as exc:
                raise ExecutionPlanError(
                    "donor process fence lifecycle evidence is invalid"
                ) from exc
            if (
                profile.to_dict() != process["profile"]
                or identity.to_dict() != process["process_birth"]
                or fenced_tree.to_dict() != process["fenced_tree"]
                or fenced_tree.members
                or process.get("revision_cid") != revision_cid
                or process.get("slice_manifest_cid") != slice_manifest_cid
                or process.get("slice_id") != slice_id
                or process.get("donor_lane_id") != item.donor_lane_id
                or identity.profile_id != profile.profile_id
                or identity.run_id != profile.run_id
                or identity.target_id != profile.target_id
            ):
                raise ExecutionPlanError("donor process fence evidence is mixed")
            launch_birth_cid = process.get("launch_process_birth_cid")
            if not isinstance(launch_birth_cid, str) or not launch_birth_cid:
                raise ExecutionPlanError("donor launch birth evidence is absent")
            launch_birth_binding = _load_plan_bound_process_birth_chain_locked(
                self.plan_revision_store,
                revision_cid=revision_cid,
                slice_id=slice_id,
                lane_id=item.donor_lane_id,
            )
            if (
                launch_birth_binding is None
                or launch_birth_binding[0] != launch_birth_cid
            ):
                raise ExecutionPlanError(
                    "donor launch birth is not its bounded chain head"
                )
            launch_birth = launch_birth_binding[1].to_dict()
            if (
                launch_birth.get("schema")
                != PLAN_BOUND_PROCESS_BIRTH_SCHEMA
                or launch_birth.get("revision_cid") != revision_cid
                or launch_birth.get("slice_manifest_cid") != slice_manifest_cid
                or launch_birth.get("slice_id") != slice_id
                or launch_birth.get("lane_id") != item.donor_lane_id
                or launch_birth.get("process_birth") != process["process_birth"]
                or launch_birth.get("profile") != process["profile"]
                or launch_birth.get("task_ids") != list(execution_slice.task_ids)
                or launch_birth.get("task_cids") != list(execution_slice.task_cids)
            ):
                raise ExecutionPlanError("donor launch birth evidence is mixed")

            attempt = cas_payload(item.attempt_absence_cid, "attempt absence")
            if set(attempt) != {
                "schema", "revision_cid", "slice_manifest_cid", "slice_id",
                "task_ids", "task_cids", "state_path", "state_identity",
                "state", "never_attempted",
            } or (
                attempt.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/plan-slice-attempt-absence@1"
                or attempt.get("revision_cid") != revision_cid
                or attempt.get("slice_manifest_cid") != slice_manifest_cid
                or attempt.get("slice_id") != slice_id
                or attempt.get("task_ids") != list(execution_slice.task_ids)
                or attempt.get("task_cids") != list(execution_slice.task_cids)
                or attempt.get("never_attempted") is not True
                or not isinstance(attempt.get("state"), Mapping)
                or not isinstance(attempt.get("state_identity"), Mapping)
            ):
                raise ExecutionPlanError("attempt absence evidence is mixed")

            claims = cas_payload(item.claim_absence_cid, "claim absence")
            if set(claims) != {
                "schema", "revision_cid", "slice_manifest_cid", "slice_id",
                "task_ids", "task_cids", "claims",
            } or (
                claims.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/plan-slice-claim-absence@1"
                or claims.get("revision_cid") != revision_cid
                or claims.get("slice_manifest_cid") != slice_manifest_cid
                or claims.get("slice_id") != slice_id
                or claims.get("task_ids") != list(execution_slice.task_ids)
                or claims.get("task_cids") != list(execution_slice.task_cids)
                or not isinstance(claims.get("claims"), list)
            ):
                raise ExecutionPlanError("claim absence evidence is mixed")
            claim_rows = claims["claims"]
            if len(claim_rows) != len(execution_slice.task_ids):
                raise ExecutionPlanError("claim absence evidence is partial")
            for row, task_id, task_cid in zip(
                claim_rows,
                execution_slice.task_ids,
                execution_slice.task_cids,
                strict=True,
            ):
                if not isinstance(row, Mapping) or set(row) != {
                    "task_id", "task_cid", "claim_path", "state",
                    "artifact_identity",
                } or (
                    row.get("task_id") != task_id
                    or row.get("task_cid") != task_cid
                    or row.get("state") != "absent"
                    or not isinstance(row.get("claim_path"), str)
                    or not isinstance(row.get("artifact_identity"), Mapping)
                    or row["artifact_identity"].get("state") != "absent"
                ):
                    raise ExecutionPlanError("claim absence row is mixed")
        return reassignment_cid, record

    def validate_slice_owner(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
        slice_id: str,
        lane_id: str,
        reassignment_cid: str = "",
    ) -> ConfiguredBoardExecutionSlice:
        """Require the launch lane to equal the current canonical slice owner."""

        with self.plan_revision_store._thread_lock:  # noqa: SLF001
            with self.plan_revision_store._guard():  # noqa: SLF001
                return self._validate_slice_owner_locked(
                    revision_cid=revision_cid,
                    slice_manifest_cid=slice_manifest_cid,
                    slice_id=slice_id,
                    lane_id=lane_id,
                    reassignment_cid=reassignment_cid,
                )

    def _validate_slice_owner_locked(
        self,
        *,
        revision_cid: str,
        slice_manifest_cid: str,
        slice_id: str,
        lane_id: str,
        reassignment_cid: str = "",
    ) -> ConfiguredBoardExecutionSlice:
        """Validate ownership with the canonical store lock already held."""

        revision_cid = _text(revision_cid, "revision_cid")
        slice_manifest_cid = _text(
            slice_manifest_cid, "slice_manifest_cid"
        )
        slice_id = _text(slice_id, "slice_id")
        lane_id = _text(lane_id, "lane_id")
        reassignment_cid = str(reassignment_cid or "").strip()
        active = _secure_store_active(self.plan_revision_store)
        if active is None or active.revision_cid != revision_cid:
            raise ExecutionSliceViolationError(
                "slice owner validation requires the active revision"
            )
        revision_payload = _secure_store_cas(
            self.plan_revision_store,
            revision_cid,
        )
        revision = PlanRevision.from_dict(revision_payload)
        if revision.to_dict() != revision_payload:
            raise ExecutionSliceViolationError(
                "slice owner revision changed during typed decode"
            )
        if revision.materialization_transaction_cid != slice_manifest_cid:
            raise ExecutionSliceViolationError(
                "slice owner validation observed a foreign manifest"
            )
        manifest = ConfiguredBoardExecutionSlices.from_dict(
            _secure_store_cas(self.plan_revision_store, slice_manifest_cid)
        )
        matches = tuple(
            item for item in manifest.slices if item.slice_id == slice_id
        )
        if len(matches) != 1:
            raise ExecutionSliceViolationError(
                "slice owner validation target is absent or duplicated"
            )
        current = self._load_slice_reassignment_locked(
            revision_cid=revision_cid,
            slice_id=slice_id,
        )
        if current is None:
            if reassignment_cid or lane_id != matches[0].lane_id:
                raise ExecutionSliceViolationError(
                    "launch lane does not own the immutable slice"
                )
        elif current[0] != reassignment_cid or current[1].recipient_lane_id != lane_id:
            raise ExecutionSliceViolationError(
                "launch lane lost the same-revision reassignment CAS"
            )
        return matches[0]

def load_plan_revision_store_binding(
    store: PlanRevisionStore,
    *,
    execution_slice_task_ids: Iterable[str] = (),
    execution_slice_task_cids: Iterable[str] = (),
) -> ActivePlanBinding:
    """Bind the real store while preserving its separate revision CID key.

    ``PlanRevision.to_dict()`` intentionally serializes semantic content only;
    its CAS key is exposed by ``revision_cid`` on the typed value.  The common
    binder expects that key alongside the body, so join them explicitly rather
    than weakening mixed-revision checks or persisting a second projection.
    """

    if not isinstance(store, PlanRevisionStore):
        raise TypeError("store must be PlanRevisionStore")
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            return _load_plan_revision_store_binding_locked(
                store,
                execution_slice_task_ids=execution_slice_task_ids,
                execution_slice_task_cids=execution_slice_task_cids,
            )


def _load_plan_revision_store_binding_locked(
    store: PlanRevisionStore,
    *,
    execution_slice_task_ids: Iterable[str] = (),
    execution_slice_task_cids: Iterable[str] = (),
) -> ActivePlanBinding:
    """Bind strict store artifacts with the canonical guard already held."""

    active = _secure_store_active(store)
    if active is None:
        raise ExecutionPlanError("plan revision store has no active revision")
    if active.quarantined:
        raise ExecutionPlanError("plan revision store is quarantined")
    stored_revision = _secure_store_cas(store, active.revision_cid)
    revision = PlanRevision.from_dict(stored_revision)
    if revision.to_dict() != stored_revision:
        raise ExecutionPlanError("plan revision changed during typed decode")
    revision_payload = revision.to_dict()
    revision_payload["revision_cid"] = revision.revision_cid
    execution_plan = _secure_store_cas(store, revision.execution_plan_cid)
    return bind_active_plan_revision(
        active=active,
        revision=revision_payload,
        execution_plan=execution_plan,
        execution_slice_task_ids=execution_slice_task_ids,
        execution_slice_task_cids=execution_slice_task_cids,
    )


__all__ = [
    "ADAPTIVE_EXECUTION_PLAN_SCHEMA", "CONFIGURED_BOARD_EXECUTION_SLICES_SCHEMA",
    "EXECUTION_SLICE_SCHEMA", "PARALLELISM_DECISION_RECEIPT_SCHEMA",
    "PLAN_BOUND_EXECUTION_LEASE_SCHEMA", "PLAN_BOUND_PROPOSAL_DISPOSITION_SCHEMA",
    "PLAN_BOUND_PROCESS_BIRTH_SCHEMA", "PLAN_BOUND_PROCESS_BIRTH_EXHAUSTED_SCHEMA",
    "PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA", "PLAN_BOUND_MERGE_ENQUEUE_INTENT_SCHEMA",
    "PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA",
    "PLAN_BOUND_TERMINAL_MISSING_SCHEMA", "PLAN_BOUND_WAVE_DIFF_BARRIER_SCHEMA",
    "PLAN_BOUND_WAVE_DIFF_BARRIER_WINDOW_SCHEMA", "PLAN_SLICE_REASSIGNMENT_SCHEMA",
    "MAX_PLAN_BOUND_WAVE_TRANSFERS",
    "AdaptiveExecutionPlan", "CapacitySnapshot", "ConfiguredBoardExecutionSlice",
    "ConfiguredBoardExecutionSlices", "ExecutionClaimConflictError", "ExecutionPlanError",
    "ExecutionReplanRequired", "ExecutionSlice", "ExecutionSliceViolationError",
    "ExecutionTask", "InvocationBudget", "ParallelismDecisionReceipt",
    "PlanBoundExecutionLease", "PlanBoundProcessBirth",
    "PlanBoundProcessBirthExhausted", "PlanBoundProposalDisposition",
    "PlanBoundTerminalMissing",
    "PlanBoundWaveDiffBarrier", "PlanSliceReassignment",
    "ProductionParallelPlanAdapter", "load_plan_revision_store_binding",
    "plan_bound_execution_lease_key", "plan_bound_process_birth_key",
    "plan_bound_process_birth_exhausted_key",
    "plan_bound_proposal_disposition_key",
    "plan_bound_terminal_missing_key",
    "plan_bound_wave_diff_barrier_key",
]
