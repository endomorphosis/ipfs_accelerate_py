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

from ..planning.parallel_plan_compiler import ParallelExecutionPlan, ParallelPlanCompiler
from ..planning.plan_revision_contracts import PlanAuthorityRoots, PlanDelta, PlanRevision
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
MAX_TASKS: Final[int] = 4096
MAX_SLICE_REASSIGNMENTS: Final[int] = 4096
MAX_AUTHORITY_JSON_BYTES: Final[int] = 1_048_576

_PLAN_BOUND_EXECUTION_LEASE_PHASES: Final[tuple[str, ...]] = (
    "reserved",
    "claimed",
    "workspace_prepared",
    "provider_ready",
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
            if self.provider_ready != (self.phase in {"provider_ready", "scope_drift"}):
                raise ExecutionPlanError("workspace provider-ready phase is mixed")
            if self.phase == "workspace_prepared" and (
                self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
            ):
                raise ExecutionPlanError("prepared workspace carries proposal result")
            if self.phase == "provider_ready" and (
                self.proposal_id
                or self.proposal_receipt_id
                or self.proposal_reason_codes
                or self.actual_changed_paths
                or self.merge_enqueue_reached
            ):
                raise ExecutionPlanError("provider-ready lease carries proposal result")
            if self.phase == "scope_drift" and (
                not self.proposal_id
                or not self.proposal_receipt_id
                or "path_outside_scope" not in self.proposal_reason_codes
                or not self.actual_changed_paths
                or self.merge_enqueue_reached
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
            schema=payload["schema"],
        )
        if _canonical(dict(payload)) != _canonical(result.to_dict()):
            raise ExecutionPlanError(
                "plan-bound execution lease failed exact semantic round trip"
            )
        return result


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

    birth_payload = _secure_store_cas(store, record.process_birth_cid)
    if (
        birth_payload.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/plan-bound-process-birth@1"
        or birth_payload.get("revision_cid") != revision_cid
        or birth_payload.get("slice_manifest_cid") != record.slice_manifest_cid
        or birth_payload.get("slice_id") != slice_id
        or birth_payload.get("lane_id") != lane_id
        or birth_payload.get("task_ids") != list(record.task_ids)
        or birth_payload.get("task_cids") != list(record.task_cids)
        or birth_payload.get("process_birth") != record.process_birth
    ):
        raise ExecutionPlanError("execution lease process birth is mixed")
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
            if len(chain) > MAX_SLICE_REASSIGNMENTS:
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
            launch_birth = cas_payload(launch_birth_cid, "launch process birth")
            if (
                launch_birth.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/plan-bound-process-birth@1"
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
    "PLAN_BOUND_EXECUTION_LEASE_SCHEMA", "PLAN_SLICE_REASSIGNMENT_SCHEMA",
    "AdaptiveExecutionPlan", "CapacitySnapshot", "ConfiguredBoardExecutionSlice",
    "ConfiguredBoardExecutionSlices", "ExecutionClaimConflictError", "ExecutionPlanError",
    "ExecutionReplanRequired", "ExecutionSlice", "ExecutionSliceViolationError",
    "ExecutionTask", "InvocationBudget", "ParallelismDecisionReceipt",
    "PlanBoundExecutionLease", "PlanSliceReassignment",
    "ProductionParallelPlanAdapter", "load_plan_revision_store_binding",
    "plan_bound_execution_lease_key",
]
