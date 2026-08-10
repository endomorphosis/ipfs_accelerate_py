"""DCR-064: schedule proof-carrying repair plans within leases, lanes, and budgets.

Interfaces
----------
* ``PathLease@1`` — exact permitted write paths bound to a fencing token.
* ``RepairResourceSchedule@1`` — deterministic lane/lease/budget assignment for
  one acyclic repair plan under a fixed policy.

Normative rules (fail-closed)
-----------------------------
* Same plan + policy always yields the same schedule identity and wave layout.
* Overlapping write paths, write roots, endpoints, and exclusive solver slots
  never share an execution wave.
* Explicit dependencies always dominate sharding/lane labels (strict sharding
  cannot override the dependency DAG).
* Assignments cover lanes, path leases, fencing tokens, timeouts, retry
  budgets, and validation resources without minting write authority or
  invoking a provider/model.
* Wave packing is list-scheduled and progress-guaranteed: every ready node is
  eventually admitted, so starvation/deadlock probes terminate on acyclic
  plans.

Predicted symbols: :class:`RepairResourceScheduler`, :class:`PathLeasePlan`.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence / schemas
# ---------------------------------------------------------------------------

PATH_LEASE_INTERFACE: Final[str] = "PathLease@1"
REPAIR_RESOURCE_SCHEDULE_INTERFACE: Final[str] = "RepairResourceSchedule@1"
DCR_RESOURCE_SCHEDULE_EVIDENCE: Final[str] = "dcr/resource-schedule@1"
REPAIR_RESOURCE_SCHEDULER_VERSION: Final[int] = 1

PATH_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-path-lease@1"
)
PATH_LEASE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-path-lease-plan@1"
)
SCHEDULED_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-scheduled-node@1"
)
REPAIR_RESOURCE_SCHEDULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-repair-resource-schedule@1"
)
RESOURCE_SCHEDULE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-resource-schedule-policy@1"
)
DEFAULT_RESOURCE_SCHEDULES_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/resource-schedules.json"
)

MAX_NODES: Final[int] = 4_096
MAX_LANES: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_DURATION_MS: Final[int] = 86_400_000
DEFAULT_TIMEOUT_MS: Final[int] = 600_000
DEFAULT_RETRY_BUDGET: Final[int] = 2
DEFAULT_LEASE_DURATION_MS: Final[int] = 900_000
DEFAULT_HEARTBEAT_MS: Final[int] = 30_000
DEFAULT_NODE_DURATION_MS: Final[int] = 60_000

# Resource classes that consume exclusive solver slots by default.
_SOLVER_RESOURCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "cpu-proof-solver",
        "cpu-proof-kernel",
        "cpu-proof-translate",
        "cpu-proof-type-check",
        "solver",
        "proof-solver",
    }
)

_VALIDATION_RESOURCE_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "cpu-validation",
        "validation",
        "cpu-proof-type-check",
    }
)

_WRITE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "operator_apply",
        "pin_update",
        "write",
        "edit",
        "materialize",
    }
)


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class RepairResourceSchedulerError(ContractValidationError):
    """Malformed plan/policy input or closed-boundary scheduling violation."""


class ScheduleDisposition(str, Enum):
    """Closed outcomes for one repair resource schedule."""

    SCHEDULED = "scheduled"
    REJECTED = "rejected"


class ConflictKind(str, Enum):
    """Closed conflict-surface vocabulary for serialization decisions."""

    PATH = "path"
    ROOT = "root"
    ENDPOINT = "endpoint"
    SOLVER = "solver"
    DEPENDENCY = "dependency"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(getattr(value, "value", value)).strip()
    if required and not text:
        raise RepairResourceSchedulerError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise RepairResourceSchedulerError(f"{name} exceeds byte bound")
    if "\x00" in text:
        raise RepairResourceSchedulerError(f"{name} contains NUL")
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _boolean(value: Any, name: str, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RepairResourceSchedulerError(f"{name} must be a boolean")


def _integer(
    value: Any,
    name: str,
    *,
    default: int | None = None,
    minimum: int = 0,
    maximum: int = MAX_DURATION_MS,
) -> int:
    if value in (None, ""):
        if default is None:
            raise RepairResourceSchedulerError(f"{name} is required")
        return default
    if isinstance(value, bool):
        raise RepairResourceSchedulerError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RepairResourceSchedulerError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise RepairResourceSchedulerError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _safe_relative_path(value: Any, *, field: str) -> str:
    text = _text(value, field, required=True, limit=MAX_PATH_BYTES)
    normalized = text.replace("\\", "/").removeprefix("./").rstrip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise RepairResourceSchedulerError(f"invalid repository path for {field}")
    return path.as_posix()


def _paths(values: Any, *, field: str) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)):
        items = (values,)
    elif isinstance(values, Mapping):
        items = values.keys()
    elif isinstance(values, Sequence):
        items = values
    else:
        raise RepairResourceSchedulerError(f"{field} must be a path sequence")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = _safe_relative_path(item, field=field)
        if path not in seen:
            seen.add(path)
            out.append(path)
    return tuple(sorted(out))


def _ids(values: Any, *, field: str) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise RepairResourceSchedulerError(f"{field} must be a sequence of strings")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, field, required=True)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {name: getattr(value, name) for name in fields}
    raise RepairResourceSchedulerError("records must be mappings or expose to_dict()")


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _path_overlap_set(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    hits: set[str] = set()
    for a in left:
        for b in right:
            if _paths_overlap(a, b):
                hits.add(a if len(a) <= len(b) else b)
    return tuple(sorted(hits))


def _is_solver_class(resource_class: str) -> bool:
    return resource_class in _SOLVER_RESOURCE_CLASSES or "solver" in resource_class


def _is_validation_class(resource_class: str, kind: str) -> bool:
    return (
        resource_class in _VALIDATION_RESOURCE_CLASSES
        or kind in {"validation", "validate"}
        or resource_class.startswith("cpu-validation")
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSchedulePolicy(CanonicalContract):
    """Deterministic admission and packing policy for repair plan scheduling."""

    SCHEMA: ClassVar[str] = RESOURCE_SCHEDULE_POLICY_SCHEMA

    max_lanes: int = 4
    max_wave_width: int = 4
    max_solver_concurrency: int = 1
    max_validation_concurrency: int = 2
    max_root_writers: int = 1
    default_timeout_ms: int = DEFAULT_TIMEOUT_MS
    default_retry_budget: int = DEFAULT_RETRY_BUDGET
    lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS
    heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_MS
    default_node_duration_ms: int = DEFAULT_NODE_DURATION_MS
    base_fence_epoch: int = 1
    serialize_shared_roots: bool = True
    serialize_endpoints: bool = True
    serialize_solver_resources: bool = True
    policy_id: str = "policy:dcr-resource-schedule@1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_lanes",
            _integer(self.max_lanes, "max_lanes", minimum=1, maximum=MAX_LANES),
        )
        object.__setattr__(
            self,
            "max_wave_width",
            _integer(
                self.max_wave_width,
                "max_wave_width",
                minimum=1,
                maximum=MAX_LANES,
            ),
        )
        object.__setattr__(
            self,
            "max_solver_concurrency",
            _integer(
                self.max_solver_concurrency,
                "max_solver_concurrency",
                minimum=1,
                maximum=MAX_LANES,
            ),
        )
        object.__setattr__(
            self,
            "max_validation_concurrency",
            _integer(
                self.max_validation_concurrency,
                "max_validation_concurrency",
                minimum=1,
                maximum=MAX_LANES,
            ),
        )
        object.__setattr__(
            self,
            "max_root_writers",
            _integer(
                self.max_root_writers,
                "max_root_writers",
                minimum=1,
                maximum=MAX_LANES,
            ),
        )
        object.__setattr__(
            self,
            "default_timeout_ms",
            _integer(
                self.default_timeout_ms,
                "default_timeout_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "default_retry_budget",
            _integer(
                self.default_retry_budget,
                "default_retry_budget",
                minimum=0,
                maximum=1_000,
            ),
        )
        object.__setattr__(
            self,
            "lease_duration_ms",
            _integer(
                self.lease_duration_ms,
                "lease_duration_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "heartbeat_interval_ms",
            _integer(
                self.heartbeat_interval_ms,
                "heartbeat_interval_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "default_node_duration_ms",
            _integer(
                self.default_node_duration_ms,
                "default_node_duration_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "base_fence_epoch",
            _integer(
                self.base_fence_epoch,
                "base_fence_epoch",
                minimum=0,
                maximum=2**31 - 1,
            ),
        )
        object.__setattr__(
            self,
            "serialize_shared_roots",
            _boolean(self.serialize_shared_roots, "serialize_shared_roots", default=True),
        )
        object.__setattr__(
            self,
            "serialize_endpoints",
            _boolean(self.serialize_endpoints, "serialize_endpoints", default=True),
        )
        object.__setattr__(
            self,
            "serialize_solver_resources",
            _boolean(
                self.serialize_solver_resources,
                "serialize_solver_resources",
                default=True,
            ),
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id, "policy_id"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_lanes": self.max_lanes,
            "max_wave_width": self.max_wave_width,
            "max_solver_concurrency": self.max_solver_concurrency,
            "max_validation_concurrency": self.max_validation_concurrency,
            "max_root_writers": self.max_root_writers,
            "default_timeout_ms": self.default_timeout_ms,
            "default_retry_budget": self.default_retry_budget,
            "lease_duration_ms": self.lease_duration_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "default_node_duration_ms": self.default_node_duration_ms,
            "base_fence_epoch": self.base_fence_epoch,
            "serialize_shared_roots": self.serialize_shared_roots,
            "serialize_endpoints": self.serialize_endpoints,
            "serialize_solver_resources": self.serialize_solver_resources,
            "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourceSchedulePolicy":
        if payload is None:
            return cls()
        data = _mapping(payload)
        known = {
            "policy_id",
            "max_lanes",
            "max_wave_width",
            "max_solver_concurrency",
            "max_validation_concurrency",
            "max_root_writers",
            "default_timeout_ms",
            "default_retry_budget",
            "lease_duration_ms",
            "heartbeat_interval_ms",
            "default_node_duration_ms",
            "base_fence_epoch",
            "serialize_shared_roots",
            "serialize_endpoints",
            "serialize_solver_resources",
        }
        return cls(**{key: data[key] for key in known if key in data})

    @property
    def policy_cid(self) -> str:
        return self.content_id

    def budget_counters(self) -> dict[str, int]:
        """Integer budget projection embedded in schedule evidence."""

        return {
            "max_lanes": self.max_lanes,
            "max_wave_width": self.max_wave_width,
            "max_solver_concurrency": self.max_solver_concurrency,
            "max_validation_concurrency": self.max_validation_concurrency,
            "max_root_writers": self.max_root_writers,
            "default_timeout_ms": self.default_timeout_ms,
            "default_retry_budget": self.default_retry_budget,
            "lease_duration_ms": self.lease_duration_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "default_node_duration_ms": self.default_node_duration_ms,
            "base_fence_epoch": self.base_fence_epoch,
        }


# ---------------------------------------------------------------------------
# Plan nodes (normalized)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulableNode:
    """Normalized executable node accepted by the resource scheduler."""

    node_id: str
    kind: str
    depends_on: tuple[str, ...]
    write_set: tuple[str, ...]
    owner_root: str
    resource_class: str
    validation_ref: str = ""
    endpoints: tuple[str, ...] = ()
    duration_ms: int = DEFAULT_NODE_DURATION_MS
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    retry_budget: int = DEFAULT_RETRY_BUDGET
    shard_hint: str = ""
    lane_hint: str = ""
    exclusive_group: str = ""
    evidence_cid: str = ""
    operator_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "kind",
            _text(self.kind, "kind").lower().replace("-", "_").replace(" ", "_"),
        )
        object.__setattr__(self, "depends_on", _ids(self.depends_on, field="depends_on"))
        object.__setattr__(self, "write_set", _paths(self.write_set, field="write_set"))
        object.__setattr__(
            self,
            "owner_root",
            _optional_text(self.owner_root, "owner_root"),
        )
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class or "cpu-medium", "resource_class"),
        )
        object.__setattr__(
            self,
            "validation_ref",
            _optional_text(self.validation_ref, "validation_ref"),
        )
        endpoints = list(_ids(self.endpoints, field="endpoints"))
        if self.validation_ref and self.validation_ref not in endpoints:
            endpoints.append(self.validation_ref)
        object.__setattr__(self, "endpoints", tuple(sorted(set(endpoints))))
        object.__setattr__(
            self,
            "duration_ms",
            _integer(
                self.duration_ms,
                "duration_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _integer(
                self.timeout_ms,
                "timeout_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "retry_budget",
            _integer(self.retry_budget, "retry_budget", minimum=0, maximum=1_000),
        )
        object.__setattr__(self, "shard_hint", _optional_text(self.shard_hint, "shard_hint"))
        object.__setattr__(self, "lane_hint", _optional_text(self.lane_hint, "lane_hint"))
        object.__setattr__(
            self,
            "exclusive_group",
            _optional_text(self.exclusive_group, "exclusive_group"),
        )
        object.__setattr__(
            self,
            "evidence_cid",
            _optional_text(self.evidence_cid, "evidence_cid"),
        )
        object.__setattr__(
            self,
            "operator_ref",
            _optional_text(self.operator_ref, "operator_ref"),
        )
        if self.node_id in self.depends_on:
            raise RepairResourceSchedulerError(
                f"dependency cycle on self for {self.node_id}"
            )

    @property
    def is_writer(self) -> bool:
        return bool(self.write_set) or self.kind in _WRITE_KINDS

    @property
    def is_solver(self) -> bool:
        return _is_solver_class(self.resource_class)

    @property
    def is_validation(self) -> bool:
        return _is_validation_class(self.resource_class, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "write_set": list(self.write_set),
            "owner_root": self.owner_root,
            "resource_class": self.resource_class,
            "validation_ref": self.validation_ref,
            "endpoints": list(self.endpoints),
            "duration_ms": self.duration_ms,
            "timeout_ms": self.timeout_ms,
            "retry_budget": self.retry_budget,
            "shard_hint": self.shard_hint,
            "lane_hint": self.lane_hint,
            "exclusive_group": self.exclusive_group,
            "evidence_cid": self.evidence_cid,
            "operator_ref": self.operator_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | Any) -> "SchedulableNode":
        data = _mapping(payload)
        write_set = (
            data.get("write_set")
            or data.get("write_paths")
            or data.get("predicted_files")
            or data.get("outputs")
            or ()
        )
        endpoints = data.get("endpoints") or data.get("endpoint_ids") or ()
        operator_raw = (
            data.get("operator_ref")
            or data.get("operator_id")
            or data.get("operator")
            or ""
        )
        if isinstance(operator_raw, Mapping):
            operator_ref = str(
                operator_raw.get("operator_id")
                or operator_raw.get("operator_ref")
                or operator_raw.get("id")
                or operator_raw.get("kind")
                or ""
            )
        else:
            operator_ref = str(operator_raw or "")
        kind_raw = data.get("kind") or data.get("node_kind") or "operator_apply"
        kind = str(getattr(kind_raw, "value", kind_raw) or "operator_apply")
        return cls(
            node_id=str(
                data.get("node_id")
                or data.get("task_id")
                or data.get("id")
                or data.get("task_cid")
                or ""
            ),
            kind=kind,
            depends_on=tuple(data.get("depends_on") or data.get("dependencies") or ()),
            write_set=tuple(write_set),
            owner_root=str(data.get("owner_root") or data.get("owner") or ""),
            resource_class=str(data.get("resource_class") or "cpu-medium"),
            validation_ref=str(data.get("validation_ref") or data.get("validation") or ""),
            endpoints=tuple(endpoints),
            duration_ms=int(
                data.get("duration_ms")
                or data.get("estimated_duration_ms")
                or DEFAULT_NODE_DURATION_MS
            ),
            timeout_ms=int(data.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
            retry_budget=int(
                data.get("retry_budget")
                if data.get("retry_budget") is not None
                else DEFAULT_RETRY_BUDGET
            ),
            shard_hint=str(
                data.get("shard_hint")
                or data.get("shard_key")
                or data.get("shard_id")
                or ""
            ),
            lane_hint=str(
                data.get("lane_hint")
                or data.get("parallel_lane")
                or data.get("lane_label")
                or ""
            ),
            exclusive_group=str(data.get("exclusive_group") or ""),
            evidence_cid=str(data.get("evidence_cid") or ""),
            operator_ref=operator_ref,
        )


# ---------------------------------------------------------------------------
# Path lease / assignment contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathLease(CanonicalContract):
    """Bounded write-path lease with a fencing token (``PathLease@1``)."""

    SCHEMA: ClassVar[str] = PATH_LEASE_SCHEMA
    INTERFACE: ClassVar[str] = PATH_LEASE_INTERFACE

    lease_id: str
    fencing_token: str
    permitted_write_paths: tuple[str, ...]
    owner_root: str = ""
    node_ids: tuple[str, ...] = ()
    fence_epoch: int = 1
    lease_duration_ms: int = DEFAULT_LEASE_DURATION_MS
    heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_MS
    lease_scope: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "fencing_token", _text(self.fencing_token, "fencing_token")
        )
        paths = _paths(self.permitted_write_paths, field="permitted_write_paths")
        if not paths:
            raise RepairResourceSchedulerError(
                "path lease requires at least one permitted write path"
            )
        object.__setattr__(self, "permitted_write_paths", paths)
        object.__setattr__(self, "owner_root", _optional_text(self.owner_root, "owner_root"))
        object.__setattr__(self, "node_ids", _ids(self.node_ids, field="node_ids"))
        object.__setattr__(
            self,
            "fence_epoch",
            _integer(self.fence_epoch, "fence_epoch", minimum=0, maximum=2**31 - 1),
        )
        object.__setattr__(
            self,
            "lease_duration_ms",
            _integer(
                self.lease_duration_ms,
                "lease_duration_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "heartbeat_interval_ms",
            _integer(
                self.heartbeat_interval_ms,
                "heartbeat_interval_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "lease_scope",
            _optional_text(self.lease_scope, "lease_scope") or self.lease_id,
        )

    def contains(self, path: str) -> bool:
        candidate = _safe_relative_path(path, field="path")
        return any(_paths_overlap(candidate, owned) for owned in self.permitted_write_paths)

    def overlaps(self, other: "PathLease") -> bool:
        return bool(_path_overlap_set(self.permitted_write_paths, other.permitted_write_paths))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "permitted_write_paths": list(self.permitted_write_paths),
            "owner_root": self.owner_root,
            "node_ids": list(self.node_ids),
            "fence_epoch": self.fence_epoch,
            "lease_duration_ms": self.lease_duration_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "lease_scope": self.lease_scope,
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
            "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathLease":
        data = _mapping(payload)
        return cls(
            lease_id=str(data.get("lease_id") or ""),
            fencing_token=str(data.get("fencing_token") or data.get("fence_token") or ""),
            permitted_write_paths=tuple(
                data.get("permitted_write_paths") or data.get("write_set") or ()
            ),
            owner_root=str(data.get("owner_root") or ""),
            node_ids=tuple(data.get("node_ids") or ()),
            fence_epoch=int(data.get("fence_epoch") or 1),
            lease_duration_ms=int(
                data.get("lease_duration_ms") or DEFAULT_LEASE_DURATION_MS
            ),
            heartbeat_interval_ms=int(
                data.get("heartbeat_interval_ms") or DEFAULT_HEARTBEAT_MS
            ),
            lease_scope=str(data.get("lease_scope") or ""),
        )


@dataclass(frozen=True)
class PathLeasePlan(CanonicalContract):
    """Deterministic collection of path leases for one repair schedule."""

    SCHEMA: ClassVar[str] = PATH_LEASE_PLAN_SCHEMA

    plan_id: str
    leases: tuple[PathLease, ...]
    conflict_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        leases = tuple(
            lease if isinstance(lease, PathLease) else PathLease.from_dict(lease)
            for lease in (self.leases or ())
        )
        # Stable order by lease_id for identity.
        leases = tuple(sorted(leases, key=lambda item: item.lease_id))
        object.__setattr__(self, "leases", leases)
        pairs: list[tuple[str, str]] = []
        for left, right in self.conflict_pairs or ():
            a, b = sorted((_text(left, "conflict_pair"), _text(right, "conflict_pair")))
            if a != b and (a, b) not in pairs:
                pairs.append((a, b))
        object.__setattr__(self, "conflict_pairs", tuple(sorted(pairs)))

    def lease_for_node(self, node_id: str) -> PathLease | None:
        for lease in self.leases:
            if node_id in lease.node_ids:
                return lease
        return None

    def _payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "leases": [lease.to_dict() for lease in self.leases],
            "conflict_pairs": [list(pair) for pair in self.conflict_pairs],
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
            "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathLeasePlan":
        data = _mapping(payload)
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            leases=tuple(data.get("leases") or ()),
            conflict_pairs=tuple(
                tuple(pair) for pair in (data.get("conflict_pairs") or ())
            ),
        )


@dataclass(frozen=True)
class ConflictEdge:
    """One undirected conflict edge between schedulable nodes."""

    left_node_id: str
    right_node_id: str
    kinds: tuple[str, ...]
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        left = _text(self.left_node_id, "left_node_id")
        right = _text(self.right_node_id, "right_node_id")
        if left == right:
            raise RepairResourceSchedulerError("conflict edge requires distinct nodes")
        ordered = tuple(sorted((left, right)))
        object.__setattr__(self, "left_node_id", ordered[0])
        object.__setattr__(self, "right_node_id", ordered[1])
        kinds = tuple(sorted(set(_ids(self.kinds, field="kinds"))))
        if not kinds:
            raise RepairResourceSchedulerError("conflict edge requires at least one kind")
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "evidence", _ids(self.evidence, field="evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_node_id": self.left_node_id,
            "right_node_id": self.right_node_id,
            "kinds": list(self.kinds),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ScheduledNodeAssignment(CanonicalContract):
    """One node's lane, lease, fence, timeout, retry, and validation binding."""

    SCHEMA: ClassVar[str] = SCHEDULED_NODE_SCHEMA

    node_id: str
    wave: int
    lane: int
    shard_id: str
    resource_class: str
    lease_id: str
    fencing_token: str
    fence_epoch: int
    timeout_ms: int
    retry_budget: int
    validation_resource: str
    duration_ms: int
    start_offset_ms: int
    depends_on: tuple[str, ...] = ()
    write_set: tuple[str, ...] = ()
    owner_root: str = ""
    endpoints: tuple[str, ...] = ()
    exclusive_group: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(
            self, "wave", _integer(self.wave, "wave", minimum=0, maximum=MAX_NODES)
        )
        object.__setattr__(
            self, "lane", _integer(self.lane, "lane", minimum=0, maximum=MAX_LANES)
        )
        object.__setattr__(self, "shard_id", _text(self.shard_id, "shard_id"))
        object.__setattr__(
            self, "resource_class", _text(self.resource_class, "resource_class")
        )
        object.__setattr__(self, "lease_id", _optional_text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "fencing_token", _optional_text(self.fencing_token, "fencing_token")
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _integer(self.fence_epoch, "fence_epoch", minimum=0, maximum=2**31 - 1),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _integer(self.timeout_ms, "timeout_ms", minimum=1, maximum=MAX_DURATION_MS),
        )
        object.__setattr__(
            self,
            "retry_budget",
            _integer(self.retry_budget, "retry_budget", minimum=0, maximum=1_000),
        )
        object.__setattr__(
            self,
            "validation_resource",
            _text(self.validation_resource or "validation:none", "validation_resource"),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _integer(
                self.duration_ms, "duration_ms", minimum=1, maximum=MAX_DURATION_MS
            ),
        )
        object.__setattr__(
            self,
            "start_offset_ms",
            _integer(
                self.start_offset_ms,
                "start_offset_ms",
                minimum=0,
                maximum=MAX_DURATION_MS * MAX_NODES,
            ),
        )
        object.__setattr__(self, "depends_on", _ids(self.depends_on, field="depends_on"))
        object.__setattr__(self, "write_set", _paths(self.write_set, field="write_set"))
        object.__setattr__(self, "owner_root", _optional_text(self.owner_root, "owner_root"))
        object.__setattr__(self, "endpoints", _ids(self.endpoints, field="endpoints"))
        object.__setattr__(
            self, "exclusive_group", _optional_text(self.exclusive_group, "exclusive_group")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "wave": self.wave,
            "lane": self.lane,
            "shard_id": self.shard_id,
            "resource_class": self.resource_class,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "fence_epoch": self.fence_epoch,
            "timeout_ms": self.timeout_ms,
            "retry_budget": self.retry_budget,
            "validation_resource": self.validation_resource,
            "duration_ms": self.duration_ms,
            "start_offset_ms": self.start_offset_ms,
            "depends_on": list(self.depends_on),
            "write_set": list(self.write_set),
            "owner_root": self.owner_root,
            "endpoints": list(self.endpoints),
            "exclusive_group": self.exclusive_group,
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
            "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScheduledNodeAssignment":
        data = _mapping(payload)
        return cls(
            node_id=str(data.get("node_id") or ""),
            wave=int(data.get("wave") or 0),
            lane=int(data.get("lane") or 0),
            shard_id=str(data.get("shard_id") or ""),
            resource_class=str(data.get("resource_class") or "cpu-medium"),
            lease_id=str(data.get("lease_id") or ""),
            fencing_token=str(data.get("fencing_token") or ""),
            fence_epoch=int(data.get("fence_epoch") or 0),
            timeout_ms=int(data.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
            retry_budget=int(
                data.get("retry_budget")
                if data.get("retry_budget") is not None
                else DEFAULT_RETRY_BUDGET
            ),
            validation_resource=str(data.get("validation_resource") or "validation:none"),
            duration_ms=int(data.get("duration_ms") or DEFAULT_NODE_DURATION_MS),
            start_offset_ms=int(data.get("start_offset_ms") or 0),
            depends_on=tuple(data.get("depends_on") or ()),
            write_set=tuple(data.get("write_set") or ()),
            owner_root=str(data.get("owner_root") or ""),
            endpoints=tuple(data.get("endpoints") or ()),
            exclusive_group=str(data.get("exclusive_group") or ""),
        )


@dataclass(frozen=True)
class ScheduleWave:
    """One concurrent execution wave of non-conflicting nodes."""

    wave: int
    node_ids: tuple[str, ...]
    width: int
    resource_usage: Mapping[str, int] = field(default_factory=dict)
    start_offset_ms: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "wave", _integer(self.wave, "wave", minimum=0, maximum=MAX_NODES)
        )
        node_ids = _ids(self.node_ids, field="node_ids")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "width", len(node_ids))
        usage = {
            str(key): _integer(value, f"resource_usage[{key}]", minimum=0, maximum=MAX_LANES)
            for key, value in dict(self.resource_usage or {}).items()
        }
        object.__setattr__(self, "resource_usage", MappingProxyType(dict(sorted(usage.items()))))
        object.__setattr__(
            self,
            "start_offset_ms",
            _integer(
                self.start_offset_ms,
                "start_offset_ms",
                minimum=0,
                maximum=MAX_DURATION_MS * MAX_NODES,
            ),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _integer(
                self.duration_ms,
                "duration_ms",
                minimum=0,
                maximum=MAX_DURATION_MS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave": self.wave,
            "node_ids": list(self.node_ids),
            "width": self.width,
            "resource_usage": dict(self.resource_usage),
            "start_offset_ms": self.start_offset_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class RepairResourceSchedule(CanonicalContract):
    """Deterministic resource schedule for one repair plan (``RepairResourceSchedule@1``)."""

    SCHEMA: ClassVar[str] = REPAIR_RESOURCE_SCHEDULE_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_RESOURCE_SCHEDULE_INTERFACE

    plan_id: str
    policy_id: str
    disposition: ScheduleDisposition
    assignments: tuple[ScheduledNodeAssignment, ...]
    waves: tuple[ScheduleWave, ...]
    path_lease_plan: PathLeasePlan
    conflict_graph: tuple[ConflictEdge, ...]
    critical_path: tuple[str, ...]
    critical_path_duration_ms: int
    budgets: Mapping[str, int]
    topological_order: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        try:
            disposition = ScheduleDisposition(
                str(getattr(self.disposition, "value", self.disposition))
            )
        except ValueError as exc:
            raise RepairResourceSchedulerError(
                f"unsupported disposition: {self.disposition!r}"
            ) from exc
        object.__setattr__(self, "disposition", disposition)

        assignments = tuple(
            item
            if isinstance(item, ScheduledNodeAssignment)
            else ScheduledNodeAssignment.from_dict(item)
            for item in (self.assignments or ())
        )
        assignments = tuple(sorted(assignments, key=lambda item: (item.wave, item.lane, item.node_id)))
        object.__setattr__(self, "assignments", assignments)

        waves = tuple(
            item if isinstance(item, ScheduleWave) else ScheduleWave(**_mapping(item))
            for item in (self.waves or ())
        )
        waves = tuple(sorted(waves, key=lambda item: item.wave))
        object.__setattr__(self, "waves", waves)

        lease_plan = self.path_lease_plan
        if not isinstance(lease_plan, PathLeasePlan):
            lease_plan = PathLeasePlan.from_dict(_mapping(lease_plan))
        object.__setattr__(self, "path_lease_plan", lease_plan)

        edges = tuple(
            item if isinstance(item, ConflictEdge) else ConflictEdge(**_mapping(item))
            for item in (self.conflict_graph or ())
        )
        edges = tuple(
            sorted(edges, key=lambda item: (item.left_node_id, item.right_node_id, item.kinds))
        )
        object.__setattr__(self, "conflict_graph", edges)
        object.__setattr__(
            self, "critical_path", _ids(self.critical_path, field="critical_path")
        )
        object.__setattr__(
            self,
            "critical_path_duration_ms",
            _integer(
                self.critical_path_duration_ms,
                "critical_path_duration_ms",
                minimum=0,
                maximum=MAX_DURATION_MS * MAX_NODES,
            ),
        )
        budgets = {
            str(key): _integer(
                value,
                f"budgets[{key}]",
                minimum=0,
                maximum=MAX_DURATION_MS * MAX_NODES,
            )
            for key, value in dict(self.budgets or {}).items()
        }
        object.__setattr__(self, "budgets", MappingProxyType(dict(sorted(budgets.items()))))
        object.__setattr__(
            self,
            "topological_order",
            _ids(self.topological_order, field="topological_order"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, field="reason_codes")
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_write_authority", False)

    @property
    def ok(self) -> bool:
        return self.disposition is ScheduleDisposition.SCHEDULED

    @property
    def schedule_cid(self) -> str:
        return self.content_id

    def assignment_map(self) -> Mapping[str, ScheduledNodeAssignment]:
        return MappingProxyType({item.node_id: item for item in self.assignments})

    def concurrent_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for wave in self.waves:
            ids = list(wave.node_ids)
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    pairs.append(tuple(sorted((left, right))))  # type: ignore[arg-type]
        return tuple(sorted(set(pairs)))

    def evidence_subset(self) -> dict[str, Any]:
        """Project the DCR-064 evidence subset."""

        return {
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
            "schedule_cid": self.schedule_cid,
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "lane_shard": [
                {
                    "node_id": item.node_id,
                    "lane": item.lane,
                    "shard": item.shard_id,
                    "wave": item.wave,
                }
                for item in self.assignments
            ],
            "conflict_graph": [edge.to_dict() for edge in self.conflict_graph],
            "lease_fence": [
                {
                    "node_id": item.node_id,
                    "lease_id": item.lease_id,
                    "fencing_token": item.fencing_token,
                    "fence_epoch": item.fence_epoch,
                }
                for item in self.assignments
            ],
            "budgets": dict(self.budgets),
            "critical_path": list(self.critical_path),
            "critical_path_duration_ms": self.critical_path_duration_ms,
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "disposition": self.disposition.value,
            "assignments": [item.to_dict() for item in self.assignments],
            "waves": [wave.to_dict() for wave in self.waves],
            "path_lease_plan": self.path_lease_plan.to_dict(),
            "conflict_graph": [edge.to_dict() for edge in self.conflict_graph],
            "critical_path": list(self.critical_path),
            "critical_path_duration_ms": self.critical_path_duration_ms,
            "budgets": dict(self.budgets),
            "topological_order": list(self.topological_order),
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
            "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
            "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairResourceSchedule":
        data = _mapping(payload)
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            policy_id=str(data.get("policy_id") or ""),
            disposition=str(data.get("disposition") or ScheduleDisposition.REJECTED.value),
            assignments=tuple(data.get("assignments") or ()),
            waves=tuple(data.get("waves") or ()),
            path_lease_plan=data.get("path_lease_plan") or {"plan_id": "", "leases": []},
            conflict_graph=tuple(data.get("conflict_graph") or ()),
            critical_path=tuple(data.get("critical_path") or ()),
            critical_path_duration_ms=int(data.get("critical_path_duration_ms") or 0),
            budgets=dict(data.get("budgets") or {}),
            topological_order=tuple(data.get("topological_order") or ()),
            reason_codes=tuple(data.get("reason_codes") or ()),
        )


# ---------------------------------------------------------------------------
# Normalization / graph construction
# ---------------------------------------------------------------------------


def _extract_plan_id(plan: Any, explicit: str | None = None) -> str:
    if explicit:
        return _text(explicit, "plan_id")
    if isinstance(plan, Mapping):
        for key in ("plan_id", "id", "schedule_plan_id"):
            if plan.get(key):
                return _text(plan[key], "plan_id")
    plan_id = getattr(plan, "plan_id", None)
    if plan_id:
        return _text(plan_id, "plan_id")
    return "plan:anonymous"


def _extract_nodes(plan: Any) -> tuple[SchedulableNode, ...]:
    if plan is None:
        raise RepairResourceSchedulerError("plan is required")
    if isinstance(plan, (list, tuple)):
        raw_nodes: Sequence[Any] = plan
    else:
        data = _mapping(plan)
        if "nodes" in data:
            raw_nodes = tuple(data.get("nodes") or ())
        elif "tasks" in data:
            raw_nodes = tuple(data.get("tasks") or ())
        elif "assignments" in data:
            raw_nodes = tuple(data.get("assignments") or ())
        else:
            # Single-node plan mapping.
            raw_nodes = (data,)
    if not raw_nodes:
        raise RepairResourceSchedulerError("plan has no schedulable nodes")
    if len(raw_nodes) > MAX_NODES:
        raise RepairResourceSchedulerError("plan exceeds node bound")

    nodes = tuple(SchedulableNode.from_dict(item) for item in raw_nodes)
    by_id: dict[str, SchedulableNode] = {}
    for node in nodes:
        if node.node_id in by_id:
            raise RepairResourceSchedulerError(f"duplicate node_id: {node.node_id}")
        by_id[node.node_id] = node
    for node in nodes:
        for dep in node.depends_on:
            if dep not in by_id:
                raise RepairResourceSchedulerError(f"unknown dependency: {dep}")
    return nodes


def topological_order(nodes: Sequence[SchedulableNode]) -> tuple[str, ...]:
    """Deterministic Kahn topological order; raises on cycles."""

    by_id = {node.node_id: node for node in nodes}
    indegree = {node_id: 0 for node_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in node.depends_on:
            children[dep].append(node.node_id)
            indegree[node.node_id] += 1
    ready = deque(sorted(node_id for node_id, deg in indegree.items() if deg == 0))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        if len(ready) > 1:
            ready = deque(sorted(ready))
    if len(order) != len(by_id):
        raise RepairResourceSchedulerError("dependency cycle in plan")
    return tuple(order)


def _pairwise_conflicts(
    left: SchedulableNode,
    right: SchedulableNode,
    policy: ResourceSchedulePolicy,
) -> ConflictEdge | None:
    kinds: list[str] = []
    evidence: list[str] = []

    path_hits = _path_overlap_set(left.write_set, right.write_set)
    if path_hits:
        kinds.append(ConflictKind.PATH.value)
        evidence.extend(path_hits)

    if (
        policy.serialize_shared_roots
        and left.is_writer
        and right.is_writer
        and left.owner_root
        and left.owner_root == right.owner_root
    ):
        kinds.append(ConflictKind.ROOT.value)
        evidence.append(f"root:{left.owner_root}")

    if policy.serialize_endpoints:
        shared_endpoints = sorted(set(left.endpoints) & set(right.endpoints))
        if shared_endpoints:
            kinds.append(ConflictKind.ENDPOINT.value)
            evidence.extend(shared_endpoints)

    if left.exclusive_group and left.exclusive_group == right.exclusive_group:
        kinds.append(ConflictKind.SOLVER.value)
        evidence.append(f"exclusive:{left.exclusive_group}")

    if not kinds:
        return None
    return ConflictEdge(
        left_node_id=left.node_id,
        right_node_id=right.node_id,
        kinds=tuple(kinds),
        evidence=tuple(sorted(set(evidence))),
    )


def build_conflict_graph(
    nodes: Sequence[SchedulableNode],
    policy: ResourceSchedulePolicy,
) -> tuple[ConflictEdge, ...]:
    """Build undirected conflict edges for path/root/endpoint exclusivity."""

    ordered = sorted(nodes, key=lambda node: node.node_id)
    edges: list[ConflictEdge] = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            edge = _pairwise_conflicts(left, right, policy)
            if edge is not None:
                edges.append(edge)
    return tuple(edges)


def _critical_path(
    nodes: Sequence[SchedulableNode],
    order: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    by_id = {node.node_id: node for node in nodes}
    finish: dict[str, int] = {}
    pred: dict[str, str | None] = {}
    for node_id in order:
        node = by_id[node_id]
        best_pred: str | None = None
        best_start = 0
        for dep in node.depends_on:
            dep_finish = finish[dep]
            if dep_finish >= best_start:
                best_start = dep_finish
                best_pred = dep
        finish[node_id] = best_start + node.duration_ms
        pred[node_id] = best_pred
    if not finish:
        return (), 0
    terminal = max(sorted(finish), key=lambda node_id: (finish[node_id], node_id))
    path: list[str] = []
    cursor: str | None = terminal
    while cursor is not None:
        path.append(cursor)
        cursor = pred[cursor]
    path.reverse()
    return tuple(path), finish[terminal]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _validation_resource_for(node: SchedulableNode, wave: int, lane: int) -> str:
    if node.validation_ref:
        ref = node.validation_ref
        if ref.startswith("validation:"):
            return ref
        return f"validation:{ref}"
    if node.is_validation:
        return f"validation:slot:{wave}:{lane}"
    return "validation:none"


def _lease_identity(
    *,
    plan_id: str,
    node: SchedulableNode,
    wave: int,
    fence_epoch: int,
) -> tuple[str, str]:
    material = {
        "plan_id": plan_id,
        "node_id": node.node_id,
        "write_set": list(node.write_set),
        "owner_root": node.owner_root,
        "wave": wave,
        "fence_epoch": fence_epoch,
        "interface": PATH_LEASE_INTERFACE,
    }
    digest = content_identity(material)
    lease_id = f"lease:{digest}"
    fencing_token = f"fence:{digest}"
    return lease_id, fencing_token


def _fits_in_wave(
    node: SchedulableNode,
    selected: Sequence[SchedulableNode],
    *,
    policy: ResourceSchedulePolicy,
    hard_conflicts: Mapping[str, set[str]],
) -> bool:
    if len(selected) >= min(policy.max_lanes, policy.max_wave_width):
        return False

    solvers = sum(1 for item in selected if item.is_solver)
    if (
        policy.serialize_solver_resources
        and node.is_solver
        and solvers >= policy.max_solver_concurrency
    ):
        return False

    validations = sum(1 for item in selected if item.is_validation)
    if node.is_validation and validations >= policy.max_validation_concurrency:
        return False

    if policy.serialize_shared_roots and node.is_writer and node.owner_root:
        root_writers = sum(
            1
            for item in selected
            if item.is_writer and item.owner_root == node.owner_root
        )
        if root_writers >= policy.max_root_writers:
            return False

    for peer in selected:
        if node.node_id in hard_conflicts.get(peer.node_id, set()):
            return False
        # Solver class packing is budget-based above; still block identical
        # exclusive groups even when solver concurrency allows multiple slots.
        if node.exclusive_group and node.exclusive_group == peer.exclusive_group:
            return False
    return True


class RepairResourceScheduler:
    """Deterministic lane/lease/budget scheduler for repair plan DAGs."""

    INTERFACE: ClassVar[str] = REPAIR_RESOURCE_SCHEDULE_INTERFACE

    def __init__(self, policy: ResourceSchedulePolicy | Mapping[str, Any] | None = None) -> None:
        if isinstance(policy, ResourceSchedulePolicy):
            self.policy = policy
        else:
            self.policy = ResourceSchedulePolicy.from_dict(policy)

    def schedule(
        self,
        plan: Any,
        *,
        plan_id: str | None = None,
        policy: ResourceSchedulePolicy | Mapping[str, Any] | None = None,
    ) -> RepairResourceSchedule:
        """Compile a deterministic resource schedule for ``plan``."""

        active_policy = (
            self.policy
            if policy is None
            else (
                policy
                if isinstance(policy, ResourceSchedulePolicy)
                else ResourceSchedulePolicy.from_dict(policy)
            )
        )
        try:
            nodes = _extract_nodes(plan)
            resolved_plan_id = _extract_plan_id(plan, plan_id)
            order = topological_order(nodes)
            return self._schedule_nodes(
                nodes,
                order=order,
                plan_id=resolved_plan_id,
                policy=active_policy,
            )
        except RepairResourceSchedulerError as exc:
            failed_plan_id = plan_id or _optional_text(
                getattr(plan, "plan_id", None) or (plan.get("plan_id") if isinstance(plan, Mapping) else None),
                "plan_id",
            ) or "plan:rejected"
            return RepairResourceSchedule(
                plan_id=failed_plan_id,
                policy_id=active_policy.policy_id,
                disposition=ScheduleDisposition.REJECTED,
                assignments=(),
                waves=(),
                path_lease_plan=PathLeasePlan(plan_id=failed_plan_id, leases=()),
                conflict_graph=(),
                critical_path=(),
                critical_path_duration_ms=0,
                budgets=active_policy.budget_counters(),
                topological_order=(),
                reason_codes=(str(exc),),
            )

    def _schedule_nodes(
        self,
        nodes: Sequence[SchedulableNode],
        *,
        order: Sequence[str],
        plan_id: str,
        policy: ResourceSchedulePolicy,
    ) -> RepairResourceSchedule:
        by_id = {node.node_id: node for node in nodes}
        order_index = {node_id: index for index, node_id in enumerate(order)}
        conflict_edges = build_conflict_graph(nodes, policy)
        hard_conflicts: dict[str, set[str]] = defaultdict(set)
        for edge in conflict_edges:
            hard_conflicts[edge.left_node_id].add(edge.right_node_id)
            hard_conflicts[edge.right_node_id].add(edge.left_node_id)

        remaining = set(by_id)
        completed: set[str] = set()
        assignments: list[ScheduledNodeAssignment] = []
        waves: list[ScheduleWave] = []
        leases: list[PathLease] = []
        lease_conflicts: list[tuple[str, str]] = []
        current_time_ms = 0
        wave_number = 0
        # Bounded progress: each outer iteration schedules at least one node
        # (the earliest ready node always fits an empty wave).
        max_iterations = len(by_id) + 1
        iterations = 0
        while remaining:
            iterations += 1
            if iterations > max_iterations:
                raise RepairResourceSchedulerError("scheduler failed to make progress")
            ready = [
                by_id[node_id]
                for node_id in sorted(
                    remaining,
                    key=lambda item: (order_index[item], item),
                )
                if all(dep in completed for dep in by_id[node_id].depends_on)
            ]
            if not ready:
                raise RepairResourceSchedulerError("deadlock: no ready nodes despite remainder")

            selected: list[SchedulableNode] = []
            for node in ready:
                if _fits_in_wave(
                    node,
                    selected,
                    policy=policy,
                    hard_conflicts=hard_conflicts,
                ):
                    selected.append(node)

            # Progress guarantee: first ready node always packs into an empty wave.
            if not selected:
                selected = [ready[0]]

            wave_duration = max(node.duration_ms for node in selected)
            usage: dict[str, int] = defaultdict(int)
            for lane, node in enumerate(selected):
                usage[node.resource_class] += 1
                if node.is_solver:
                    usage["solver_slots"] += 1
                if node.is_validation:
                    usage["validation_slots"] += 1
                if node.is_writer and node.owner_root:
                    usage[f"root:{node.owner_root}"] += 1

                fence_epoch = policy.base_fence_epoch + wave_number
                lease_id = ""
                fencing_token = ""
                if node.write_set:
                    lease_id, fencing_token = _lease_identity(
                        plan_id=plan_id,
                        node=node,
                        wave=wave_number,
                        fence_epoch=fence_epoch,
                    )
                    lease = PathLease(
                        lease_id=lease_id,
                        fencing_token=fencing_token,
                        permitted_write_paths=node.write_set,
                        owner_root=node.owner_root,
                        node_ids=(node.node_id,),
                        fence_epoch=fence_epoch,
                        lease_duration_ms=policy.lease_duration_ms,
                        heartbeat_interval_ms=policy.heartbeat_interval_ms,
                        lease_scope=f"{plan_id}:{node.node_id}",
                    )
                    # Track lease conflicts for evidence; concurrent packing already
                    # forbids overlapping writers in the same wave.
                    for prior in leases:
                        if lease.overlaps(prior):
                            lease_conflicts.append(
                                tuple(sorted((prior.lease_id, lease.lease_id)))  # type: ignore[arg-type]
                            )
                    leases.append(lease)

                # Strict sharding cannot override dependencies: shard labels are
                # advisory only after dependency readiness has been enforced.
                shard_id = (
                    node.shard_hint
                    or node.lane_hint
                    or f"shard:w{wave_number}:l{lane}"
                )
                timeout_ms = max(node.timeout_ms, policy.default_timeout_ms)
                retry_budget = (
                    node.retry_budget
                    if node.retry_budget is not None
                    else policy.default_retry_budget
                )
                # Prefer the more restrictive of node-declared and policy defaults
                # only when node used the module default and policy is tighter.
                if node.retry_budget == DEFAULT_RETRY_BUDGET:
                    retry_budget = policy.default_retry_budget
                if node.timeout_ms == DEFAULT_TIMEOUT_MS:
                    timeout_ms = policy.default_timeout_ms

                assignments.append(
                    ScheduledNodeAssignment(
                        node_id=node.node_id,
                        wave=wave_number,
                        lane=lane,
                        shard_id=shard_id,
                        resource_class=node.resource_class,
                        lease_id=lease_id,
                        fencing_token=fencing_token,
                        fence_epoch=fence_epoch,
                        timeout_ms=timeout_ms,
                        retry_budget=retry_budget,
                        validation_resource=_validation_resource_for(
                            node, wave_number, lane
                        ),
                        duration_ms=node.duration_ms,
                        start_offset_ms=current_time_ms,
                        depends_on=node.depends_on,
                        write_set=node.write_set,
                        owner_root=node.owner_root,
                        endpoints=node.endpoints,
                        exclusive_group=node.exclusive_group,
                    )
                )
                remaining.remove(node.node_id)
                completed.add(node.node_id)

            waves.append(
                ScheduleWave(
                    wave=wave_number,
                    node_ids=tuple(item.node_id for item in selected),
                    width=len(selected),
                    resource_usage=dict(usage),
                    start_offset_ms=current_time_ms,
                    duration_ms=wave_duration,
                )
            )
            current_time_ms += wave_duration
            wave_number += 1

        critical, critical_ms = _critical_path(nodes, order)
        path_lease_plan = PathLeasePlan(
            plan_id=plan_id,
            leases=tuple(leases),
            conflict_pairs=tuple(sorted(set(lease_conflicts))),
        )
        budgets = {
            **policy.budget_counters(),
            "makespan_ms": current_time_ms,
            "wave_count": len(waves),
            "assignment_count": len(assignments),
        }
        return RepairResourceSchedule(
            plan_id=plan_id,
            policy_id=policy.policy_id,
            disposition=ScheduleDisposition.SCHEDULED,
            assignments=tuple(assignments),
            waves=tuple(waves),
            path_lease_plan=path_lease_plan,
            conflict_graph=conflict_edges,
            critical_path=critical,
            critical_path_duration_ms=critical_ms,
            budgets=budgets,
            topological_order=tuple(order),
            reason_codes=(ScheduleDisposition.SCHEDULED.value,),
        )


def schedule_repair_resources(
    plan: Any,
    *,
    plan_id: str | None = None,
    policy: ResourceSchedulePolicy | Mapping[str, Any] | None = None,
) -> RepairResourceSchedule:
    """Module-level entry point for DCR-064 resource scheduling."""

    return RepairResourceScheduler(policy=policy).schedule(plan, plan_id=plan_id)


def simulate_schedule_execution(
    schedule: RepairResourceSchedule,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Simulate wave execution to prove starvation/deadlock probes terminate.

    The simulator is pure and bounded: it walks the precomputed waves in order
    and records concurrent writers. It never blocks waiting for runtime locks.
    """

    if not schedule.ok:
        return {
            "terminated": True,
            "ok": False,
            "reason": "schedule_rejected",
            "steps": 0,
            "completed": [],
            "concurrent_write_violations": [],
        }

    assignment_map = schedule.assignment_map()
    completed: list[str] = []
    concurrent_write_violations: list[dict[str, Any]] = []
    steps = 0
    bound = max_steps if max_steps is not None else max(1, len(schedule.assignments) * 4 + 8)

    for wave in schedule.waves:
        steps += 1
        if steps > bound:
            return {
                "terminated": False,
                "ok": False,
                "reason": "step_bound_exceeded",
                "steps": steps,
                "completed": completed,
                "concurrent_write_violations": concurrent_write_violations,
            }
        # Dependency readiness check (strict sharding cannot skip deps).
        for node_id in wave.node_ids:
            assignment = assignment_map[node_id]
            missing = [dep for dep in assignment.depends_on if dep not in completed]
            if missing:
                return {
                    "terminated": True,
                    "ok": False,
                    "reason": "dependency_not_ready",
                    "node_id": node_id,
                    "missing": missing,
                    "steps": steps,
                    "completed": completed,
                    "concurrent_write_violations": concurrent_write_violations,
                }
        writers = [
            (node_id, assignment_map[node_id].write_set)
            for node_id in wave.node_ids
            if assignment_map[node_id].write_set
        ]
        for i, (left_id, left_paths) in enumerate(writers):
            for right_id, right_paths in writers[i + 1 :]:
                overlap = _path_overlap_set(left_paths, right_paths)
                if overlap:
                    concurrent_write_violations.append(
                        {
                            "wave": wave.wave,
                            "left": left_id,
                            "right": right_id,
                            "paths": list(overlap),
                        }
                    )
        completed.extend(wave.node_ids)

    return {
        "terminated": True,
        "ok": not concurrent_write_violations
        and len(completed) == len(schedule.assignments),
        "reason": "completed" if not concurrent_write_violations else "write_conflict",
        "steps": steps,
        "completed": completed,
        "concurrent_write_violations": concurrent_write_violations,
        "makespan_ms": schedule.budgets.get("makespan_ms", 0),
    }


def materialize_resource_schedules(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a compact recipe-style fixture for DCR-064 validation."""

    policy = ResourceSchedulePolicy(
        policy_id="policy:dcr064-fixture",
        max_lanes=2,
        max_wave_width=2,
        max_solver_concurrency=1,
        max_validation_concurrency=1,
        max_root_writers=1,
        default_timeout_ms=120_000,
        default_retry_budget=1,
        lease_duration_ms=300_000,
        base_fence_epoch=7,
    )
    plan = {
        "plan_id": "plan:dcr064-fixture",
        "nodes": [
            {
                "node_id": "node:a",
                "kind": "operator_apply",
                "owner_root": "ipfs-accelerate",
                "write_set": [
                    "external/ipfs_accelerate/pkg/a.py",
                ],
                "resource_class": "cpu-proof-solver",
                "depends_on": [],
                "validation_ref": "validation:a",
                "duration_ms": 10_000,
            },
            {
                "node_id": "node:b",
                "kind": "operator_apply",
                "owner_root": "ipfs-datasets",
                "write_set": [
                    "external/ipfs_datasets/pkg/b.py",
                ],
                "resource_class": "cpu-medium",
                "depends_on": [],
                "validation_ref": "validation:b",
                "duration_ms": 12_000,
            },
            {
                "node_id": "node:c",
                "kind": "operator_apply",
                "owner_root": "ipfs-accelerate",
                "write_set": [
                    "external/ipfs_accelerate/pkg/a.py",
                    "external/ipfs_accelerate/pkg/c.py",
                ],
                "resource_class": "cpu-proof-solver",
                "depends_on": ["node:a"],
                "validation_ref": "validation:c",
                "duration_ms": 15_000,
            },
            {
                "node_id": "node:d",
                "kind": "validation",
                "owner_root": "ipfs-datasets",
                "write_set": [],
                "resource_class": "cpu-validation",
                "depends_on": ["node:b"],
                "validation_ref": "validation:d",
                "duration_ms": 8_000,
            },
        ],
    }
    schedule = schedule_repair_resources(plan, policy=policy)
    simulation = simulate_schedule_execution(schedule)
    payload = {
        "artifact_schema": REPAIR_RESOURCE_SCHEDULE_SCHEMA,
        "evidence_id": DCR_RESOURCE_SCHEDULE_EVIDENCE,
        "interfaces": {
            "path_lease": PATH_LEASE_INTERFACE,
            "repair_resource_schedule": REPAIR_RESOURCE_SCHEDULE_INTERFACE,
        },
        "version": REPAIR_RESOURCE_SCHEDULER_VERSION,
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "policy": policy.to_dict(),
        "plan": plan,
        "schedule": schedule.to_dict(),
        "evidence_subset": schedule.evidence_subset(),
        "simulation": simulation,
    }
    if destination is None:
        root = Path(repo_root) if repo_root else Path.cwd()
        destination = root / DEFAULT_RESOURCE_SCHEDULES_REL
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "ConflictEdge",
    "ConflictKind",
    "DCR_RESOURCE_SCHEDULE_EVIDENCE",
    "DEFAULT_RESOURCE_SCHEDULES_REL",
    "PATH_LEASE_INTERFACE",
    "PATH_LEASE_PLAN_SCHEMA",
    "PATH_LEASE_SCHEMA",
    "PathLease",
    "PathLeasePlan",
    "REPAIR_RESOURCE_SCHEDULE_INTERFACE",
    "REPAIR_RESOURCE_SCHEDULE_SCHEMA",
    "REPAIR_RESOURCE_SCHEDULER_VERSION",
    "RESOURCE_SCHEDULE_POLICY_SCHEMA",
    "RepairResourceSchedule",
    "RepairResourceScheduler",
    "RepairResourceSchedulerError",
    "ResourceSchedulePolicy",
    "SCHEDULED_NODE_SCHEMA",
    "SchedulableNode",
    "ScheduleDisposition",
    "ScheduleWave",
    "ScheduledNodeAssignment",
    "build_conflict_graph",
    "materialize_resource_schedules",
    "schedule_repair_resources",
    "simulate_schedule_execution",
    "topological_order",
]
