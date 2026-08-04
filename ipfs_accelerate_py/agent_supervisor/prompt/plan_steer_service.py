"""Revision-bound plan-steer preview (PDR-028 / ``PlanSteerService@1``).

``preview_steer`` is a **proposal-only** operation.  It:

1. loads and integrity-checks the exact base plan, task-source, run, lease,
   worktree, merge, and event-cursor snapshot;
2. partitions completed / accepted / claimed / running / settling / unstarted /
   blocked / superseded (/ failed) populations;
3. binds the current tree scan and impact evidence;
4. generates a closed, history-preserving :class:`PlanDelta`;
5. applies the delta to an in-memory copy;
6. validates lifecycle immutability, population history, graph closure bounds,
   and (optionally) full multi-gate admission; and
7. returns a body-free, read-only, restart-serializable preview receipt.

The service never writes a task source, never mutates claimed/running specs in
place, and fails closed on any stale base/root/revision/cursor/claimed/lease/
fence/policy binding.  Deferred supersession, successor tasks, and separate
lifecycle requests are explicit delta items rather than silent side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Final, Iterable

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
    PlanRevisionContractError,
    PlanRevisionLifecycleError,
    PlanRevisionStaleRootError,
    PlanSteerRequest,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    assert_delta_preserves_history,
    assert_population_history_intact,
    closed_delta_operations,
    is_history_immutable,
    plan_revision_cid,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)


PLAN_STEER_SERVICE_INTERFACE: Final[str] = "PlanSteerService@1"
PLAN_STEER_SERVICE_VERSION: Final[int] = 1
PLAN_STEER_SERVICE_CONTRACT_VERSION: Final[int] = 1

PLAN_STEER_LIVE_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-steer-live-state@1"
)
PLAN_STEER_POPULATION_PARTITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-steer-population-partition@1"
)
PLAN_STEER_SCAN_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-steer-scan-impact@1"
)
PLAN_STEER_PREVIEW_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-steer-preview-receipt@1"
)
PLAN_STEER_REJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-steer-rejection@1"
)

MAX_TASK_RECORDS: Final[int] = 4_096
MAX_GOAL_RECORDS: Final[int] = 1_024
MAX_REJECTION_REASONS: Final[int] = 256
MAX_IMPACT_PATHS: Final[int] = 4_096
MAX_RECEIPT_BYTES: Final[int] = 262_144

# Lifecycle states that may never be edited in place by a steer delta.
_RUNNING_FAMILY: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.CLAIMED,
        LifecycleState.RUNNING,
        LifecycleState.SETTLING,
    }
)
_TERMINAL_HISTORY: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.COMPLETED,
        LifecycleState.ACCEPTED,
    }
)
_IMMUTABLE_STATES: Final[frozenset[LifecycleState]] = (
    _RUNNING_FAMILY | _TERMINAL_HISTORY
)

# Ops that may name an immutable target only as a non-mutating effect.
_SAFE_ON_IMMUTABLE: Final[frozenset[PlanDeltaOperation]] = frozenset(
    {
        PlanDeltaOperation.ADD_TASK,
        PlanDeltaOperation.ADD_GOAL,
        PlanDeltaOperation.ATTACH_EVIDENCE,
        PlanDeltaOperation.RECORD_UNCERTAINTY,
        PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
        PlanDeltaOperation.BLOCK_UNSTARTED_TASK,
        PlanDeltaOperation.UNBLOCK_TASK,
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "source_body",
        "source_text",
        "transcript",
        "password",
        "secret",
        "token",
        "api_key",
        "private_key",
        "credential",
        "access_token",
        "session_token",
        "authorization",
        "cookie",
    }
)


# ---------------------------------------------------------------------------
# Errors / codes
# ---------------------------------------------------------------------------


class PlanSteerServiceError(ValueError):
    """Base error for the steer preview service."""

    def __init__(
        self,
        message: str,
        *,
        code: "PlanSteerRejectionCode | str | None" = None,
    ) -> None:
        super().__init__(message)
        if code is None:
            self.code: str = PlanSteerRejectionCode.SERVICE_ERROR.value
        elif isinstance(code, PlanSteerRejectionCode):
            self.code = code.value
        else:
            self.code = str(code)


class PlanSteerStaleError(PlanSteerServiceError):
    """A bound root, cursor, claim, lease, fence, or policy is stale."""


class PlanSteerLifecycleError(PlanSteerServiceError):
    """A delta would edit immutable or running history."""


class PlanSteerAdmissionError(PlanSteerServiceError):
    """The resulting plan failed admission or validation."""


class PlanSteerVerdict(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


class PlanSteerRejectionCode(str, Enum):
    SERVICE_ERROR = "service_error"
    MALFORMED_REQUEST = "malformed_request"
    MALFORMED_STATE = "malformed_state"
    STALE_BASE = "stale_base"
    STALE_ROOT = "stale_root"
    STALE_REVISION = "stale_revision"
    STALE_CURSOR = "stale_cursor"
    STALE_CLAIMED = "stale_claimed"
    STALE_ACCEPTED = "stale_accepted"
    STALE_LEASE = "stale_lease"
    STALE_FENCE = "stale_fence"
    STALE_POLICY = "stale_policy"
    STALE_RUN = "stale_run"
    STALE_WORKTREE = "stale_worktree"
    STALE_MERGE = "stale_merge"
    STALE_SCAN = "stale_scan"
    POPULATION_INTEGRITY = "population_integrity"
    LIFECYCLE_VIOLATION = "lifecycle_violation"
    RUNNING_EDIT = "running_edit"
    FORBIDDEN_OPERATION = "forbidden_operation"
    AFFECTED_POPULATION_EXCEEDED = "affected_population_exceeded"
    EMPTY_DELTA = "empty_delta"
    INVALID_DELTA = "invalid_delta"
    HISTORY_SHRINK = "history_shrink"
    ADMISSION_FAILED = "admission_failed"
    BODY_PRESENT = "body_present"
    WRITE_ATTEMPTED = "write_attempted"
    MISSING_BASE_PLAN = "missing_base_plan"
    MISSING_SCAN = "missing_scan"
    QUERY_FAILED = "query_failed"


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise PlanSteerServiceError(
            "steer value exceeds depth bound",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, bool) or not isinstance(value, int):
            return value
        return value
    if isinstance(value, float):
        raise PlanSteerServiceError(
            "floating point values are not canonical steer data",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PlanSteerServiceError(
                "steer mapping keys must be strings",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        return {
            key: _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _plain(converter(), depth=depth + 1)
    raise PlanSteerServiceError(
        f"unsupported steer value: {type(value).__name__}",
        code=PlanSteerRejectionCode.MALFORMED_STATE,
    )


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise PlanSteerServiceError(
            f"{field_name} may not contain floating-point values",
            code=PlanSteerRejectionCode.BODY_PRESENT,
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanSteerServiceError(
                    f"{field_name} has a non-string key",
                    code=PlanSteerRejectionCode.BODY_PRESENT,
                )
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or any(
                marker in normalized for marker in ("password", "private_key", "api_key")
            ):
                raise PlanSteerServiceError(
                    f"{field_name} may not contain secrets or source bodies",
                    code=PlanSteerRejectionCode.BODY_PRESENT,
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise PlanSteerServiceError(
            f"{field_name} may not contain binary bodies",
            code=PlanSteerRejectionCode.BODY_PRESENT,
        )


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise PlanSteerServiceError(
            f"{name} must be a string",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    if value != value.strip() or "\x00" in value:
        raise PlanSteerServiceError(
            f"{name} must not contain surrounding whitespace or NUL",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    if required and not value:
        raise PlanSteerServiceError(
            f"{name} is required",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    return value


def _optional_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    return _text(value, name, required=False)


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanSteerServiceError(
            f"{name} must be an integer",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    if value < minimum:
        raise PlanSteerServiceError(
            f"{name} is below the supported minimum",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlanSteerServiceError(
            f"{name} must be boolean",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    return value


def _strings(
    value: Any,
    name: str,
    *,
    limit: int = MAX_IMPACT_PATHS,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanSteerServiceError(
            f"{name} must be a sequence of strings",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    else:
        raw = value
    if len(raw) > limit:
        raise PlanSteerServiceError(
            f"{name} exceeds its item bound",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    result = tuple(sorted({_text(item, name) for item in raw}))
    if required and not result:
        raise PlanSteerServiceError(
            f"{name} must not be empty",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        )
    return result


def _freeze(value: Any) -> Any:
    plain = _plain(value)
    if isinstance(plain, dict):
        return MappingProxyType({key: _freeze(item) for key, item in plain.items()})
    if isinstance(plain, list):
        return tuple(_freeze(item) for item in plain)
    return plain


def _lifecycle(value: Any, name: str = "lifecycle_state") -> LifecycleState:
    if isinstance(value, LifecycleState):
        return value
    text = _text(value, name).casefold().replace("-", "_")
    aliases = {
        "ready": LifecycleState.UNSTARTED,
        "pending": LifecycleState.UNSTARTED,
        "todo": LifecycleState.UNSTARTED,
        "open": LifecycleState.UNSTARTED,
        "in_progress": LifecycleState.RUNNING,
        "active": LifecycleState.RUNNING,
        "done": LifecycleState.COMPLETED,
        "success": LifecycleState.COMPLETED,
        "passed": LifecycleState.ACCEPTED,
        "cancelled": LifecycleState.CANCELLED,
        "canceled": LifecycleState.CANCELLED,
    }
    if text in aliases:
        return aliases[text]
    try:
        return LifecycleState(text)
    except ValueError as exc:
        raise PlanSteerServiceError(
            f"{name} has an unsupported lifecycle state {value!r}",
            code=PlanSteerRejectionCode.MALFORMED_STATE,
        ) from exc


def _population(
    kind: PopulationKind, members: Iterable[str]
) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=tuple(sorted(set(members))))


def _cid_or_identity(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required)
    if not text and not required:
        return ""
    return text


def _record_cid(record: Mapping[str, Any] | Any, *, default_name: str) -> str:
    if isinstance(record, Mapping):
        for key in (
            "task_cid",
            "goal_cid",
            "content_id",
            "cid",
            "record_cid",
            "id",
        ):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return plan_revision_cid({"namespace": default_name, "record": _plain(record)})
    for attr in (
        "task_cid",
        "goal_cid",
        "content_id",
        "cid",
        "record_cid",
    ):
        value = getattr(record, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    converter = getattr(record, "to_dict", None)
    if callable(converter):
        return _record_cid(converter(), default_name=default_name)
    raise PlanSteerServiceError(
        f"{default_name} record is missing a content identity",
        code=PlanSteerRejectionCode.MALFORMED_STATE,
    )


def _record_lifecycle(record: Mapping[str, Any] | Any) -> LifecycleState:
    if isinstance(record, Mapping):
        for key in (
            "lifecycle_state",
            "lifecycle",
            "state",
            "status",
            "task_status",
        ):
            if key in record and record[key] not in (None, ""):
                return _lifecycle(record[key], key)
        return LifecycleState.UNSTARTED
    for attr in (
        "lifecycle_state",
        "lifecycle",
        "state",
        "status",
        "task_status",
    ):
        value = getattr(record, attr, None)
        if value not in (None, ""):
            return _lifecycle(value, attr)
    return LifecycleState.UNSTARTED


def _record_spec_revision(record: Mapping[str, Any] | Any) -> str:
    if isinstance(record, Mapping):
        for key in (
            "spec_revision",
            "expected_target_spec_revision",
            "revision",
            "content_id",
        ):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for attr in (
        "spec_revision",
        "expected_target_spec_revision",
        "revision",
        "content_id",
    ):
        value = getattr(record, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ---------------------------------------------------------------------------
# Live state / partitions / scan impact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanSteerTaskRecord:
    """Body-free task observation used for population partitioning."""

    task_cid: str
    lifecycle_state: LifecycleState
    spec_revision: str = ""
    goal_cid: str = ""
    attempt_id: str = ""
    lease_id: str = ""
    worktree_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_cid", _cid_or_identity(self.task_cid, "task_cid")
        )
        object.__setattr__(
            self,
            "lifecycle_state",
            _lifecycle(self.lifecycle_state, "lifecycle_state"),
        )
        object.__setattr__(
            self,
            "spec_revision",
            _optional_text(self.spec_revision, "spec_revision"),
        )
        object.__setattr__(
            self, "goal_cid", _optional_text(self.goal_cid, "goal_cid")
        )
        object.__setattr__(
            self, "attempt_id", _optional_text(self.attempt_id, "attempt_id")
        )
        object.__setattr__(
            self, "lease_id", _optional_text(self.lease_id, "lease_id")
        )
        object.__setattr__(
            self, "worktree_id", _optional_text(self.worktree_id, "worktree_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "lifecycle_state": self.lifecycle_state.value,
            "spec_revision": self.spec_revision,
            "goal_cid": self.goal_cid,
            "attempt_id": self.attempt_id,
            "lease_id": self.lease_id,
            "worktree_id": self.worktree_id,
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Any) -> "PlanSteerTaskRecord":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            converter = getattr(value, "to_dict", None)
            if callable(converter):
                value = converter()
            else:
                value = {
                    "task_cid": _record_cid(value, default_name="task"),
                    "lifecycle_state": _record_lifecycle(value).value,
                    "spec_revision": _record_spec_revision(value),
                }
        assert isinstance(value, Mapping)
        return cls(
            task_cid=_record_cid(value, default_name="task"),
            lifecycle_state=_record_lifecycle(value),
            spec_revision=_optional_text(
                value.get("spec_revision")
                or value.get("expected_target_spec_revision")
                or "",
                "spec_revision",
            ),
            goal_cid=_optional_text(value.get("goal_cid", ""), "goal_cid"),
            attempt_id=_optional_text(
                value.get("attempt_id") or value.get("run_attempt_id") or "",
                "attempt_id",
            ),
            lease_id=_optional_text(value.get("lease_id", ""), "lease_id"),
            worktree_id=_optional_text(
                value.get("worktree_id") or value.get("worktree_root") or "",
                "worktree_id",
            ),
        )


@dataclass(frozen=True)
class PlanSteerGoalRecord:
    """Body-free goal observation used for population partitioning."""

    goal_cid: str
    lifecycle_state: LifecycleState
    spec_revision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "goal_cid", _cid_or_identity(self.goal_cid, "goal_cid")
        )
        object.__setattr__(
            self,
            "lifecycle_state",
            _lifecycle(self.lifecycle_state, "lifecycle_state"),
        )
        object.__setattr__(
            self,
            "spec_revision",
            _optional_text(self.spec_revision, "spec_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_cid": self.goal_cid,
            "lifecycle_state": self.lifecycle_state.value,
            "spec_revision": self.spec_revision,
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Any) -> "PlanSteerGoalRecord":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            converter = getattr(value, "to_dict", None)
            if callable(converter):
                value = converter()
            else:
                value = {
                    "goal_cid": _record_cid(value, default_name="goal"),
                    "lifecycle_state": _record_lifecycle(value).value,
                    "spec_revision": _record_spec_revision(value),
                }
        assert isinstance(value, Mapping)
        return cls(
            goal_cid=_record_cid(value, default_name="goal"),
            lifecycle_state=_record_lifecycle(value),
            spec_revision=_optional_text(
                value.get("spec_revision", ""), "spec_revision"
            ),
        )


@dataclass(frozen=True)
class PlanSteerPopulationPartition(CanonicalContract):
    """Closed partition of the live task population for one steer preview."""

    SCHEMA: ClassVar[str] = PLAN_STEER_POPULATION_PARTITION_SCHEMA

    completed: PlanPopulationDigest
    accepted: PlanPopulationDigest
    claimed: PlanPopulationDigest
    running: PlanPopulationDigest
    settling: PlanPopulationDigest
    unstarted: PlanPopulationDigest
    blocked: PlanPopulationDigest
    superseded: PlanPopulationDigest
    failed: PlanPopulationDigest = field(
        default_factory=lambda: PlanPopulationDigest(
            kind=PopulationKind.FAILED, member_cids=()
        )
    )

    def __post_init__(self) -> None:
        expected = {
            "completed": PopulationKind.COMPLETED,
            "accepted": PopulationKind.ACCEPTED,
            "claimed": PopulationKind.CLAIMED,
            "running": PopulationKind.RUNNING,
            "settling": PopulationKind.SETTLING,
            "unstarted": PopulationKind.UNSTARTED,
            "blocked": PopulationKind.BLOCKED,
            "superseded": PopulationKind.SUPERSEDED,
            "failed": PopulationKind.FAILED,
        }
        for name, kind in expected.items():
            pop = getattr(self, name)
            if isinstance(pop, PlanPopulationDigest):
                decoded = pop
            elif isinstance(pop, Mapping):
                decoded = PlanPopulationDigest.from_dict(pop) if "schema" in pop else PlanPopulationDigest(**pop)
            else:
                raise PlanSteerServiceError(
                    f"{name} must be a PlanPopulationDigest",
                    code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
                )
            if decoded.kind is not kind:
                raise PlanSteerServiceError(
                    f"{name}.kind must be {kind.value}",
                    code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
                )
            object.__setattr__(self, name, decoded)
        # Populations must be disjoint across the closed partition.
        seen: dict[str, str] = {}
        for name in expected:
            pop = getattr(self, name)
            assert isinstance(pop, PlanPopulationDigest)
            for member in pop.member_cids:
                prior = seen.get(member)
                if prior is not None:
                    raise PlanSteerServiceError(
                        f"task {member} appears in both {prior} and {name}",
                        code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
                    )
                seen[member] = name
        _assert_body_free(self._payload(), "population partition")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_STEER_SERVICE_CONTRACT_VERSION,
            "completed": self.completed.to_dict(),
            "accepted": self.accepted.to_dict(),
            "claimed": self.claimed.to_dict(),
            "running": self.running.to_dict(),
            "settling": self.settling.to_dict(),
            "unstarted": self.unstarted.to_dict(),
            "blocked": self.blocked.to_dict(),
            "superseded": self.superseded.to_dict(),
            "failed": self.failed.to_dict(),
        }

    @property
    def partition_cid(self) -> str:
        return self.content_id

    @property
    def claimed_family_digest(self) -> str:
        """Digest covering claimed + running + settling (request claimed set)."""

        members = (
            set(self.claimed.member_cids)
            | set(self.running.member_cids)
            | set(self.settling.member_cids)
        )
        return _population(PopulationKind.CLAIMED, members).digest

    @property
    def accepted_or_completed_digest(self) -> str:
        members = set(self.accepted.member_cids) | set(self.completed.member_cids)
        return _population(PopulationKind.ACCEPTED, members).digest

    def lifecycle_of(self, task_cid: str) -> LifecycleState | None:
        mapping = (
            (self.completed, LifecycleState.COMPLETED),
            (self.accepted, LifecycleState.ACCEPTED),
            (self.claimed, LifecycleState.CLAIMED),
            (self.running, LifecycleState.RUNNING),
            (self.settling, LifecycleState.SETTLING),
            (self.unstarted, LifecycleState.UNSTARTED),
            (self.blocked, LifecycleState.BLOCKED),
            (self.superseded, LifecycleState.SUPERSEDED),
            (self.failed, LifecycleState.FAILED),
        )
        for pop, state in mapping:
            if task_cid in pop.member_cids:
                return state
        return None

    def all_member_cids(self) -> frozenset[str]:
        members: set[str] = set()
        for name in (
            "completed",
            "accepted",
            "claimed",
            "running",
            "settling",
            "unstarted",
            "blocked",
            "superseded",
            "failed",
        ):
            pop = getattr(self, name)
            members.update(pop.member_cids)
        return frozenset(members)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerPopulationPartition":
        if not isinstance(payload, Mapping):
            raise PlanSteerServiceError(
                "population partition payload must be a mapping",
                code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise PlanSteerServiceError(
                "unsupported population partition schema",
                code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
            )
        names = (
            "completed",
            "accepted",
            "claimed",
            "running",
            "settling",
            "unstarted",
            "blocked",
            "superseded",
            "failed",
        )
        values = {name: payload[name] for name in names if name in payload}
        value = cls(**values)
        supplied = payload.get("content_id") or payload.get("cid")
        if supplied not in (None, "") and supplied != value.content_id:
            raise PlanSteerServiceError(
                "population partition identity does not match content",
                code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
            )
        return value


@dataclass(frozen=True)
class PlanSteerScanImpact(CanonicalContract):
    """Body-free current-tree scan + impact binding for a steer preview."""

    SCHEMA: ClassVar[str] = PLAN_STEER_SCAN_IMPACT_SCHEMA

    scan_receipt_cid: str
    repository_root_cid: str
    dirty_worktree_root: str
    base_plan_root: str
    impacted_paths: tuple[str, ...] = ()
    impacted_symbols: tuple[str, ...] = ()
    added_paths: tuple[str, ...] = ()
    modified_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    renamed_paths: tuple[str, ...] = ()
    policy_admitted_untracked_paths: tuple[str, ...] = ()
    taskboard_drift_refs: tuple[str, ...] = ()
    accepted_output_drift_refs: tuple[str, ...] = ()
    truncation_refs: tuple[str, ...] = ()
    instability_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "scan_receipt_cid",
            "repository_root_cid",
            "dirty_worktree_root",
            "base_plan_root",
        ):
            object.__setattr__(
                self, name, _cid_or_identity(getattr(self, name), name)
            )
        for name in (
            "impacted_paths",
            "added_paths",
            "modified_paths",
            "deleted_paths",
            "renamed_paths",
            "policy_admitted_untracked_paths",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name)
            )
        for name in (
            "impacted_symbols",
            "taskboard_drift_refs",
            "accepted_output_drift_refs",
            "truncation_refs",
            "instability_refs",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name)
            )
        _assert_body_free(self._payload(), "scan impact")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_STEER_SERVICE_CONTRACT_VERSION,
            "scan_receipt_cid": self.scan_receipt_cid,
            "repository_root_cid": self.repository_root_cid,
            "dirty_worktree_root": self.dirty_worktree_root,
            "base_plan_root": self.base_plan_root,
            "impacted_paths": list(self.impacted_paths),
            "impacted_symbols": list(self.impacted_symbols),
            "added_paths": list(self.added_paths),
            "modified_paths": list(self.modified_paths),
            "deleted_paths": list(self.deleted_paths),
            "renamed_paths": list(self.renamed_paths),
            "policy_admitted_untracked_paths": list(
                self.policy_admitted_untracked_paths
            ),
            "taskboard_drift_refs": list(self.taskboard_drift_refs),
            "accepted_output_drift_refs": list(self.accepted_output_drift_refs),
            "truncation_refs": list(self.truncation_refs),
            "instability_refs": list(self.instability_refs),
        }

    @property
    def impact_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerScanImpact":
        if not isinstance(payload, Mapping):
            raise PlanSteerServiceError(
                "scan impact payload must be a mapping",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise PlanSteerServiceError(
                "unsupported scan impact schema",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        names = (
            "scan_receipt_cid",
            "repository_root_cid",
            "dirty_worktree_root",
            "base_plan_root",
            "impacted_paths",
            "impacted_symbols",
            "added_paths",
            "modified_paths",
            "deleted_paths",
            "renamed_paths",
            "policy_admitted_untracked_paths",
            "taskboard_drift_refs",
            "accepted_output_drift_refs",
            "truncation_refs",
            "instability_refs",
        )
        values = {name: payload[name] for name in names if name in payload}
        value = cls(**values)
        supplied = payload.get("content_id") or payload.get("cid")
        if supplied not in (None, "") and supplied != value.content_id:
            raise PlanSteerServiceError(
                "scan impact identity does not match content",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        return value


@dataclass(frozen=True)
class PlanSteerLiveState(CanonicalContract):
    """Exact plan/task/run/lease/worktree/merge snapshot for integrity checks."""

    SCHEMA: ClassVar[str] = PLAN_STEER_LIVE_STATE_SCHEMA

    current_roots: PlanAuthorityRoots
    plan_revision: int
    event_cursor: str
    accepted_evidence_root: str
    base_plan_root: str
    base_admitted_plan_root: str = ""
    completion_revision: str = ""
    supervisor_run_id: str = ""
    supervisor_state_revision: str = ""
    lease_id: str = ""
    fencing_epoch: int = 0
    tasks: tuple[PlanSteerTaskRecord, ...] = ()
    goals: tuple[PlanSteerGoalRecord, ...] = ()
    lease_contract: PlanLeaseContract = field(default_factory=PlanLeaseContract)
    worktree_contract: PlanWorktreeContract = field(
        default_factory=PlanWorktreeContract
    )
    merge_strategy: PlanMergeStrategy = field(
        default_factory=lambda: PlanMergeStrategy(kind=MergeStrategyKind.SERIAL)
    )
    base_plan: PlanRevision | None = None
    scan: Mapping[str, Any] = field(default_factory=dict)
    impact: Mapping[str, Any] = field(default_factory=dict)
    merge_state: Mapping[str, Any] = field(default_factory=dict)
    run_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        roots = self.current_roots
        if isinstance(roots, Mapping):
            roots = (
                PlanAuthorityRoots.from_dict(roots)
                if "schema" in roots
                else PlanAuthorityRoots(**roots)
            )
        elif not isinstance(roots, PlanAuthorityRoots):
            raise PlanSteerServiceError(
                "current_roots must be PlanAuthorityRoots",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "current_roots", roots)
        object.__setattr__(
            self,
            "plan_revision",
            _int(self.plan_revision, "plan_revision", minimum=1),
        )
        object.__setattr__(
            self, "event_cursor", _cid_or_identity(self.event_cursor, "event_cursor")
        )
        object.__setattr__(
            self,
            "accepted_evidence_root",
            _cid_or_identity(self.accepted_evidence_root, "accepted_evidence_root"),
        )
        object.__setattr__(
            self,
            "base_plan_root",
            _cid_or_identity(self.base_plan_root, "base_plan_root"),
        )
        object.__setattr__(
            self,
            "base_admitted_plan_root",
            _cid_or_identity(
                self.base_admitted_plan_root,
                "base_admitted_plan_root",
                required=False,
            ),
        )
        for name in (
            "completion_revision",
            "supervisor_run_id",
            "supervisor_state_revision",
            "lease_id",
        ):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "fencing_epoch",
            _int(self.fencing_epoch, "fencing_epoch", minimum=0),
        )

        tasks_raw = self.tasks
        if tasks_raw is None:
            tasks_seq: Sequence[Any] = ()
        elif isinstance(tasks_raw, Sequence) and not isinstance(
            tasks_raw, (str, bytes, bytearray)
        ):
            tasks_seq = tasks_raw
        else:
            raise PlanSteerServiceError(
                "tasks must be a sequence",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if len(tasks_seq) > MAX_TASK_RECORDS:
            raise PlanSteerServiceError(
                "tasks exceeds its item bound",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        tasks = tuple(
            sorted(
                (PlanSteerTaskRecord.from_value(item) for item in tasks_seq),
                key=lambda item: item.task_cid,
            )
        )
        seen_tasks: set[str] = set()
        for task in tasks:
            if task.task_cid in seen_tasks:
                raise PlanSteerServiceError(
                    "duplicate task_cid in live state",
                    code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
                )
            seen_tasks.add(task.task_cid)
        object.__setattr__(self, "tasks", tasks)

        goals_raw = self.goals
        if goals_raw is None:
            goals_seq: Sequence[Any] = ()
        elif isinstance(goals_raw, Sequence) and not isinstance(
            goals_raw, (str, bytes, bytearray)
        ):
            goals_seq = goals_raw
        else:
            raise PlanSteerServiceError(
                "goals must be a sequence",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if len(goals_seq) > MAX_GOAL_RECORDS:
            raise PlanSteerServiceError(
                "goals exceeds its item bound",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        goals = tuple(
            sorted(
                (PlanSteerGoalRecord.from_value(item) for item in goals_seq),
                key=lambda item: item.goal_cid,
            )
        )
        object.__setattr__(self, "goals", goals)

        lease = self.lease_contract
        if isinstance(lease, Mapping):
            lease = (
                PlanLeaseContract.from_dict(lease)
                if "schema" in lease
                else PlanLeaseContract(**lease)
            )
        elif not isinstance(lease, PlanLeaseContract):
            raise PlanSteerServiceError(
                "lease_contract must be PlanLeaseContract",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "lease_contract", lease)

        worktree = self.worktree_contract
        if isinstance(worktree, Mapping):
            worktree = (
                PlanWorktreeContract.from_dict(worktree)
                if "schema" in worktree
                else PlanWorktreeContract(**worktree)
            )
        elif not isinstance(worktree, PlanWorktreeContract):
            raise PlanSteerServiceError(
                "worktree_contract must be PlanWorktreeContract",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "worktree_contract", worktree)

        merge = self.merge_strategy
        if isinstance(merge, Mapping):
            merge = (
                PlanMergeStrategy.from_dict(merge)
                if "schema" in merge
                else PlanMergeStrategy(**merge)
            )
        elif not isinstance(merge, PlanMergeStrategy):
            raise PlanSteerServiceError(
                "merge_strategy must be PlanMergeStrategy",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "merge_strategy", merge)

        if self.base_plan is not None and not isinstance(
            self.base_plan, PlanRevision
        ):
            if isinstance(self.base_plan, Mapping):
                object.__setattr__(
                    self,
                    "base_plan",
                    PlanRevision.from_dict(self.base_plan)
                    if "schema" in self.base_plan
                    else PlanRevision(**self.base_plan),
                )
            else:
                raise PlanSteerServiceError(
                    "base_plan must be PlanRevision when provided",
                    code=PlanSteerRejectionCode.MISSING_BASE_PLAN,
                )

        object.__setattr__(self, "scan", _freeze(self.scan or {}))
        object.__setattr__(self, "impact", _freeze(self.impact or {}))
        object.__setattr__(self, "merge_state", _freeze(self.merge_state or {}))
        object.__setattr__(self, "run_state", _freeze(self.run_state or {}))
        _assert_body_free(self._payload(), "plan steer live state")

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": PLAN_STEER_SERVICE_CONTRACT_VERSION,
            "current_roots": self.current_roots.to_dict(),
            "plan_revision": self.plan_revision,
            "event_cursor": self.event_cursor,
            "accepted_evidence_root": self.accepted_evidence_root,
            "base_plan_root": self.base_plan_root,
            "base_admitted_plan_root": self.base_admitted_plan_root,
            "completion_revision": self.completion_revision,
            "supervisor_run_id": self.supervisor_run_id,
            "supervisor_state_revision": self.supervisor_state_revision,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "tasks": [task.to_dict() for task in self.tasks],
            "goals": [goal.to_dict() for goal in self.goals],
            "lease_contract": self.lease_contract.to_dict(),
            "worktree_contract": self.worktree_contract.to_dict(),
            "merge_strategy": self.merge_strategy.to_dict(),
            "scan": dict(self.scan),
            "impact": dict(self.impact),
            "merge_state": dict(self.merge_state),
            "run_state": dict(self.run_state),
        }
        if self.base_plan is not None:
            payload["base_plan"] = self.base_plan.to_dict()
        return payload

    @property
    def state_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerLiveState":
        if not isinstance(payload, Mapping):
            raise PlanSteerServiceError(
                "live state payload must be a mapping",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise PlanSteerServiceError(
                "unsupported live state schema",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        names = (
            "current_roots",
            "plan_revision",
            "event_cursor",
            "accepted_evidence_root",
            "base_plan_root",
            "base_admitted_plan_root",
            "completion_revision",
            "supervisor_run_id",
            "supervisor_state_revision",
            "lease_id",
            "fencing_epoch",
            "tasks",
            "goals",
            "lease_contract",
            "worktree_contract",
            "merge_strategy",
            "base_plan",
            "scan",
            "impact",
            "merge_state",
            "run_state",
        )
        values = {name: payload[name] for name in names if name in payload}
        value = cls(**values)
        supplied = payload.get("content_id") or payload.get("cid")
        if supplied not in (None, "") and supplied != value.content_id:
            raise PlanSteerServiceError(
                "live state identity does not match content",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        return value


# ---------------------------------------------------------------------------
# Rejection / preview receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanSteerRejection(CanonicalContract):
    """One typed rejection reason attached to a steer preview."""

    SCHEMA: ClassVar[str] = PLAN_STEER_REJECTION_SCHEMA

    code: PlanSteerRejectionCode
    message: str
    detail_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.code, PlanSteerRejectionCode):
            code = self.code
        else:
            try:
                code = PlanSteerRejectionCode(str(self.code))
            except ValueError as exc:
                raise PlanSteerServiceError(
                    f"unknown rejection code {self.code!r}",
                    code=PlanSteerRejectionCode.SERVICE_ERROR,
                ) from exc
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(
            self, "detail_ids", _strings(self.detail_ids, "detail_ids")
        )
        _assert_body_free(self._payload(), "steer rejection")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_STEER_SERVICE_CONTRACT_VERSION,
            "code": self.code.value,
            "message": self.message,
            "detail_ids": list(self.detail_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerRejection":
        if not isinstance(payload, Mapping):
            raise PlanSteerServiceError(
                "rejection payload must be a mapping",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise PlanSteerServiceError(
                "unsupported rejection schema",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        return cls(
            code=payload.get("code", PlanSteerRejectionCode.SERVICE_ERROR),
            message=payload.get("message", "unknown rejection"),
            detail_ids=tuple(payload.get("detail_ids") or ()),
        )


@dataclass(frozen=True)
class PlanSteerPreviewReceipt(CanonicalContract):
    """Body-free, read-only, restart-serializable steer preview receipt."""

    SCHEMA: ClassVar[str] = PLAN_STEER_PREVIEW_RECEIPT_SCHEMA

    request_cid: str
    base_plan_root: str
    base_plan_revision: int
    base_admitted_plan_root: str
    current_roots: PlanAuthorityRoots
    population_partition: PlanSteerPopulationPartition
    scan_impact: PlanSteerScanImpact
    event_cursor: str
    claimed_population_digest: str
    accepted_population_digest: str
    accepted_evidence_root: str
    lease_id: str
    fencing_epoch: int
    verdict: PlanSteerVerdict
    read_only: bool = True
    wrote_task_source: bool = False
    restart_serializable: bool = True
    delta_cid: str = ""
    candidate_plan_root: str = ""
    candidate_plan_revision: int = 0
    scan_receipt_cid: str = ""
    query_plan_cid: str = ""
    evidence_bundle_cid: str = ""
    admission_receipt_cid: str = ""
    execution_plan_cid: str = ""
    state_cid: str = ""
    expected_effects: tuple[str, ...] = ()
    deferred_item_keys: tuple[str, ...] = ()
    successor_item_keys: tuple[str, ...] = ()
    lifecycle_request_item_keys: tuple[str, ...] = ()
    materializable_item_keys: tuple[str, ...] = ()
    rejection_reasons: tuple[PlanSteerRejection, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    service_interface: str = PLAN_STEER_SERVICE_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_cid", _cid_or_identity(self.request_cid, "request_cid")
        )
        object.__setattr__(
            self,
            "base_plan_root",
            _cid_or_identity(self.base_plan_root, "base_plan_root"),
        )
        object.__setattr__(
            self,
            "base_plan_revision",
            _int(self.base_plan_revision, "base_plan_revision", minimum=1),
        )
        object.__setattr__(
            self,
            "base_admitted_plan_root",
            _cid_or_identity(
                self.base_admitted_plan_root,
                "base_admitted_plan_root",
                required=False,
            ),
        )
        roots = self.current_roots
        if isinstance(roots, Mapping):
            roots = (
                PlanAuthorityRoots.from_dict(roots)
                if "schema" in roots
                else PlanAuthorityRoots(**roots)
            )
        elif not isinstance(roots, PlanAuthorityRoots):
            raise PlanSteerServiceError(
                "current_roots must be PlanAuthorityRoots",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "current_roots", roots)

        partition = self.population_partition
        if isinstance(partition, Mapping):
            partition = PlanSteerPopulationPartition.from_dict(partition)
        elif not isinstance(partition, PlanSteerPopulationPartition):
            raise PlanSteerServiceError(
                "population_partition must be PlanSteerPopulationPartition",
                code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
            )
        object.__setattr__(self, "population_partition", partition)

        impact = self.scan_impact
        if isinstance(impact, Mapping):
            impact = PlanSteerScanImpact.from_dict(impact)
        elif not isinstance(impact, PlanSteerScanImpact):
            raise PlanSteerServiceError(
                "scan_impact must be PlanSteerScanImpact",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        object.__setattr__(self, "scan_impact", impact)

        for name in (
            "event_cursor",
            "claimed_population_digest",
            "accepted_population_digest",
            "accepted_evidence_root",
        ):
            object.__setattr__(
                self, name, _cid_or_identity(getattr(self, name), name)
            )
        object.__setattr__(
            self, "lease_id", _optional_text(self.lease_id, "lease_id")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _int(self.fencing_epoch, "fencing_epoch", minimum=0),
        )
        if isinstance(self.verdict, PlanSteerVerdict):
            verdict = self.verdict
        else:
            verdict = PlanSteerVerdict(str(self.verdict))
        object.__setattr__(self, "verdict", verdict)
        for name in ("read_only", "wrote_task_source", "restart_serializable"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if not self.read_only:
            raise PlanSteerServiceError(
                "steer preview receipts must be read_only",
                code=PlanSteerRejectionCode.WRITE_ATTEMPTED,
            )
        if self.wrote_task_source:
            raise PlanSteerServiceError(
                "steer preview must not write the task source",
                code=PlanSteerRejectionCode.WRITE_ATTEMPTED,
            )
        if not self.restart_serializable:
            raise PlanSteerServiceError(
                "steer preview receipts must be restart_serializable",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        for name in (
            "delta_cid",
            "candidate_plan_root",
            "scan_receipt_cid",
            "query_plan_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
            "state_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid_or_identity(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "candidate_plan_revision",
            _int(
                self.candidate_plan_revision,
                "candidate_plan_revision",
                minimum=0,
            ),
        )
        for name in (
            "expected_effects",
            "deferred_item_keys",
            "successor_item_keys",
            "lifecycle_request_item_keys",
            "materializable_item_keys",
            "artifact_refs",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), name)
            )
        raw_reasons = self.rejection_reasons
        if raw_reasons is None:
            reasons_seq: Sequence[Any] = ()
        elif isinstance(raw_reasons, Sequence) and not isinstance(
            raw_reasons, (str, bytes, bytearray)
        ):
            reasons_seq = raw_reasons
        else:
            raise PlanSteerServiceError(
                "rejection_reasons must be a sequence",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if len(reasons_seq) > MAX_REJECTION_REASONS:
            raise PlanSteerServiceError(
                "rejection_reasons exceeds its item bound",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        reasons: list[PlanSteerRejection] = []
        for item in reasons_seq:
            if isinstance(item, PlanSteerRejection):
                reasons.append(item)
            elif isinstance(item, Mapping):
                reasons.append(PlanSteerRejection.from_dict(item))
            else:
                raise PlanSteerServiceError(
                    "rejection_reasons items must be PlanSteerRejection",
                    code=PlanSteerRejectionCode.MALFORMED_STATE,
                )
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(sorted(reasons, key=lambda item: item.content_id)),
        )
        object.__setattr__(
            self,
            "service_interface",
            _text(self.service_interface, "service_interface"),
        )
        if self.service_interface != PLAN_STEER_SERVICE_INTERFACE:
            raise PlanSteerServiceError(
                "service_interface must be PlanSteerService@1",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if self.verdict is PlanSteerVerdict.ADMITTED and not self.delta_cid:
            raise PlanSteerServiceError(
                "admitted steer preview requires a delta_cid",
                code=PlanSteerRejectionCode.INVALID_DELTA,
            )
        if self.verdict is PlanSteerVerdict.REJECTED and not self.rejection_reasons:
            raise PlanSteerServiceError(
                "rejected steer preview requires rejection_reasons",
                code=PlanSteerRejectionCode.SERVICE_ERROR,
            )
        payload = self._payload()
        _assert_body_free(payload, "plan steer preview receipt")
        encoded = content_identity(payload)  # force identity construction
        del encoded
        # Bound the durable receipt size.
        import json

        if len(json.dumps(payload, separators=(",", ":"), sort_keys=True)) > MAX_RECEIPT_BYTES:
            raise PlanSteerServiceError(
                "preview receipt exceeds its serialized byte bound",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PLAN_STEER_SERVICE_CONTRACT_VERSION,
            "service_interface": self.service_interface,
            "request_cid": self.request_cid,
            "base_plan_root": self.base_plan_root,
            "base_plan_revision": self.base_plan_revision,
            "base_admitted_plan_root": self.base_admitted_plan_root,
            "current_roots": self.current_roots.to_dict(),
            "population_partition": self.population_partition.to_dict(),
            "scan_impact": self.scan_impact.to_dict(),
            "event_cursor": self.event_cursor,
            "claimed_population_digest": self.claimed_population_digest,
            "accepted_population_digest": self.accepted_population_digest,
            "accepted_evidence_root": self.accepted_evidence_root,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "verdict": self.verdict.value,
            "read_only": self.read_only,
            "wrote_task_source": self.wrote_task_source,
            "restart_serializable": self.restart_serializable,
            "delta_cid": self.delta_cid,
            "candidate_plan_root": self.candidate_plan_root,
            "candidate_plan_revision": self.candidate_plan_revision,
            "scan_receipt_cid": self.scan_receipt_cid,
            "query_plan_cid": self.query_plan_cid,
            "evidence_bundle_cid": self.evidence_bundle_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "state_cid": self.state_cid,
            "expected_effects": list(self.expected_effects),
            "deferred_item_keys": list(self.deferred_item_keys),
            "successor_item_keys": list(self.successor_item_keys),
            "lifecycle_request_item_keys": list(self.lifecycle_request_item_keys),
            "materializable_item_keys": list(self.materializable_item_keys),
            "rejection_reasons": [
                reason.to_dict() for reason in self.rejection_reasons
            ],
            "artifact_refs": list(self.artifact_refs),
        }

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def admitted(self) -> bool:
        return self.verdict is PlanSteerVerdict.ADMITTED

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(reason.code.value for reason in self.rejection_reasons)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanSteerPreviewReceipt":
        if not isinstance(payload, Mapping):
            raise PlanSteerServiceError(
                "preview receipt payload must be a mapping",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise PlanSteerServiceError(
                "unsupported preview receipt schema",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        names = (
            "request_cid",
            "base_plan_root",
            "base_plan_revision",
            "base_admitted_plan_root",
            "current_roots",
            "population_partition",
            "scan_impact",
            "event_cursor",
            "claimed_population_digest",
            "accepted_population_digest",
            "accepted_evidence_root",
            "lease_id",
            "fencing_epoch",
            "verdict",
            "read_only",
            "wrote_task_source",
            "restart_serializable",
            "delta_cid",
            "candidate_plan_root",
            "candidate_plan_revision",
            "scan_receipt_cid",
            "query_plan_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
            "state_cid",
            "expected_effects",
            "deferred_item_keys",
            "successor_item_keys",
            "lifecycle_request_item_keys",
            "materializable_item_keys",
            "rejection_reasons",
            "artifact_refs",
            "service_interface",
        )
        values = {name: payload[name] for name in names if name in payload}
        value = cls(**values)
        supplied = payload.get("content_id") or payload.get("cid") or payload.get(
            "receipt_cid"
        )
        if supplied not in (None, "") and supplied != value.content_id:
            raise PlanSteerServiceError(
                "preview receipt identity does not match content",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )
        return value


# ---------------------------------------------------------------------------
# Materials envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanSteerPreviewMaterials:
    """Inputs for one revision-bound steer preview.

    The service is read-only: materials never include a task-source writer.
    """

    request: PlanSteerRequest
    live_state: PlanSteerLiveState
    proposed_delta_items: tuple[PlanDeltaItem, ...] = ()
    query_plan_cid: str = ""
    evidence_bundle_cid: str = ""
    admission_receipt_cid: str = ""
    execution_plan_cid: str = ""
    expected_effects: tuple[str, ...] = ()
    # Optional independent admission callback over the candidate plan mapping.
    # Signature: (candidate_plan_mapping) -> (admitted: bool, receipt_cid: str, reasons: Sequence[str])
    admit_candidate: Callable[[Mapping[str, Any]], Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, PlanSteerRequest):
            if isinstance(self.request, Mapping):
                object.__setattr__(
                    self,
                    "request",
                    PlanSteerRequest.from_dict(self.request)
                    if "schema" in self.request
                    else PlanSteerRequest(**self.request),
                )
            else:
                raise PlanSteerServiceError(
                    "request must be PlanSteerRequest",
                    code=PlanSteerRejectionCode.MALFORMED_REQUEST,
                )
        if not isinstance(self.live_state, PlanSteerLiveState):
            if isinstance(self.live_state, Mapping):
                object.__setattr__(
                    self,
                    "live_state",
                    PlanSteerLiveState.from_dict(self.live_state)
                    if "schema" in self.live_state
                    else PlanSteerLiveState(**self.live_state),
                )
            else:
                raise PlanSteerServiceError(
                    "live_state must be PlanSteerLiveState",
                    code=PlanSteerRejectionCode.MALFORMED_STATE,
                )
        items_raw = self.proposed_delta_items
        if items_raw is None:
            items_seq: Sequence[Any] = ()
        elif isinstance(items_raw, Sequence) and not isinstance(
            items_raw, (str, bytes, bytearray)
        ):
            items_seq = items_raw
        else:
            raise PlanSteerServiceError(
                "proposed_delta_items must be a sequence",
                code=PlanSteerRejectionCode.INVALID_DELTA,
            )
        items: list[PlanDeltaItem] = []
        for item in items_seq:
            if isinstance(item, PlanDeltaItem):
                items.append(item)
            elif isinstance(item, Mapping):
                items.append(
                    PlanDeltaItem.from_dict(item)
                    if "schema" in item
                    else PlanDeltaItem(**item)
                )
            else:
                raise PlanSteerServiceError(
                    "proposed_delta_items must contain PlanDeltaItem records",
                    code=PlanSteerRejectionCode.INVALID_DELTA,
                )
        object.__setattr__(self, "proposed_delta_items", tuple(items))
        for name in (
            "query_plan_cid",
            "evidence_bundle_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
        ):
            object.__setattr__(
                self,
                name,
                _cid_or_identity(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "expected_effects",
            _strings(self.expected_effects, "expected_effects"),
        )
        if self.admit_candidate is not None and not callable(self.admit_candidate):
            raise PlanSteerServiceError(
                "admit_candidate must be callable when provided",
                code=PlanSteerRejectionCode.MALFORMED_STATE,
            )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PlanSteerService:
    """Revision-bound steer preview.  Interface: ``PlanSteerService@1``.

    Preview is body-free, read-only, and restart-serializable.  Apply is owned
    by a later task (plan revision store / plan supervisor facade).
    """

    INTERFACE: Final[str] = PLAN_STEER_SERVICE_INTERFACE
    VERSION: Final[int] = PLAN_STEER_SERVICE_VERSION

    def __init__(
        self,
        *,
        query_planner: Any | None = None,
        receipt_store: MutableMapping[str, Mapping[str, Any]] | Any | None = None,
        root_observer: Callable[..., Mapping[str, Any]] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.query_planner = query_planner
        self.receipt_store = receipt_store
        self.root_observer = root_observer
        self._clock_ms = clock_ms or (lambda: 0)
        self._preview_by_request: dict[str, PlanSteerPreviewReceipt] = {}
        self._wrote_task_source = False

    # -- public API --------------------------------------------------------

    def preview_steer(
        self,
        materials: PlanSteerPreviewMaterials
        | Mapping[str, Any]
        | PlanSteerRequest,
        live_state: PlanSteerLiveState | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PlanSteerPreviewReceipt:
        """Produce an admitted or rejected steer preview without writes."""

        if self._wrote_task_source:
            raise PlanSteerServiceError(
                "steer service instance previously attempted a write",
                code=PlanSteerRejectionCode.WRITE_ATTEMPTED,
            )
        bundle = self._coerce_materials(materials, live_state, kwargs)
        request = bundle.request
        existing = self._preview_by_request.get(request.request_cid)
        if existing is not None:
            # Restart-serializable idempotent return for identical request CID.
            return existing

        rejections: list[PlanSteerRejection] = []
        partition: PlanSteerPopulationPartition | None = None
        scan_impact: PlanSteerScanImpact | None = None
        delta: PlanDelta | None = None
        candidate: PlanRevision | None = None
        query_plan_cid = bundle.query_plan_cid
        evidence_bundle_cid = bundle.evidence_bundle_cid
        admission_receipt_cid = bundle.admission_receipt_cid
        execution_plan_cid = bundle.execution_plan_cid

        try:
            self._integrity_check(request, bundle.live_state)
            partition = self.partition_populations(bundle.live_state)
            self._verify_population_bindings(request, partition, bundle.live_state)
            scan_impact = self.bind_scan_and_impact(request, bundle.live_state)
            if self.query_planner is not None and not query_plan_cid:
                query_plan_cid = self._compile_query_plan(request, scan_impact)
            items = self._resolve_delta_items(
                request, partition, scan_impact, bundle.proposed_delta_items
            )
            delta = self.generate_closed_delta(
                request, bundle.live_state, partition, scan_impact, items
            )
            candidate = self.apply_delta_in_memory(
                request, bundle.live_state, partition, delta, scan_impact
            )
            self.validate_resulting_plan(
                request,
                bundle.live_state,
                partition,
                delta,
                candidate,
            )
            if bundle.admit_candidate is not None:
                admission_receipt_cid = self._run_admission(
                    bundle.admit_candidate, candidate, admission_receipt_cid
                )
            elif not admission_receipt_cid:
                # Self-admission receipt for the validated candidate.
                admission_receipt_cid = plan_revision_cid(
                    {
                        "namespace": "plan-steer-self-admission",
                        "candidate_plan_root": candidate.plan_root_cid,
                        "delta_cid": delta.delta_cid,
                        "request_cid": request.request_cid,
                        "verdict": "admitted",
                    }
                )
        except (
            PlanSteerServiceError,
            PlanRevisionContractError,
            PlanRevisionLifecycleError,
            PlanRevisionStaleRootError,
        ) as exc:
            code = getattr(exc, "code", None)
            if isinstance(exc, PlanRevisionStaleRootError):
                code = PlanSteerRejectionCode.STALE_ROOT
            elif isinstance(exc, PlanRevisionLifecycleError):
                code = PlanSteerRejectionCode.LIFECYCLE_VIOLATION
            elif isinstance(exc, PlanRevisionContractError) and code is None:
                code = PlanSteerRejectionCode.INVALID_DELTA
            if code is None:
                code = PlanSteerRejectionCode.SERVICE_ERROR
            if not isinstance(code, PlanSteerRejectionCode):
                try:
                    code = PlanSteerRejectionCode(str(code))
                except ValueError:
                    code = PlanSteerRejectionCode.SERVICE_ERROR
            rejections.append(
                PlanSteerRejection(code=code, message=str(exc) or code.value)
            )

        if partition is None:
            # Minimal empty partition for rejected previews that failed before
            # partitioning so the receipt remains serializable.
            partition = self._empty_partition()
        if scan_impact is None:
            scan_impact = self._fallback_scan_impact(request, bundle.live_state)

        if rejections:
            receipt = self._build_receipt(
                request=request,
                live_state=bundle.live_state,
                partition=partition,
                scan_impact=scan_impact,
                verdict=PlanSteerVerdict.REJECTED,
                delta=delta,
                candidate=candidate,
                query_plan_cid=query_plan_cid,
                evidence_bundle_cid=evidence_bundle_cid,
                admission_receipt_cid=admission_receipt_cid,
                execution_plan_cid=execution_plan_cid,
                expected_effects=bundle.expected_effects,
                rejections=tuple(rejections),
            )
        else:
            assert delta is not None and candidate is not None
            receipt = self._build_receipt(
                request=request,
                live_state=bundle.live_state,
                partition=partition,
                scan_impact=scan_impact,
                verdict=PlanSteerVerdict.ADMITTED,
                delta=delta,
                candidate=candidate,
                query_plan_cid=query_plan_cid,
                evidence_bundle_cid=evidence_bundle_cid,
                admission_receipt_cid=admission_receipt_cid,
                execution_plan_cid=execution_plan_cid,
                expected_effects=bundle.expected_effects
                or tuple(delta.expected_effects),
                rejections=(),
            )
        self._persist(receipt)
        self._preview_by_request[request.request_cid] = receipt
        # Hard guarantee: preview never writes task sources.
        if self._wrote_task_source:
            raise PlanSteerServiceError(
                "steer preview attempted a task-source write",
                code=PlanSteerRejectionCode.WRITE_ATTEMPTED,
            )
        return receipt

    # Alias used by the later PlanSupervisorService facade.
    preview = preview_steer

    # -- partitioning ------------------------------------------------------

    def partition_populations(
        self, live_state: PlanSteerLiveState | Mapping[str, Any]
    ) -> PlanSteerPopulationPartition:
        """Partition live task records into closed lifecycle populations."""

        state = (
            live_state
            if isinstance(live_state, PlanSteerLiveState)
            else PlanSteerLiveState.from_dict(live_state)
            if isinstance(live_state, Mapping) and "schema" in live_state
            else PlanSteerLiveState(**live_state)  # type: ignore[arg-type]
        )
        buckets: dict[str, list[str]] = {
            "completed": [],
            "accepted": [],
            "claimed": [],
            "running": [],
            "settling": [],
            "unstarted": [],
            "blocked": [],
            "superseded": [],
            "failed": [],
        }
        mapping = {
            LifecycleState.COMPLETED: "completed",
            LifecycleState.ACCEPTED: "accepted",
            LifecycleState.CLAIMED: "claimed",
            LifecycleState.RUNNING: "running",
            LifecycleState.SETTLING: "settling",
            LifecycleState.UNSTARTED: "unstarted",
            LifecycleState.READY: "unstarted",
            LifecycleState.PROPOSED: "unstarted",
            LifecycleState.ADMITTED: "unstarted",
            LifecycleState.BLOCKED: "blocked",
            LifecycleState.SUPERSEDED: "superseded",
            LifecycleState.FAILED: "failed",
            LifecycleState.CANCELLED: "failed",
            LifecycleState.QUARANTINED: "failed",
        }
        for task in state.tasks:
            bucket = mapping.get(task.lifecycle_state)
            if bucket is None:
                raise PlanSteerServiceError(
                    f"task {task.task_cid} has unpartitionable state "
                    f"{task.lifecycle_state.value}",
                    code=PlanSteerRejectionCode.POPULATION_INTEGRITY,
                )
            buckets[bucket].append(task.task_cid)
        return PlanSteerPopulationPartition(
            completed=_population(PopulationKind.COMPLETED, buckets["completed"]),
            accepted=_population(PopulationKind.ACCEPTED, buckets["accepted"]),
            claimed=_population(PopulationKind.CLAIMED, buckets["claimed"]),
            running=_population(PopulationKind.RUNNING, buckets["running"]),
            settling=_population(PopulationKind.SETTLING, buckets["settling"]),
            unstarted=_population(PopulationKind.UNSTARTED, buckets["unstarted"]),
            blocked=_population(PopulationKind.BLOCKED, buckets["blocked"]),
            superseded=_population(
                PopulationKind.SUPERSEDED, buckets["superseded"]
            ),
            failed=_population(PopulationKind.FAILED, buckets["failed"]),
        )

    # -- scan / impact -----------------------------------------------------

    def bind_scan_and_impact(
        self,
        request: PlanSteerRequest,
        live_state: PlanSteerLiveState,
    ) -> PlanSteerScanImpact:
        """Bind the current tree scan and impact evidence to the request roots."""

        scan = dict(live_state.scan)
        impact = dict(live_state.impact)
        scan_cid = _optional_text(
            scan.get("scan_cid")
            or scan.get("scan_receipt_cid")
            or scan.get("content_id")
            or impact.get("scan_receipt_cid")
            or "",
            "scan_receipt_cid",
        )
        if not scan_cid:
            raise PlanSteerServiceError(
                "live state is missing a current scan receipt",
                code=PlanSteerRejectionCode.MISSING_SCAN,
            )
        repo_root = _optional_text(
            scan.get("repository_root_cid")
            or impact.get("repository_root_cid")
            or live_state.current_roots.repository_root_cid,
            "repository_root_cid",
        )
        dirty_root = _optional_text(
            scan.get("dirty_worktree_root")
            or impact.get("dirty_worktree_root")
            or live_state.current_roots.dirty_worktree_root,
            "dirty_worktree_root",
        )
        if repo_root != live_state.current_roots.repository_root_cid:
            raise PlanSteerStaleError(
                "scan repository_root_cid differs from live roots",
                code=PlanSteerRejectionCode.STALE_SCAN,
            )
        if dirty_root != live_state.current_roots.dirty_worktree_root:
            raise PlanSteerStaleError(
                "scan dirty_worktree_root differs from live roots",
                code=PlanSteerRejectionCode.STALE_SCAN,
            )
        if dirty_root != request.roots.dirty_worktree_root:
            raise PlanSteerStaleError(
                "scan dirty_worktree_root is stale relative to the request",
                code=PlanSteerRejectionCode.STALE_SCAN,
            )
        if repo_root != request.roots.repository_root_cid:
            raise PlanSteerStaleError(
                "scan repository_root_cid is stale relative to the request",
                code=PlanSteerRejectionCode.STALE_SCAN,
            )

        def _paths(*keys: str) -> tuple[str, ...]:
            collected: list[str] = []
            for source in (impact, scan):
                for key in keys:
                    raw = source.get(key)
                    if raw is None:
                        continue
                    if isinstance(raw, str):
                        collected.append(raw)
                    elif isinstance(raw, Sequence) and not isinstance(
                        raw, (str, bytes, bytearray)
                    ):
                        collected.extend(str(item) for item in raw)
            # Prefer repository-relative paths already validated by contracts.
            cleaned: list[str] = []
            for item in collected:
                text = item.strip()
                if not text or text.startswith("/") or "\\" in text or ".." in text:
                    continue
                cleaned.append(text)
            return tuple(sorted(set(cleaned)))

        return PlanSteerScanImpact(
            scan_receipt_cid=scan_cid,
            repository_root_cid=repo_root,
            dirty_worktree_root=dirty_root,
            base_plan_root=request.base_materialized_plan_root,
            impacted_paths=_paths(
                "impacted_paths", "changed_paths", "affected_paths"
            ),
            impacted_symbols=_paths(
                "impacted_symbols", "changed_symbols", "affected_symbols"
            ),
            added_paths=_paths("added_paths", "added"),
            modified_paths=_paths("modified_paths", "modified"),
            deleted_paths=_paths("deleted_paths", "deleted"),
            renamed_paths=_paths("renamed_paths", "renamed"),
            policy_admitted_untracked_paths=_paths(
                "policy_admitted_untracked_paths", "untracked_paths"
            ),
            taskboard_drift_refs=_paths(
                "taskboard_drift_refs", "status_drift_refs"
            ),
            accepted_output_drift_refs=_paths("accepted_output_drift_refs"),
            truncation_refs=_paths("truncation_refs", "truncations"),
            instability_refs=_paths("instability_refs", "instabilities"),
        )

    # -- delta generation --------------------------------------------------

    def generate_closed_delta(
        self,
        request: PlanSteerRequest,
        live_state: PlanSteerLiveState,
        partition: PlanSteerPopulationPartition,
        scan_impact: PlanSteerScanImpact,
        items: Sequence[PlanDeltaItem],
    ) -> PlanDelta:
        """Build a closed PlanDelta and enforce lifecycle-safe history rules."""

        if not items:
            raise PlanSteerServiceError(
                "steer preview requires a non-empty closed delta",
                code=PlanSteerRejectionCode.EMPTY_DELTA,
            )
        allowed = set(request.allowed_delta_operations)
        closed = set(closed_delta_operations())
        affected_goals: set[str] = set()
        affected_tasks: set[str] = set()
        affected_paths: set[str] = set()
        deferred_keys: list[str] = []
        expected_effects: list[str] = []

        normalized: list[PlanDeltaItem] = []
        for item in items:
            if item.operation.value not in closed:
                raise PlanSteerServiceError(
                    f"delta operation {item.operation.value} is outside the closed vocabulary",
                    code=PlanSteerRejectionCode.FORBIDDEN_OPERATION,
                )
            if item.operation.value not in allowed:
                raise PlanSteerServiceError(
                    f"delta operation {item.operation.value} is not allowed by the request",
                    code=PlanSteerRejectionCode.FORBIDDEN_OPERATION,
                )
            if (
                item.operation is PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION
                and not request.may_request_lifecycle_action
            ):
                raise PlanSteerServiceError(
                    "lifecycle requests are not permitted by this steer request",
                    code=PlanSteerRejectionCode.FORBIDDEN_OPERATION,
                )
            self._assert_item_lifecycle_safe(item, partition)
            normalized.append(item)
            affected_goals.update(item.affected_goal_cids)
            affected_tasks.update(item.affected_task_cids)
            if item.target_cid:
                affected_tasks.add(item.target_cid)
            affected_paths.update(item.affected_paths)
            if item.effect_class is DeltaEffectClass.DEFERRED:
                deferred_keys.append(item.item_key)
            expected_effects.extend(item.expected_effects)

        if len(affected_goals) > request.max_affected_goals:
            raise PlanSteerServiceError(
                "affected goal population exceeds request budget",
                code=PlanSteerRejectionCode.AFFECTED_POPULATION_EXCEEDED,
            )
        if len(affected_tasks) > request.max_affected_tasks:
            raise PlanSteerServiceError(
                "affected task population exceeds request budget",
                code=PlanSteerRejectionCode.AFFECTED_POPULATION_EXCEEDED,
            )
        if len(affected_paths) > request.max_affected_paths:
            raise PlanSteerServiceError(
                "affected path population exceeds request budget",
                code=PlanSteerRejectionCode.AFFECTED_POPULATION_EXCEEDED,
            )

        delta = PlanDelta(
            base_plan_root=request.base_materialized_plan_root,
            base_plan_revision=request.plan_revision,
            request_cid=request.request_cid,
            roots=request.roots,
            items=tuple(normalized),
            expected_effects=tuple(sorted(set(expected_effects))),
            deferred_item_keys=tuple(sorted(set(deferred_keys))),
            claimed_population_digest=partition.claimed_family_digest,
            accepted_population_digest=request.accepted_population.digest,
            scan_receipt_cid=scan_impact.scan_receipt_cid,
            evidence_bundle_cid="",
            admission_receipt_cid="",
        )
        assert_delta_preserves_history(delta)
        return delta

    def apply_delta_in_memory(
        self,
        request: PlanSteerRequest,
        live_state: PlanSteerLiveState,
        partition: PlanSteerPopulationPartition,
        delta: PlanDelta,
        scan_impact: PlanSteerScanImpact,
    ) -> PlanRevision:
        """Apply a closed delta onto an in-memory candidate plan revision."""

        prior_completed = set(partition.completed.member_cids) | set(
            partition.accepted.member_cids
        )
        # Accepted is a subset of completed history for integrity checks.
        prior_accepted = set(partition.accepted.member_cids)
        prior_claimed = (
            set(partition.claimed.member_cids)
            | set(partition.running.member_cids)
            | set(partition.settling.member_cids)
        )

        next_completed = set(prior_completed)
        next_accepted = set(prior_accepted)
        next_claimed = set(prior_claimed)
        next_blocked = set(partition.blocked.member_cids)
        next_superseded = set(partition.superseded.member_cids)
        next_unstarted = set(partition.unstarted.member_cids)
        added: set[str] = set()
        deferred: set[str] = set()
        retained = set(partition.all_member_cids())
        deleted: set[str] = set()

        for item in delta.items:
            op = item.operation
            target = item.target_cid
            after = item.after_record_cid
            if op is PlanDeltaOperation.ADD_TASK:
                member = after or plan_revision_cid(
                    {"item_key": item.item_key, "op": op.value}
                )
                added.add(member)
                if item.effect_class is DeltaEffectClass.DEFERRED:
                    deferred.add(member)
                else:
                    next_unstarted.add(member)
                retained.add(member)
            elif op is PlanDeltaOperation.ADD_GOAL:
                member = after or plan_revision_cid(
                    {"item_key": item.item_key, "op": op.value}
                )
                added.add(member)
                retained.add(member)
            elif op is PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK:
                if target in prior_claimed or target in prior_completed:
                    # Should have been deferred or rejected already.
                    if item.effect_class is DeltaEffectClass.DEFERRED:
                        deferred.add(after or target)
                        added.add(after or target)
                    else:
                        raise PlanSteerLifecycleError(
                            f"cannot supersede immutable target {target}",
                            code=PlanSteerRejectionCode.RUNNING_EDIT,
                        )
                else:
                    if target:
                        next_unstarted.discard(target)
                        next_blocked.discard(target)
                        next_superseded.add(target)
                    if after:
                        added.add(after)
                        next_unstarted.add(after)
                        retained.add(after)
            elif op is PlanDeltaOperation.BLOCK_UNSTARTED_TASK:
                if target:
                    if target in prior_claimed or is_history_immutable(
                        partition.lifecycle_of(target) or LifecycleState.UNSTARTED
                    ):
                        if partition.lifecycle_of(target) in _RUNNING_FAMILY:
                            raise PlanSteerLifecycleError(
                                f"cannot block running/claimed task {target}",
                                code=PlanSteerRejectionCode.RUNNING_EDIT,
                            )
                    next_unstarted.discard(target)
                    next_blocked.add(target)
            elif op is PlanDeltaOperation.UNBLOCK_TASK:
                if target and target in next_blocked:
                    next_blocked.discard(target)
                    next_unstarted.add(target)
            elif op is PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION:
                # Explicit separate lifecycle request; no in-place mutation.
                deferred.add(item.item_key)
            elif op in (
                PlanDeltaOperation.ATTACH_EVIDENCE,
                PlanDeltaOperation.RECORD_UNCERTAINTY,
            ):
                # Evidence-only; history membership unchanged.
                pass
            elif op in (
                PlanDeltaOperation.SPLIT_UNSTARTED_TASK,
                PlanDeltaOperation.COALESCE_UNSTARTED_TASKS,
                PlanDeltaOperation.REWIRE_UNSTARTED_DEPENDENCY,
                PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
                PlanDeltaOperation.ASSIGN_PARALLEL_CONTRACT,
                PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
                PlanDeltaOperation.SUPERSEDE_GOAL,
            ):
                if target and partition.lifecycle_of(target) in _IMMUTABLE_STATES:
                    raise PlanSteerLifecycleError(
                        f"operation {op.value} cannot mutate immutable target {target}",
                        code=PlanSteerRejectionCode.RUNNING_EDIT,
                    )
                if after:
                    added.add(after)
                    retained.add(after)
                    if op in (
                        PlanDeltaOperation.SPLIT_UNSTARTED_TASK,
                        PlanDeltaOperation.COALESCE_UNSTARTED_TASKS,
                        PlanDeltaOperation.SUPERSEDE_GOAL,
                        PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
                    ):
                        next_unstarted.add(after)
                        if target:
                            next_superseded.add(target)
                            next_unstarted.discard(target)

        assert_population_history_intact(
            prior_completed=prior_completed,
            prior_accepted=prior_accepted,
            prior_claimed=prior_claimed,
            next_completed=next_completed,
            next_accepted=next_accepted,
            next_claimed=next_claimed,
            deleted_cids=deleted,
        )

        parent_root = request.base_materialized_plan_root
        candidate_root = plan_revision_cid(
            {
                "namespace": "plan-steer-candidate",
                "parent_plan_root": parent_root,
                "delta_cid": delta.delta_cid,
                "request_cid": request.request_cid,
                "semantic_revision": request.plan_revision + 1,
            }
        )
        # PlanRevision requires a non-empty admission_receipt_cid.  Preview
        # binds a provisional self-admission identity; an injected admission
        # callback may replace it on the preview receipt without rewriting the
        # candidate graph identity used for history checks.
        provisional_admission = plan_revision_cid(
            {
                "namespace": "plan-steer-provisional-admission",
                "request_cid": request.request_cid,
                "delta_cid": delta.delta_cid,
                "candidate_plan_root": candidate_root,
            }
        )
        base = live_state.base_plan
        resource = (
            base.resource_contract if base is not None else PlanResourceContract()
        )
        provider = (
            base.provider_contract if base is not None else PlanProviderContract()
        )
        lease = live_state.lease_contract
        retry = base.retry_contract if base is not None else PlanRetryContract()
        worktree = live_state.worktree_contract
        merge = live_state.merge_strategy
        conflict = (
            base.conflict_contract
            if base is not None
            else PlanConflictContract(
                predicted_files=tuple(scan_impact.impacted_paths[:64])
            )
        )
        completion = (
            base.completion_rule
            if base is not None
            else PlanCompletionRule(authority=CompletionAuthority.VALIDATION_GATE)
        )
        validation_dag = (
            base.validation_dag
            if base is not None and base.validation_dag
            else (
                PlanValidationNode(
                    validation_key="validation:steer-preview",
                    argv=("python", "-m", "pytest", "-q"),
                ),
            )
        )
        goal_members = {goal.goal_cid for goal in live_state.goals} | {
            cid for cid in added if cid.startswith("goal:") or "goal" in cid
        }
        if base is not None:
            goal_members |= set(base.goal_population.member_cids)
        task_members = set(partition.all_member_cids()) | added
        query_plan_cid = (
            base.query_plan_cid
            if base is not None and base.query_plan_cid
            else plan_revision_cid(
                {
                    "namespace": "plan-steer-query-placeholder",
                    "request_cid": request.request_cid,
                }
            )
        )
        evidence_bundle_cid = (
            base.evidence_bundle_cid
            if base is not None and base.evidence_bundle_cid
            else plan_revision_cid(
                {
                    "namespace": "plan-steer-evidence-placeholder",
                    "request_cid": request.request_cid,
                    "scan": scan_impact.scan_receipt_cid,
                }
            )
        )
        execution_plan_cid = (
            base.execution_plan_cid
            if base is not None and base.execution_plan_cid
            else plan_revision_cid(
                {
                    "namespace": "plan-steer-execution-placeholder",
                    "request_cid": request.request_cid,
                    "candidate_plan_root": candidate_root,
                }
            )
        )

        return PlanRevision(
            plan_root_cid=candidate_root,
            semantic_revision=request.plan_revision + 1,
            parent_plan_root=parent_root,
            origin=PlanOrigin.STEER,
            roots=request.roots,
            request_cid=request.request_cid,
            delta_cid=delta.delta_cid,
            scan_receipt_cid=scan_impact.scan_receipt_cid,
            query_plan_cid=query_plan_cid,
            evidence_bundle_cid=evidence_bundle_cid,
            admission_receipt_cid=provisional_admission,
            execution_plan_cid=execution_plan_cid,
            goal_population=_population(PopulationKind.RETAINED, goal_members),
            task_population=_population(PopulationKind.RETAINED, task_members),
            added_population=_population(PopulationKind.ADDED, added),
            superseded_population=_population(
                PopulationKind.SUPERSEDED, next_superseded
            ),
            retained_population=_population(PopulationKind.RETAINED, retained),
            deferred_population=_population(PopulationKind.DEFERRED, deferred),
            claimed_population=_population(PopulationKind.CLAIMED, next_claimed),
            completed_population=_population(
                PopulationKind.COMPLETED, next_completed
            ),
            blocked_population=_population(PopulationKind.BLOCKED, next_blocked),
            resource_contract=resource,
            provider_contract=provider,
            lease_contract=lease,
            retry_contract=retry,
            worktree_contract=worktree,
            merge_strategy=merge,
            conflict_contract=conflict,
            completion_rule=completion,
            validation_dag=validation_dag,
            event_cursor=request.event_cursor,
        )

    def validate_resulting_plan(
        self,
        request: PlanSteerRequest,
        live_state: PlanSteerLiveState,
        partition: PlanSteerPopulationPartition,
        delta: PlanDelta,
        candidate: PlanRevision,
    ) -> None:
        """Validate the complete resulting plan after in-memory delta apply."""

        if candidate.origin is not PlanOrigin.STEER:
            raise PlanSteerAdmissionError(
                "candidate plan origin must be steer",
                code=PlanSteerRejectionCode.ADMISSION_FAILED,
            )
        if candidate.parent_plan_root != request.base_materialized_plan_root:
            raise PlanSteerAdmissionError(
                "candidate parent_plan_root must equal the base materialized root",
                code=PlanSteerRejectionCode.ADMISSION_FAILED,
            )
        if candidate.semantic_revision != request.plan_revision + 1:
            raise PlanSteerAdmissionError(
                "candidate semantic_revision must be base revision + 1",
                code=PlanSteerRejectionCode.ADMISSION_FAILED,
            )
        if candidate.delta_cid != delta.delta_cid:
            raise PlanSteerAdmissionError(
                "candidate delta_cid must match the generated delta",
                code=PlanSteerRejectionCode.ADMISSION_FAILED,
            )
        if not candidate.roots.matches(request.roots):
            raise PlanSteerStaleError(
                "candidate roots drifted from the request",
                code=PlanSteerRejectionCode.STALE_ROOT,
            )
        # Claimed family must still be present after the steer.
        prior_claimed = (
            set(partition.claimed.member_cids)
            | set(partition.running.member_cids)
            | set(partition.settling.member_cids)
        )
        next_claimed = set(candidate.claimed_population.member_cids)
        next_completed = set(candidate.completed_population.member_cids)
        if not prior_claimed.issubset(next_claimed | next_completed):
            raise PlanSteerLifecycleError(
                "claimed population disappeared without a terminal transition",
                code=PlanSteerRejectionCode.HISTORY_SHRINK,
            )
        prior_accepted = set(partition.accepted.member_cids)
        next_accepted_or_completed = next_completed | set(
            # accepted may be folded into completed population for revisions
            candidate.completed_population.member_cids
        )
        if not prior_accepted.issubset(next_accepted_or_completed):
            raise PlanSteerLifecycleError(
                "accepted history cannot shrink across a steer revision",
                code=PlanSteerRejectionCode.HISTORY_SHRINK,
            )
        # Running work must never appear as a superseded member.
        for item in delta.items:
            state = item.expected_target_lifecycle
            if state in _RUNNING_FAMILY and item.operation in {
                PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
                PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
                PlanDeltaOperation.SPLIT_UNSTARTED_TASK,
                PlanDeltaOperation.COALESCE_UNSTARTED_TASKS,
                PlanDeltaOperation.REWIRE_UNSTARTED_DEPENDENCY,
                PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
                PlanDeltaOperation.ASSIGN_PARALLEL_CONTRACT,
            }:
                if item.effect_class is not DeltaEffectClass.DEFERRED:
                    raise PlanSteerLifecycleError(
                        "running work may only be superseded as an explicit deferred item",
                        code=PlanSteerRejectionCode.RUNNING_EDIT,
                    )
        assert_delta_preserves_history(delta)

    # -- integrity ---------------------------------------------------------

    def _integrity_check(
        self, request: PlanSteerRequest, live_state: PlanSteerLiveState
    ) -> None:
        try:
            request.require_fresh(
                roots=live_state.current_roots,
                plan_revision=live_state.plan_revision,
                event_cursor=live_state.event_cursor,
                claimed_digest=self._live_claimed_digest(live_state),
                accepted_evidence_root=live_state.accepted_evidence_root,
            )
        except PlanRevisionStaleRootError as exc:
            message = str(exc).casefold()
            if "revision" in message:
                code = PlanSteerRejectionCode.STALE_REVISION
            elif "cursor" in message:
                code = PlanSteerRejectionCode.STALE_CURSOR
            elif "claimed" in message:
                code = PlanSteerRejectionCode.STALE_CLAIMED
            elif "accepted" in message:
                code = PlanSteerRejectionCode.STALE_ACCEPTED
            elif "root" in message or "authority" in message:
                code = PlanSteerRejectionCode.STALE_ROOT
            else:
                code = PlanSteerRejectionCode.STALE_ROOT
            raise PlanSteerStaleError(str(exc), code=code) from exc

        if live_state.base_plan_root != request.base_materialized_plan_root:
            raise PlanSteerStaleError(
                "base materialized plan root is stale",
                code=PlanSteerRejectionCode.STALE_BASE,
            )
        if live_state.base_admitted_plan_root and (
            live_state.base_admitted_plan_root != request.base_admitted_plan_root
        ):
            raise PlanSteerStaleError(
                "base admitted plan root is stale",
                code=PlanSteerRejectionCode.STALE_BASE,
            )
        if live_state.plan_revision != request.plan_revision:
            raise PlanSteerStaleError(
                "plan revision is stale",
                code=PlanSteerRejectionCode.STALE_REVISION,
            )
        if live_state.event_cursor != request.event_cursor:
            raise PlanSteerStaleError(
                "event cursor is stale",
                code=PlanSteerRejectionCode.STALE_CURSOR,
            )
        if live_state.accepted_evidence_root != request.accepted_evidence_root:
            raise PlanSteerStaleError(
                "accepted evidence root is stale",
                code=PlanSteerRejectionCode.STALE_ACCEPTED,
            )
        if request.lease_id and live_state.lease_id != request.lease_id:
            raise PlanSteerStaleError(
                "lease_id is stale",
                code=PlanSteerRejectionCode.STALE_LEASE,
            )
        if request.fencing_epoch and live_state.fencing_epoch != request.fencing_epoch:
            raise PlanSteerStaleError(
                "fencing epoch is stale",
                code=PlanSteerRejectionCode.STALE_FENCE,
            )
        if live_state.lease_contract.fencing_epoch and request.fencing_epoch:
            if live_state.lease_contract.fencing_epoch != request.fencing_epoch:
                raise PlanSteerStaleError(
                    "lease contract fencing epoch is stale",
                    code=PlanSteerRejectionCode.STALE_FENCE,
                )
        if request.supervisor_run_id and (
            live_state.supervisor_run_id != request.supervisor_run_id
        ):
            raise PlanSteerStaleError(
                "supervisor run id is stale",
                code=PlanSteerRejectionCode.STALE_RUN,
            )
        if request.supervisor_state_revision and (
            live_state.supervisor_state_revision
            != request.supervisor_state_revision
        ):
            raise PlanSteerStaleError(
                "supervisor state revision is stale",
                code=PlanSteerRejectionCode.STALE_RUN,
            )
        # Policy / catalog / IR roots already covered by require_fresh roots.
        if live_state.current_roots.policy_root != request.roots.policy_root:
            raise PlanSteerStaleError(
                "policy root is stale",
                code=PlanSteerRejectionCode.STALE_POLICY,
            )
        if live_state.base_plan is not None:
            base = live_state.base_plan
            if base.plan_root_cid != request.base_materialized_plan_root:
                raise PlanSteerStaleError(
                    "base plan root does not match the request",
                    code=PlanSteerRejectionCode.STALE_BASE,
                )
            if base.semantic_revision != request.plan_revision:
                raise PlanSteerStaleError(
                    "base plan semantic revision is stale",
                    code=PlanSteerRejectionCode.STALE_REVISION,
                )
            if not base.roots.matches(request.roots):
                raise PlanSteerStaleError(
                    "base plan authority roots are stale",
                    code=PlanSteerRejectionCode.STALE_ROOT,
                )
        if self.root_observer is not None:
            observed = self.root_observer(request, live_state)
            if not isinstance(observed, Mapping):
                raise PlanSteerStaleError(
                    "root observer did not return a mapping",
                    code=PlanSteerRejectionCode.STALE_ROOT,
                )
            expected = {
                "repository_root_cid": request.roots.repository_root_cid,
                "dirty_worktree_root": request.roots.dirty_worktree_root,
                "policy_root": request.roots.policy_root,
                "task_source_revision": request.roots.task_source_revision,
                "plan_revision": request.plan_revision,
                "event_cursor": request.event_cursor,
            }
            for key, value in expected.items():
                if key in observed and observed[key] != value:
                    raise PlanSteerStaleError(
                        f"observed {key} is stale relative to the request",
                        code=PlanSteerRejectionCode.STALE_ROOT,
                    )

    def _verify_population_bindings(
        self,
        request: PlanSteerRequest,
        partition: PlanSteerPopulationPartition,
        live_state: PlanSteerLiveState,
    ) -> None:
        live_claimed = partition.claimed_family_digest
        if live_claimed != request.claimed_population.digest:
            # Allow the request claimed population to be exactly the union set.
            request_members = set(request.claimed_population.member_cids)
            live_members = (
                set(partition.claimed.member_cids)
                | set(partition.running.member_cids)
                | set(partition.settling.member_cids)
            )
            if request_members != live_members:
                raise PlanSteerStaleError(
                    "claimed population digest is stale",
                    code=PlanSteerRejectionCode.STALE_CLAIMED,
                )
        accepted_members = set(request.accepted_population.member_cids)
        live_accepted = set(partition.accepted.member_cids) | set(
            partition.completed.member_cids
        )
        if request.accepted_population.kind is PopulationKind.ACCEPTED:
            if not accepted_members.issubset(live_accepted):
                raise PlanSteerStaleError(
                    "accepted population is stale",
                    code=PlanSteerRejectionCode.STALE_ACCEPTED,
                )
        elif request.accepted_population.kind is PopulationKind.COMPLETED:
            if not accepted_members.issubset(
                set(partition.completed.member_cids)
                | set(partition.accepted.member_cids)
            ):
                raise PlanSteerStaleError(
                    "completed population is stale",
                    code=PlanSteerRejectionCode.STALE_ACCEPTED,
                )

    def _live_claimed_digest(self, live_state: PlanSteerLiveState) -> str:
        members = [
            task.task_cid
            for task in live_state.tasks
            if task.lifecycle_state in _RUNNING_FAMILY
        ]
        return _population(PopulationKind.CLAIMED, members).digest

    def _assert_item_lifecycle_safe(
        self,
        item: PlanDeltaItem,
        partition: PlanSteerPopulationPartition,
    ) -> None:
        target = item.target_cid
        if not target:
            return
        observed = partition.lifecycle_of(target)
        if observed is not None and observed is not item.expected_target_lifecycle:
            # Allow claimed-family aliases (claimed request vs running observation).
            aliases = {
                LifecycleState.CLAIMED: _RUNNING_FAMILY,
                LifecycleState.RUNNING: _RUNNING_FAMILY,
                LifecycleState.SETTLING: _RUNNING_FAMILY,
            }
            allowed = aliases.get(item.expected_target_lifecycle, {item.expected_target_lifecycle})
            if observed not in allowed:
                raise PlanSteerLifecycleError(
                    f"delta item {item.item_key} expected lifecycle "
                    f"{item.expected_target_lifecycle.value} but live state is "
                    f"{observed.value}",
                    code=PlanSteerRejectionCode.LIFECYCLE_VIOLATION,
                )
        state = item.expected_target_lifecycle
        if state in _IMMUTABLE_STATES:
            if item.operation not in _SAFE_ON_IMMUTABLE:
                raise PlanSteerLifecycleError(
                    f"operation {item.operation.value} cannot target "
                    f"{state.value} history; use ADD_TASK successors, "
                    f"ATTACH_EVIDENCE, RECORD_UNCERTAINTY, or "
                    f"REQUEST_LIFECYCLE_ACTION instead",
                    code=PlanSteerRejectionCode.RUNNING_EDIT,
                )
            # Safe ops may name immutable targets only as non-mutating
            # successors, evidence, uncertainty, or separate lifecycle
            # requests.  Deferred supersession of running work is expressed
            # as a DEFERRED ADD_TASK successor, never SUPERSEDE_* in place.

    # -- delta item resolution ---------------------------------------------

    def _resolve_delta_items(
        self,
        request: PlanSteerRequest,
        partition: PlanSteerPopulationPartition,
        scan_impact: PlanSteerScanImpact,
        proposed: Sequence[PlanDeltaItem],
    ) -> tuple[PlanDeltaItem, ...]:
        if proposed:
            return tuple(proposed)
        # Deterministic fallback: build a single successor ADD_TASK from the
        # directive metadata / impact paths so the preview remains usable
        # without a model.  Never invent lifecycle mutations.
        meta = dict(request.redacted_directive_metadata)
        paths = _strings(
            meta.get("affected_paths")
            or meta.get("changed_paths")
            or scan_impact.impacted_paths,
            "affected_paths",
        )
        if not paths and scan_impact.modified_paths:
            paths = scan_impact.modified_paths
        if PlanDeltaOperation.ADD_TASK.value not in request.allowed_delta_operations:
            raise PlanSteerServiceError(
                "no proposed delta items and ADD_TASK is not allowed",
                code=PlanSteerRejectionCode.EMPTY_DELTA,
            )
        running_targets = (
            list(partition.running.member_cids)
            or list(partition.claimed.member_cids)
            or list(partition.settling.member_cids)
        )
        target = running_targets[0] if running_targets else ""
        target_state = (
            partition.lifecycle_of(target)
            if target
            else LifecycleState.UNSTARTED
        )
        if target_state is None:
            target_state = LifecycleState.UNSTARTED
        after_cid = plan_revision_cid(
            {
                "namespace": "plan-steer-successor",
                "request_cid": request.request_cid,
                "target": target,
                "paths": list(paths[:16]),
            }
        )
        effect = (
            DeltaEffectClass.DEFERRED
            if target and target_state in _RUNNING_FAMILY
            else DeltaEffectClass.MATERIALIZABLE_NOW
        )
        item = PlanDeltaItem(
            item_key="delta:steer-successor-1",
            operation=PlanDeltaOperation.ADD_TASK,
            target_cid=target,
            expected_target_lifecycle=target_state,
            expected_target_spec_revision="",
            before_digest="",
            after_record_cid=after_cid,
            effect_class=effect,
            rationale="Deterministic successor task for steer impact.",
            provenance={
                "source": "plan-steer-service-deterministic",
                "directive_cid": request.directive_cid,
            },
            expected_effects=("append-task",),
            affected_task_cids=(after_cid,) + ((target,) if target else ()),
            affected_paths=paths[: request.max_affected_paths],
            preconditions=(
                (f"target-terminal:{target}",)
                if effect is DeltaEffectClass.DEFERRED and target
                else ()
            ),
        )
        return (item,)

    # -- admission / query -------------------------------------------------

    def _compile_query_plan(
        self, request: PlanSteerRequest, scan_impact: PlanSteerScanImpact
    ) -> str:
        try:
            plan = self.query_planner.compile(request)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - fail closed with typed code
            raise PlanSteerServiceError(
                f"query planner failed: {exc}",
                code=PlanSteerRejectionCode.QUERY_FAILED,
            ) from exc
        for attr in ("plan_id", "query_plan_id", "content_id", "cid"):
            value = getattr(plan, attr, None)
            if isinstance(value, str) and value:
                return value
        converter = getattr(plan, "to_dict", None)
        if callable(converter):
            return plan_revision_cid(
                {"namespace": "plan-steer-query", "plan": converter()}
            )
        return plan_revision_cid(
            {
                "namespace": "plan-steer-query",
                "request_cid": request.request_cid,
                "scan": scan_impact.scan_receipt_cid,
            }
        )

    def _run_admission(
        self,
        admit_candidate: Callable[[Mapping[str, Any]], Any],
        candidate: PlanRevision,
        existing_receipt_cid: str,
    ) -> str:
        result = admit_candidate(candidate.to_dict())
        if isinstance(result, bool):
            if not result:
                raise PlanSteerAdmissionError(
                    "candidate plan failed admission",
                    code=PlanSteerRejectionCode.ADMISSION_FAILED,
                )
            return existing_receipt_cid or plan_revision_cid(
                {
                    "namespace": "plan-steer-admission",
                    "candidate": candidate.plan_root_cid,
                    "admitted": True,
                }
            )
        if isinstance(result, Mapping):
            admitted = bool(result.get("admitted", result.get("verdict") == "admitted"))
            receipt = str(
                result.get("receipt_cid")
                or result.get("admission_receipt_cid")
                or result.get("content_id")
                or ""
            )
            if not admitted:
                reasons = result.get("reasons") or result.get("rejection_reasons") or ()
                message = "candidate plan failed admission"
                if isinstance(reasons, Sequence) and reasons:
                    message = f"{message}: {reasons[0]}"
                raise PlanSteerAdmissionError(
                    message,
                    code=PlanSteerRejectionCode.ADMISSION_FAILED,
                )
            return receipt or existing_receipt_cid
        if isinstance(result, Sequence) and not isinstance(
            result, (str, bytes, bytearray)
        ):
            # (admitted, receipt_cid, reasons)
            admitted = bool(result[0]) if result else False
            receipt = str(result[1]) if len(result) > 1 else ""
            if not admitted:
                raise PlanSteerAdmissionError(
                    "candidate plan failed admission",
                    code=PlanSteerRejectionCode.ADMISSION_FAILED,
                )
            return receipt or existing_receipt_cid
        raise PlanSteerAdmissionError(
            "admit_candidate returned an unsupported result",
            code=PlanSteerRejectionCode.ADMISSION_FAILED,
        )

    # -- receipt construction ----------------------------------------------

    def _build_receipt(
        self,
        *,
        request: PlanSteerRequest,
        live_state: PlanSteerLiveState,
        partition: PlanSteerPopulationPartition,
        scan_impact: PlanSteerScanImpact,
        verdict: PlanSteerVerdict,
        delta: PlanDelta | None,
        candidate: PlanRevision | None,
        query_plan_cid: str,
        evidence_bundle_cid: str,
        admission_receipt_cid: str,
        execution_plan_cid: str,
        expected_effects: Sequence[str],
        rejections: Sequence[PlanSteerRejection],
    ) -> PlanSteerPreviewReceipt:
        deferred: list[str] = []
        successors: list[str] = []
        lifecycle_requests: list[str] = []
        materializable: list[str] = []
        if delta is not None:
            for item in delta.items:
                if item.effect_class is DeltaEffectClass.DEFERRED:
                    deferred.append(item.item_key)
                if item.effect_class is DeltaEffectClass.LIFECYCLE_REQUEST or (
                    item.operation is PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION
                ):
                    lifecycle_requests.append(item.item_key)
                if (
                    item.operation is PlanDeltaOperation.ADD_TASK
                    and item.target_cid
                    and partition.lifecycle_of(item.target_cid) in _RUNNING_FAMILY
                ):
                    successors.append(item.item_key)
                if item.effect_class is DeltaEffectClass.MATERIALIZABLE_NOW:
                    materializable.append(item.item_key)

        artifact_refs = {
            request.request_cid,
            request.base_materialized_plan_root,
            scan_impact.scan_receipt_cid,
            partition.partition_cid,
            scan_impact.impact_cid,
            live_state.state_cid,
        }
        if delta is not None:
            artifact_refs.add(delta.delta_cid)
        if candidate is not None:
            artifact_refs.add(candidate.plan_root_cid)
        for cid in (
            query_plan_cid,
            evidence_bundle_cid,
            admission_receipt_cid,
            execution_plan_cid,
        ):
            if cid:
                artifact_refs.add(cid)

        return PlanSteerPreviewReceipt(
            request_cid=request.request_cid,
            base_plan_root=request.base_materialized_plan_root,
            base_plan_revision=request.plan_revision,
            base_admitted_plan_root=request.base_admitted_plan_root,
            current_roots=request.roots,
            population_partition=partition,
            scan_impact=scan_impact,
            event_cursor=request.event_cursor,
            claimed_population_digest=partition.claimed_family_digest,
            accepted_population_digest=request.accepted_population.digest,
            accepted_evidence_root=request.accepted_evidence_root,
            lease_id=request.lease_id or live_state.lease_id,
            fencing_epoch=request.fencing_epoch or live_state.fencing_epoch,
            verdict=verdict,
            read_only=True,
            wrote_task_source=False,
            restart_serializable=True,
            delta_cid=delta.delta_cid if delta is not None else "",
            candidate_plan_root=(
                candidate.plan_root_cid if candidate is not None else ""
            ),
            candidate_plan_revision=(
                candidate.semantic_revision if candidate is not None else 0
            ),
            scan_receipt_cid=scan_impact.scan_receipt_cid,
            query_plan_cid=query_plan_cid,
            evidence_bundle_cid=evidence_bundle_cid,
            admission_receipt_cid=admission_receipt_cid,
            execution_plan_cid=execution_plan_cid,
            state_cid=live_state.state_cid,
            expected_effects=tuple(expected_effects),
            deferred_item_keys=tuple(sorted(set(deferred))),
            successor_item_keys=tuple(sorted(set(successors))),
            lifecycle_request_item_keys=tuple(sorted(set(lifecycle_requests))),
            materializable_item_keys=tuple(sorted(set(materializable))),
            rejection_reasons=tuple(rejections),
            artifact_refs=tuple(sorted(artifact_refs)),
        )

    def _empty_partition(self) -> PlanSteerPopulationPartition:
        return PlanSteerPopulationPartition(
            completed=_population(PopulationKind.COMPLETED, ()),
            accepted=_population(PopulationKind.ACCEPTED, ()),
            claimed=_population(PopulationKind.CLAIMED, ()),
            running=_population(PopulationKind.RUNNING, ()),
            settling=_population(PopulationKind.SETTLING, ()),
            unstarted=_population(PopulationKind.UNSTARTED, ()),
            blocked=_population(PopulationKind.BLOCKED, ()),
            superseded=_population(PopulationKind.SUPERSEDED, ()),
            failed=_population(PopulationKind.FAILED, ()),
        )

    def _fallback_scan_impact(
        self, request: PlanSteerRequest, live_state: PlanSteerLiveState
    ) -> PlanSteerScanImpact:
        scan_cid = plan_revision_cid(
            {
                "namespace": "plan-steer-missing-scan",
                "request_cid": request.request_cid,
            }
        )
        return PlanSteerScanImpact(
            scan_receipt_cid=scan_cid,
            repository_root_cid=request.roots.repository_root_cid,
            dirty_worktree_root=request.roots.dirty_worktree_root,
            base_plan_root=request.base_materialized_plan_root,
        )

    def _coerce_materials(
        self,
        materials: PlanSteerPreviewMaterials | Mapping[str, Any] | PlanSteerRequest,
        live_state: PlanSteerLiveState | Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> PlanSteerPreviewMaterials:
        if isinstance(materials, PlanSteerPreviewMaterials):
            return materials
        if isinstance(materials, PlanSteerRequest):
            if live_state is None:
                raise PlanSteerServiceError(
                    "live_state is required when passing a PlanSteerRequest",
                    code=PlanSteerRejectionCode.MALFORMED_STATE,
                )
            return PlanSteerPreviewMaterials(
                request=materials,
                live_state=live_state  # type: ignore[arg-type]
                if isinstance(live_state, PlanSteerLiveState)
                else PlanSteerLiveState.from_dict(live_state)
                if isinstance(live_state, Mapping) and "schema" in live_state
                else PlanSteerLiveState(**live_state),  # type: ignore[arg-type]
                proposed_delta_items=tuple(kwargs.get("proposed_delta_items") or ()),
                query_plan_cid=str(kwargs.get("query_plan_cid") or ""),
                evidence_bundle_cid=str(kwargs.get("evidence_bundle_cid") or ""),
                admission_receipt_cid=str(
                    kwargs.get("admission_receipt_cid") or ""
                ),
                execution_plan_cid=str(kwargs.get("execution_plan_cid") or ""),
                expected_effects=tuple(kwargs.get("expected_effects") or ()),
                admit_candidate=kwargs.get("admit_candidate"),
            )
        if isinstance(materials, Mapping):
            payload = dict(materials)
            if live_state is not None and "live_state" not in payload:
                payload["live_state"] = live_state
            for key, value in kwargs.items():
                payload.setdefault(key, value)
            return PlanSteerPreviewMaterials(**payload)  # type: ignore[arg-type]
        raise PlanSteerServiceError(
            "materials must be PlanSteerPreviewMaterials, PlanSteerRequest, or mapping",
            code=PlanSteerRejectionCode.MALFORMED_REQUEST,
        )

    def _persist(self, receipt: PlanSteerPreviewReceipt) -> None:
        if self.receipt_store is None:
            return
        record = receipt.to_record()
        if isinstance(self.receipt_store, MutableMapping):
            existing = self.receipt_store.get(receipt.content_id)
            if existing is not None and dict(existing) != record:
                raise PlanSteerServiceError(
                    "receipt store contains a conflicting canonical receipt",
                    code=PlanSteerRejectionCode.SERVICE_ERROR,
                )
            self.receipt_store[receipt.content_id] = record
            return
        put = getattr(self.receipt_store, "put", None) or getattr(
            self.receipt_store, "store", None
        )
        if not callable(put):
            raise PlanSteerServiceError(
                "receipt_store must be a mutable mapping or implement put",
                code=PlanSteerRejectionCode.SERVICE_ERROR,
            )
        put(receipt.content_id, record)


def preview_steer(
    materials: PlanSteerPreviewMaterials | Mapping[str, Any] | PlanSteerRequest,
    live_state: PlanSteerLiveState | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PlanSteerPreviewReceipt:
    """Module-level convenience wrapper around :class:`PlanSteerService`."""

    return PlanSteerService().preview_steer(materials, live_state, **kwargs)


def construct_plan_steer_live_state(
    **fields: Any,
) -> PlanSteerLiveState:
    """Build a :class:`PlanSteerLiveState` from keyword fields."""

    return PlanSteerLiveState(**fields)


def partition_live_task_populations(
    tasks: Sequence[Mapping[str, Any] | PlanSteerTaskRecord | Any],
) -> PlanSteerPopulationPartition:
    """Partition arbitrary task observations without a full live-state object."""

    state = PlanSteerLiveState(
        current_roots=PlanAuthorityRoots(
            repository_id="repository:partition-only",
            repository_root_cid=plan_revision_cid({"fixture": "repo"}),
            dirty_worktree_root=plan_revision_cid({"fixture": "dirty"}),
            task_source_id="task-source:partition",
            task_source_revision=plan_revision_cid({"fixture": "ts"}),
            policy_root=plan_revision_cid({"fixture": "policy"}),
            intent_ir_root=plan_revision_cid({"fixture": "intent"}),
            legal_ir_root=plan_revision_cid({"fixture": "legal"}),
            security_ir_root=plan_revision_cid({"fixture": "security"}),
            program_root=plan_revision_cid({"fixture": "program"}),
            capability_catalog_root=plan_revision_cid({"fixture": "cap"}),
            provider_catalog_root=plan_revision_cid({"fixture": "prov"}),
            usage_policy_root=plan_revision_cid({"fixture": "usage"}),
        ),
        plan_revision=1,
        event_cursor=plan_revision_cid({"fixture": "cursor"}),
        accepted_evidence_root=plan_revision_cid({"fixture": "accepted"}),
        base_plan_root=plan_revision_cid({"fixture": "base"}),
        tasks=tuple(tasks),
    )
    return PlanSteerService().partition_populations(state)


__all__ = [
    "PLAN_STEER_SERVICE_INTERFACE",
    "PLAN_STEER_SERVICE_VERSION",
    "PlanSteerAdmissionError",
    "PlanSteerGoalRecord",
    "PlanSteerLifecycleError",
    "PlanSteerLiveState",
    "PlanSteerPopulationPartition",
    "PlanSteerPreviewMaterials",
    "PlanSteerPreviewReceipt",
    "PlanSteerRejection",
    "PlanSteerRejectionCode",
    "PlanSteerScanImpact",
    "PlanSteerService",
    "PlanSteerServiceError",
    "PlanSteerStaleError",
    "PlanSteerTaskRecord",
    "PlanSteerVerdict",
    "construct_plan_steer_live_state",
    "partition_live_task_populations",
    "preview_steer",
]
