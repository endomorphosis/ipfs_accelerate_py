"""Compile benchmark/Doctor residuals into bounded derived goals and tasks.

Interface: ``PlannerDoctorRefill@1`` (PDR-081)

This module is the production gate between live epoch residuals and the
separate derived runtime DuckDB task source:

* at most 8 goals and 24 tasks per epoch, and 48 open tasks overall;
* every proposal carries exact source roots, goal/subgoal/task hierarchy,
  minimal files/context, acceptance/validation, resource/conflict/dependency
  edges, and a stop policy;
* duplicates and semantically unchanged failures back off without re-emitting
  work; exact replay of an admitted population is a no-op;
* candidates cannot edit protected anchors, authorize themselves, lower
  thresholds, or mark parent work complete; and
* generated work enters DuckDB only after independent formal-plan compilation,
  structural admission, and parallel-plan compilation.

The seed plan, objectives heap, seed taskboard, scheduler config, authority
policy, benchmark anchors, and promotion policy remain operator-owned and are
never written by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .doctor_plan_refill import (
    DEFAULT_DERIVED_RUNTIME_SOURCE_ID,
    DERIVED_RUNTIME_SOURCE_GATE,
    DoctorPlanRefillMemory,
    DoctorPlanRefillMemoryEntry,
    DoctorPlanResidual,
    DoctorResidualKind,
    dedupe_residuals,
    extract_residuals_from_fixed_point,
    fixed_point_is_successful,
    residual_fingerprint,
    residual_identity_key,
)
from .objective_graph import ObjectiveWorkKind, ObjectiveWorkProposal
from ..planning.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
)
from ..planning.parallel_plan_compiler import (
    ParallelPlanCompiler,
    ParallelPlanOutcome,
)
from ..proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

PLANNER_DOCTOR_REFILL_INTERFACE: Final[str] = "PlannerDoctorRefill@1"
PLANNER_DOCTOR_REFILL_VERSION: Final[str] = "1.0.0"
CONTRACT_VERSION: Final[int] = 1
PRODUCER_ID: Final[str] = "planner-doctor-refill@1"

PLANNER_DOCTOR_REFILL_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-refill-policy@1"
)
PLANNER_DOCTOR_REFILL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-refill-receipt@1"
)
PLANNER_DOCTOR_REFILL_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-refill-memory@1"
)
DERIVED_GOAL_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/derived-goal-proposal@1"
)
DERIVED_TASK_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/derived-task-proposal@1"
)
DERIVED_ADMISSION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/derived-refill-admission@1"
)
DERIVED_COMPILATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/derived-refill-compilation@1"
)

# Authority is hard-off: refill proposes and may admit only into the derived
# runtime source after independent gates; it never completes or mutates seed.
REFILL_AUTHORIZES_COMPLETION: Final[bool] = False
REFILL_AUTHORIZES_MUTATION: Final[bool] = False
REFILL_AUTHORIZES_SEED_BOARD_EDIT: Final[bool] = False
REFILL_AUTHORIZES_THRESHOLD_LOWER: Final[bool] = False
REFILL_AUTHORIZES_SELF_AUTHORIZATION: Final[bool] = False

DEFAULT_PARENT_GOAL_ID: Final[str] = "PDR-G090"
DEFAULT_MAX_GOALS_PER_EPOCH: Final[int] = 8
DEFAULT_MAX_TASKS_PER_EPOCH: Final[int] = 24
DEFAULT_MAX_OPEN_TASKS: Final[int] = 48
DEFAULT_MAX_DEPTH: Final[int] = 3
DEFAULT_MAX_RETRIES: Final[int] = 2
DEFAULT_MAX_PATHS: Final[int] = 8
DEFAULT_MAX_CONTEXT_PATHS: Final[int] = 8
DEFAULT_BACKOFF_IDENTICAL_ATTEMPTS: Final[int] = 1
DEFAULT_COOLDOWN_SECONDS: Final[int] = 3_600
DEFAULT_RESOURCE_CLASS: Final[str] = "cpu-medium"
DEFAULT_TOKEN_CLASS: Final[str] = "medium"
DEFAULT_STOP_POLICY: Final[str] = (
    "stop:planner-doctor-derived-refill@1:"
    "max-goals=8;max-tasks=24;max-open=48;max-retries=2;"
    "unchanged-backoff;no-seed-mutation;no-self-authorization"
)
DEFAULT_ACTOR_ID: Final[str] = "actor:planner-doctor-refill"
DEFAULT_OWNER_ACTOR_ID: Final[str] = "owner:planner-doctor-refill"
DEFAULT_DERIVED_SOURCE_ROLE: Final[str] = "derived_runtime"

MAX_ID_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_RESIDUALS: Final[int] = 128
MAX_MEMORY_ENTRIES: Final[int] = 4_096
MAX_VALIDATION_COMMANDS: Final[int] = 16

# Operator-protected anchors. Derived work may never list these as outputs.
DEFAULT_PROTECTED_ANCHORS: Final[tuple[str, ...]] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md",
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
    "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
    "config/agent_supervisor_planner_doctor_authority_policy.json",
    "config/agent_supervisor_planner_doctor_authority_policy.seal.json",
    "test/api/test_agent_supervisor_planner_doctor_authority_policy.py",
    "config/agent_supervisor_planner_doctor_benchmark.json",
    "config/agent_supervisor_planner_doctor_benchmark.seal.json",
    "docs/architecture/agent_supervisor_planner_doctor_benchmark.md",
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json",
    "test/api/test_agent_supervisor_planner_doctor_benchmark_contract.py",
)

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-=]{0,511}$"
)
_GOAL_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9._:-]{0,255}$"
)

_AUTHORITY_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "completion_authority",
        "mutation_authority",
        "claims_completion",
        "may_mutate",
        "seed_board_edit",
        "authorize_self",
        "self_authorization",
        "lower_threshold",
        "threshold_override",
        "mark_complete",
        "mark_parent_complete",
        "automatic_promotion",
        "mutate_seed_board",
    }
)

_OBJECTIVE_FAMILY_PREFIX: Final[str] = "objective-family/v1/planner-doctor-derived/"
_OBJECTIVE_INSTANCE_PREFIX: Final[str] = "objective-instance/v1/planner-doctor-derived/"


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class PlannerDoctorRefillError(ValueError):
    """A planner-doctor residual, policy, or receipt is malformed."""


class PlannerDoctorRefillAuthorityError(PlannerDoctorRefillError):
    """Raised when a residual attempts to claim forbidden authority."""


class PlannerDoctorRefillBoundsError(PlannerDoctorRefillError):
    """A residual population or field exceeds a hard bound."""


class PlannerDoctorRefillAdmissionError(PlannerDoctorRefillError):
    """Independent plan/admission/parallel compilation refused the work."""


class ResidualSourceKind(str, Enum):
    """Closed residual origins accepted by the derived refill compiler."""

    DOCTOR = "doctor"
    BENCHMARK = "benchmark"
    PROOF = "proof"
    SECURITY = "security"
    CAPABILITY = "capability"
    RESOURCE = "resource"
    NOVELTY = "novelty"
    OTHER = "other"


class PlannerDoctorRefillDisposition(str, Enum):
    """Stable outcomes of one derived refill pass."""

    FIXED_POINT_CLOSED = "fixed_point_closed"
    COMPILED = "compiled"
    ADMITTED = "admitted"
    MATERIALIZED = "materialized"
    DUPLICATE_BACKOFF = "duplicate_backoff"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    BOUND_EXCEEDED = "bound_exceeded"
    OPEN_WORK_CEILING = "open_work_ceiling"
    ADMISSION_REJECTED = "admission_rejected"
    EMPTY_INPUT = "empty_input"
    REPLAY_NOOP = "replay_noop"


class ResidualDisposition(str, Enum):
    """Per-residual admission decision."""

    GOAL = "goal"
    SUBGOAL = "subgoal"
    TASK = "task"
    DUPLICATE = "duplicate"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    BOUND_REJECTED = "bound_rejected"
    ANCHOR_REJECTED = "anchor_rejected"
    AUTHORITY_REJECTED = "authority_rejected"
    MALFORMED = "malformed"
    FIXED_POINT_SKIP = "fixed_point_skip"


# ---------------------------------------------------------------------------
# Normalization helpers
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
        raise PlannerDoctorRefillError(f"{name} must be a string")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise PlannerDoctorRefillError(f"{name} must be normalized single-line text")
    if required and not text:
        raise PlannerDoctorRefillError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise PlannerDoctorRefillBoundsError(f"{name} exceeds its byte bound")
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, limit=MAX_ID_BYTES)
    if not text:
        return ""
    if not _IDENTIFIER_RE.fullmatch(text):
        raise PlannerDoctorRefillError(f"{name} is malformed")
    return text


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = 64,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise PlannerDoctorRefillError(f"{name} must be a sequence of identifiers")
    if len(items) > maximum:
        raise PlannerDoctorRefillBoundsError(f"{name} exceeds its item bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = _identifier(raw, name, required=True)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if required and not out:
        raise PlannerDoctorRefillError(f"{name} must not be empty")
    return tuple(out)


def _command_strings(
    values: Any,
    name: str,
    *,
    maximum: int = MAX_VALIDATION_COMMANDS,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise PlannerDoctorRefillError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise PlannerDoctorRefillBoundsError(f"{name} exceeds its item bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _text(raw, name, required=True, limit=MAX_TEXT_BYTES)
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _normalize_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    normalized = posixpath.normpath(raw)
    if normalized in (".",):
        return ""
    if (
        normalized.startswith("/")
        or normalized == ".."
        or normalized.startswith("../")
        or "/../" in f"/{normalized}/"
    ):
        raise PlannerDoctorRefillError(
            "paths must be repository-relative and non-escaping"
        )
    return normalized


def _paths(
    values: Any,
    name: str,
    *,
    maximum: int = DEFAULT_MAX_PATHS,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise PlannerDoctorRefillError(f"{name} must be a sequence of paths")
    if len(items) > maximum:
        raise PlannerDoctorRefillBoundsError(f"{name} exceeds its path bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        path = _normalize_path(raw)
        if not path or path in seen:
            continue
        if len(path.encode("utf-8")) > MAX_PATH_BYTES:
            raise PlannerDoctorRefillBoundsError(f"{name} path exceeds its byte bound")
        seen.add(path)
        out.append(path)
    return tuple(out)


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise PlannerDoctorRefillError(f"{name} must be a boolean")


def _nonneg_int(value: Any, name: str, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerDoctorRefillError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise PlannerDoctorRefillBoundsError(f"{name} out of bounds")
    return value


def _positive_int(value: Any, name: str, *, maximum: int = 1_000_000) -> int:
    number = _nonneg_int(value, name, maximum=maximum)
    if number < 1:
        raise PlannerDoctorRefillError(f"{name} must be >= 1")
    return number


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerDoctorRefillError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PlannerDoctorRefillError(f"{name} must be between 0 and 1")
    return number


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise PlannerDoctorRefillError(f"{name} has unknown value {text!r}") from exc


def _mapping_proxy(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PlannerDoctorRefillError(f"{name} must be a mapping")
    try:
        canonical = json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise PlannerDoctorRefillError(f"{name} must be canonical JSON data") from exc
    if not isinstance(canonical, dict):
        raise PlannerDoctorRefillError(f"{name} must be a mapping")
    return MappingProxyType(canonical)


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reject_authority_claims(metadata: Mapping[str, Any], *, where: str) -> None:
    for key, value in metadata.items():
        norm = str(key).lower().replace("-", "_")
        if norm in _AUTHORITY_FORBIDDEN_KEYS and value not in (False, None, "", 0):
            raise PlannerDoctorRefillAuthorityError(
                f"{where} cannot claim {key}"
            )


def _path_hits_protected(
    path: str, protected: Sequence[str]
) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    for anchor in protected:
        target = anchor.replace("\\", "/").strip("/")
        if normalized == target or normalized.startswith(target + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# Policy / memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerDoctorRefillPolicy:
    """Hard bounds and gates for one derived residual compilation epoch."""

    max_goals_per_epoch: int = DEFAULT_MAX_GOALS_PER_EPOCH
    max_tasks_per_epoch: int = DEFAULT_MAX_TASKS_PER_EPOCH
    max_open_tasks: int = DEFAULT_MAX_OPEN_TASKS
    max_depth: int = DEFAULT_MAX_DEPTH
    max_retries: int = DEFAULT_MAX_RETRIES
    max_paths: int = DEFAULT_MAX_PATHS
    max_context_paths: int = DEFAULT_MAX_CONTEXT_PATHS
    max_residuals: int = DEFAULT_MAX_TASKS_PER_EPOCH
    backoff_identical_attempts: int = DEFAULT_BACKOFF_IDENTICAL_ATTEMPTS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    resource_class: str = DEFAULT_RESOURCE_CLASS
    token_class: str = DEFAULT_TOKEN_CLASS
    stop_policy: str = DEFAULT_STOP_POLICY
    derived_runtime_source_id: str = DEFAULT_DERIVED_RUNTIME_SOURCE_ID
    protected_anchors: tuple[str, ...] = DEFAULT_PROTECTED_ANCHORS
    # Derived admission is enabled by this module (PDR-081); still never
    # mutates seed boards.
    derived_runtime_admission_enabled: bool = True
    require_independent_compilation: bool = True
    require_parallel_compilation: bool = True
    parallel_review_only: bool = True
    actor_id: str = DEFAULT_ACTOR_ID
    owner_actor_id: str = DEFAULT_OWNER_ACTOR_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_goals_per_epoch",
            _positive_int(self.max_goals_per_epoch, "max_goals_per_epoch", maximum=8),
        )
        object.__setattr__(
            self,
            "max_tasks_per_epoch",
            _positive_int(self.max_tasks_per_epoch, "max_tasks_per_epoch", maximum=24),
        )
        object.__setattr__(
            self,
            "max_open_tasks",
            _positive_int(self.max_open_tasks, "max_open_tasks", maximum=48),
        )
        # Enforce the normative ceilings even if callers pass the defaults via
        # alternate constructions; values above the hard program bounds fail.
        if self.max_goals_per_epoch > 8:
            raise PlannerDoctorRefillBoundsError("max_goals_per_epoch cannot exceed 8")
        if self.max_tasks_per_epoch > 24:
            raise PlannerDoctorRefillBoundsError("max_tasks_per_epoch cannot exceed 24")
        if self.max_open_tasks > 48:
            raise PlannerDoctorRefillBoundsError("max_open_tasks cannot exceed 48")
        for name, maximum in (
            ("max_depth", 8),
            ("max_retries", 16),
            ("max_paths", 64),
            ("max_context_paths", 64),
            ("max_residuals", MAX_RESIDUALS),
            ("backoff_identical_attempts", 64),
            ("cooldown_seconds", 86_400),
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=maximum),
            )
        parent = _text(self.parent_goal_id, "parent_goal_id")
        if not _GOAL_ID_RE.fullmatch(parent):
            raise PlannerDoctorRefillError("parent_goal_id is malformed")
        object.__setattr__(self, "parent_goal_id", parent)
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, "resource_class", limit=64),
        )
        object.__setattr__(
            self, "token_class", _text(self.token_class, "token_class", limit=64)
        )
        object.__setattr__(
            self,
            "stop_policy",
            _text(self.stop_policy, "stop_policy", limit=MAX_TEXT_BYTES),
        )
        object.__setattr__(
            self,
            "derived_runtime_source_id",
            _identifier(self.derived_runtime_source_id, "derived_runtime_source_id"),
        )
        anchors = _paths(
            self.protected_anchors,
            "protected_anchors",
            maximum=64,
        )
        object.__setattr__(self, "protected_anchors", anchors or DEFAULT_PROTECTED_ANCHORS)
        for flag in (
            "derived_runtime_admission_enabled",
            "require_independent_compilation",
            "require_parallel_compilation",
            "parallel_review_only",
        ):
            object.__setattr__(self, flag, _bool(getattr(self, flag), flag))
        object.__setattr__(
            self, "actor_id", _identifier(self.actor_id, "actor_id")
        )
        object.__setattr__(
            self,
            "owner_actor_id",
            _identifier(self.owner_actor_id, "owner_actor_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_REFILL_POLICY_SCHEMA,
            "max_goals_per_epoch": self.max_goals_per_epoch,
            "max_tasks_per_epoch": self.max_tasks_per_epoch,
            "max_open_tasks": self.max_open_tasks,
            "max_depth": self.max_depth,
            "max_retries": self.max_retries,
            "max_paths": self.max_paths,
            "max_context_paths": self.max_context_paths,
            "max_residuals": self.max_residuals,
            "backoff_identical_attempts": self.backoff_identical_attempts,
            "cooldown_seconds": self.cooldown_seconds,
            "parent_goal_id": self.parent_goal_id,
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "stop_policy": self.stop_policy,
            "derived_runtime_source_id": self.derived_runtime_source_id,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
            "protected_anchors": list(self.protected_anchors),
            "derived_runtime_admission_enabled": (
                self.derived_runtime_admission_enabled
            ),
            "require_independent_compilation": (
                self.require_independent_compilation
            ),
            "require_parallel_compilation": self.require_parallel_compilation,
            "parallel_review_only": self.parallel_review_only,
            "actor_id": self.actor_id,
            "owner_actor_id": self.owner_actor_id,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "threshold_lower_authority": False,
            "self_authorization": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | None
    ) -> "PlannerDoctorRefillPolicy":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise PlannerDoctorRefillError("policy must be a mapping")
        fields = set(cls.__dataclass_fields__)
        unknown = set(payload) - fields - {
            "schema",
            "derived_runtime_gate",
            "completion_authority",
            "mutation_authority",
            "seed_board_edit",
            "threshold_lower_authority",
            "self_authorization",
        }
        if unknown:
            raise PlannerDoctorRefillError(
                f"policy has unknown fields: {sorted(unknown)}"
            )
        for flag in (
            "completion_authority",
            "mutation_authority",
            "seed_board_edit",
            "threshold_lower_authority",
            "self_authorization",
        ):
            if payload.get(flag):
                raise PlannerDoctorRefillAuthorityError(
                    f"policy cannot claim {flag}"
                )
        kwargs = {key: payload[key] for key in fields if key in payload}
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class PlannerDoctorRefillMemory:
    """Durable backoff / dedupe / open-work state for derived refill."""

    entries: tuple[DoctorPlanRefillMemoryEntry, ...] = ()
    open_task_count: int = 0
    last_source_identity: str = ""
    last_plan_root_cid: str = ""
    now_epoch_s: int = 0

    def __post_init__(self) -> None:
        if len(self.entries) > MAX_MEMORY_ENTRIES:
            raise PlannerDoctorRefillBoundsError("memory entries exceed bound")
        normalized: list[DoctorPlanRefillMemoryEntry] = []
        seen: set[str] = set()
        for item in self.entries:
            if isinstance(item, DoctorPlanRefillMemoryEntry):
                entry = item
            elif isinstance(item, Mapping):
                entry = DoctorPlanRefillMemoryEntry.from_dict(item)
            else:
                raise PlannerDoctorRefillError("memory entries must be objects")
            if entry.identity_key in seen:
                continue
            seen.add(entry.identity_key)
            normalized.append(entry)
        object.__setattr__(self, "entries", tuple(normalized))
        object.__setattr__(
            self,
            "open_task_count",
            _nonneg_int(self.open_task_count, "open_task_count", maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "last_source_identity",
            _optional_text(
                self.last_source_identity, "last_source_identity", limit=MAX_ID_BYTES
            ),
        )
        object.__setattr__(
            self,
            "last_plan_root_cid",
            _optional_text(
                self.last_plan_root_cid, "last_plan_root_cid", limit=MAX_ID_BYTES
            ),
        )
        object.__setattr__(
            self,
            "now_epoch_s",
            _nonneg_int(self.now_epoch_s, "now_epoch_s", maximum=2**63 - 1),
        )

    def by_identity(self) -> dict[str, DoctorPlanRefillMemoryEntry]:
        return {entry.identity_key: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_REFILL_MEMORY_SCHEMA,
            "entries": [entry.to_dict() for entry in self.entries],
            "open_task_count": self.open_task_count,
            "last_source_identity": self.last_source_identity,
            "last_plan_root_cid": self.last_plan_root_cid,
            "now_epoch_s": self.now_epoch_s,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any] | None
    ) -> "PlannerDoctorRefillMemory":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise PlannerDoctorRefillError("memory must be a mapping")
        return cls(
            entries=tuple(payload.get("entries") or ()),
            open_task_count=int(payload.get("open_task_count") or 0),
            last_source_identity=str(payload.get("last_source_identity") or ""),
            last_plan_root_cid=str(payload.get("last_plan_root_cid") or ""),
            now_epoch_s=int(payload.get("now_epoch_s") or 0),
        )


# ---------------------------------------------------------------------------
# Residual input
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DerivedResidual:
    """One benchmark or Doctor residual ready for derived goal/task compilation."""

    issue_id: str
    source_kind: ResidualSourceKind = ResidualSourceKind.DOCTOR
    obligation_id: str = ""
    root_id: str = ""
    attempt_id: str = ""
    source_root: str = ""
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    parent_subgoal_id: str = ""
    predicted_files: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    resource_class: str = DEFAULT_RESOURCE_CLASS
    token_class: str = DEFAULT_TOKEN_CLASS
    stop_policy: str = ""
    title: str = ""
    rationale: str = ""
    unchanged_failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    residual_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_id", _identifier(self.issue_id, "issue_id"))
        object.__setattr__(
            self, "source_kind", _enum(self.source_kind, ResidualSourceKind, "source_kind")
        )
        for name in (
            "obligation_id",
            "root_id",
            "attempt_id",
            "source_root",
            "parent_subgoal_id",
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, required=False),
            )
        parent = _optional_text(self.parent_goal_id, "parent_goal_id")
        if parent and not _GOAL_ID_RE.fullmatch(parent):
            raise PlannerDoctorRefillError("parent_goal_id is malformed")
        object.__setattr__(
            self, "parent_goal_id", parent or DEFAULT_PARENT_GOAL_ID
        )
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files")
        )
        object.__setattr__(
            self,
            "context_paths",
            _paths(
                self.context_paths,
                "context_paths",
                maximum=DEFAULT_MAX_CONTEXT_PATHS,
            ),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _ids(self.predicted_symbols, "predicted_symbols", maximum=32),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _command_strings(self.validation_commands, "validation_commands"),
        )
        if self.acceptance_criteria is None:
            acceptance_items: Sequence[Any] = ()
        elif isinstance(self.acceptance_criteria, str):
            acceptance_items = (self.acceptance_criteria,)
        elif isinstance(self.acceptance_criteria, Sequence) and not isinstance(
            self.acceptance_criteria, (bytes, bytearray)
        ):
            acceptance_items = self.acceptance_criteria
        else:
            raise PlannerDoctorRefillError(
                "acceptance_criteria must be a sequence of strings"
            )
        if len(acceptance_items) > 16:
            raise PlannerDoctorRefillBoundsError(
                "acceptance_criteria exceeds its item bound"
            )
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(
                _text(item, "acceptance_criteria", required=True)
                for item in acceptance_items
            ),
        )
        object.__setattr__(
            self, "dependencies", _ids(self.dependencies, "dependencies", maximum=32)
        )
        object.__setattr__(
            self, "conflicts", _ids(self.conflicts, "conflicts", maximum=32)
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs", maximum=64)
        )
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, "resource_class", limit=64),
        )
        object.__setattr__(
            self, "token_class", _text(self.token_class, "token_class", limit=64)
        )
        object.__setattr__(
            self,
            "stop_policy",
            _optional_text(self.stop_policy, "stop_policy"),
        )
        object.__setattr__(self, "title", _optional_text(self.title, "title"))
        object.__setattr__(
            self, "rationale", _optional_text(self.rationale, "rationale")
        )
        object.__setattr__(
            self, "unchanged_failure", _bool(self.unchanged_failure, "unchanged_failure")
        )
        object.__setattr__(
            self, "metadata", _mapping_proxy(self.metadata, "metadata")
        )
        _reject_authority_claims(self.metadata, where="derived residual")
        rid = _optional_text(self.residual_id, "residual_id", limit=MAX_ID_BYTES)
        object.__setattr__(self, "residual_id", rid or self.identity_key)
        if not self.source_root and self.root_id:
            object.__setattr__(self, "source_root", self.root_id)

    @property
    def identity_key(self) -> str:
        return residual_identity_key(
            issue_id=self.issue_id,
            obligation_id=self.obligation_id,
            root_id=self.root_id or self.source_root,
            attempt_id=self.attempt_id,
        )

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def mutation_authority(self) -> bool:
        return False

    def minimal_paths(self, *, maximum: int = DEFAULT_MAX_PATHS) -> tuple[str, ...]:
        files = list(self.predicted_files)
        for path in self.context_paths:
            if path not in files:
                files.append(path)
        return tuple(files[:maximum])

    def fingerprint(self) -> str:
        payload = {
            "issue_id": self.issue_id,
            "source_kind": self.source_kind.value,
            "obligation_id": self.obligation_id,
            "root_id": self.root_id or self.source_root,
            "attempt_id": self.attempt_id,
            "predicted_files": list(self.predicted_files),
            "context_paths": list(self.context_paths),
            "acceptance_criteria": list(self.acceptance_criteria),
            "validation_commands": list(self.validation_commands),
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "unchanged_failure": self.unchanged_failure,
            "title": self.title,
            "rationale": self.rationale,
        }
        return _stable_digest(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "issue_id": self.issue_id,
            "source_kind": self.source_kind.value,
            "obligation_id": self.obligation_id,
            "root_id": self.root_id,
            "attempt_id": self.attempt_id,
            "source_root": self.source_root,
            "parent_goal_id": self.parent_goal_id,
            "parent_subgoal_id": self.parent_subgoal_id,
            "predicted_files": list(self.predicted_files),
            "context_paths": list(self.context_paths),
            "predicted_symbols": list(self.predicted_symbols),
            "validation_commands": list(self.validation_commands),
            "acceptance_criteria": list(self.acceptance_criteria),
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "evidence_refs": list(self.evidence_refs),
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "stop_policy": self.stop_policy,
            "title": self.title,
            "rationale": self.rationale,
            "unchanged_failure": self.unchanged_failure,
            "metadata": dict(self.metadata),
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DerivedResidual":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorRefillError("residual payload must be a mapping")
        for flag in (
            "completion_authority",
            "mutation_authority",
            "seed_board_edit",
            "threshold_lower_authority",
            "self_authorization",
        ):
            if payload.get(flag):
                raise PlannerDoctorRefillAuthorityError(
                    f"{flag} cannot be true on a derived residual"
                )
        return cls(
            issue_id=payload["issue_id"],
            source_kind=payload.get("source_kind", ResidualSourceKind.DOCTOR),
            obligation_id=payload.get("obligation_id", ""),
            root_id=payload.get("root_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            source_root=payload.get("source_root", ""),
            parent_goal_id=payload.get("parent_goal_id", DEFAULT_PARENT_GOAL_ID),
            parent_subgoal_id=payload.get("parent_subgoal_id", ""),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            context_paths=tuple(payload.get("context_paths") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            validation_commands=tuple(payload.get("validation_commands") or ()),
            acceptance_criteria=tuple(payload.get("acceptance_criteria") or ()),
            dependencies=tuple(payload.get("dependencies") or ()),
            conflicts=tuple(payload.get("conflicts") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            resource_class=payload.get("resource_class", DEFAULT_RESOURCE_CLASS),
            token_class=payload.get("token_class", DEFAULT_TOKEN_CLASS),
            stop_policy=payload.get("stop_policy", ""),
            title=payload.get("title", ""),
            rationale=payload.get("rationale", ""),
            unchanged_failure=bool(payload.get("unchanged_failure", False)),
            metadata=payload.get("metadata") or {},
            residual_id=payload.get("residual_id", ""),
        )

    @classmethod
    def from_doctor_residual(
        cls,
        residual: DoctorPlanResidual | Mapping[str, Any],
        *,
        source_kind: ResidualSourceKind | str = ResidualSourceKind.DOCTOR,
    ) -> "DerivedResidual":
        if isinstance(residual, Mapping):
            residual = DoctorPlanResidual.from_dict(residual)
        if not isinstance(residual, DoctorPlanResidual):
            raise PlannerDoctorRefillError("doctor residual is required")
        kind = ResidualSourceKind.DOCTOR
        if residual.kind is DoctorResidualKind.CAPABILITY_GAP:
            kind = ResidualSourceKind.CAPABILITY
        elif residual.kind is DoctorResidualKind.SECURITY:
            kind = ResidualSourceKind.SECURITY
        if source_kind not in (ResidualSourceKind.DOCTOR, "doctor"):
            kind = (
                source_kind
                if isinstance(source_kind, ResidualSourceKind)
                else ResidualSourceKind(str(source_kind))
            )
        return cls(
            issue_id=residual.issue_id,
            source_kind=kind,
            obligation_id=residual.obligation_id,
            root_id=residual.root_id,
            attempt_id=residual.attempt_id,
            source_root=residual.root_id,
            parent_goal_id=residual.parent_goal_id or DEFAULT_PARENT_GOAL_ID,
            predicted_files=residual.predicted_files,
            context_paths=residual.context_paths,
            predicted_symbols=residual.predicted_symbols,
            validation_commands=residual.validation_commands,
            acceptance_criteria=(
                f"resolve residual identity {residual.identity_key}",
                "no completion authority",
                "no seed board mutation",
            ),
            dependencies=tuple(
                item
                for item in (residual.obligation_id, residual.parent_task_cid)
                if item
            ),
            evidence_refs=residual.evidence_refs,
            title=residual.title,
            rationale=residual.rationale,
            unchanged_failure=residual.unchanged_failure
            or residual.kind is DoctorResidualKind.UNCHANGED_FAILURE,
            metadata=dict(residual.metadata),
            residual_id=residual.residual_id,
        )


def normalize_residuals(
    residuals: Sequence[Any] | None = None,
    *,
    doctor_residuals: Sequence[Any] | None = None,
    benchmark_residuals: Sequence[Any] | None = None,
    fixed_point: Any = None,
    root_id: str = "",
    attempt_id: str = "",
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID,
) -> tuple[DerivedResidual, ...]:
    """Normalize heterogeneous residual inputs into DerivedResidual records."""

    collected: list[DerivedResidual] = []

    def _append(raw: Any, *, default_kind: ResidualSourceKind) -> None:
        if isinstance(raw, DerivedResidual):
            collected.append(raw)
            return
        if isinstance(raw, DoctorPlanResidual):
            collected.append(
                DerivedResidual.from_doctor_residual(raw, source_kind=default_kind)
            )
            return
        if not isinstance(raw, Mapping):
            raise PlannerDoctorRefillError("residual must be a mapping or typed object")
        if "source_kind" in raw or "acceptance_criteria" in raw or "source_root" in raw:
            payload = dict(raw)
            payload.setdefault("source_kind", default_kind.value)
            collected.append(DerivedResidual.from_dict(payload))
            return
        # Doctor residual shape.
        collected.append(
            DerivedResidual.from_doctor_residual(
                DoctorPlanResidual.from_dict(raw), source_kind=default_kind
            )
        )

    for raw in residuals or ():
        _append(raw, default_kind=ResidualSourceKind.DOCTOR)
    for raw in doctor_residuals or ():
        _append(raw, default_kind=ResidualSourceKind.DOCTOR)
    for raw in benchmark_residuals or ():
        _append(raw, default_kind=ResidualSourceKind.BENCHMARK)

    if fixed_point is not None and not fixed_point_is_successful(fixed_point):
        for residual in extract_residuals_from_fixed_point(
            fixed_point,
            root_id=root_id,
            attempt_id=attempt_id,
            parent_goal_id=parent_goal_id,
        ):
            collected.append(DerivedResidual.from_doctor_residual(residual))

    return tuple(collected)


def dedupe_derived_residuals(
    residuals: Sequence[DerivedResidual],
) -> tuple[tuple[DerivedResidual, ...], tuple[str, ...]]:
    unique: list[DerivedResidual] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for residual in residuals:
        key = residual.identity_key
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        unique.append(residual)
    return tuple(unique), tuple(duplicates)


# ---------------------------------------------------------------------------
# Goal / task proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DerivedGoalProposal:
    """One derived goal or subgoal with exact hierarchy and source roots."""

    goal_id: str
    goal_cid: str
    title: str
    parent_goal_id: str = ""
    source_root: str = ""
    residual_ids: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    scope_paths: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    resource_class: str = DEFAULT_RESOURCE_CLASS
    stop_policy: str = DEFAULT_STOP_POLICY
    depth: int = 0
    is_subgoal: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "goal_cid", _identifier(self.goal_cid, "goal_cid"))
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(
            self,
            "parent_goal_id",
            _identifier(self.parent_goal_id, "parent_goal_id", required=False),
        )
        object.__setattr__(
            self,
            "source_root",
            _identifier(self.source_root, "source_root", required=False),
        )
        object.__setattr__(
            self, "residual_ids", _ids(self.residual_ids, "residual_ids", maximum=64)
        )
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(
                _text(item, "acceptance_criteria")
                for item in (
                    self.acceptance_criteria
                    if isinstance(self.acceptance_criteria, Sequence)
                    and not isinstance(self.acceptance_criteria, (str, bytes))
                    else ()
                )
            )[:16],
        )
        object.__setattr__(
            self, "scope_paths", _paths(self.scope_paths, "scope_paths")
        )
        object.__setattr__(
            self,
            "context_paths",
            _paths(self.context_paths, "context_paths", maximum=DEFAULT_MAX_CONTEXT_PATHS),
        )
        object.__setattr__(
            self, "dependencies", _ids(self.dependencies, "dependencies", maximum=32)
        )
        object.__setattr__(
            self, "conflicts", _ids(self.conflicts, "conflicts", maximum=32)
        )
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, "resource_class", limit=64),
        )
        object.__setattr__(
            self, "stop_policy", _text(self.stop_policy, "stop_policy")
        )
        object.__setattr__(
            self, "depth", _nonneg_int(self.depth, "depth", maximum=16)
        )
        object.__setattr__(
            self, "is_subgoal", _bool(self.is_subgoal, "is_subgoal")
        )
        object.__setattr__(
            self, "rationale", _optional_text(self.rationale, "rationale")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DERIVED_GOAL_PROPOSAL_SCHEMA,
            "goal_id": self.goal_id,
            "goal_cid": self.goal_cid,
            "title": self.title,
            "parent_goal_id": self.parent_goal_id,
            "source_root": self.source_root,
            "residual_ids": list(self.residual_ids),
            "acceptance_criteria": list(self.acceptance_criteria),
            "scope_paths": list(self.scope_paths),
            "context_paths": list(self.context_paths),
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "resource_class": self.resource_class,
            "stop_policy": self.stop_policy,
            "depth": self.depth,
            "is_subgoal": self.is_subgoal,
            "rationale": self.rationale,
            "completion_authority": False,
            "mutation_authority": False,
        }


@dataclass(frozen=True, slots=True)
class DerivedTaskProposal:
    """One derived task with minimal files/context and validation."""

    task_id: str
    task_cid: str
    goal_id: str
    goal_cid: str
    title: str
    source_root: str = ""
    residual_id: str = ""
    identity_key: str = ""
    predicted_files: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    resource_class: str = DEFAULT_RESOURCE_CLASS
    token_class: str = DEFAULT_TOKEN_CLASS
    stop_policy: str = DEFAULT_STOP_POLICY
    rationale: str = ""
    depth: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "task_cid", _identifier(self.task_cid, "task_cid"))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "goal_cid", _identifier(self.goal_cid, "goal_cid"))
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(
            self,
            "source_root",
            _identifier(self.source_root, "source_root", required=False),
        )
        object.__setattr__(
            self,
            "residual_id",
            _identifier(self.residual_id, "residual_id", required=False),
        )
        object.__setattr__(
            self,
            "identity_key",
            _optional_text(self.identity_key, "identity_key", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files")
        )
        object.__setattr__(
            self,
            "context_paths",
            _paths(
                self.context_paths,
                "context_paths",
                maximum=DEFAULT_MAX_CONTEXT_PATHS,
            ),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _ids(self.predicted_symbols, "predicted_symbols", maximum=32),
        )
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(
                _text(item, "acceptance_criteria")
                for item in (
                    self.acceptance_criteria
                    if isinstance(self.acceptance_criteria, Sequence)
                    and not isinstance(self.acceptance_criteria, (str, bytes))
                    else ()
                )
            )[:16],
        )
        object.__setattr__(
            self,
            "validation_commands",
            _command_strings(self.validation_commands, "validation_commands"),
        )
        object.__setattr__(
            self, "dependencies", _ids(self.dependencies, "dependencies", maximum=32)
        )
        object.__setattr__(
            self, "conflicts", _ids(self.conflicts, "conflicts", maximum=32)
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs", maximum=64)
        )
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, "resource_class", limit=64),
        )
        object.__setattr__(
            self, "token_class", _text(self.token_class, "token_class", limit=64)
        )
        object.__setattr__(
            self, "stop_policy", _text(self.stop_policy, "stop_policy")
        )
        object.__setattr__(
            self, "rationale", _optional_text(self.rationale, "rationale")
        )
        object.__setattr__(
            self, "depth", _nonneg_int(self.depth, "depth", maximum=16)
        )

    def to_work_proposal(self, *, parent_goal_id: str) -> ObjectiveWorkProposal:
        family_key = (
            _OBJECTIVE_FAMILY_PREFIX
            + _stable_digest({"goal": self.goal_id, "task": self.task_id})[
                len("sha256:") : len("sha256:") + 16
            ]
        )
        instance_key = (
            _OBJECTIVE_INSTANCE_PREFIX
            + (self.identity_key or self.task_cid).split(":")[-1][:32]
        )
        return ObjectiveWorkProposal(
            kind=ObjectiveWorkKind.TASK,
            title=self.title,
            parent_goal_id=parent_goal_id or self.goal_id,
            parent_objective_terms=(self.goal_id, "planner-doctor-derived"),
            expected_evidence_delta=self.evidence_refs
            or (f"task:{self.task_id}", "derived-refill"),
            dependencies=self.dependencies,
            predicted_files=self.predicted_files,
            predicted_symbols=self.predicted_symbols,
            validation_commands=self.validation_commands
            or (
                "python -m pytest "
                "test/api/test_agent_supervisor_planner_doctor_refill.py -q",
            ),
            confidence=0.7,
            estimated_cost=1.0,
            novelty=1.0,
            depth=self.depth,
            source=PRODUCER_ID,
            source_id=self.residual_id or self.task_id,
            rationale=self.rationale
            or "Bounded derived work compiled from planner/Doctor residual.",
            family_key=family_key,
            instance_key=instance_key,
            semantic_key="",
            canonical_id="",
            acceptance_subset=self.acceptance_criteria
            or (
                "resolve residual with evidence",
                "no completion authority",
                "no seed board mutation",
            ),
            preconditions=(
                "independent formal plan compilation",
                "independent structural admission",
                "independent parallel plan compilation",
                f"source_root={self.source_root}" if self.source_root else "source_root bound",
            ),
            effects=(
                f"target residual {self.residual_id or self.task_id}",
                "derived runtime only; no seed mutation",
            ),
            evidence_subset=self.evidence_refs,
            conflicts=self.conflicts,
            context_paths=self.context_paths or self.predicted_files,
            resource_class=self.resource_class,
            token_class=self.token_class,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DERIVED_TASK_PROPOSAL_SCHEMA,
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "goal_id": self.goal_id,
            "goal_cid": self.goal_cid,
            "title": self.title,
            "source_root": self.source_root,
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "predicted_files": list(self.predicted_files),
            "context_paths": list(self.context_paths),
            "predicted_symbols": list(self.predicted_symbols),
            "acceptance_criteria": list(self.acceptance_criteria),
            "validation_commands": list(self.validation_commands),
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "evidence_refs": list(self.evidence_refs),
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "stop_policy": self.stop_policy,
            "rationale": self.rationale,
            "depth": self.depth,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
        }


@dataclass(frozen=True, slots=True)
class ResidualDecision:
    residual_id: str
    identity_key: str
    disposition: ResidualDisposition
    reason_codes: tuple[str, ...] = ()
    goal_id: str = ""
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "goal_id": self.goal_id,
            "task_id": self.task_id,
        }


# ---------------------------------------------------------------------------
# Compilation / admission receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DerivedCompilationReceipt:
    """Independent formal + parallel compilation identities for one population."""

    formal_plan_id: str
    source_identity: str
    plan_root_cid: str
    parallel_plan_digest: str
    parallel_outcome: str
    goal_count: int
    task_count: int
    formal_status: str
    admitted_for_parallel: bool
    formal_input: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formal_plan_id",
            _identifier(self.formal_plan_id, "formal_plan_id"),
        )
        object.__setattr__(
            self,
            "source_identity",
            _identifier(self.source_identity, "source_identity"),
        )
        object.__setattr__(
            self,
            "plan_root_cid",
            _identifier(self.plan_root_cid, "plan_root_cid", required=False)
            or self.formal_plan_id,
        )
        object.__setattr__(
            self,
            "parallel_plan_digest",
            _identifier(self.parallel_plan_digest, "parallel_plan_digest"),
        )
        object.__setattr__(
            self,
            "parallel_outcome",
            _text(self.parallel_outcome, "parallel_outcome", limit=64),
        )
        object.__setattr__(
            self,
            "goal_count",
            _nonneg_int(self.goal_count, "goal_count", maximum=64),
        )
        object.__setattr__(
            self,
            "task_count",
            _nonneg_int(self.task_count, "task_count", maximum=256),
        )
        object.__setattr__(
            self,
            "formal_status",
            _text(self.formal_status, "formal_status", limit=64),
        )
        object.__setattr__(
            self,
            "admitted_for_parallel",
            _bool(self.admitted_for_parallel, "admitted_for_parallel"),
        )
        object.__setattr__(
            self, "formal_input", _mapping_proxy(self.formal_input, "formal_input")
        )

    @property
    def receipt_cid(self) -> str:
        return content_identity(
            {
                "schema": DERIVED_COMPILATION_RECEIPT_SCHEMA,
                "formal_plan_id": self.formal_plan_id,
                "source_identity": self.source_identity,
                "plan_root_cid": self.plan_root_cid,
                "parallel_plan_digest": self.parallel_plan_digest,
                "parallel_outcome": self.parallel_outcome,
                "goal_count": self.goal_count,
                "task_count": self.task_count,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DERIVED_COMPILATION_RECEIPT_SCHEMA,
            "receipt_cid": self.receipt_cid,
            "formal_plan_id": self.formal_plan_id,
            "source_identity": self.source_identity,
            "plan_root_cid": self.plan_root_cid,
            "parallel_plan_digest": self.parallel_plan_digest,
            "parallel_outcome": self.parallel_outcome,
            "goal_count": self.goal_count,
            "task_count": self.task_count,
            "formal_status": self.formal_status,
            "admitted_for_parallel": self.admitted_for_parallel,
            # formal_input is large; omit body by default, keep identity only.
            "formal_input_bound": bool(self.formal_input),
        }


@dataclass(frozen=True, slots=True)
class DerivedAdmissionReceipt:
    """Structural admission for derived refill (independent of candidate claims)."""

    admitted: bool
    admission_receipt_cid: str
    compilation: DerivedCompilationReceipt
    reason_codes: tuple[str, ...] = ()
    protected_anchor_hits: tuple[str, ...] = ()
    open_task_count: int = 0
    stop_policy: str = DEFAULT_STOP_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self,
            "admission_receipt_cid",
            _identifier(self.admission_receipt_cid, "admission_receipt_cid"),
        )
        if not isinstance(self.compilation, DerivedCompilationReceipt):
            raise PlannerDoctorRefillError("compilation receipt is required")
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", maximum=64)
        )
        object.__setattr__(
            self,
            "protected_anchor_hits",
            _paths(self.protected_anchor_hits, "protected_anchor_hits", maximum=64),
        )
        object.__setattr__(
            self,
            "open_task_count",
            _nonneg_int(self.open_task_count, "open_task_count", maximum=1_000_000),
        )
        object.__setattr__(
            self, "stop_policy", _text(self.stop_policy, "stop_policy")
        )
        if self.admitted and self.reason_codes:
            # Soft warnings allowed only when not hard-fail codes.
            hard = {
                "formal_not_compiled",
                "parallel_not_admitted",
                "protected_anchor",
                "authority_claim",
                "open_work_ceiling",
                "goal_bound",
                "task_bound",
                "missing_source_root",
            }
            if any(code in hard for code in self.reason_codes):
                raise PlannerDoctorRefillAdmissionError(
                    "admitted receipt cannot carry hard rejection codes"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DERIVED_ADMISSION_RECEIPT_SCHEMA,
            "admitted": self.admitted,
            "admission_receipt_cid": self.admission_receipt_cid,
            "compilation": self.compilation.to_dict(),
            "reason_codes": list(self.reason_codes),
            "protected_anchor_hits": list(self.protected_anchor_hits),
            "open_task_count": self.open_task_count,
            "stop_policy": self.stop_policy,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "threshold_lower_authority": False,
            "self_authorization": False,
        }


@dataclass(frozen=True, slots=True)
class PlannerDoctorRefillReceipt:
    """Result of one derived residual compilation epoch."""

    disposition: PlannerDoctorRefillDisposition
    residuals: tuple[DerivedResidual, ...] = ()
    decisions: tuple[ResidualDecision, ...] = ()
    goals: tuple[DerivedGoalProposal, ...] = ()
    tasks: tuple[DerivedTaskProposal, ...] = ()
    work_proposals: tuple[ObjectiveWorkProposal, ...] = ()
    compilation: DerivedCompilationReceipt | None = None
    admission: DerivedAdmissionReceipt | None = None
    materialization: Mapping[str, Any] = field(default_factory=dict)
    backoff_identity_keys: tuple[str, ...] = ()
    duplicate_identity_keys: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    policy: PlannerDoctorRefillPolicy = field(
        default_factory=PlannerDoctorRefillPolicy
    )
    next_memory: PlannerDoctorRefillMemory = field(
        default_factory=PlannerDoctorRefillMemory
    )
    repository_tree_id: str = ""
    epoch_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PlannerDoctorRefillDisposition, "disposition"),
        )
        if not isinstance(self.policy, PlannerDoctorRefillPolicy):
            object.__setattr__(
                self, "policy", PlannerDoctorRefillPolicy.from_dict(self.policy)
            )
        if not isinstance(self.next_memory, PlannerDoctorRefillMemory):
            object.__setattr__(
                self,
                "next_memory",
                PlannerDoctorRefillMemory.from_dict(self.next_memory),
            )
        object.__setattr__(
            self,
            "materialization",
            _mapping_proxy(self.materialization, "materialization"),
        )
        object.__setattr__(
            self,
            "backoff_identity_keys",
            _ids(self.backoff_identity_keys, "backoff_identity_keys", maximum=256),
        )
        object.__setattr__(
            self,
            "duplicate_identity_keys",
            _ids(self.duplicate_identity_keys, "duplicate_identity_keys", maximum=256),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", maximum=64)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _identifier(self.repository_tree_id, "repository_tree_id", required=False),
        )
        object.__setattr__(
            self,
            "epoch_id",
            _identifier(self.epoch_id, "epoch_id", required=False),
        )
        # Authority hardening on the receipt itself.
        if self.materialization:
            _reject_authority_claims(self.materialization, where="materialization")

    @property
    def emits_work(self) -> bool:
        return bool(self.goals or self.tasks or self.work_proposals)

    @property
    def derived_runtime_admitted(self) -> bool:
        return bool(self.admission and self.admission.admitted)

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def mutation_authority(self) -> bool:
        return False

    @property
    def receipt_id(self) -> str:
        return content_identity(
            {
                "schema": PLANNER_DOCTOR_REFILL_RECEIPT_SCHEMA,
                "disposition": self.disposition.value,
                "goal_ids": [goal.goal_id for goal in self.goals],
                "task_ids": [task.task_id for task in self.tasks],
                "reason_codes": list(self.reason_codes),
                "admission": (
                    self.admission.admission_receipt_cid if self.admission else ""
                ),
                "source_identity": (
                    self.compilation.source_identity if self.compilation else ""
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_REFILL_RECEIPT_SCHEMA,
            "interface": PLANNER_DOCTOR_REFILL_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "receipt_id": self.receipt_id,
            "disposition": self.disposition.value,
            "emits_work": self.emits_work,
            "residuals": [item.to_dict() for item in self.residuals],
            "decisions": [item.to_dict() for item in self.decisions],
            "goals": [item.to_dict() for item in self.goals],
            "tasks": [item.to_dict() for item in self.tasks],
            "work_proposals": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in self.work_proposals
            ],
            "compilation": self.compilation.to_dict() if self.compilation else None,
            "admission": self.admission.to_dict() if self.admission else None,
            "materialization": dict(self.materialization),
            "backoff_identity_keys": list(self.backoff_identity_keys),
            "duplicate_identity_keys": list(self.duplicate_identity_keys),
            "reason_codes": list(self.reason_codes),
            "policy": self.policy.to_dict(),
            "next_memory": self.next_memory.to_dict(),
            "repository_tree_id": self.repository_tree_id,
            "epoch_id": self.epoch_id,
            "derived_runtime_admitted": self.derived_runtime_admitted,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
            "derived_runtime_source_id": self.policy.derived_runtime_source_id,
            "stop_policy": self.policy.stop_policy,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "threshold_lower_authority": False,
            "self_authorization": False,
        }


# ---------------------------------------------------------------------------
# Hierarchy compilation
# ---------------------------------------------------------------------------


def _goal_cid_for(goal_id: str, source_root: str) -> str:
    return content_identity(
        {
            "namespace": "planner-doctor-derived-goal",
            "goal_id": goal_id,
            "source_root": source_root,
        }
    )


def _task_cid_for(task_id: str, goal_cid: str, identity_key: str) -> str:
    return content_identity(
        {
            "namespace": "planner-doctor-derived-task",
            "task_id": task_id,
            "goal_cid": goal_cid,
            "identity_key": identity_key,
        }
    )


def _group_key(residual: DerivedResidual) -> str:
    # One goal per source root + source kind keeps hierarchy shallow and stable.
    root = residual.source_root or residual.root_id or "root:unbound"
    return f"{residual.source_kind.value}::{root}"


def compile_hierarchy(
    residuals: Sequence[DerivedResidual],
    *,
    policy: PlannerDoctorRefillPolicy,
    memory: PlannerDoctorRefillMemory,
) -> tuple[
    tuple[DerivedGoalProposal, ...],
    tuple[DerivedTaskProposal, ...],
    tuple[ResidualDecision, ...],
    tuple[str, ...],
    tuple[str, ...],
    list[str],
    dict[str, DoctorPlanRefillMemoryEntry],
]:
    """Compile residuals into bounded goals/subgoals/tasks with backoff."""

    unique, duplicates = dedupe_derived_residuals(residuals)
    if len(unique) > policy.max_residuals:
        unique = unique[: policy.max_residuals]

    memory_index = memory.by_identity()
    next_entries: dict[str, DoctorPlanRefillMemoryEntry] = dict(memory_index)
    decisions: list[ResidualDecision] = []
    backoff_keys: list[str] = []
    reason_codes: list[str] = []

    if memory.open_task_count >= policy.max_open_tasks:
        reason_codes.append("open_work_ceiling")
        for residual in unique:
            decisions.append(
                ResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=ResidualDisposition.BOUND_REJECTED,
                    reason_codes=("open_work_ceiling",),
                )
            )
        return (), (), tuple(decisions), tuple(duplicates), tuple(backoff_keys), reason_codes, next_entries

    # Group residuals by source root for goal formation.
    groups: dict[str, list[DerivedResidual]] = {}
    for residual in unique:
        fingerprint = residual.fingerprint()
        prior = memory_index.get(residual.identity_key)

        if residual.unchanged_failure:
            backoff_keys.append(residual.identity_key)
            decisions.append(
                ResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=ResidualDisposition.UNCHANGED_BACKOFF,
                    reason_codes=("unchanged_failure",),
                )
            )
            next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                identity_key=residual.identity_key,
                attempt_count=(prior.attempt_count + 1) if prior else 1,
                last_fingerprint=fingerprint,
                last_seen_epoch_s=memory.now_epoch_s,
                last_disposition=ResidualDisposition.UNCHANGED_BACKOFF.value,
            )
            continue

        if (
            prior is not None
            and prior.last_fingerprint == fingerprint
            and prior.attempt_count >= policy.backoff_identical_attempts
        ):
            if (
                policy.cooldown_seconds > 0
                and memory.now_epoch_s > 0
                and prior.last_seen_epoch_s > 0
                and (memory.now_epoch_s - prior.last_seen_epoch_s)
                < policy.cooldown_seconds
            ) or policy.cooldown_seconds == 0 or memory.now_epoch_s == 0:
                backoff_keys.append(residual.identity_key)
                decisions.append(
                    ResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=ResidualDisposition.UNCHANGED_BACKOFF,
                        reason_codes=("identical_fingerprint_backoff",),
                    )
                )
                next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                    identity_key=residual.identity_key,
                    attempt_count=prior.attempt_count + 1,
                    last_fingerprint=fingerprint,
                    last_seen_epoch_s=memory.now_epoch_s,
                    last_disposition=ResidualDisposition.UNCHANGED_BACKOFF.value,
                )
                continue

        # Reject protected-anchor edit attempts.
        paths = residual.minimal_paths(maximum=policy.max_paths)
        hits = [p for p in paths if _path_hits_protected(p, policy.protected_anchors)]
        if hits:
            decisions.append(
                ResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=ResidualDisposition.ANCHOR_REJECTED,
                    reason_codes=("protected_anchor",),
                )
            )
            reason_codes.append("protected_anchor")
            continue

        groups.setdefault(_group_key(residual), []).append(residual)

    goals: list[DerivedGoalProposal] = []
    tasks: list[DerivedTaskProposal] = []
    open_budget = policy.max_open_tasks - memory.open_task_count
    task_budget = min(policy.max_tasks_per_epoch, open_budget)
    goal_budget = policy.max_goals_per_epoch

    # Parent goal (depth 0) once, then subgoals per group, then tasks.
    parent_goal_id = policy.parent_goal_id
    # Use first residual's source root for the parent if present.
    first_root = ""
    if groups:
        first_residual = next(iter(groups.values()))[0]
        first_root = first_residual.source_root or first_residual.root_id

    if groups and goal_budget >= 1:
        parent = DerivedGoalProposal(
            goal_id=parent_goal_id,
            goal_cid=_goal_cid_for(parent_goal_id, first_root),
            title=f"Derived residual epoch under {parent_goal_id}",
            parent_goal_id="",
            source_root=first_root,
            residual_ids=tuple(
                residual.residual_id
                for items in groups.values()
                for residual in items
            )[:64],
            acceptance_criteria=(
                "compile only evidence-covered residuals",
                "never mutate seed anchors",
                "stop on open-work and epoch bounds",
            ),
            scope_paths=(),
            context_paths=(),
            resource_class=policy.resource_class,
            stop_policy=policy.stop_policy,
            depth=0,
            is_subgoal=False,
            rationale="Root goal for planner-doctor derived refill (PDR-081).",
        )
        goals.append(parent)
        goal_budget -= 1

    for group_index, (group_key, items) in enumerate(sorted(groups.items())):
        if task_budget <= 0:
            reason_codes.append("task_bound")
            for residual in items:
                decisions.append(
                    ResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=ResidualDisposition.BOUND_REJECTED,
                        reason_codes=("task_bound",),
                    )
                )
            continue

        source_root = items[0].source_root or items[0].root_id
        subgoal_id = f"{parent_goal_id}.S{group_index + 1:02d}"
        subgoal_cid = _goal_cid_for(subgoal_id, source_root)
        if goal_budget >= 1 and policy.max_depth >= 1:
            scope: list[str] = []
            for residual in items:
                for path in residual.minimal_paths(maximum=policy.max_paths):
                    if path not in scope:
                        scope.append(path)
            subgoal = DerivedGoalProposal(
                goal_id=subgoal_id,
                goal_cid=subgoal_cid,
                title=f"Derived subgoal for {group_key}",
                parent_goal_id=parent_goal_id,
                source_root=source_root,
                residual_ids=tuple(item.residual_id for item in items),
                acceptance_criteria=(
                    "resolve each residual with evidence",
                    "respect stop policy bounds",
                ),
                scope_paths=tuple(scope[: policy.max_paths]),
                context_paths=tuple(scope[: policy.max_context_paths]),
                resource_class=policy.resource_class,
                stop_policy=policy.stop_policy,
                depth=1,
                is_subgoal=True,
                rationale=f"Subgoal grouping residuals under {group_key}.",
            )
            goals.append(subgoal)
            goal_budget -= 1
            task_parent_goal_id = subgoal_id
            task_parent_goal_cid = subgoal_cid
        else:
            if not goals:
                # Ensure at least one goal when tasks emit.
                goals.append(
                    DerivedGoalProposal(
                        goal_id=parent_goal_id,
                        goal_cid=_goal_cid_for(parent_goal_id, source_root),
                        title=f"Derived residual epoch under {parent_goal_id}",
                        source_root=source_root,
                        acceptance_criteria=("bounded derived work",),
                        stop_policy=policy.stop_policy,
                        depth=0,
                    )
                )
            task_parent_goal_id = goals[-1].goal_id
            task_parent_goal_cid = goals[-1].goal_cid

        for residual_index, residual in enumerate(items):
            if task_budget <= 0:
                decisions.append(
                    ResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=ResidualDisposition.BOUND_REJECTED,
                        reason_codes=("task_bound",),
                    )
                )
                reason_codes.append("task_bound")
                continue

            task_id = f"PDR-D{group_index + 1:02d}{residual_index + 1:02d}"
            task_cid = _task_cid_for(
                task_id, task_parent_goal_cid, residual.identity_key
            )
            paths = residual.minimal_paths(maximum=policy.max_paths)
            context = residual.context_paths[: policy.max_context_paths] or paths
            acceptance = residual.acceptance_criteria or (
                f"resolve residual identity {residual.identity_key}",
                "no completion authority",
                "no seed board mutation",
            )
            validation = residual.validation_commands or (
                "python -m pytest "
                "test/api/test_agent_supervisor_planner_doctor_refill.py -q",
            )
            title = residual.title or (
                f"Derived residual successor for {residual.issue_id}"
            )
            task = DerivedTaskProposal(
                task_id=task_id,
                task_cid=task_cid,
                goal_id=task_parent_goal_id,
                goal_cid=task_parent_goal_cid,
                title=title,
                source_root=source_root,
                residual_id=residual.residual_id,
                identity_key=residual.identity_key,
                predicted_files=paths,
                context_paths=context,
                predicted_symbols=residual.predicted_symbols,
                acceptance_criteria=acceptance,
                validation_commands=validation,
                dependencies=residual.dependencies,
                conflicts=residual.conflicts,
                evidence_refs=residual.evidence_refs
                or (
                    f"issue:{residual.issue_id}",
                    f"kind:{residual.source_kind.value}",
                ),
                resource_class=residual.resource_class or policy.resource_class,
                token_class=residual.token_class or policy.token_class,
                stop_policy=residual.stop_policy or policy.stop_policy,
                rationale=residual.rationale
                or "Bounded derived task for planner-doctor residual.",
                depth=min(2, policy.max_depth),
            )
            tasks.append(task)
            task_budget -= 1
            decisions.append(
                ResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=ResidualDisposition.TASK,
                    reason_codes=("compiled_task",),
                    goal_id=task_parent_goal_id,
                    task_id=task_id,
                )
            )
            prior = memory_index.get(residual.identity_key)
            next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                identity_key=residual.identity_key,
                attempt_count=(prior.attempt_count + 1) if prior else 1,
                last_fingerprint=residual.fingerprint(),
                last_seen_epoch_s=memory.now_epoch_s,
                last_disposition=ResidualDisposition.TASK.value,
            )

    if duplicates:
        reason_codes.append("duplicates_collapsed")
    if backoff_keys and not tasks:
        reason_codes.append("all_residuals_backed_off")

    # Goal bound notice.
    if len(goals) >= policy.max_goals_per_epoch and len(groups) > len(
        [g for g in goals if g.is_subgoal]
    ):
        reason_codes.append("goal_bound")

    return (
        tuple(goals),
        tuple(tasks),
        tuple(decisions),
        tuple(dict.fromkeys(duplicates)),
        tuple(dict.fromkeys(backoff_keys)),
        reason_codes,
        next_entries,
    )


def build_formal_plan_input(
    goals: Sequence[DerivedGoalProposal],
    tasks: Sequence[DerivedTaskProposal],
    *,
    repository_tree_id: str,
    policy: PlannerDoctorRefillPolicy,
) -> dict[str, Any]:
    """Project derived goals/tasks into FormalPlanCompiler records."""

    tree_id = _identifier(repository_tree_id, "repository_tree_id")
    if not goals:
        raise PlannerDoctorRefillError("formal plan input requires at least one goal")
    if not tasks:
        raise PlannerDoctorRefillError("formal plan input requires at least one task")

    root = next((goal for goal in goals if not goal.is_subgoal), goals[0])

    # The formal compiler binds tasks to objective goal_id aliases.  Keep the
    # hierarchy in DerivedGoalProposal records, but project a single root
    # objective so every taskboard row resolves cleanly.
    objectives = [
        {
            "goal_id": root.goal_id,
            "goal_cid": root.goal_cid,
            "owner_actor_id": policy.owner_actor_id,
            "title": root.title,
            "acceptance_criteria": list(
                root.acceptance_criteria
                or ("bounded derived residual work",)
            ),
        }
    ]

    taskboard: list[dict[str, Any]] = []
    ast_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    for task in tasks:
        paths = task.predicted_files or task.context_paths
        if not paths:
            paths = (
                "ipfs_accelerate_py/agent_supervisor/objectives/"
                "planner_doctor_refill.py",
            )
        primary = paths[0]
        symbol_cid = content_identity(
            {
                "namespace": "planner-doctor-derived-symbol",
                "task_cid": task.task_cid,
                "path": primary,
            }
        )
        effects = [
            {
                "effect_id": content_identity(
                    {
                        "namespace": "planner-doctor-derived-effect",
                        "task_cid": task.task_cid,
                        "path": path,
                    }
                ),
                "operation": "assign",
                "fluent_id": f"output:{path}",
                "path": path,
                "value": "modify",
            }
            for path in paths[: policy.max_paths]
        ]
        # Bind every task to the root objective alias (not subgoal cid).
        taskboard.append(
            {
                "task_id": task.task_id,
                "task_cid": task.task_cid,
                "goal_id": root.goal_id,
                "actor_id": policy.actor_id,
                "depends_on": list(task.dependencies),
                "resource_needs": [task.resource_class, "duckdb"],
                "changed_ast_scopes": [symbol_cid],
                "acceptance_criteria": list(
                    task.acceptance_criteria
                    or ("resolve residual with evidence",)
                ),
                "validation_commands": list(task.validation_commands),
                "effects": effects,
            }
        )
        ast_records.append(
            {
                "symbol_cid": symbol_cid,
                "tree_cid": tree_id,
                "task_cid": task.task_cid,
                "symbol": task.task_id.replace("-", "_"),
            }
        )
        evidence_cid = content_identity(
            {
                "namespace": "planner-doctor-derived-evidence",
                "task_cid": task.task_cid,
                "residual_id": task.residual_id,
            }
        )
        evidence_records.append(
            {
                "evidence_cid": evidence_cid,
                "task_cid": task.task_cid,
                "kind": "test",
            }
        )

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-input@1",
        "repository_tree_id": tree_id,
        "objectives": objectives,
        "taskboard": taskboard,
        "ast_records": ast_records,
        "proof_policy": {
            "policy_cid": content_identity(
                {
                    "namespace": "planner-doctor-derived-policy",
                    "tree": tree_id,
                    "stop_policy": policy.stop_policy,
                }
            ),
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
        },
        "evidence_records": evidence_records,
    }


def _parallel_tasks_from_proposals(
    tasks: Sequence[DerivedTaskProposal],
) -> list[dict[str, Any]]:
    population: list[dict[str, Any]] = []
    for task in tasks:
        outputs = list(task.predicted_files or task.context_paths)
        if not outputs:
            outputs = [
                "ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py"
            ]
        population.append(
            {
                "task_id": task.task_id,
                "task_cid": task.task_cid,
                "outputs": outputs,
                "produces": [f"leaf:{task.task_id}"],
                "depends_on": list(task.dependencies),
                "duration_ms": 1_000,
                "resource_contract": {
                    "resource_class": "cpu-small",
                    "resource_stage": "implementation",
                    "cpu_slots": 1,
                    "process_slots": 1,
                    "memory_bytes": 100,
                    "disk_bytes": 100,
                },
                "lease_contract": {
                    "lease_scope": "task",
                    "lease_duration_ms": 20_000,
                    "heartbeat_interval_ms": 2_000,
                },
                "worktree_contract": {
                    "policy": "isolated",
                    "isolation_required": True,
                },
                "merge_strategy": {
                    "merge_train_id": "merge-train:derived-runtime",
                    "post_merge_validation": list(task.validation_commands[:1])
                    or ["validation:derived"],
                },
            }
        )
    return population


def compile_independent_gates(
    goals: Sequence[DerivedGoalProposal],
    tasks: Sequence[DerivedTaskProposal],
    *,
    policy: PlannerDoctorRefillPolicy,
    repository_tree_id: str,
    current_time_ms: int = 1_000_000,
) -> DerivedCompilationReceipt:
    """Run independent formal plan + parallel plan compilation."""

    formal_input = build_formal_plan_input(
        goals, tasks, repository_tree_id=repository_tree_id, policy=policy
    )
    formal_result = FormalPlanCompiler().compile(formal_input)
    # Abstractions are expected for descriptive fields; only hard ERROR / non-
    # compiled status fails the independent formal gate.
    hard_errors = [
        issue
        for issue in (formal_result.issues or ())
        if str(getattr(getattr(issue, "severity", None), "value", getattr(issue, "severity", ""))).lower()
        == "error"
        or str(getattr(issue, "severity", "")).lower().endswith("error")
    ]
    if (
        formal_result.status is not CompilationStatus.COMPILED
        or not formal_result.plan_id
        or not formal_result.source_identity
        or hard_errors
    ):
        diagnostics = "; ".join(
            getattr(issue, "message", str(issue))
            for issue in (hard_errors or formal_result.issues or ())[:5]
        )
        raise PlannerDoctorRefillAdmissionError(
            "formal plan compilation failed"
            + (f": {diagnostics}" if diagnostics else "")
        )

    # Second independent compile (identity must match).
    recompile = FormalPlanCompiler().compile(formal_input)
    if (
        recompile.plan_id != formal_result.plan_id
        or recompile.source_identity != formal_result.source_identity
    ):
        raise PlannerDoctorRefillAdmissionError(
            "independent formal recompilation disagreed with the first compile"
        )

    parallel_digest = ""
    parallel_outcome = ParallelPlanOutcome.REJECTED.value
    parallel_admitted = False
    if policy.require_parallel_compilation:
        parallel = ParallelPlanCompiler().compile(
            tasks=_parallel_tasks_from_proposals(tasks),
            requested_width=1,
            capacity_snapshot={
                "snapshot_id": "capacity:derived-refill",
                "observed_at_ms": current_time_ms,
                "fresh_until_ms": current_time_ms + 60_000,
                "cpu_slots": 8,
                "process_slots": 8,
                "memory_bytes": 8_000,
                "gpu_memory_bytes": 2_000,
                "disk_bytes": 20_000,
                "resource_class_slots": {"cpu-small": 8},
            },
            repository_snapshot={
                "tree_id": repository_tree_id,
                "snapshot_id": f"repository-snapshot:{repository_tree_id}",
                "fencing_epoch": 1,
            },
            current_time_ms=current_time_ms,
            review_only=policy.parallel_review_only,
            protected_paths=policy.protected_anchors,
        )
        parallel_outcome = (
            parallel.outcome.value
            if hasattr(parallel.outcome, "value")
            else str(parallel.outcome)
        )
        parallel_admitted = bool(parallel.admitted)
        parallel_digest = str(
            getattr(parallel, "input_digest", "")
            or content_identity(
                {
                    "namespace": "parallel-plan-fallback",
                    "tasks": [task.task_cid for task in tasks],
                    "outcome": parallel_outcome,
                }
            )
        )
        if not parallel_admitted:
            raise PlannerDoctorRefillAdmissionError(
                f"parallel plan compilation rejected: {parallel_outcome}"
            )
    else:
        parallel_digest = content_identity(
            {
                "namespace": "parallel-plan-skipped",
                "tasks": [task.task_cid for task in tasks],
            }
        )
        parallel_outcome = "skipped"
        parallel_admitted = True

    return DerivedCompilationReceipt(
        formal_plan_id=formal_result.plan_id,
        source_identity=formal_result.source_identity,
        plan_root_cid=formal_result.plan_id,
        parallel_plan_digest=parallel_digest,
        parallel_outcome=parallel_outcome,
        goal_count=len(goals),
        task_count=len(tasks),
        formal_status=formal_result.status.value
        if hasattr(formal_result.status, "value")
        else str(formal_result.status),
        admitted_for_parallel=parallel_admitted,
        formal_input=formal_input,
    )


def admit_compiled_population(
    goals: Sequence[DerivedGoalProposal],
    tasks: Sequence[DerivedTaskProposal],
    *,
    compilation: DerivedCompilationReceipt,
    policy: PlannerDoctorRefillPolicy,
    open_task_count: int,
) -> DerivedAdmissionReceipt:
    """Structurally admit a compiled population fail-closed."""

    reasons: list[str] = []
    hits: list[str] = []

    if compilation.formal_status not in {
        CompilationStatus.COMPILED.value,
        "compiled",
    }:
        reasons.append("formal_not_compiled")
    if policy.require_parallel_compilation and not compilation.admitted_for_parallel:
        reasons.append("parallel_not_admitted")
    if len(goals) > policy.max_goals_per_epoch:
        reasons.append("goal_bound")
    if len(tasks) > policy.max_tasks_per_epoch:
        reasons.append("task_bound")
    if open_task_count + len(tasks) > policy.max_open_tasks:
        reasons.append("open_work_ceiling")

    for task in tasks:
        if not task.source_root and not compilation.formal_input.get(
            "repository_tree_id"
        ):
            reasons.append("missing_source_root")
        for path in task.predicted_files:
            if _path_hits_protected(path, policy.protected_anchors):
                hits.append(path)
                reasons.append("protected_anchor")
        # Candidates cannot lower thresholds or mark complete via task fields.
        for criterion in task.acceptance_criteria:
            lowered = criterion.casefold()
            if "lower threshold" in lowered or "mark complete" in lowered:
                reasons.append("authority_claim")

    for goal in goals:
        for path in goal.scope_paths:
            if _path_hits_protected(path, policy.protected_anchors):
                hits.append(path)
                reasons.append("protected_anchor")

    unique_reasons = tuple(dict.fromkeys(reasons))
    admitted = not unique_reasons
    body = {
        "schema": DERIVED_ADMISSION_RECEIPT_SCHEMA,
        "admitted": admitted,
        "formal_plan_id": compilation.formal_plan_id,
        "source_identity": compilation.source_identity,
        "parallel_plan_digest": compilation.parallel_plan_digest,
        "goal_count": len(goals),
        "task_count": len(tasks),
        "reason_codes": list(unique_reasons),
        "protected_anchor_hits": list(dict.fromkeys(hits)),
        "stop_policy": policy.stop_policy,
        "completion_authority": False,
        "mutation_authority": False,
        "seed_board_edit": False,
        "threshold_lower_authority": False,
        "self_authorization": False,
        "producer_id": PRODUCER_ID,
    }
    return DerivedAdmissionReceipt(
        admitted=admitted,
        admission_receipt_cid=content_identity(body),
        compilation=compilation,
        reason_codes=unique_reasons,
        protected_anchor_hits=tuple(dict.fromkeys(hits)),
        open_task_count=open_task_count + (len(tasks) if admitted else 0),
        stop_policy=policy.stop_policy,
    )


# ---------------------------------------------------------------------------
# DuckDB materialization
# ---------------------------------------------------------------------------


def materialize_admitted_work(
    admission: DerivedAdmissionReceipt,
    *,
    duckdb_source: Any,
    repository_tree_id: str = "",
    receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Write admitted work into the separate derived DuckDB source.

    Requires :meth:`DuckDBTaskSource.materialize_derived_runtime` (PDR-081).
    """

    if not admission.admitted:
        raise PlannerDoctorRefillAdmissionError(
            "cannot materialize work that failed admission"
        )
    compilation = admission.compilation
    formal_input = dict(compilation.formal_input)
    if not formal_input:
        raise PlannerDoctorRefillAdmissionError(
            "compilation receipt is missing formal_input for materialization"
        )
    materialize = getattr(duckdb_source, "materialize_derived_runtime", None)
    if not callable(materialize):
        raise PlannerDoctorRefillAdmissionError(
            "duckdb source does not implement materialize_derived_runtime"
        )
    return materialize(
        formal_input,
        formal_plan_id=compilation.formal_plan_id,
        source_identity=compilation.source_identity,
        parallel_plan_digest=compilation.parallel_plan_digest,
        admission_receipt_cid=admission.admission_receipt_cid,
        repository_tree_id=repository_tree_id
        or str(formal_input.get("repository_tree_id") or ""),
        plan_root_cid=compilation.plan_root_cid,
        receipt={
            **dict(receipt or {}),
            "producer_id": PRODUCER_ID,
            "stop_policy": admission.stop_policy,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
        },
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def refill_planner_doctor_residuals(
    residuals: Sequence[Any] | None = None,
    *,
    doctor_residuals: Sequence[Any] | None = None,
    benchmark_residuals: Sequence[Any] | None = None,
    fixed_point: Any = None,
    memory: PlannerDoctorRefillMemory | Mapping[str, Any] | None = None,
    policy: PlannerDoctorRefillPolicy | Mapping[str, Any] | None = None,
    repository_tree_id: str = "",
    root_id: str = "",
    attempt_id: str = "",
    epoch_id: str = "",
    duckdb_source: Any = None,
    materialize: bool = False,
    current_time_ms: int = 1_000_000,
) -> PlannerDoctorRefillReceipt:
    """Compile benchmark/Doctor residuals into bounded derived goals and tasks.

    When ``materialize`` is true and ``duckdb_source`` is provided, admitted
    work is written only after independent formal/admission/parallel gates.
    Exact replay of an already-admitted population is a no-op at the DuckDB
    layer.
    """

    resolved_policy = (
        policy
        if isinstance(policy, PlannerDoctorRefillPolicy)
        else PlannerDoctorRefillPolicy.from_dict(policy)
    )
    resolved_memory = (
        memory
        if isinstance(memory, PlannerDoctorRefillMemory)
        else PlannerDoctorRefillMemory.from_dict(memory)
    )

    if fixed_point is not None and fixed_point_is_successful(fixed_point):
        return PlannerDoctorRefillReceipt(
            disposition=PlannerDoctorRefillDisposition.FIXED_POINT_CLOSED,
            reason_codes=("fixed_point_residual_free",),
            policy=resolved_policy,
            next_memory=resolved_memory,
            repository_tree_id=repository_tree_id,
            epoch_id=epoch_id,
        )

    collected = normalize_residuals(
        residuals,
        doctor_residuals=doctor_residuals,
        benchmark_residuals=benchmark_residuals,
        fixed_point=fixed_point,
        root_id=root_id,
        attempt_id=attempt_id,
        parent_goal_id=resolved_policy.parent_goal_id,
    )
    if not collected:
        return PlannerDoctorRefillReceipt(
            disposition=PlannerDoctorRefillDisposition.EMPTY_INPUT,
            reason_codes=("no_residuals",),
            policy=resolved_policy,
            next_memory=resolved_memory,
            repository_tree_id=repository_tree_id,
            epoch_id=epoch_id,
        )

    tree_id = (
        _identifier(repository_tree_id, "repository_tree_id", required=False)
        or _identifier(root_id, "root_id", required=False)
        or next(
            (
                residual.source_root or residual.root_id
                for residual in collected
                if residual.source_root or residual.root_id
            ),
            "",
        )
    )
    if not tree_id:
        tree_id = "tree:derived-unbound"

    (
        goals,
        tasks,
        decisions,
        duplicates,
        backoff_keys,
        reason_codes,
        next_entries,
    ) = compile_hierarchy(
        collected, policy=resolved_policy, memory=resolved_memory
    )

    if not tasks:
        if backoff_keys and not any(
            d.disposition is ResidualDisposition.TASK for d in decisions
        ):
            disposition = PlannerDoctorRefillDisposition.UNCHANGED_BACKOFF
        elif "open_work_ceiling" in reason_codes:
            disposition = PlannerDoctorRefillDisposition.OPEN_WORK_CEILING
        elif duplicates and not goals:
            disposition = PlannerDoctorRefillDisposition.DUPLICATE_BACKOFF
        else:
            disposition = PlannerDoctorRefillDisposition.EMPTY_INPUT
        return PlannerDoctorRefillReceipt(
            disposition=disposition,
            residuals=collected,
            decisions=decisions,
            goals=goals,
            tasks=(),
            work_proposals=(),
            backoff_identity_keys=backoff_keys,
            duplicate_identity_keys=duplicates,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            policy=resolved_policy,
            next_memory=PlannerDoctorRefillMemory(
                entries=tuple(next_entries.values())[:MAX_MEMORY_ENTRIES],
                open_task_count=resolved_memory.open_task_count,
                last_source_identity=resolved_memory.last_source_identity,
                last_plan_root_cid=resolved_memory.last_plan_root_cid,
                now_epoch_s=resolved_memory.now_epoch_s,
            ),
            repository_tree_id=tree_id,
            epoch_id=epoch_id,
        )

    work_proposals = tuple(
        task.to_work_proposal(parent_goal_id=task.goal_id) for task in tasks
    )

    try:
        compilation = compile_independent_gates(
            goals,
            tasks,
            policy=resolved_policy,
            repository_tree_id=tree_id,
            current_time_ms=current_time_ms,
        )
    except PlannerDoctorRefillAdmissionError as exc:
        return PlannerDoctorRefillReceipt(
            disposition=PlannerDoctorRefillDisposition.ADMISSION_REJECTED,
            residuals=collected,
            decisions=decisions,
            goals=goals,
            tasks=tasks,
            work_proposals=work_proposals,
            backoff_identity_keys=backoff_keys,
            duplicate_identity_keys=duplicates,
            reason_codes=tuple(
                dict.fromkeys([*reason_codes, "compilation_failed"])
            ),
            policy=resolved_policy,
            next_memory=PlannerDoctorRefillMemory(
                entries=tuple(next_entries.values())[:MAX_MEMORY_ENTRIES],
                open_task_count=resolved_memory.open_task_count,
                last_source_identity=resolved_memory.last_source_identity,
                last_plan_root_cid=resolved_memory.last_plan_root_cid,
                now_epoch_s=resolved_memory.now_epoch_s,
            ),
            repository_tree_id=tree_id,
            epoch_id=epoch_id,
            materialization={"error": str(exc)},
        )

    # Exact population replay: if the source identity matches last admitted,
    # skip re-emission growth (still returns compiled receipt).
    if (
        resolved_memory.last_source_identity
        and resolved_memory.last_source_identity == compilation.source_identity
    ):
        return PlannerDoctorRefillReceipt(
            disposition=PlannerDoctorRefillDisposition.REPLAY_NOOP,
            residuals=collected,
            decisions=decisions,
            goals=goals,
            tasks=tasks,
            work_proposals=work_proposals,
            compilation=compilation,
            backoff_identity_keys=backoff_keys,
            duplicate_identity_keys=duplicates,
            reason_codes=tuple(
                dict.fromkeys([*reason_codes, "exact_source_identity_replay"])
            ),
            policy=resolved_policy,
            next_memory=resolved_memory,
            repository_tree_id=tree_id,
            epoch_id=epoch_id,
        )

    admission = admit_compiled_population(
        goals,
        tasks,
        compilation=compilation,
        policy=resolved_policy,
        open_task_count=resolved_memory.open_task_count,
    )
    if not admission.admitted:
        return PlannerDoctorRefillReceipt(
            disposition=PlannerDoctorRefillDisposition.ADMISSION_REJECTED,
            residuals=collected,
            decisions=decisions,
            goals=goals,
            tasks=tasks,
            work_proposals=work_proposals,
            compilation=compilation,
            admission=admission,
            backoff_identity_keys=backoff_keys,
            duplicate_identity_keys=duplicates,
            reason_codes=tuple(
                dict.fromkeys([*reason_codes, *admission.reason_codes])
            ),
            policy=resolved_policy,
            next_memory=PlannerDoctorRefillMemory(
                entries=tuple(next_entries.values())[:MAX_MEMORY_ENTRIES],
                open_task_count=resolved_memory.open_task_count,
                last_source_identity=resolved_memory.last_source_identity,
                last_plan_root_cid=resolved_memory.last_plan_root_cid,
                now_epoch_s=resolved_memory.now_epoch_s,
            ),
            repository_tree_id=tree_id,
            epoch_id=epoch_id,
        )

    materialization: Mapping[str, Any] = {}
    disposition = PlannerDoctorRefillDisposition.ADMITTED
    if materialize:
        if duckdb_source is None:
            raise PlannerDoctorRefillError(
                "materialize=True requires duckdb_source"
            )
        if not resolved_policy.derived_runtime_admission_enabled:
            raise PlannerDoctorRefillAuthorityError(
                "derived runtime admission is disabled on policy"
            )
        materialization = materialize_admitted_work(
            admission,
            duckdb_source=duckdb_source,
            repository_tree_id=tree_id,
        )
        if materialization.get("replayed"):
            disposition = PlannerDoctorRefillDisposition.REPLAY_NOOP
        else:
            disposition = PlannerDoctorRefillDisposition.MATERIALIZED
    else:
        disposition = PlannerDoctorRefillDisposition.COMPILED
        # Still mark admitted when gates pass without materializing.
        if admission.admitted:
            disposition = PlannerDoctorRefillDisposition.ADMITTED

    next_memory = PlannerDoctorRefillMemory(
        entries=tuple(next_entries.values())[:MAX_MEMORY_ENTRIES],
        open_task_count=admission.open_task_count
        if disposition
        in {
            PlannerDoctorRefillDisposition.ADMITTED,
            PlannerDoctorRefillDisposition.MATERIALIZED,
            PlannerDoctorRefillDisposition.COMPILED,
        }
        else resolved_memory.open_task_count,
        last_source_identity=compilation.source_identity,
        last_plan_root_cid=compilation.plan_root_cid,
        now_epoch_s=resolved_memory.now_epoch_s,
    )

    return PlannerDoctorRefillReceipt(
        disposition=disposition,
        residuals=collected,
        decisions=decisions,
        goals=goals,
        tasks=tasks,
        work_proposals=work_proposals,
        compilation=compilation,
        admission=admission,
        materialization=materialization,
        backoff_identity_keys=backoff_keys,
        duplicate_identity_keys=duplicates,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        policy=resolved_policy,
        next_memory=next_memory,
        repository_tree_id=tree_id,
        epoch_id=epoch_id,
    )


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


@dataclass
class PlannerDoctorRefill:
    """Stateful helper for repeated residual → derived work compilation."""

    INTERFACE: Final[str] = PLANNER_DOCTOR_REFILL_INTERFACE
    VERSION: Final[str] = PLANNER_DOCTOR_REFILL_VERSION

    policy: PlannerDoctorRefillPolicy = field(
        default_factory=PlannerDoctorRefillPolicy
    )
    memory: PlannerDoctorRefillMemory = field(
        default_factory=PlannerDoctorRefillMemory
    )

    @property
    def producer_id(self) -> str:
        return PRODUCER_ID

    def refill(
        self,
        residuals: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> PlannerDoctorRefillReceipt:
        receipt = refill_planner_doctor_residuals(
            residuals,
            memory=self.memory,
            policy=self.policy,
            **kwargs,
        )
        self.memory = receipt.next_memory
        return receipt

    def compile_and_admit(
        self,
        residuals: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> PlannerDoctorRefillReceipt:
        kwargs.setdefault("materialize", False)
        return self.refill(residuals, **kwargs)

    def materialize(
        self,
        residuals: Sequence[Any] | None = None,
        *,
        duckdb_source: Any,
        **kwargs: Any,
    ) -> PlannerDoctorRefillReceipt:
        return self.refill(
            residuals, duckdb_source=duckdb_source, materialize=True, **kwargs
        )


def create_planner_doctor_refill(**kwargs: Any) -> PlannerDoctorRefill:
    """Construct a :class:`PlannerDoctorRefill` instance."""

    policy = kwargs.pop("policy", None)
    memory = kwargs.pop("memory", None)
    if kwargs:
        raise PlannerDoctorRefillError(
            f"unknown create_planner_doctor_refill kwargs: {sorted(kwargs)}"
        )
    return PlannerDoctorRefill(
        policy=(
            policy
            if isinstance(policy, PlannerDoctorRefillPolicy)
            else PlannerDoctorRefillPolicy.from_dict(policy)
        ),
        memory=(
            memory
            if isinstance(memory, PlannerDoctorRefillMemory)
            else PlannerDoctorRefillMemory.from_dict(memory)
        ),
    )


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_DERIVED_RUNTIME_SOURCE_ID",
    "DEFAULT_MAX_GOALS_PER_EPOCH",
    "DEFAULT_MAX_OPEN_TASKS",
    "DEFAULT_MAX_TASKS_PER_EPOCH",
    "DEFAULT_PARENT_GOAL_ID",
    "DEFAULT_PROTECTED_ANCHORS",
    "DEFAULT_STOP_POLICY",
    "DERIVED_RUNTIME_SOURCE_GATE",
    "PLANNER_DOCTOR_REFILL_INTERFACE",
    "PLANNER_DOCTOR_REFILL_RECEIPT_SCHEMA",
    "PLANNER_DOCTOR_REFILL_VERSION",
    "PRODUCER_ID",
    "REFILL_AUTHORIZES_COMPLETION",
    "REFILL_AUTHORIZES_MUTATION",
    "REFILL_AUTHORIZES_SEED_BOARD_EDIT",
    "REFILL_AUTHORIZES_SELF_AUTHORIZATION",
    "REFILL_AUTHORIZES_THRESHOLD_LOWER",
    "DerivedAdmissionReceipt",
    "DerivedCompilationReceipt",
    "DerivedGoalProposal",
    "DerivedResidual",
    "DerivedTaskProposal",
    "PlannerDoctorRefill",
    "PlannerDoctorRefillAdmissionError",
    "PlannerDoctorRefillAuthorityError",
    "PlannerDoctorRefillBoundsError",
    "PlannerDoctorRefillDisposition",
    "PlannerDoctorRefillError",
    "PlannerDoctorRefillMemory",
    "PlannerDoctorRefillPolicy",
    "PlannerDoctorRefillReceipt",
    "ResidualDecision",
    "ResidualDisposition",
    "ResidualSourceKind",
    "admit_compiled_population",
    "build_formal_plan_input",
    "compile_hierarchy",
    "compile_independent_gates",
    "create_planner_doctor_refill",
    "dedupe_derived_residuals",
    "materialize_admitted_work",
    "normalize_residuals",
    "refill_planner_doctor_residuals",
]
