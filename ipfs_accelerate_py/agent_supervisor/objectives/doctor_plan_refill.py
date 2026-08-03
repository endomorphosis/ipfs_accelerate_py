"""Feed Doctor residuals into plan steering and bounded derived refill (PDR-055).

Interface: ``DoctorPlanResidual@1`` / ``DoctorPlanRefill@1``

When a Doctor fixed-point run is residual-free, this module emits no work.
Otherwise unsupported, open, or failed obligations become small targeted
successor tasks rather than repeated broad LLM prompts.

Normative rules:

* residuals are deduplicated by exact issue / obligation / root / attempt
  identities;
* where a residual maps onto an existing plan node, emission is an append-only
  successor (``add_task`` / ``add_goal``) — never an in-place edit of
  active/accepted task specs or seed boards;
* residuals that cannot map emit bounded :class:`ObjectiveWorkProposal`
  records;
* unchanged failures back off without re-emitting work;
* capability gaps name the exact provider / conformance work required;
* no completion or mutation authority is granted;
* generated tasks target minimal files and context; admission into the
  separate derived runtime task source is gated until PDR-081 enables it.

Conflict policy: residuals propose plan deltas or derived work; they never
edit active/accepted task specs or seed boards.
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

from .objective_graph import ObjectiveWorkKind, ObjectiveWorkProposal
from ..planning.plan_revision_contracts import (
    DeltaEffectClass,
    LifecycleState,
    PlanDeltaItem,
    PlanDeltaOperation,
)
from ..proof.formal_verification_contracts import content_identity

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DOCTOR_PLAN_RESIDUAL_INTERFACE: Final[str] = "DoctorPlanResidual@1"
DOCTOR_PLAN_REFILL_INTERFACE: Final[str] = "DoctorPlanRefill@1"
DOCTOR_PLAN_REFILL_VERSION: Final[str] = "1.0.0"
CONTRACT_VERSION: Final[int] = 1
PRODUCER_ID: Final[str] = "doctor-plan-refill@1"

DOCTOR_PLAN_RESIDUAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-plan-residual@1"
)
DOCTOR_PLAN_REFILL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-plan-refill-receipt@1"
)
DOCTOR_PLAN_REFILL_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-plan-refill-policy@1"
)
DOCTOR_PLAN_REFILL_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-plan-refill-memory@1"
)
DOCTOR_PLAN_SUCCESSOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-plan-successor@1"
)

# Authority is intentionally hard-off: refill proposes only.
REFILL_AUTHORIZES_COMPLETION: Final[bool] = False
REFILL_AUTHORIZES_MUTATION: Final[bool] = False
REFILL_AUTHORIZES_SEED_BOARD_EDIT: Final[bool] = False

# Derived runtime source admission is owned by PDR-081.  This module may
# *label* proposals for that source but never admits them while disabled.
DERIVED_RUNTIME_SOURCE_GATE: Final[str] = "PDR-081"
DEFAULT_DERIVED_RUNTIME_SOURCE_ID: Final[str] = (
    "task-source:derived-runtime/planner-doctor"
)

DEFAULT_PARENT_GOAL_ID: Final[str] = "PDR-G060"
DEFAULT_MAX_RESIDUALS: Final[int] = 24
DEFAULT_MAX_SUCCESSORS: Final[int] = 16
DEFAULT_MAX_PROPOSALS: Final[int] = 16
DEFAULT_MAX_PATHS_PER_RESIDUAL: Final[int] = 8
DEFAULT_MAX_CONTEXT_PATHS: Final[int] = 8
DEFAULT_BACKOFF_IDENTICAL_ATTEMPTS: Final[int] = 1
DEFAULT_COOLDOWN_SECONDS: Final[int] = 900
DEFAULT_MAX_OPEN_DERIVED_TASKS: Final[int] = 48
DEFAULT_MAX_GOALS_PER_EPOCH: Final[int] = 8
DEFAULT_MAX_TASKS_PER_EPOCH: Final[int] = 24
DEFAULT_RESOURCE_CLASS: Final[str] = "cpu-medium"
DEFAULT_TOKEN_CLASS: Final[str] = "medium"

MAX_ID_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_RESIDUALS: Final[int] = 128
MAX_MEMORY_ENTRIES: Final[int] = 4_096
MAX_STRATEGY_IDS: Final[int] = 64
MAX_REASON_CODES: Final[int] = 64

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-=]{0,511}$"
)
_GOAL_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9._:-]{0,255}$"
)

# Family namespace for typed objective work derived from doctor residuals.
_OBJECTIVE_FAMILY_PREFIX: Final[str] = "objective-family/v1/doctor-residual/"
_OBJECTIVE_INSTANCE_PREFIX: Final[str] = "objective-instance/v1/doctor-residual/"


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class DoctorPlanRefillError(ValueError):
    """A doctor plan refill residual, policy, or receipt is malformed."""


class DoctorPlanRefillAuthorityError(DoctorPlanRefillError):
    """Raised when a residual attempts to claim completion or mutation power."""


class DoctorPlanRefillBoundsError(DoctorPlanRefillError):
    """A residual population or field exceeds a hard bound."""


class DoctorResidualKind(str, Enum):
    """Closed residual kinds emitted by Doctor fixed-point / diagnosis."""

    OPEN_OBLIGATION = "open_obligation"
    FAILED_OBLIGATION = "failed_obligation"
    UNSUPPORTED = "unsupported"
    CAPABILITY_GAP = "capability_gap"
    FRONTIER = "frontier"
    COUNTEREXAMPLE = "counterexample"
    UNCHANGED_FAILURE = "unchanged_failure"
    CACHE_STALE = "cache_stale"
    SECURITY = "security"
    OTHER = "other"


class DoctorPlanRefillDisposition(str, Enum):
    """Stable outcomes of one refill pass."""

    FIXED_POINT_CLOSED = "fixed_point_closed"
    APPEND_ONLY_SUCCESSORS = "append_only_successors"
    WORK_PROPOSALS = "work_proposals"
    MIXED = "mixed"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    CAPABILITY_GAP = "capability_gap"
    BOUND_EXCEEDED = "bound_exceeded"
    EMPTY_INPUT = "empty_input"
    DERIVED_RUNTIME_GATED = "derived_runtime_gated"


class DoctorResidualDisposition(str, Enum):
    """Per-residual admission decision."""

    MAPPED_SUCCESSOR = "mapped_successor"
    WORK_PROPOSAL = "work_proposal"
    CAPABILITY_GAP_PROPOSAL = "capability_gap_proposal"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    DUPLICATE = "duplicate"
    BOUND_REJECTED = "bound_rejected"
    MALFORMED = "malformed"
    FIXED_POINT_SKIP = "fixed_point_skip"


class DoctorPlanTargetSource(str, Enum):
    """Where generated work may land once independently admitted."""

    PLAN_STEER_DELTA = "plan_steer_delta"
    OBJECTIVE_HEAP = "objective_heap"
    DERIVED_RUNTIME = "derived_runtime"
    NONE = "none"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise DoctorPlanRefillError(f"{name} must be a string")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise DoctorPlanRefillError(f"{name} must be normalized single-line text")
    if required and not text:
        raise DoctorPlanRefillError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise DoctorPlanRefillBoundsError(f"{name} exceeds its byte bound")
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, limit=MAX_ID_BYTES)
    if not text:
        return ""
    if not _IDENTIFIER_RE.fullmatch(text):
        raise DoctorPlanRefillError(f"{name} is malformed")
    return text


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_RESIDUALS,
    preserve_order: bool = True,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise DoctorPlanRefillError(f"{name} must be a sequence of identifiers")
    if len(items) > maximum:
        raise DoctorPlanRefillBoundsError(f"{name} exceeds its item bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = _identifier(raw, name, required=True)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if required and not out:
        raise DoctorPlanRefillError(f"{name} must not be empty")
    if preserve_order:
        return tuple(out)
    return tuple(sorted(out))


def _command_strings(
    values: Any,
    name: str,
    *,
    maximum: int = 16,
) -> tuple[str, ...]:
    """Normalize validation commands (spaces allowed; newlines/nulls forbidden)."""

    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise DoctorPlanRefillError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise DoctorPlanRefillBoundsError(f"{name} exceeds its item bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _text(raw, name, required=True, limit=MAX_TEXT_BYTES)
        # Collapse internal whitespace but keep a single-line shell command.
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _paths(values: Any, name: str, *, maximum: int = DEFAULT_MAX_PATHS_PER_RESIDUAL) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise DoctorPlanRefillError(f"{name} must be a sequence of paths")
    if len(items) > maximum:
        raise DoctorPlanRefillBoundsError(f"{name} exceeds its path bound")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        path = _normalize_path(raw)
        if not path:
            continue
        if path in seen:
            continue
        if len(path.encode("utf-8")) > MAX_PATH_BYTES:
            raise DoctorPlanRefillBoundsError(f"{name} path exceeds its byte bound")
        seen.add(path)
        out.append(path)
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
        raise DoctorPlanRefillError("paths must be repository-relative and non-escaping")
    return normalized


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise DoctorPlanRefillError(f"{name} must be a boolean")


def _nonneg_int(value: Any, name: str, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DoctorPlanRefillError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise DoctorPlanRefillBoundsError(f"{name} out of bounds")
    return value


def _positive_int(value: Any, name: str, *, maximum: int = 1_000_000) -> int:
    number = _nonneg_int(value, name, maximum=maximum)
    if number < 1:
        raise DoctorPlanRefillError(f"{name} must be >= 1")
    return number


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DoctorPlanRefillError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DoctorPlanRefillError(f"{name} must be between 0 and 1")
    return number


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise DoctorPlanRefillError(f"{name} has unknown value {text!r}") from exc


def _mapping_proxy(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise DoctorPlanRefillError(f"{name} must be a mapping")
    try:
        canonical = json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise DoctorPlanRefillError(f"{name} must be canonical JSON data") from exc
    if not isinstance(canonical, dict):
        raise DoctorPlanRefillError(f"{name} must be a mapping")
    return MappingProxyType(canonical)


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _content_cid(payload: Mapping[str, Any]) -> str:
    return content_identity(dict(payload))


# ---------------------------------------------------------------------------
# Residual identity
# ---------------------------------------------------------------------------


def residual_identity_payload(
    *,
    issue_id: str,
    obligation_id: str,
    root_id: str,
    attempt_id: str,
) -> dict[str, str]:
    """Exact issue/obligation/root/attempt identity tuple used for dedupe."""

    return {
        "issue_id": _identifier(issue_id, "issue_id"),
        "obligation_id": _identifier(obligation_id, "obligation_id", required=False),
        "root_id": _identifier(root_id, "root_id", required=False),
        "attempt_id": _identifier(attempt_id, "attempt_id", required=False),
    }


def residual_identity_key(
    *,
    issue_id: str,
    obligation_id: str = "",
    root_id: str = "",
    attempt_id: str = "",
) -> str:
    """Stable content-addressed residual identity (exact four-tuple)."""

    payload = residual_identity_payload(
        issue_id=issue_id,
        obligation_id=obligation_id,
        root_id=root_id,
        attempt_id=attempt_id,
    )
    return "doctor-residual:" + _stable_digest(payload)[len("sha256:") :]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoctorPlanResidual:
    """One unsupported / open / failed Doctor residual (``DoctorPlanResidual@1``).

    Residuals are evidence of incomplete Doctor closure.  They never authorize
    completion, mutation, or seed-board edits.
    """

    issue_id: str
    obligation_id: str = ""
    root_id: str = ""
    attempt_id: str = ""
    kind: DoctorResidualKind = DoctorResidualKind.OPEN_OBLIGATION
    finding_id: str = ""
    frontier_id: str = ""
    counterexample_id: str = ""
    plan_id: str = ""
    transaction_id: str = ""
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    parent_task_cid: str = ""
    parent_goal_cid: str = ""
    predicted_files: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    attempted_strategies: tuple[str, ...] = ()
    required_capability: str = ""
    required_provider: str = ""
    required_conformance: str = ""
    cache_hit: bool = False
    unchanged_failure: bool = False
    reason_codes: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    title: str = ""
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    residual_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_id", _identifier(self.issue_id, "issue_id"))
        object.__setattr__(
            self,
            "obligation_id",
            _identifier(self.obligation_id, "obligation_id", required=False),
        )
        object.__setattr__(
            self, "root_id", _identifier(self.root_id, "root_id", required=False)
        )
        object.__setattr__(
            self,
            "attempt_id",
            _identifier(self.attempt_id, "attempt_id", required=False),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, DoctorResidualKind, "kind")
        )
        for name in (
            "finding_id",
            "frontier_id",
            "counterexample_id",
            "plan_id",
            "transaction_id",
            "parent_task_cid",
            "parent_goal_cid",
            "required_capability",
            "required_provider",
            "required_conformance",
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, required=False),
            )
        parent_goal = _optional_text(self.parent_goal_id, "parent_goal_id")
        if parent_goal and not _GOAL_ID_RE.fullmatch(parent_goal):
            raise DoctorPlanRefillError("parent_goal_id is malformed")
        object.__setattr__(
            self, "parent_goal_id", parent_goal or DEFAULT_PARENT_GOAL_ID
        )
        object.__setattr__(
            self,
            "predicted_files",
            _paths(self.predicted_files, "predicted_files"),
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
            "attempted_strategies",
            _ids(self.attempted_strategies, "attempted_strategies", maximum=MAX_STRATEGY_IDS),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _command_strings(self.validation_commands, "validation_commands"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _ids(self.evidence_refs, "evidence_refs", maximum=64),
        )
        object.__setattr__(self, "cache_hit", _bool(self.cache_hit, "cache_hit"))
        object.__setattr__(
            self, "unchanged_failure", _bool(self.unchanged_failure, "unchanged_failure")
        )
        object.__setattr__(self, "title", _optional_text(self.title, "title"))
        object.__setattr__(
            self, "rationale", _optional_text(self.rationale, "rationale")
        )
        object.__setattr__(
            self, "metadata", _mapping_proxy(self.metadata, "metadata")
        )
        rid = _optional_text(self.residual_id, "residual_id", limit=MAX_ID_BYTES)
        object.__setattr__(self, "residual_id", rid or self.identity_key)
        # Authority hardening: residual metadata may never claim authority.
        forbidden = {
            "completion_authority",
            "mutation_authority",
            "claims_completion",
            "may_mutate",
            "seed_board_edit",
        }
        for key in self.metadata:
            if str(key).lower().replace("-", "_") in forbidden:
                raise DoctorPlanRefillAuthorityError(
                    "doctor residuals cannot claim completion or mutation authority"
                )
        if self.kind is DoctorResidualKind.CAPABILITY_GAP:
            if not (
                self.required_capability
                or self.required_provider
                or self.required_conformance
            ):
                raise DoctorPlanRefillError(
                    "capability_gap residuals must name required_capability, "
                    "required_provider, or required_conformance"
                )

    @property
    def identity_key(self) -> str:
        return residual_identity_key(
            issue_id=self.issue_id,
            obligation_id=self.obligation_id,
            root_id=self.root_id,
            attempt_id=self.attempt_id,
        )

    @property
    def identity_payload(self) -> dict[str, str]:
        return residual_identity_payload(
            issue_id=self.issue_id,
            obligation_id=self.obligation_id,
            root_id=self.root_id,
            attempt_id=self.attempt_id,
        )

    @property
    def is_capability_gap(self) -> bool:
        return self.kind is DoctorResidualKind.CAPABILITY_GAP

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def mutation_authority(self) -> bool:
        return False

    def minimal_paths(self) -> tuple[str, ...]:
        """Predicted files first, then context — already path-bounded."""

        files = list(self.predicted_files)
        for path in self.context_paths:
            if path not in files:
                files.append(path)
        return tuple(files[: DEFAULT_MAX_PATHS_PER_RESIDUAL])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PLAN_RESIDUAL_SCHEMA,
            "interface": DOCTOR_PLAN_RESIDUAL_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "issue_id": self.issue_id,
            "obligation_id": self.obligation_id,
            "root_id": self.root_id,
            "attempt_id": self.attempt_id,
            "kind": self.kind.value,
            "finding_id": self.finding_id,
            "frontier_id": self.frontier_id,
            "counterexample_id": self.counterexample_id,
            "plan_id": self.plan_id,
            "transaction_id": self.transaction_id,
            "parent_goal_id": self.parent_goal_id,
            "parent_task_cid": self.parent_task_cid,
            "parent_goal_cid": self.parent_goal_cid,
            "predicted_files": list(self.predicted_files),
            "context_paths": list(self.context_paths),
            "predicted_symbols": list(self.predicted_symbols),
            "attempted_strategies": list(self.attempted_strategies),
            "required_capability": self.required_capability,
            "required_provider": self.required_provider,
            "required_conformance": self.required_conformance,
            "cache_hit": self.cache_hit,
            "unchanged_failure": self.unchanged_failure,
            "reason_codes": list(self.reason_codes),
            "validation_commands": list(self.validation_commands),
            "evidence_refs": list(self.evidence_refs),
            "title": self.title,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPlanResidual":
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("residual payload must be a mapping")
        known = {
            "schema",
            "interface",
            "contract_version",
            "producer_id",
            "residual_id",
            "identity_key",
            "issue_id",
            "obligation_id",
            "root_id",
            "attempt_id",
            "kind",
            "finding_id",
            "frontier_id",
            "counterexample_id",
            "plan_id",
            "transaction_id",
            "parent_goal_id",
            "parent_task_cid",
            "parent_goal_cid",
            "predicted_files",
            "context_paths",
            "predicted_symbols",
            "attempted_strategies",
            "required_capability",
            "required_provider",
            "required_conformance",
            "cache_hit",
            "unchanged_failure",
            "reason_codes",
            "validation_commands",
            "evidence_refs",
            "title",
            "rationale",
            "metadata",
            "completion_authority",
            "mutation_authority",
            "seed_board_edit",
        }
        unknown = set(payload) - known
        if unknown:
            raise DoctorPlanRefillError(
                f"residual payload has unknown fields: {sorted(unknown)}"
            )
        for flag in ("completion_authority", "mutation_authority", "seed_board_edit"):
            if flag in payload and payload[flag]:
                raise DoctorPlanRefillAuthorityError(
                    f"{flag} cannot be true on a doctor residual"
                )
        return cls(
            issue_id=payload["issue_id"],
            obligation_id=payload.get("obligation_id", ""),
            root_id=payload.get("root_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            kind=payload.get("kind", DoctorResidualKind.OPEN_OBLIGATION),
            finding_id=payload.get("finding_id", ""),
            frontier_id=payload.get("frontier_id", ""),
            counterexample_id=payload.get("counterexample_id", ""),
            plan_id=payload.get("plan_id", ""),
            transaction_id=payload.get("transaction_id", ""),
            parent_goal_id=payload.get("parent_goal_id", DEFAULT_PARENT_GOAL_ID),
            parent_task_cid=payload.get("parent_task_cid", ""),
            parent_goal_cid=payload.get("parent_goal_cid", ""),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            context_paths=tuple(payload.get("context_paths") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            attempted_strategies=tuple(payload.get("attempted_strategies") or ()),
            required_capability=payload.get("required_capability", ""),
            required_provider=payload.get("required_provider", ""),
            required_conformance=payload.get("required_conformance", ""),
            cache_hit=bool(payload.get("cache_hit", False)),
            unchanged_failure=bool(payload.get("unchanged_failure", False)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            validation_commands=tuple(payload.get("validation_commands") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            title=payload.get("title", ""),
            rationale=payload.get("rationale", ""),
            metadata=payload.get("metadata") or {},
            residual_id=payload.get("residual_id", ""),
        )


@dataclass(frozen=True, slots=True)
class DoctorPlanRefillPolicy:
    """Hard bounds and gates for one doctor residual refill pass."""

    max_residuals: int = DEFAULT_MAX_RESIDUALS
    max_successors: int = DEFAULT_MAX_SUCCESSORS
    max_proposals: int = DEFAULT_MAX_PROPOSALS
    max_paths_per_residual: int = DEFAULT_MAX_PATHS_PER_RESIDUAL
    max_context_paths: int = DEFAULT_MAX_CONTEXT_PATHS
    backoff_identical_attempts: int = DEFAULT_BACKOFF_IDENTICAL_ATTEMPTS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    max_open_derived_tasks: int = DEFAULT_MAX_OPEN_DERIVED_TASKS
    max_goals_per_epoch: int = DEFAULT_MAX_GOALS_PER_EPOCH
    max_tasks_per_epoch: int = DEFAULT_MAX_TASKS_PER_EPOCH
    # PDR-081 gate: derived runtime source admission is disabled by default.
    derived_runtime_admission_enabled: bool = False
    derived_runtime_source_id: str = DEFAULT_DERIVED_RUNTIME_SOURCE_ID
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    resource_class: str = DEFAULT_RESOURCE_CLASS
    token_class: str = DEFAULT_TOKEN_CLASS
    prefer_append_only_successors: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_residuals",
            _positive_int(self.max_residuals, "max_residuals", maximum=MAX_RESIDUALS),
        )
        for name, maximum in (
            ("max_successors", MAX_RESIDUALS),
            ("max_proposals", MAX_RESIDUALS),
            ("max_paths_per_residual", 64),
            ("max_context_paths", 64),
            ("backoff_identical_attempts", 64),
            ("cooldown_seconds", 86_400),
            ("max_open_derived_tasks", 1_024),
            ("max_goals_per_epoch", 64),
            ("max_tasks_per_epoch", 256),
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=maximum),
            )
        object.__setattr__(
            self,
            "derived_runtime_admission_enabled",
            _bool(
                self.derived_runtime_admission_enabled,
                "derived_runtime_admission_enabled",
            ),
        )
        object.__setattr__(
            self,
            "derived_runtime_source_id",
            _identifier(
                self.derived_runtime_source_id, "derived_runtime_source_id"
            ),
        )
        parent = _text(self.parent_goal_id, "parent_goal_id")
        if not _GOAL_ID_RE.fullmatch(parent):
            raise DoctorPlanRefillError("parent_goal_id is malformed")
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
            "prefer_append_only_successors",
            _bool(self.prefer_append_only_successors, "prefer_append_only_successors"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PLAN_REFILL_POLICY_SCHEMA,
            "max_residuals": self.max_residuals,
            "max_successors": self.max_successors,
            "max_proposals": self.max_proposals,
            "max_paths_per_residual": self.max_paths_per_residual,
            "max_context_paths": self.max_context_paths,
            "backoff_identical_attempts": self.backoff_identical_attempts,
            "cooldown_seconds": self.cooldown_seconds,
            "max_open_derived_tasks": self.max_open_derived_tasks,
            "max_goals_per_epoch": self.max_goals_per_epoch,
            "max_tasks_per_epoch": self.max_tasks_per_epoch,
            "derived_runtime_admission_enabled": (
                self.derived_runtime_admission_enabled
            ),
            "derived_runtime_source_id": self.derived_runtime_source_id,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
            "parent_goal_id": self.parent_goal_id,
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "prefer_append_only_successors": self.prefer_append_only_successors,
            "completion_authority": False,
            "mutation_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "DoctorPlanRefillPolicy":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("policy must be a mapping")
        fields = set(cls.__dataclass_fields__)
        unknown = set(payload) - fields - {"schema", "derived_runtime_gate",
                                           "completion_authority", "mutation_authority"}
        if unknown:
            raise DoctorPlanRefillError(
                f"policy has unknown fields: {sorted(unknown)}"
            )
        kwargs = {key: payload[key] for key in fields if key in payload}
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class DoctorPlanRefillMemoryEntry:
    """One previously observed residual fingerprint for backoff / dedupe."""

    identity_key: str
    attempt_count: int = 1
    last_fingerprint: str = ""
    last_seen_epoch_s: int = 0
    last_disposition: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity_key", _identifier(self.identity_key, "identity_key")
        )
        object.__setattr__(
            self,
            "attempt_count",
            _positive_int(self.attempt_count, "attempt_count", maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "last_fingerprint",
            _optional_text(self.last_fingerprint, "last_fingerprint", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "last_seen_epoch_s",
            _nonneg_int(self.last_seen_epoch_s, "last_seen_epoch_s", maximum=2**63 - 1),
        )
        object.__setattr__(
            self,
            "last_disposition",
            _optional_text(self.last_disposition, "last_disposition", limit=128),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_key": self.identity_key,
            "attempt_count": self.attempt_count,
            "last_fingerprint": self.last_fingerprint,
            "last_seen_epoch_s": self.last_seen_epoch_s,
            "last_disposition": self.last_disposition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPlanRefillMemoryEntry":
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("memory entry must be a mapping")
        return cls(
            identity_key=payload["identity_key"],
            attempt_count=int(payload.get("attempt_count", 1)),
            last_fingerprint=str(payload.get("last_fingerprint") or ""),
            last_seen_epoch_s=int(payload.get("last_seen_epoch_s") or 0),
            last_disposition=str(payload.get("last_disposition") or ""),
        )


@dataclass(frozen=True, slots=True)
class DoctorPlanRefillMemory:
    """Durable backoff / dedupe state for doctor residual refill."""

    entries: tuple[DoctorPlanRefillMemoryEntry, ...] = ()
    open_derived_task_count: int = 0
    now_epoch_s: int = 0

    def __post_init__(self) -> None:
        if len(self.entries) > MAX_MEMORY_ENTRIES:
            raise DoctorPlanRefillBoundsError("memory entries exceed bound")
        normalized: list[DoctorPlanRefillMemoryEntry] = []
        seen: set[str] = set()
        for item in self.entries:
            if isinstance(item, DoctorPlanRefillMemoryEntry):
                entry = item
            elif isinstance(item, Mapping):
                entry = DoctorPlanRefillMemoryEntry.from_dict(item)
            else:
                raise DoctorPlanRefillError("memory entries must be objects")
            if entry.identity_key in seen:
                continue
            seen.add(entry.identity_key)
            normalized.append(entry)
        object.__setattr__(self, "entries", tuple(normalized))
        object.__setattr__(
            self,
            "open_derived_task_count",
            _nonneg_int(
                self.open_derived_task_count,
                "open_derived_task_count",
                maximum=1_000_000,
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
            "schema": DOCTOR_PLAN_REFILL_MEMORY_SCHEMA,
            "entries": [entry.to_dict() for entry in self.entries],
            "open_derived_task_count": self.open_derived_task_count,
            "now_epoch_s": self.now_epoch_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "DoctorPlanRefillMemory":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("memory must be a mapping")
        return cls(
            entries=tuple(payload.get("entries") or ()),
            open_derived_task_count=int(payload.get("open_derived_task_count") or 0),
            now_epoch_s=int(payload.get("now_epoch_s") or 0),
        )


@dataclass(frozen=True, slots=True)
class DoctorPlanNode:
    """Minimal plan node used for append-only successor mapping."""

    node_cid: str
    kind: str  # "task" | "goal"
    lifecycle: str = LifecycleState.UNSTARTED.value
    goal_id: str = ""
    obligation_ids: tuple[str, ...] = ()
    issue_ids: tuple[str, ...] = ()
    predicted_files: tuple[str, ...] = ()
    title: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_cid", _identifier(self.node_cid, "node_cid")
        )
        kind = _text(self.kind, "kind", limit=32).lower()
        if kind not in {"task", "goal"}:
            raise DoctorPlanRefillError("plan node kind must be task or goal")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "lifecycle",
            _optional_text(self.lifecycle, "lifecycle", limit=64)
            or LifecycleState.UNSTARTED.value,
        )
        object.__setattr__(
            self, "goal_id", _optional_text(self.goal_id, "goal_id", limit=256)
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=64),
        )
        object.__setattr__(
            self, "issue_ids", _ids(self.issue_ids, "issue_ids", maximum=64)
        )
        object.__setattr__(
            self,
            "predicted_files",
            _paths(self.predicted_files, "predicted_files"),
        )
        object.__setattr__(self, "title", _optional_text(self.title, "title"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_cid": self.node_cid,
            "kind": self.kind,
            "lifecycle": self.lifecycle,
            "goal_id": self.goal_id,
            "obligation_ids": list(self.obligation_ids),
            "issue_ids": list(self.issue_ids),
            "predicted_files": list(self.predicted_files),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorPlanNode":
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("plan node must be a mapping")
        return cls(
            node_cid=payload["node_cid"],
            kind=payload.get("kind", "task"),
            lifecycle=payload.get("lifecycle", LifecycleState.UNSTARTED.value),
            goal_id=payload.get("goal_id", ""),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            issue_ids=tuple(payload.get("issue_ids") or ()),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            title=payload.get("title", ""),
        )


@dataclass(frozen=True, slots=True)
class DoctorPlanContext:
    """Existing plan surface used for append-only successor mapping."""

    plan_root: str = ""
    plan_revision: int = 1
    nodes: tuple[DoctorPlanNode, ...] = ()
    allowed_delta_operations: tuple[str, ...] = (
        PlanDeltaOperation.ADD_TASK.value,
        PlanDeltaOperation.ADD_GOAL.value,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plan_root", _identifier(self.plan_root, "plan_root", required=False)
        )
        object.__setattr__(
            self,
            "plan_revision",
            _positive_int(self.plan_revision, "plan_revision", maximum=2**31 - 1),
        )
        nodes: list[DoctorPlanNode] = []
        for item in self.nodes:
            if isinstance(item, DoctorPlanNode):
                nodes.append(item)
            elif isinstance(item, Mapping):
                nodes.append(DoctorPlanNode.from_dict(item))
            else:
                raise DoctorPlanRefillError("plan nodes must be DoctorPlanNode")
        if len(nodes) > MAX_RESIDUALS * 4:
            raise DoctorPlanRefillBoundsError("plan nodes exceed bound")
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(
            self,
            "allowed_delta_operations",
            _ids(
                self.allowed_delta_operations,
                "allowed_delta_operations",
                maximum=32,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_root": self.plan_root,
            "plan_revision": self.plan_revision,
            "nodes": [node.to_dict() for node in self.nodes],
            "allowed_delta_operations": list(self.allowed_delta_operations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "DoctorPlanContext":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise DoctorPlanRefillError("plan context must be a mapping")
        return cls(
            plan_root=payload.get("plan_root", ""),
            plan_revision=int(payload.get("plan_revision") or 1),
            nodes=tuple(payload.get("nodes") or ()),
            allowed_delta_operations=tuple(
                payload.get("allowed_delta_operations")
                or (
                    PlanDeltaOperation.ADD_TASK.value,
                    PlanDeltaOperation.ADD_GOAL.value,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class DoctorPlanSuccessor:
    """Append-only successor task proposed against an existing plan node."""

    residual_id: str
    identity_key: str
    parent_node_cid: str
    parent_lifecycle: str
    operation: str
    delta_item: PlanDeltaItem
    target_source: DoctorPlanTargetSource = DoctorPlanTargetSource.PLAN_STEER_DELTA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "residual_id", _identifier(self.residual_id, "residual_id")
        )
        object.__setattr__(
            self, "identity_key", _identifier(self.identity_key, "identity_key")
        )
        object.__setattr__(
            self,
            "parent_node_cid",
            _identifier(self.parent_node_cid, "parent_node_cid"),
        )
        object.__setattr__(
            self,
            "parent_lifecycle",
            _text(self.parent_lifecycle, "parent_lifecycle", limit=64),
        )
        object.__setattr__(
            self, "operation", _text(self.operation, "operation", limit=64)
        )
        if not isinstance(self.delta_item, PlanDeltaItem):
            raise DoctorPlanRefillError("delta_item must be PlanDeltaItem")
        if self.delta_item.operation not in (
            PlanDeltaOperation.ADD_TASK,
            PlanDeltaOperation.ADD_GOAL,
        ):
            raise DoctorPlanRefillAuthorityError(
                "doctor successors must be append-only add_task/add_goal"
            )
        object.__setattr__(
            self,
            "target_source",
            _enum(self.target_source, DoctorPlanTargetSource, "target_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PLAN_SUCCESSOR_SCHEMA,
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "parent_node_cid": self.parent_node_cid,
            "parent_lifecycle": self.parent_lifecycle,
            "operation": self.operation,
            "delta_item": self.delta_item.to_dict(),
            "target_source": self.target_source.value,
            "completion_authority": False,
            "mutation_authority": False,
        }


@dataclass(frozen=True, slots=True)
class DoctorResidualDecision:
    """Per-residual decision recorded on the refill receipt."""

    residual_id: str
    identity_key: str
    disposition: DoctorResidualDisposition
    reason_codes: tuple[str, ...] = ()
    successor: DoctorPlanSuccessor | None = None
    work_proposal: ObjectiveWorkProposal | None = None
    target_source: DoctorPlanTargetSource = DoctorPlanTargetSource.NONE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "residual_id", _identifier(self.residual_id, "residual_id")
        )
        object.__setattr__(
            self, "identity_key", _identifier(self.identity_key, "identity_key")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorResidualDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        if self.successor is not None and not isinstance(
            self.successor, DoctorPlanSuccessor
        ):
            raise DoctorPlanRefillError("successor must be DoctorPlanSuccessor")
        if self.work_proposal is not None and not isinstance(
            self.work_proposal, ObjectiveWorkProposal
        ):
            raise DoctorPlanRefillError(
                "work_proposal must be ObjectiveWorkProposal"
            )
        object.__setattr__(
            self,
            "target_source",
            _enum(self.target_source, DoctorPlanTargetSource, "target_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "identity_key": self.identity_key,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "successor": self.successor.to_dict() if self.successor else None,
            "work_proposal": (
                self.work_proposal.to_dict() if self.work_proposal else None
            ),
            "target_source": self.target_source.value,
            "completion_authority": False,
            "mutation_authority": False,
        }


@dataclass(frozen=True, slots=True)
class DoctorPlanRefillReceipt:
    """Body-free receipt for one doctor residual → plan/refill pass."""

    disposition: DoctorPlanRefillDisposition
    residuals: tuple[DoctorPlanResidual, ...] = ()
    decisions: tuple[DoctorResidualDecision, ...] = ()
    successors: tuple[DoctorPlanSuccessor, ...] = ()
    work_proposals: tuple[ObjectiveWorkProposal, ...] = ()
    backoff_identity_keys: tuple[str, ...] = ()
    duplicate_identity_keys: tuple[str, ...] = ()
    capability_gap_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    policy: DoctorPlanRefillPolicy = field(default_factory=DoctorPlanRefillPolicy)
    plan_root: str = ""
    plan_revision: int = 0
    fixed_point_complete: bool = False
    derived_runtime_admitted: bool = False
    receipt_id: str = ""
    next_memory: DoctorPlanRefillMemory = field(default_factory=DoctorPlanRefillMemory)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorPlanRefillDisposition, "disposition"),
        )
        residuals = tuple(
            item
            if isinstance(item, DoctorPlanResidual)
            else DoctorPlanResidual.from_dict(item)
            for item in self.residuals
        )
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "successors", tuple(self.successors))
        object.__setattr__(self, "work_proposals", tuple(self.work_proposals))
        object.__setattr__(
            self,
            "backoff_identity_keys",
            _ids(self.backoff_identity_keys, "backoff_identity_keys", maximum=MAX_RESIDUALS),
        )
        object.__setattr__(
            self,
            "duplicate_identity_keys",
            _ids(
                self.duplicate_identity_keys,
                "duplicate_identity_keys",
                maximum=MAX_RESIDUALS,
            ),
        )
        object.__setattr__(
            self,
            "capability_gap_ids",
            _ids(self.capability_gap_ids, "capability_gap_ids", maximum=MAX_RESIDUALS),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        if not isinstance(self.policy, DoctorPlanRefillPolicy):
            object.__setattr__(
                self, "policy", DoctorPlanRefillPolicy.from_dict(self.policy)
            )
        object.__setattr__(
            self, "plan_root", _identifier(self.plan_root, "plan_root", required=False)
        )
        object.__setattr__(
            self,
            "plan_revision",
            _nonneg_int(self.plan_revision, "plan_revision", maximum=2**31 - 1),
        )
        object.__setattr__(
            self,
            "fixed_point_complete",
            _bool(self.fixed_point_complete, "fixed_point_complete"),
        )
        object.__setattr__(
            self,
            "derived_runtime_admitted",
            _bool(self.derived_runtime_admitted, "derived_runtime_admitted"),
        )
        if self.derived_runtime_admitted and not self.policy.derived_runtime_admission_enabled:
            raise DoctorPlanRefillAuthorityError(
                "derived runtime admission requires PDR-081 gate enablement"
            )
        if not isinstance(self.next_memory, DoctorPlanRefillMemory):
            object.__setattr__(
                self, "next_memory", DoctorPlanRefillMemory.from_dict(self.next_memory)
            )
        rid = _optional_text(self.receipt_id, "receipt_id", limit=MAX_ID_BYTES)
        object.__setattr__(self, "receipt_id", rid or self.content_id)

    @property
    def content_id(self) -> str:
        # Avoid floats from ObjectiveWorkProposal (confidence/cost/novelty) so
        # the receipt identity stays compatible with canonical proof contracts.
        return _stable_digest(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_PLAN_REFILL_RECEIPT_SCHEMA,
            "interface": DOCTOR_PLAN_REFILL_INTERFACE,
            "disposition": self.disposition.value,
            "residual_ids": [item.residual_id for item in self.residuals],
            "identity_keys": [item.identity_key for item in self.residuals],
            "decision_dispositions": [
                item.disposition.value for item in self.decisions
            ],
            "successor_item_keys": [
                item.delta_item.item_key for item in self.successors
            ],
            "work_proposal_ids": [
                getattr(item, "canonical_id", "") or getattr(item, "source_id", "")
                for item in self.work_proposals
            ],
            "backoff_identity_keys": list(self.backoff_identity_keys),
            "duplicate_identity_keys": list(self.duplicate_identity_keys),
            "capability_gap_ids": list(self.capability_gap_ids),
            "reason_codes": list(self.reason_codes),
            "plan_root": self.plan_root,
            "plan_revision": self.plan_revision,
            "fixed_point_complete": self.fixed_point_complete,
            "derived_runtime_admitted": self.derived_runtime_admitted,
            "completion_authority": False,
            "mutation_authority": False,
        }

    @property
    def emits_work(self) -> bool:
        return bool(self.successors or self.work_proposals)

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def mutation_authority(self) -> bool:
        return False

    def proposed_delta_items(self) -> tuple[PlanDeltaItem, ...]:
        return tuple(item.delta_item for item in self.successors)

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": DOCTOR_PLAN_REFILL_RECEIPT_SCHEMA,
            "interface": DOCTOR_PLAN_REFILL_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "disposition": self.disposition.value,
            "residuals": [item.to_dict() for item in self.residuals],
            "decisions": [item.to_dict() for item in self.decisions],
            "successors": [item.to_dict() for item in self.successors],
            "work_proposals": [item.to_dict() for item in self.work_proposals],
            "backoff_identity_keys": list(self.backoff_identity_keys),
            "duplicate_identity_keys": list(self.duplicate_identity_keys),
            "capability_gap_ids": list(self.capability_gap_ids),
            "reason_codes": list(self.reason_codes),
            "policy": self.policy.to_dict(),
            "plan_root": self.plan_root,
            "plan_revision": self.plan_revision,
            "fixed_point_complete": self.fixed_point_complete,
            "derived_runtime_admitted": self.derived_runtime_admitted,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
            "derived_runtime_source_id": self.policy.derived_runtime_source_id,
            "emits_work": self.emits_work,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "next_memory": self.next_memory.to_dict(),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


# ---------------------------------------------------------------------------
# Extraction from fixed-point / outcome shapes
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def fixed_point_is_successful(source: Any) -> bool:
    """Return True when a Doctor fixed-point source is residual-free success."""

    if source is None:
        return False
    if isinstance(source, bool):
        return source
    complete = getattr(source, "complete", None)
    if isinstance(complete, bool) and complete:
        residual_free = getattr(source, "residual_free", None)
        if residual_free is False:
            return False
        residual_ids = getattr(source, "residual_finding_ids", None)
        if residual_ids:
            return False
        open_frontiers = getattr(source, "open_frontier_ids", None)
        if open_frontiers:
            return False
        return True
    payload = _as_mapping(source)
    if payload is None:
        return False
    if payload.get("complete") is True or payload.get("residual_free") is True:
        if payload.get("residual_finding_ids") or payload.get("open_frontier_ids"):
            return False
        if payload.get("claims_completion") is False and payload.get("complete") is not True:
            return False
        # Nested report/fixed_point forms from DoctorFixedPointOutcome.
        report = payload.get("report")
        if isinstance(report, Mapping) and report.get("complete") is False:
            return False
        fixed_point = payload.get("fixed_point")
        if isinstance(fixed_point, Mapping):
            if fixed_point.get("complete") is False:
                return False
            if fixed_point.get("residual_finding_ids") or fixed_point.get(
                "open_frontier_ids"
            ):
                return False
        return bool(
            payload.get("complete")
            or payload.get("residual_free")
            or (
                isinstance(fixed_point, Mapping)
                and fixed_point.get("complete") is True
            )
        )
    return False


def extract_residuals_from_fixed_point(
    source: Any,
    *,
    plan_id: str = "",
    transaction_id: str = "",
    root_id: str = "",
    attempt_id: str = "",
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID,
    default_paths: Sequence[str] = (),
) -> tuple[DoctorPlanResidual, ...]:
    """Project Doctor fixed-point / outcome evidence into residual records.

    A successful residual-free fixed point yields an empty tuple.
    """

    if fixed_point_is_successful(source):
        return ()

    residual_finding_ids: list[str] = []
    open_frontier_ids: list[str] = []
    reason_codes: list[str] = []
    capability_gaps: list[Mapping[str, Any]] = []
    residual_payloads: list[Mapping[str, Any]] = []

    def _collect(payload: Mapping[str, Any]) -> None:
        for key in ("residual_finding_ids", "residual_ids", "open_obligation_ids"):
            raw = payload.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                residual_finding_ids.extend(str(item) for item in raw if str(item).strip())
        raw_frontiers = payload.get("open_frontier_ids")
        if isinstance(raw_frontiers, Sequence) and not isinstance(
            raw_frontiers, (str, bytes)
        ):
            open_frontier_ids.extend(
                str(item) for item in raw_frontiers if str(item).strip()
            )
        raw_reasons = payload.get("reason_codes")
        if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, (str, bytes)):
            reason_codes.extend(str(item) for item in raw_reasons if str(item).strip())
        for key in ("capability_gaps", "missing_capabilities"):
            raw_gaps = payload.get(key)
            if isinstance(raw_gaps, Sequence) and not isinstance(raw_gaps, (str, bytes)):
                for gap in raw_gaps:
                    if isinstance(gap, Mapping):
                        capability_gaps.append(gap)
                    elif str(gap).strip():
                        capability_gaps.append({"required_capability": str(gap).strip()})
        raw_residuals = payload.get("residuals")
        if isinstance(raw_residuals, Sequence) and not isinstance(
            raw_residuals, (str, bytes)
        ):
            for item in raw_residuals:
                if isinstance(item, Mapping):
                    residual_payloads.append(item)
                elif str(item).strip():
                    residual_finding_ids.append(str(item).strip())
        nested_fp = payload.get("fixed_point")
        if isinstance(nested_fp, Mapping):
            _collect(nested_fp)
        nested_report = payload.get("report")
        if isinstance(nested_report, Mapping):
            _collect(nested_report)
        for iter_key in ("iteration_receipts", "iterations"):
            iterations = payload.get(iter_key)
            if isinstance(iterations, Sequence) and not isinstance(
                iterations, (str, bytes)
            ):
                for item in iterations:
                    nested = _as_mapping(item)
                    if nested is not None:
                        _collect(nested)

    # Attribute-based collection for typed receipts.
    for attr, sink in (
        ("residual_finding_ids", residual_finding_ids),
        ("open_frontier_ids", open_frontier_ids),
        ("reason_codes", reason_codes),
    ):
        raw = getattr(source, attr, None)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            sink.extend(str(item) for item in raw if str(item).strip())

    payload = _as_mapping(source)
    if payload is not None:
        _collect(payload)

    # Explicit residual list attributes.
    explicit = getattr(source, "residuals", None)
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        for item in explicit:
            if isinstance(item, DoctorPlanResidual):
                residual_payloads.append(item.to_dict())
            elif isinstance(item, Mapping):
                residual_payloads.append(item)
            elif str(item).strip():
                residual_finding_ids.append(str(item).strip())

    resolved_plan_id = _identifier(
        plan_id or getattr(source, "plan_id", "") or "",
        "plan_id",
        required=False,
    )
    resolved_txn = _identifier(
        transaction_id or getattr(source, "transaction_id", "") or "",
        "transaction_id",
        required=False,
    )
    resolved_root = _identifier(
        root_id
        or getattr(source, "candidate_tree_id", "")
        or getattr(source, "committed_tree_cid", "")
        or "",
        "root_id",
        required=False,
    )
    resolved_attempt = _identifier(attempt_id, "attempt_id", required=False)
    paths = _paths(default_paths, "default_paths")

    residuals: list[DoctorPlanResidual] = []
    seen: set[str] = set()

    for item in residual_payloads:
        residual = (
            item
            if isinstance(item, DoctorPlanResidual)
            else DoctorPlanResidual.from_dict(
                {
                    **dict(item),
                    "plan_id": item.get("plan_id") or resolved_plan_id,
                    "transaction_id": item.get("transaction_id") or resolved_txn,
                    "root_id": item.get("root_id") or resolved_root,
                    "attempt_id": item.get("attempt_id") or resolved_attempt,
                    "parent_goal_id": item.get("parent_goal_id") or parent_goal_id,
                }
            )
        )
        if residual.identity_key in seen:
            continue
        seen.add(residual.identity_key)
        residuals.append(residual)

    for finding_id in residual_finding_ids:
        key = residual_identity_key(
            issue_id=finding_id,
            obligation_id=finding_id,
            root_id=resolved_root,
            attempt_id=resolved_attempt,
        )
        if key in seen:
            continue
        seen.add(key)
        residuals.append(
            DoctorPlanResidual(
                issue_id=finding_id,
                obligation_id=finding_id,
                root_id=resolved_root,
                attempt_id=resolved_attempt,
                kind=DoctorResidualKind.OPEN_OBLIGATION,
                finding_id=finding_id,
                plan_id=resolved_plan_id,
                transaction_id=resolved_txn,
                parent_goal_id=parent_goal_id,
                predicted_files=paths,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
                title=f"Resolve Doctor residual {finding_id}",
                rationale="Open or failed Doctor residual after fixed-point iteration.",
            )
        )

    for frontier_id in open_frontier_ids:
        key = residual_identity_key(
            issue_id=f"frontier:{frontier_id}",
            obligation_id=frontier_id,
            root_id=resolved_root,
            attempt_id=resolved_attempt,
        )
        if key in seen:
            continue
        seen.add(key)
        residuals.append(
            DoctorPlanResidual(
                issue_id=f"frontier:{frontier_id}",
                obligation_id=frontier_id,
                root_id=resolved_root,
                attempt_id=resolved_attempt,
                kind=DoctorResidualKind.FRONTIER,
                frontier_id=frontier_id,
                plan_id=resolved_plan_id,
                transaction_id=resolved_txn,
                parent_goal_id=parent_goal_id,
                predicted_files=paths,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
                title=f"Close Doctor frontier {frontier_id}",
                rationale="Open required impact frontier after Doctor fixed-point.",
            )
        )

    for gap in capability_gaps:
        capability = str(
            gap.get("required_capability")
            or gap.get("capability")
            or gap.get("capability_id")
            or ""
        ).strip()
        provider = str(
            gap.get("required_provider") or gap.get("provider") or ""
        ).strip()
        conformance = str(
            gap.get("required_conformance")
            or gap.get("conformance")
            or gap.get("conformance_work")
            or ""
        ).strip()
        issue = (
            capability
            or provider
            or conformance
            or str(gap.get("issue_id") or "capability-gap")
        )
        key = residual_identity_key(
            issue_id=f"capability:{issue}",
            obligation_id=capability or issue,
            root_id=resolved_root,
            attempt_id=resolved_attempt,
        )
        if key in seen:
            continue
        seen.add(key)
        residuals.append(
            DoctorPlanResidual(
                issue_id=f"capability:{issue}",
                obligation_id=capability or issue,
                root_id=resolved_root,
                attempt_id=resolved_attempt,
                kind=DoctorResidualKind.CAPABILITY_GAP,
                plan_id=resolved_plan_id,
                transaction_id=resolved_txn,
                parent_goal_id=parent_goal_id,
                required_capability=capability,
                required_provider=provider,
                required_conformance=conformance,
                predicted_files=paths,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
                title=_capability_gap_title(capability, provider, conformance),
                rationale=(
                    "Doctor fixed-point lost or lacks a required capability; "
                    "emit exact provider/conformance work."
                ),
            )
        )

    return tuple(residuals)


def _capability_gap_title(
    capability: str, provider: str, conformance: str
) -> str:
    parts: list[str] = []
    if provider:
        parts.append(f"provider={provider}")
    if capability:
        parts.append(f"capability={capability}")
    if conformance:
        parts.append(f"conformance={conformance}")
    detail = ", ".join(parts) if parts else "unspecified"
    return f"Capability gap: {detail}"


# ---------------------------------------------------------------------------
# Mapping / proposal builders
# ---------------------------------------------------------------------------


def residual_fingerprint(residual: DoctorPlanResidual) -> str:
    """Semantic fingerprint for unchanged-failure backoff (excludes attempt time)."""

    return _stable_digest(
        {
            "identity": residual.identity_payload,
            "kind": residual.kind.value,
            "finding_id": residual.finding_id,
            "frontier_id": residual.frontier_id,
            "counterexample_id": residual.counterexample_id,
            "predicted_files": list(residual.predicted_files),
            "context_paths": list(residual.context_paths),
            "predicted_symbols": list(residual.predicted_symbols),
            "attempted_strategies": list(residual.attempted_strategies),
            "required_capability": residual.required_capability,
            "required_provider": residual.required_provider,
            "required_conformance": residual.required_conformance,
            "reason_codes": list(residual.reason_codes),
            "unchanged_failure": residual.unchanged_failure,
        }
    )


def _match_plan_node(
    residual: DoctorPlanResidual, plan: DoctorPlanContext
) -> DoctorPlanNode | None:
    if residual.parent_task_cid:
        for node in plan.nodes:
            if node.kind == "task" and node.node_cid == residual.parent_task_cid:
                return node
    if residual.parent_goal_cid:
        for node in plan.nodes:
            if node.kind == "goal" and node.node_cid == residual.parent_goal_cid:
                return node
    if residual.obligation_id:
        for node in plan.nodes:
            if residual.obligation_id in node.obligation_ids:
                return node
    if residual.issue_id:
        for node in plan.nodes:
            if residual.issue_id in node.issue_ids:
                return node
    # Path overlap as a last, conservative mapping signal.
    residual_paths = set(residual.minimal_paths())
    if residual_paths:
        for node in plan.nodes:
            if residual_paths.intersection(node.predicted_files):
                return node
    return None


def _lifecycle_for_node(node: DoctorPlanNode) -> LifecycleState:
    try:
        return LifecycleState(node.lifecycle)
    except ValueError:
        return LifecycleState.UNSTARTED


def build_append_only_successor(
    residual: DoctorPlanResidual,
    parent: DoctorPlanNode,
    *,
    policy: DoctorPlanRefillPolicy,
) -> DoctorPlanSuccessor:
    """Build an append-only ADD_TASK (or ADD_GOAL) delta for a mapped residual."""

    after_payload = {
        "kind": "doctor_residual_successor",
        "identity": residual.identity_payload,
        "parent_node_cid": parent.node_cid,
        "residual_id": residual.residual_id,
        "title": residual.title or f"Doctor residual {residual.issue_id}",
        "predicted_files": list(residual.minimal_paths()),
        "validation_commands": list(residual.validation_commands),
    }
    after_cid = _content_cid(after_payload)
    lifecycle = _lifecycle_for_node(parent)
    deferred = lifecycle in {
        LifecycleState.CLAIMED,
        LifecycleState.RUNNING,
        LifecycleState.SETTLING,
        LifecycleState.COMPLETED,
        LifecycleState.ACCEPTED,
    }
    # Residual repair work is always an append-only task successor.
    operation = PlanDeltaOperation.ADD_TASK

    item_key = "delta:doctor-residual:" + residual.identity_key.split(":")[-1][:48]
    rationale = residual.rationale or (
        f"Append-only successor for Doctor residual {residual.issue_id}"
    )
    effect = (
        DeltaEffectClass.DEFERRED if deferred else DeltaEffectClass.MATERIALIZABLE_NOW
    )
    preconditions: tuple[str, ...] = ()
    if deferred:
        preconditions = (f"target-terminal:{parent.node_cid}",)

    delta = PlanDeltaItem(
        item_key=item_key,
        operation=operation,
        target_cid=parent.node_cid,
        expected_target_lifecycle=lifecycle,
        expected_target_spec_revision="",
        before_digest="",
        after_record_cid=after_cid,
        effect_class=effect,
        rationale=rationale[: MAX_TEXT_BYTES],
        provenance={
            "source": PRODUCER_ID,
            "residual_id": residual.residual_id,
            "identity_key": residual.identity_key,
            "issue_id": residual.issue_id,
            "obligation_id": residual.obligation_id,
            "root_id": residual.root_id,
            "attempt_id": residual.attempt_id,
            "append_only": True,
            "completion_authority": False,
            "mutation_authority": False,
        },
        preconditions=preconditions,
        expected_effects=("append-task", f"resolve:{residual.issue_id}"),
        affected_goal_cids=(
            (parent.node_cid,) if parent.kind == "goal" else ()
        ),
        affected_task_cids=(
            (after_cid, parent.node_cid)
            if parent.kind == "task"
            else (after_cid,)
        ),
        affected_paths=residual.minimal_paths(),
        dependency_impact=(parent.node_cid,),
        conflict_impact=(),
        resource_impact=(policy.resource_class,),
    )
    return DoctorPlanSuccessor(
        residual_id=residual.residual_id,
        identity_key=residual.identity_key,
        parent_node_cid=parent.node_cid,
        parent_lifecycle=lifecycle.value,
        operation=operation.value,
        delta_item=delta,
        target_source=DoctorPlanTargetSource.PLAN_STEER_DELTA,
    )


def build_work_proposal(
    residual: DoctorPlanResidual,
    *,
    policy: DoctorPlanRefillPolicy,
    target_source: DoctorPlanTargetSource,
) -> ObjectiveWorkProposal:
    """Emit a bounded ObjectiveWorkProposal for an unmapped residual."""

    if residual.is_capability_gap:
        title = residual.title or _capability_gap_title(
            residual.required_capability,
            residual.required_provider,
            residual.required_conformance,
        )
        evidence = tuple(
            item
            for item in (
                residual.required_capability
                and f"capability:{residual.required_capability}",
                residual.required_provider
                and f"provider:{residual.required_provider}",
                residual.required_conformance
                and f"conformance:{residual.required_conformance}",
                "doctor-capability-gap",
            )
            if item
        )
        rationale = (
            residual.rationale
            or "Capability gap names exact provider/conformance work; no mutation."
        )
    else:
        title = residual.title or f"Doctor residual successor for {residual.issue_id}"
        evidence = (
            f"issue:{residual.issue_id}",
            f"obligation:{residual.obligation_id}" if residual.obligation_id else "",
            f"root:{residual.root_id}" if residual.root_id else "",
            f"attempt:{residual.attempt_id}" if residual.attempt_id else "",
            "doctor-residual",
        )
        evidence = tuple(item for item in evidence if item)
        rationale = residual.rationale or (
            "Bounded derived work for an unmapped Doctor residual."
        )

    family_key = (
        _OBJECTIVE_FAMILY_PREFIX
        + residual.kind.value
        + "/"
        + _stable_digest({"issue": residual.issue_id, "kind": residual.kind.value})[
            len("sha256:") : len("sha256:") + 16
        ]
    )
    instance_key = (
        _OBJECTIVE_INSTANCE_PREFIX
        + residual.identity_key.split(":")[-1][:32]
    )

    paths = residual.minimal_paths()[: policy.max_paths_per_residual]
    context = residual.context_paths[: policy.max_context_paths]
    if not context:
        context = paths

    acceptance = (
        f"resolve residual identity {residual.identity_key}",
        "no completion authority",
        "no seed board mutation",
    )
    if residual.is_capability_gap:
        acceptance = acceptance + (
            "name exact provider/conformance work",
            *(
                (f"provider={residual.required_provider}",)
                if residual.required_provider
                else ()
            ),
            *(
                (f"capability={residual.required_capability}",)
                if residual.required_capability
                else ()
            ),
            *(
                (f"conformance={residual.required_conformance}",)
                if residual.required_conformance
                else ()
            ),
        )

    preconditions = (
        f"doctor residual {residual.issue_id} is open",
        "derived runtime admission gated by PDR-081"
        if target_source is DoctorPlanTargetSource.DERIVED_RUNTIME
        else "proposal-only admission",
    )
    effects = (
        f"target residual {residual.issue_id}",
        "proposal only; no mutation authority",
    )

    return ObjectiveWorkProposal(
        kind=ObjectiveWorkKind.TASK,
        title=title,
        parent_goal_id=residual.parent_goal_id or policy.parent_goal_id,
        parent_objective_terms=(
            residual.parent_goal_id or policy.parent_goal_id,
            "doctor-residual",
            residual.kind.value,
        ),
        expected_evidence_delta=evidence,
        dependencies=tuple(
            item
            for item in (residual.obligation_id, residual.parent_task_cid)
            if item
        ),
        predicted_files=paths,
        predicted_symbols=residual.predicted_symbols,
        validation_commands=residual.validation_commands
        or (
            "python -m pytest "
            "test/api/test_agent_supervisor_doctor_plan_refill.py -q",
        ),
        confidence=0.55 if residual.is_capability_gap else 0.7,
        estimated_cost=1.0,
        novelty=1.0,
        depth=1,
        estimated_tokens=0,
        retry_count=0,
        source=PRODUCER_ID,
        source_id=residual.residual_id,
        rationale=rationale,
        family_key=family_key,
        instance_key=instance_key,
        # semantic_key / canonical_id are derived by ObjectiveWorkProposal.
        semantic_key="",
        canonical_id="",
        acceptance_subset=acceptance,
        preconditions=preconditions,
        effects=effects,
        evidence_subset=evidence + residual.evidence_refs,
        conflicts=(),
        context_paths=context,
        resource_class=policy.resource_class,
        token_class=policy.token_class,
        merge_fate="",
        rejection_reasons=(),
    )


# ---------------------------------------------------------------------------
# Main refill entry
# ---------------------------------------------------------------------------


def dedupe_residuals(
    residuals: Sequence[DoctorPlanResidual | Mapping[str, Any]],
) -> tuple[tuple[DoctorPlanResidual, ...], tuple[str, ...]]:
    """Deduplicate residuals by exact issue/obligation/root/attempt identity."""

    unique: list[DoctorPlanResidual] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for raw in residuals:
        residual = (
            raw
            if isinstance(raw, DoctorPlanResidual)
            else DoctorPlanResidual.from_dict(raw)
        )
        key = residual.identity_key
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        unique.append(residual)
    return tuple(unique), tuple(duplicates)


def refill_doctor_plan_residuals(
    residuals: Sequence[DoctorPlanResidual | Mapping[str, Any]] | None = None,
    *,
    fixed_point: Any = None,
    plan: DoctorPlanContext | Mapping[str, Any] | None = None,
    memory: DoctorPlanRefillMemory | Mapping[str, Any] | None = None,
    policy: DoctorPlanRefillPolicy | Mapping[str, Any] | None = None,
    plan_id: str = "",
    transaction_id: str = "",
    root_id: str = "",
    attempt_id: str = "",
) -> DoctorPlanRefillReceipt:
    """Map Doctor residuals into append-only plan successors or work proposals.

    Successful fixed points emit no work.  Unchanged failures back off.
    Capability gaps name exact provider/conformance work.  No completion or
    mutation authority is granted.  Derived runtime admission stays gated
    until PDR-081 enables it on the policy.
    """

    resolved_policy = (
        policy
        if isinstance(policy, DoctorPlanRefillPolicy)
        else DoctorPlanRefillPolicy.from_dict(policy)
    )
    resolved_plan = (
        plan
        if isinstance(plan, DoctorPlanContext)
        else DoctorPlanContext.from_dict(plan)
    )
    resolved_memory = (
        memory
        if isinstance(memory, DoctorPlanRefillMemory)
        else DoctorPlanRefillMemory.from_dict(memory)
    )

    if fixed_point is not None and fixed_point_is_successful(fixed_point):
        return DoctorPlanRefillReceipt(
            disposition=DoctorPlanRefillDisposition.FIXED_POINT_CLOSED,
            residuals=(),
            decisions=(),
            successors=(),
            work_proposals=(),
            reason_codes=("fixed_point_residual_free",),
            policy=resolved_policy,
            plan_root=resolved_plan.plan_root,
            plan_revision=resolved_plan.plan_revision,
            fixed_point_complete=True,
            derived_runtime_admitted=False,
            next_memory=resolved_memory,
        )

    collected: list[DoctorPlanResidual | Mapping[str, Any]] = []
    if residuals:
        collected.extend(list(residuals))
    if fixed_point is not None:
        collected.extend(
            extract_residuals_from_fixed_point(
                fixed_point,
                plan_id=plan_id,
                transaction_id=transaction_id,
                root_id=root_id,
                attempt_id=attempt_id,
                parent_goal_id=resolved_policy.parent_goal_id,
            )
        )

    if not collected:
        return DoctorPlanRefillReceipt(
            disposition=DoctorPlanRefillDisposition.EMPTY_INPUT,
            residuals=(),
            decisions=(),
            successors=(),
            work_proposals=(),
            reason_codes=("no_residuals",),
            policy=resolved_policy,
            plan_root=resolved_plan.plan_root,
            plan_revision=resolved_plan.plan_revision,
            fixed_point_complete=False,
            derived_runtime_admitted=False,
            next_memory=resolved_memory,
        )

    unique, duplicates = dedupe_residuals(collected)
    if len(unique) > resolved_policy.max_residuals:
        unique = unique[: resolved_policy.max_residuals]
        bound_exceeded = True
    else:
        bound_exceeded = False

    memory_index = resolved_memory.by_identity()
    decisions: list[DoctorResidualDecision] = []
    successors: list[DoctorPlanSuccessor] = []
    proposals: list[ObjectiveWorkProposal] = []
    backoff_keys: list[str] = []
    capability_gap_ids: list[str] = []
    next_entries: dict[str, DoctorPlanRefillMemoryEntry] = dict(memory_index)
    allowed_ops = frozenset(resolved_plan.allowed_delta_operations)
    add_task_allowed = PlanDeltaOperation.ADD_TASK.value in allowed_ops

    for residual in unique:
        fingerprint = residual_fingerprint(residual)
        prior = memory_index.get(residual.identity_key)

        # Unchanged failure backoff.
        if residual.unchanged_failure or residual.kind is DoctorResidualKind.UNCHANGED_FAILURE:
            backoff_keys.append(residual.identity_key)
            decisions.append(
                DoctorResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF,
                    reason_codes=("unchanged_failure",),
                )
            )
            next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                identity_key=residual.identity_key,
                attempt_count=(prior.attempt_count + 1) if prior else 1,
                last_fingerprint=fingerprint,
                last_seen_epoch_s=resolved_memory.now_epoch_s,
                last_disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF.value,
            )
            continue

        if (
            prior is not None
            and prior.last_fingerprint == fingerprint
            and prior.attempt_count >= resolved_policy.backoff_identical_attempts
        ):
            # Cooldown window (when timestamps are available).
            if (
                resolved_policy.cooldown_seconds > 0
                and resolved_memory.now_epoch_s > 0
                and prior.last_seen_epoch_s > 0
                and (
                    resolved_memory.now_epoch_s - prior.last_seen_epoch_s
                    < resolved_policy.cooldown_seconds
                )
            ):
                backoff_keys.append(residual.identity_key)
                decisions.append(
                    DoctorResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF,
                        reason_codes=("identical_fingerprint_cooldown",),
                    )
                )
                next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                    identity_key=residual.identity_key,
                    attempt_count=prior.attempt_count + 1,
                    last_fingerprint=fingerprint,
                    last_seen_epoch_s=resolved_memory.now_epoch_s,
                    last_disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF.value,
                )
                continue
            if resolved_policy.cooldown_seconds == 0 or resolved_memory.now_epoch_s == 0:
                backoff_keys.append(residual.identity_key)
                decisions.append(
                    DoctorResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF,
                        reason_codes=("identical_fingerprint_backoff",),
                    )
                )
                next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                    identity_key=residual.identity_key,
                    attempt_count=prior.attempt_count + 1,
                    last_fingerprint=fingerprint,
                    last_seen_epoch_s=resolved_memory.now_epoch_s,
                    last_disposition=DoctorResidualDisposition.UNCHANGED_BACKOFF.value,
                )
                continue

        # Capability gaps always become named provider/conformance work.
        if residual.is_capability_gap:
            if len(proposals) >= resolved_policy.max_proposals:
                decisions.append(
                    DoctorResidualDecision(
                        residual_id=residual.residual_id,
                        identity_key=residual.identity_key,
                        disposition=DoctorResidualDisposition.BOUND_REJECTED,
                        reason_codes=("proposal_bound",),
                    )
                )
                continue
            target = _resolve_target_source(resolved_policy, mapped=False)
            proposal = build_work_proposal(
                residual, policy=resolved_policy, target_source=target
            )
            proposals.append(proposal)
            capability_gap_ids.append(residual.residual_id)
            decisions.append(
                DoctorResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=DoctorResidualDisposition.CAPABILITY_GAP_PROPOSAL,
                    reason_codes=("capability_gap",),
                    work_proposal=proposal,
                    target_source=target,
                )
            )
            next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                identity_key=residual.identity_key,
                attempt_count=(prior.attempt_count + 1) if prior else 1,
                last_fingerprint=fingerprint,
                last_seen_epoch_s=resolved_memory.now_epoch_s,
                last_disposition=DoctorResidualDisposition.CAPABILITY_GAP_PROPOSAL.value,
            )
            continue

        # Prefer append-only successor mapping onto the existing plan.
        parent = None
        if resolved_policy.prefer_append_only_successors and add_task_allowed:
            parent = _match_plan_node(residual, resolved_plan)

        if parent is not None and len(successors) < resolved_policy.max_successors:
            successor = build_append_only_successor(
                residual, parent, policy=resolved_policy
            )
            successors.append(successor)
            decisions.append(
                DoctorResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=DoctorResidualDisposition.MAPPED_SUCCESSOR,
                    reason_codes=("append_only_successor",),
                    successor=successor,
                    target_source=DoctorPlanTargetSource.PLAN_STEER_DELTA,
                )
            )
            next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
                identity_key=residual.identity_key,
                attempt_count=(prior.attempt_count + 1) if prior else 1,
                last_fingerprint=fingerprint,
                last_seen_epoch_s=resolved_memory.now_epoch_s,
                last_disposition=DoctorResidualDisposition.MAPPED_SUCCESSOR.value,
            )
            continue

        # Otherwise emit a bounded ObjectiveWorkProposal.
        if len(proposals) >= resolved_policy.max_proposals:
            decisions.append(
                DoctorResidualDecision(
                    residual_id=residual.residual_id,
                    identity_key=residual.identity_key,
                    disposition=DoctorResidualDisposition.BOUND_REJECTED,
                    reason_codes=("proposal_bound",),
                )
            )
            continue

        target = _resolve_target_source(resolved_policy, mapped=False)
        proposal = build_work_proposal(
            residual, policy=resolved_policy, target_source=target
        )
        proposals.append(proposal)
        decisions.append(
            DoctorResidualDecision(
                residual_id=residual.residual_id,
                identity_key=residual.identity_key,
                disposition=DoctorResidualDisposition.WORK_PROPOSAL,
                reason_codes=(
                    ("unmapped_residual",)
                    if parent is None
                    else ("successor_bound_or_op_denied",)
                ),
                work_proposal=proposal,
                target_source=target,
            )
        )
        next_entries[residual.identity_key] = DoctorPlanRefillMemoryEntry(
            identity_key=residual.identity_key,
            attempt_count=(prior.attempt_count + 1) if prior else 1,
            last_fingerprint=fingerprint,
            last_seen_epoch_s=resolved_memory.now_epoch_s,
            last_disposition=DoctorResidualDisposition.WORK_PROPOSAL.value,
        )

    # Derive overall disposition.
    disposition = _aggregate_disposition(
        successors=successors,
        proposals=proposals,
        backoff_keys=backoff_keys,
        capability_gap_ids=capability_gap_ids,
        bound_exceeded=bound_exceeded,
        policy=resolved_policy,
    )
    reason_codes: list[str] = []
    if duplicates:
        reason_codes.append("duplicates_collapsed")
    if bound_exceeded:
        reason_codes.append("residual_bound_truncated")
    if not resolved_policy.derived_runtime_admission_enabled and proposals:
        reason_codes.append("derived_runtime_gated_until_pdr_081")
    if backoff_keys and not successors and not proposals:
        reason_codes.append("all_residuals_backed_off")

    next_memory = DoctorPlanRefillMemory(
        entries=tuple(next_entries.values())[:MAX_MEMORY_ENTRIES],
        open_derived_task_count=resolved_memory.open_derived_task_count
        + (
            len(proposals)
            if resolved_policy.derived_runtime_admission_enabled
            else 0
        ),
        now_epoch_s=resolved_memory.now_epoch_s,
    )

    return DoctorPlanRefillReceipt(
        disposition=disposition,
        residuals=unique,
        decisions=tuple(decisions),
        successors=tuple(successors),
        work_proposals=tuple(proposals),
        backoff_identity_keys=tuple(dict.fromkeys(backoff_keys)),
        duplicate_identity_keys=tuple(dict.fromkeys(duplicates)),
        capability_gap_ids=tuple(dict.fromkeys(capability_gap_ids)),
        reason_codes=tuple(reason_codes),
        policy=resolved_policy,
        plan_root=resolved_plan.plan_root,
        plan_revision=resolved_plan.plan_revision,
        fixed_point_complete=False,
        derived_runtime_admitted=(
            resolved_policy.derived_runtime_admission_enabled and bool(proposals)
        ),
        next_memory=next_memory,
    )


def _resolve_target_source(
    policy: DoctorPlanRefillPolicy, *, mapped: bool
) -> DoctorPlanTargetSource:
    if mapped:
        return DoctorPlanTargetSource.PLAN_STEER_DELTA
    if policy.derived_runtime_admission_enabled:
        # Still only *label* the derived source; actual CAS write is PDR-081.
        if (
            policy.max_open_derived_tasks > 0
        ):
            return DoctorPlanTargetSource.DERIVED_RUNTIME
    return DoctorPlanTargetSource.OBJECTIVE_HEAP


def _aggregate_disposition(
    *,
    successors: Sequence[DoctorPlanSuccessor],
    proposals: Sequence[ObjectiveWorkProposal],
    backoff_keys: Sequence[str],
    capability_gap_ids: Sequence[str],
    bound_exceeded: bool,
    policy: DoctorPlanRefillPolicy,
) -> DoctorPlanRefillDisposition:
    if bound_exceeded and not successors and not proposals:
        return DoctorPlanRefillDisposition.BOUND_EXCEEDED
    has_succ = bool(successors)
    has_prop = bool(proposals)
    only_gaps = has_prop and len(capability_gap_ids) == len(proposals) and not has_succ
    if only_gaps:
        return DoctorPlanRefillDisposition.CAPABILITY_GAP
    if has_succ and has_prop:
        return DoctorPlanRefillDisposition.MIXED
    if has_succ:
        return DoctorPlanRefillDisposition.APPEND_ONLY_SUCCESSORS
    if has_prop:
        if (
            not policy.derived_runtime_admission_enabled
            and all(
                True for _ in proposals
            )
        ):
            # Proposals exist but derived runtime remains gated.
            return DoctorPlanRefillDisposition.WORK_PROPOSALS
        return DoctorPlanRefillDisposition.WORK_PROPOSALS
    if backoff_keys:
        return DoctorPlanRefillDisposition.UNCHANGED_BACKOFF
    return DoctorPlanRefillDisposition.EMPTY_INPUT


# ---------------------------------------------------------------------------
# Plan-supervisor integration helpers
# ---------------------------------------------------------------------------


def build_steer_proposed_delta_items(
    receipt: DoctorPlanRefillReceipt,
) -> tuple[PlanDeltaItem, ...]:
    """Return append-only delta items suitable for PlanSteerPreviewMaterials."""

    return receipt.proposed_delta_items()


def build_plan_steer_refill_materials(
    receipt: DoctorPlanRefillReceipt,
    *,
    request: Any,
    live_state: Any,
) -> dict[str, Any]:
    """Build proposal-only materials for :class:`PlanSupervisorService`.

    Returns a mapping that can be passed as ``materials`` to
    ``preview_steer``.  Never includes writers, completion flags, or seed
    board mutation handles.
    """

    return {
        "request": request,
        "live_state": live_state,
        "proposed_delta_items": list(build_steer_proposed_delta_items(receipt)),
        "expected_effects": [
            "doctor-residual-refill",
            *(
                f"resolve:{residual.issue_id}"
                for residual in receipt.residuals[:16]
            ),
        ],
        "doctor_plan_refill": {
            "receipt_id": receipt.receipt_id,
            "disposition": receipt.disposition.value,
            "emits_work": receipt.emits_work,
            "successor_count": len(receipt.successors),
            "proposal_count": len(receipt.work_proposals),
            "completion_authority": False,
            "mutation_authority": False,
            "derived_runtime_admitted": receipt.derived_runtime_admitted,
            "derived_runtime_gate": DERIVED_RUNTIME_SOURCE_GATE,
        },
    }


def doctor_residuals_for_steer(
    *,
    residuals: Sequence[DoctorPlanResidual | Mapping[str, Any]] | None = None,
    fixed_point: Any = None,
    plan: DoctorPlanContext | Mapping[str, Any] | None = None,
    memory: DoctorPlanRefillMemory | Mapping[str, Any] | None = None,
    policy: DoctorPlanRefillPolicy | Mapping[str, Any] | None = None,
    request: Any = None,
    live_state: Any = None,
) -> dict[str, Any]:
    """Convenience: refill then package steer materials + receipt.

    The returned mapping is proposal-only and never grants apply authority.
    """

    receipt = refill_doctor_plan_residuals(
        residuals,
        fixed_point=fixed_point,
        plan=plan,
        memory=memory,
        policy=policy,
    )
    materials = None
    if request is not None and live_state is not None:
        materials = build_plan_steer_refill_materials(
            receipt, request=request, live_state=live_state
        )
    return {
        "receipt": receipt,
        "materials": materials,
        "proposed_delta_items": list(receipt.proposed_delta_items()),
        "work_proposals": list(receipt.work_proposals),
        "completion_authority": False,
        "mutation_authority": False,
        "read_only": True,
    }


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


@dataclass
class DoctorPlanRefill:
    """Stateful helper for repeated residual → plan refill passes."""

    INTERFACE: Final[str] = DOCTOR_PLAN_REFILL_INTERFACE
    VERSION: Final[str] = DOCTOR_PLAN_REFILL_VERSION

    policy: DoctorPlanRefillPolicy = field(default_factory=DoctorPlanRefillPolicy)
    memory: DoctorPlanRefillMemory = field(default_factory=DoctorPlanRefillMemory)

    @property
    def producer_id(self) -> str:
        return PRODUCER_ID

    def refill(
        self,
        residuals: Sequence[DoctorPlanResidual | Mapping[str, Any]] | None = None,
        *,
        fixed_point: Any = None,
        plan: DoctorPlanContext | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> DoctorPlanRefillReceipt:
        receipt = refill_doctor_plan_residuals(
            residuals,
            fixed_point=fixed_point,
            plan=plan,
            memory=self.memory,
            policy=self.policy,
            **kwargs,
        )
        self.memory = receipt.next_memory
        return receipt

    def refill_fixed_point(
        self,
        fixed_point: Any,
        *,
        plan: DoctorPlanContext | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> DoctorPlanRefillReceipt:
        return self.refill(fixed_point=fixed_point, plan=plan, **kwargs)


def create_doctor_plan_refill(
    **kwargs: Any,
) -> DoctorPlanRefill:
    """Construct a :class:`DoctorPlanRefill` instance."""

    policy = kwargs.pop("policy", None)
    memory = kwargs.pop("memory", None)
    if kwargs:
        raise DoctorPlanRefillError(
            f"unknown create_doctor_plan_refill kwargs: {sorted(kwargs)}"
        )
    return DoctorPlanRefill(
        policy=(
            policy
            if isinstance(policy, DoctorPlanRefillPolicy)
            else DoctorPlanRefillPolicy.from_dict(policy)
        ),
        memory=(
            memory
            if isinstance(memory, DoctorPlanRefillMemory)
            else DoctorPlanRefillMemory.from_dict(memory)
        ),
    )


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_DERIVED_RUNTIME_SOURCE_ID",
    "DEFAULT_PARENT_GOAL_ID",
    "DERIVED_RUNTIME_SOURCE_GATE",
    "DOCTOR_PLAN_REFILL_INTERFACE",
    "DOCTOR_PLAN_REFILL_RECEIPT_SCHEMA",
    "DOCTOR_PLAN_RESIDUAL_INTERFACE",
    "DOCTOR_PLAN_RESIDUAL_SCHEMA",
    "PRODUCER_ID",
    "REFILL_AUTHORIZES_COMPLETION",
    "REFILL_AUTHORIZES_MUTATION",
    "REFILL_AUTHORIZES_SEED_BOARD_EDIT",
    "DoctorPlanContext",
    "DoctorPlanNode",
    "DoctorPlanRefill",
    "DoctorPlanRefillAuthorityError",
    "DoctorPlanRefillBoundsError",
    "DoctorPlanRefillDisposition",
    "DoctorPlanRefillError",
    "DoctorPlanRefillMemory",
    "DoctorPlanRefillMemoryEntry",
    "DoctorPlanRefillPolicy",
    "DoctorPlanRefillReceipt",
    "DoctorPlanResidual",
    "DoctorPlanSuccessor",
    "DoctorPlanTargetSource",
    "DoctorResidualDecision",
    "DoctorResidualDisposition",
    "DoctorResidualKind",
    "build_append_only_successor",
    "build_plan_steer_refill_materials",
    "build_steer_proposed_delta_items",
    "build_work_proposal",
    "create_doctor_plan_refill",
    "dedupe_residuals",
    "doctor_residuals_for_steer",
    "extract_residuals_from_fixed_point",
    "fixed_point_is_successful",
    "refill_doctor_plan_residuals",
    "residual_fingerprint",
    "residual_identity_key",
    "residual_identity_payload",
]
