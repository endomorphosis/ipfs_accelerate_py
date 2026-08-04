"""Deterministic, bounded critique of proof-directed plan candidates.

``PlanCritic@1`` is an independent admission input, not a wrapper around
candidate-supplied ``valid`` or ``admitted`` booleans.  It replays canonical
records where a repository contract exists and derives the remaining graph,
coverage, effect, gate, conflict, and resource facts from primitive fields.

The critic is intentionally tolerant at its input boundary: formal plans,
symbolic candidates, plan revisions, and compact test/adapter mappings can all
be inspected.  Its output is a closed, content-addressed ``PlanCritique@1``
record with bounded minimal cores, typed counterexamples, and exact record ids
which a replanner may change.  Accepted/completed/claimed/running records are
never advertised as repairable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


PLAN_CRITIQUE_INTERFACE: Final[str] = "PlanCritique@1"
# The task contract names the service and its result with one interface.  Keep
# both constant names for callers which distinguish producer from artifact.
PLAN_CRITIC_INTERFACE: Final[str] = PLAN_CRITIQUE_INTERFACE
PLAN_CRITIQUE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-critique@1"
)
PLAN_CRITIQUE_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-critique-finding@1"
)
PLAN_UNSAT_CORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-unsat-core@1"
)
PLAN_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/typed-plan-counterexample@1"
)
PLAN_CRITIQUE_VERSION: Final[int] = 1

_ID_RE = re.compile(r"^[^\x00\r\n\t]{1,2048}$")
_DIGEST_RE = re.compile(r"^(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_.-])(?:api[_-]?key|authorization|cookie|credential|hidden[_-]?"
    r"witness|password|passwd|private[_-]?key|private[_-]?witness|secret|token)"
    r"(?:$|[_.-])",
    re.IGNORECASE,
)
_FAIL_STATUSES = frozenset(
    {
        "blocked",
        "abstain",
        "abstained",
        "conflict",
        "counterexample",
        "deny",
        "denied",
        "detached",
        "error",
        "failed",
        "failure",
        "infeasible",
        "inconclusive",
        "invalid",
        "missing",
        "rejected",
        "reject",
        "stale",
        "unknown",
        "unsafe",
        "unverified",
        "violated",
    }
)
_IMMUTABLE_STATES = frozenset(
    {
        "accepted",
        "claimed",
        "completed",
        "executing",
        "historical",
        "history",
        "in_progress",
        "running",
        "started",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "candidate_id",
        "content_id",
        "graph_id",
        "identity",
        "plan_id",
        "portfolio_id",
        "record_id",
        "revision_id",
        "semantic_id",
        "snapshot_id",
    }
)


class PlanCriticError(ValueError):
    """A critique input or serialized critique is malformed."""


class PlanCritiqueBoundsError(PlanCriticError):
    """A critique cannot be represented inside its explicit bounds."""


class PlanDefectKind(str, Enum):
    SCHEMA_FAILURE = "schema_failure"
    IDENTITY_MISMATCH = "identity_mismatch"
    DUPLICATE_IDENTITY = "duplicate_identity"
    DEPENDENCY_CYCLE = "dependency_cycle"
    CYCLE = "dependency_cycle"
    ORPHAN_RECORD = "orphan_record"
    ORPHAN = "orphan_record"
    ORPHAN_TASK = "orphan_record"
    UNCOVERED_GOAL = "uncovered_goal"
    COVERAGE_FAILURE = "uncovered_goal"
    CONTRADICTION = "contradiction"
    UNSATISFIED_ASSUMPTION = "unsatisfied_assumption"
    MISSING_CONSUMER = "missing_consumer"
    INVALID_EFFECT = "invalid_effect"
    POLICY_FAILURE = "policy_failure"
    POLICY_VIOLATION = "policy_failure"
    IR_FAILURE = "ir_failure"
    IR_VIOLATION = "ir_failure"
    SECURITY_FAILURE = "security_failure"
    SECURITY_VIOLATION = "security_failure"
    PROOF_FAILURE = "proof_failure"
    CONFLICT_FAILURE = "conflict_failure"
    FALSE_PARALLELISM = "false_parallelism"
    RESOURCE_INFEASIBLE = "resource_infeasible"
    STALE_EVIDENCE = "stale_evidence"
    LIFECYCLE_FAILURE = "lifecycle_failure"
    LIFECYCLE_VIOLATION = "lifecycle_failure"
    BOUND_EXCEEDED = "bound_exceeded"


PlanCritiqueFindingKind = PlanDefectKind
CritiqueFindingKind = PlanDefectKind


class PlanCritiqueSeverity(str, Enum):
    ERROR = "error"
    REVIEW = "review"
    WARNING = "warning"


CritiqueSeverity = PlanCritiqueSeverity


class PlanCritiqueDecision(str, Enum):
    ACCEPTED = "accepted"
    READY = "accepted"
    REVIEW_REQUIRED = "review_required"
    REPAIR_REQUIRED = "repair_required"
    REJECTED = "rejected"
    BLOCKED = "rejected"


def _identifier(value: Any, name: str, *, allow_empty: bool = False) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise PlanCriticError(f"{name} must be a string")
    result = raw.strip()
    if not result and allow_empty:
        return ""
    if not result or not _ID_RE.fullmatch(result):
        raise PlanCriticError(f"{name} must be a bounded identifier")
    return result


def _text(value: Any, name: str, *, maximum: int = 2_048) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise PlanCriticError(f"{name} must not be empty")
    if "\x00" in result or len(result) > maximum:
        raise PlanCriticError(f"{name} exceeds its text bound")
    return result


def _sequence(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _ids(value: Any, name: str, *, maximum: int = 4_096) -> tuple[str, ...]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        values: Iterable[Any] = value.keys()
    else:
        values = _sequence(value)
    for item in values:
        if isinstance(item, Mapping):
            item = _record_id(item)
        if item in (None, ""):
            continue
        result.add(_identifier(str(item), name))
    if len(result) > maximum:
        raise PlanCritiqueBoundsError(f"{name} exceeds {maximum} entries")
    return tuple(sorted(result))


def _integer(
    value: Any,
    name: str,
    *,
    default: int = 0,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise PlanCriticError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PlanCriticError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise PlanCriticError(f"{name} is outside its allowed range")
    return result


def _mapping(value: Any, name: str = "record") -> dict[str, Any]:
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
        return {key: getattr(value, key) for key in fields}
    raise PlanCriticError(f"{name} must be a mapping or expose to_dict()")


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 10:
        raise PlanCritiqueBoundsError("critique payload exceeds nesting bound")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanCriticError("critique payload contains a non-finite number")
        return format(value, ".12g")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE_KEY_RE.search(str(key))
                else _plain(item, depth=depth + 1)
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    return str(value)


def _digest(label: str, value: Any) -> str:
    body = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{label}:sha256:{hashlib.sha256(body).hexdigest()}"


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _record_id(record: Mapping[str, Any], fallback: str = "") -> str:
    for name in (
        "record_id",
        "effect_id",
        "constraint_id",
        "assumption_id",
        "obligation_id",
        "consumer_id",
        "step_id",
        "task_id",
        "goal_id",
        "candidate_id",
        "id",
    ):
        value = record.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return fallback


def _record_values(value: Any) -> tuple[dict[str, Any], ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in (
                "id",
                "task_id",
                "goal_id",
                "record_id",
                "constraint_id",
                "effect_id",
                "assumption_id",
                "obligation_id",
                "consumer_id",
                "candidate_id",
                "step_id",
                "plan_id",
                "portfolio_id",
                "revision_id",
            )
        ):
            return (_mapping(value),)
        result = []
        for key, item in value.items():
            record = _mapping(item) if isinstance(item, Mapping) else {"value": item}
            if not _record_id(record):
                record["id"] = str(key)
            result.append(record)
        return tuple(result)
    return tuple(
        _mapping(item) if isinstance(item, Mapping) or hasattr(item, "to_dict") else {"id": str(item)}
        for item in _sequence(value)
    )


def _field(source: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in source and source[name] not in (None, "", (), [], {}):
            return source[name]
    return default


def _selected_symbolic(source: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the selected symbolic record without trusting a convenience flag."""

    selected = source.get("selected")
    if isinstance(selected, Mapping):
        symbolic = selected.get("symbolic_candidate", selected)
        if isinstance(symbolic, Mapping):
            return symbolic
    selected_snapshot_id = str(source.get("selected_snapshot_id") or "")
    snapshots = _record_values(source.get("snapshots"))
    if selected_snapshot_id:
        snapshots = tuple(
            item
            for item in snapshots
            if str(item.get("snapshot_id") or "") == selected_snapshot_id
        )
    if len(snapshots) == 1:
        symbolic = snapshots[0].get("symbolic_candidate")
        if isinstance(symbolic, Mapping):
            return symbolic
    return {}


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    unknown = set(payload) - set(expected)
    if unknown:
        raise PlanCriticError(
            f"{label} uses unknown fields: {', '.join(sorted(map(str, unknown)))}"
        )


@dataclass(frozen=True)
class PlanCritiqueBounds:
    max_records: int = 4_096
    max_findings: int = 256
    max_core_items: int = 16
    max_counterexamples: int = 256
    max_witness_items: int = 32
    max_output_bytes: int = 512 * 1_024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1, maximum=1_000_000),
            )

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCritiqueBounds":
        _reject_unknown_fields(payload, set(cls.__dataclass_fields__), "critique bounds")
        return cls(
            **{
                name: payload.get(name, field.default)
                for name, field in cls.__dataclass_fields__.items()
            }
        )


@dataclass(frozen=True)
class PlanUnsatCore:
    kind: PlanDefectKind
    constraint_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    minimal: bool = True
    bounded: bool = True
    core_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PlanDefectKind(self.kind))
        for name in ("constraint_ids", "record_ids", "assumption_ids", "evidence_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if not self.constraint_ids and not self.record_ids and not self.assumption_ids:
            raise PlanCriticError("an unsat core must identify at least one record")
        if not isinstance(self.minimal, bool) or not isinstance(self.bounded, bool):
            raise PlanCriticError("unsat core flags must be booleans")
        material = self.to_dict(include_identity=False)
        computed = _digest("plan-unsat-core", material)
        if self.core_id and self.core_id != computed:
            raise PlanCriticError("unsat core identity mismatch")
        object.__setattr__(self, "core_id", computed)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(sorted({*self.constraint_ids, *self.record_ids, *self.assumption_ids}))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_UNSAT_CORE_SCHEMA,
            "kind": self.kind.value,
            "constraint_ids": list(self.constraint_ids),
            "record_ids": list(self.record_ids),
            "assumption_ids": list(self.assumption_ids),
            "evidence_ids": list(self.evidence_ids),
            "minimal": self.minimal,
            "bounded": self.bounded,
        }
        if include_identity:
            payload["core_id"] = self.core_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanUnsatCore":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "kind",
                "constraint_ids",
                "record_ids",
                "assumption_ids",
                "evidence_ids",
                "minimal",
                "bounded",
                "core_id",
            },
            "unsat core",
        )
        if payload.get("schema") not in (None, PLAN_UNSAT_CORE_SCHEMA):
            raise PlanCriticError("unsupported unsat core schema")
        return cls(
            kind=payload.get("kind", ""),
            constraint_ids=tuple(payload.get("constraint_ids") or ()),
            record_ids=tuple(payload.get("record_ids") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            minimal=payload.get("minimal", True),
            bounded=payload.get("bounded", True),
            core_id=payload.get("core_id", ""),
        )


MinimalUnsatCore = PlanUnsatCore
UnsatCore = PlanUnsatCore


@dataclass(frozen=True)
class TypedPlanCounterexample:
    kind: PlanDefectKind
    violated_property: str
    record_ids: tuple[str, ...]
    repairable_record_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    finite_bounds: Mapping[str, int] = field(default_factory=dict)
    witness: Mapping[str, Any] = field(default_factory=dict)
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PlanDefectKind(self.kind))
        object.__setattr__(
            self, "violated_property", _identifier(self.violated_property, "violated_property")
        )
        for name in (
            "record_ids",
            "repairable_record_ids",
            "assumption_ids",
            "evidence_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        bounds = {
            _identifier(str(key), "finite bound key"): _integer(
                value, f"finite_bounds.{key}", minimum=0
            )
            for key, value in sorted(dict(self.finite_bounds).items())
        }
        object.__setattr__(self, "finite_bounds", MappingProxyType(bounds))
        object.__setattr__(self, "witness", _freeze(_plain(dict(self.witness))))
        computed = _digest(
            "typed-plan-counterexample", self.to_dict(include_identity=False)
        )
        if self.counterexample_id and self.counterexample_id != computed:
            raise PlanCriticError("counterexample identity mismatch")
        object.__setattr__(self, "counterexample_id", computed)

    @property
    def semantic_id(self) -> str:
        return self.counterexample_id

    @property
    def counterexample_type(self) -> str:
        return self.kind.value

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_COUNTEREXAMPLE_SCHEMA,
            "kind": self.kind.value,
            "counterexample_type": self.kind.value,
            "violated_property": self.violated_property,
            "record_ids": list(self.record_ids),
            "repairable_record_ids": list(self.repairable_record_ids),
            "assumption_ids": list(self.assumption_ids),
            "evidence_ids": list(self.evidence_ids),
            "finite_bounds": dict(self.finite_bounds),
            "witness": _plain(self.witness),
        }
        if include_identity:
            payload["counterexample_id"] = self.counterexample_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedPlanCounterexample":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "kind",
                "counterexample_type",
                "violated_property",
                "record_ids",
                "repairable_record_ids",
                "assumption_ids",
                "evidence_ids",
                "finite_bounds",
                "witness",
                "counterexample_id",
            },
            "plan counterexample",
        )
        if payload.get("schema") not in (None, PLAN_COUNTEREXAMPLE_SCHEMA):
            raise PlanCriticError("unsupported plan counterexample schema")
        return cls(
            kind=payload.get("kind", payload.get("counterexample_type", "")),
            violated_property=payload.get("violated_property", ""),
            record_ids=tuple(payload.get("record_ids") or ()),
            repairable_record_ids=tuple(
                payload.get("repairable_record_ids") or ()
            ),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            finite_bounds=payload.get("finite_bounds") or {},
            witness=payload.get("witness") or {},
            counterexample_id=payload.get("counterexample_id", ""),
        )


PlanCounterexample = TypedPlanCounterexample


@dataclass(frozen=True)
class PlanCritiqueFinding:
    kind: PlanDefectKind
    severity: PlanCritiqueSeverity
    message: str
    record_ids: tuple[str, ...]
    repairable_record_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    unsat_core_id: str = ""
    counterexample_id: str = ""
    reason_code: str = ""
    finding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PlanDefectKind(self.kind))
        object.__setattr__(self, "severity", PlanCritiqueSeverity(self.severity))
        object.__setattr__(self, "message", _text(self.message, "message"))
        for name in ("record_ids", "repairable_record_ids", "evidence_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        for name in ("unsat_core_id", "counterexample_id", "reason_code"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, allow_empty=True),
            )
        if not set(self.repairable_record_ids) <= set(self.record_ids):
            raise PlanCriticError("repairable ids must be a subset of finding record ids")
        computed = _digest("plan-critique-finding", self.to_dict(include_identity=False))
        if self.finding_id and self.finding_id != computed:
            raise PlanCriticError("finding identity mismatch")
        object.__setattr__(self, "finding_id", computed)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_CRITIQUE_FINDING_SCHEMA,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "record_ids": list(self.record_ids),
            "repairable_record_ids": list(self.repairable_record_ids),
            "evidence_ids": list(self.evidence_ids),
            "unsat_core_id": self.unsat_core_id,
            "counterexample_id": self.counterexample_id,
            "reason_code": self.reason_code or self.kind.value,
        }
        if include_identity:
            payload["finding_id"] = self.finding_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCritiqueFinding":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "kind",
                "severity",
                "message",
                "record_ids",
                "repairable_record_ids",
                "evidence_ids",
                "unsat_core_id",
                "counterexample_id",
                "reason_code",
                "finding_id",
            },
            "critique finding",
        )
        if payload.get("schema") not in (None, PLAN_CRITIQUE_FINDING_SCHEMA):
            raise PlanCriticError("unsupported critique finding schema")
        return cls(
            kind=payload.get("kind", ""),
            severity=payload.get("severity", ""),
            message=payload.get("message", ""),
            record_ids=tuple(payload.get("record_ids") or ()),
            repairable_record_ids=tuple(
                payload.get("repairable_record_ids") or ()
            ),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            unsat_core_id=payload.get("unsat_core_id", ""),
            counterexample_id=payload.get("counterexample_id", ""),
            reason_code=payload.get("reason_code", ""),
            finding_id=payload.get("finding_id", ""),
        )


CritiqueFinding = PlanCritiqueFinding


@dataclass(frozen=True)
class PlanCritique:
    source_plan_id: str
    decision: PlanCritiqueDecision
    findings: tuple[PlanCritiqueFinding, ...]
    unsat_cores: tuple[PlanUnsatCore, ...]
    counterexamples: tuple[TypedPlanCounterexample, ...]
    checked_record_ids: tuple[str, ...]
    repairable_record_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    bounds: PlanCritiqueBounds
    truncated: bool = False
    interface: str = PLAN_CRITIQUE_INTERFACE
    critic_interface: str = PLAN_CRITIC_INTERFACE
    critique_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_plan_id", _identifier(self.source_plan_id, "source_plan_id")
        )
        object.__setattr__(self, "decision", PlanCritiqueDecision(self.decision))
        object.__setattr__(
            self,
            "findings",
            tuple(
                item if isinstance(item, PlanCritiqueFinding) else PlanCritiqueFinding.from_dict(item)
                for item in self.findings
            ),
        )
        object.__setattr__(
            self,
            "unsat_cores",
            tuple(
                item if isinstance(item, PlanUnsatCore) else PlanUnsatCore.from_dict(item)
                for item in self.unsat_cores
            ),
        )
        object.__setattr__(
            self,
            "counterexamples",
            tuple(
                item
                if isinstance(item, TypedPlanCounterexample)
                else TypedPlanCounterexample.from_dict(item)
                for item in self.counterexamples
            ),
        )
        for name in ("checked_record_ids", "repairable_record_ids", "evidence_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if not isinstance(self.bounds, PlanCritiqueBounds):
            object.__setattr__(self, "bounds", PlanCritiqueBounds.from_dict(self.bounds))
        if not isinstance(self.truncated, bool):
            raise PlanCriticError("truncated must be boolean")
        if self.interface != PLAN_CRITIQUE_INTERFACE:
            raise PlanCriticError("unsupported PlanCritique interface")
        if self.critic_interface != PLAN_CRITIC_INTERFACE:
            raise PlanCriticError("unsupported PlanCritic interface")
        if len(self.findings) > self.bounds.max_findings:
            raise PlanCritiqueBoundsError("critique exceeds max_findings")
        if len(self.counterexamples) > self.bounds.max_counterexamples:
            raise PlanCritiqueBoundsError("critique exceeds max_counterexamples")
        core_ids = {item.core_id for item in self.unsat_cores}
        counterexample_ids = {item.counterexample_id for item in self.counterexamples}
        if any(item.unsat_core_id and item.unsat_core_id not in core_ids for item in self.findings):
            raise PlanCriticError("finding references an unknown unsat core")
        if any(
            item.counterexample_id
            and item.counterexample_id not in counterexample_ids
            for item in self.findings
        ):
            raise PlanCriticError("finding references an unknown counterexample")
        expected_repairable = tuple(
            sorted(
                {
                    record_id
                    for item in self.findings
                    for record_id in item.repairable_record_ids
                }
            )
        )
        if self.repairable_record_ids != expected_repairable:
            raise PlanCriticError("repairable record projection is inconsistent")
        computed = content_identity(self.to_dict(include_identity=False))
        if self.critique_id and self.critique_id != computed:
            raise PlanCriticError("critique identity mismatch")
        object.__setattr__(self, "critique_id", computed)
        if len(self.to_json().encode("utf-8")) > self.bounds.max_output_bytes:
            raise PlanCritiqueBoundsError("serialized critique exceeds max_output_bytes")

    @property
    def plan_id(self) -> str:
        return self.source_plan_id

    @property
    def accepted(self) -> bool:
        return self.decision is PlanCritiqueDecision.ACCEPTED

    @property
    def admitted(self) -> bool:
        return self.accepted

    @property
    def ready(self) -> bool:
        return self.accepted

    @property
    def requires_repair(self) -> bool:
        return self.decision is PlanCritiqueDecision.REPAIR_REQUIRED

    @property
    def blocked(self) -> bool:
        return self.decision is PlanCritiqueDecision.REJECTED

    @property
    def candidate_id(self) -> str:
        return self.source_plan_id

    @property
    def finding_kinds(self) -> tuple[PlanDefectKind, ...]:
        return tuple(sorted({item.kind for item in self.findings}, key=lambda item: item.value))

    @property
    def issues(self) -> tuple[PlanCritiqueFinding, ...]:
        return self.findings

    @property
    def minimal_unsat_cores(self) -> tuple[PlanUnsatCore, ...]:
        return self.unsat_cores

    @property
    def failure_signature(self) -> str:
        return _digest(
            "plan-critique-failure",
            {
                "source_plan_id": self.source_plan_id,
                "findings": [
                    {
                        "kind": item.kind.value,
                        "record_ids": item.record_ids,
                        "evidence_ids": item.evidence_ids,
                    }
                    for item in self.findings
                    if item.severity is PlanCritiqueSeverity.ERROR
                ],
            },
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_CRITIQUE_SCHEMA,
            "version": PLAN_CRITIQUE_VERSION,
            "interface": self.interface,
            "critic_interface": self.critic_interface,
            "source_plan_id": self.source_plan_id,
            "plan_id": self.source_plan_id,
            "decision": self.decision.value,
            "accepted": self.accepted,
            "findings": [item.to_dict() for item in self.findings],
            "unsat_cores": [item.to_dict() for item in self.unsat_cores],
            "counterexamples": [item.to_dict() for item in self.counterexamples],
            "checked_record_ids": list(self.checked_record_ids),
            "repairable_record_ids": list(self.repairable_record_ids),
            "evidence_ids": list(self.evidence_ids),
            "bounds": self.bounds.to_dict(),
            "truncated": self.truncated,
            "authority": {
                "proof_authority": False,
                "completion_authority": False,
                "proposal_repair_only": True,
                "independently_recomputed": True,
            },
        }
        if include_identity:
            payload["critique_id"] = self.critique_id
            payload["failure_signature"] = self.failure_signature
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCritique":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "version",
                "interface",
                "critic_interface",
                "source_plan_id",
                "plan_id",
                "decision",
                "accepted",
                "findings",
                "unsat_cores",
                "counterexamples",
                "checked_record_ids",
                "repairable_record_ids",
                "evidence_ids",
                "bounds",
                "truncated",
                "authority",
                "critique_id",
                "failure_signature",
            },
            "plan critique",
        )
        if payload.get("schema") != PLAN_CRITIQUE_SCHEMA:
            raise PlanCriticError("unsupported plan critique schema")
        if payload.get("version") != PLAN_CRITIQUE_VERSION:
            raise PlanCriticError("unsupported plan critique version")
        result = cls(
            source_plan_id=payload.get("source_plan_id", payload.get("plan_id", "")),
            decision=payload.get("decision", ""),
            findings=tuple(payload.get("findings") or ()),
            unsat_cores=tuple(payload.get("unsat_cores") or ()),
            counterexamples=tuple(payload.get("counterexamples") or ()),
            checked_record_ids=tuple(payload.get("checked_record_ids") or ()),
            repairable_record_ids=tuple(
                payload.get("repairable_record_ids") or ()
            ),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            bounds=payload.get("bounds") or {},
            truncated=payload.get("truncated", False),
            interface=payload.get("interface", ""),
            critic_interface=payload.get("critic_interface", ""),
            critique_id=payload.get("critique_id", ""),
        )
        if payload.get("plan_id") not in (None, result.source_plan_id):
            raise PlanCriticError("plan identity projection is inconsistent")
        if payload.get("accepted") not in (None, result.accepted):
            raise PlanCriticError("accepted projection is inconsistent")
        if payload.get("failure_signature") not in (None, result.failure_signature):
            raise PlanCriticError("failure signature projection is inconsistent")
        expected_authority = result.to_dict(include_identity=False)["authority"]
        if payload.get("authority") not in (None, expected_authority):
            raise PlanCriticError("critique authority projection is inconsistent")
        return result

    @classmethod
    def from_json(cls, value: str) -> "PlanCritique":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise PlanCriticError("plan critique JSON must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class PlanCritiqueRequest:
    plan: Any
    obligation_graph: Any = None
    evidence: Any = None
    policy: Any = None
    ir: Any = None
    security: Any = None
    proof: Any = None
    parallel_plan: Any = None
    resources: Any = None
    consumers: Any = None
    expected_effects: Any = None


class _Collector:
    def __init__(
        self,
        bounds: PlanCritiqueBounds,
        repairable: set[str],
        immutable: set[str],
    ) -> None:
        self.bounds = bounds
        self.repairable = repairable
        self.immutable = immutable
        self.findings: list[PlanCritiqueFinding] = []
        self.cores: list[PlanUnsatCore] = []
        self.counterexamples: list[TypedPlanCounterexample] = []
        self.truncated = False
        self._keys: set[tuple[Any, ...]] = set()

    def add(
        self,
        kind: PlanDefectKind,
        message: str,
        records: Iterable[str],
        *,
        evidence: Iterable[str] = (),
        assumptions: Iterable[str] = (),
        constraints: Iterable[str] = (),
        witness: Mapping[str, Any] | None = None,
        severity: PlanCritiqueSeverity = PlanCritiqueSeverity.ERROR,
        core: bool = False,
        reason_code: str = "",
    ) -> None:
        record_ids = tuple(sorted({str(item) for item in records if str(item)}))
        evidence_ids = tuple(sorted({str(item) for item in evidence if str(item)}))
        assumption_ids = tuple(sorted({str(item) for item in assumptions if str(item)}))
        constraint_ids = tuple(sorted({str(item) for item in constraints if str(item)}))
        key = (kind.value, record_ids, evidence_ids, assumption_ids, constraint_ids)
        if key in self._keys:
            return
        self._keys.add(key)
        if len(self.findings) >= self.bounds.max_findings:
            self.truncated = True
            return
        core_record = None
        if core and (record_ids or assumption_ids or constraint_ids):
            core_record = PlanUnsatCore(
                kind=kind,
                constraint_ids=constraint_ids[: self.bounds.max_core_items],
                record_ids=record_ids[: self.bounds.max_core_items],
                assumption_ids=assumption_ids[: self.bounds.max_core_items],
                evidence_ids=evidence_ids[: self.bounds.max_core_items],
                minimal=True,
                bounded=True,
            )
            self.cores.append(core_record)
        repairable_ids = tuple(
            item
            for item in record_ids
            if item in self.repairable and item not in self.immutable
        )
        counterexample = None
        if severity is PlanCritiqueSeverity.ERROR and (
            len(self.counterexamples) < self.bounds.max_counterexamples
        ):
            compact_witness = dict(list(sorted((witness or {}).items()))[: self.bounds.max_witness_items])
            counterexample = TypedPlanCounterexample(
                kind=kind,
                violated_property=kind.value,
                record_ids=record_ids,
                repairable_record_ids=repairable_ids,
                assumption_ids=assumption_ids,
                evidence_ids=evidence_ids,
                finite_bounds={
                    "max_core_items": self.bounds.max_core_items,
                    "max_witness_items": self.bounds.max_witness_items,
                },
                witness=compact_witness,
            )
            self.counterexamples.append(counterexample)
        self.findings.append(
            PlanCritiqueFinding(
                kind=kind,
                severity=severity,
                message=message,
                record_ids=record_ids,
                repairable_record_ids=repairable_ids,
                evidence_ids=evidence_ids,
                unsat_core_id=core_record.core_id if core_record else "",
                counterexample_id=(
                    counterexample.counterexample_id if counterexample else ""
                ),
                reason_code=reason_code or kind.value,
            )
        )


def _shortest_cycle(
    nodes: Iterable[str], edges: Mapping[str, set[str]], maximum: int
) -> tuple[str, ...]:
    best: tuple[str, ...] = ()
    for start in sorted(set(nodes)):
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(start, (start,))])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) > maximum:
                continue
            for child in sorted(edges.get(node, ())):
                if child == start:
                    candidate = path
                    if not best or (len(candidate), candidate) < (len(best), best):
                        best = candidate
                    queue.clear()
                    break
                if child not in seen:
                    seen.add(child)
                    queue.append((child, (*path, child)))
    return best


def _topological_width(nodes: set[str], edges: Mapping[str, set[str]]) -> int:
    if not nodes:
        return 0
    remaining = set(nodes)
    completed: set[str] = set()
    maximum = 0
    while remaining:
        ready = sorted(
            node for node in remaining if edges.get(node, set()) <= completed
        )
        if not ready:
            return 0
        maximum = max(maximum, len(ready))
        completed.update(ready)
        remaining.difference_update(ready)
    return maximum


def _status_failure(record: Mapping[str, Any]) -> bool:
    status = str(
        _field(record, "status", "decision", "outcome", "disposition", default="")
    ).strip().casefold()
    if status in _FAIL_STATUSES:
        return True
    checks = (
        ("admitted", True),
        ("allowed", True),
        ("current", True),
        ("feasible", True),
        ("passed", True),
        ("safe", True),
        ("satisfied", True),
        ("valid", True),
        ("verified", True),
    )
    return any(name in record and record[name] is not expected for name, expected in checks)


def _evidence_ids(value: Any) -> tuple[str, ...]:
    result: set[str] = set()
    for record in _record_values(value):
        record_id = _record_id(record)
        if record_id:
            result.add(record_id)
        result.update(
            _ids(
                _field(record, "evidence_ids", "receipt_ids", "source_ids", default=()),
                "evidence id",
            )
        )
    return tuple(sorted(result))


class PlanCritic:
    """Independently derive deterministic defects and repair coordinates."""

    interface = PLAN_CRITIC_INTERFACE

    def __init__(self, *, bounds: PlanCritiqueBounds | Mapping[str, Any] | None = None) -> None:
        if bounds is None:
            bounds = PlanCritiqueBounds()
        elif isinstance(bounds, Mapping):
            bounds = PlanCritiqueBounds.from_dict(bounds)
        if not isinstance(bounds, PlanCritiqueBounds):
            raise PlanCriticError("bounds must be PlanCritiqueBounds or a mapping")
        self.bounds = bounds

    def critique(
        self,
        plan: Any = None,
        *,
        candidate: Any = None,
        obligation_graph: Any = None,
        evidence: Any = None,
        evidence_bundle: Any = None,
        policy: Any = None,
        intent_ir: Any = None,
        ir: Any = None,
        security_ir: Any = None,
        security: Any = None,
        proof: Any = None,
        proof_receipts: Any = None,
        parallel_plan: Any = None,
        resource_snapshot: Any = None,
        resources: Any = None,
        consumers: Any = None,
        expected_effects: Any = None,
        required_goal_ids: Iterable[str] = (),
        required_consumer_ids: Iterable[str] = (),
        required_assumption_ids: Iterable[str] = (),
        **context: Any,
    ) -> PlanCritique:
        if isinstance(plan, PlanCritiqueRequest):
            request = plan
            plan = request.plan
            obligation_graph = obligation_graph or request.obligation_graph
            evidence = evidence or request.evidence
            policy = policy or request.policy
            ir = ir or request.ir
            security = security or request.security
            proof = proof or request.proof
            parallel_plan = parallel_plan or request.parallel_plan
            resources = resources or request.resources
            consumers = consumers or request.consumers
            expected_effects = expected_effects or request.expected_effects
        plan = plan if plan is not None else candidate
        if plan is None:
            raise PlanCriticError("plan or candidate is required")
        source = _mapping(plan, "plan")
        if "plan" in source and not any(
            key in source for key in ("tasks", "steps", "branch", "schema")
        ):
            envelope = source
            source = _mapping(envelope["plan"], "plan")
            obligation_graph = obligation_graph or envelope.get("obligation_graph")
            evidence = evidence or envelope.get("evidence")
            policy = policy or envelope.get("policy")
            ir = ir or envelope.get("ir")
            security = security or envelope.get("security")
            proof = proof or envelope.get("proof")
            parallel_plan = parallel_plan or envelope.get("parallel_plan")
            resources = resources or envelope.get("resources")
            consumers = consumers or envelope.get("consumers")

        evidence = evidence if evidence is not None else evidence_bundle
        ir = ir if ir is not None else intent_ir
        security = security if security is not None else security_ir
        proof = proof if proof is not None else proof_receipts
        resources = resources if resources is not None else resource_snapshot

        tasks = self._tasks(source)
        task_by_id: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        immutable: set[str] = set()
        repairable: set[str] = set()
        checked: set[str] = set()
        for index, task in enumerate(tasks):
            task_id = _record_id(task, f"task:{index}")
            if task_id in task_by_id:
                duplicates.add(task_id)
            task_by_id[task_id] = task
            checked.add(task_id)
            state = str(_field(task, "lifecycle_state", "state", "status", default="")).casefold()
            if state in _IMMUTABLE_STATES:
                immutable.add(task_id)
            else:
                repairable.add(task_id)
        source_plan_id = self._source_plan_id(source)
        checked.add(source_plan_id)
        source_state = str(
            _field(source, "lifecycle_state", "state", "status", default="")
        ).casefold()
        (immutable if source_state in _IMMUTABLE_STATES else repairable).add(
            source_plan_id
        )
        for collection in ("goals", "constraints", "assumptions", "effects"):
            for index, record in enumerate(_record_values(source.get(collection))):
                record_id = _record_id(record, f"{collection[:-1]}:{index}")
                checked.add(record_id)
                state = str(_field(record, "lifecycle_state", "state", "status", default="")).casefold()
                (immutable if state in _IMMUTABLE_STATES else repairable).add(record_id)

        collector = _Collector(self.bounds, repairable, immutable)
        if len(checked) > self.bounds.max_records:
            collector.add(
                PlanDefectKind.BOUND_EXCEEDED,
                "plan record population exceeds the deterministic critique bound",
                tuple(sorted(checked))[: self.bounds.max_core_items],
                witness={"record_count": len(checked), "max_records": self.bounds.max_records},
                core=True,
            )
            collector.truncated = True

        self._check_identity(source, source_plan_id, collector)
        self._check_obligation_graph(obligation_graph, collector)
        if not source.get("schema"):
            for record_id, record in task_by_id.items():
                self._check_record_identity(record, record_id, collector)
            for collection in ("goals", "constraints", "assumptions", "effects"):
                for index, record in enumerate(_record_values(source.get(collection))):
                    self._check_record_identity(
                        record,
                        _record_id(record, f"{collection[:-1]}:{index}"),
                        collector,
                    )
        if duplicates:
            for item in sorted(duplicates):
                collector.add(
                    PlanDefectKind.DUPLICATE_IDENTITY,
                    f"record identity {item!r} occurs more than once",
                    (item,),
                    core=True,
                )
        edges = self._check_graph(task_by_id, collector)
        self._check_coverage(
            source,
            task_by_id,
            obligation_graph,
            set(required_goal_ids),
            collector,
        )
        self._check_assumptions(
            source,
            evidence,
            set(required_assumption_ids),
            collector,
        )
        self._check_contradictions(source, task_by_id, collector)
        self._check_consumers(
            source,
            consumers,
            set(required_consumer_ids),
            collector,
        )
        self._check_effects(source, expected_effects, task_by_id, collector)
        self._check_gate("policy", PlanDefectKind.POLICY_FAILURE, policy or source.get("policy_receipts"), collector)
        self._check_gate("ir", PlanDefectKind.IR_FAILURE, ir or source.get("ir_receipts"), collector)
        self._check_gate("security", PlanDefectKind.SECURITY_FAILURE, security or source.get("security_receipts"), collector)
        self._check_gate("proof", PlanDefectKind.PROOF_FAILURE, proof or source.get("proof_receipts"), collector)
        self._check_required_gate_bindings(
            source,
            policy=policy,
            ir=ir,
            security=security,
            proof=proof,
            collector=collector,
        )
        self._check_candidate_claims(source, evidence, collector)
        self._check_parallelism(
            source,
            parallel_plan,
            task_by_id,
            edges,
            collector,
        )
        self._check_resources(source, parallel_plan, resources, task_by_id, collector)

        if collector.truncated and not any(
            item.kind is PlanDefectKind.BOUND_EXCEEDED for item in collector.findings
        ) and len(collector.findings) < self.bounds.max_findings:
            collector.add(
                PlanDefectKind.BOUND_EXCEEDED,
                "additional deterministic findings were omitted at the output bound",
                (source_plan_id,),
                witness={"max_findings": self.bounds.max_findings},
            )
        findings = tuple(
            sorted(
                collector.findings,
                key=lambda item: (
                    item.severity.value,
                    item.kind.value,
                    item.record_ids,
                    item.finding_id,
                ),
            )
        )
        errors = [item for item in findings if item.severity is PlanCritiqueSeverity.ERROR]
        if not errors:
            decision = (
                PlanCritiqueDecision.REVIEW_REQUIRED
                if any(item.severity is PlanCritiqueSeverity.REVIEW for item in findings)
                else PlanCritiqueDecision.ACCEPTED
            )
        elif any(item.repairable_record_ids for item in errors):
            decision = PlanCritiqueDecision.REPAIR_REQUIRED
        else:
            decision = PlanCritiqueDecision.REJECTED
        evidence_population = set(_evidence_ids(evidence))
        for value in (policy, ir, security, proof, parallel_plan, resources):
            evidence_population.update(_evidence_ids(value))
        return PlanCritique(
            source_plan_id=source_plan_id,
            decision=decision,
            findings=findings,
            unsat_cores=tuple(sorted(collector.cores, key=lambda item: item.core_id)),
            counterexamples=tuple(
                sorted(
                    collector.counterexamples,
                    key=lambda item: item.counterexample_id,
                )
            ),
            checked_record_ids=tuple(sorted(checked | {source_plan_id})),
            repairable_record_ids=tuple(
                sorted(
                    {
                        record_id
                        for item in findings
                        for record_id in item.repairable_record_ids
                    }
                )
            ),
            evidence_ids=tuple(sorted(evidence_population)),
            bounds=self.bounds,
            truncated=collector.truncated,
        )

    analyze = critique
    evaluate = critique
    criticize = critique

    def _source_plan_id(self, source: Mapping[str, Any]) -> str:
        for name in ("plan_id", "candidate_id", "portfolio_id", "revision_id", "content_id"):
            value = source.get(name)
            if value not in (None, ""):
                return str(value)
        material = {key: value for key, value in source.items() if key not in _IDENTITY_KEYS}
        return content_identity(_plain(material))

    def _check_identity(
        self,
        source: Mapping[str, Any],
        source_plan_id: str,
        collector: _Collector,
    ) -> None:
        schema = str(source.get("schema") or "")
        parsers: dict[str, tuple[str, str]] = {
            "ipfs_accelerate_py/agent-supervisor/formal-work-plan@1": (
                "ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts",
                "FormalWorkPlan",
            ),
            "ipfs_accelerate_py/agent-supervisor/obligation-graph@1": (
                "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler",
                "ObligationGraph",
            ),
            "ipfs_accelerate_py/agent-supervisor/symbolic-candidate@1": (
                "ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner",
                "SymbolicCandidateRecord",
            ),
            "ipfs_accelerate_py/agent-supervisor/symbolic-candidate-portfolio@1": (
                "ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner",
                "SymbolicCandidatePortfolio",
            ),
            "ipfs_accelerate_py/agent-supervisor/plan-revision@1": (
                "ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts",
                "PlanRevision",
            ),
        }
        if schema in parsers:
            module_name, class_name = parsers[schema]
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name).from_dict(source)
            except Exception as exc:
                message = str(exc)
                lowered = message.casefold()
                kind = PlanDefectKind.SCHEMA_FAILURE
                if "identity" in lowered or "content id" in lowered:
                    kind = PlanDefectKind.IDENTITY_MISMATCH
                elif "acyclic" in lowered or "cycle" in lowered:
                    kind = PlanDefectKind.DEPENDENCY_CYCLE
                elif "unknown" in lowered or "unbound" in lowered:
                    kind = PlanDefectKind.ORPHAN_RECORD
                elif "conflict" in lowered:
                    kind = PlanDefectKind.CONTRADICTION
                collector.add(
                    kind,
                    f"canonical schema replay failed: {message[:1024]}",
                    (source_plan_id,),
                    core=kind
                    in {
                        PlanDefectKind.DEPENDENCY_CYCLE,
                        PlanDefectKind.CONTRADICTION,
                    },
                )
        elif schema and not schema.startswith("ipfs_accelerate_py/agent-supervisor/"):
            collector.add(
                PlanDefectKind.SCHEMA_FAILURE,
                f"plan uses unsupported schema {schema!r}",
                (source_plan_id,),
            )
        claimed = str(source.get("content_id") or source.get("identity") or "")
        if claimed and _DIGEST_RE.fullmatch(claimed):
            material = {
                key: value
                for key, value in source.items()
                if key not in _IDENTITY_KEYS
            }
            recomputed = content_identity(_plain(material))
            if claimed != recomputed and schema not in parsers:
                collector.add(
                    PlanDefectKind.IDENTITY_MISMATCH,
                    "claimed plan identity does not match canonical plan content",
                    (source_plan_id,),
                    witness={"claimed": claimed, "recomputed": recomputed},
                )

    def _check_record_identity(
        self,
        record: Mapping[str, Any],
        record_id: str,
        collector: _Collector,
    ) -> None:
        claimed = str(
            _field(
                record,
                "content_id",
                "identity",
                "record_id",
                "canonical_id",
                default="",
            )
        )
        if not claimed or not _DIGEST_RE.fullmatch(claimed):
            return
        material = {
            key: value for key, value in record.items() if key not in _IDENTITY_KEYS
        }
        expected = {
            content_identity(_plain(material)),
            _canonical_sha256(material),
        }
        if claimed not in expected:
            collector.add(
                PlanDefectKind.IDENTITY_MISMATCH,
                f"record {record_id!r} identity does not match canonical content",
                (record_id,),
                witness={"claimed": claimed, "recomputed": sorted(expected)},
            )

    def _check_obligation_graph(
        self,
        obligation_graph: Any,
        collector: _Collector,
    ) -> None:
        """Replay graph identity and independently inspect primitive edges."""

        if obligation_graph is None:
            return
        graph = _mapping(obligation_graph, "obligation_graph")
        graph_id = str(
            graph.get("graph_id")
            or graph.get("content_id")
            or self._source_plan_id(graph)
        )
        self._check_identity(graph, graph_id, collector)

        nodes = {
            _record_id(record, f"obligation:{index}"): record
            for index, record in enumerate(_record_values(graph.get("nodes")))
        }
        node_ids = set(nodes)
        edges: dict[str, set[str]] = {item: set() for item in node_ids}
        for index, refinement in enumerate(
            _record_values(graph.get("refinements"))
        ):
            refinement_id = _record_id(
                refinement, f"refinement:{index}"
            )
            parent = str(refinement.get("parent_obligation_id") or "")
            children = set(
                _ids(
                    refinement.get("child_obligation_ids"),
                    "child obligation id",
                )
            )
            missing = ({parent} | children) - node_ids
            for orphan in sorted(item for item in missing if item):
                collector.add(
                    PlanDefectKind.ORPHAN_RECORD,
                    f"refinement {refinement_id!r} references unknown "
                    f"obligation {orphan!r}",
                    (refinement_id, orphan),
                    core=True,
                )
            if parent in node_ids:
                edges[parent].update(children & node_ids)

        cycle = _shortest_cycle(
            node_ids,
            edges,
            self.bounds.max_core_items,
        )
        if cycle:
            collector.add(
                PlanDefectKind.DEPENDENCY_CYCLE,
                "obligation refinement graph contains a cycle",
                cycle,
                constraints=tuple(
                    f"refinement:{cycle[index - 1]}->{cycle[index]}"
                    for index in range(len(cycle))
                ),
                witness={"cycle": list(cycle)},
                core=True,
            )

        for obligation_id, node in nodes.items():
            status = str(node.get("status") or "").casefold()
            if status == "contradicted":
                collector.add(
                    PlanDefectKind.CONTRADICTION,
                    f"obligation {obligation_id!r} is contradicted",
                    (obligation_id,),
                    assumptions=_ids(
                        node.get("assumption_refs"),
                        "assumption id",
                    ),
                    witness={"status": status},
                    core=True,
                )

        candidate_ids = {
            _record_id(record, f"candidate:{index}")
            for index, record in enumerate(
                _record_values(graph.get("task_candidates"))
            )
        }
        for index, candidate in enumerate(
            _record_values(graph.get("task_candidates"))
        ):
            candidate_id = _record_id(candidate, f"candidate:{index}")
            dependencies = set(
                _ids(
                    candidate.get("depends_on_candidate_ids"),
                    "candidate dependency id",
                )
            )
            closures = set(
                _ids(
                    candidate.get("closes_obligation_ids"),
                    "closed obligation id",
                )
            )
            for orphan in sorted(dependencies - candidate_ids):
                collector.add(
                    PlanDefectKind.ORPHAN_RECORD,
                    f"task candidate {candidate_id!r} depends on unknown "
                    f"candidate {orphan!r}",
                    (candidate_id, orphan),
                    core=True,
                )
            for orphan in sorted(closures - node_ids):
                collector.add(
                    PlanDefectKind.ORPHAN_RECORD,
                    f"task candidate {candidate_id!r} closes unknown "
                    f"obligation {orphan!r}",
                    (candidate_id, orphan),
                    core=True,
                )

        for index, assumption in enumerate(
            _record_values(graph.get("assumptions"))
        ):
            assumption_id = _record_id(
                assumption, f"assumption:{index}"
            )
            status = str(assumption.get("status") or "").casefold()
            if status in {"invalid", "unknown", "stale", "unverified"}:
                collector.add(
                    PlanDefectKind.UNSATISFIED_ASSUMPTION,
                    f"graph assumption {assumption_id!r} is {status}",
                    (assumption_id,),
                    assumptions=(assumption_id,),
                    witness={"status": status},
                    core=True,
                )

    def _tasks(self, source: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        direct = _field(source, "tasks", "steps", "records")
        if direct is not None:
            return _record_values(direct)
        symbolic = _selected_symbolic(source)
        if symbolic:
            schedule = symbolic.get("schedule")
            if isinstance(schedule, Mapping):
                ids = {
                    item
                    for wave in _sequence(schedule.get("waves"))
                    for item in _sequence(wave)
                }
                dependencies: dict[str, list[str]] = defaultdict(list)
                for edge in _sequence(schedule.get("dependency_edges")):
                    pair = _sequence(edge)
                    if len(pair) == 2:
                        dependencies[str(pair[1])].append(str(pair[0]))
                return tuple(
                    {
                        "task_id": str(item),
                        "depends_on": dependencies.get(str(item), ()),
                    }
                    for item in sorted(ids, key=str)
                )
            task_ids = symbolic.get("task_candidate_ids")
            if task_ids:
                return tuple(
                    {"task_id": str(item)} for item in _sequence(task_ids)
                )
        schedule = source.get("schedule")
        if isinstance(schedule, Mapping):
            dependencies: dict[str, list[str]] = defaultdict(list)
            for edge in _sequence(schedule.get("dependency_edges")):
                pair = _sequence(edge)
                if len(pair) == 2:
                    dependencies[str(pair[1])].append(str(pair[0]))
            ids = {
                str(item)
                for wave in _sequence(schedule.get("waves"))
                for item in _sequence(wave)
            }
            return tuple(
                {"task_id": item, "depends_on": dependencies.get(item, [])}
                for item in sorted(ids)
            )
        return ()

    def _check_graph(
        self,
        tasks: Mapping[str, Mapping[str, Any]],
        collector: _Collector,
    ) -> dict[str, set[str]]:
        task_ids = set(tasks)
        edges: dict[str, set[str]] = {}
        for task_id, task in tasks.items():
            dependencies = set(
                _ids(
                    _field(
                        task,
                        "depends_on",
                        "dependency_ids",
                        "dependencies",
                        "predecessor_ids",
                        "requires",
                        default=(),
                    ),
                    "dependency id",
                )
            )
            edges[task_id] = dependencies & task_ids
            for missing in sorted(dependencies - task_ids):
                collector.add(
                    PlanDefectKind.ORPHAN_RECORD,
                    f"task {task_id!r} depends on unknown record {missing!r}",
                    (task_id, missing),
                    core=True,
                )
        cycle = _shortest_cycle(task_ids, edges, self.bounds.max_core_items)
        if cycle:
            collector.add(
                PlanDefectKind.DEPENDENCY_CYCLE,
                "task dependency graph contains a cycle",
                cycle,
                constraints=tuple(
                    f"dependency:{cycle[index - 1]}->{cycle[index]}"
                    for index in range(len(cycle))
                ),
                witness={"cycle": list(cycle)},
                core=True,
            )
        return edges

    def _check_coverage(
        self,
        source: Mapping[str, Any],
        tasks: Mapping[str, Mapping[str, Any]],
        obligation_graph: Any,
        required: set[str],
        collector: _Collector,
    ) -> None:
        goals = _record_values(source.get("goals"))
        declared_goals = {
            _record_id(goal, f"goal:{index}") for index, goal in enumerate(goals)
        }
        required.update(
            declared_goals
        )
        required.update(
            _ids(
                _field(
                    source,
                    "required_goal_ids",
                    "goal_ids",
                    "root_obligation_ids",
                    default=(),
                ),
                "goal id",
            )
        )
        covered = set(
            _ids(
                _field(
                    source,
                    "covered_goal_ids",
                    "covered_obligation_ids",
                    default=(),
                ),
                "covered goal id",
            )
        )
        symbolic = _selected_symbolic(source)
        claimed_symbolic_coverage = set(
            _ids(
                symbolic.get("covered_obligation_ids"),
                "covered obligation id",
            )
        )
        covered.update(claimed_symbolic_coverage)
        for task in tasks.values():
            task_id = _record_id(task)
            task_goals = set(
                _ids(
                    _field(
                        task,
                        "goal_id",
                        "goal_ids",
                        "closes_goal_ids",
                        "closes_obligation_ids",
                        default=(),
                    ),
                    "covered goal id",
                )
            )
            covered.update(task_goals)
            if declared_goals:
                for unknown in sorted(task_goals - declared_goals):
                    collector.add(
                        PlanDefectKind.ORPHAN_RECORD,
                        f"task {task_id!r} references unknown goal {unknown!r}",
                        (task_id, unknown),
                        core=True,
                    )
                if not task_goals:
                    collector.add(
                        PlanDefectKind.ORPHAN_RECORD,
                        f"task {task_id!r} is not attached to a declared goal",
                        (task_id,),
                        core=True,
                    )
        graph = _mapping(obligation_graph) if obligation_graph is not None else {}
        graph_roots = set(
            _ids(graph.get("root_obligation_ids"), "root obligation id")
        )
        required.update(graph_roots)
        if graph:
            graph_candidates = {
                _record_id(item): item
                for item in _record_values(graph.get("task_candidates"))
            }
            selected_task_ids = set(tasks)
            independently_covered = {
                obligation_id
                for task_id in selected_task_ids
                for obligation_id in _ids(
                    graph_candidates.get(task_id, {}).get("closes_obligation_ids"),
                    "covered obligation id",
                )
            }
            for unknown in sorted(
                selected_task_ids - set(graph_candidates)
                if graph_candidates
                else ()
            ):
                collector.add(
                    PlanDefectKind.ORPHAN_RECORD,
                    f"selected task {unknown!r} is absent from the obligation graph",
                    (unknown,),
                    core=True,
                )
            if graph_candidates:
                for false_claim in sorted(
                    claimed_symbolic_coverage - independently_covered
                ):
                    collector.add(
                        PlanDefectKind.UNCOVERED_GOAL,
                        f"candidate claims unproduced obligation {false_claim!r}",
                        (false_claim,),
                        witness={
                            "claimed": false_claim,
                            "independently_covered": sorted(
                                independently_covered
                            ),
                        },
                        core=True,
                    )
                covered.difference_update(claimed_symbolic_coverage)
                covered.update(independently_covered)

            nodes = {
                _record_id(item): item
                for item in _record_values(graph.get("nodes"))
            }
            refinements = {
                _record_id(item): item
                for item in _record_values(graph.get("refinements"))
            }
            by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for refinement in refinements.values():
                by_parent[
                    str(refinement.get("parent_obligation_id") or "")
                ].append(refinement)

            def satisfied(obligation_id: str, visiting: frozenset[str]) -> bool:
                if obligation_id in visiting:
                    return False
                node = nodes.get(obligation_id, {})
                if str(node.get("status") or "").casefold() == "discharged":
                    return True
                next_visiting = visiting | {obligation_id}
                records = by_parent.get(obligation_id, ())
                for refinement in records:
                    children = _ids(
                        refinement.get("child_obligation_ids"),
                        "child obligation id",
                    )
                    kind = str(refinement.get("kind") or "").casefold()
                    if kind == "or" and any(
                        satisfied(child, next_visiting) for child in children
                    ):
                        return True
                    if kind == "and" and all(
                        satisfied(child, next_visiting) for child in children
                    ):
                        return obligation_id in independently_covered
                return obligation_id in independently_covered

            covered.update(
                root
                for root in graph_roots
                if satisfied(root, frozenset())
            )
        for item in sorted(required - covered):
            collector.add(
                PlanDefectKind.UNCOVERED_GOAL,
                f"required goal or obligation {item!r} has no selected producer",
                (item,),
                witness={"required": item, "covered": sorted(covered)},
                core=True,
            )

    def _check_assumptions(
        self,
        source: Mapping[str, Any],
        evidence: Any,
        required: set[str],
        collector: _Collector,
    ) -> None:
        assumption_records = _record_values(source.get("assumptions"))
        required.update(
            _record_id(item, f"assumption:{index}")
            for index, item in enumerate(assumption_records)
        )
        for task in _record_values(_field(source, "tasks", "steps", default=())):
            required.update(
                _ids(
                    _field(
                        task,
                        "required_consumer_ids",
                        "consumer_ids",
                        "consumer_refs",
                        "downstream_task_ids",
                        "used_by",
                        "required_by",
                        default=(),
                    ),
                    "consumer id",
                )
            )
        for effect in _record_values(
            _field(source, "effects", "actual_effects", default=())
        ):
            required.update(
                _ids(
                    _field(
                        effect,
                        "required_consumer_ids",
                        "consumer_ids",
                        "consumer_refs",
                        default=(),
                    ),
                    "consumer id",
                )
            )
        required.update(_ids(source.get("required_assumption_ids"), "assumption id"))
        candidate = source.get("plan") if isinstance(source.get("plan"), Mapping) else source
        required.update(_ids(candidate.get("assumptions"), "assumption id"))
        satisfied = set(
            _ids(
                _field(
                    _mapping(evidence) if evidence is not None else {},
                    "satisfied_assumption_ids",
                    "validated_assumptions",
                    "proved_assumption_ids",
                    default=(),
                ),
                "satisfied assumption id",
            )
        )
        if not satisfied:
            satisfied.update(
                _ids(candidate.get("validated_assumptions"), "validated assumption")
            )
            satisfied.update(
                _record_id(item)
                for item in assumption_records
                if str(item.get("status") or "").casefold()
                in {"active", "satisfied", "valid", "verified"}
            )
        for item in sorted(required - satisfied):
            collector.add(
                PlanDefectKind.UNSATISFIED_ASSUMPTION,
                f"plan assumption {item!r} has no current satisfying evidence",
                (item,),
                assumptions=(item,),
                core=True,
            )

    def _check_contradictions(
        self,
        source: Mapping[str, Any],
        tasks: Mapping[str, Mapping[str, Any]],
        collector: _Collector,
    ) -> None:
        values: dict[tuple[str, str], list[tuple[str, Any]]] = defaultdict(list)
        records = list(_record_values(source.get("constraints")))
        records.extend(_record_values(source.get("facts")))
        for index, record in enumerate(records):
            record_id = _record_id(record, f"constraint:{index}")
            predicate = str(
                _field(record, "predicate_id", "predicate", "property", "name", default="")
            )
            scope = str(_field(record, "scope_id", "subject_id", "target_id", default=""))
            value = _field(record, "truth", "value", "polarity", "expected", default=None)
            if predicate and value is not None:
                values[(scope, predicate)].append((record_id, _plain(value)))
        for key, entries in sorted(values.items()):
            by_value: dict[str, tuple[str, Any]] = {}
            for record_id, value in entries:
                by_value.setdefault(json.dumps(value, sort_keys=True), (record_id, value))
            if len(by_value) > 1:
                minimal = tuple(item[0] for item in list(by_value.values())[:2])
                collector.add(
                    PlanDefectKind.CONTRADICTION,
                    f"constraints assign incompatible values to {key[1]!r}",
                    minimal,
                    constraints=minimal,
                    witness={"scope": key[0], "predicate": key[1], "values": [item[1] for item in list(by_value.values())[:2]]},
                    core=True,
                )
        for task_id, task in tasks.items():
            for conflict in _ids(
                _field(task, "unresolved_conflicts", "conflicts", default=()),
                "conflict id",
            ):
                collector.add(
                    PlanDefectKind.CONFLICT_FAILURE,
                    f"task {task_id!r} retains unresolved conflict {conflict!r}",
                    (task_id, conflict),
                    core=True,
                )

    def _check_consumers(
        self,
        source: Mapping[str, Any],
        consumers: Any,
        required: set[str],
        collector: _Collector,
    ) -> None:
        required.update(
            _ids(
                _field(
                    source,
                    "required_consumer_ids",
                    "expected_consumer_ids",
                    "consumer_ids",
                    default=(),
                ),
                "consumer id",
            )
        )
        consumer_source = _mapping(consumers) if consumers is not None else source
        resolved = set(
            _ids(
                _field(
                    consumer_source,
                    "resolved_consumer_ids",
                    "covered_consumer_ids",
                    "consumer_dispositions",
                    default=(),
                ),
                "resolved consumer id",
            )
        )
        for record in _record_values(consumer_source.get("consumers")):
            if str(record.get("status") or "resolved").casefold() in {
                "resolved",
                "retained",
                "updated",
                "unaffected",
            }:
                resolved.add(_record_id(record))
        for item in sorted(required - resolved):
            collector.add(
                PlanDefectKind.MISSING_CONSUMER,
                f"mandatory consumer {item!r} has no disposition",
                (item,),
                witness={"required_consumer": item},
                core=True,
            )

    def _check_effects(
        self,
        source: Mapping[str, Any],
        expected_effects: Any,
        tasks: Mapping[str, Mapping[str, Any]],
        collector: _Collector,
    ) -> None:
        expected = set(
            _ids(
                expected_effects
                if expected_effects is not None
                else _field(source, "expected_effects", "expected_effect_ids", default=()),
                "expected effect id",
            )
        )
        actual_records = list(_record_values(_field(source, "effects", "actual_effects", default=())))
        actual = {
            _record_id(record, f"effect:{index}")
            for index, record in enumerate(actual_records)
        }
        for task_id, task in tasks.items():
            expected.update(_ids(task.get("expected_effects"), "expected effect id"))
            actual.update(_ids(_field(task, "effect_ids", "effects", default=()), "effect id"))
            if task.get("valid_effects") is False:
                collector.add(
                    PlanDefectKind.INVALID_EFFECT,
                    f"task {task_id!r} declares invalid effects",
                    (task_id,),
                )
        for item in sorted(expected - actual):
            collector.add(
                PlanDefectKind.INVALID_EFFECT,
                f"expected effect {item!r} is missing",
                (item,),
                witness={"expected": item, "actual_effect_ids": sorted(actual)},
                core=True,
            )
        for item in sorted(actual - expected) if expected else ():
            collector.add(
                PlanDefectKind.INVALID_EFFECT,
                f"undeclared effect {item!r} is present",
                (item,),
                witness={"unexpected": item},
                core=True,
            )
        task_ids = set(tasks)
        assignments: dict[tuple[str, str], tuple[str, Any]] = {}
        for index, effect in enumerate(actual_records):
            effect_id = _record_id(effect, f"effect:{index}")
            owner = str(_field(effect, "task_id", "producer_id", default=""))
            target = str(_field(effect, "target_id", "fluent_id", "path", default=""))
            operation = str(effect.get("operation") or "").casefold()
            if owner and owner not in task_ids:
                collector.add(
                    PlanDefectKind.INVALID_EFFECT,
                    f"effect {effect_id!r} references unknown producer {owner!r}",
                    (effect_id, owner),
                    core=True,
                )
            if operation and operation not in {
                "assign",
                "create",
                "delete",
                "emit",
                "initiate",
                "terminate",
                "update",
            }:
                collector.add(
                    PlanDefectKind.INVALID_EFFECT,
                    f"effect {effect_id!r} uses unsupported operation {operation!r}",
                    (effect_id,),
                )
            if operation == "assign" and target:
                transition = str(_field(effect, "event_id", "task_id", default=""))
                key = (transition, target)
                value = _plain(effect.get("value"))
                prior = assignments.get(key)
                if prior is not None and prior[1] != value:
                    collector.add(
                        PlanDefectKind.CONTRADICTION,
                        f"effects assign incompatible values to {target!r}",
                        (prior[0], effect_id),
                        constraints=(prior[0], effect_id),
                        witness={"target": target, "values": [prior[1], value]},
                        core=True,
                    )
                assignments[key] = (effect_id, value)

    def _check_gate(
        self,
        label: str,
        kind: PlanDefectKind,
        value: Any,
        collector: _Collector,
    ) -> None:
        if value in (None, "", (), [], {}):
            return
        records = _record_values(value)
        if not records and isinstance(value, Mapping):
            records = (_mapping(value),)
        for index, record in enumerate(records):
            if not _status_failure(record):
                continue
            record_id = _record_id(record, f"{label}:gate:{index}")
            evidence = _ids(
                _field(record, "evidence_ids", "receipt_ids", "source_ids", default=()),
                "gate evidence id",
            )
            status = str(
                _field(record, "status", "decision", "outcome", default="failed")
            )
            collector.add(
                kind,
                f"{label} gate {record_id!r} failed closed with status {status!r}",
                (record_id,),
                evidence=evidence,
                witness={"gate": label, "status": status},
                core=True,
            )

    def _check_required_gate_bindings(
        self,
        source: Mapping[str, Any],
        *,
        policy: Any,
        ir: Any,
        security: Any,
        proof: Any,
        collector: _Collector,
    ) -> None:
        candidate = (
            source.get("plan")
            if isinstance(source.get("plan"), Mapping)
            else source
        )
        symbolic = _selected_symbolic(source)
        if isinstance(symbolic.get("plan"), Mapping):
            candidate = symbolic["plan"]

        task_records = _record_values(
            _field(source, "tasks", "steps", default=())
        )

        def required_ids(*names: str) -> set[str]:
            result = set(_ids(_field(source, *names, default=()), "required gate id"))
            result.update(
                _ids(_field(candidate, *names, default=()), "required gate id")
            )
            for task in task_records:
                result.update(
                    _ids(_field(task, *names, default=()), "required gate id")
                )
            return result

        def satisfied_ids(value: Any) -> set[str]:
            result = set(_evidence_ids(value))
            for record in _record_values(value):
                result.update(
                    _ids(
                        _field(
                            record,
                            "subject_ids",
                            "obligation_ids",
                            "requirement_ids",
                            "check_ids",
                            "satisfied_ids",
                            default=(),
                        ),
                        "satisfied gate id",
                    )
                )
            return result

        specifications = (
            (
                PlanDefectKind.POLICY_FAILURE,
                policy,
                required_ids("required_policy_ids", "policy_requirement_ids"),
                "policy",
            ),
            (
                PlanDefectKind.IR_FAILURE,
                ir,
                required_ids(
                    "validation_requirement_ids",
                    "validation_requirement_refs",
                    "required_ir_check_ids",
                ),
                "IR/validation",
            ),
            (
                PlanDefectKind.SECURITY_FAILURE,
                security,
                required_ids(
                    "required_security_gate_ids",
                    "security_requirement_ids",
                ),
                "security",
            ),
            (
                PlanDefectKind.PROOF_FAILURE,
                proof,
                required_ids(
                    "proof_requirement_ids",
                    "proof_requirement_refs",
                    "proof_obligation_ids",
                    "required_proof_obligations",
                ),
                "proof",
            ),
        )
        for kind, value, required, label in specifications:
            if not required:
                continue
            satisfied = satisfied_ids(value)
            for missing in sorted(required - satisfied):
                collector.add(
                    kind,
                    f"required {label} record {missing!r} is absent",
                    (missing,),
                    evidence=satisfied,
                    core=True,
                )

    def _check_candidate_claims(
        self,
        source: Mapping[str, Any],
        evidence: Any,
        collector: _Collector,
    ) -> None:
        candidate = source.get("plan") if isinstance(source.get("plan"), Mapping) else source
        symbolic = _selected_symbolic(source)
        if isinstance(symbolic.get("plan"), Mapping):
            candidate = symbolic["plan"]
        candidate_id = str(
            _field(candidate, "candidate_id", default=self._source_plan_id(source))
        )
        for violation in _ids(candidate.get("authority_violations"), "authority violation"):
            collector.add(
                PlanDefectKind.POLICY_FAILURE,
                f"candidate retains authority violation {violation!r}",
                (candidate_id, violation),
                core=True,
            )
        changed = set(_ids(candidate.get("changed_scopes"), "changed scope"))
        authorized = set(_ids(candidate.get("authorized_scopes"), "authorized scope"))
        for scope in sorted(changed - authorized):
            collector.add(
                PlanDefectKind.POLICY_FAILURE,
                f"changed scope {scope!r} is not authorized",
                (candidate_id, scope),
                core=True,
            )
        required_semantics = set(
            _ids(candidate.get("semantic_requirements"), "semantic requirement")
        )
        supported = set(_ids(candidate.get("supported_semantics"), "supported semantic"))
        for semantic in sorted(required_semantics - supported):
            collector.add(
                PlanDefectKind.IR_FAILURE,
                f"required semantic {semantic!r} is unsupported",
                (candidate_id, semantic),
                core=True,
            )
        if candidate.get("validation_feasible") is False:
            collector.add(
                PlanDefectKind.IR_FAILURE,
                "candidate validation is infeasible",
                (candidate_id,),
            )
        if candidate.get("proof_feasible") is False:
            collector.add(
                PlanDefectKind.PROOF_FAILURE,
                "candidate proof obligations are infeasible",
                (candidate_id,),
            )
        if evidence is not None:
            evidence_map = _mapping(evidence)
            current = evidence_map.get("current")
            if current is False or str(evidence_map.get("status") or "").casefold() == "stale":
                collector.add(
                    PlanDefectKind.STALE_EVIDENCE,
                    "plan evidence is stale at the current authority roots",
                    (candidate_id,),
                    evidence=_evidence_ids(evidence),
                )

    def _check_parallelism(
        self,
        source: Mapping[str, Any],
        parallel_plan: Any,
        tasks: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, set[str]],
        collector: _Collector,
    ) -> None:
        parallel = _mapping(parallel_plan) if parallel_plan is not None else {}
        if not parallel:
            parallel = _mapping(source.get("parallel_plan")) if source.get("parallel_plan") is not None else {}
        if not parallel:
            return
        plan_id = str(
            _field(parallel, "plan_id", "record_id", default=self._source_plan_id(source))
        )
        graph_width = _topological_width(set(tasks), edges)
        claimed_graph = _integer(
            _field(parallel, "graph_width", default=0),
            "graph_width",
            default=0,
        )
        requested = _integer(
            _field(parallel, "requested_width", "parallel_width", "claimed_width", default=0),
            "requested_width",
            default=0,
        )
        admitted = _integer(
            _field(parallel, "admitted_width", default=0),
            "admitted_width",
            default=0,
        )
        if claimed_graph and claimed_graph != graph_width:
            collector.add(
                PlanDefectKind.FALSE_PARALLELISM,
                "declared graph width does not match the recomputed dependency width",
                (plan_id,),
                witness={"declared_graph_width": claimed_graph, "recomputed_graph_width": graph_width},
                core=True,
            )
        if requested > graph_width and graph_width >= 0:
            collector.add(
                PlanDefectKind.FALSE_PARALLELISM,
                "requested parallel width exceeds independently recomputed ready width",
                (plan_id,),
                witness={"requested_width": requested, "recomputed_graph_width": graph_width},
                core=True,
            )
        if admitted > graph_width:
            collector.add(
                PlanDefectKind.FALSE_PARALLELISM,
                "admitted parallel width exceeds independently recomputed ready width",
                (plan_id,),
                witness={"admitted_width": admitted, "recomputed_graph_width": graph_width},
                core=True,
            )
        conflicts: set[frozenset[str]] = set()
        for record in _record_values(parallel.get("conflicts")):
            if record.get("blocking", True):
                left = str(record.get("left_task_id") or "")
                right = str(record.get("right_task_id") or "")
                if left and right:
                    conflicts.add(frozenset((left, right)))
        for wave_index, wave in enumerate(
            _sequence(
                _field(parallel, "execution_waves", "ready_waves", "waves", default=())
            )
        ):
            record = _mapping(wave) if isinstance(wave, Mapping) else {"task_ids": wave}
            wave_tasks = set(
                _ids(
                    _field(
                        record,
                        "task_ids",
                        "admitted_task_ids",
                        "graph_ready_task_ids",
                        default=(),
                    ),
                    "wave task id",
                )
            )
            for task_id in wave_tasks:
                dependency_in_wave = edges.get(task_id, set()) & wave_tasks
                if dependency_in_wave:
                    dependency = sorted(dependency_in_wave)[0]
                    collector.add(
                        PlanDefectKind.FALSE_PARALLELISM,
                        f"execution wave {wave_index} co-schedules a dependency edge",
                        (task_id, dependency, plan_id),
                        constraints=(f"dependency:{dependency}->{task_id}",),
                        core=True,
                    )
            for pair in conflicts:
                if pair <= wave_tasks:
                    collector.add(
                        PlanDefectKind.FALSE_PARALLELISM,
                        f"execution wave {wave_index} co-schedules a blocking conflict",
                        (*sorted(pair), plan_id),
                        constraints=(f"conflict:{':'.join(sorted(pair))}",),
                        core=True,
                    )
        issues = _record_values(parallel.get("issues"))
        for issue in issues:
            code = str(issue.get("code") or "")
            if code in {"fake_lane_label", "dependency_cycle"}:
                collector.add(
                    PlanDefectKind.FALSE_PARALLELISM,
                    f"parallel compiler reported {code!r}",
                    (*_ids(issue.get("task_ids"), "issue task id"), plan_id),
                    evidence=_ids(issue.get("evidence"), "issue evidence id"),
                    core=True,
                )

    def _check_resources(
        self,
        source: Mapping[str, Any],
        parallel_plan: Any,
        resources: Any,
        tasks: Mapping[str, Mapping[str, Any]],
        collector: _Collector,
    ) -> None:
        parallel = _mapping(parallel_plan) if parallel_plan is not None else {}
        feasibility = _mapping(parallel.get("resource_feasibility")) if parallel.get("resource_feasibility") is not None else {}
        if feasibility and feasibility.get("feasible") is False:
            plan_id = str(_field(parallel, "plan_id", default=self._source_plan_id(source)))
            collector.add(
                PlanDefectKind.RESOURCE_INFEASIBLE,
                "parallel plan resource feasibility check failed",
                (plan_id,),
                witness={
                    key: value
                    for key, value in feasibility.items()
                    if key.endswith("_feasible") or key in {"required_totals", "available_host"}
                },
                core=True,
            )
        snapshot = _mapping(resources) if resources is not None else {}
        available = _mapping(
            _field(
                snapshot,
                "available",
                "capacity",
                "available_resources",
                "available_host",
                default=feasibility.get("available_host", {}),
            )
        )
        required = _mapping(
            _field(
                source,
                "required_resources",
                "resource_requirements",
                default=feasibility.get("required_totals", {}),
            )
        )
        for task in tasks.values():
            for key, value in _mapping(
                _field(task, "required_resources", "resources", default={})
            ).items():
                try:
                    required[str(key)] = _integer(
                        required.get(str(key), 0),
                        f"required_resources.{key}",
                    ) + _integer(value, f"task resource {key}")
                except PlanCriticError:
                    continue
        if not required and not available:
            return
        for key in sorted(set(required) | set(available)):
            try:
                needed = _integer(required.get(key, 0), f"required.{key}")
                capacity = _integer(available.get(key, 0), f"available.{key}")
            except PlanCriticError:
                continue
            if needed > capacity:
                record_ids = tuple(
                    sorted(
                        task_id
                        for task_id, task in tasks.items()
                        if key
                        in _mapping(
                            _field(task, "required_resources", "resources", default={})
                        )
                    )
                ) or (self._source_plan_id(source),)
                collector.add(
                    PlanDefectKind.RESOURCE_INFEASIBLE,
                    f"required {key!r} capacity exceeds the fresh resource snapshot",
                    record_ids,
                    evidence=_evidence_ids(snapshot),
                    constraints=(f"resource:{key}",),
                    witness={"resource": key, "required": needed, "available": capacity},
                    core=True,
                )


def critique_plan(plan: Any = None, **kwargs: Any) -> PlanCritique:
    bounds = kwargs.pop("bounds", None)
    return PlanCritic(bounds=bounds).critique(plan, **kwargs)


evaluate_plan_critique = critique_plan
criticize_plan = critique_plan
critique_candidate = critique_plan

# Compatibility spellings used by the surrounding planning packages.
CritiqueBounds = PlanCritiqueBounds
PlanCriticFinding = PlanCritiqueFinding
PlanCritiqueCounterexample = TypedPlanCounterexample
PlanCritiqueResult = PlanCritique
DeterministicPlanCritic = PlanCritic


__all__ = [
    "CritiqueFinding",
    "CritiqueFindingKind",
    "CritiqueBounds",
    "CritiqueSeverity",
    "DeterministicPlanCritic",
    "MinimalUnsatCore",
    "PLAN_COUNTEREXAMPLE_SCHEMA",
    "PLAN_CRITIC_INTERFACE",
    "PLAN_CRITIQUE_FINDING_SCHEMA",
    "PLAN_CRITIQUE_INTERFACE",
    "PLAN_CRITIQUE_SCHEMA",
    "PLAN_CRITIQUE_VERSION",
    "PLAN_UNSAT_CORE_SCHEMA",
    "PlanCounterexample",
    "PlanCritic",
    "PlanCriticError",
    "PlanCriticFinding",
    "PlanCritique",
    "PlanCritiqueBounds",
    "PlanCritiqueBoundsError",
    "PlanCritiqueDecision",
    "PlanCritiqueCounterexample",
    "PlanCritiqueFinding",
    "PlanCritiqueFindingKind",
    "PlanCritiqueRequest",
    "PlanCritiqueResult",
    "PlanCritiqueSeverity",
    "PlanDefectKind",
    "PlanUnsatCore",
    "TypedPlanCounterexample",
    "UnsatCore",
    "criticize_plan",
    "critique_candidate",
    "critique_plan",
    "evaluate_plan_critique",
]
