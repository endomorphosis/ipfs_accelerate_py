"""Plan-trace conformance and evidence-backed goal completion.

This module is the trust boundary between an accepted
:class:`~.formal_planning_contracts.FormalWorkPlan`, the transitions that were
actually observed, and a goal-completion decision.  A consistent plan is only
evidence about the plan: it is never treated as implementation, test, proof,
protocol, or runtime evidence.

All semantic inputs are content addressed.  A conformance receipt binds the
exact plan, completion policy, repository tree, AST scopes, premises, and
known counterexamples.  Changing any of those inputs invalidates the receipt.
The complete evaluation packet can be stored as canonical JSON or DuckDB and
replayed without relying on process-local state.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_planning_contracts import (
    EventKind,
    FormalWorkPlan,
    PlanConformanceLevel,
    PlanEvent,
)
from .formal_verification_contracts import canonical_json, content_identity
from .goal_completion import GoalState, normalize_goal_state


FORMAL_PLAN_CONFORMANCE_VERSION: Final = 1
FORMAL_PLAN_CONFORMANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-conformance@1"
)
CONFORMANCE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-conformance-binding@1"
)
EXECUTION_EVENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-execution-event@1"
)
COMPLETION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-completion-evidence@1"
)
COMPLETION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-completion-policy@1"
)
GOAL_COMPLETION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-goal-completion@1"
)
CONFORMANCE_REPLAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-conformance-replay@1"
)


class ConformanceValidationError(ValueError):
    """Raised when conformance input cannot be interpreted canonically."""


class TransitionDisposition(str, Enum):
    """How an observed or expected transition relates to the accepted plan."""

    MATCHED = "matched"
    SKIPPED = "skipped"
    REORDERED = "reordered"
    UNAUTHORIZED = "unauthorized"
    FAILED = "failed"
    OVERRIDDEN = "overridden"
    SUPERSEDED = "superseded"


class ConformanceVerdict(str, Enum):
    """Overall result of comparing one finite execution to one plan."""

    CONFORMANT = "conformant"
    INCOMPLETE = "incomplete"
    VIOLATED = "violated"
    INVALIDATED = "invalidated"


class CompletionEvidenceKind(str, Enum):
    """Independent lanes which may be required by completion policy."""

    CODE = "code"
    TEST = "test"
    KERNEL = "kernel"
    MODEL_CHECK = "model_check"
    PROTOCOL = "protocol"
    RUNTIME = "runtime"


# Public short spelling used by callers which already import EvidenceKind from
# another proof module.
FormalEvidenceKind = CompletionEvidenceKind


class EvidenceCheckStatus(str, Enum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"
    UNBOUND = "unbound"
    INVALIDATED = "invalidated"


class InvalidationCause(str, Enum):
    PLAN_CHANGED = "plan_changed"
    POLICY_CHANGED = "policy_changed"
    REPOSITORY_TREE_CHANGED = "repository_tree_changed"
    AST_CHANGED = "ast_changed"
    PREMISE_CHANGED = "premise_changed"
    COUNTEREXAMPLE_CHANGED = "counterexample_changed"


_KIND_ALIASES: Final[Mapping[str, CompletionEvidenceKind]] = {
    "code": CompletionEvidenceKind.CODE,
    "code_change": CompletionEvidenceKind.CODE,
    "source": CompletionEvidenceKind.CODE,
    "implementation": CompletionEvidenceKind.CODE,
    "implementation_evidence": CompletionEvidenceKind.CODE,
    "artifact": CompletionEvidenceKind.CODE,
    "test": CompletionEvidenceKind.TEST,
    "tests": CompletionEvidenceKind.TEST,
    "unit_test": CompletionEvidenceKind.TEST,
    "integration_test": CompletionEvidenceKind.TEST,
    "pytest": CompletionEvidenceKind.TEST,
    "validation": CompletionEvidenceKind.TEST,
    "kernel": CompletionEvidenceKind.KERNEL,
    "kernel_check": CompletionEvidenceKind.KERNEL,
    "kernel_verification": CompletionEvidenceKind.KERNEL,
    "code_proof": CompletionEvidenceKind.KERNEL,
    "proof_receipt": CompletionEvidenceKind.KERNEL,
    "proof": CompletionEvidenceKind.KERNEL,
    "model": CompletionEvidenceKind.MODEL_CHECK,
    "model_check": CompletionEvidenceKind.MODEL_CHECK,
    "model_checking": CompletionEvidenceKind.MODEL_CHECK,
    "model_checker": CompletionEvidenceKind.MODEL_CHECK,
    "model-check": CompletionEvidenceKind.MODEL_CHECK,
    "tla": CompletionEvidenceKind.MODEL_CHECK,
    "smt": CompletionEvidenceKind.MODEL_CHECK,
    "protocol": CompletionEvidenceKind.PROTOCOL,
    "protocol_check": CompletionEvidenceKind.PROTOCOL,
    "protocol_verification": CompletionEvidenceKind.PROTOCOL,
    "proverif": CompletionEvidenceKind.PROTOCOL,
    "tamarin": CompletionEvidenceKind.PROTOCOL,
    "runtime": CompletionEvidenceKind.RUNTIME,
    "mtl": CompletionEvidenceKind.RUNTIME,
    "runtime_monitor": CompletionEvidenceKind.RUNTIME,
    "runtime_mtl": CompletionEvidenceKind.RUNTIME,
    "temporal_monitor": CompletionEvidenceKind.RUNTIME,
}

_PASS_VERDICTS: Final = frozenset(
    {
        "accepted",
        "complete",
        "completed",
        "conformant",
        "current",
        "ok",
        "pass",
        "passed",
        "proved",
        "satisfied",
        "success",
        "succeeded",
        "verified",
    }
)
_FAIL_VERDICTS: Final = frozenset(
    {
        "cancelled",
        "counterexample",
        "error",
        "fail",
        "failed",
        "invalid",
        "rejected",
        "stale",
        "timeout",
        "violated",
    }
)


def _text(value: Any, *, field_name: str, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ConformanceValidationError(f"{field_name} is required")
    return result


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ConformanceValidationError("expected a string sequence")
    return tuple(sorted({_text(item, field_name="identifier", required=True) for item in values}))


def _mapping(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ConformanceValidationError("expected a mapping")
    # Round-tripping through canonical JSON rejects non-public/non-canonical
    # values and detaches the immutable record from caller-owned structures.
    return json.loads(canonical_json(value))


def _timestamp(value: datetime | str | int | float | None, *, required: bool = False) -> str:
    if value in (None, ""):
        if required:
            raise ConformanceValidationError("timestamp is required")
        return ""
    if isinstance(value, bool):
        raise ConformanceValidationError("boolean is not a timestamp")
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ConformanceValidationError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value or "").strip().lower()


def _event_kind(value: Any) -> str:
    normalized = _enum_value(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "done": EventKind.COMPLETED.value,
        "complete": EventKind.COMPLETED.value,
        "running": EventKind.STARTED.value,
        "start": EventKind.STARTED.value,
        "error": EventKind.FAILED.value,
        "success": EventKind.COMPLETED.value,
        "succeeded": EventKind.COMPLETED.value,
        "proof": EventKind.EVIDENCE_PRODUCED.value,
    }
    return aliases.get(normalized, normalized or EventKind.EXECUTED.value)


def _evidence_kind(value: Any) -> CompletionEvidenceKind:
    if isinstance(value, CompletionEvidenceKind):
        return value
    normalized = _enum_value(value).replace(" ", "_")
    try:
        return _KIND_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(item.value for item in CompletionEvidenceKind)
        raise ConformanceValidationError(
            f"unknown completion evidence kind {value!r}; expected one of: {choices}"
        ) from exc


@dataclass(frozen=True)
class ConformanceBinding:
    """Semantic inputs to which conformance and evidence are bound."""

    plan_id: str
    policy_id: str
    repository_tree_id: str
    ast_scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("plan_id", "policy_id", "repository_tree_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        for name in ("ast_scope_ids", "premise_ids", "counterexample_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name)))

    @property
    def binding_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def ast_ids(self) -> tuple[str, ...]:
        return self.ast_scope_ids

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "ast_scope_ids": list(self.ast_scope_ids),
            "premise_ids": list(self.premise_ids),
            "counterexample_ids": list(self.counterexample_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFORMANCE_BINDING_SCHEMA,
            "binding_id": self.binding_id,
            **self._identity_payload(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceBinding":
        result = cls(
            plan_id=payload.get("plan_id", ""),
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get(
                "repository_tree_id", payload.get("tree_id", "")
            ),
            ast_scope_ids=tuple(
                payload.get("ast_scope_ids", payload.get("ast_ids", ())) or ()
            ),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            counterexample_ids=tuple(payload.get("counterexample_ids") or ()),
        )
        claimed = payload.get("binding_id")
        if claimed and claimed != result.binding_id:
            raise ConformanceValidationError("conformance binding identity mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ConformanceBinding":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class CompletionPolicy:
    """Configured, independent evidence and transition requirements."""

    required_evidence: tuple[CompletionEvidenceKind, ...] = tuple(
        CompletionEvidenceKind
    )
    max_age_seconds: Mapping[str, float | int | str | None] = field(default_factory=dict)
    allow_overridden: bool = False
    allow_superseded: bool = False
    require_artifact_id: bool = True
    require_current_freshness: bool = True
    require_exact_binding: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kinds = tuple(
            sorted(
                {_evidence_kind(item) for item in self.required_evidence},
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "required_evidence", kinds)
        # The shared formal-contract identity boundary intentionally rejects
        # JSON floats. Preserve whole-second bounds as integers and encode
        # fractional seconds as canonical decimal strings.
        ages: dict[str, int | str | None] = {}
        if not isinstance(self.max_age_seconds, Mapping):
            raise ConformanceValidationError("max_age_seconds must be a mapping")
        for key, value in self.max_age_seconds.items():
            kind = _evidence_kind(key).value
            if value is None:
                ages[kind] = None
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ConformanceValidationError(
                    f"max_age_seconds[{kind}] must be numeric or null"
                ) from exc
            if isinstance(value, bool) or not math.isfinite(numeric) or numeric < 0:
                raise ConformanceValidationError(
                    f"max_age_seconds[{kind}] must be non-negative or null"
                )
            ages[kind] = (
                int(numeric)
                if numeric.is_integer()
                else format(numeric, ".15g")
            )
        object.__setattr__(self, "max_age_seconds", dict(sorted(ages.items())))
        for name in (
            "allow_overridden",
            "allow_superseded",
            "require_artifact_id",
            "require_current_freshness",
            "require_exact_binding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConformanceValidationError(f"{name} must be boolean")
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def policy_id(self) -> str:
        return content_identity(self._identity_payload())

    def max_age_for(self, kind: CompletionEvidenceKind) -> float | None:
        value = self.max_age_seconds.get(kind.value)
        return None if value is None else float(value)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "required_evidence": [item.value for item in self.required_evidence],
            "max_age_seconds": dict(self.max_age_seconds),
            "allow_overridden": self.allow_overridden,
            "allow_superseded": self.allow_superseded,
            "require_artifact_id": self.require_artifact_id,
            "require_current_freshness": self.require_current_freshness,
            "require_exact_binding": self.require_exact_binding,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPLETION_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            **self._identity_payload(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionPolicy":
        result = cls(
            required_evidence=tuple(
                payload.get("required_evidence", payload.get("required_kinds", ()))
                or ()
            ),
            max_age_seconds=payload.get("max_age_seconds") or {},
            allow_overridden=payload.get("allow_overridden", False),
            allow_superseded=payload.get("allow_superseded", False),
            require_artifact_id=payload.get("require_artifact_id", True),
            require_current_freshness=payload.get(
                "require_current_freshness", True
            ),
            require_exact_binding=payload.get("require_exact_binding", True),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("policy_id")
        if claimed and claimed != result.policy_id:
            raise ConformanceValidationError("completion policy identity mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "CompletionPolicy":
        return cls.from_dict(json.loads(payload))


def binding_for_plan(
    plan: FormalWorkPlan,
    policy: CompletionPolicy,
    *,
    repository_tree_id: str | None = None,
    ast_scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    counterexample_ids: Sequence[str] = (),
) -> ConformanceBinding:
    """Create the exact semantic binding used by a completion evaluation."""

    return ConformanceBinding(
        plan_id=plan.plan_id,
        policy_id=policy.policy_id,
        repository_tree_id=repository_tree_id or plan.repository_tree_id,
        ast_scope_ids=tuple(ast_scope_ids),
        premise_ids=tuple(premise_ids),
        counterexample_ids=tuple(counterexample_ids),
    )


@dataclass(frozen=True)
class CanonicalExecutionEvent:
    """Storage-independent observed supervisor transition."""

    event_id: str
    task_id: str
    kind: str
    actor_id: str
    sequence: int
    plan_event_id: str = ""
    status: str = ""
    authorized: bool | None = None
    overrides_event_id: str = ""
    supersedes_event_id: str = ""
    plan_id: str = ""
    repository_tree_id: str = ""
    provenance_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "task_id", "actor_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(self, "kind", _event_kind(self.kind))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ConformanceValidationError("event sequence must be an integer")
        for name in (
            "plan_event_id",
            "status",
            "overrides_event_id",
            "supersedes_event_id",
            "plan_id",
            "repository_tree_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        if self.authorized is not None and not isinstance(self.authorized, bool):
            raise ConformanceValidationError("authorized must be boolean or null")
        object.__setattr__(self, "provenance_ids", _strings(self.provenance_ids))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def execution_event_id(self) -> str:
        return content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "actor_id": self.actor_id,
            "sequence": self.sequence,
            "plan_event_id": self.plan_event_id,
            "status": self.status,
            "authorized": self.authorized,
            "overrides_event_id": self.overrides_event_id,
            "supersedes_event_id": self.supersedes_event_id,
            "plan_id": self.plan_id,
            "repository_tree_id": self.repository_tree_id,
            "provenance_ids": list(self.provenance_ids),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_EVENT_SCHEMA,
            "execution_event_id": self.execution_event_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, fallback_sequence: int = 0
    ) -> "CanonicalExecutionEvent":
        metadata = payload.get("metadata") or payload.get("details") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        event_id = (
            payload.get("event_id")
            or payload.get("execution_event_id")
            or payload.get("transition_id")
            or payload.get("id")
        )
        plan_event_id = (
            payload.get("plan_event_id")
            or payload.get("expected_event_id")
            or payload.get("accepted_transition_id")
            or ""
        )
        # If a raw trace uses the accepted event ID as its event ID, exact
        # matching still works without manufacturing a second identifier.
        result = cls(
            event_id=event_id,
            task_id=payload.get(
                "task_id", payload.get("work_item_id", payload.get("task", ""))
            ),
            kind=payload.get(
                "kind",
                payload.get(
                    "event_kind",
                    payload.get("transition", payload.get("event_type", payload.get("status", ""))),
                ),
            ),
            actor_id=payload.get(
                "actor_id",
                payload.get("agent_id", payload.get("principal_id", payload.get("actor", ""))),
            ),
            sequence=int(
                payload.get(
                    "sequence",
                    payload.get(
                        "logical_time",
                        payload.get("ordinal", payload.get("index", fallback_sequence)),
                    ),
                )
            ),
            plan_event_id=plan_event_id,
            status=payload.get("status", ""),
            authorized=payload.get(
                "authorized", payload.get("authorization_granted", None)
            ),
            overrides_event_id=payload.get(
                "overrides_event_id", payload.get("overrides", "")
            ),
            supersedes_event_id=payload.get(
                "supersedes_event_id", payload.get("supersedes", "")
            ),
            plan_id=payload.get("plan_id", payload.get("accepted_plan_id", "")),
            repository_tree_id=payload.get(
                "repository_tree_id", payload.get("tree_id", "")
            ),
            provenance_ids=tuple(
                payload.get("provenance_ids", payload.get("receipt_ids", ())) or ()
            ),
            metadata=metadata,
        )
        claimed = payload.get("execution_event_id")
        # execution_event_id is also accepted as the primary event ID by
        # legacy records; only validate it when an explicit event_id exists.
        if payload.get("event_id") and claimed and claimed != result.execution_event_id:
            raise ConformanceValidationError("execution event identity mismatch")
        return result

    @classmethod
    def from_json(
        cls, payload: str | bytes | bytearray
    ) -> "CanonicalExecutionEvent":
        return cls.from_dict(json.loads(payload))


ExecutionEvent = CanonicalExecutionEvent


@dataclass(frozen=True)
class TransitionFinding:
    disposition: TransitionDisposition
    expected_event_id: str = ""
    observed_event_id: str = ""
    task_id: str = ""
    expected_index: int | None = None
    observed_index: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", TransitionDisposition(self.disposition)
        )
        for name in ("expected_event_id", "observed_event_id", "task_id", "reason"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )

    @property
    def finding_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "disposition": self.disposition.value,
            "expected_event_id": self.expected_event_id,
            "observed_event_id": self.observed_event_id,
            "task_id": self.task_id,
            "expected_index": self.expected_index,
            "observed_index": self.observed_index,
            "reason": self.reason,
        }
        if include_id:
            result["finding_id"] = self.finding_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionFinding":
        result = cls(
            disposition=payload.get("disposition", TransitionDisposition.UNAUTHORIZED),
            expected_event_id=payload.get("expected_event_id", ""),
            observed_event_id=payload.get("observed_event_id", ""),
            task_id=payload.get("task_id", ""),
            expected_index=payload.get("expected_index"),
            observed_index=payload.get("observed_index"),
            reason=payload.get("reason", ""),
        )
        if payload.get("finding_id") and payload["finding_id"] != result.finding_id:
            raise ConformanceValidationError("transition finding identity mismatch")
        return result


def changed_bindings(
    prior: ConformanceBinding, current: ConformanceBinding
) -> tuple[InvalidationCause, ...]:
    """Return every semantic input family which changed."""

    causes: list[InvalidationCause] = []
    comparisons = (
        ("plan_id", InvalidationCause.PLAN_CHANGED),
        ("policy_id", InvalidationCause.POLICY_CHANGED),
        ("repository_tree_id", InvalidationCause.REPOSITORY_TREE_CHANGED),
        ("ast_scope_ids", InvalidationCause.AST_CHANGED),
        ("premise_ids", InvalidationCause.PREMISE_CHANGED),
        ("counterexample_ids", InvalidationCause.COUNTEREXAMPLE_CHANGED),
    )
    for field_name, cause in comparisons:
        if getattr(prior, field_name) != getattr(current, field_name):
            causes.append(cause)
    return tuple(causes)


def invalidate_plan_conformance(
    prior: "PlanConformanceResult | Mapping[str, Any]",
    current_binding: ConformanceBinding | Mapping[str, Any],
) -> "PlanConformanceResult":
    """Invalidate a receipt when any of its semantic bindings changed.

    An unchanged binding returns the original receipt.  This makes invalidation
    processing idempotent during daemon restart and event replay.
    """

    if not isinstance(prior, PlanConformanceResult):
        prior = PlanConformanceResult.from_dict(prior)
    if not isinstance(current_binding, ConformanceBinding):
        current_binding = ConformanceBinding.from_dict(current_binding)
    causes = changed_bindings(prior.binding, current_binding)
    if not causes:
        return prior
    return PlanConformanceResult(
        plan_id=current_binding.plan_id,
        binding=current_binding,
        verdict=ConformanceVerdict.INVALIDATED,
        findings=prior.findings,
        expected_event_ids=prior.expected_event_ids,
        observed_event_ids=prior.observed_event_ids,
        invalidation_causes=causes,
    )


@dataclass(frozen=True)
class PlanConformanceResult:
    plan_id: str
    binding: ConformanceBinding
    verdict: ConformanceVerdict
    findings: tuple[TransitionFinding, ...]
    expected_event_ids: tuple[str, ...]
    observed_event_ids: tuple[str, ...]
    invalidation_causes: tuple[InvalidationCause, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, field_name="plan_id", required=True))
        if not isinstance(self.binding, ConformanceBinding):
            object.__setattr__(self, "binding", ConformanceBinding.from_dict(self.binding))
        object.__setattr__(self, "verdict", ConformanceVerdict(self.verdict))
        object.__setattr__(
            self,
            "findings",
            tuple(
                item
                if isinstance(item, TransitionFinding)
                else TransitionFinding.from_dict(item)
                for item in self.findings
            ),
        )
        object.__setattr__(self, "expected_event_ids", tuple(self.expected_event_ids))
        object.__setattr__(self, "observed_event_ids", tuple(self.observed_event_ids))
        object.__setattr__(
            self,
            "invalidation_causes",
            tuple(InvalidationCause(item) for item in self.invalidation_causes),
        )

    @property
    def conformant(self) -> bool:
        return self.verdict is ConformanceVerdict.CONFORMANT

    @property
    def level(self) -> PlanConformanceLevel:
        if self.verdict is ConformanceVerdict.CONFORMANT:
            return PlanConformanceLevel.BOUNDED_CONFORMANT
        if self.verdict in (ConformanceVerdict.VIOLATED, ConformanceVerdict.INVALIDATED):
            return PlanConformanceLevel.VIOLATED
        return PlanConformanceLevel.INCONCLUSIVE

    @property
    def conformance_level(self) -> PlanConformanceLevel:
        return self.level

    @property
    def receipt_id(self) -> str:
        return content_identity(self._identity_payload())

    def by_disposition(
        self, disposition: TransitionDisposition | str
    ) -> tuple[TransitionFinding, ...]:
        selected = TransitionDisposition(disposition)
        return tuple(item for item in self.findings if item.disposition is selected)

    @property
    def skipped(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.SKIPPED)

    @property
    def reordered(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.REORDERED)

    @property
    def unauthorized(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.UNAUTHORIZED)

    @property
    def failed(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.FAILED)

    @property
    def overridden(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.OVERRIDDEN)

    @property
    def superseded(self) -> tuple[TransitionFinding, ...]:
        return self.by_disposition(TransitionDisposition.SUPERSEDED)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "binding": self.binding.to_dict(),
            "verdict": self.verdict.value,
            "findings": [item.to_dict() for item in self.findings],
            "expected_event_ids": list(self.expected_event_ids),
            "observed_event_ids": list(self.observed_event_ids),
            "invalidation_causes": [item.value for item in self.invalidation_causes],
        }

    def to_dict(self) -> dict[str, Any]:
        grouped = {
            kind.value: [item.to_dict() for item in self.by_disposition(kind)]
            for kind in TransitionDisposition
        }
        return {
            "schema": FORMAL_PLAN_CONFORMANCE_SCHEMA,
            "version": FORMAL_PLAN_CONFORMANCE_VERSION,
            "receipt_id": self.receipt_id,
            "conformance_level": self.level.value,
            **self._identity_payload(),
            "transitions": grouped,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanConformanceResult":
        result = cls(
            plan_id=payload.get("plan_id", ""),
            binding=ConformanceBinding.from_dict(payload.get("binding") or {}),
            verdict=payload.get("verdict", ConformanceVerdict.INCOMPLETE),
            findings=tuple(
                TransitionFinding.from_dict(item)
                for item in payload.get("findings", ())
            ),
            expected_event_ids=tuple(payload.get("expected_event_ids") or ()),
            observed_event_ids=tuple(payload.get("observed_event_ids") or ()),
            invalidation_causes=tuple(payload.get("invalidation_causes") or ()),
        )
        if payload.get("receipt_id") and payload["receipt_id"] != result.receipt_id:
            raise ConformanceValidationError("conformance receipt identity mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "PlanConformanceResult":
        return cls.from_dict(json.loads(payload))


def _canonical_events(
    events: Iterable[CanonicalExecutionEvent | Mapping[str, Any]],
) -> tuple[CanonicalExecutionEvent, ...]:
    normalized = [
        item
        if isinstance(item, CanonicalExecutionEvent)
        else CanonicalExecutionEvent.from_dict(item, fallback_sequence=index)
        for index, item in enumerate(events)
    ]
    identities: set[str] = set()
    event_ids: set[str] = set()
    for item in normalized:
        if item.execution_event_id in identities:
            raise ConformanceValidationError(
                f"duplicate execution event: {item.event_id}"
            )
        if item.event_id in event_ids:
            raise ConformanceValidationError(
                f"duplicate execution event_id: {item.event_id}"
            )
        identities.add(item.execution_event_id)
        event_ids.add(item.event_id)
    return tuple(sorted(normalized, key=lambda item: (item.sequence, item.event_id)))


class FormalPlanConformanceEvaluator:
    """Deterministically compare canonical execution events with a plan."""

    def evaluate(
        self,
        plan: FormalWorkPlan | Mapping[str, Any],
        events: Iterable[CanonicalExecutionEvent | Mapping[str, Any]],
        *,
        policy: CompletionPolicy | Mapping[str, Any] | None = None,
        binding: ConformanceBinding | Mapping[str, Any] | None = None,
        prior: PlanConformanceResult | Mapping[str, Any] | None = None,
        ast_scope_ids: Sequence[str] = (),
        premise_ids: Sequence[str] = (),
        counterexample_ids: Sequence[str] = (),
        repository_tree_id: str | None = None,
    ) -> PlanConformanceResult:
        if not isinstance(plan, FormalWorkPlan):
            plan = FormalWorkPlan.from_dict(plan)
        if policy is None:
            policy = CompletionPolicy()
        elif not isinstance(policy, CompletionPolicy):
            policy = CompletionPolicy.from_dict(policy)
        if binding is None:
            binding = binding_for_plan(
                plan,
                policy,
                repository_tree_id=repository_tree_id,
                ast_scope_ids=ast_scope_ids,
                premise_ids=premise_ids,
                counterexample_ids=counterexample_ids,
            )
        elif not isinstance(binding, ConformanceBinding):
            binding = ConformanceBinding.from_dict(binding)
        if prior is not None and not isinstance(prior, PlanConformanceResult):
            prior = PlanConformanceResult.from_dict(prior)

        invalidations: list[InvalidationCause] = []
        expected_binding = binding_for_plan(
            plan,
            policy,
            repository_tree_id=repository_tree_id or binding.repository_tree_id,
            ast_scope_ids=binding.ast_scope_ids,
            premise_ids=binding.premise_ids,
            counterexample_ids=binding.counterexample_ids,
        )
        invalidations.extend(changed_bindings(binding, expected_binding))
        binding = expected_binding
        if prior is not None:
            invalidations.extend(changed_bindings(prior.binding, binding))

        observed = _canonical_events(events)
        expected = tuple(
            sorted(plan.events, key=lambda item: (item.logical_time, item.event_id))
        )
        expected_by_id = {item.event_id: item for item in expected}
        expected_index = {item.event_id: index for index, item in enumerate(expected)}
        task_actors = {item.task_id: set(item.actor_ids) for item in plan.tasks}
        matched_expected: dict[str, CanonicalExecutionEvent] = {}
        handled_expected: set[str] = set()
        findings: list[TransitionFinding] = []
        matched_order: list[tuple[int, int, PlanEvent, CanonicalExecutionEvent]] = []

        def add(
            disposition: TransitionDisposition,
            actual: CanonicalExecutionEvent | None = None,
            planned: PlanEvent | None = None,
            reason: str = "",
            observed_position: int | None = None,
        ) -> None:
            findings.append(
                TransitionFinding(
                    disposition=disposition,
                    expected_event_id=planned.event_id if planned else "",
                    observed_event_id=actual.event_id if actual else "",
                    task_id=(
                        planned.task_id if planned else (actual.task_id if actual else "")
                    ),
                    expected_index=(
                        expected_index.get(planned.event_id) if planned else None
                    ),
                    observed_index=observed_position,
                    reason=reason,
                )
            )

        for observed_position, actual in enumerate(observed):
            reference = actual.plan_event_id
            if not reference and actual.event_id in expected_by_id:
                reference = actual.event_id

            override_reference = actual.overrides_event_id
            supersede_reference = actual.supersedes_event_id
            actual_status = actual.status.lower().replace("-", "_")
            metadata = actual.metadata
            if not override_reference and actual_status == "overridden":
                override_reference = str(
                    metadata.get("target_event_id")
                    or metadata.get("overridden_event_id")
                    or reference
                )
            if not supersede_reference and actual_status == "superseded":
                supersede_reference = str(
                    metadata.get("target_event_id")
                    or metadata.get("superseded_event_id")
                    or reference
                )

            if override_reference:
                planned = expected_by_id.get(override_reference)
                add(
                    TransitionDisposition.OVERRIDDEN,
                    actual,
                    planned,
                    "an observed transition explicitly overrode an accepted transition",
                    observed_position,
                )
                if planned:
                    handled_expected.add(planned.event_id)
                continue
            if supersede_reference:
                planned = expected_by_id.get(supersede_reference)
                add(
                    TransitionDisposition.SUPERSEDED,
                    actual,
                    planned,
                    "an observed transition explicitly superseded an accepted transition",
                    observed_position,
                )
                if planned:
                    handled_expected.add(planned.event_id)
                continue

            planned = expected_by_id.get(reference) if reference else None
            if planned is None:
                candidates = [
                    item
                    for item in expected
                    if item.event_id not in matched_expected
                    and item.event_id not in handled_expected
                    and item.task_id == actual.task_id
                    and item.kind.value == actual.kind
                ]
                if candidates:
                    actor_matches = [
                        item for item in candidates if item.actor_id == actual.actor_id
                    ]
                    planned = (actor_matches or candidates)[0]

            if actual.plan_id and actual.plan_id != binding.plan_id:
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    "execution event is bound to a different accepted plan",
                    observed_position,
                )
                continue
            if (
                actual.repository_tree_id
                and actual.repository_tree_id != binding.repository_tree_id
            ):
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    "execution event is bound to a different repository tree",
                    observed_position,
                )
                continue
            authorized_actor = (
                planned is not None
                and actual.actor_id == planned.actor_id
                and actual.actor_id in task_actors.get(actual.task_id, set())
            )
            if actual.authorized is False or not authorized_actor:
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    (
                        "actor lacks accepted transition authority"
                        if planned
                        else "transition does not occur in the accepted plan"
                    ),
                    observed_position,
                )
                continue
            if planned is None:
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    None,
                    "transition does not occur in the accepted plan",
                    observed_position,
                )
                continue
            if planned.event_id in matched_expected:
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    "accepted transition was executed more than once",
                    observed_position,
                )
                continue

            if (
                actual.kind == EventKind.FAILED.value
                or actual_status in _FAIL_VERDICTS
            ) and planned.kind is not EventKind.FAILED:
                matched_expected[planned.event_id] = actual
                add(
                    TransitionDisposition.FAILED,
                    actual,
                    planned,
                    "accepted transition failed instead of reaching its intended outcome",
                    observed_position,
                )
                continue

            matched_expected[planned.event_id] = actual
            matched_order.append(
                (
                    expected_index[planned.event_id],
                    observed_position,
                    planned,
                    actual,
                )
            )
            add(
                TransitionDisposition.MATCHED,
                actual,
                planned,
                "observed transition matches the accepted plan",
                observed_position,
            )

        # Equal logical times are concurrent in the bounded plan and therefore
        # have no ordering relationship.  The event ID tie-breaker is only for
        # canonical serialization, never a semantic sequencing constraint.
        greatest_logical_time = -1
        for _planned_index, observed_position, planned, actual in matched_order:
            if planned.logical_time < greatest_logical_time:
                add(
                    TransitionDisposition.REORDERED,
                    actual,
                    planned,
                    "transition occurred after a transition that is later in the accepted plan",
                    observed_position,
                )
            greatest_logical_time = max(greatest_logical_time, planned.logical_time)

        for planned in expected:
            if (
                planned.event_id not in matched_expected
                and planned.event_id not in handled_expected
            ):
                add(
                    TransitionDisposition.SKIPPED,
                    None,
                    planned,
                    "accepted transition was not observed",
                )

        # A plan without explicit transitions cannot attest execution.  Keep a
        # focused skipped finding per task rather than interpreting silence as
        # conformance.
        if not expected:
            for task in plan.tasks:
                findings.append(
                    TransitionFinding(
                        disposition=TransitionDisposition.SKIPPED,
                        task_id=task.task_id,
                        reason="accepted plan has no observable transition for the task",
                    )
                )

        # Findings are generated in canonical observed/expected order.  Stable
        # de-duplication removes a matched marker only when the same event also
        # received the more specific reordered classification.
        reordered_ids = {
            item.observed_event_id
            for item in findings
            if item.disposition is TransitionDisposition.REORDERED
        }
        findings = [
            item
            for item in findings
            if not (
                item.disposition is TransitionDisposition.MATCHED
                and item.observed_event_id in reordered_ids
            )
        ]

        dispositions = {item.disposition for item in findings}
        violating = {
            TransitionDisposition.REORDERED,
            TransitionDisposition.UNAUTHORIZED,
            TransitionDisposition.FAILED,
        }
        if not policy.allow_overridden:
            violating.add(TransitionDisposition.OVERRIDDEN)
        if not policy.allow_superseded:
            violating.add(TransitionDisposition.SUPERSEDED)
        if invalidations:
            verdict = ConformanceVerdict.INVALIDATED
        elif dispositions & violating:
            verdict = ConformanceVerdict.VIOLATED
        elif TransitionDisposition.SKIPPED in dispositions:
            verdict = ConformanceVerdict.INCOMPLETE
        else:
            verdict = ConformanceVerdict.CONFORMANT

        return PlanConformanceResult(
            plan_id=plan.plan_id,
            binding=binding,
            verdict=verdict,
            findings=tuple(findings),
            expected_event_ids=tuple(item.event_id for item in expected),
            observed_event_ids=tuple(item.event_id for item in observed),
            invalidation_causes=tuple(dict.fromkeys(invalidations)),
        )


def evaluate_plan_conformance(
    plan: FormalWorkPlan | Mapping[str, Any],
    events: Iterable[CanonicalExecutionEvent | Mapping[str, Any]],
    **kwargs: Any,
) -> PlanConformanceResult:
    """Functional entry point for deterministic plan-trace comparison."""

    return FormalPlanConformanceEvaluator().evaluate(plan, events, **kwargs)


compare_plan_conformance = evaluate_plan_conformance
check_plan_conformance = evaluate_plan_conformance


@dataclass(frozen=True)
class FormalCompletionEvidence:
    """One durable result from an independently configured evidence lane."""

    kind: CompletionEvidenceKind
    goal_id: str
    artifact_id: str
    binding: ConformanceBinding
    observed_at: str
    verdict: str = "passed"
    freshness: str = "current"
    expires_at: str = ""
    provider_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _evidence_kind(self.kind))
        object.__setattr__(self, "goal_id", _text(self.goal_id, field_name="goal_id", required=True))
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, field_name="artifact_id"))
        if not isinstance(self.binding, ConformanceBinding):
            object.__setattr__(
                self, "binding", ConformanceBinding.from_dict(self.binding)
            )
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, required=True)
        )
        object.__setattr__(self, "expires_at", _timestamp(self.expires_at))
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict).replace("-", "_")
        )
        object.__setattr__(
            self, "freshness", _enum_value(self.freshness).replace("-", "_")
        )
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, field_name="provider_id")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))
        actual = content_identity(self._identity_payload())
        if self.evidence_id and self.evidence_id != actual:
            raise ConformanceValidationError("completion evidence identity mismatch")
        object.__setattr__(self, "evidence_id", actual)

    @property
    def passed(self) -> bool:
        return self.verdict in _PASS_VERDICTS

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "artifact_id": self.artifact_id,
            "binding": self.binding.to_dict(),
            "observed_at": self.observed_at,
            "verdict": self.verdict,
            "freshness": self.freshness,
            "expires_at": self.expires_at,
            "provider_id": self.provider_id,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPLETION_EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            **self._identity_payload(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalCompletionEvidence":
        binding_payload = payload.get("binding")
        if not binding_payload:
            binding_payload = {
                "plan_id": payload.get("plan_id", ""),
                "policy_id": payload.get("policy_id", ""),
                "repository_tree_id": payload.get(
                    "repository_tree_id", payload.get("tree_id", "")
                ),
                "ast_scope_ids": payload.get(
                    "ast_scope_ids", payload.get("ast_ids", ())
                ),
                "premise_ids": payload.get("premise_ids", ()),
                "counterexample_ids": payload.get("counterexample_ids", ()),
            }
        verdict: Any = payload.get("verdict", payload.get("status", ""))
        if "passed" in payload:
            verdict = "passed" if payload.get("passed") is True else "failed"
        freshness: Any = payload.get(
            "freshness", payload.get("freshness_status", "current")
        )
        if isinstance(freshness, Mapping):
            if freshness.get("invalidated") is True or freshness.get("stale") is True:
                freshness = "invalidated"
            elif freshness.get("fresh") is True or freshness.get("current") is True:
                freshness = "current"
            else:
                freshness = freshness.get("status", "unknown")
        result = cls(
            kind=payload.get(
                "kind", payload.get("evidence_kind", payload.get("lane", ""))
            ),
            goal_id=payload.get(
                "goal_id", payload.get("subject_id", payload.get("objective_id", ""))
            ),
            artifact_id=payload.get(
                "artifact_id",
                payload.get(
                    "receipt_id",
                    payload.get("provenance_cid", payload.get("receipt_cid", "")),
                ),
            ),
            binding=ConformanceBinding.from_dict(binding_payload),
            observed_at=payload.get(
                "observed_at",
                payload.get(
                    "finished_at",
                    payload.get("generated_at", payload.get("created_at", "")),
                ),
            ),
            verdict=verdict or "failed",
            freshness=freshness,
            expires_at=payload.get("expires_at", payload.get("fresh_until", "")),
            provider_id=payload.get(
                "provider_id", payload.get("producer_id", payload.get("verifier_id", ""))
            ),
            metadata=payload.get("metadata") or {},
            evidence_id=payload.get("evidence_id", ""),
        )
        return result

    @classmethod
    def from_json(
        cls, payload: str | bytes | bytearray
    ) -> "FormalCompletionEvidence":
        return cls.from_dict(json.loads(payload))


CompletionEvidenceRecord = FormalCompletionEvidence
FormalEvidence = FormalCompletionEvidence


@dataclass(frozen=True)
class EvidenceCheck:
    kind: CompletionEvidenceKind
    status: EvidenceCheckStatus
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _evidence_kind(self.kind))
        object.__setattr__(self, "status", EvidenceCheckStatus(self.status))
        object.__setattr__(self, "evidence_ids", _strings(self.evidence_ids))
        object.__setattr__(self, "reason", _text(self.reason, field_name="reason"))

    @property
    def satisfied(self) -> bool:
        return self.status is EvidenceCheckStatus.SATISFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "satisfied": self.satisfied,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceCheck":
        return cls(
            kind=payload.get("kind", ""),
            status=payload.get("status", EvidenceCheckStatus.MISSING),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            reason=payload.get("reason", ""),
        )


@dataclass(frozen=True)
class CompletionEvidenceResult:
    policy_id: str
    binding_id: str
    checks: tuple[EvidenceCheck, ...]

    @property
    def satisfied(self) -> bool:
        return bool(self.checks) and all(item.satisfied for item in self.checks)

    @property
    def missing_kinds(self) -> tuple[CompletionEvidenceKind, ...]:
        return tuple(
            item.kind
            for item in self.checks
            if item.status is EvidenceCheckStatus.MISSING
        )

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "policy_id": self.policy_id,
            "binding_id": self.binding_id,
            "satisfied": self.satisfied,
            "checks": [item.to_dict() for item in self.checks],
        }
        if include_id:
            result["result_id"] = self.result_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionEvidenceResult":
        result = cls(
            policy_id=str(payload.get("policy_id", "")),
            binding_id=str(payload.get("binding_id", "")),
            checks=tuple(
                EvidenceCheck.from_dict(item) for item in payload.get("checks", ())
            ),
        )
        if payload.get("result_id") and payload["result_id"] != result.result_id:
            raise ConformanceValidationError("evidence result identity mismatch")
        return result


def evaluate_completion_evidence(
    goal_id: str,
    evidence: Iterable[FormalCompletionEvidence | Mapping[str, Any]],
    *,
    policy: CompletionPolicy,
    binding: ConformanceBinding,
    evaluated_at: datetime | str | int | float,
) -> CompletionEvidenceResult:
    """Evaluate every configured evidence lane independently."""

    now = _timestamp(evaluated_at, required=True)
    now_epoch = _epoch(now)
    normalized = tuple(
        item
        if isinstance(item, FormalCompletionEvidence)
        else FormalCompletionEvidence.from_dict(item)
        for item in evidence
    )
    checks: list[EvidenceCheck] = []
    for kind in policy.required_evidence:
        candidates = [
            item for item in normalized if item.kind is kind and item.goal_id == goal_id
        ]
        if not candidates:
            checks.append(
                EvidenceCheck(
                    kind,
                    EvidenceCheckStatus.MISSING,
                    reason=f"no {kind.value} evidence was supplied for the goal",
                )
            )
            continue

        statuses: list[tuple[EvidenceCheckStatus, str, FormalCompletionEvidence]] = []
        for item in candidates:
            if policy.require_exact_binding and item.binding != binding:
                causes = ", ".join(
                    cause.value for cause in changed_bindings(item.binding, binding)
                )
                statuses.append(
                    (
                        EvidenceCheckStatus.INVALIDATED,
                        f"evidence binding changed: {causes or 'unknown binding mismatch'}",
                        item,
                    )
                )
                continue
            if policy.require_artifact_id and not item.artifact_id:
                statuses.append(
                    (
                        EvidenceCheckStatus.UNBOUND,
                        "evidence has no durable artifact or receipt identity",
                        item,
                    )
                )
                continue
            if not item.passed:
                statuses.append(
                    (
                        EvidenceCheckStatus.FAILED,
                        f"evidence verdict is {item.verdict or 'not passing'}",
                        item,
                    )
                )
                continue
            stale_marker = item.freshness in {
                "expired",
                "invalidated",
                "stale",
                "superseded",
            }
            expired = bool(item.expires_at and _epoch(item.expires_at) < now_epoch)
            max_age = policy.max_age_for(kind)
            too_old = bool(
                max_age is not None
                and now_epoch - _epoch(item.observed_at) > max_age
            )
            future = _epoch(item.observed_at) > now_epoch
            if (
                stale_marker
                or expired
                or too_old
                or future
                or (
                    policy.require_current_freshness
                    and item.freshness not in {"current", "fresh", "valid"}
                )
            ):
                statuses.append(
                    (
                        EvidenceCheckStatus.STALE,
                        "evidence is expired, invalidated, outside its age bound, or not current",
                        item,
                    )
                )
                continue
            statuses.append(
                (EvidenceCheckStatus.SATISFIED, "fresh passing evidence", item)
            )

        passing = [item for item in statuses if item[0] is EvidenceCheckStatus.SATISFIED]
        if passing:
            checks.append(
                EvidenceCheck(
                    kind,
                    EvidenceCheckStatus.SATISFIED,
                    tuple(item.evidence_id for _, _, item in passing),
                    "at least one exactly bound, fresh, passing receipt satisfies the lane",
                )
            )
        else:
            # Prefer the most actionable/highest-trust failure classification.
            priority = {
                EvidenceCheckStatus.INVALIDATED: 0,
                EvidenceCheckStatus.UNBOUND: 1,
                EvidenceCheckStatus.FAILED: 2,
                EvidenceCheckStatus.STALE: 3,
            }
            status, reason, _item = sorted(
                statuses, key=lambda value: priority[value[0]]
            )[0]
            checks.append(
                EvidenceCheck(
                    kind,
                    status,
                    tuple(item.evidence_id for _, _, item in statuses),
                    reason,
                )
            )
    return CompletionEvidenceResult(
        policy_id=policy.policy_id,
        binding_id=binding.binding_id,
        checks=tuple(checks),
    )


@dataclass(frozen=True)
class CompletionAdmissionGate:
    """Proposal/DAG authority boundary evaluated before goal completion."""

    admitted: bool
    proposal_receipt_id: str = ""
    validation_dag_receipt_id: str = ""
    reason_codes: tuple[str, ...] = ()
    validation_policy_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", bool(self.admitted))
        object.__setattr__(
            self,
            "proposal_receipt_id",
            _text(self.proposal_receipt_id, field_name="proposal_receipt_id"),
        )
        object.__setattr__(
            self,
            "validation_dag_receipt_id",
            _text(
                self.validation_dag_receipt_id,
                field_name="validation_dag_receipt_id",
            ),
        )
        object.__setattr__(
            self,
            "validation_policy_id",
            _text(self.validation_policy_id, field_name="validation_policy_id"),
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        if self.admitted and self.reason_codes:
            raise ConformanceValidationError(
                "admitted completion gate cannot contain rejection reasons"
            )
        if not self.admitted and not self.reason_codes:
            raise ConformanceValidationError(
                "rejected completion gate requires a reason"
            )

    @property
    def gate_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "admitted": self.admitted,
            "proposal_receipt_id": self.proposal_receipt_id,
            "validation_dag_receipt_id": self.validation_dag_receipt_id,
            "reason_codes": self.reason_codes,
        }
        if self.validation_policy_id:
            payload["validation_policy_id"] = self.validation_policy_id
        if include_id:
            payload["gate_id"] = self.gate_id
        return payload


def evaluate_completion_admission(
    *,
    proposal_validation: Any = None,
    validation_dag: Any = None,
    required: bool = False,
    expected_validation_policy_id: str = "",
) -> CompletionAdmissionGate:
    """Fail closed when rejected output is offered as completion evidence."""

    reasons: list[str] = []
    proposal_receipt_id = ""
    dag_receipt_id = ""
    validation_policy_id = ""
    expected_validation_policy_id = _text(
        expected_validation_policy_id,
        field_name="expected_validation_policy_id",
    )
    proposal_result = None
    if proposal_validation is None:
        if required:
            reasons.append("proposal_validation_missing")
    else:
        from .proposal_validation import ProposalValidationResult

        proposal_result = (
            proposal_validation
            if isinstance(proposal_validation, ProposalValidationResult)
            else ProposalValidationResult.from_dict(proposal_validation)
        )
        proposal_receipt_id = proposal_result.receipt.receipt_id
        if not proposal_result.accepted:
            reasons.append("proposal_validation_rejected")

    if validation_dag is not None:
        from .validation_scheduler import ValidationDAGReceipt

        dag = (
            validation_dag
            if isinstance(validation_dag, ValidationDAGReceipt)
            else ValidationDAGReceipt.from_dict(validation_dag)
        )
        dag_receipt_id = dag.receipt_id
        validation_policy_id = dag.policy_id
        if proposal_result is None:
            reasons.append("validation_dag_without_proposal")
        elif dag.proposal_receipt_id != proposal_receipt_id:
            reasons.append("validation_dag_proposal_mismatch")
        if proposal_result is not None and (
            dag.repository_tree_id
            != proposal_result.proposal.repository_tree_id
            or dag.objective_id != proposal_result.proposal.objective_id
        ):
            reasons.append("validation_dag_authority_mismatch")
        if (
            expected_validation_policy_id
            and dag.policy_id != expected_validation_policy_id
        ):
            reasons.append("validation_dag_policy_mismatch")
        if not dag.nodes:
            reasons.append("validation_dag_empty")
        if dag.uncovered_impact:
            reasons.append("validation_dag_uncovered_impact")
        if getattr(dag, "coverage_complete", None) is False:
            reasons.append("validation_dag_incomplete")
        if not dag.passed:
            reasons.append("validation_dag_failed")
    elif required or expected_validation_policy_id:
        reasons.append("validation_dag_missing")

    return CompletionAdmissionGate(
        admitted=not reasons,
        proposal_receipt_id=proposal_receipt_id,
        validation_dag_receipt_id=dag_receipt_id,
        validation_policy_id=validation_policy_id,
        reason_codes=tuple(reasons),
    )


@dataclass(frozen=True)
class FormalGoalCompletionDecision:
    goal_id: str
    previous_state: GoalState
    state: GoalState
    conformance: PlanConformanceResult
    evidence_result: CompletionEvidenceResult
    evaluated_at: str
    reason_codes: tuple[str, ...] = ()
    plan_consistency: str = ""
    completion_admission: CompletionAdmissionGate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, field_name="goal_id", required=True))
        object.__setattr__(self, "previous_state", normalize_goal_state(self.previous_state))
        object.__setattr__(self, "state", normalize_goal_state(self.state))
        if not isinstance(self.conformance, PlanConformanceResult):
            object.__setattr__(
                self,
                "conformance",
                PlanConformanceResult.from_dict(self.conformance),
            )
        if not isinstance(self.evidence_result, CompletionEvidenceResult):
            object.__setattr__(
                self,
                "evidence_result",
                CompletionEvidenceResult.from_dict(self.evidence_result),
            )
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, required=True)
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        object.__setattr__(
            self,
            "plan_consistency",
            _enum_value(self.plan_consistency),
        )
        if self.completion_admission is not None and not isinstance(
            self.completion_admission, CompletionAdmissionGate
        ):
            object.__setattr__(
                self,
                "completion_admission",
                CompletionAdmissionGate(
                    admitted=self.completion_admission.get("admitted", False),
                    proposal_receipt_id=self.completion_admission.get(
                        "proposal_receipt_id", ""
                    ),
                    validation_dag_receipt_id=self.completion_admission.get(
                        "validation_dag_receipt_id", ""
                    ),
                    validation_policy_id=self.completion_admission.get(
                        "validation_policy_id", ""
                    ),
                    reason_codes=tuple(
                        self.completion_admission.get("reason_codes") or ()
                    ),
                ),
            )

    @property
    def verified(self) -> bool:
        return self.state is GoalState.VERIFIED_COMPLETE

    @property
    def closeable(self) -> bool:
        return self.verified

    @property
    def reopened(self) -> bool:
        return self.state is GoalState.REOPENED

    @property
    def decision_id(self) -> str:
        # Evaluation time is observational metadata.  It does not change the
        # semantic replay verdict when the same packet is evaluated again.
        return content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            "goal_id": self.goal_id,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "conformance_receipt_id": self.conformance.receipt_id,
            "evidence_result_id": self.evidence_result.result_id,
            "reason_codes": list(self.reason_codes),
            "plan_consistency": self.plan_consistency,
        }
        if self.completion_admission is not None:
            payload["completion_admission_gate_id"] = (
                self.completion_admission.gate_id
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_COMPLETION_SCHEMA,
            "version": FORMAL_PLAN_CONFORMANCE_VERSION,
            "decision_id": self.decision_id,
            **self._identity_payload(),
            "evaluated_at": self.evaluated_at,
            "verified": self.verified,
            "closeable": self.closeable,
            "reopened": self.reopened,
            "conformance": self.conformance.to_dict(),
            "evidence_result": self.evidence_result.to_dict(),
            "completion_admission": (
                self.completion_admission.to_dict()
                if self.completion_admission is not None
                else None
            ),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalGoalCompletionDecision":
        result = cls(
            goal_id=payload.get("goal_id", ""),
            previous_state=payload.get("previous_state", GoalState.ACTIVE),
            state=payload.get("state", GoalState.PROVISIONALLY_COMPLETE),
            conformance=PlanConformanceResult.from_dict(
                payload.get("conformance") or {}
            ),
            evidence_result=CompletionEvidenceResult.from_dict(
                payload.get("evidence_result") or {}
            ),
            evaluated_at=payload.get("evaluated_at", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            plan_consistency=payload.get("plan_consistency", ""),
            completion_admission=payload.get("completion_admission"),
        )
        if payload.get("decision_id") and payload["decision_id"] != result.decision_id:
            raise ConformanceValidationError("goal completion decision identity mismatch")
        return result

    @classmethod
    def from_json(
        cls, payload: str | bytes | bytearray
    ) -> "FormalGoalCompletionDecision":
        return cls.from_dict(json.loads(payload))


GoalCompletionConformanceDecision = FormalGoalCompletionDecision


def evaluate_formal_goal_completion(
    goal_id: str,
    plan: FormalWorkPlan | Mapping[str, Any],
    events: Iterable[CanonicalExecutionEvent | Mapping[str, Any]],
    evidence: Iterable[FormalCompletionEvidence | Mapping[str, Any]],
    *,
    policy: CompletionPolicy | Mapping[str, Any] | None = None,
    binding: ConformanceBinding | Mapping[str, Any] | None = None,
    previous_state: GoalState | str = GoalState.ACTIVE,
    prior_conformance: PlanConformanceResult | Mapping[str, Any] | None = None,
    evaluated_at: datetime | str | int | float | None = None,
    repository_tree_id: str | None = None,
    ast_scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    counterexample_ids: Sequence[str] = (),
    plan_consistency: Any = "",
    proposal_validation: Any = None,
    validation_dag: Any = None,
    require_proposal_validation: bool = False,
    expected_validation_policy_id: str = "",
) -> FormalGoalCompletionDecision:
    """Bind trace conformance and independent evidence into goal completion.

    ``plan_consistency`` is retained for diagnostics only.  It deliberately
    cannot satisfy any completion lane or compensate for a missing receipt.
    """

    if not isinstance(plan, FormalWorkPlan):
        plan = FormalWorkPlan.from_dict(plan)
    if policy is None:
        policy = CompletionPolicy()
    elif not isinstance(policy, CompletionPolicy):
        policy = CompletionPolicy.from_dict(policy)
    if binding is None:
        binding = binding_for_plan(
            plan,
            policy,
            repository_tree_id=repository_tree_id,
            ast_scope_ids=ast_scope_ids,
            premise_ids=premise_ids,
            counterexample_ids=counterexample_ids,
        )
    elif not isinstance(binding, ConformanceBinding):
        binding = ConformanceBinding.from_dict(binding)
    evaluation_time = _timestamp(
        evaluated_at if evaluated_at is not None else datetime.now(timezone.utc),
        required=True,
    )
    previous = normalize_goal_state(previous_state)
    conformance = evaluate_plan_conformance(
        plan,
        events,
        policy=policy,
        binding=binding,
        prior=prior_conformance,
        repository_tree_id=repository_tree_id,
    )
    evidence_result = evaluate_completion_evidence(
        goal_id,
        evidence,
        policy=policy,
        binding=conformance.binding,
        evaluated_at=evaluation_time,
    )
    reasons: list[str] = []
    if not conformance.conformant:
        reasons.append(f"plan_conformance_{conformance.verdict.value}")
    for cause in conformance.invalidation_causes:
        reasons.append(cause.value)
    for check in evidence_result.checks:
        if not check.satisfied:
            reasons.append(f"{check.kind.value}_evidence_{check.status.value}")
    admission = (
        evaluate_completion_admission(
            proposal_validation=proposal_validation,
            validation_dag=validation_dag,
            required=require_proposal_validation,
            expected_validation_policy_id=expected_validation_policy_id,
        )
        if (
            proposal_validation is not None
            or validation_dag is not None
            or require_proposal_validation
            or expected_validation_policy_id
        )
        else None
    )
    if admission is not None:
        reasons.extend(admission.reason_codes)

    if (
        conformance.conformant
        and evidence_result.satisfied
        and (admission is None or admission.admitted)
    ):
        state = GoalState.VERIFIED_COMPLETE
    elif previous is GoalState.VERIFIED_COMPLETE:
        state = GoalState.REOPENED
        reasons.append("verified_goal_evidence_regressed")
    else:
        state = GoalState.PROVISIONALLY_COMPLETE

    return FormalGoalCompletionDecision(
        goal_id=goal_id,
        previous_state=previous,
        state=state,
        conformance=conformance,
        evidence_result=evidence_result,
        evaluated_at=evaluation_time,
        reason_codes=tuple(reasons),
        plan_consistency=_enum_value(plan_consistency),
        completion_admission=admission,
    )


bind_goal_completion = evaluate_formal_goal_completion
evaluate_goal_completion_with_conformance = evaluate_formal_goal_completion


@dataclass(frozen=True)
class ConformanceReplayPacket:
    """Complete deterministic input and output for restart-safe replay."""

    goal_id: str
    plan: FormalWorkPlan
    events: tuple[CanonicalExecutionEvent, ...]
    evidence: tuple[FormalCompletionEvidence, ...]
    policy: CompletionPolicy
    binding: ConformanceBinding
    previous_state: GoalState
    evaluated_at: str
    prior_conformance: PlanConformanceResult | None = None
    plan_consistency: str = ""
    stored_decision: FormalGoalCompletionDecision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, field_name="goal_id", required=True))
        if not isinstance(self.plan, FormalWorkPlan):
            object.__setattr__(self, "plan", FormalWorkPlan.from_dict(self.plan))
        object.__setattr__(self, "events", _canonical_events(self.events))
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    (
                        item
                        if isinstance(item, FormalCompletionEvidence)
                        else FormalCompletionEvidence.from_dict(item)
                        for item in self.evidence
                    ),
                    key=lambda item: item.evidence_id,
                )
            ),
        )
        if not isinstance(self.policy, CompletionPolicy):
            object.__setattr__(self, "policy", CompletionPolicy.from_dict(self.policy))
        if not isinstance(self.binding, ConformanceBinding):
            object.__setattr__(self, "binding", ConformanceBinding.from_dict(self.binding))
        object.__setattr__(self, "previous_state", normalize_goal_state(self.previous_state))
        object.__setattr__(self, "evaluated_at", _timestamp(self.evaluated_at, required=True))
        if self.prior_conformance is not None and not isinstance(
            self.prior_conformance, PlanConformanceResult
        ):
            object.__setattr__(
                self,
                "prior_conformance",
                PlanConformanceResult.from_dict(self.prior_conformance),
            )
        if self.stored_decision is not None and not isinstance(
            self.stored_decision, FormalGoalCompletionDecision
        ):
            object.__setattr__(
                self,
                "stored_decision",
                FormalGoalCompletionDecision.from_dict(self.stored_decision),
            )
        object.__setattr__(self, "plan_consistency", _enum_value(self.plan_consistency))

    @property
    def packet_id(self) -> str:
        return content_identity(self._identity_payload())

    def evaluate(self) -> FormalGoalCompletionDecision:
        return evaluate_formal_goal_completion(
            self.goal_id,
            self.plan,
            self.events,
            self.evidence,
            policy=self.policy,
            binding=self.binding,
            previous_state=self.previous_state,
            prior_conformance=self.prior_conformance,
            evaluated_at=self.evaluated_at,
            plan_consistency=self.plan_consistency,
        )

    def replay(self, *, verify_stored: bool = True) -> FormalGoalCompletionDecision:
        result = self.evaluate()
        if (
            verify_stored
            and self.stored_decision is not None
            and result.decision_id != self.stored_decision.decision_id
        ):
            raise ConformanceValidationError(
                "replayed conformance verdict differs from stored decision"
            )
        return result

    def with_decision(
        self, decision: FormalGoalCompletionDecision | None = None
    ) -> "ConformanceReplayPacket":
        return ConformanceReplayPacket(
            goal_id=self.goal_id,
            plan=self.plan,
            events=self.events,
            evidence=self.evidence,
            policy=self.policy,
            binding=self.binding,
            previous_state=self.previous_state,
            evaluated_at=self.evaluated_at,
            prior_conformance=self.prior_conformance,
            plan_consistency=self.plan_consistency,
            stored_decision=decision or self.evaluate(),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "plan_id": self.plan.plan_id,
            "event_ids": [item.execution_event_id for item in self.events],
            "evidence_ids": [item.evidence_id for item in self.evidence],
            "policy_id": self.policy.policy_id,
            "binding_id": self.binding.binding_id,
            "previous_state": self.previous_state.value,
            "evaluated_at": self.evaluated_at,
            "prior_conformance_receipt_id": (
                self.prior_conformance.receipt_id if self.prior_conformance else ""
            ),
            "plan_consistency": self.plan_consistency,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFORMANCE_REPLAY_SCHEMA,
            "version": FORMAL_PLAN_CONFORMANCE_VERSION,
            "packet_id": self.packet_id,
            **self._identity_payload(),
            "plan": self.plan.to_record(),
            "events": [item.to_dict() for item in self.events],
            "evidence": [item.to_dict() for item in self.evidence],
            "policy": self.policy.to_dict(),
            "binding": self.binding.to_dict(),
            "prior_conformance": (
                self.prior_conformance.to_dict() if self.prior_conformance else None
            ),
            "stored_decision": (
                self.stored_decision.to_dict() if self.stored_decision else None
            ),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceReplayPacket":
        result = cls(
            goal_id=payload.get("goal_id", ""),
            plan=FormalWorkPlan.from_dict(payload.get("plan") or {}),
            events=tuple(
                CanonicalExecutionEvent.from_dict(item, fallback_sequence=index)
                for index, item in enumerate(payload.get("events", ()))
            ),
            evidence=tuple(
                FormalCompletionEvidence.from_dict(item)
                for item in payload.get("evidence", ())
            ),
            policy=CompletionPolicy.from_dict(payload.get("policy") or {}),
            binding=ConformanceBinding.from_dict(payload.get("binding") or {}),
            previous_state=payload.get("previous_state", GoalState.ACTIVE),
            evaluated_at=payload.get("evaluated_at", ""),
            prior_conformance=(
                PlanConformanceResult.from_dict(payload["prior_conformance"])
                if payload.get("prior_conformance")
                else None
            ),
            plan_consistency=payload.get("plan_consistency", ""),
            stored_decision=(
                FormalGoalCompletionDecision.from_dict(payload["stored_decision"])
                if payload.get("stored_decision")
                else None
            ),
        )
        if payload.get("packet_id") and payload["packet_id"] != result.packet_id:
            raise ConformanceValidationError("conformance replay packet identity mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ConformanceReplayPacket":
        return cls.from_dict(json.loads(payload))


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_conformance_evidence(
    path: Path | str,
    packet: ConformanceReplayPacket | Mapping[str, Any],
    *,
    include_decision: bool = True,
) -> Path:
    """Persist a replay packet to canonical JSON or a normalized DuckDB store."""

    if not isinstance(packet, ConformanceReplayPacket):
        packet = ConformanceReplayPacket.from_dict(packet)
    if include_decision and packet.stored_decision is None:
        packet = packet.with_decision()
    target = Path(path)
    if target.suffix.lower() not in {".duckdb", ".db"}:
        _atomic_write(target, packet.to_json())
        return target
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("duckdb is required for DuckDB conformance evidence") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(target))
    try:
        connection.execute("BEGIN TRANSACTION")
        for table in (
            "formal_conformance_events",
            "formal_completion_evidence",
            "formal_conformance_packets",
        ):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute(
            "CREATE TABLE formal_conformance_packets ("
            "packet_id VARCHAR PRIMARY KEY, goal_id VARCHAR, plan_id VARCHAR, "
            "policy_id VARCHAR, binding_id VARCHAR, verdict VARCHAR, payload_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE formal_conformance_events ("
            "execution_event_id VARCHAR PRIMARY KEY, packet_id VARCHAR, sequence BIGINT, "
            "event_id VARCHAR, task_id VARCHAR, kind VARCHAR, disposition VARCHAR, payload_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE formal_completion_evidence ("
            "evidence_id VARCHAR PRIMARY KEY, packet_id VARCHAR, kind VARCHAR, goal_id VARCHAR, "
            "artifact_id VARCHAR, freshness VARCHAR, verdict VARCHAR, payload_json VARCHAR)"
        )
        decision = packet.stored_decision or packet.evaluate()
        connection.execute(
            "INSERT INTO formal_conformance_packets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                packet.packet_id,
                packet.goal_id,
                packet.plan.plan_id,
                packet.policy.policy_id,
                packet.binding.binding_id,
                decision.conformance.verdict.value,
                packet.to_json(),
            ),
        )
        dispositions: dict[str, str] = {}
        for finding in decision.conformance.findings:
            if finding.observed_event_id:
                dispositions[finding.observed_event_id] = finding.disposition.value
        if packet.events:
            connection.executemany(
                "INSERT INTO formal_conformance_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.execution_event_id,
                        packet.packet_id,
                        item.sequence,
                        item.event_id,
                        item.task_id,
                        item.kind,
                        dispositions.get(item.event_id, ""),
                        canonical_json(item.to_dict()),
                    )
                    for item in packet.events
                ],
            )
        if packet.evidence:
            connection.executemany(
                "INSERT INTO formal_completion_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.evidence_id,
                        packet.packet_id,
                        item.kind.value,
                        item.goal_id,
                        item.artifact_id,
                        item.freshness,
                        item.verdict,
                        canonical_json(item.to_dict()),
                    )
                    for item in packet.evidence
                ],
            )
        connection.execute("COMMIT")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        connection.close()
    return target


def read_conformance_evidence(path: Path | str) -> ConformanceReplayPacket:
    """Load an exact replay packet from JSON or DuckDB evidence."""

    source = Path(path)
    if source.suffix.lower() not in {".duckdb", ".db"}:
        return ConformanceReplayPacket.from_json(source.read_text(encoding="utf-8"))
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("duckdb is required for DuckDB conformance evidence") from exc
    connection = duckdb.connect(str(source), read_only=True)
    try:
        rows = connection.execute(
            "SELECT packet_id, goal_id, plan_id, policy_id, binding_id, verdict, "
            "payload_json FROM formal_conformance_packets ORDER BY packet_id"
        ).fetchall()
        event_rows = connection.execute(
            "SELECT payload_json FROM formal_conformance_events "
            "ORDER BY sequence, event_id"
        ).fetchall()
        evidence_rows = connection.execute(
            "SELECT payload_json FROM formal_completion_evidence "
            "ORDER BY evidence_id"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise ConformanceValidationError(
            f"expected exactly one conformance packet, found {len(rows)}"
        )
    packet = ConformanceReplayPacket.from_json(rows[0][6])
    indexed = rows[0]
    expected_index = (
        packet.packet_id,
        packet.goal_id,
        packet.plan.plan_id,
        packet.policy.policy_id,
        packet.binding.binding_id,
        (
            (packet.stored_decision or packet.evaluate()).conformance.verdict.value
        ),
    )
    if tuple(indexed[:6]) != expected_index:
        raise ConformanceValidationError(
            "DuckDB conformance packet indexes do not match canonical payload"
        )
    stored_events = tuple(
        CanonicalExecutionEvent.from_json(row[0]) for row in event_rows
    )
    stored_evidence = tuple(
        FormalCompletionEvidence.from_json(row[0]) for row in evidence_rows
    )
    if (
        tuple(item.execution_event_id for item in stored_events)
        != tuple(item.execution_event_id for item in packet.events)
        or tuple(item.evidence_id for item in stored_evidence)
        != tuple(item.evidence_id for item in packet.evidence)
    ):
        raise ConformanceValidationError(
            "DuckDB conformance projections do not match canonical payload"
        )
    return packet


def replay_conformance_evidence(
    source: Path | str | ConformanceReplayPacket | Mapping[str, Any],
    *,
    verify_stored: bool = True,
) -> FormalGoalCompletionDecision:
    """Recompute a decision from persisted evidence and verify its identity."""

    if isinstance(source, (str, Path)):
        packet = read_conformance_evidence(source)
    elif isinstance(source, ConformanceReplayPacket):
        packet = source
    else:
        packet = ConformanceReplayPacket.from_dict(source)
    return packet.replay(verify_stored=verify_stored)


class PlanConformanceEvidenceStore:
    """Small path-bound facade for restart-safe conformance evidence."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def write(
        self,
        packet: ConformanceReplayPacket | Mapping[str, Any],
        *,
        include_decision: bool = True,
    ) -> Path:
        return write_conformance_evidence(
            self.path, packet, include_decision=include_decision
        )

    save = write

    def read(self) -> ConformanceReplayPacket:
        return read_conformance_evidence(self.path)

    load = read

    def replay(
        self, *, verify_stored: bool = True
    ) -> FormalGoalCompletionDecision:
        return replay_conformance_evidence(
            self.path, verify_stored=verify_stored
        )


replay_plan_conformance = replay_conformance_evidence
write_formal_plan_conformance = write_conformance_evidence
read_formal_plan_conformance = read_conformance_evidence
PlanConformanceReplayPacket = ConformanceReplayPacket
FormalPlanConformanceResult = PlanConformanceResult
FormalPlanCompletionPolicy = CompletionPolicy
FormalPlanConformanceStore = PlanConformanceEvidenceStore
PlanConformancePolicy = CompletionPolicy
ConformanceStatus = ConformanceVerdict
ConformanceIssueKind = TransitionDisposition
EvidenceType = CompletionEvidenceKind
evaluate_goal_completion = evaluate_formal_goal_completion


__all__ = [
    "CONFORMANCE_BINDING_SCHEMA",
    "CONFORMANCE_REPLAY_SCHEMA",
    "COMPLETION_EVIDENCE_SCHEMA",
    "COMPLETION_POLICY_SCHEMA",
    "EXECUTION_EVENT_SCHEMA",
    "FORMAL_PLAN_CONFORMANCE_SCHEMA",
    "FORMAL_PLAN_CONFORMANCE_VERSION",
    "GOAL_COMPLETION_SCHEMA",
    "CanonicalExecutionEvent",
    "CompletionEvidenceKind",
    "CompletionEvidenceRecord",
    "CompletionEvidenceResult",
    "CompletionAdmissionGate",
    "CompletionPolicy",
    "ConformanceBinding",
    "ConformanceIssueKind",
    "ConformanceReplayPacket",
    "ConformanceStatus",
    "ConformanceValidationError",
    "ConformanceVerdict",
    "EvidenceCheck",
    "EvidenceCheckStatus",
    "EvidenceType",
    "ExecutionEvent",
    "FormalCompletionEvidence",
    "FormalEvidence",
    "FormalEvidenceKind",
    "FormalGoalCompletionDecision",
    "FormalPlanCompletionPolicy",
    "FormalPlanConformanceEvaluator",
    "FormalPlanConformanceResult",
    "FormalPlanConformanceStore",
    "GoalCompletionConformanceDecision",
    "InvalidationCause",
    "PlanConformanceReplayPacket",
    "PlanConformanceEvidenceStore",
    "PlanConformancePolicy",
    "PlanConformanceResult",
    "TransitionDisposition",
    "TransitionFinding",
    "bind_goal_completion",
    "binding_for_plan",
    "changed_bindings",
    "check_plan_conformance",
    "compare_plan_conformance",
    "evaluate_completion_evidence",
    "evaluate_completion_admission",
    "evaluate_formal_goal_completion",
    "evaluate_goal_completion",
    "evaluate_goal_completion_with_conformance",
    "evaluate_plan_conformance",
    "invalidate_plan_conformance",
    "read_conformance_evidence",
    "read_formal_plan_conformance",
    "replay_conformance_evidence",
    "replay_plan_conformance",
    "write_conformance_evidence",
    "write_formal_plan_conformance",
]
