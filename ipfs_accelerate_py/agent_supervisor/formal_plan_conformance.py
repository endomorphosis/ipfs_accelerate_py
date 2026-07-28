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
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_planning_contracts import (
    EventKind,
    FormalWorkPlan,
    PlanConformanceLevel,
    PlanEvent,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    ProofReceipt,
    assurance_satisfies,
    canonical_json,
    content_identity,
    derive_assurance,
)
from .goal_completion import GoalState, normalize_goal_state


FORMAL_PLAN_CONFORMANCE_VERSION: Final = 1
FORMAL_PLAN_CONFORMANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-conformance@1"
)
REQUIRES_PROOF_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-requires-proof-admission@1"
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
POST_MERGE_COMPLETION_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/post-merge-completion-admission@1"
)
CONFORMANCE_REPLAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-conformance-replay@1"
)
STRICT_VALIDATION_OBJECTIVE_ID: Final = "ASI-G040"
STRICT_VALIDATION_OBJECTIVE_REVISION: Final = "ASI-G040@asi-089"
STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION: Final = (
    "strict-validation-completion@1"
)
STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION: Final = (
    "strict-validation-completion-policy@1"
)
STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS: Final = 2
STRICT_VALIDATION_PRODUCING_TASK_IDS: Final[tuple[str, ...]] = (
    "ASI-010",
    "ASI-011",
    "ASI-012",
)
STRICT_VALIDATION_CHILD_GOAL_IDS: Final[tuple[str, ...]] = (
    "ASI-G100",
    "ASI-G101",
    "ASI-G102",
)
STRICT_VALIDATION_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "Schema, authority, patch, path, AST/interface, impact-test, "
        "semantic/proof, merge, and freshness gates are explicit. Validation "
        "declarations bind canonical impact targets, DAG dependencies, and "
        "downstream authority gates"
    ),
    (
        "the receipt covers the complete selected population and schedules "
        "only dependency-ready checks under bounded parallelism. No required "
        "gate may be omitted, seeded adversarial defects do not escape, and "
        "failed output yields bounded typed diagnostics while closing proof, "
        "merge, freshness, and completion authority."
    ),
)
STRICT_VALIDATION_GATE_KINDS: Final[tuple[str, ...]] = (
    "schema",
    "authority",
    "patch",
    "path",
    "ast_interface",
    "impact_test",
    "semantic_proof",
    "merge",
    "freshness",
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
    GOAL_CHANGED = "goal_changed"
    PLAN_CHANGED = "plan_changed"
    POLICY_CHANGED = "policy_changed"
    REPOSITORY_TREE_CHANGED = "repository_tree_changed"
    AST_CHANGED = "ast_changed"
    PREMISE_CHANGED = "premise_changed"
    COUNTEREXAMPLE_CHANGED = "counterexample_changed"
    TOOLCHAIN_CHANGED = "toolchain_changed"


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
    goal_id: str = ""
    toolchain_id: str = ""
    ast_scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("plan_id", "policy_id", "repository_tree_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        for name in ("goal_id", "toolchain_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
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
            "goal_id": self.goal_id,
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "toolchain_id": self.toolchain_id,
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
            goal_id=payload.get("goal_id", payload.get("objective_id", "")),
            plan_id=payload.get("plan_id", ""),
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get(
                "repository_tree_id", payload.get("tree_id", "")
            ),
            toolchain_id=payload.get("toolchain_id", ""),
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
    goal_id: str = "",
    toolchain_id: str = "",
    ast_scope_ids: Sequence[str] = (),
    premise_ids: Sequence[str] = (),
    counterexample_ids: Sequence[str] = (),
) -> ConformanceBinding:
    """Create the exact semantic binding used by a completion evaluation."""

    return ConformanceBinding(
        goal_id=(
            goal_id
            or (
                plan.goals[0].goal_id
                if len(plan.goals) == 1
                else ""
            )
        ),
        plan_id=plan.plan_id,
        policy_id=policy.policy_id,
        repository_tree_id=repository_tree_id or plan.repository_tree_id,
        toolchain_id=(
            toolchain_id
            or str(policy.metadata.get("toolchain_id") or "").strip()
        ),
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
        ("goal_id", InvalidationCause.GOAL_CHANGED),
        ("plan_id", InvalidationCause.PLAN_CHANGED),
        ("policy_id", InvalidationCause.POLICY_CHANGED),
        ("repository_tree_id", InvalidationCause.REPOSITORY_TREE_CHANGED),
        ("ast_scope_ids", InvalidationCause.AST_CHANGED),
        ("premise_ids", InvalidationCause.PREMISE_CHANGED),
        ("counterexample_ids", InvalidationCause.COUNTEREXAMPLE_CHANGED),
        ("toolchain_id", InvalidationCause.TOOLCHAIN_CHANGED),
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
        goal_id: str = "",
        toolchain_id: str = "",
        ast_scope_ids: Sequence[str] | None = None,
        premise_ids: Sequence[str] | None = None,
        counterexample_ids: Sequence[str] | None = None,
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
                goal_id=goal_id,
                toolchain_id=toolchain_id,
                ast_scope_ids=ast_scope_ids or (),
                premise_ids=premise_ids or (),
                counterexample_ids=counterexample_ids or (),
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
            goal_id=goal_id or binding.goal_id,
            toolchain_id=toolchain_id or binding.toolchain_id,
            ast_scope_ids=(
                tuple(ast_scope_ids)
                if ast_scope_ids is not None
                else binding.ast_scope_ids
            ),
            premise_ids=(
                tuple(premise_ids)
                if premise_ids is not None
                else binding.premise_ids
            ),
            counterexample_ids=(
                tuple(counterexample_ids)
                if counterexample_ids is not None
                else binding.counterexample_ids
            ),
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

        accepted_goal_ids = {item.goal_id for item in plan.goals}
        if binding.goal_id and binding.goal_id not in accepted_goal_ids:
            add(
                TransitionDisposition.UNAUTHORIZED,
                reason="conformance binding names a goal outside the accepted plan",
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
                authorized_override = (
                    planned is not None
                    and actual.plan_id == binding.plan_id
                    and actual.repository_tree_id == binding.repository_tree_id
                    and actual.task_id == planned.task_id
                    and actual.actor_id == planned.actor_id
                    and actual.actor_id in task_actors.get(planned.task_id, set())
                    and actual.authorized is not False
                )
                if not authorized_override:
                    add(
                        TransitionDisposition.UNAUTHORIZED,
                        actual,
                        planned,
                        "override is not exactly bound to an authorized accepted transition",
                        observed_position,
                    )
                    continue
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
                authorized_supersede = (
                    planned is not None
                    and actual.plan_id == binding.plan_id
                    and actual.repository_tree_id == binding.repository_tree_id
                    and actual.task_id == planned.task_id
                    and actual.actor_id == planned.actor_id
                    and actual.actor_id in task_actors.get(planned.task_id, set())
                    and actual.authorized is not False
                )
                if not authorized_supersede:
                    add(
                        TransitionDisposition.UNAUTHORIZED,
                        actual,
                        planned,
                        "supersession is not exactly bound to an authorized accepted transition",
                        observed_position,
                    )
                    continue
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

            if planned is not None and actual.task_id != planned.task_id:
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    "execution event task does not match the accepted transition",
                    observed_position,
                )
                continue
            if (
                planned is not None
                and actual.kind != planned.kind.value
                and actual.kind != EventKind.FAILED.value
                and actual_status not in _FAIL_VERDICTS
            ):
                add(
                    TransitionDisposition.UNAUTHORIZED,
                    actual,
                    planned,
                    "execution event kind does not match the accepted transition",
                    observed_position,
                )
                continue
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

        passing = [
            item
            for item in statuses
            if item[0] is EvidenceCheckStatus.SATISFIED
        ]
        rejected = [
            item
            for item in statuses
            if item[0] is not EvidenceCheckStatus.SATISFIED
        ]
        if passing and not rejected:
            checks.append(
                EvidenceCheck(
                    kind,
                    EvidenceCheckStatus.SATISFIED,
                    tuple(item.evidence_id for _, _, item in passing),
                    "every submitted receipt is exactly bound, fresh, and passing",
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
                rejected or statuses,
                key=lambda value: priority[value[0]],
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
    code_proof_result_ids: tuple[str, ...] = ()
    proof_candidate_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.admitted, bool):
            raise ConformanceValidationError("admitted must be boolean")
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
        object.__setattr__(
            self, "code_proof_result_ids", _strings(self.code_proof_result_ids)
        )
        object.__setattr__(
            self,
            "proof_candidate_receipt_ids",
            _strings(self.proof_candidate_receipt_ids),
        )
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
        if self.code_proof_result_ids:
            payload["code_proof_result_ids"] = self.code_proof_result_ids
        if self.proof_candidate_receipt_ids:
            payload["proof_candidate_receipt_ids"] = (
                self.proof_candidate_receipt_ids
            )
        if include_id:
            payload["gate_id"] = self.gate_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionAdmissionGate":
        result = cls(
            admitted=payload.get("admitted", False),
            proposal_receipt_id=str(payload.get("proposal_receipt_id") or ""),
            validation_dag_receipt_id=str(
                payload.get("validation_dag_receipt_id") or ""
            ),
            validation_policy_id=str(
                payload.get("validation_policy_id") or ""
            ),
            code_proof_result_ids=tuple(
                payload.get("code_proof_result_ids") or ()
            ),
            proof_candidate_receipt_ids=tuple(
                payload.get("proof_candidate_receipt_ids") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("gate_id") and payload["gate_id"] != result.gate_id:
            raise ConformanceValidationError(
                "completion admission gate identity mismatch"
            )
        return result


@dataclass(frozen=True)
class PostMergeCompletionAdmissionGate:
    """Re-derived authority projection for one merged-tree evidence receipt.

    This gate deliberately stores the authoritative aggregate receipt identity
    rather than copying its component verdicts.  Callers must construct it
    through :func:`evaluate_post_merge_completion_admission`, which replays the
    receipt against the current repository tree before projecting authority.
    """

    admitted: bool
    post_merge_evidence_receipt_id: str = ""
    revalidated_receipt_id: str = ""
    evidence_graph_id: str = ""
    repository_id: str = ""
    merged_tree_id: str = ""
    merge_commit_id: str = ""
    covered_acceptance_criteria: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.admitted, bool):
            raise ConformanceValidationError("admitted must be boolean")
        for name in (
            "post_merge_evidence_receipt_id",
            "revalidated_receipt_id",
            "evidence_graph_id",
            "repository_id",
            "merged_tree_id",
            "merge_commit_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "covered_acceptance_criteria",
            _strings(self.covered_acceptance_criteria),
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        if self.admitted and self.reason_codes:
            raise ConformanceValidationError(
                "admitted post-merge gate cannot contain rejection reasons"
            )
        if not self.admitted and not self.reason_codes:
            raise ConformanceValidationError(
                "rejected post-merge gate requires a reason"
            )

    @property
    def receipt_id(self) -> str:
        """Compatibility spelling for the aggregate receipt identity."""

        return self.post_merge_evidence_receipt_id

    @property
    def graph_id(self) -> str:
        """Compatibility spelling for the rebuilt graph identity."""

        return self.evidence_graph_id

    @property
    def gate_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": POST_MERGE_COMPLETION_ADMISSION_SCHEMA,
            "admitted": self.admitted,
            "post_merge_evidence_receipt_id": (
                self.post_merge_evidence_receipt_id
            ),
            "revalidated_receipt_id": self.revalidated_receipt_id,
            "evidence_graph_id": self.evidence_graph_id,
            "repository_id": self.repository_id,
            "merged_tree_id": self.merged_tree_id,
            "merge_commit_id": self.merge_commit_id,
            "covered_acceptance_criteria": (
                self.covered_acceptance_criteria
            ),
            "reason_codes": self.reason_codes,
        }
        if include_id:
            payload["gate_id"] = self.gate_id
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PostMergeCompletionAdmissionGate":
        if not isinstance(payload, Mapping):
            raise ConformanceValidationError(
                "post-merge completion admission must be a mapping"
            )
        schema = str(
            payload.get("schema") or POST_MERGE_COMPLETION_ADMISSION_SCHEMA
        )
        if schema != POST_MERGE_COMPLETION_ADMISSION_SCHEMA:
            raise ConformanceValidationError(
                f"unsupported post-merge completion admission schema: {schema}"
            )
        result = cls(
            admitted=payload.get("admitted", False),
            post_merge_evidence_receipt_id=str(
                payload.get("post_merge_evidence_receipt_id")
                or payload.get("receipt_id")
                or ""
            ),
            revalidated_receipt_id=str(
                payload.get("revalidated_receipt_id") or ""
            ),
            evidence_graph_id=str(
                payload.get("evidence_graph_id")
                or payload.get("graph_id")
                or ""
            ),
            repository_id=str(payload.get("repository_id") or ""),
            merged_tree_id=str(
                payload.get("merged_tree_id")
                or payload.get("repository_tree_id")
                or ""
            ),
            merge_commit_id=str(
                payload.get("merge_commit_id")
                or payload.get("merge_commit")
                or ""
            ),
            covered_acceptance_criteria=tuple(
                payload.get("covered_acceptance_criteria")
                or payload.get("acceptance_criteria")
                or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("gate_id") and payload["gate_id"] != result.gate_id:
            raise ConformanceValidationError(
                "post-merge completion admission identity mismatch"
            )
        return result


def evaluate_post_merge_completion_admission(
    post_merge_evidence: Any = None,
    *,
    current_repository_tree_id: str,
    expected_repository_id: str = "",
    expected_merge_commit_id: str = "",
    expected_evidence_graph_id: str = "",
    expected_acceptance_criteria: Sequence[str] | None = None,
    now: datetime | str | None = None,
) -> PostMergeCompletionAdmissionGate:
    """Replay and bind the sole merged-tree completion authority boundary.

    The evidence assembler owns the detailed proposal, validation, semantic,
    protocol, legal/logic, theorem, proof, merge, freshness, contradiction,
    and coverage checks.  This adapter calls its canonical verifier and then
    independently binds the resulting authority to the caller's current tree
    and, when supplied, expected repository, merge commit, graph, and exact
    acceptance population.  Provider verdicts and pre-merge summaries are not
    accepted as substitutes.
    """

    from .code_evidence_graph import (
        POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA,
        PostMergeEvidenceReceipt,
        verify_post_merge_evidence,
    )

    current_tree = _text(
        current_repository_tree_id,
        field_name="current_repository_tree_id",
        required=True,
    )
    expected_repository = _text(
        expected_repository_id,
        field_name="expected_repository_id",
    )
    expected_commit = _text(
        expected_merge_commit_id,
        field_name="expected_merge_commit_id",
    )
    expected_graph = _text(
        expected_evidence_graph_id,
        field_name="expected_evidence_graph_id",
    )
    expected_criteria = tuple(
        POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA
        if expected_acceptance_criteria is None
        else expected_acceptance_criteria
    )
    expected_criteria_set = set(_strings(expected_criteria))

    if post_merge_evidence is None:
        return PostMergeCompletionAdmissionGate(
            admitted=False,
            merged_tree_id=current_tree,
            covered_acceptance_criteria=(),
            reason_codes=("post_merge_evidence_missing",),
        )

    try:
        receipt = (
            post_merge_evidence
            if isinstance(post_merge_evidence, PostMergeEvidenceReceipt)
            else PostMergeEvidenceReceipt.from_dict(post_merge_evidence)
        )
        # Revalidation reconstructs every authority projection and returns a
        # closed receipt for ordinary evidence failures.  Identity/schema
        # tampering remains exceptional and is intentionally not softened.
        verified = verify_post_merge_evidence(
            receipt,
            current_repository_tree_id=current_tree,
            now=now,
        )
    except ConformanceValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConformanceValidationError(
            f"invalid post-merge evidence receipt: {exc}"
        ) from exc

    reasons = list(verified.reason_codes)
    receipt_criteria = tuple(verified.acceptance_criteria)
    receipt_criteria_set = set(_strings(receipt_criteria))
    exact_criteria = bool(
        len(receipt_criteria) == len(expected_criteria)
        and len(receipt_criteria_set) == len(receipt_criteria)
        and receipt_criteria_set == expected_criteria_set
    )

    if verified.merged_tree_id != current_tree:
        reasons.append("post_merge_tree_mismatch")
    if (
        expected_repository
        and verified.repository_id != expected_repository
    ):
        reasons.append("post_merge_repository_mismatch")
    if expected_commit and verified.merge_commit_id != expected_commit:
        reasons.append("post_merge_commit_mismatch")
    if expected_graph and verified.graph_id != expected_graph:
        reasons.append("post_merge_evidence_graph_mismatch")
    if not exact_criteria:
        reasons.append("post_merge_acceptance_criteria_mismatch")
    if not verified.accepted:
        reasons.append("post_merge_evidence_not_accepted")
    if not verified.authoritative:
        reasons.append("post_merge_evidence_not_authoritative")
    if not verified.merge_eligible:
        reasons.append("post_merge_evidence_merge_ineligible")
    if not getattr(verified, "merge_authoritative", verified.merge_eligible):
        reasons.append("post_merge_merge_not_authoritative")
    if not verified.completion_authoritative:
        reasons.append("post_merge_completion_not_authoritative")
    if not verified.freshness_authoritative:
        reasons.append("post_merge_freshness_not_authoritative")

    reason_codes = tuple(dict.fromkeys(reasons))
    return PostMergeCompletionAdmissionGate(
        admitted=not reason_codes,
        post_merge_evidence_receipt_id=receipt.receipt_id,
        revalidated_receipt_id=verified.receipt_id,
        evidence_graph_id=verified.graph_id,
        repository_id=verified.repository_id,
        merged_tree_id=verified.merged_tree_id,
        merge_commit_id=verified.merge_commit_id,
        covered_acceptance_criteria=receipt_criteria,
        reason_codes=reason_codes,
    )


# Compatibility names for callers that describe the same boundary as a gate
# or verifier rather than an admission evaluation.
PostMergeCompletionGate = PostMergeCompletionAdmissionGate
verify_post_merge_completion_admission = (
    evaluate_post_merge_completion_admission
)


@dataclass(frozen=True)
class RequiresProofCheck:
    """One requires_proof(property_id, assurance) admission decision."""

    property_id: str
    required_assurance: AssuranceLevel
    admitted: bool
    reason_codes: tuple[str, ...] = ()
    precondition_id: str = ""
    task_id: str = ""
    receipt_id: str = ""
    derived_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    cache_status: str = ""
    from_cache: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "property_id", _text(self.property_id, field_name="property_id")
        )
        object.__setattr__(
            self,
            "precondition_id",
            _text(self.precondition_id, field_name="precondition_id"),
        )
        object.__setattr__(self, "task_id", _text(self.task_id, field_name="task_id"))
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, field_name="receipt_id")
        )
        object.__setattr__(
            self, "cache_status", _text(self.cache_status, field_name="cache_status")
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        if not isinstance(self.admitted, bool):
            raise ConformanceValidationError("admitted must be boolean")
        if not isinstance(self.from_cache, bool):
            raise ConformanceValidationError("from_cache must be boolean")
        required = self.required_assurance
        if not isinstance(required, AssuranceLevel):
            required = AssuranceLevel(str(required))
        object.__setattr__(self, "required_assurance", required)
        derived = self.derived_assurance
        if not isinstance(derived, AssuranceLevel):
            derived = AssuranceLevel(str(derived))
        object.__setattr__(self, "derived_assurance", derived)
        if self.admitted and self.reason_codes:
            raise ConformanceValidationError(
                "admitted requires_proof check cannot contain rejection reasons"
            )
        if not self.admitted and not self.reason_codes:
            raise ConformanceValidationError(
                "rejected requires_proof check requires a reason"
            )

    @property
    def check_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "property_id": self.property_id,
            "required_assurance": self.required_assurance.value,
            "admitted": self.admitted,
            "reason_codes": list(self.reason_codes),
            "precondition_id": self.precondition_id,
            "task_id": self.task_id,
            "receipt_id": self.receipt_id,
            "derived_assurance": self.derived_assurance.value,
            "cache_status": self.cache_status,
            "from_cache": self.from_cache,
        }
        if include_id:
            payload["check_id"] = self.check_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiresProofCheck":
        if not isinstance(payload, Mapping):
            raise ConformanceValidationError("requires_proof check must be a mapping")
        result = cls(
            property_id=str(payload.get("property_id") or ""),
            required_assurance=AssuranceLevel(
                str(
                    payload.get("required_assurance")
                    or AssuranceLevel.KERNEL_VERIFIED.value
                )
            ),
            admitted=bool(payload.get("admitted", False)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            precondition_id=str(payload.get("precondition_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            derived_assurance=AssuranceLevel(
                str(
                    payload.get("derived_assurance")
                    or AssuranceLevel.UNVERIFIED.value
                )
            ),
            cache_status=str(payload.get("cache_status") or ""),
            from_cache=bool(payload.get("from_cache", False)),
        )
        claimed = str(payload.get("check_id") or "")
        if claimed and claimed != result.check_id:
            raise ConformanceValidationError(
                "requires_proof check identity mismatch"
            )
        return result


@dataclass(frozen=True)
class RequiresProofAdmissionResult:
    """Plan-level admission over every requires_proof precondition.

    Fail-closed: missing cache-backed receipts, insufficient re-derived
    assurance, and candidate-only evidence never admit work.
    """

    admitted: bool
    plan_id: str = ""
    repository_tree_id: str = ""
    checks: tuple[RequiresProofCheck, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.admitted, bool):
            raise ConformanceValidationError("admitted must be boolean")
        object.__setattr__(self, "plan_id", _text(self.plan_id, field_name="plan_id"))
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        checks = tuple(
            item
            if isinstance(item, RequiresProofCheck)
            else RequiresProofCheck.from_dict(item)
            for item in self.checks
        )
        object.__setattr__(self, "checks", checks)
        if self.admitted and self.reason_codes:
            raise ConformanceValidationError(
                "admitted requires_proof result cannot contain rejection reasons"
            )
        if self.admitted and any(not item.admitted for item in checks):
            raise ConformanceValidationError(
                "admitted requires_proof result cannot include failed checks"
            )
        if not self.admitted and not self.reason_codes and checks:
            # Aggregate reasons from checks when the caller omitted them.
            aggregated = tuple(
                dict.fromkeys(
                    code
                    for item in checks
                    for code in item.reason_codes
                )
            )
            if aggregated:
                object.__setattr__(self, "reason_codes", aggregated)
        if not self.admitted and not self.reason_codes and checks:
            raise ConformanceValidationError(
                "rejected requires_proof result requires a reason"
            )

    @property
    def admission_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REQUIRES_PROOF_ADMISSION_SCHEMA,
            "admitted": self.admitted,
            "plan_id": self.plan_id,
            "repository_tree_id": self.repository_tree_id,
            "checks": [item.to_dict() for item in self.checks],
            "reason_codes": list(self.reason_codes),
        }
        if include_id:
            payload["admission_id"] = self.admission_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiresProofAdmissionResult":
        if not isinstance(payload, Mapping):
            raise ConformanceValidationError(
                "requires_proof admission must be a mapping"
            )
        schema = str(payload.get("schema") or REQUIRES_PROOF_ADMISSION_SCHEMA)
        if schema != REQUIRES_PROOF_ADMISSION_SCHEMA:
            raise ConformanceValidationError(
                f"unsupported requires_proof admission schema: {schema}"
            )
        result = cls(
            admitted=bool(payload.get("admitted", False)),
            plan_id=str(payload.get("plan_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            checks=tuple(payload.get("checks") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("admission_id") or "")
        if claimed and claimed != result.admission_id:
            raise ConformanceValidationError(
                "requires_proof admission identity mismatch"
            )
        return result


def _coerce_assurance_level(value: Any, *, default: AssuranceLevel) -> AssuranceLevel:
    if value in (None, ""):
        return default
    if isinstance(value, AssuranceLevel):
        return value
    return AssuranceLevel(str(value))


def _requires_proof_bindings_from_plan(
    plan: FormalWorkPlan,
) -> tuple[dict[str, str], ...]:
    from .formal_plan_compiler import requires_proof_precondition_bindings

    return requires_proof_precondition_bindings(plan)


def _lookup_requires_proof_receipt(
    *,
    property_id: str,
    precondition_id: str,
    required_assurance: AssuranceLevel,
    proof_cache: Any,
    cache_keys: Mapping[str, Any],
    receipts: Mapping[str, Any],
) -> tuple[ProofReceipt | None, str, bool, tuple[str, ...]]:
    """Return (receipt, cache_status, from_cache, reason_codes)."""

    for key in (precondition_id, property_id):
        if not key:
            continue
        if key in receipts:
            raw = receipts[key]
            try:
                receipt = (
                    raw
                    if isinstance(raw, ProofReceipt)
                    else ProofReceipt.from_dict(raw)
                )
            except (TypeError, ValueError) as exc:
                raise ConformanceValidationError(
                    f"invalid proof receipt for {key}: {exc}"
                ) from exc
            return receipt, "direct", False, ()

    cache_key = None
    for key in (precondition_id, property_id):
        if key and key in cache_keys:
            cache_key = cache_keys[key]
            break
    if proof_cache is None or cache_key is None:
        return None, "miss", False, ("proof_receipt_missing",)

    from .formal_verification_cache import CacheLookupStatus

    try:
        lookup = proof_cache.lookup(
            cache_key, required_assurance=required_assurance
        )
    except (TypeError, ValueError) as exc:
        raise ConformanceValidationError(
            f"invalid proof-cache lookup for {property_id}: {exc}"
        ) from exc

    status = getattr(lookup, "status", None)
    status_value = (
        status.value if isinstance(status, CacheLookupStatus) else str(status or "")
    )
    if status is CacheLookupStatus.HIT or status_value == "hit":
        receipt = getattr(lookup, "receipt", None) or getattr(
            lookup, "kernel_receipt", None
        )
        if receipt is None:
            return None, status_value or "hit", True, ("proof_receipt_missing",)
        if not isinstance(receipt, ProofReceipt):
            try:
                receipt = ProofReceipt.from_dict(receipt)
            except (TypeError, ValueError) as exc:
                raise ConformanceValidationError(
                    f"invalid cached proof receipt for {property_id}: {exc}"
                ) from exc
        return receipt, status_value or "hit", True, ()

    reason_codes = tuple(getattr(lookup, "reason_codes", ()) or ())
    kernel = getattr(lookup, "kernel_receipt", None)
    if kernel is not None:
        if not isinstance(kernel, ProofReceipt):
            try:
                kernel = ProofReceipt.from_dict(kernel)
            except (TypeError, ValueError):
                kernel = None
        if kernel is not None:
            return (
                kernel,
                status_value or "rejected",
                True,
                reason_codes or ("proof_receipt_rejected",),
            )
    if status is CacheLookupStatus.MISS or status_value == "miss":
        return None, "miss", False, ("proof_receipt_missing",)
    return (
        None,
        status_value or "rejected",
        False,
        reason_codes or ("proof_receipt_missing",),
    )


def evaluate_requires_proof_preconditions(
    plan: FormalWorkPlan | Mapping[str, Any] | None = None,
    *,
    proof_cache: Any = None,
    cache_keys: Mapping[str, Any] | None = None,
    receipts: Mapping[str, Any] | None = None,
    catalog: Any = None,
    bindings: Sequence[Mapping[str, Any]] | None = None,
) -> RequiresProofAdmissionResult:
    """Admit work only when every requires_proof precondition is cache-satisfied.

    Acceptance rules (CBP-090):
    - missing receipt fails admission
    - cache hit with re-derived assurance that satisfies the declared level admits
    - candidate-only evidence never admits

    ``cache_keys`` and ``receipts`` may be keyed by ``property_id`` or
    ``precondition_id``.  When both are supplied, direct receipts are consulted
    first (useful for candidate-only negative tests that cannot be stored as
    authoritative cache entries).
    """

    typed_plan: FormalWorkPlan | None = None
    plan_id = ""
    repository_tree_id = ""
    if plan is not None:
        if isinstance(plan, FormalWorkPlan):
            typed_plan = plan
        else:
            typed_plan = FormalWorkPlan.from_dict(plan)
        plan_id = typed_plan.plan_id
        repository_tree_id = typed_plan.repository_tree_id

    if bindings is None:
        if typed_plan is None:
            binding_rows: tuple[dict[str, str], ...] = ()
        else:
            binding_rows = _requires_proof_bindings_from_plan(typed_plan)
    else:
        binding_rows = tuple(dict(item) for item in bindings)

    if catalog is None:
        from .code_property_catalog import DEFAULT_CODE_PROPERTY_CATALOG

        catalog = DEFAULT_CODE_PROPERTY_CATALOG

    key_map = dict(cache_keys or {})
    receipt_map = dict(receipts or {})
    checks: list[RequiresProofCheck] = []
    aggregate_reasons: list[str] = []

    for row in binding_rows:
        property_id = str(
            row.get("property_id") or row.get("property") or ""
        ).strip()
        precondition_id = str(row.get("precondition_id") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        required = _coerce_assurance_level(
            row.get("assurance") or row.get("required_assurance"),
            default=AssuranceLevel.KERNEL_VERIFIED,
        )
        if not property_id:
            check = RequiresProofCheck(
                property_id="",
                required_assurance=required,
                admitted=False,
                reason_codes=("requires_proof_property_missing",),
                precondition_id=precondition_id,
                task_id=task_id,
                cache_status="invalid",
            )
            checks.append(check)
            aggregate_reasons.extend(check.reason_codes)
            continue

        prop = catalog.get(property_id) if catalog is not None else None
        if catalog is not None and prop is None:
            check = RequiresProofCheck(
                property_id=property_id,
                required_assurance=required,
                admitted=False,
                reason_codes=("unknown_property_id",),
                precondition_id=precondition_id,
                task_id=task_id,
                cache_status="invalid",
            )
            checks.append(check)
            aggregate_reasons.extend(check.reason_codes)
            continue

        receipt, cache_status, from_cache, lookup_reasons = (
            _lookup_requires_proof_receipt(
                property_id=property_id,
                precondition_id=precondition_id,
                required_assurance=required,
                proof_cache=proof_cache,
                cache_keys=key_map,
                receipts=receipt_map,
            )
        )
        if receipt is None:
            reasons = tuple(lookup_reasons) or ("proof_receipt_missing",)
            check = RequiresProofCheck(
                property_id=property_id,
                required_assurance=required,
                admitted=False,
                reason_codes=reasons,
                precondition_id=precondition_id,
                task_id=task_id,
                cache_status=cache_status,
                from_cache=from_cache,
            )
            checks.append(check)
            aggregate_reasons.extend(check.reason_codes)
            continue

        # Always re-derive assurance from typed evidence; never trust claimed levels.
        derived = derive_assurance(
            receipt.evidence,
            obligation_id=receipt.obligation_id,
            kernel_id=receipt.kernel_id,
            freshness=receipt.freshness,
        )

        reasons: list[str] = []
        if not assurance_satisfies(derived, required):
            if derived is AssuranceLevel.CANDIDATE:
                reasons.append("candidate_only")
            else:
                reasons.append("required_assurance_not_satisfied")
        if lookup_reasons and not assurance_satisfies(derived, required):
            for code in lookup_reasons:
                if code not in reasons and code != "proof_receipt_missing":
                    reasons.append(code)

        admitted = not reasons
        check = RequiresProofCheck(
            property_id=property_id,
            required_assurance=required,
            admitted=admitted,
            reason_codes=tuple(dict.fromkeys(reasons)),
            precondition_id=precondition_id,
            task_id=task_id,
            receipt_id=receipt.receipt_id,
            derived_assurance=derived,
            cache_status=cache_status,
            from_cache=from_cache,
        )
        checks.append(check)
        if not admitted:
            aggregate_reasons.extend(check.reason_codes)

    # Plans with no requires_proof preconditions admit vacuously.
    admitted = not any(not item.admitted for item in checks)
    return RequiresProofAdmissionResult(
        admitted=admitted,
        plan_id=plan_id,
        repository_tree_id=repository_tree_id,
        checks=tuple(checks),
        reason_codes=tuple(dict.fromkeys(aggregate_reasons)),
    )


# Compatibility aliases used by callers/tests.
evaluate_requires_proof_admission = evaluate_requires_proof_preconditions
RequiresProofAdmissionGate = RequiresProofAdmissionResult


def evaluate_completion_admission(
    *,
    proposal_validation: Any = None,
    validation_dag: Any = None,
    required: bool = False,
    expected_validation_policy_id: str = "",
    code_proof_results: Iterable[Any] = (),
    code_proof_receipts: Iterable[Any] = (),
    implementation_obligations: Any = None,
    required_code_assurance: Any = "kernel_verified",
    require_code_proof: bool = False,
) -> CompletionAdmissionGate:
    """Fail closed when rejected output or a proof candidate is offered.

    Proposal admission and a passing validation DAG only authorize derivation
    of implementation obligations.  A required gate that reaches that point
    automatically requires code proof.  Entering the proof boundary through
    ``require_code_proof`` or any proof/obligation input also requires replay
    of the accepted proposal and complete validation DAG.  Only canonical
    receipts revalidated against the fresh, exactly bound obligation
    population create positive proof authority; provider candidates and
    detached result summaries remain explicit rejected inputs.
    """

    # Materialize caller iterables exactly once.  Apart from making generator
    # inputs deterministic, this lets the gate recognize that a caller has
    # entered the code-proof boundary before deciding which earlier authority
    # records are mandatory.
    proof_result_inputs = tuple(code_proof_results or ())
    proof_receipts = tuple(code_proof_receipts or ())
    proof_boundary_requested = bool(
        proof_result_inputs
        or proof_receipts
        or implementation_obligations is not None
        or require_code_proof
    )

    reasons: list[str] = []
    proposal_receipt_id = ""
    dag_receipt_id = ""
    validation_policy_id = ""
    dag_passed = False
    code_proof_result_ids: list[str] = []
    proof_candidate_receipt_ids: list[str] = []
    expected_validation_policy_id = _text(
        expected_validation_policy_id,
        field_name="expected_validation_policy_id",
    )
    proposal_result = None
    if proposal_validation is None:
        if required or proof_boundary_requested:
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
        dag_passed = dag.passed
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
    elif required or expected_validation_policy_id or proof_boundary_requested:
        reasons.append("validation_dag_missing")

    from .code_proof_obligations import (
        CodeProofReceiptBindingResult,
        ImplementationObligationSet,
        validate_code_proof_receipt_bindings,
    )
    from .formal_verification_contracts import (
        AssuranceLevel,
        ProofReceipt,
        ProofVerdict,
    )

    normalized_proof_results: list[CodeProofReceiptBindingResult] = []
    for item in proof_result_inputs:
        try:
            result = (
                item
                if isinstance(item, CodeProofReceiptBindingResult)
                else CodeProofReceiptBindingResult.from_dict(item)
            )
        except (TypeError, ValueError) as exc:
            raise ConformanceValidationError(
                f"invalid code-proof binding result: {exc}"
            ) from exc
        normalized_proof_results.append(result)

    revalidated_result_ids: set[str] = set()
    obligation_set = None
    if implementation_obligations is not None:
        try:
            obligation_set = (
                implementation_obligations
                if isinstance(
                    implementation_obligations, ImplementationObligationSet
                )
                else ImplementationObligationSet.from_dict(
                    implementation_obligations
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConformanceValidationError(
                f"invalid implementation obligation set: {exc}"
            ) from exc
    if proof_receipts and obligation_set is None:
        reasons.append("code_proof_obligations_missing")
    if obligation_set is not None:
        if not obligation_set.complete:
            reasons.append("code_proof_obligations_incomplete")
        if proposal_result is None:
            reasons.append("code_proof_without_proposal")
        else:
            binding = obligation_set.binding
            if (
                binding.proposal_validation_receipt_id != proposal_receipt_id
                or binding.proposal_accepted is not True
                or binding.accepted_plan_id
                != proposal_result.proposal.accepted_plan_id
                or binding.repository_id
                != proposal_result.proposal.repository_id
                or binding.repository_tree_id
                != proposal_result.proposal.repository_tree_id
                or (
                    dag_receipt_id
                    and binding.validation_dag_receipt_id != dag_receipt_id
                )
                or (
                    validation_policy_id
                    and binding.validation_policy_id != validation_policy_id
                )
                or (
                    validation_dag is not None
                    and binding.repository_tree_id != dag.repository_tree_id
                )
            ):
                reasons.append("code_proof_authority_chain_mismatch")
        for item in proof_receipts:
            try:
                receipt = (
                    item
                    if isinstance(item, ProofReceipt)
                    else ProofReceipt.from_dict(item)
                )
                revalidated = validate_code_proof_receipt_bindings(
                    receipt,
                    obligation_set,
                    required_assurance=AssuranceLevel(
                        required_code_assurance
                    ),
                )
                normalized_proof_results.append(revalidated)
                revalidated_result_ids.add(revalidated.result_id)
            except (TypeError, ValueError) as exc:
                raise ConformanceValidationError(
                    f"invalid canonical code-proof receipt: {exc}"
                ) from exc

    # Stable de-duplication allows callers to retain a diagnostic result while
    # also supplying the canonical receipt needed for positive authority.
    normalized_proof_results = list(
        {
            result.result_id: result for result in normalized_proof_results
        }.values()
    )
    valid_obligation_ids: set[str] = set()
    for result in normalized_proof_results:
        code_proof_result_ids.append(result.result_id)
        if result.authoritative_assurance is AssuranceLevel.CANDIDATE:
            proof_candidate_receipt_ids.append(result.receipt_id)
            reasons.append("code_proof_candidate_only")
        if (
            result.authoritative_verdict is not ProofVerdict.PROVED
            or result.authoritative_assurance.rank
            < AssuranceLevel.KERNEL_VERIFIED.rank
        ):
            reasons.append("code_proof_not_authoritative")
        if not result.valid:
            reasons.append("code_proof_binding_rejected")
        elif result.result_id in revalidated_result_ids:
            valid_obligation_ids.add(result.obligation_id)
        else:
            # A detached serialized/in-memory summary is useful diagnostic
            # input, but only revalidating the canonical receipt against the
            # obligation set may create positive proof authority.
            reasons.append("code_proof_unverified_summary")
    proof_required = proof_boundary_requested or bool(
        required
        and proposal_result is not None
        and proposal_result.accepted
        and dag_passed
    )
    if proof_required:
        if obligation_set is None or not proof_receipts:
            reasons.append("code_proof_missing")
        elif valid_obligation_ids != set(obligation_set.obligation_ids):
            reasons.append("code_proof_population_incomplete")

    return CompletionAdmissionGate(
        admitted=not reasons,
        proposal_receipt_id=proposal_receipt_id,
        validation_dag_receipt_id=dag_receipt_id,
        validation_policy_id=validation_policy_id,
        code_proof_result_ids=tuple(code_proof_result_ids),
        proof_candidate_receipt_ids=tuple(proof_candidate_receipt_ids),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def evaluate_transitive_impact_admission_closure(
    *,
    proposal_validation: Any,
    validation_dag: Any,
) -> CompletionAdmissionGate:
    """Replay the G101 witness across proof and completion boundaries.

    The operational validation is successful evidence precisely when its
    seeded transitive defect makes the DAG fail closed.  This helper requires
    that exact proof-boundary classification and returns the canonical
    completion gate showing that admission remained closed.
    """

    from .code_proof_obligations import (
        transitive_impact_blocks_proof_derivation,
    )

    if not transitive_impact_blocks_proof_derivation(
        proposal_validation,
        validation_dag,
    ):
        raise ConformanceValidationError(
            "validation DAG is not a closed transitive-impact witness"
        )
    gate = evaluate_completion_admission(
        proposal_validation=proposal_validation,
        validation_dag=validation_dag,
        required=True,
    )
    if gate.admitted or "validation_dag_failed" not in gate.reason_codes:
        raise ConformanceValidationError(
            "transitive-impact witness did not close completion admission"
        )
    return gate


def evaluate_strict_validation_completion(
    *,
    repository_id: str,
    repository_tree: str,
    producing_tasks: Sequence[Any] = (),
    child_goals: Sequence[Any] = (),
    proposal_validation: Any = None,
    validation_projection: Any = None,
    proof_projection: Any = None,
    current_state: Any = "active",
    evidence: Sequence[Any] = (),
    tasks_complete: bool = False,
    coverage: Any = None,
    analyzer_health: Any = None,
    exhaustion_quorum: Any = None,
    required_exhaustive_receipts: int = (
        STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS
    ),
    now: Any = None,
    freshness_seconds: float = 3600.0,
    clock_skew_seconds: float = 300.0,
    analysis_inconclusive: bool = False,
    blocked_reason: str = "",
) -> Any:
    """Evaluate the closed ASI-G040 parent completion boundary.

    The operational proposal, DAG, and proof records remain evidence for their
    respective child goals.  They cannot complete the parent by themselves.
    Parent verification additionally fixes the complete producer, child,
    criterion, gate, analyzer, and exhaustion populations to this revision and
    current tree, then delegates the state transition to the canonical
    two-phase goal-completion lifecycle.
    """

    from .goal_completion import evaluate_goal_completion

    if (
        isinstance(required_exhaustive_receipts, bool)
        or not isinstance(required_exhaustive_receipts, int)
        or required_exhaustive_receipts
        != STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS
    ):
        raise ValueError(
            "required_exhaustive_receipts must equal the configured "
            f"ASI-G040 count {STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS}"
        )
    for name, value in (
        ("freshness_seconds", freshness_seconds),
        ("clock_skew_seconds", clock_skew_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a non-negative number")

    def payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return dict(converted)
        return {}

    def normalized(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def normalized_gate(value: Any) -> str:
        return (
            normalized(value)
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

    def parsed_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str) and value.strip():
            try:
                result = datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)

    current = parsed_datetime(now) or datetime.now(timezone.utc)
    max_age = timedelta(seconds=float(freshness_seconds))
    clock_skew = timedelta(seconds=float(clock_skew_seconds))

    def fresh(value: Any) -> bool:
        observed = parsed_datetime(value)
        return bool(
            observed is not None
            and observed <= current + clock_skew
            and current - observed <= max_age
        )

    repository_id = str(repository_id or "").strip()
    repository_tree = str(repository_tree or "").strip()
    expected_binding = {
        "repository_id": repository_id,
        "tree_id": repository_tree,
        "objective_id": STRICT_VALIDATION_OBJECTIVE_ID,
        "objective_revision": STRICT_VALIDATION_OBJECTIVE_REVISION,
        "analyzer_version": (
            STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    terminal_states = frozenset(
        {
            "complete",
            "completed",
            "passed",
            "success",
            "succeeded",
            "verified",
            "verified_complete",
        }
    )

    producer_values = [payload(item) for item in producing_tasks]
    producer_ids = [
        str(item.get("task_id", item.get("id", "")) or "").strip()
        for item in producer_values
    ]
    producer_population_complete = bool(
        repository_id
        and repository_tree
        and len(producer_ids) == len(set(producer_ids))
        and set(producer_ids) == set(STRICT_VALIDATION_PRODUCING_TASK_IDS)
        and len(producer_ids) == len(STRICT_VALIDATION_PRODUCING_TASK_IDS)
        and all(
            normalized(item.get("status", item.get("state", "")))
            in terminal_states
            for item in producer_values
        )
    )

    child_values = [payload(item) for item in child_goals]
    child_ids = [
        str(item.get("goal_id", item.get("id", "")) or "").strip()
        for item in child_values
    ]

    def child_current(child: Mapping[str, Any]) -> bool:
        gate_value = child.get("completion_gate", child.get("gate"))
        gate = gate_value if isinstance(gate_value, Mapping) else {}
        evaluated_value = gate.get("evaluated_evidence")
        evaluated = (
            evaluated_value
            if isinstance(evaluated_value, Mapping)
            else {}
        )
        validations = evaluated.get("validation_evidence")
        proof_requirements = child.get(
            "proof_requirements",
            evaluated.get("proof_requirements", ()),
        )
        if isinstance(proof_requirements, Mapping):
            proof_requirements = (proof_requirements,)
        validation_records_current = bool(
            isinstance(validations, list)
            and validations
            and all(
                isinstance(item, Mapping)
                and item.get("valid", item.get("verified")) is True
                and isinstance(item.get("evidence"), Mapping)
                and item["evidence"].get("repository_id") == repository_id
                and item["evidence"].get("repository_tree")
                == repository_tree
                for item in validations
            )
        )
        proof_records_conclusive = bool(
            isinstance(proof_requirements, (list, tuple))
            and proof_requirements
            and all(
                isinstance(item, Mapping)
                and str(
                    item.get(
                        "repository_tree",
                        item.get("tree_id", ""),
                    )
                    or ""
                ).strip()
                == repository_tree
                and bool(
                    str(
                        item.get(
                            "provenance_id",
                            item.get(
                                "proof_receipt_id",
                                item.get("receipt_id", ""),
                            ),
                        )
                        or ""
                    ).strip()
                )
                and normalized(item.get("required_assurance"))
                not in {"", "unverified", "candidate"}
                and normalized(item.get("authoritative_assurance"))
                not in {"", "unverified", "candidate"}
                and item.get("assurance_satisfied") is True
                and normalized(item.get("proof_verdict")) == "proved"
                and normalized(item.get("freshness")) == "current"
                and item.get("contradicted", False) is False
                and not tuple(item.get("reason_codes") or ())
                for item in proof_requirements
            )
        )
        return bool(
            normalized(child.get("state", child.get("next_state", "")))
            == GoalState.VERIFIED_COMPLETE.value
            and child.get("verified") is True
            and gate.get("passed") is True
            and evaluated.get("repository_id") == repository_id
            and evaluated.get("repository_tree") == repository_tree
            and fresh(evaluated.get("evaluated_at"))
            and validation_records_current
            and proof_records_conclusive
        )

    child_population_complete = bool(
        len(child_ids) == len(set(child_ids))
        and set(child_ids) == set(STRICT_VALIDATION_CHILD_GOAL_IDS)
        and len(child_ids) == len(STRICT_VALIDATION_CHILD_GOAL_IDS)
        and all(child_current(child) for child in child_values)
    )

    # Reconstruct all three producer-owned records before joining their gate
    # populations.  A scheduler vocabulary projection or caller-authored
    # ``qualifies=True`` mapping is not evidence for proposal/proof gates.
    proposal_owned = {"schema", "authority", "patch", "path", "ast_interface"}
    scheduler_owned = {"impact_test", "semantic_proof", "merge", "freshness"}
    proof_owned = {"semantic_proof"}
    proposal_gate_kinds: set[str] = set()
    scheduler_gate_kinds: set[str] = set()
    proof_gate_kinds: set[str] = set()
    proposal_complete = False
    validation_projection_complete = False
    proof_projection_complete = False

    try:
        from .proposal_validation import (
            ProposalValidationReceipt,
            ProposalValidationResult,
        )

        if isinstance(proposal_validation, ProposalValidationResult):
            proposal_result = proposal_validation
            proposal_receipt = proposal_result.receipt
        elif isinstance(proposal_validation, ProposalValidationReceipt):
            proposal_result = None
            proposal_receipt = proposal_validation
        elif isinstance(proposal_validation, Mapping):
            if "proposal" in proposal_validation:
                proposal_result = ProposalValidationResult.from_dict(
                    proposal_validation
                )
                proposal_receipt = proposal_result.receipt
            else:
                proposal_result = None
                proposal_receipt = ProposalValidationReceipt.from_dict(
                    proposal_validation
                )
        else:
            raise ValueError("proposal validation is missing")
        proposal_gate_evidence = dict(
            proposal_receipt.proposal_gate_evidence
        )
        proposal_gates = proposal_gate_evidence.get("gates")
        proposal_gates = (
            proposal_gates if isinstance(proposal_gates, Mapping) else {}
        )
        proposal_gate_kinds = {
            normalized_gate(name) for name in proposal_gates
        }
        proposal_complete = bool(
            proposal_receipt.accepted
            and (
                proposal_result is None
                or proposal_result.accepted
            )
            and proposal_receipt.repository_tree_id == repository_tree
            and proposal_receipt.objective_id
            in STRICT_VALIDATION_CHILD_GOAL_IDS
            and proposal_gate_evidence.get("all_owned_gates_passed") is True
            and proposal_gate_evidence.get("completion_authoritative") is False
            and proposal_gate_kinds == proposal_owned
            and all(
                isinstance(value, Mapping)
                and value.get("passed") is True
                for value in proposal_gates.values()
            )
        )
    except (TypeError, ValueError):
        proposal_complete = False

    try:
        from .validation_scheduler import (
            StrictValidationDAGCompletionEvidence,
        )

        scheduler_evidence = (
            validation_projection
            if isinstance(
                validation_projection,
                StrictValidationDAGCompletionEvidence,
            )
            else StrictValidationDAGCompletionEvidence.from_dict(
                payload(validation_projection)
            )
        )
        scheduler_payload = scheduler_evidence.to_dict()
        scheduler_gate_kinds = {
            normalized_gate(item)
            for item in scheduler_evidence.scheduler_gate_kinds
        }
        validation_projection_complete = bool(
            scheduler_evidence.objective_id
            == STRICT_VALIDATION_OBJECTIVE_ID
            and scheduler_evidence.child_objective_id
            in STRICT_VALIDATION_CHILD_GOAL_IDS
            and scheduler_evidence.repository_tree_id == repository_tree
            and scheduler_evidence.operational_receipt_id
            and scheduler_evidence.evidence_id
            and scheduler_evidence.qualifies
            and scheduler_evidence.completion_authoritative is False
            and scheduler_payload.get("completion_authoritative") is False
            and scheduler_gate_kinds == scheduler_owned
        )
    except (TypeError, ValueError):
        validation_projection_complete = False

    try:
        from .code_proof_obligations import (
            StrictValidationProofCompletionEvidence,
        )

        proof_evidence = (
            proof_projection
            if isinstance(
                proof_projection,
                StrictValidationProofCompletionEvidence,
            )
            else StrictValidationProofCompletionEvidence.from_dict(
                payload(proof_projection)
            )
        )
        proof_gate_kinds = {
            normalized_gate(item) for item in proof_evidence.gate_kinds
        }
        proof_projection_complete = bool(
            proof_evidence.objective_id == STRICT_VALIDATION_OBJECTIVE_ID
            and proof_evidence.child_objective_id
            in STRICT_VALIDATION_CHILD_GOAL_IDS
            and proof_evidence.repository_id == repository_id
            and proof_evidence.repository_tree_id == repository_tree
            and proof_evidence.operational_receipt_id
            and proof_evidence.evidence_id
            and proof_evidence.qualifies
            and proof_evidence.completion_authoritative is False
            and proof_gate_kinds == proof_owned
        )
    except (TypeError, ValueError):
        proof_projection_complete = False

    producer_gate_join_complete = bool(
        proposal_complete
        and validation_projection_complete
        and proof_projection_complete
        and proposal_gate_kinds | scheduler_gate_kinds | proof_gate_kinds
        == set(STRICT_VALIDATION_GATE_KINDS)
    )

    expected_criteria = {
        normalized(item) for item in STRICT_VALIDATION_ACCEPTANCE_CRITERIA
    }
    evidence_values = [payload(item) for item in evidence]
    receipt_ids_by_criterion: dict[str, set[str]] = {}
    evidence_criteria: list[str] = []
    for record in evidence_values:
        source_value = record.get("evidence", record)
        source = (
            dict(source_value)
            if isinstance(source_value, Mapping)
            else record
        )
        criterion = normalized(
            source.get(
                "acceptance_criterion",
                source.get("criterion", source.get("acceptance", "")),
            )
        )
        evidence_criteria.append(criterion)
        receipt_id = str(
            source.get(
                "provenance_cid",
                source.get(
                    "receipt_id",
                    source.get("evidence_id", source.get("receipt_cid", "")),
                ),
            )
            or ""
        ).strip()
        if criterion and receipt_id:
            receipt_ids_by_criterion.setdefault(criterion, set()).add(
                receipt_id
            )
    evidence_population_complete = bool(
        len(evidence_values) == len(expected_criteria)
        and len(evidence_criteria) == len(set(evidence_criteria))
        and set(evidence_criteria) == expected_criteria
        and all(
            len(receipt_ids_by_criterion.get(criterion, set())) == 1
            for criterion in expected_criteria
        )
    )

    coverage_value = payload(coverage)
    rows_value = coverage_value.get("criteria")
    rows = rows_value if isinstance(rows_value, list) else []

    def row_criterion(row: Mapping[str, Any]) -> str:
        return normalized(
            row.get(
                "criterion",
                row.get(
                    "acceptance_criterion",
                    row.get("acceptance", ""),
                ),
            )
        )

    def implementation_bound(row: Mapping[str, Any]) -> bool:
        for name in (
            "implementation",
            "implementation_binding",
            "changed_files",
            "predicted_files",
            "ast_symbols",
            "interfaces",
        ):
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                return True
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and any(str(item or "").strip() for item in value)
            ):
                return True
        return False

    def validation_ids(row: Mapping[str, Any]) -> set[str]:
        raw = row.get(
            "validation_receipt_ids",
            row.get("validation_receipt_id", ()),
        )
        if isinstance(raw, str):
            raw = (raw,)
        if not (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes, bytearray))
        ):
            return set()
        return {
            str(item or "").strip()
            for item in raw
            if str(item or "").strip()
        }

    row_keys = [
        row_criterion(row) for row in rows if isinstance(row, Mapping)
    ]
    coverage_bound = bool(
        evidence_population_complete
        and coverage_value.get("verified") is True
        and coverage_value.get("repository_id") == repository_id
        and coverage_value.get("repository_tree") == repository_tree
        and len(row_keys) == len(set(row_keys)) == len(expected_criteria)
        and set(row_keys) == expected_criteria
        and all(
            isinstance(row, Mapping)
            and implementation_bound(row)
            and len(validation_ids(row)) == 1
            and validation_ids(row)
            == receipt_ids_by_criterion.get(row_criterion(row), set())
            for row in rows
        )
    )
    if not coverage_bound:
        coverage_value = {
            **coverage_value,
            "verified": False,
            "passed": False,
            "reason_codes": [
                (
                    "validation_evidence_population_incomplete"
                    if not evidence_population_complete
                    else "coverage_validation_receipt_unbound"
                )
            ],
        }

    health_value = payload(analyzer_health)
    health_binding_value = health_value.get("binding")
    health_binding = (
        dict(health_binding_value)
        if isinstance(health_binding_value, Mapping)
        else {}
    )
    health_valid = bool(
        all(expected_binding.values())
        and health_binding == expected_binding
        and normalized(health_value.get("status")) == "healthy"
        and health_value.get("healthy") is True
        and health_value.get("safe_for_completion_reasoning") is True
    )
    if not health_valid:
        health_value = {
            **health_value,
            "healthy": False,
            "safe_for_completion_reasoning": False,
        }

    quorum_value = payload(exhaustion_quorum)
    members_value = quorum_value.get("members")
    members = members_value if isinstance(members_value, list) else []
    quorum_binding_value = quorum_value.get("binding")
    quorum_binding = (
        dict(quorum_binding_value)
        if isinstance(quorum_binding_value, Mapping)
        else {}
    )

    def independent_member_field(name: str) -> bool:
        values = [
            str(member.get(name) or "").strip()
            for member in members
            if isinstance(member, Mapping)
        ]
        return bool(
            len(values) == len(members)
            and all(values)
            and len(values) == len(set(values))
        )

    quorum_valid = bool(
        quorum_value.get("required_members")
        == STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("member_count") == len(members)
        and len(members) == STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("satisfied") is True
        and quorum_value.get("quorum_met") is True
        and health_valid
        and quorum_binding == expected_binding
        and quorum_binding == health_binding
        and independent_member_field("member_id")
        and independent_member_field("evidence_channel")
        and independent_member_field("receipt_cid")
        and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and normalized(member.get("scan_mode")) == "exhaustive"
            and fresh(member.get("finished_at"))
            and isinstance(member.get("binding"), Mapping)
            and dict(member["binding"]) == expected_binding
            for member in members
        )
    )
    if not quorum_valid:
        quorum_value = {
            **quorum_value,
            "satisfied": False,
            "quorum_met": False,
        }

    return evaluate_goal_completion(
        current_state=current_state,
        acceptance_criteria=STRICT_VALIDATION_ACCEPTANCE_CRITERIA,
        evidence=evidence,
        tasks_complete=bool(
            tasks_complete
            and producer_population_complete
            and child_population_complete
            and producer_gate_join_complete
        ),
        repository_tree=repository_tree,
        repository_id=repository_id,
        now=current,
        freshness_seconds=float(freshness_seconds),
        clock_skew_seconds=float(clock_skew_seconds),
        coverage=coverage_value,
        analyzer_health=health_value,
        exhaustion_quorum=quorum_value,
        child_goals=child_values,
        analysis_inconclusive=analysis_inconclusive,
        blocked_reason=blocked_reason,
        require_completion_gate=True,
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
                CompletionAdmissionGate.from_dict(self.completion_admission),
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
    toolchain_id: str = "",
    ast_scope_ids: Sequence[str] | None = None,
    premise_ids: Sequence[str] | None = None,
    counterexample_ids: Sequence[str] | None = None,
    plan_consistency: Any = "",
    proposal_validation: Any = None,
    validation_dag: Any = None,
    require_proposal_validation: bool = False,
    expected_validation_policy_id: str = "",
    code_proof_results: Iterable[Any] = (),
    code_proof_receipts: Iterable[Any] = (),
    implementation_obligations: Any = None,
    required_code_assurance: Any = "kernel_verified",
    require_code_proof: bool = False,
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
            goal_id=goal_id,
            toolchain_id=toolchain_id,
            ast_scope_ids=ast_scope_ids or (),
            premise_ids=premise_ids or (),
            counterexample_ids=counterexample_ids or (),
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
        goal_id=goal_id,
        toolchain_id=toolchain_id,
        repository_tree_id=repository_tree_id,
        ast_scope_ids=ast_scope_ids,
        premise_ids=premise_ids,
        counterexample_ids=counterexample_ids,
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
    proof_results = tuple(code_proof_results or ())
    proof_receipts = tuple(code_proof_receipts or ())
    admission = (
        evaluate_completion_admission(
            proposal_validation=proposal_validation,
            validation_dag=validation_dag,
            required=require_proposal_validation,
            expected_validation_policy_id=expected_validation_policy_id,
            code_proof_results=proof_results,
            code_proof_receipts=proof_receipts,
            implementation_obligations=implementation_obligations,
            required_code_assurance=required_code_assurance,
            require_code_proof=require_code_proof,
        )
        if (
            proposal_validation is not None
            or validation_dag is not None
            or require_proposal_validation
            or expected_validation_policy_id
            or proof_results
            or proof_receipts
            or implementation_obligations is not None
            or require_code_proof
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
    "POST_MERGE_COMPLETION_ADMISSION_SCHEMA",
    "STRICT_VALIDATION_ACCEPTANCE_CRITERIA",
    "STRICT_VALIDATION_CHILD_GOAL_IDS",
    "STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION",
    "STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION",
    "STRICT_VALIDATION_GATE_KINDS",
    "STRICT_VALIDATION_OBJECTIVE_ID",
    "STRICT_VALIDATION_OBJECTIVE_REVISION",
    "STRICT_VALIDATION_PRODUCING_TASK_IDS",
    "STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS",
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
    "PostMergeCompletionAdmissionGate",
    "PostMergeCompletionGate",
    "REQUIRES_PROOF_ADMISSION_SCHEMA",
    "RequiresProofAdmissionGate",
    "RequiresProofAdmissionResult",
    "RequiresProofCheck",
    "TransitionDisposition",
    "TransitionFinding",
    "bind_goal_completion",
    "binding_for_plan",
    "changed_bindings",
    "check_plan_conformance",
    "compare_plan_conformance",
    "evaluate_completion_evidence",
    "evaluate_completion_admission",
    "evaluate_post_merge_completion_admission",
    "evaluate_requires_proof_admission",
    "evaluate_requires_proof_preconditions",
    "evaluate_transitive_impact_admission_closure",
    "evaluate_formal_goal_completion",
    "evaluate_goal_completion",
    "evaluate_goal_completion_with_conformance",
    "evaluate_plan_conformance",
    "evaluate_strict_validation_completion",
    "verify_post_merge_completion_admission",
    "invalidate_plan_conformance",
    "read_conformance_evidence",
    "read_formal_plan_conformance",
    "replay_conformance_evidence",
    "replay_plan_conformance",
    "write_conformance_evidence",
    "write_formal_plan_conformance",
]
