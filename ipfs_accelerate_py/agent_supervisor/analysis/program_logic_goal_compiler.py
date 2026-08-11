"""Compile task and repair evidence into finite program-logic goals (LPR-006).

``ProgramLogicGoalCompiler`` turns exact RPR artifacts — broken traces, call
requirements, contract deltas, consumer obligations, missing-input records,
required-behavior contracts, and memory-safety facets — into a finite inventory
of :class:`ProgramLogicGoal` records with explicit source bindings.

Authority rules (fail-closed):

* Free-form objective/prose text may *nominate* a goal but never satisfies it
  and never becomes an axiom or discharged claim.
* Source precedence is retained on every binding; nominating routes cannot
  promote to semantic authority.
* One required goal exists per resolved consumer/facet pair.
* Dynamic, native, and unsupported semantics remain first-class residuals —
  they never silently disappear during decomposition.
* Conflicting intent (same consumer/facet with incompatible expected facts)
  yields a diagnostic rather than an invented resolution.
* Counterexample targets are retained wherever the selected logic supports
  a negative or counterexample family.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    DeltaDisposition,
    DeltaKind,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)
from .contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    CallRequirementContract,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    TraceDisposition,
)
from .program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    LogicFacetRef,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicPredictionError,
    ProofStatus,
    SourceAuthorityClass,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

GOAL_SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-source-binding@1"
)
GOAL_DIAGNOSTIC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-diagnostic@1"
)
PROGRAM_LOGIC_GOAL_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-goal-compilation@1"
)
PRODUCER_ID: Final[str] = "program-logic-goal-compiler@1"
CONTRACT_VERSION: Final[int] = 1

MAX_GOALS: Final[int] = 512
MAX_BINDINGS: Final[int] = 1024
MAX_DIAGNOSTICS: Final[int] = 256
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_REF_BYTES: Final[int] = 512


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProgramLogicGoalCompilerError(ContractValidationError):
    """Malformed or unsafe goal-compilation input."""


class ProgramLogicGoalCompilerAuthorityError(ProgramLogicGoalCompilerError):
    """Root, identity, or authority promotion failure."""


class ProgramLogicGoalCompilerBoundsError(ProgramLogicGoalCompilerError):
    """A compilation budget was exceeded."""


class ProgramLogicGoalConflictError(ProgramLogicGoalCompilerError):
    """Conflicting intent that cannot be silently resolved."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class GoalObligationKind(str, Enum):
    """Semantic obligation dimensions compiled from plan §4.2 goal families."""

    CALLER_INPUT_ACCEPTANCE = "caller_input_acceptance"
    VALUE_SUFFICIENCY = "value_sufficiency"
    OUTPUT_REFINEMENT = "output_refinement"
    TOTALITY = "totality"
    NULLABILITY = "nullability"
    RANGE = "range"
    ALLOWED_ERRORS = "allowed_errors"
    PERMITTED_EFFECTS = "permitted_effects"
    CAPABILITIES = "capabilities"
    AUTHORIZATION = "authorization"
    RESOURCES = "resources"
    TEMPORAL = "temporal"
    STATE = "state"
    CONCURRENCY = "concurrency"
    SCHEMA = "schema"
    CONSTRUCTOR = "constructor"
    SERIALIZATION = "serialization"
    REGISTRATION = "registration"
    PLACEMENT = "placement"
    OWNERSHIP = "ownership"
    LIFETIME = "lifetime"
    INFORMATION_PROVENANCE = "information_provenance"
    MEMORY_SAFETY = "memory_safety"
    CONSISTENCY = "consistency"
    IDEMPOTENCE = "idempotence"
    ATOMICITY = "atomicity"
    ORDERING = "ordering"
    COMPATIBILITY = "compatibility"
    CONSUMER_MIGRATION = "consumer_migration"


class GoalSourceKind(str, Enum):
    """Closed classes of exact evidence that may bind a compiled goal."""

    BROKEN_TRACE = "broken_trace"
    CALL_REQUIREMENT = "call_requirement"
    CONTRACT_DELTA = "contract_delta"
    CONSUMER_OBLIGATION = "consumer_obligation"
    MISSING_INPUT = "missing_input"
    REQUIRED_BEHAVIOR = "required_behavior"
    MEMORY_SAFETY = "memory_safety"
    TASK_PROSE = "task_prose"
    OBJECTIVE_TEXT = "objective_text"
    IMPLEMENTATION_HYPOTHESIS = "implementation_hypothesis"


class CompilationDisposition(str, Enum):
    """Outcome of one goal-compilation attempt."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ABSTAINED = "abstained"


class GoalDiagnosticKind(str, Enum):
    """Closed diagnostic reasons emitted during compilation."""

    CONFLICTING_INTENT = "conflicting_intent"
    PROSE_NOMINATION = "prose_nomination"
    UNSUPPORTED_SEMANTIC = "unsupported_semantic"
    DYNAMIC_FRONTIER = "dynamic_frontier"
    NATIVE_BOUNDARY = "native_boundary"
    MISSING_FACET = "missing_facet"
    FRONTIER_CONSUMER = "frontier_consumer"
    ROOT_MISMATCH = "root_mismatch"
    NON_AUTHORITATIVE_SOURCE = "non_authoritative_source"


# Map delta clause kinds onto goal obligation kinds.
_DELTA_TO_OBLIGATION: Final[Mapping[DeltaKind, GoalObligationKind]] = {
    DeltaKind.PARAMETER_ADD: GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
    DeltaKind.PARAMETER_REMOVE: GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
    DeltaKind.PARAMETER_RENAME: GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
    DeltaKind.PARAMETER_REORDER: GoalObligationKind.ORDERING,
    DeltaKind.PARAMETER_DEFAULT: GoalObligationKind.VALUE_SUFFICIENCY,
    DeltaKind.PARAMETER_KEYWORD: GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
    DeltaKind.PARAMETER_VARIANCE: GoalObligationKind.RANGE,
    DeltaKind.RESULT_CHANGE: GoalObligationKind.OUTPUT_REFINEMENT,
    DeltaKind.GENERIC_CHANGE: GoalObligationKind.SCHEMA,
    DeltaKind.NULLABILITY_CHANGE: GoalObligationKind.NULLABILITY,
    DeltaKind.SCHEMA_CHANGE: GoalObligationKind.SCHEMA,
    DeltaKind.SERIALIZATION_CHANGE: GoalObligationKind.SERIALIZATION,
    DeltaKind.PROTOCOL_CHANGE: GoalObligationKind.COMPATIBILITY,
    DeltaKind.SYNC_ASYNC_CHANGE: GoalObligationKind.TEMPORAL,
    DeltaKind.CANCELLATION_CHANGE: GoalObligationKind.TEMPORAL,
    DeltaKind.ERROR_CHANGE: GoalObligationKind.ALLOWED_ERRORS,
    DeltaKind.EFFECT_CHANGE: GoalObligationKind.PERMITTED_EFFECTS,
    DeltaKind.CAPABILITY_CHANGE: GoalObligationKind.CAPABILITIES,
    DeltaKind.AUTHORIZATION_CHANGE: GoalObligationKind.AUTHORIZATION,
    DeltaKind.LIFECYCLE_CHANGE: GoalObligationKind.STATE,
    DeltaKind.TEMPORAL_STATE_CHANGE: GoalObligationKind.TEMPORAL,
    DeltaKind.CONSISTENCY_CHANGE: GoalObligationKind.CONSISTENCY,
    DeltaKind.RESOURCE_CHANGE: GoalObligationKind.RESOURCES,
    DeltaKind.MEMORY_FACET_CHANGE: GoalObligationKind.MEMORY_SAFETY,
    DeltaKind.SYMBOL_MOVE: GoalObligationKind.PLACEMENT,
    DeltaKind.SYMBOL_RENAME: GoalObligationKind.PLACEMENT,
    DeltaKind.SYMBOL_REEXPORT: GoalObligationKind.REGISTRATION,
    DeltaKind.SYMBOL_REGISTRATION: GoalObligationKind.REGISTRATION,
    DeltaKind.VISIBILITY_CHANGE: GoalObligationKind.REGISTRATION,
    DeltaKind.CONSTRUCTOR_INTRO: GoalObligationKind.CONSTRUCTOR,
    DeltaKind.CONSTRUCTOR_REMOVE: GoalObligationKind.CONSTRUCTOR,
    DeltaKind.FIELD_INTRO: GoalObligationKind.SCHEMA,
    DeltaKind.FIELD_REMOVE: GoalObligationKind.SCHEMA,
    DeltaKind.METHOD_INTRO: GoalObligationKind.PLACEMENT,
    DeltaKind.METHOD_REMOVE: GoalObligationKind.PLACEMENT,
    DeltaKind.CLASS_INTRO: GoalObligationKind.PLACEMENT,
    DeltaKind.CLASS_REMOVE: GoalObligationKind.PLACEMENT,
    DeltaKind.DATA_STRUCTURE_INTRO: GoalObligationKind.SCHEMA,
    DeltaKind.DATA_STRUCTURE_REMOVE: GoalObligationKind.SCHEMA,
    DeltaKind.INTERFACE_INTRO: GoalObligationKind.COMPATIBILITY,
    DeltaKind.INTERFACE_REMOVE: GoalObligationKind.COMPATIBILITY,
    DeltaKind.FACTORY_INTRO: GoalObligationKind.CONSTRUCTOR,
    DeltaKind.FACTORY_REMOVE: GoalObligationKind.CONSTRUCTOR,
}

# Obligation → primary logic facet kind.
_OBLIGATION_FACET: Final[Mapping[GoalObligationKind, LogicFacetKind]] = {
    GoalObligationKind.CALLER_INPUT_ACCEPTANCE: LogicFacetKind.TYPE,
    GoalObligationKind.VALUE_SUFFICIENCY: LogicFacetKind.INFORMATION,
    GoalObligationKind.OUTPUT_REFINEMENT: LogicFacetKind.TYPE,
    GoalObligationKind.TOTALITY: LogicFacetKind.TYPE,
    GoalObligationKind.NULLABILITY: LogicFacetKind.TYPE,
    GoalObligationKind.RANGE: LogicFacetKind.TYPE,
    GoalObligationKind.ALLOWED_ERRORS: LogicFacetKind.ERROR,
    GoalObligationKind.PERMITTED_EFFECTS: LogicFacetKind.EFFECT,
    GoalObligationKind.CAPABILITIES: LogicFacetKind.AUTHORIZATION,
    GoalObligationKind.AUTHORIZATION: LogicFacetKind.AUTHORIZATION,
    GoalObligationKind.RESOURCES: LogicFacetKind.RESOURCE,
    GoalObligationKind.TEMPORAL: LogicFacetKind.TEMPORAL,
    GoalObligationKind.STATE: LogicFacetKind.STATE,
    GoalObligationKind.CONCURRENCY: LogicFacetKind.STATE,
    GoalObligationKind.SCHEMA: LogicFacetKind.SCHEMA,
    GoalObligationKind.CONSTRUCTOR: LogicFacetKind.SCHEMA,
    GoalObligationKind.SERIALIZATION: LogicFacetKind.SCHEMA,
    GoalObligationKind.REGISTRATION: LogicFacetKind.PLACEMENT,
    GoalObligationKind.PLACEMENT: LogicFacetKind.PLACEMENT,
    GoalObligationKind.OWNERSHIP: LogicFacetKind.MEMORY,
    GoalObligationKind.LIFETIME: LogicFacetKind.LIFETIME,
    GoalObligationKind.INFORMATION_PROVENANCE: LogicFacetKind.INFORMATION,
    GoalObligationKind.MEMORY_SAFETY: LogicFacetKind.MEMORY,
    GoalObligationKind.CONSISTENCY: LogicFacetKind.STATE,
    GoalObligationKind.IDEMPOTENCE: LogicFacetKind.STATE,
    GoalObligationKind.ATOMICITY: LogicFacetKind.STATE,
    GoalObligationKind.ORDERING: LogicFacetKind.TEMPORAL,
    GoalObligationKind.COMPATIBILITY: LogicFacetKind.SCHEMA,
    GoalObligationKind.CONSUMER_MIGRATION: LogicFacetKind.PLACEMENT,
}

# Trace dispositions that leave an explicit residual rather than a closed goal.
_DYNAMIC_TRACE: Final[frozenset[TraceDisposition]] = frozenset(
    {
        TraceDisposition.DYNAMIC,
        TraceDisposition.EXTERNAL,
        TraceDisposition.UNSUPPORTED,
        TraceDisposition.AMBIGUOUS,
    }
)

# Consumer dispositions that require at least one migration goal.
_RESOLVED_CONSUMERS: Final[frozenset[ConsumerDisposition]] = frozenset(
    {
        ConsumerDisposition.MIGRATE,
        ConsumerDisposition.ADAPTER,
        ConsumerDisposition.UPSTREAM,
        ConsumerDisposition.REVIEW_ONLY,
    }
)

_NOMINATING_SOURCE_KINDS: Final[frozenset[GoalSourceKind]] = frozenset(
    {
        GoalSourceKind.TASK_PROSE,
        GoalSourceKind.OBJECTIVE_TEXT,
        GoalSourceKind.IMPLEMENTATION_HYPOTHESIS,
    }
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "theorem_text",
        "proof_script",
        "prompt_body",
        "objective_text",
        "prose",
    }
)

_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "access_token",
        "refresh_token",
        "bearer",
        "credential",
        "ssh_key",
        "client_secret",
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _text(
    value: Any, name: str, *, required: bool = False, limit: int = MAX_TEXT_BYTES
) -> str:
    if value is None:
        if required:
            raise ProgramLogicGoalCompilerError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise ProgramLogicGoalCompilerError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise ProgramLogicGoalCompilerError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise ProgramLogicGoalCompilerBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True)
    if any(ch.isspace() for ch in text):
        raise ProgramLogicGoalCompilerError(f"{name} must be a compact identifier")
    if len(text.encode("utf-8")) > MAX_REF_BYTES:
        raise ProgramLogicGoalCompilerBoundsError(f"{name} exceeds its byte bound")
    return text


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise ProgramLogicGoalCompilerError(
            f"{name} must be one of: {choices}"
        ) from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramLogicGoalCompilerError(f"{name} must be a boolean")
    return value


def _ids(
    values: Iterable[Any] | None,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values or ():
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            result.append(text)
    if len(result) > limit:
        raise ProgramLogicGoalCompilerBoundsError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(result))
    if required and not ordered:
        raise ProgramLogicGoalCompilerError(f"{name} must not be empty")
    return ordered


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, float):
        raise ProgramLogicGoalCompilerError(
            f"{name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramLogicGoalCompilerError(f"{name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ProgramLogicGoalCompilerError(
                    f"{name} may not contain source bodies"
                )
            if normalized in _SECRET_KEY_MARKERS:
                raise ProgramLogicGoalCompilerError(
                    f"{name} may not contain secret material"
                )
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, name)
    elif isinstance(value, (bytes, bytearray)):
        raise ProgramLogicGoalCompilerError(f"{name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_RECORD_BYTES:
        raise ProgramLogicGoalCompilerBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ProgramLogicGoalCompilerAuthorityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ProgramLogicGoalCompilerError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, CONTRACT_VERSION):
        raise ProgramLogicGoalCompilerError(
            f"{name} has an unsupported contract version"
        )
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ProgramLogicGoalCompilerError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    return {
        field_name: payload[field_name]
        for field_name in fields
        if field_name in payload
    }


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**value)
        )
    raise ProgramLogicGoalCompilerError("roots must be ProgramLogicAuthorityRoots")


def _digest(parts: Mapping[str, Any]) -> str:
    payload = _canonical_json(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _stable_id(prefix: str, **parts: Any) -> str:
    return f"{prefix}:{_digest(parts)}"


def all_obligation_kinds() -> tuple[GoalObligationKind, ...]:
    return tuple(GoalObligationKind)


def all_source_kinds() -> tuple[GoalSourceKind, ...]:
    return tuple(GoalSourceKind)


def obligation_facet_kind(kind: GoalObligationKind | str) -> LogicFacetKind:
    key = kind if isinstance(kind, GoalObligationKind) else GoalObligationKind(kind)
    return _OBLIGATION_FACET[key]


def delta_obligation_kind(kind: DeltaKind | str) -> GoalObligationKind:
    key = kind if isinstance(kind, DeltaKind) else DeltaKind(kind)
    try:
        return _DELTA_TO_OBLIGATION[key]
    except KeyError as exc:
        raise ProgramLogicGoalCompilerError(
            f"delta kind has no obligation mapping: {key!r}"
        ) from exc


def is_nominating_source(kind: GoalSourceKind | str) -> bool:
    key = kind if isinstance(kind, GoalSourceKind) else GoalSourceKind(kind)
    return key in _NOMINATING_SOURCE_KINDS


def source_authority_for_kind(kind: GoalSourceKind) -> SourceAuthorityClass:
    if kind in _NOMINATING_SOURCE_KINDS:
        return SourceAuthorityClass.NOMINATING
    if kind is GoalSourceKind.MEMORY_SAFETY:
        return SourceAuthorityClass.CONFORMANCE
    return SourceAuthorityClass.AUTHORITATIVE


def _facet_token(kind: LogicFacetKind | GoalObligationKind | str) -> str:
    """Compact token safe for identifiers (avoids secret-marker substrings)."""
    if isinstance(kind, LogicFacetKind):
        raw = kind.value
    elif isinstance(kind, GoalObligationKind):
        raw = kind.value
    else:
        raw = str(kind)
    # Prediction contracts reject the substring "authorization:" as secret-like.
    if raw == "authorization":
        return "authz"
    return raw.replace("authorization", "authz")


def _facet_id(kind: LogicFacetKind | GoalObligationKind, *parts: str) -> str:
    token = _facet_token(
        obligation_facet_kind(kind) if isinstance(kind, GoalObligationKind) else kind
    )
    tail = ":".join(part for part in parts if part)
    return f"facet:{token}:{tail}" if tail else f"facet:{token}"


# ---------------------------------------------------------------------------
# GoalSourceBinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalSourceBinding(CanonicalContract):
    """Exact source binding for one compiled goal.

    Retains source precedence, actual/expected fact references, counterexample
    targets, assumptions, and root/bound references.  Nominating-only bindings
    (prose, objective text, implementation hypotheses) cannot satisfy a goal.
    """

    SCHEMA: ClassVar[str] = GOAL_SOURCE_BINDING_SCHEMA

    binding_id: str
    goal_id: str
    source_kind: GoalSourceKind
    source_ref: str
    source_authority: SourceAuthorityClass
    obligation_kind: GoalObligationKind
    consumer_id: str = ""
    facet_id: str = ""
    actual_fact_ref: str = ""
    expected_fact_ref: str = ""
    counterexample_target_ref: str = ""
    assumption_refs: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    nominating_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _identifier(self.binding_id, "binding_id")
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "source_kind", _enum(self.source_kind, GoalSourceKind, "source_kind")
        )
        object.__setattr__(
            self, "source_ref", _identifier(self.source_ref, "source_ref")
        )
        object.__setattr__(
            self,
            "source_authority",
            _enum(self.source_authority, SourceAuthorityClass, "source_authority"),
        )
        object.__setattr__(
            self,
            "obligation_kind",
            _enum(self.obligation_kind, GoalObligationKind, "obligation_kind"),
        )
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "facet_id", _text(self.facet_id, "facet_id"))
        object.__setattr__(
            self, "actual_fact_ref", _text(self.actual_fact_ref, "actual_fact_ref")
        )
        object.__setattr__(
            self,
            "expected_fact_ref",
            _text(self.expected_fact_ref, "expected_fact_ref"),
        )
        object.__setattr__(
            self,
            "counterexample_target_ref",
            _text(self.counterexample_target_ref, "counterexample_target_ref"),
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        object.__setattr__(
            self, "nominating_only", _bool(self.nominating_only, "nominating_only")
        )
        if self.source_kind in _NOMINATING_SOURCE_KINDS and not self.nominating_only:
            raise ProgramLogicGoalCompilerAuthorityError(
                "prose/hypothesis sources must be nominating_only"
            )
        if self.nominating_only and self.source_authority not in {
            SourceAuthorityClass.NOMINATING,
            SourceAuthorityClass.DIAGNOSTIC,
            SourceAuthorityClass.NONE,
        }:
            raise ProgramLogicGoalCompilerAuthorityError(
                "nominating-only bindings cannot claim authoritative source authority"
            )
        if (
            not self.nominating_only
            and self.source_authority is SourceAuthorityClass.NOMINATING
        ):
            raise ProgramLogicGoalCompilerAuthorityError(
                "non-nominating bindings cannot use nominating source authority"
            )
        _bounded(self, "goal source binding")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "binding_id": self.binding_id,
            "goal_id": self.goal_id,
            "source_kind": self.source_kind.value,
            "source_ref": self.source_ref,
            "source_authority": self.source_authority.value,
            "obligation_kind": self.obligation_kind.value,
            "consumer_id": self.consumer_id,
            "facet_id": self.facet_id,
            "actual_fact_ref": self.actual_fact_ref,
            "expected_fact_ref": self.expected_fact_ref,
            "counterexample_target_ref": self.counterexample_target_ref,
            "assumption_refs": list(self.assumption_refs),
            "bound_refs": list(self.bound_refs),
            "nominating_only": self.nominating_only,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalSourceBinding":
        fields = (
            "binding_id",
            "goal_id",
            "source_kind",
            "source_ref",
            "source_authority",
            "obligation_kind",
            "consumer_id",
            "facet_id",
            "actual_fact_ref",
            "expected_fact_ref",
            "counterexample_target_ref",
            "assumption_refs",
            "bound_refs",
            "nominating_only",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "goal source binding")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# GoalDiagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalDiagnostic(CanonicalContract):
    """Bounded diagnostic for conflict, residual, or nomination events."""

    SCHEMA: ClassVar[str] = GOAL_DIAGNOSTIC_SCHEMA

    diagnostic_id: str
    kind: GoalDiagnosticKind
    reason_ref: str
    related_goal_ids: tuple[str, ...] = ()
    related_source_refs: tuple[str, ...] = ()
    consumer_id: str = ""
    facet_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostic_id", _identifier(self.diagnostic_id, "diagnostic_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, GoalDiagnosticKind, "kind")
        )
        object.__setattr__(
            self, "reason_ref", _identifier(self.reason_ref, "reason_ref")
        )
        object.__setattr__(
            self, "related_goal_ids", _ids(self.related_goal_ids, "related_goal_ids")
        )
        object.__setattr__(
            self,
            "related_source_refs",
            _ids(self.related_source_refs, "related_source_refs"),
        )
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "facet_id", _text(self.facet_id, "facet_id"))
        _bounded(self, "goal diagnostic")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "diagnostic_id": self.diagnostic_id,
            "kind": self.kind.value,
            "reason_ref": self.reason_ref,
            "related_goal_ids": list(self.related_goal_ids),
            "related_source_refs": list(self.related_source_refs),
            "consumer_id": self.consumer_id,
            "facet_id": self.facet_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDiagnostic":
        fields = (
            "diagnostic_id",
            "kind",
            "reason_ref",
            "related_goal_ids",
            "related_source_refs",
            "consumer_id",
            "facet_id",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "goal diagnostic")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# ProgramLogicGoalCompilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramLogicGoalCompilation(CanonicalContract):
    """Finite positive/negative obligation inventory with source bindings."""

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_GOAL_COMPILATION_SCHEMA

    roots: ProgramLogicAuthorityRoots
    compilation_id: str
    disposition: CompilationDisposition
    goals: tuple[ProgramLogicGoal, ...]
    source_bindings: tuple[GoalSourceBinding, ...]
    diagnostics: tuple[GoalDiagnostic, ...] = ()
    residual_refs: tuple[str, ...] = ()
    unsupported_refs: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "compilation_id", _identifier(self.compilation_id, "compilation_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CompilationDisposition, "disposition"),
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

        goals = tuple(self.goals)
        if len(goals) > MAX_GOALS:
            raise ProgramLogicGoalCompilerBoundsError("goals exceed compilation bound")
        if not all(isinstance(item, ProgramLogicGoal) for item in goals):
            raise ProgramLogicGoalCompilerError("goals must be ProgramLogicGoal values")
        goal_ids = [item.goal_id for item in goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ProgramLogicGoalCompilerError("goals must have unique goal_ids")
        object.__setattr__(
            self,
            "goals",
            tuple(sorted(goals, key=lambda item: item.goal_id)),
        )

        bindings = tuple(self.source_bindings)
        if len(bindings) > MAX_BINDINGS:
            raise ProgramLogicGoalCompilerBoundsError(
                "source bindings exceed compilation bound"
            )
        if not all(isinstance(item, GoalSourceBinding) for item in bindings):
            raise ProgramLogicGoalCompilerError(
                "source_bindings must be GoalSourceBinding values"
            )
        binding_ids = [item.binding_id for item in bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ProgramLogicGoalCompilerError("source bindings must be unique")
        known_goals = {item.goal_id for item in self.goals}
        for binding in bindings:
            if binding.goal_id not in known_goals:
                raise ProgramLogicGoalCompilerError(
                    "source binding goal_id must reference a compiled goal"
                )
        object.__setattr__(
            self,
            "source_bindings",
            tuple(sorted(bindings, key=lambda item: item.binding_id)),
        )

        diagnostics = tuple(self.diagnostics)
        if len(diagnostics) > MAX_DIAGNOSTICS:
            raise ProgramLogicGoalCompilerBoundsError(
                "diagnostics exceed compilation bound"
            )
        if not all(isinstance(item, GoalDiagnostic) for item in diagnostics):
            raise ProgramLogicGoalCompilerError(
                "diagnostics must be GoalDiagnostic values"
            )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(sorted(diagnostics, key=lambda item: item.diagnostic_id)),
        )

        object.__setattr__(
            self, "residual_refs", _ids(self.residual_refs, "residual_refs")
        )
        object.__setattr__(
            self, "unsupported_refs", _ids(self.unsupported_refs, "unsupported_refs")
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if self.disposition is CompilationDisposition.CONFLICT:
            if not any(
                item.kind is GoalDiagnosticKind.CONFLICTING_INTENT
                for item in self.diagnostics
            ):
                raise ProgramLogicGoalCompilerError(
                    "conflict disposition requires a conflicting_intent diagnostic"
                )
        if self.disposition is CompilationDisposition.COMPLETE:
            if self.residual_refs or self.unsupported_refs:
                raise ProgramLogicGoalCompilerError(
                    "complete compilation cannot retain residuals or unsupported refs"
                )
            if any(
                item.kind is GoalDiagnosticKind.CONFLICTING_INTENT
                for item in self.diagnostics
            ):
                raise ProgramLogicGoalCompilerError(
                    "complete compilation cannot include conflicting_intent diagnostics"
                )
        _bounded(self, "program logic goal compilation")

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(item.goal_id for item in self.goals)

    @property
    def required_goals(self) -> tuple[ProgramLogicGoal, ...]:
        return tuple(
            item
            for item in self.goals
            if item.disposition
            in {GoalDisposition.OPEN, GoalDisposition.PLANNED, GoalDisposition.ADMITTED}
        )

    def goals_for_consumer(self, consumer_id: str) -> tuple[ProgramLogicGoal, ...]:
        consumer = _identifier(consumer_id, "consumer_id")
        goal_ids = {
            binding.goal_id
            for binding in self.source_bindings
            if binding.consumer_id == consumer
        }
        return tuple(item for item in self.goals if item.goal_id in goal_ids)

    def goals_for_obligation(
        self, kind: GoalObligationKind | str
    ) -> tuple[ProgramLogicGoal, ...]:
        obligation = _enum(kind, GoalObligationKind, "obligation_kind")
        goal_ids = {
            binding.goal_id
            for binding in self.source_bindings
            if binding.obligation_kind is obligation
        }
        return tuple(item for item in self.goals if item.goal_id in goal_ids)

    def binding_for_goal(self, goal_id: str) -> tuple[GoalSourceBinding, ...]:
        target = _identifier(goal_id, "goal_id")
        return tuple(item for item in self.source_bindings if item.goal_id == target)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "compilation_id": self.compilation_id,
            "disposition": self.disposition.value,
            "goals": [item.to_dict() for item in self.goals],
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "residual_refs": list(self.residual_refs),
            "unsupported_refs": list(self.unsupported_refs),
            "bound_refs": list(self.bound_refs),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicGoalCompilation":
        fields = (
            "roots",
            "compilation_id",
            "disposition",
            "goals",
            "source_bindings",
            "diagnostics",
            "residual_refs",
            "unsupported_refs",
            "bound_refs",
            "invalidation_refs",
            "producer_id",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "program logic goal compilation"
        )
        values["roots"] = _roots(values["roots"])
        goals_raw = values.get("goals") or ()
        values["goals"] = tuple(
            item
            if isinstance(item, ProgramLogicGoal)
            else ProgramLogicGoal.from_dict(item)
            for item in goals_raw
        )
        bindings_raw = values.get("source_bindings") or ()
        values["source_bindings"] = tuple(
            item
            if isinstance(item, GoalSourceBinding)
            else GoalSourceBinding.from_dict(item)
            for item in bindings_raw
        )
        diagnostics_raw = values.get("diagnostics") or ()
        values["diagnostics"] = tuple(
            item
            if isinstance(item, GoalDiagnostic)
            else GoalDiagnostic.from_dict(item)
            for item in diagnostics_raw
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Prose nomination (cannot satisfy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProseGoalNomination:
    """Task/objective prose may nominate a goal kind but cannot satisfy it."""

    obligation_kind: GoalObligationKind
    statement_ref: str
    source_kind: GoalSourceKind = GoalSourceKind.TASK_PROSE
    subject_symbol_id: str = ""
    consumer_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_kind",
            _enum(self.obligation_kind, GoalObligationKind, "obligation_kind"),
        )
        object.__setattr__(
            self, "statement_ref", _identifier(self.statement_ref, "statement_ref")
        )
        object.__setattr__(
            self, "source_kind", _enum(self.source_kind, GoalSourceKind, "source_kind")
        )
        if self.source_kind not in _NOMINATING_SOURCE_KINDS:
            raise ProgramLogicGoalCompilerAuthorityError(
                "prose nominations must use nominating source kinds"
            )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _text(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )


# ---------------------------------------------------------------------------
# Internal draft record used during compilation
# ---------------------------------------------------------------------------


@dataclass
class _DraftGoal:
    goal_id: str
    family: GoalFamily
    obligation_kind: GoalObligationKind
    disposition: GoalDisposition
    positive_statement_ref: str
    negative_target_ref: str = ""
    counterexample_target_ref: str = ""
    subject_symbol_id: str = ""
    source_refs: list[str] | None = None
    assumption_refs: list[str] | None = None
    unsupported: bool = False
    nominating_only: bool = False
    consumer_id: str = ""
    facet_id: str = ""
    actual_fact_ref: str = ""
    expected_fact_ref: str = ""
    source_kind: GoalSourceKind = GoalSourceKind.CONTRACT_DELTA
    source_ref: str = ""
    source_authority: SourceAuthorityClass = SourceAuthorityClass.AUTHORITATIVE
    bound_refs: list[str] | None = None

    def __post_init__(self) -> None:
        if self.source_refs is None:
            self.source_refs = []
        if self.assumption_refs is None:
            self.assumption_refs = []
        if self.bound_refs is None:
            self.bound_refs = []


# ---------------------------------------------------------------------------
# Root consistency helpers
# ---------------------------------------------------------------------------


def _repair_root_ids(roots: AuthorityRoots) -> tuple[str, str, str]:
    return roots.repository_id, roots.tree_id, roots.policy_id


def _propagation_root_ids(
    roots: PropagationAuthorityRoots,
) -> tuple[str, str, str, str]:
    return (
        roots.repository_id,
        roots.candidate_tree_id,
        roots.base_tree_id,
        roots.policy_id,
    )


def _check_repair_roots(
    logic_roots: ProgramLogicAuthorityRoots, artifact_roots: AuthorityRoots, label: str
) -> None:
    if artifact_roots.repository_id != logic_roots.repository_id:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} repository_id must match compilation roots"
        )
    if artifact_roots.tree_id != logic_roots.tree_id:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} tree_id must match compilation roots"
        )
    if artifact_roots.policy_id != logic_roots.policy_id:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} policy_id must match compilation roots"
        )


def _check_propagation_roots(
    logic_roots: ProgramLogicAuthorityRoots,
    artifact_roots: PropagationAuthorityRoots,
    label: str,
) -> None:
    if artifact_roots.repository_id != logic_roots.repository_id:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} repository_id must match compilation roots"
        )
    if logic_roots.tree_id not in {
        artifact_roots.base_tree_id,
        artifact_roots.candidate_tree_id,
    }:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} tree_id is stale relative to compilation roots"
        )
    if artifact_roots.policy_id != logic_roots.policy_id:
        raise ProgramLogicGoalCompilerAuthorityError(
            f"{label} policy_id must match compilation roots"
        )


# ---------------------------------------------------------------------------
# ProgramLogicGoalCompiler
# ---------------------------------------------------------------------------


class ProgramLogicGoalCompiler:
    """Compile exact RPR evidence into a finite :class:`ProgramLogicGoal` set.

    The compiler never invents axioms from free-form objective text.  Prose may
    only produce nominating-only bindings that cannot discharge a goal.
    """

    def __init__(self, roots: ProgramLogicAuthorityRoots) -> None:
        self.roots = _roots(roots)

    def compile(
        self,
        *,
        broken_traces: Sequence[BrokenContractTrace] = (),
        call_requirements: Sequence[CallRequirementContract] = (),
        contract_deltas: Sequence[ProgramContractDelta] = (),
        consumer_obligations: Sequence[ConsumerMigrationObligation] = (),
        missing_inputs: Sequence[MissingInputRequirement] = (),
        behavior_contracts: Sequence[RequiredBehaviorContract] = (),
        memory_facets: Sequence[MemorySafetyFacet] = (),
        prose_nominations: Sequence[ProseGoalNomination | Mapping[str, Any]] = (),
    ) -> ProgramLogicGoalCompilation:
        drafts: list[_DraftGoal] = []
        diagnostics: list[GoalDiagnostic] = []
        residual_refs: list[str] = []
        unsupported_refs: list[str] = []
        bound_refs: list[str] = [
            self.roots.tree_id,
            self.roots.corpus_id,
            self.roots.policy_id,
            self.roots.translator_id,
            self.roots.toolchain_id,
            self.roots.environment_id,
        ]

        for trace in broken_traces:
            self._ingest_trace(trace, drafts, diagnostics, residual_refs, unsupported_refs)
        for requirement in call_requirements:
            self._ingest_call_requirement(
                requirement, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for delta in contract_deltas:
            self._ingest_delta(
                delta, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for obligation in consumer_obligations:
            self._ingest_consumer(
                obligation, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for missing in missing_inputs:
            self._ingest_missing_input(
                missing, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for behavior in behavior_contracts:
            self._ingest_behavior(
                behavior, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for facet in memory_facets:
            self._ingest_memory_facet(
                facet, drafts, diagnostics, residual_refs, unsupported_refs
            )
        for nomination in prose_nominations:
            self._ingest_prose(nomination, drafts, diagnostics)

        drafts, conflict_diagnostics = self._detect_conflicts(drafts)
        diagnostics.extend(conflict_diagnostics)

        # One required goal per resolved consumer/facet.
        self._assert_consumer_facet_coverage(drafts, consumer_obligations, diagnostics)

        goals, bindings = self._materialize(drafts)

        residual = tuple(sorted(set(residual_refs)))
        unsupported = tuple(sorted(set(unsupported_refs)))
        invalidation = tuple(
            sorted(
                {
                    self.roots.tree_id,
                    self.roots.corpus_id,
                    self.roots.policy_id,
                    self.roots.overlay_id,
                    self.roots.graph_id,
                    self.roots.index_id,
                    *bound_refs,
                }
            )
        )
        bound = tuple(sorted(set(bound_refs)))

        has_conflict = any(
            item.kind is GoalDiagnosticKind.CONFLICTING_INTENT for item in diagnostics
        )
        if has_conflict:
            disposition = CompilationDisposition.CONFLICT
        elif not goals and not residual and not unsupported:
            disposition = CompilationDisposition.ABSTAINED
        elif residual or unsupported or any(
            item.disposition is GoalDisposition.UNSUPPORTED for item in goals
        ):
            disposition = CompilationDisposition.PARTIAL
        else:
            disposition = CompilationDisposition.COMPLETE

        compilation_id = _stable_id(
            "compilation",
            roots=self.roots.content_id,
            goals=[item.goal_id for item in goals],
            residuals=list(residual),
            unsupported=list(unsupported),
            diagnostics=[item.diagnostic_id for item in diagnostics],
        )
        return ProgramLogicGoalCompilation(
            roots=self.roots,
            compilation_id=compilation_id,
            disposition=disposition,
            goals=goals,
            source_bindings=bindings,
            diagnostics=tuple(diagnostics),
            residual_refs=residual,
            unsupported_refs=unsupported,
            bound_refs=bound,
            invalidation_refs=invalidation,
        )

    # -- ingestion paths -----------------------------------------------------

    def _ingest_trace(
        self,
        trace: BrokenContractTrace,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        if not isinstance(trace, BrokenContractTrace):
            raise ProgramLogicGoalCompilerError(
                "broken_traces must contain BrokenContractTrace values"
            )
        _check_repair_roots(self.roots, trace.roots, "broken_trace")
        source_ref = trace.content_id
        subject = trace.caller_symbol_id
        consumer = self.roots.consumer_id

        if trace.disposition in _DYNAMIC_TRACE:
            residual_refs.append(source_ref)
            unsupported_refs.append(f"trace:{trace.disposition.value}")
            kind = (
                GoalDiagnosticKind.DYNAMIC_FRONTIER
                if trace.disposition is TraceDisposition.DYNAMIC
                else GoalDiagnosticKind.UNSUPPORTED_SEMANTIC
            )
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag", kind=kind.value, source=source_ref
                    ),
                    kind=kind,
                    reason_ref=f"reason:trace:{trace.disposition.value}",
                    related_source_refs=(source_ref,),
                    consumer_id=consumer,
                )
            )
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=GoalObligationKind.CALLER_INPUT_ACCEPTANCE.value,
                        source=source_ref,
                        subject=subject,
                    ),
                    family=GoalFamily.POSITIVE,
                    obligation_kind=GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
                    disposition=GoalDisposition.UNSUPPORTED,
                    positive_statement_ref=(
                        f"stmt:caller_input_acceptance:{trace.receiver_reference}"
                    ),
                    subject_symbol_id=subject,
                    source_refs=[source_ref],
                    unsupported=True,
                    consumer_id=consumer,
                    facet_id=_facet_id(LogicFacetKind.TYPE, subject),
                    actual_fact_ref=f"actual:receiver:{trace.receiver_reference}",
                    expected_fact_ref="expected:resolved_receiver",
                    source_kind=GoalSourceKind.BROKEN_TRACE,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    bound_refs=[trace.caller_span.path],
                )
            )
            return

        # Resolved or missing-local traces produce acceptance + counterexample goals.
        positive = _DraftGoal(
            goal_id=_stable_id(
                "goal",
                obligation=GoalObligationKind.CALLER_INPUT_ACCEPTANCE.value,
                source=source_ref,
                subject=subject,
            ),
            family=GoalFamily.POSITIVE,
            obligation_kind=GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
            disposition=GoalDisposition.OPEN,
            positive_statement_ref=(
                f"stmt:caller_input_acceptance:{trace.receiver_reference}"
            ),
            subject_symbol_id=subject,
            source_refs=[source_ref, *[ref.content_id for ref in trace.evidence_refs]],
            consumer_id=consumer,
            facet_id=_facet_id(LogicFacetKind.TYPE, subject),
            actual_fact_ref=f"actual:receiver:{trace.receiver_reference}",
            expected_fact_ref="expected:receiver_precondition",
            source_kind=GoalSourceKind.BROKEN_TRACE,
            source_ref=source_ref,
            source_authority=SourceAuthorityClass.AUTHORITATIVE,
            bound_refs=[trace.caller_span.path],
        )
        if trace.disposition is TraceDisposition.RESOLVED_MISMATCH:
            positive.family = GoalFamily.REFINEMENT
            positive.negative_target_ref = (
                f"neg:mismatch:{trace.receiver_reference}"
            )
            positive.counterexample_target_ref = (
                f"cex:caller_not_accepted:{trace.caller_symbol_id}"
            )
            # Companion counterexample goal.
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=GoalObligationKind.CALLER_INPUT_ACCEPTANCE.value,
                        source=source_ref,
                        subject=subject,
                        polarity="counterexample",
                    ),
                    family=GoalFamily.COUNTEREXAMPLE,
                    obligation_kind=GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
                    disposition=GoalDisposition.OPEN,
                    positive_statement_ref=(
                        f"stmt:caller_input_acceptance:{trace.receiver_reference}"
                    ),
                    counterexample_target_ref=positive.counterexample_target_ref,
                    subject_symbol_id=subject,
                    source_refs=list(positive.source_refs),
                    consumer_id=consumer,
                    facet_id=_facet_id(LogicFacetKind.TYPE, subject, "cex"),
                    actual_fact_ref=positive.actual_fact_ref,
                    expected_fact_ref=positive.expected_fact_ref,
                    source_kind=GoalSourceKind.BROKEN_TRACE,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    bound_refs=list(positive.bound_refs or []),
                )
            )
        drafts.append(positive)

        for frontier in trace.graph_frontier_refs:
            residual_refs.append(frontier)

    def _ingest_call_requirement(
        self,
        requirement: CallRequirementContract,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        if not isinstance(requirement, CallRequirementContract):
            raise ProgramLogicGoalCompilerError(
                "call_requirements must contain CallRequirementContract values"
            )
        _check_repair_roots(self.roots, requirement.roots, "call_requirement")
        source_ref = requirement.content_id
        subject = f"symbol:caller:{requirement.caller_span.path}"
        consumer = self.roots.consumer_id

        # Caller acceptance + output refinement from requirement surface.
        for obligation, family, statement in (
            (
                GoalObligationKind.CALLER_INPUT_ACCEPTANCE,
                GoalFamily.VALUE,
                f"stmt:caller_input_acceptance:{requirement.trace_id}",
            ),
            (
                GoalObligationKind.OUTPUT_REFINEMENT,
                GoalFamily.REFINEMENT,
                f"stmt:output_refinement:{requirement.trace_id}",
            ),
            (
                GoalObligationKind.ALLOWED_ERRORS,
                GoalFamily.BEHAVIOR,
                f"stmt:allowed_errors:{requirement.trace_id}",
            ),
            (
                GoalObligationKind.PERMITTED_EFFECTS,
                GoalFamily.BEHAVIOR,
                f"stmt:permitted_effects:{requirement.trace_id}",
            ),
            (
                GoalObligationKind.AUTHORIZATION,
                GoalFamily.BEHAVIOR,
                f"stmt:authz:{requirement.trace_id}",
            ),
            (
                GoalObligationKind.RESOURCES,
                GoalFamily.BEHAVIOR,
                f"stmt:resources:{requirement.trace_id}",
            ),
        ):
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=obligation.value,
                        source=source_ref,
                        subject=requirement.trace_id,
                    ),
                    family=family,
                    obligation_kind=obligation,
                    disposition=GoalDisposition.OPEN,
                    positive_statement_ref=statement,
                    subject_symbol_id=subject,
                    source_refs=[
                        source_ref,
                        *[ref.content_id for ref in requirement.requirement_refs],
                        *[ref.content_id for ref in requirement.evidence_refs],
                    ],
                    consumer_id=consumer,
                    facet_id=_facet_id(obligation, requirement.trace_id),
                    actual_fact_ref=f"actual:call_requirement:{requirement.trace_id}",
                    expected_fact_ref=f"expected:{_facet_token(obligation)}",
                    source_kind=GoalSourceKind.CALL_REQUIREMENT,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    bound_refs=[requirement.caller_span.path],
                    counterexample_target_ref=(
                        f"cex:{_facet_token(obligation)}:{requirement.trace_id}"
                    ),
                )
            )

        for unsupported in requirement.unsupported_clause_refs:
            unsupported_refs.append(unsupported)
            residual_refs.append(unsupported)
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag",
                        kind=GoalDiagnosticKind.UNSUPPORTED_SEMANTIC.value,
                        source=unsupported,
                    ),
                    kind=GoalDiagnosticKind.UNSUPPORTED_SEMANTIC,
                    reason_ref=f"reason:unsupported_clause:{unsupported}",
                    related_source_refs=(source_ref, unsupported),
                    consumer_id=consumer,
                )
            )

    def _ingest_delta(
        self,
        delta: ProgramContractDelta,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        if not isinstance(delta, ProgramContractDelta):
            raise ProgramLogicGoalCompilerError(
                "contract_deltas must contain ProgramContractDelta values"
            )
        _check_propagation_roots(self.roots, delta.roots, "contract_delta")
        source_ref = delta.content_id
        subject = delta.subject_symbol_id
        consumer = self.roots.consumer_id

        for clause in delta.clauses:
            obligation = delta_obligation_kind(clause.kind)
            if clause.disposition is DeltaDisposition.UNSUPPORTED:
                unsupported_refs.append(clause.clause_id)
                residual_refs.append(clause.clause_id)
                diagnostics.append(
                    GoalDiagnostic(
                        diagnostic_id=_stable_id(
                            "diag",
                            kind=GoalDiagnosticKind.UNSUPPORTED_SEMANTIC.value,
                            source=clause.clause_id,
                        ),
                        kind=GoalDiagnosticKind.UNSUPPORTED_SEMANTIC,
                        reason_ref=f"reason:delta_unsupported:{clause.clause_id}",
                        related_source_refs=(source_ref, clause.clause_id),
                        consumer_id=consumer,
                        facet_id=_facet_id(obligation, clause.clause_id),
                    )
                )
                drafts.append(
                    _DraftGoal(
                        goal_id=_stable_id(
                            "goal",
                            obligation=obligation.value,
                            source=source_ref,
                            clause=clause.clause_id,
                        ),
                        family=GoalFamily.POSITIVE,
                        obligation_kind=obligation,
                        disposition=GoalDisposition.UNSUPPORTED,
                        positive_statement_ref=(
                            f"stmt:{_facet_token(obligation)}:{clause.clause_id}"
                        ),
                        subject_symbol_id=subject,
                        source_refs=[source_ref, clause.clause_id],
                        unsupported=True,
                        consumer_id=consumer,
                        facet_id=_facet_id(obligation, clause.clause_id),
                        actual_fact_ref=clause.after_contract_ref
                        or f"actual:{clause.clause_id}",
                        expected_fact_ref=clause.before_contract_ref
                        or f"expected:{clause.clause_id}",
                        source_kind=GoalSourceKind.CONTRACT_DELTA,
                        source_ref=source_ref,
                        source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    )
                )
                continue

            if clause.disposition is DeltaDisposition.COMPATIBLE:
                # Compatible clauses do not create open required goals.
                continue

            if clause.disposition is DeltaDisposition.UNKNOWN:
                residual_refs.append(clause.clause_id)
                drafts.append(
                    _DraftGoal(
                        goal_id=_stable_id(
                            "goal",
                            obligation=obligation.value,
                            source=source_ref,
                            clause=clause.clause_id,
                        ),
                        family=GoalFamily.POSITIVE,
                        obligation_kind=obligation,
                        disposition=GoalDisposition.RESIDUAL,
                        positive_statement_ref=(
                            f"stmt:{_facet_token(obligation)}:{clause.clause_id}"
                        ),
                        subject_symbol_id=subject,
                        source_refs=[source_ref, clause.clause_id],
                        consumer_id=consumer,
                        facet_id=_facet_id(obligation, clause.clause_id),
                        actual_fact_ref=clause.after_contract_ref
                        or f"actual:{clause.clause_id}",
                        expected_fact_ref=clause.before_contract_ref
                        or f"expected:{clause.clause_id}",
                        source_kind=GoalSourceKind.CONTRACT_DELTA,
                        source_ref=source_ref,
                        source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    )
                )
                continue

            family = (
                GoalFamily.REFINEMENT
                if clause.disposition is DeltaDisposition.BREAKING
                else GoalFamily.BEHAVIOR
            )
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=obligation.value,
                        source=source_ref,
                        clause=clause.clause_id,
                    ),
                    family=family,
                    obligation_kind=obligation,
                    disposition=GoalDisposition.OPEN,
                    positive_statement_ref=(
                        f"stmt:{_facet_token(obligation)}:{clause.clause_id}"
                    ),
                    negative_target_ref=(
                        f"neg:{_facet_token(obligation)}:{clause.clause_id}"
                        if clause.disposition is DeltaDisposition.BREAKING
                        else ""
                    ),
                    counterexample_target_ref=(
                        f"cex:{_facet_token(obligation)}:{clause.clause_id}"
                    ),
                    subject_symbol_id=subject,
                    source_refs=[
                        source_ref,
                        clause.clause_id,
                        *delta.evidence_refs,
                    ],
                    consumer_id=consumer,
                    facet_id=_facet_id(obligation, clause.clause_id),
                    actual_fact_ref=clause.after_contract_ref
                    or f"actual:{clause.clause_id}",
                    expected_fact_ref=clause.before_contract_ref
                    or f"expected:{clause.clause_id}",
                    source_kind=GoalSourceKind.CONTRACT_DELTA,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                )
            )

    def _ingest_consumer(
        self,
        obligation: ConsumerMigrationObligation,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        del unsupported_refs  # reserved for future frontier taxonomy expansion
        if not isinstance(obligation, ConsumerMigrationObligation):
            raise ProgramLogicGoalCompilerError(
                "consumer_obligations must contain ConsumerMigrationObligation values"
            )
        _check_propagation_roots(self.roots, obligation.roots, "consumer_obligation")
        source_ref = obligation.content_id
        consumer = obligation.consumer_id
        subject = obligation.node.symbol_id if hasattr(obligation.node, "symbol_id") else (
            getattr(obligation.node, "node_id", consumer)
        )
        if not isinstance(subject, str) or not subject.strip():
            subject = consumer

        if obligation.disposition is ConsumerDisposition.FRONTIER:
            residual_refs.append(source_ref)
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag",
                        kind=GoalDiagnosticKind.FRONTIER_CONSUMER.value,
                        source=source_ref,
                    ),
                    kind=GoalDiagnosticKind.FRONTIER_CONSUMER,
                    reason_ref=f"reason:frontier_consumer:{consumer}",
                    related_source_refs=(source_ref,),
                    consumer_id=consumer,
                )
            )
            return

        if obligation.disposition in {
            ConsumerDisposition.COMPATIBLE,
            ConsumerDisposition.EXCLUDED,
            ConsumerDisposition.ABSTAIN,
        }:
            return

        if obligation.disposition not in _RESOLVED_CONSUMERS:
            residual_refs.append(source_ref)
            return

        # Required migration goal for the consumer itself.
        drafts.append(
            _DraftGoal(
                goal_id=_stable_id(
                    "goal",
                    obligation=GoalObligationKind.CONSUMER_MIGRATION.value,
                    source=source_ref,
                    consumer=consumer,
                ),
                family=GoalFamily.POSITIVE,
                obligation_kind=GoalObligationKind.CONSUMER_MIGRATION,
                disposition=GoalDisposition.OPEN,
                positive_statement_ref=(
                    f"stmt:consumer_migration:{consumer}:{obligation.disposition.value}"
                ),
                subject_symbol_id=subject,
                source_refs=[source_ref, *obligation.clause_ids, *obligation.proof_refs],
                consumer_id=consumer,
                facet_id=_facet_id(LogicFacetKind.PLACEMENT, consumer),
                actual_fact_ref=f"actual:consumer:{consumer}",
                expected_fact_ref=f"expected:migration:{obligation.disposition.value}",
                source_kind=GoalSourceKind.CONSUMER_OBLIGATION,
                source_ref=source_ref,
                source_authority=SourceAuthorityClass.AUTHORITATIVE,
                counterexample_target_ref=f"cex:consumer_unmigrated:{consumer}",
                bound_refs=list(obligation.invalidation_refs),
            )
        )

        # One required goal per clause/facet carried by the obligation.
        for clause_id in obligation.clause_ids:
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=GoalObligationKind.COMPATIBILITY.value,
                        source=source_ref,
                        consumer=consumer,
                        clause=clause_id,
                    ),
                    family=GoalFamily.REFINEMENT,
                    obligation_kind=GoalObligationKind.COMPATIBILITY,
                    disposition=GoalDisposition.OPEN,
                    positive_statement_ref=f"stmt:compatibility:{consumer}:{clause_id}",
                    subject_symbol_id=subject,
                    source_refs=[source_ref, clause_id],
                    consumer_id=consumer,
                    facet_id=_facet_id(LogicFacetKind.SCHEMA, consumer, clause_id),
                    actual_fact_ref=f"actual:clause:{clause_id}",
                    expected_fact_ref=f"expected:clause:{clause_id}",
                    source_kind=GoalSourceKind.CONSUMER_OBLIGATION,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    counterexample_target_ref=(
                        f"cex:clause_break:{consumer}:{clause_id}"
                    ),
                )
            )

    def _ingest_missing_input(
        self,
        missing: MissingInputRequirement,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        del diagnostics, residual_refs, unsupported_refs
        if not isinstance(missing, MissingInputRequirement):
            raise ProgramLogicGoalCompilerError(
                "missing_inputs must contain MissingInputRequirement values"
            )
        _check_propagation_roots(self.roots, missing.roots, "missing_input")
        source_ref = missing.content_id
        consumer = self.roots.consumer_id
        subject = f"symbol:param:{missing.parameter_name}"

        facet_goals: list[tuple[GoalObligationKind, GoalFamily, str, str]] = [
            (
                GoalObligationKind.VALUE_SUFFICIENCY,
                GoalFamily.VALUE,
                f"stmt:value_sufficiency:{missing.parameter_name}",
                _facet_id(LogicFacetKind.INFORMATION, missing.requirement_id),
            ),
            (
                GoalObligationKind.INFORMATION_PROVENANCE,
                GoalFamily.VALUE,
                f"stmt:information_provenance:{missing.information_content_ref}",
                _facet_id(LogicFacetKind.INFORMATION, missing.information_content_ref),
            ),
            (
                GoalObligationKind.NULLABILITY,
                GoalFamily.POSITIVE,
                f"stmt:nullability:{missing.nullability}",
                _facet_id(LogicFacetKind.TYPE, "nullability", missing.requirement_id),
            ),
            (
                GoalObligationKind.RANGE,
                GoalFamily.POSITIVE,
                f"stmt:range:{missing.type_ref}",
                _facet_id(LogicFacetKind.TYPE, "range", missing.requirement_id),
            ),
            (
                GoalObligationKind.TOTALITY,
                GoalFamily.POSITIVE,
                f"stmt:totality:{missing.parameter_name}",
                _facet_id(LogicFacetKind.TYPE, "totality", missing.requirement_id),
            ),
        ]
        if missing.allowed_error_refs:
            facet_goals.append(
                (
                    GoalObligationKind.ALLOWED_ERRORS,
                    GoalFamily.BEHAVIOR,
                    f"stmt:allowed_errors:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.ERROR, missing.requirement_id),
                )
            )
        if missing.effect_refs:
            facet_goals.append(
                (
                    GoalObligationKind.PERMITTED_EFFECTS,
                    GoalFamily.BEHAVIOR,
                    f"stmt:permitted_effects:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.EFFECT, missing.requirement_id),
                )
            )
        if missing.capability_refs or missing.authorization_refs:
            facet_goals.append(
                (
                    GoalObligationKind.AUTHORIZATION,
                    GoalFamily.BEHAVIOR,
                    f"stmt:authz:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.AUTHORIZATION, missing.requirement_id),
                )
            )
        if missing.resource_refs:
            facet_goals.append(
                (
                    GoalObligationKind.RESOURCES,
                    GoalFamily.BEHAVIOR,
                    f"stmt:resources:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.RESOURCE, missing.requirement_id),
                )
            )
        if missing.ownership_refs:
            facet_goals.append(
                (
                    GoalObligationKind.OWNERSHIP,
                    GoalFamily.POSITIVE,
                    f"stmt:ownership:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.MEMORY, "ownership", missing.requirement_id),
                )
            )
        if missing.construction_precondition_refs:
            facet_goals.append(
                (
                    GoalObligationKind.CONSTRUCTOR,
                    GoalFamily.BEHAVIOR,
                    f"stmt:constructor:{missing.requirement_id}",
                    _facet_id(LogicFacetKind.SCHEMA, "constructor", missing.requirement_id),
                )
            )

        for obligation, family, statement, facet_id in facet_goals:
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=obligation.value,
                        source=source_ref,
                        param=missing.parameter_name,
                    ),
                    family=family,
                    obligation_kind=obligation,
                    disposition=GoalDisposition.OPEN,
                    positive_statement_ref=statement,
                    subject_symbol_id=subject,
                    source_refs=[
                        source_ref,
                        missing.requirement_id,
                        *missing.proof_refs,
                    ],
                    consumer_id=consumer,
                    facet_id=facet_id,
                    actual_fact_ref=f"actual:missing:{missing.parameter_name}",
                    expected_fact_ref=f"expected:{missing.type_ref}",
                    source_kind=GoalSourceKind.MISSING_INPUT,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.AUTHORITATIVE,
                    counterexample_target_ref=(
                        f"cex:missing_value:{missing.parameter_name}"
                    ),
                    assumption_refs=list(missing.construction_precondition_refs),
                    bound_refs=[
                        f"bound:depth:{missing.propagation_depth_bound}",
                    ],
                )
            )

    def _ingest_behavior(
        self,
        behavior: RequiredBehaviorContract,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        if not isinstance(behavior, RequiredBehaviorContract):
            raise ProgramLogicGoalCompilerError(
                "behavior_contracts must contain RequiredBehaviorContract values"
            )
        _check_propagation_roots(self.roots, behavior.roots, "required_behavior")
        source_ref = behavior.content_id
        subject = behavior.subject_symbol_id
        consumer = self.roots.consumer_id
        nominating = (
            behavior.evidence_precedence
            is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            or behavior.implementation_hypothesis
        )
        authority = (
            SourceAuthorityClass.NOMINATING
            if nominating
            else SourceAuthorityClass.AUTHORITATIVE
        )
        source_kind = (
            GoalSourceKind.IMPLEMENTATION_HYPOTHESIS
            if nominating
            else GoalSourceKind.REQUIRED_BEHAVIOR
        )

        if nominating:
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag",
                        kind=GoalDiagnosticKind.NON_AUTHORITATIVE_SOURCE.value,
                        source=source_ref,
                    ),
                    kind=GoalDiagnosticKind.NON_AUTHORITATIVE_SOURCE,
                    reason_ref=f"reason:implementation_hypothesis:{behavior.behavior_id}",
                    related_source_refs=(source_ref,),
                    consumer_id=consumer,
                )
            )

        pairs: list[tuple[GoalObligationKind, Sequence[str]]] = [
            (GoalObligationKind.SCHEMA, behavior.field_refs),
            (GoalObligationKind.CONSTRUCTOR, behavior.constructor_refs),
            (GoalObligationKind.STATE, behavior.state_transition_refs),
            (GoalObligationKind.PERMITTED_EFFECTS, behavior.effect_refs),
            (GoalObligationKind.CAPABILITIES, behavior.capability_refs),
            (GoalObligationKind.AUTHORIZATION, behavior.authorization_refs),
            (GoalObligationKind.RESOURCES, behavior.resource_refs),
        ]
        if behavior.placement_decision_ref:
            pairs.append(
                (GoalObligationKind.PLACEMENT, (behavior.placement_decision_ref,))
            )
        if behavior.method_refs:
            pairs.append((GoalObligationKind.PLACEMENT, behavior.method_refs))
        if behavior.invariant_refs:
            pairs.append((GoalObligationKind.CONSISTENCY, behavior.invariant_refs))

        emitted = False
        for obligation, refs in pairs:
            if not refs and obligation is not GoalObligationKind.PLACEMENT:
                continue
            if not refs:
                continue
            emitted = True
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=obligation.value,
                        source=source_ref,
                        subject=subject,
                    ),
                    family=GoalFamily.BEHAVIOR,
                    obligation_kind=obligation,
                    disposition=(
                        GoalDisposition.OPEN
                        if not nominating
                        else GoalDisposition.OPEN
                    ),
                    positive_statement_ref=(
                        f"stmt:{_facet_token(obligation)}:{behavior.behavior_id}"
                    ),
                    subject_symbol_id=subject,
                    source_refs=[source_ref, *refs, *behavior.proof_refs],
                    consumer_id=consumer,
                    facet_id=_facet_id(obligation, behavior.behavior_id),
                    actual_fact_ref=f"actual:behavior:{behavior.behavior_id}",
                    expected_fact_ref=f"expected:{_facet_token(obligation)}",
                    source_kind=source_kind,
                    source_ref=source_ref,
                    source_authority=authority,
                    nominating_only=nominating,
                    counterexample_target_ref=(
                        f"cex:{_facet_token(obligation)}:{behavior.behavior_id}"
                    ),
                    assumption_refs=[] if not nominating else list(refs),
                )
            )

        if not emitted:
            residual_refs.append(source_ref)
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag",
                        kind=GoalDiagnosticKind.MISSING_FACET.value,
                        source=source_ref,
                    ),
                    kind=GoalDiagnosticKind.MISSING_FACET,
                    reason_ref=f"reason:empty_behavior:{behavior.behavior_id}",
                    related_source_refs=(source_ref,),
                    consumer_id=consumer,
                )
            )

    def _ingest_memory_facet(
        self,
        facet: MemorySafetyFacet,
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
        residual_refs: list[str],
        unsupported_refs: list[str],
    ) -> None:
        if not isinstance(facet, MemorySafetyFacet):
            raise ProgramLogicGoalCompilerError(
                "memory_facets must contain MemorySafetyFacet values"
            )
        _check_repair_roots(self.roots, facet.roots, "memory_safety_facet")
        source_ref = facet.content_id
        subject = f"symbol:span:{facet.subject_span.path}"
        consumer = self.roots.consumer_id

        if facet.disposition is MemorySafetyDisposition.UNSUPPORTED:
            unsupported_refs.extend(facet.unsupported_refs)
            residual_refs.append(source_ref)
            diagnostics.append(
                GoalDiagnostic(
                    diagnostic_id=_stable_id(
                        "diag",
                        kind=GoalDiagnosticKind.NATIVE_BOUNDARY.value,
                        source=source_ref,
                    ),
                    kind=GoalDiagnosticKind.NATIVE_BOUNDARY,
                    reason_ref=f"reason:memory_unsupported:{facet.language_runtime}",
                    related_source_refs=(source_ref, *facet.unsupported_refs),
                    consumer_id=consumer,
                    facet_id=_facet_id(LogicFacetKind.MEMORY, facet.language_runtime),
                )
            )
            disposition = GoalDisposition.UNSUPPORTED
            unsupported = True
        elif facet.disposition is MemorySafetyDisposition.STALE:
            residual_refs.append(source_ref)
            disposition = GoalDisposition.STALE
            unsupported = False
        elif facet.disposition is MemorySafetyDisposition.ERROR:
            residual_refs.append(source_ref)
            disposition = GoalDisposition.RESIDUAL
            unsupported = False
        else:
            disposition = GoalDisposition.OPEN
            unsupported = False

        for obligation in (
            GoalObligationKind.MEMORY_SAFETY,
            GoalObligationKind.OWNERSHIP,
            GoalObligationKind.LIFETIME,
        ):
            drafts.append(
                _DraftGoal(
                    goal_id=_stable_id(
                        "goal",
                        obligation=obligation.value,
                        source=source_ref,
                        lang=facet.language_runtime,
                    ),
                    family=GoalFamily.POSITIVE,
                    obligation_kind=obligation,
                    disposition=disposition,
                    positive_statement_ref=(
                        f"stmt:{_facet_token(obligation)}:{facet.language_runtime}"
                    ),
                    subject_symbol_id=subject,
                    source_refs=[
                        source_ref,
                        *[ref.content_id for ref in facet.evidence_refs],
                        *[ref.content_id for ref in facet.proof_refs],
                    ],
                    unsupported=unsupported,
                    consumer_id=consumer,
                    facet_id=_facet_id(obligation, facet.language_runtime),
                    actual_fact_ref=f"actual:memory:{facet.disposition.value}",
                    expected_fact_ref=f"expected:{_facet_token(obligation)}",
                    source_kind=GoalSourceKind.MEMORY_SAFETY,
                    source_ref=source_ref,
                    source_authority=SourceAuthorityClass.CONFORMANCE,
                    counterexample_target_ref=(
                        f"cex:{_facet_token(obligation)}:{facet.language_runtime}"
                    ),
                    bound_refs=[facet.subject_span.path],
                )
            )

    def _ingest_prose(
        self,
        nomination: ProseGoalNomination | Mapping[str, Any],
        drafts: list[_DraftGoal],
        diagnostics: list[GoalDiagnostic],
    ) -> None:
        if isinstance(nomination, Mapping):
            nomination = ProseGoalNomination(
                obligation_kind=nomination["obligation_kind"],
                statement_ref=nomination["statement_ref"],
                source_kind=nomination.get("source_kind", GoalSourceKind.TASK_PROSE),
                subject_symbol_id=nomination.get("subject_symbol_id", ""),
                consumer_id=nomination.get("consumer_id", ""),
            )
        if not isinstance(nomination, ProseGoalNomination):
            raise ProgramLogicGoalCompilerError(
                "prose_nominations must be ProseGoalNomination values"
            )
        consumer = nomination.consumer_id or self.roots.consumer_id
        subject = nomination.subject_symbol_id or "symbol:prose"
        source_ref = nomination.statement_ref
        goal_id = _stable_id(
            "goal",
            obligation=nomination.obligation_kind.value,
            source=source_ref,
            prose="1",
        )
        drafts.append(
            _DraftGoal(
                goal_id=goal_id,
                family=GoalFamily.POSITIVE,
                obligation_kind=nomination.obligation_kind,
                disposition=GoalDisposition.OPEN,
                positive_statement_ref=nomination.statement_ref,
                subject_symbol_id=subject,
                source_refs=[source_ref],
                nominating_only=True,
                consumer_id=consumer,
                facet_id=_facet_id(nomination.obligation_kind, "prose", source_ref),
                actual_fact_ref="",
                expected_fact_ref=nomination.statement_ref,
                source_kind=nomination.source_kind,
                source_ref=source_ref,
                source_authority=SourceAuthorityClass.NOMINATING,
            )
        )
        diagnostics.append(
            GoalDiagnostic(
                diagnostic_id=_stable_id(
                    "diag",
                    kind=GoalDiagnosticKind.PROSE_NOMINATION.value,
                    source=source_ref,
                ),
                kind=GoalDiagnosticKind.PROSE_NOMINATION,
                reason_ref=f"reason:prose_cannot_satisfy:{source_ref}",
                related_goal_ids=(goal_id,),
                related_source_refs=(source_ref,),
                consumer_id=consumer,
                facet_id=_facet_id(LogicFacetKind.INFORMATION, "prose", source_ref),
            )
        )

    # -- conflict detection --------------------------------------------------

    def _detect_conflicts(
        self, drafts: list[_DraftGoal]
    ) -> tuple[list[_DraftGoal], list[GoalDiagnostic]]:
        """Same consumer/facet with incompatible expected facts → diagnostic."""
        diagnostics: list[GoalDiagnostic] = []
        # Group authoritative (non-nominating) drafts by (consumer, obligation, facet).
        groups: dict[tuple[str, str, str], list[_DraftGoal]] = {}
        for draft in drafts:
            if draft.nominating_only or draft.unsupported:
                continue
            if draft.disposition in {
                GoalDisposition.UNSUPPORTED,
                GoalDisposition.RESIDUAL,
                GoalDisposition.STALE,
            }:
                continue
            key = (
                draft.consumer_id,
                draft.obligation_kind.value,
                draft.facet_id or draft.obligation_kind.value,
            )
            groups.setdefault(key, []).append(draft)

        for key, members in groups.items():
            expected = {
                item.expected_fact_ref
                for item in members
                if item.expected_fact_ref
            }
            if len(expected) > 1:
                goal_ids = tuple(sorted(item.goal_id for item in members))
                source_refs = tuple(sorted({item.source_ref for item in members}))
                diagnostics.append(
                    GoalDiagnostic(
                        diagnostic_id=_stable_id(
                            "diag",
                            kind=GoalDiagnosticKind.CONFLICTING_INTENT.value,
                            consumer=key[0],
                            obligation=key[1],
                            facet=key[2],
                        ),
                        kind=GoalDiagnosticKind.CONFLICTING_INTENT,
                        reason_ref=(
                            f"reason:conflicting_expected:"
                            f"{key[1]}:{'+'.join(sorted(expected))}"
                        ),
                        related_goal_ids=goal_ids,
                        related_source_refs=source_refs,
                        consumer_id=key[0],
                        facet_id=key[2],
                    )
                )
                # Mark members as residual so conflict is explicit; do not invent
                # a resolution.
                for item in members:
                    item.disposition = GoalDisposition.RESIDUAL
                    # Add consistency goal companion once per conflict group.
                consistency_id = _stable_id(
                    "goal",
                    obligation=GoalObligationKind.CONSISTENCY.value,
                    consumer=key[0],
                    facet=key[2],
                    conflict="1",
                )
                drafts.append(
                    _DraftGoal(
                        goal_id=consistency_id,
                        family=GoalFamily.CONSISTENCY,
                        obligation_kind=GoalObligationKind.CONSISTENCY,
                        disposition=GoalDisposition.OPEN,
                        positive_statement_ref=(
                            f"stmt:consistency:{key[0]}:{key[1]}:{key[2]}"
                        ),
                        subject_symbol_id=members[0].subject_symbol_id or key[0],
                        source_refs=list(source_refs),
                        consumer_id=key[0],
                        facet_id=key[2],
                        actual_fact_ref="+".join(sorted(expected)),
                        expected_fact_ref=f"expected:unique:{key[1]}",
                        source_kind=members[0].source_kind,
                        source_ref=members[0].source_ref,
                        source_authority=SourceAuthorityClass.AUTHORITATIVE,
                        counterexample_target_ref=(
                            f"cex:conflicting_intent:{key[0]}:{key[1]}"
                        ),
                    )
                )
        return drafts, diagnostics

    def _assert_consumer_facet_coverage(
        self,
        drafts: list[_DraftGoal],
        consumer_obligations: Sequence[ConsumerMigrationObligation],
        diagnostics: list[GoalDiagnostic],
    ) -> None:
        """Ensure every resolved consumer has at least one required goal."""
        del diagnostics
        required_by_consumer: dict[str, list[_DraftGoal]] = {}
        for draft in drafts:
            if not draft.consumer_id:
                continue
            if draft.nominating_only:
                continue
            if draft.disposition not in {
                GoalDisposition.OPEN,
                GoalDisposition.PLANNED,
                GoalDisposition.ADMITTED,
                GoalDisposition.RESIDUAL,
            }:
                continue
            required_by_consumer.setdefault(draft.consumer_id, []).append(draft)

        for obligation in consumer_obligations:
            if obligation.disposition not in _RESOLVED_CONSUMERS:
                continue
            if obligation.consumer_id not in required_by_consumer:
                # Compiler always emits consumer_migration for resolved consumers;
                # this is a fail-closed integrity check.
                raise ProgramLogicGoalCompilerError(
                    f"resolved consumer {obligation.consumer_id!r} lacks a required goal"
                )

    # -- materialization -----------------------------------------------------

    def _materialize(
        self, drafts: list[_DraftGoal]
    ) -> tuple[tuple[ProgramLogicGoal, ...], tuple[GoalSourceBinding, ...]]:
        goals: list[ProgramLogicGoal] = []
        bindings: list[GoalSourceBinding] = []
        seen_goal_ids: set[str] = set()

        for draft in sorted(drafts, key=lambda item: item.goal_id):
            if draft.goal_id in seen_goal_ids:
                # Deterministic merge: keep first, attach additional binding.
                existing_bindings = [
                    item for item in bindings if item.goal_id == draft.goal_id
                ]
                if existing_bindings and draft.source_ref:
                    binding = self._binding_from_draft(draft)
                    if binding.binding_id not in {
                        item.binding_id for item in bindings
                    }:
                        bindings.append(binding)
                continue
            seen_goal_ids.add(draft.goal_id)

            facet_kind = obligation_facet_kind(draft.obligation_kind)
            contract_ref = draft.expected_fact_ref or draft.positive_statement_ref
            # Memory/resource separation: never put resource: on memory facets.
            if facet_kind is LogicFacetKind.MEMORY and contract_ref.startswith(
                "resource:"
            ):
                contract_ref = f"memory:{contract_ref.split(':', 1)[-1]}"
            if facet_kind is LogicFacetKind.RESOURCE and contract_ref.startswith(
                "memory:"
            ):
                contract_ref = f"resource:{contract_ref.split(':', 1)[-1]}"
            if facet_kind is LogicFacetKind.TYPE and contract_ref.startswith(
                ("memory:", "resource:")
            ):
                contract_ref = f"type:{contract_ref.split(':', 1)[-1]}"

            required_facets: tuple[LogicFacetRef, ...] = ()
            unsupported_facets: tuple[LogicFacetRef, ...] = ()
            facet = LogicFacetRef(
                facet_id=draft.facet_id
                or _facet_id(facet_kind, draft.goal_id),
                kind=facet_kind,
                subject_symbol_id=draft.subject_symbol_id or "symbol:unknown",
                contract_ref=contract_ref if not draft.unsupported else "",
                unsupported=draft.unsupported,
            )
            if draft.unsupported:
                unsupported_facets = (facet,)
            else:
                required_facets = (facet,)

            family = draft.family
            negative = draft.negative_target_ref
            counterexample = draft.counterexample_target_ref
            if family is GoalFamily.NEGATIVE and not negative:
                negative = draft.counterexample_target_ref or (
                    f"neg:{_facet_token(draft.obligation_kind)}"
                )
            if family is GoalFamily.COUNTEREXAMPLE and not counterexample:
                counterexample = (
                    draft.negative_target_ref
                    or f"cex:{_facet_token(draft.obligation_kind)}"
                )

            assumption_authority = (
                SourceAuthorityClass.NOMINATING
                if draft.nominating_only
                else (
                    draft.source_authority
                    if draft.assumption_refs
                    else SourceAuthorityClass.NONE
                )
            )

            invalidation = tuple(
                sorted(
                    {
                        self.roots.tree_id,
                        self.roots.corpus_id,
                        self.roots.policy_id,
                        *(draft.bound_refs or ()),
                    }
                )
            )

            try:
                goal = ProgramLogicGoal(
                    roots=self.roots,
                    goal_id=draft.goal_id,
                    family=family,
                    disposition=draft.disposition,
                    positive_statement_ref=draft.positive_statement_ref,
                    negative_target_ref=negative,
                    counterexample_target_ref=counterexample,
                    affected_symbol_ids=(
                        (draft.subject_symbol_id,) if draft.subject_symbol_id else ()
                    ),
                    source_refs=tuple(sorted(set(draft.source_refs or ()))),
                    required_facets=required_facets,
                    unsupported_facets=unsupported_facets,
                    assumption_refs=tuple(sorted(set(draft.assumption_refs or ()))),
                    assumption_authority=assumption_authority,
                    proof_status=ProofStatus.UNPROVED,
                    logic_family_refs=(draft.obligation_kind.value,),
                    bound_refs=tuple(sorted(set(draft.bound_refs or ()))),
                    invalidation_refs=invalidation,
                )
            except ProgramLogicPredictionError as exc:
                raise ProgramLogicGoalCompilerError(
                    f"failed to materialize goal {draft.goal_id}: {exc}"
                ) from exc

            goals.append(goal)
            bindings.append(self._binding_from_draft(draft))

        if len(goals) > MAX_GOALS:
            raise ProgramLogicGoalCompilerBoundsError("goals exceed compilation bound")
        if len(bindings) > MAX_BINDINGS:
            raise ProgramLogicGoalCompilerBoundsError(
                "source bindings exceed compilation bound"
            )
        return tuple(goals), tuple(bindings)

    def _binding_from_draft(self, draft: _DraftGoal) -> GoalSourceBinding:
        return GoalSourceBinding(
            binding_id=_stable_id(
                "binding",
                goal=draft.goal_id,
                source=draft.source_ref or draft.goal_id,
                kind=draft.source_kind.value,
            ),
            goal_id=draft.goal_id,
            source_kind=draft.source_kind,
            source_ref=draft.source_ref or draft.goal_id,
            source_authority=draft.source_authority,
            obligation_kind=draft.obligation_kind,
            consumer_id=draft.consumer_id,
            facet_id=draft.facet_id,
            actual_fact_ref=draft.actual_fact_ref,
            expected_fact_ref=draft.expected_fact_ref,
            counterexample_target_ref=draft.counterexample_target_ref,
            assumption_refs=tuple(draft.assumption_refs or ()),
            bound_refs=tuple(draft.bound_refs or ()),
            nominating_only=draft.nominating_only,
        )


def compile_program_logic_goals(
    roots: ProgramLogicAuthorityRoots,
    **kwargs: Any,
) -> ProgramLogicGoalCompilation:
    """Module-level entry point matching other analysis compilers."""
    return ProgramLogicGoalCompiler(roots).compile(**kwargs)


__all__ = [
    "GOAL_SOURCE_BINDING_SCHEMA",
    "GOAL_DIAGNOSTIC_SCHEMA",
    "PROGRAM_LOGIC_GOAL_COMPILATION_SCHEMA",
    "PRODUCER_ID",
    "CONTRACT_VERSION",
    "MAX_GOALS",
    "MAX_BINDINGS",
    "MAX_DIAGNOSTICS",
    "ProgramLogicGoalCompilerError",
    "ProgramLogicGoalCompilerAuthorityError",
    "ProgramLogicGoalCompilerBoundsError",
    "ProgramLogicGoalConflictError",
    "GoalObligationKind",
    "GoalSourceKind",
    "CompilationDisposition",
    "GoalDiagnosticKind",
    "GoalSourceBinding",
    "GoalDiagnostic",
    "ProgramLogicGoalCompilation",
    "ProseGoalNomination",
    "ProgramLogicGoalCompiler",
    "compile_program_logic_goals",
    "all_obligation_kinds",
    "all_source_kinds",
    "obligation_facet_kind",
    "delta_obligation_kind",
    "is_nominating_source",
    "source_authority_for_kind",
    # Re-export for AST symbol surface expected by the plan.
    "GoalFamily",
]
