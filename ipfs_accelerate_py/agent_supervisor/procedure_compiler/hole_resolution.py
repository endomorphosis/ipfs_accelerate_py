"""Typed residual-hole routing and independent candidate validation.

Holes are the only path from a procedure to an approved provider.  This
module owns routing and validation, not authority: a model may propose a
bounded candidate, never a decision.  Deterministic and cache routes are
attempted before any model class.  Identical failure or unchanged evidence
suppresses another provider call.

Distilled exact-cache, rule, classifier, and local resolvers inject through
the same provider-class ports.  Route order is exact cache -> declarative
rule -> deterministic classifier -> small local model -> remote model.
No injected route may skip validation or claim correctness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..context.context_compiler import (
    ContextCompilationError,
    ContextCompiler,
    RequiredContextOverflowError,
)
from ..context.context_contracts import ContextBudget, ContextReference, ContextTier
from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from ..semantic_state.contracts import ModelRoute
from ..semantic_state.providers import capability_for_route
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    FORBIDDEN_HOLE_TYPES,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    HoleType,
    ProcedureContractError,
    ProcedureHole,
    ProviderClass,
    _bounded,
    _decode_fields,
    _enum,
    _enums,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
)


HOLE_RESOLVER_REVISION: Final[str] = "HoleResolver@1"
HOLE_VALIDATOR_REVISION: Final[str] = "HoleResolutionValidator@1"
HOLE_CALLER: Final[str] = "procedure-compiler:hole-resolver"
HOLE_STAGE: Final[str] = "typed-hole-resolution"
MAX_HOLE_ATTEMPTS: Final[int] = 4
MAX_HOLE_CONTEXT_BYTES: Final[int] = 1_048_576
MAX_HOLE_TOKENS: Final[int] = 32_768
BYTES_PER_TOKEN: Final[int] = 4

ALLOWED_HOLE_TYPES: Final[frozenset[str]] = frozenset(item.value for item in HoleType)

PROVIDER_ROUTE_ORDER: Final[tuple[ProviderClass, ...]] = (
    ProviderClass.EXACT_CACHE,
    ProviderClass.DECLARATIVE_RULE,
    ProviderClass.DETERMINISTIC_CLASSIFIER,
    ProviderClass.LOCAL_SMALL_MODEL,
    ProviderClass.REMOTE_STANDARD_MODEL,
    ProviderClass.REMOTE_STRONG_MODEL,
    ProviderClass.HUMAN,
)
LOCAL_HOLE_ROUTE_ORDER: Final[tuple[ProviderClass, ...]] = (
    ProviderClass.EXACT_CACHE,
    ProviderClass.DECLARATIVE_RULE,
    ProviderClass.DETERMINISTIC_CLASSIFIER,
    ProviderClass.LOCAL_SMALL_MODEL,
    ProviderClass.REMOTE_STANDARD_MODEL,
)

DETERMINISTIC_PROVIDER_CLASSES: Final[frozenset[ProviderClass]] = frozenset(
    {
        ProviderClass.EXACT_CACHE,
        ProviderClass.DECLARATIVE_RULE,
        ProviderClass.DETERMINISTIC_CLASSIFIER,
    }
)
MODEL_PROVIDER_CLASSES: Final[frozenset[ProviderClass]] = frozenset(
    {
        ProviderClass.LOCAL_SMALL_MODEL,
        ProviderClass.REMOTE_STANDARD_MODEL,
        ProviderClass.REMOTE_STRONG_MODEL,
    }
)

PROVIDER_CLASS_TO_MODEL_ROUTE: Final[Mapping[ProviderClass, ModelRoute]] = MappingProxyType(
    {
        ProviderClass.EXACT_CACHE: ModelRoute.DETERMINISTIC_ONLY,
        ProviderClass.DECLARATIVE_RULE: ModelRoute.DETERMINISTIC_ONLY,
        ProviderClass.DETERMINISTIC_CLASSIFIER: ModelRoute.DETERMINISTIC_ONLY,
        ProviderClass.LOCAL_SMALL_MODEL: ModelRoute.SMALL_LOCAL_MODEL,
        ProviderClass.REMOTE_STANDARD_MODEL: ModelRoute.MEDIUM_MODEL,
        ProviderClass.REMOTE_STRONG_MODEL: ModelRoute.FRONTIER_MODEL,
        ProviderClass.HUMAN: ModelRoute.HUMAN_REVIEW_REQUIRED,
    }
)

_REQUIRED_INPUT_KEYS: Final[Mapping[HoleType, tuple[str, ...]]] = MappingProxyType(
    {
        HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS: ("allowed_values",),
        HoleType.GENERATE_DOCSTRING: ("symbol",),
        HoleType.PROPOSE_BOUNDED_PATCH: ("template_ids",),
        HoleType.CLASSIFY_FAILURE: ("failure_signature",),
        HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE: ("template_ids",),
        HoleType.SUGGEST_MISSING_TEST_CASE: ("uncovered_obligation",),
        HoleType.SUGGEST_LEMMA: ("obligation",),
    }
)
_REQUIRED_OUTPUT_KEYS: Final[Mapping[HoleType, tuple[str, ...]]] = MappingProxyType(
    {
        HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS: ("selected",),
        HoleType.GENERATE_DOCSTRING: ("docstring",),
        HoleType.PROPOSE_BOUNDED_PATCH: ("template_id",),
        HoleType.CLASSIFY_FAILURE: ("failure_class",),
        HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE: ("template_id",),
        HoleType.SUGGEST_MISSING_TEST_CASE: ("test_name",),
        HoleType.SUGGEST_LEMMA: ("lemma_name",),
    }
)
_GENERIC_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "bindings",
        "artifact_version",
        "state",
        "subject_cid",
        "reference_cids",
        "labels",
        "facts",
        "created_at_ms",
    }
)
_AUTHORITY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "accept_proof",
        "authority_decision",
        "claim_completion",
        "claim_correctness",
        "claims_correctness",
        "complete_task",
        "confirmation",
        "correctness_claim",
        "disable_validation",
        "grant_authority",
        "omit_proof",
        "omit_tests",
        "policy_decision",
        "promote",
        "proof_acceptance",
        "release_promotion",
        "self_validate",
        "skip_proof",
        "skip_validation",
        "task_completion",
        "test_omission",
        "trusted_key",
        "unbounded_shell",
    }
)
_PROVIDER_OUTCOMES: Final[frozenset[str]] = frozenset({"proposed", "missed", "failed"})


class HoleResolutionError(ProcedureContractError):
    """A typed hole request, route, or candidate is unsafe."""


class HoleTypeError(HoleResolutionError):
    """The hole type is forbidden or outside the closed vocabulary."""


class HoleBoundError(HoleResolutionError):
    """A context, token, or attempt bound was exceeded."""


class HoleValidationError(HoleResolutionError):
    """A candidate failed independent validation or tried to leave candidate tier."""


class HoleResolutionAction(str, Enum):
    PROPOSE = "propose"
    FALLBACK = "fallback"
    SUPPRESS = "suppress"
    REFUSE = "refuse"


class HoleResolutionReason(str, Enum):
    CANDIDATE_PROPOSED = "candidate-proposed"
    FORBIDDEN_HOLE_TYPE = "forbidden-hole-type"
    UNKNOWN_HOLE_TYPE = "unknown-hole-type"
    PROVIDER_NOT_ALLOWED = "provider-not-allowed"
    CONTEXT_BUDGET_EXCEEDED = "context-budget-exceeded"
    TOKEN_BUDGET_EXCEEDED = "token-budget-exceeded"
    ATTEMPT_BUDGET_EXCEEDED = "attempt-budget-exceeded"
    IDENTICAL_FAILURE = "identical-failure"
    NO_NEW_EVIDENCE = "no-new-evidence"
    STALE_CONTEXT = "stale-context"
    INJECTION_REJECTED = "injection-rejected"
    SCHEMA_MISMATCH = "schema-mismatch"
    PROVIDER_UNAVAILABLE = "provider-unavailable"
    FALLBACK_REQUIRED = "fallback-required"
    AUTHORITY_FLOW_REJECTED = "authority-flow-rejected"
    EFFECT_FLOW_REJECTED = "effect-flow-rejected"
    VALIDATION_REQUIRED = "validation-required"
    CAPACITY_MISSING = "capacity-missing"
    MODEL_BEFORE_DETERMINISTIC = "model-before-deterministic"
    CANDIDATE_TIER_REQUIRED = "candidate-tier-required"


class HoleProviderOutcome(str, Enum):
    PROPOSED = "proposed"
    MISSED = "missed"
    FAILED = "failed"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise HoleResolutionError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _payload_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    frozen = _freeze(value if value is not None else {}, field_name)
    if not isinstance(frozen, Mapping):
        raise HoleResolutionError(f"{field_name} must be a mapping")
    return frozen


def _authority_hit(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _AUTHORITY_MARKERS or any(
        marker in normalized for marker in _AUTHORITY_MARKERS
    )


def _scan_forbidden_payload(value: Any, field_name: str) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                return "injection-rejected"
            key = raw_key.lower().replace("-", "_")
            if _unsafe_key(raw_key) or _authority_hit(raw_key):
                if _authority_hit(raw_key):
                    return "authority-flow-rejected"
                return "injection-rejected"
            if key in {"effect_classes", "declared_effects", "new_effects"}:
                return "effect-flow-rejected"
            nested = _scan_forbidden_payload(item, field_name)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            nested = _scan_forbidden_payload(item, field_name)
            if nested is not None:
                return nested
    return None


def _reason_from_scan(code: str) -> HoleResolutionReason:
    if code == "authority-flow-rejected":
        return HoleResolutionReason.AUTHORITY_FLOW_REJECTED
    if code == "effect-flow-rejected":
        return HoleResolutionReason.EFFECT_FLOW_REJECTED
    return HoleResolutionReason.INJECTION_REJECTED


def model_route_for_provider_class(provider_class: ProviderClass | str) -> ModelRoute:
    normalized = _enum(provider_class, ProviderClass, "provider_class")
    return PROVIDER_CLASS_TO_MODEL_ROUTE[normalized]


def provider_port_claims_authority(port: object) -> bool:
    """True when an injected provider tries to skip validation or claim correctness."""

    return bool(
        getattr(port, "can_skip_validation", False)
        or getattr(port, "can_authorize", False)
        or getattr(port, "claims_correctness", False)
        or getattr(port, "can_claim_correctness", False)
    )


def default_hole_context_compiler(
    *,
    max_input_tokens: int = 2_048,
    provider_max_input_bytes: int | None = 65_536,
) -> ContextCompiler:
    """Build a hermetic ContextCompiler for hole routing tests and defaults."""

    def _tokenizer(text: str) -> int:
        encoded = text.encode("utf-8")
        return max(1, (len(encoded) + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN)

    return ContextCompiler(
        ContextBudget(
            max_input_tokens=max_input_tokens,
            reserved_output_tokens=256,
            reserved_tool_tokens=64,
            max_items=32,
            max_item_bytes=16_384,
            max_serialized_bytes=min(262_144, MAX_HOLE_CONTEXT_BYTES),
        ),
        tokenizer=_tokenizer,
        provider_max_input_bytes=provider_max_input_bytes,
        provider_context_window=max_input_tokens + 320,
    )


@dataclass(frozen=True)
class HoleContextReference:
    """Compact content reference selected for one hole call."""

    reference_id: str
    content_id: str
    tree_id: str = ""
    byte_count: int = 0
    token_count: int = 0
    required: bool = False
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "reference_id")
        )
        object.__setattr__(self, "content_id", _identifier(self.content_id, "content_id"))
        object.__setattr__(
            self, "tree_id", _identifier(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        object.__setattr__(
            self, "token_count", _nonnegative_int(self.token_count, "token_count")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", required=False)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "content_id": self.content_id,
            "tree_id": self.tree_id,
            "byte_count": self.byte_count,
            "token_count": self.token_count,
            "required": self.required,
            "summary": self.summary,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> HoleContextReference:
        if not isinstance(payload, Mapping):
            raise HoleResolutionError("context reference must be a mapping")
        return cls(
            reference_id=payload.get("reference_id", ""),
            content_id=payload.get("content_id", payload.get("referenced_content_id", "")),
            tree_id=payload.get("tree_id", ""),
            byte_count=payload.get("byte_count", 0),
            token_count=payload.get("token_count", 0),
            required=payload.get("required", False),
            summary=payload.get("summary", ""),
        )


@dataclass(frozen=True)
class HoleAttempt:
    """One recorded provider interaction or suppressed replay."""

    attempt_index: int
    provider_class: ProviderClass
    outcome: str
    evidence_fingerprint: str
    failure_code: str = ""
    context_receipt_cid: str = ""
    output_digest: str = ""
    token_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_index",
            _nonnegative_int(self.attempt_index, "attempt_index", maximum=MAX_HOLE_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "provider_class",
            _enum(self.provider_class, ProviderClass, "provider_class"),
        )
        outcome = _identifier(self.outcome, "outcome")
        if outcome not in {"proposed", "missed", "failed", "suppressed", "refused"}:
            raise HoleResolutionError("attempt outcome is outside the closed vocabulary")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _identifier(self.evidence_fingerprint, "evidence_fingerprint"),
        )
        object.__setattr__(
            self, "failure_code", _identifier(self.failure_code, "failure_code", required=False)
        )
        object.__setattr__(
            self,
            "context_receipt_cid",
            _identifier(self.context_receipt_cid, "context_receipt_cid", required=False),
        )
        object.__setattr__(
            self, "output_digest", _identifier(self.output_digest, "output_digest", required=False)
        )
        object.__setattr__(
            self,
            "token_count",
            _nonnegative_int(self.token_count, "token_count", maximum=MAX_HOLE_TOKENS),
        )

    @property
    def counts_against_attempt_budget(self) -> bool:
        return self.outcome in _PROVIDER_OUTCOMES

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "provider_class": self.provider_class.value,
            "outcome": self.outcome,
            "evidence_fingerprint": self.evidence_fingerprint,
            "failure_code": self.failure_code,
            "context_receipt_cid": self.context_receipt_cid,
            "output_digest": self.output_digest,
            "token_count": self.token_count,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> HoleAttempt:
        if not isinstance(payload, Mapping):
            raise HoleResolutionError("hole attempt must be a mapping")
        return cls(
            attempt_index=payload.get("attempt_index", 0),
            provider_class=payload.get("provider_class", ""),
            outcome=payload.get("outcome", ""),
            evidence_fingerprint=payload.get("evidence_fingerprint", ""),
            failure_code=payload.get("failure_code", ""),
            context_receipt_cid=payload.get("context_receipt_cid", ""),
            output_digest=payload.get("output_digest", ""),
            token_count=payload.get("token_count", 0),
        )


def _attempt_records(values: Any) -> tuple[HoleAttempt, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise HoleResolutionError("prior_attempts must be a sequence")
    if len(raw) > MAX_HOLE_ATTEMPTS:
        raise HoleBoundError("prior_attempts exceeds the attempt bound")
    attempts: list[HoleAttempt] = []
    for item in raw:
        if isinstance(item, HoleAttempt):
            attempts.append(item)
        elif isinstance(item, Mapping):
            attempts.append(HoleAttempt.from_record(item))
        else:
            raise HoleResolutionError("prior_attempts must contain HoleAttempt records")
    return tuple(attempts)


def _context_records(values: Any) -> tuple[HoleContextReference, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise HoleResolutionError("context_references must be a sequence")
    if len(raw) > MAX_ITEMS:
        raise HoleBoundError("context_references exceeds its item bound")
    result: list[HoleContextReference] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, HoleContextReference):
            record = item
        elif isinstance(item, Mapping):
            record = HoleContextReference.from_record(item)
        else:
            raise HoleResolutionError("context_references must contain HoleContextReference records")
        if record.reference_id in seen:
            raise HoleResolutionError("context_references contains a duplicate reference_id")
        seen.add(record.reference_id)
        result.append(record)
    return tuple(result)


@dataclass(frozen=True)
class ProviderCapacitySnapshot:
    """Current typed capacity for one approved provider class."""

    provider_class: ProviderClass
    available: bool
    remaining_calls: int = 0
    max_context_bytes: int = 0
    max_tokens: int = 0
    provider_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_class",
            _enum(self.provider_class, ProviderClass, "provider_class"),
        )
        object.__setattr__(self, "available", _bool(self.available, "available"))
        object.__setattr__(
            self,
            "remaining_calls",
            _nonnegative_int(self.remaining_calls, "remaining_calls", maximum=MAX_HOLE_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "max_context_bytes",
            _nonnegative_int(
                self.max_context_bytes, "max_context_bytes", maximum=MAX_HOLE_CONTEXT_BYTES
            ),
        )
        object.__setattr__(
            self,
            "max_tokens",
            _nonnegative_int(self.max_tokens, "max_tokens", maximum=MAX_HOLE_TOKENS),
        )
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, "provider_id", required=False)
        )

    @property
    def model_route(self) -> ModelRoute:
        return model_route_for_provider_class(self.provider_class)

    def admits(self, *, context_bytes: int, tokens: int) -> bool:
        if not self.available or self.remaining_calls < 1:
            return False
        if self.max_context_bytes and context_bytes > self.max_context_bytes:
            return False
        if self.max_tokens and tokens > self.max_tokens:
            return False
        return True

    def to_record(self) -> dict[str, Any]:
        return {
            "provider_class": self.provider_class.value,
            "available": self.available,
            "remaining_calls": self.remaining_calls,
            "max_context_bytes": self.max_context_bytes,
            "max_tokens": self.max_tokens,
            "provider_id": self.provider_id,
            "model_route": self.model_route.value,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ProviderCapacitySnapshot) -> ProviderCapacitySnapshot:
        if isinstance(payload, ProviderCapacitySnapshot):
            return payload
        if not isinstance(payload, Mapping):
            raise HoleResolutionError("provider capacity must be a mapping")
        return cls(
            provider_class=payload.get("provider_class", ""),
            available=payload.get("available", False),
            remaining_calls=payload.get("remaining_calls", 0),
            max_context_bytes=payload.get("max_context_bytes", 0),
            max_tokens=payload.get("max_tokens", 0),
            provider_id=payload.get("provider_id", ""),
        )


@dataclass(frozen=True)
class HoleProviderResult:
    """Bounded proposal, miss, or failure from an injected provider port."""

    outcome: HoleProviderOutcome
    output: Mapping[str, Any] = field(default_factory=dict)
    token_count: int = 0
    failure_code: str = ""
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcome", _enum(self.outcome, HoleProviderOutcome, "outcome")
        )
        object.__setattr__(self, "output", _payload_mapping(self.output, "output"))
        object.__setattr__(
            self,
            "token_count",
            _nonnegative_int(self.token_count, "token_count", maximum=MAX_HOLE_TOKENS),
        )
        object.__setattr__(
            self, "failure_code", _identifier(self.failure_code, "failure_code", required=False)
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, "evidence_ids", identifiers=True),
        )


class HoleProvider(Protocol):
    """Injected proposal producer.  Implementations cannot validate or admit."""

    def propose(
        self,
        request: "HoleRequest",
        compiled: "CompiledHoleContext",
    ) -> HoleProviderResult:
        ...


@dataclass(frozen=True)
class CompiledHoleContext:
    """Replayable ContextCompiler receipt used for stale and suppression checks."""

    repository_id: str
    tree_id: str
    capsule_id: str
    receipt_cid: str
    input_tokens: int
    serialized_bytes: int
    selected_reference_ids: tuple[str, ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "capsule_id", "receipt_cid", "evidence_fingerprint"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "input_tokens",
            _nonnegative_int(self.input_tokens, "input_tokens", maximum=MAX_HOLE_TOKENS),
        )
        object.__setattr__(
            self,
            "serialized_bytes",
            _nonnegative_int(
                self.serialized_bytes, "serialized_bytes", maximum=MAX_HOLE_CONTEXT_BYTES
            ),
        )
        object.__setattr__(
            self,
            "selected_reference_ids",
            _strings(
                self.selected_reference_ids,
                "selected_reference_ids",
                identifiers=True,
                preserve_order=True,
            ),
        )


def _normalize_hole_type(value: Any) -> HoleType:
    if isinstance(value, HoleType):
        if value.value in FORBIDDEN_HOLE_TYPES:
            raise HoleTypeError("forbidden hole types cannot be requested")
        return value
    if type(value) is not str:
        raise HoleTypeError("hole_type must be a closed HoleType identifier")
    raw = value.strip()
    if raw in FORBIDDEN_HOLE_TYPES:
        raise HoleTypeError("forbidden hole types cannot be requested")
    if raw not in ALLOWED_HOLE_TYPES:
        raise HoleTypeError("hole_type is outside the allowed typed-hole vocabulary")
    return HoleType(raw)


@dataclass(frozen=True)
class HoleRequest(CanonicalContract):
    """Closed request that may call only its declared approved providers."""

    SCHEMA: ClassVar[str] = _schema_name("HoleRequest")

    bindings: ArtifactBindings
    hole_id: str
    hole_type: HoleType
    input_schema_ref: str
    output_schema_ref: str
    allowed_provider_classes: tuple[ProviderClass, ...]
    context_budget_bytes: int
    validation_observation_ids: tuple[str, ...]
    fallback_step_id: str
    maximum_attempts: int
    input_payload: Mapping[str, Any] = field(default_factory=dict)
    context_references: tuple[HoleContextReference, ...] = ()
    authority_requirement_ids: tuple[str, ...] = ()
    effect_classes: tuple[EffectClass, ...] = ()
    token_budget: int = 1_024
    prior_attempts: tuple[HoleAttempt, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "hole_type", _normalize_hole_type(self.hole_type))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        providers = _enums(
            self.allowed_provider_classes,
            ProviderClass,
            "allowed_provider_classes",
            limit=len(ProviderClass),
            required=True,
        )
        object.__setattr__(self, "allowed_provider_classes", providers)
        object.__setattr__(
            self,
            "context_budget_bytes",
            _positive_int(
                self.context_budget_bytes,
                "context_budget_bytes",
                maximum=MAX_HOLE_CONTEXT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "validation_observation_ids",
            _strings(
                self.validation_observation_ids,
                "validation_observation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "fallback_step_id", _identifier(self.fallback_step_id, "fallback_step_id")
        )
        object.__setattr__(
            self,
            "maximum_attempts",
            _positive_int(self.maximum_attempts, "maximum_attempts", maximum=MAX_HOLE_ATTEMPTS),
        )
        payload = _payload_mapping(self.input_payload, "input_payload")
        scan = _scan_forbidden_payload(payload, "input_payload")
        if scan is not None:
            raise HoleResolutionError("hole input contains a forbidden authority or injection field")
        object.__setattr__(self, "input_payload", payload)
        object.__setattr__(self, "context_references", _context_records(self.context_references))
        object.__setattr__(
            self,
            "authority_requirement_ids",
            _strings(
                self.authority_requirement_ids,
                "authority_requirement_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "effect_classes",
            _enums(self.effect_classes, EffectClass, "effect_classes", limit=8),
        )
        if not set(self.effect_classes).issubset(
            {EffectClass.OBSERVE, EffectClass.MODEL_REQUEST}
        ):
            raise HoleResolutionError("typed holes cannot declare effect classes beyond observe/model_request")
        object.__setattr__(
            self,
            "token_budget",
            _positive_int(self.token_budget, "token_budget", maximum=MAX_HOLE_TOKENS),
        )
        object.__setattr__(self, "prior_attempts", _attempt_records(self.prior_attempts))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleResolutionError("hole requests remain candidate-tier")
        required_keys = _REQUIRED_INPUT_KEYS[self.hole_type]
        missing = tuple(key for key in required_keys if key not in payload)
        if missing:
            raise HoleResolutionError("hole input is missing required schema fields")
        if (
            any(
                provider in MODEL_PROVIDER_CLASSES
                for provider in providers
            )
            and self.context_budget_bytes == 0
        ):
            raise HoleResolutionError("remote holes require a nonzero context budget")
        _bounded(self, "HoleRequest")

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def provider_call_count(self) -> int:
        return sum(1 for item in self.prior_attempts if item.counts_against_attempt_budget)

    def allows_provider(self, provider_class: ProviderClass | str) -> bool:
        normalized = _enum(provider_class, ProviderClass, "provider_class")
        return normalized in self.allowed_provider_classes

    def with_attempts(self, attempts: Sequence[HoleAttempt]) -> HoleRequest:
        return HoleRequest(
            bindings=self.bindings,
            hole_id=self.hole_id,
            hole_type=self.hole_type,
            input_schema_ref=self.input_schema_ref,
            output_schema_ref=self.output_schema_ref,
            allowed_provider_classes=self.allowed_provider_classes,
            context_budget_bytes=self.context_budget_bytes,
            validation_observation_ids=self.validation_observation_ids,
            fallback_step_id=self.fallback_step_id,
            maximum_attempts=self.maximum_attempts,
            input_payload=dict(self.input_payload),
            context_references=self.context_references,
            authority_requirement_ids=self.authority_requirement_ids,
            effect_classes=self.effect_classes,
            token_budget=self.token_budget,
            prior_attempts=tuple(attempts),
            state=ArtifactState.CANDIDATE,
        )

    @classmethod
    def from_procedure_hole(
        cls,
        hole: ProcedureHole,
        bindings: ArtifactBindings,
        *,
        input_payload: Mapping[str, Any] | None = None,
        context_references: Sequence[HoleContextReference | Mapping[str, Any]] = (),
        token_budget: int | None = None,
        prior_attempts: Sequence[HoleAttempt | Mapping[str, Any]] = (),
        effect_classes: Sequence[EffectClass | str] | None = None,
    ) -> HoleRequest:
        if not isinstance(hole, ProcedureHole):
            raise HoleResolutionError("from_procedure_hole requires a ProcedureHole")
        declared_effects = effect_classes if effect_classes is not None else hole.effect_classes
        return cls(
            bindings=bindings,
            hole_id=hole.hole_id,
            hole_type=hole.hole_type,
            input_schema_ref=hole.input_schema_ref,
            output_schema_ref=hole.output_schema_ref,
            allowed_provider_classes=hole.allowed_provider_classes,
            context_budget_bytes=hole.context_budget_bytes,
            validation_observation_ids=hole.validation_observation_ids,
            fallback_step_id=hole.fallback_step_id,
            maximum_attempts=hole.maximum_attempts,
            input_payload=input_payload or {},
            context_references=tuple(context_references),
            authority_requirement_ids=hole.authority_requirement_ids,
            effect_classes=declared_effects,
            token_budget=token_budget if token_budget is not None else max(32, hole.context_budget_bytes // BYTES_PER_TOKEN),
            prior_attempts=tuple(prior_attempts),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "allowed_provider_classes": tuple(
                item.value for item in self.allowed_provider_classes
            ),
            "context_budget_bytes": self.context_budget_bytes,
            "validation_observation_ids": self.validation_observation_ids,
            "fallback_step_id": self.fallback_step_id,
            "maximum_attempts": self.maximum_attempts,
            "input_payload": dict(self.input_payload),
            "context_references": tuple(item.to_record() for item in self.context_references),
            "authority_requirement_ids": self.authority_requirement_ids,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "token_budget": self.token_budget,
            "prior_attempts": tuple(item.to_record() for item in self.prior_attempts),
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleRequest:
        if not isinstance(payload, Mapping):
            raise HoleResolutionError("HoleRequest payload must be a mapping")
        body = dict(payload)
        keys = set(body).difference({"schema", "contract_version", "content_id", "cid"})
        if keys and keys <= _GENERIC_ENVELOPE_FIELDS and "facts" in body:
            facts = body.get("facts")
            if not isinstance(facts, Mapping):
                raise HoleResolutionError("generic HoleRequest facts must be a mapping")
            merged = {
                "schema": cls.SCHEMA,
                "contract_version": body.get("contract_version", PROCEDURE_CONTRACT_VERSION),
                "bindings": body.get("bindings"),
                "state": body.get("state", ArtifactState.CANDIDATE.value),
                **dict(facts),
            }
            if "hole_id" not in merged:
                merged["hole_id"] = body.get("subject_cid", "")
            body = merged
        fields = (
            "bindings",
            "hole_id",
            "hole_type",
            "input_schema_ref",
            "output_schema_ref",
            "allowed_provider_classes",
            "context_budget_bytes",
            "validation_observation_ids",
            "fallback_step_id",
            "maximum_attempts",
            "input_payload",
            "context_references",
            "authority_requirement_ids",
            "effect_classes",
            "token_budget",
            "prior_attempts",
            "state",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class HoleCandidate(CanonicalContract):
    """Provider output that remains a candidate until independent validation."""

    SCHEMA: ClassVar[str] = _schema_name("HoleCandidate")

    bindings: ArtifactBindings
    request_cid: str
    hole_id: str
    hole_type: HoleType
    output_schema_ref: str
    provider_class: ProviderClass
    output: Mapping[str, Any]
    context_receipt_cid: str
    evidence_fingerprint: str
    token_count: int = 0
    state: ArtifactState = ArtifactState.CANDIDATE
    validated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "request_cid", _identifier(self.request_cid, "request_cid"))
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "hole_type", _normalize_hole_type(self.hole_type))
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        object.__setattr__(
            self, "provider_class", _enum(self.provider_class, ProviderClass, "provider_class")
        )
        output = _payload_mapping(self.output, "output")
        scan = _scan_forbidden_payload(output, "output")
        if scan is not None:
            raise HoleValidationError("hole candidate contains a forbidden field")
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "context_receipt_cid",
            _identifier(self.context_receipt_cid, "context_receipt_cid"),
        )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _identifier(self.evidence_fingerprint, "evidence_fingerprint"),
        )
        object.__setattr__(
            self,
            "token_count",
            _nonnegative_int(self.token_count, "token_count", maximum=MAX_HOLE_TOKENS),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleValidationError("hole outputs remain candidates until validation")
        object.__setattr__(self, "validated", _bool(self.validated, "validated"))
        if self.validated:
            raise HoleValidationError("provider output cannot self-validate")
        _bounded(self, "HoleCandidate")

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def output_digest(self) -> str:
        return content_identity({"output": dict(self.output), "schema": self.output_schema_ref})

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "request_cid": self.request_cid,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "output_schema_ref": self.output_schema_ref,
            "provider_class": self.provider_class.value,
            "output": dict(self.output),
            "context_receipt_cid": self.context_receipt_cid,
            "evidence_fingerprint": self.evidence_fingerprint,
            "token_count": self.token_count,
            "state": ArtifactState.CANDIDATE.value,
            "validated": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleCandidate:
        fields = (
            "bindings",
            "request_cid",
            "hole_id",
            "hole_type",
            "output_schema_ref",
            "provider_class",
            "output",
            "context_receipt_cid",
            "evidence_fingerprint",
            "token_count",
            "state",
            "validated",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class HoleResolution(CanonicalContract):
    """Typed routing result.  Never grants authority or skips validation."""

    SCHEMA: ClassVar[str] = _schema_name("HoleResolution")

    bindings: ArtifactBindings
    request_cid: str
    hole_id: str
    action: HoleResolutionAction
    reason_code: HoleResolutionReason
    fallback_step_id: str
    provider_class: str = ""
    candidate_cid: str = ""
    context_receipt_cid: str = ""
    evidence_fingerprint: str = ""
    attempts_used: int = 0
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    remains_candidate: bool = True
    resolver_revision: str = HOLE_RESOLVER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "request_cid", _identifier(self.request_cid, "request_cid"))
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(
            self, "action", _enum(self.action, HoleResolutionAction, "action")
        )
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, HoleResolutionReason, "reason_code")
        )
        object.__setattr__(
            self, "fallback_step_id", _identifier(self.fallback_step_id, "fallback_step_id")
        )
        object.__setattr__(
            self,
            "provider_class",
            _identifier(self.provider_class, "provider_class", required=False),
        )
        object.__setattr__(
            self, "candidate_cid", _identifier(self.candidate_cid, "candidate_cid", required=False)
        )
        object.__setattr__(
            self,
            "context_receipt_cid",
            _identifier(self.context_receipt_cid, "context_receipt_cid", required=False),
        )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _identifier(self.evidence_fingerprint, "evidence_fingerprint", required=False),
        )
        object.__setattr__(
            self,
            "attempts_used",
            _nonnegative_int(self.attempts_used, "attempts_used", maximum=MAX_HOLE_ATTEMPTS),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleValidationError("hole resolutions remain candidate-tier")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise HoleResolutionError("hole resolutions cannot authorize")
        object.__setattr__(
            self, "remains_candidate", _bool(self.remains_candidate, "remains_candidate")
        )
        if not self.remains_candidate:
            raise HoleValidationError("hole outputs remain candidates until validation")
        object.__setattr__(
            self, "resolver_revision", _identifier(self.resolver_revision, "resolver_revision")
        )
        if self.resolver_revision != HOLE_RESOLVER_REVISION:
            raise HoleResolutionError("hole resolver revision is not current")
        if self.action is HoleResolutionAction.PROPOSE and not self.candidate_cid:
            raise HoleResolutionError("a propose result requires a candidate")
        _bounded(self, "HoleResolution")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_skip_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "request_cid": self.request_cid,
            "hole_id": self.hole_id,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "fallback_step_id": self.fallback_step_id,
            "provider_class": self.provider_class,
            "candidate_cid": self.candidate_cid,
            "context_receipt_cid": self.context_receipt_cid,
            "evidence_fingerprint": self.evidence_fingerprint,
            "attempts_used": self.attempts_used,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "remains_candidate": True,
            "resolver_revision": HOLE_RESOLVER_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleResolution:
        fields = (
            "bindings",
            "request_cid",
            "hole_id",
            "action",
            "reason_code",
            "fallback_step_id",
            "provider_class",
            "candidate_cid",
            "context_receipt_cid",
            "evidence_fingerprint",
            "attempts_used",
            "state",
            "can_authorize",
            "remains_candidate",
            "resolver_revision",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class HoleValidationReceipt(CanonicalContract):
    """Independent check of a hole candidate.  Acceptance never promotes."""

    SCHEMA: ClassVar[str] = _schema_name("HoleValidationReceipt")

    bindings: ArtifactBindings
    request_cid: str
    candidate_cid: str
    hole_id: str
    accepted: bool
    reason_code: HoleResolutionReason
    observation_ids: tuple[str, ...]
    state: ArtifactState = ArtifactState.CANDIDATE
    remains_candidate: bool = True
    can_authorize: bool = False
    validator_revision: str = HOLE_VALIDATOR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "request_cid", _identifier(self.request_cid, "request_cid"))
        object.__setattr__(
            self, "candidate_cid", _identifier(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, HoleResolutionReason, "reason_code")
        )
        object.__setattr__(
            self,
            "observation_ids",
            _strings(self.observation_ids, "observation_ids", identifiers=True),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleValidationError("validated hole fills remain candidates")
        object.__setattr__(
            self, "remains_candidate", _bool(self.remains_candidate, "remains_candidate")
        )
        if not self.remains_candidate:
            raise HoleValidationError("hole outputs remain candidates until validation")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise HoleValidationError("hole validation cannot authorize")
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != HOLE_VALIDATOR_REVISION:
            raise HoleValidationError("hole validator revision is not current")
        _bounded(self, "HoleValidationReceipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "request_cid": self.request_cid,
            "candidate_cid": self.candidate_cid,
            "hole_id": self.hole_id,
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
            "observation_ids": self.observation_ids,
            "state": ArtifactState.CANDIDATE.value,
            "remains_candidate": True,
            "can_authorize": False,
            "validator_revision": HOLE_VALIDATOR_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleValidationReceipt:
        fields = (
            "bindings",
            "request_cid",
            "candidate_cid",
            "hole_id",
            "accepted",
            "reason_code",
            "observation_ids",
            "state",
            "remains_candidate",
            "can_authorize",
            "validator_revision",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


class HoleResolutionValidator:
    """Independent structural, injection, freshness, and authority-flow checks."""

    def __init__(self, *, current_tree_id: str = "") -> None:
        self._current_tree_id = (
            _identifier(current_tree_id, "current_tree_id", required=False)
            if current_tree_id
            else ""
        )

    def check_output_schema(
        self, request: HoleRequest, output: Mapping[str, Any]
    ) -> HoleResolutionReason | None:
        if not isinstance(output, Mapping):
            return HoleResolutionReason.SCHEMA_MISMATCH
        scan = _scan_forbidden_payload(output, "output")
        if scan is not None:
            return _reason_from_scan(scan)
        declared_schema = output.get("schema_ref", request.output_schema_ref)
        if declared_schema != request.output_schema_ref:
            return HoleResolutionReason.SCHEMA_MISMATCH
        required = _REQUIRED_OUTPUT_KEYS[request.hole_type]
        if any(key not in output for key in required):
            return HoleResolutionReason.SCHEMA_MISMATCH
        if request.hole_type is HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS:
            allowed = request.input_payload.get("allowed_values", ())
            if output.get("selected") not in tuple(allowed):
                return HoleResolutionReason.SCHEMA_MISMATCH
        if request.hole_type in {
            HoleType.PROPOSE_BOUNDED_PATCH,
            HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE,
        }:
            allowed = request.input_payload.get("template_ids", ())
            if output.get("template_id") not in tuple(allowed):
                return HoleResolutionReason.SCHEMA_MISMATCH
        extra_effects = output.get("effect_classes", ())
        if extra_effects and not set(extra_effects).issubset(
            {item.value for item in request.effect_classes} | {EffectClass.OBSERVE.value}
        ):
            return HoleResolutionReason.EFFECT_FLOW_REJECTED
        extra_authority = output.get("authority_requirement_ids", ())
        if extra_authority and not set(extra_authority).issubset(
            set(request.authority_requirement_ids)
        ):
            return HoleResolutionReason.AUTHORITY_FLOW_REJECTED
        return None

    def check_freshness(
        self,
        request: HoleRequest,
        *,
        compiled: CompiledHoleContext | None = None,
        current_tree_id: str = "",
    ) -> HoleResolutionReason | None:
        tree_id = current_tree_id or self._current_tree_id
        if tree_id and tree_id != request.bindings.tree_id:
            return HoleResolutionReason.STALE_CONTEXT
        for item in request.context_references:
            if item.tree_id and item.tree_id != request.bindings.tree_id:
                return HoleResolutionReason.STALE_CONTEXT
        if compiled is not None:
            if compiled.tree_id != request.bindings.tree_id:
                return HoleResolutionReason.STALE_CONTEXT
            if compiled.repository_id != request.bindings.repository_id:
                return HoleResolutionReason.STALE_CONTEXT
        return None

    def validate_candidate(
        self,
        request: HoleRequest,
        candidate: HoleCandidate,
        *,
        compiled: CompiledHoleContext | None = None,
        current_tree_id: str = "",
        observations: Sequence[str] = (),
    ) -> HoleValidationReceipt:
        if not isinstance(candidate, HoleCandidate):
            raise HoleValidationError("validation requires a HoleCandidate")
        if candidate.request_cid != request.content_id:
            raise HoleValidationError("candidate is not bound to the hole request")
        if candidate.hole_id != request.hole_id or candidate.hole_type is not request.hole_type:
            raise HoleValidationError("candidate does not match the typed hole")
        if candidate.state is not ArtifactState.CANDIDATE or candidate.validated:
            reason = HoleResolutionReason.CANDIDATE_TIER_REQUIRED
            accepted = False
        else:
            reason = self.check_freshness(
                request, compiled=compiled, current_tree_id=current_tree_id
            )
            if reason is None:
                reason = self.check_output_schema(request, candidate.output)
            accepted = reason is None
            if accepted:
                observed = tuple(
                    _identifier(item, "observation_ids") for item in observations
                )
                required = set(request.validation_observation_ids)
                if not required.issubset(set(observed)):
                    accepted = False
                    reason = HoleResolutionReason.VALIDATION_REQUIRED
                else:
                    reason = HoleResolutionReason.CANDIDATE_PROPOSED
        return HoleValidationReceipt(
            bindings=request.bindings,
            request_cid=request.content_id,
            candidate_cid=candidate.content_id,
            hole_id=request.hole_id,
            accepted=accepted,
            reason_code=reason or HoleResolutionReason.CANDIDATE_PROPOSED,
            observation_ids=tuple(observations),
        )

    def validate_resolution(
        self,
        request: HoleRequest,
        resolution: HoleResolution,
        candidate: HoleCandidate | None = None,
        *,
        compiled: CompiledHoleContext | None = None,
        current_tree_id: str = "",
        observations: Sequence[str] = (),
    ) -> HoleValidationReceipt:
        if resolution.request_cid != request.content_id:
            raise HoleValidationError("resolution is not bound to the hole request")
        if resolution.can_authorize or not resolution.remains_candidate:
            raise HoleValidationError("hole resolutions cannot leave the candidate tier")
        if resolution.action is HoleResolutionAction.PROPOSE:
            if candidate is None:
                raise HoleValidationError("propose resolutions require the candidate artifact")
            return self.validate_candidate(
                request,
                candidate,
                compiled=compiled,
                current_tree_id=current_tree_id,
                observations=observations,
            )
        return HoleValidationReceipt(
            bindings=request.bindings,
            request_cid=request.content_id,
            candidate_cid=resolution.candidate_cid or resolution.content_id,
            hole_id=request.hole_id,
            accepted=False,
            reason_code=resolution.reason_code,
            observation_ids=tuple(observations),
        )


def _capacity_index(
    values: Sequence[ProviderCapacitySnapshot | Mapping[str, Any]] | None,
) -> dict[ProviderClass, ProviderCapacitySnapshot]:
    if values is None:
        return {}
    index: dict[ProviderClass, ProviderCapacitySnapshot] = {}
    for item in values:
        snapshot = ProviderCapacitySnapshot.from_record(item)
        index[snapshot.provider_class] = snapshot
    return index


def _as_context_reference(
    item: HoleContextReference, request: HoleRequest
) -> ContextReference:
    return ContextReference(
        reference_id=item.reference_id,
        kind="hole-evidence",
        tier=ContextTier.INVARIANT if item.required else ContextTier.EVIDENCE,
        referenced_content_id=item.content_id,
        repository_id=request.bindings.repository_id,
        tree_id=item.tree_id or request.bindings.tree_id,
        summary=item.summary,
        byte_count=item.byte_count,
        token_count=item.token_count,
        metadata={"required": item.required},
    )


def evidence_fingerprint(
    request: HoleRequest, compiled: CompiledHoleContext | None = None
) -> str:
    payload = {
        "hole_id": request.hole_id,
        "hole_type": request.hole_type.value,
        "input_payload": dict(request.input_payload),
        "tree_id": request.bindings.tree_id,
        "reference_ids": tuple(item.reference_id for item in request.context_references),
        "reference_content_ids": tuple(item.content_id for item in request.context_references),
        "receipt_cid": compiled.receipt_cid if compiled is not None else "",
        "selected_reference_ids": compiled.selected_reference_ids if compiled is not None else (),
    }
    return content_identity(payload)


def _failure_fingerprint(
    request: HoleRequest,
    compiled: CompiledHoleContext,
    *,
    failure_code: str,
    output_digest: str = "",
) -> str:
    return content_identity(
        {
            "evidence": evidence_fingerprint(request, compiled),
            "failure_code": failure_code,
            "output_digest": output_digest,
        }
    )


class HoleResolver:
    """Route an allowed typed hole through approved providers, then stop at a candidate."""

    def __init__(
        self,
        context_compiler: ContextCompiler | None = None,
        *,
        providers: Mapping[ProviderClass | str, HoleProvider] | None = None,
        capacity: Sequence[ProviderCapacitySnapshot | Mapping[str, Any]] | None = None,
        validator: HoleResolutionValidator | None = None,
        current_tree_id: str = "",
    ) -> None:
        compiler = context_compiler or default_hole_context_compiler()
        if not isinstance(compiler, ContextCompiler):
            raise HoleResolutionError("hole routing requires a ContextCompiler")
        self._compiler = compiler
        self._validator = validator or HoleResolutionValidator(current_tree_id=current_tree_id)
        self._current_tree_id = (
            _identifier(current_tree_id, "current_tree_id", required=False)
            if current_tree_id
            else ""
        )
        ports: dict[ProviderClass, HoleProvider] = {}
        for key, port in dict(providers or {}).items():
            ports[_enum(key, ProviderClass, "provider_class")] = port
        self._providers = ports
        self._capacity = _capacity_index(capacity)
        self._ledger: dict[str, list[HoleAttempt]] = {}
        self._candidates: dict[str, HoleCandidate] = {}
        self._candidates_by_request: dict[str, HoleCandidate] = {}
        self._failures: dict[str, str] = {}

    @property
    def validator(self) -> HoleResolutionValidator:
        return self._validator

    def _ledger_key(self, request: HoleRequest) -> str:
        return request.hole_id + ":" + request.bindings.tree_id

    def _attempts_for(self, request: HoleRequest) -> tuple[HoleAttempt, ...]:
        ledger = tuple(self._ledger.get(self._ledger_key(request), ()))
        if request.prior_attempts:
            combined = list(request.prior_attempts)
            seen = {(item.attempt_index, item.provider_class, item.outcome, item.evidence_fingerprint) for item in combined}
            for item in ledger:
                marker = (item.attempt_index, item.provider_class, item.outcome, item.evidence_fingerprint)
                if marker not in seen:
                    combined.append(item)
                    seen.add(marker)
            return tuple(combined)
        return ledger

    def _record_attempt(self, request: HoleRequest, attempt: HoleAttempt) -> None:
        key = self._ledger_key(request)
        self._ledger.setdefault(key, []).append(attempt)

    def compile_context(self, request: HoleRequest) -> CompiledHoleContext:
        evidence = tuple(_as_context_reference(item, request) for item in request.context_references)
        try:
            result = self._compiler.compile(
                repository_id=request.bindings.repository_id,
                tree_id=request.bindings.tree_id,
                objective_id=request.bindings.objective_id,
                objective_revision=request.bindings.contract_revision,
                policy_id=request.bindings.policy_revision,
                policy_revision=request.bindings.policy_revision,
                caller=HOLE_CALLER,
                stage=HOLE_STAGE,
                goal={
                    "hole_id": request.hole_id,
                    "hole_type": request.hole_type.value,
                    "task_id": request.bindings.task_id,
                },
                authority={
                    "mode": "proposal",
                    "requirement_ids": list(request.authority_requirement_ids),
                },
                scope={
                    "tree_id": request.bindings.tree_id,
                    "effect_classes": [item.value for item in request.effect_classes],
                },
                acceptance={
                    "validation_observation_ids": list(request.validation_observation_ids),
                    "candidate_until_validated": True,
                },
                evidence=evidence,
            )
        except (RequiredContextOverflowError, ContextCompilationError) as exc:
            raise HoleBoundError("compiled hole context exceeds its budget") from exc
        serialized_bytes = len(result.capsule.canonical_bytes())
        selected = tuple(item.reference_id for item in result.capsule.evidence)
        fingerprint = content_identity(
            {
                "hole_id": request.hole_id,
                "hole_type": request.hole_type.value,
                "input_payload": dict(request.input_payload),
                "tree_id": request.bindings.tree_id,
                "reference_ids": tuple(item.reference_id for item in request.context_references),
                "reference_content_ids": tuple(
                    item.content_id for item in request.context_references
                ),
                "receipt_cid": result.receipt.content_id,
                "selected_reference_ids": selected,
            }
        )
        return CompiledHoleContext(
            repository_id=result.receipt.repository_id,
            tree_id=result.receipt.tree_id,
            capsule_id=result.capsule.capsule_id,
            receipt_cid=result.receipt.content_id,
            input_tokens=result.receipt.input_tokens,
            serialized_bytes=serialized_bytes,
            selected_reference_ids=selected,
            evidence_fingerprint=fingerprint,
        )

    def _ordered_providers(self, request: HoleRequest) -> tuple[ProviderClass, ...]:
        allowed = set(request.allowed_provider_classes)
        return tuple(item for item in PROVIDER_ROUTE_ORDER if item in allowed)

    def _capacity_for(self, provider_class: ProviderClass) -> ProviderCapacitySnapshot | None:
        return self._capacity.get(provider_class)

    def _call_provider(
        self,
        provider_class: ProviderClass,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        port = self._providers.get(provider_class)
        if port is None:
            return HoleProviderResult(
                outcome=HoleProviderOutcome.MISSED, failure_code="provider-not-injected"
            )
        if provider_port_claims_authority(port):
            return HoleProviderResult(
                outcome=HoleProviderOutcome.FAILED,
                failure_code="authority-flow-rejected",
            )
        try:
            result = port.propose(request, compiled)
        except Exception:
            return HoleProviderResult(
                outcome=HoleProviderOutcome.FAILED, failure_code="provider-error"
            )
        if isinstance(result, HoleProviderResult):
            scan = _scan_forbidden_payload(result.output, "output")
            if scan is not None:
                return HoleProviderResult(
                    outcome=HoleProviderOutcome.FAILED, failure_code=scan
                )
            return result
        if isinstance(result, Mapping):
            output = result.get("output", {})
            scan = _scan_forbidden_payload(output, "output")
            if scan is not None:
                return HoleProviderResult(
                    outcome=HoleProviderOutcome.FAILED, failure_code=scan
                )
            try:
                return HoleProviderResult(
                    outcome=result.get("outcome", HoleProviderOutcome.FAILED),
                    output=output if isinstance(output, Mapping) else {},
                    token_count=result.get("token_count", 0),
                    failure_code=result.get("failure_code", ""),
                    evidence_ids=result.get("evidence_ids", ()),
                )
            except ProcedureContractError:
                return HoleProviderResult(
                    outcome=HoleProviderOutcome.FAILED,
                    failure_code="injection-rejected",
                )
        raise HoleResolutionError("provider port returned an unsupported result")

    def _resolution(
        self,
        request: HoleRequest,
        *,
        action: HoleResolutionAction,
        reason: HoleResolutionReason,
        compiled: CompiledHoleContext | None = None,
        provider_class: ProviderClass | None = None,
        candidate: HoleCandidate | None = None,
        attempts_used: int = 0,
    ) -> HoleResolution:
        return HoleResolution(
            bindings=request.bindings,
            request_cid=request.content_id,
            hole_id=request.hole_id,
            action=action,
            reason_code=reason,
            fallback_step_id=request.fallback_step_id,
            provider_class="" if provider_class is None else provider_class.value,
            candidate_cid="" if candidate is None else candidate.content_id,
            context_receipt_cid="" if compiled is None else compiled.receipt_cid,
            evidence_fingerprint="" if compiled is None else compiled.evidence_fingerprint,
            attempts_used=attempts_used,
        )

    def resolve(
        self,
        request: HoleRequest | Mapping[str, Any],
        *,
        current_tree_id: str = "",
    ) -> HoleResolution:
        if isinstance(request, Mapping):
            request = HoleRequest.from_dict(request) if request.get("schema") == HoleRequest.SCHEMA else HoleRequest(**request)
        elif not isinstance(request, HoleRequest):
            raise HoleResolutionError("resolve requires a HoleRequest")

        tree_id = current_tree_id or self._current_tree_id
        stale = self._validator.check_freshness(request, current_tree_id=tree_id)
        if stale is not None:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=stale,
                attempts_used=request.provider_call_count,
            )

        prior = self._attempts_for(request)
        attempts_used = sum(1 for item in prior if item.counts_against_attempt_budget)
        if attempts_used >= request.maximum_attempts:
            return self._resolution(
                request,
                action=HoleResolutionAction.FALLBACK,
                reason=HoleResolutionReason.ATTEMPT_BUDGET_EXCEEDED,
                attempts_used=attempts_used,
            )

        try:
            compiled = self.compile_context(request)
        except HoleBoundError:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=HoleResolutionReason.CONTEXT_BUDGET_EXCEEDED,
                attempts_used=attempts_used,
            )
        stale = self._validator.check_freshness(
            request, compiled=compiled, current_tree_id=tree_id
        )
        if stale is not None:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=stale,
                compiled=compiled,
                attempts_used=attempts_used,
            )
        if compiled.serialized_bytes > request.context_budget_bytes:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=HoleResolutionReason.CONTEXT_BUDGET_EXCEEDED,
                compiled=compiled,
                attempts_used=attempts_used,
            )
        if compiled.input_tokens > request.token_budget:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=HoleResolutionReason.TOKEN_BUDGET_EXCEEDED,
                compiled=compiled,
                attempts_used=attempts_used,
            )

        fingerprint = compiled.evidence_fingerprint
        matching = [item for item in prior if item.evidence_fingerprint == fingerprint]
        if matching:
            last = matching[-1]
            if last.outcome == "failed":
                self._record_attempt(
                    request,
                    HoleAttempt(
                        attempt_index=len(prior),
                        provider_class=last.provider_class,
                        outcome="suppressed",
                        evidence_fingerprint=fingerprint,
                        failure_code=last.failure_code or "identical-failure",
                        context_receipt_cid=compiled.receipt_cid,
                    ),
                )
                return self._resolution(
                    request,
                    action=HoleResolutionAction.SUPPRESS,
                    reason=HoleResolutionReason.IDENTICAL_FAILURE,
                    compiled=compiled,
                    provider_class=last.provider_class,
                    attempts_used=attempts_used,
                )
            self._record_attempt(
                request,
                HoleAttempt(
                    attempt_index=len(prior),
                    provider_class=last.provider_class,
                    outcome="suppressed",
                    evidence_fingerprint=fingerprint,
                    failure_code="no-new-evidence",
                    context_receipt_cid=compiled.receipt_cid,
                    output_digest=last.output_digest,
                ),
            )
            cached = self._candidates.get(fingerprint)
            return self._resolution(
                request,
                action=HoleResolutionAction.SUPPRESS,
                reason=HoleResolutionReason.NO_NEW_EVIDENCE,
                compiled=compiled,
                provider_class=last.provider_class,
                candidate=cached,
                attempts_used=attempts_used,
            )

        ordered = self._ordered_providers(request)
        if not ordered:
            return self._resolution(
                request,
                action=HoleResolutionAction.REFUSE,
                reason=HoleResolutionReason.PROVIDER_NOT_ALLOWED,
                compiled=compiled,
                attempts_used=attempts_used,
            )

        deterministic_pending = [item for item in ordered if item in DETERMINISTIC_PROVIDER_CLASSES]
        called_model_early = False
        last_reason = HoleResolutionReason.FALLBACK_REQUIRED
        last_provider: ProviderClass | None = None
        next_index = len(prior)

        for provider_class in ordered:
            if attempts_used >= request.maximum_attempts:
                last_reason = HoleResolutionReason.ATTEMPT_BUDGET_EXCEEDED
                break
            if provider_class in MODEL_PROVIDER_CLASSES and deterministic_pending:
                called_model_early = True
                last_reason = HoleResolutionReason.MODEL_BEFORE_DETERMINISTIC
                break
            if not request.allows_provider(provider_class):
                last_reason = HoleResolutionReason.PROVIDER_NOT_ALLOWED
                continue
            capability = capability_for_route(model_route_for_provider_class(provider_class))
            snapshot = self._capacity_for(provider_class)
            if snapshot is None or not snapshot.admits(
                context_bytes=compiled.serialized_bytes, tokens=compiled.input_tokens
            ):
                last_reason = (
                    HoleResolutionReason.CAPACITY_MISSING
                    if snapshot is None
                    else HoleResolutionReason.PROVIDER_UNAVAILABLE
                )
                last_provider = provider_class
                if provider_class in DETERMINISTIC_PROVIDER_CLASSES:
                    deterministic_pending = [
                        item for item in deterministic_pending if item != provider_class
                    ]
                continue
            if capability.value not in {
                "deterministic",
                "small_local",
                "medium",
                "frontier",
                "human_review",
            }:
                last_reason = HoleResolutionReason.PROVIDER_NOT_ALLOWED
                continue

            result = self._call_provider(provider_class, request, compiled)
            outcome = result.outcome.value
            failure_code = result.failure_code
            output_digest = ""
            if result.outcome is HoleProviderOutcome.FAILED and result.failure_code in {
                "injection-rejected",
                "authority-flow-rejected",
                "effect-flow-rejected",
            }:
                last_reason = _reason_from_scan(result.failure_code)
            if result.outcome is HoleProviderOutcome.PROPOSED:
                schema_reason = self._validator.check_output_schema(request, result.output)
                if schema_reason is not None:
                    outcome = HoleProviderOutcome.FAILED.value
                    failure_code = schema_reason.value
                    last_reason = schema_reason
                elif result.token_count > request.token_budget:
                    outcome = HoleProviderOutcome.FAILED.value
                    failure_code = HoleResolutionReason.TOKEN_BUDGET_EXCEEDED.value
                    last_reason = HoleResolutionReason.TOKEN_BUDGET_EXCEEDED
                else:
                    try:
                        candidate = HoleCandidate(
                            bindings=request.bindings,
                            request_cid=request.content_id,
                            hole_id=request.hole_id,
                            hole_type=request.hole_type,
                            output_schema_ref=request.output_schema_ref,
                            provider_class=provider_class,
                            output=result.output,
                            context_receipt_cid=compiled.receipt_cid,
                            evidence_fingerprint=fingerprint,
                            token_count=result.token_count,
                        )
                    except (HoleResolutionError, ProcedureContractError):
                        outcome = HoleProviderOutcome.FAILED.value
                        failure_code = HoleResolutionReason.INJECTION_REJECTED.value
                        last_reason = HoleResolutionReason.INJECTION_REJECTED
                    else:
                        output_digest = candidate.output_digest
                        attempt = HoleAttempt(
                            attempt_index=next_index,
                            provider_class=provider_class,
                            outcome="proposed",
                            evidence_fingerprint=fingerprint,
                            context_receipt_cid=compiled.receipt_cid,
                            output_digest=output_digest,
                            token_count=result.token_count,
                        )
                        self._record_attempt(request, attempt)
                        self._candidates[fingerprint] = candidate
                        self._candidates_by_request[request.content_id] = candidate
                        return self._resolution(
                            request,
                            action=HoleResolutionAction.PROPOSE,
                            reason=HoleResolutionReason.CANDIDATE_PROPOSED,
                            compiled=compiled,
                            provider_class=provider_class,
                            candidate=candidate,
                            attempts_used=attempts_used + 1,
                        )
            attempt = HoleAttempt(
                attempt_index=next_index,
                provider_class=provider_class,
                outcome=outcome,
                evidence_fingerprint=fingerprint,
                failure_code=failure_code
                or (
                    "miss" if result.outcome is HoleProviderOutcome.MISSED else "provider-failed"
                ),
                context_receipt_cid=compiled.receipt_cid,
                output_digest=output_digest,
                token_count=result.token_count,
            )
            self._record_attempt(request, attempt)
            if outcome == "failed":
                self._failures[
                    _failure_fingerprint(
                        request, compiled, failure_code=attempt.failure_code
                    )
                ] = attempt.failure_code
                last_reason = (
                    last_reason
                    if last_reason
                    in {
                        HoleResolutionReason.INJECTION_REJECTED,
                        HoleResolutionReason.SCHEMA_MISMATCH,
                        HoleResolutionReason.AUTHORITY_FLOW_REJECTED,
                        HoleResolutionReason.EFFECT_FLOW_REJECTED,
                    }
                    else HoleResolutionReason.FALLBACK_REQUIRED
                )
            attempts_used += 1
            next_index += 1
            last_provider = provider_class
            if provider_class in DETERMINISTIC_PROVIDER_CLASSES:
                deterministic_pending = [
                    item for item in deterministic_pending if item != provider_class
                ]

        if called_model_early:
            action = HoleResolutionAction.REFUSE
            reason = HoleResolutionReason.MODEL_BEFORE_DETERMINISTIC
        elif last_reason is HoleResolutionReason.ATTEMPT_BUDGET_EXCEEDED:
            action = HoleResolutionAction.FALLBACK
            reason = last_reason
        elif last_reason in {
            HoleResolutionReason.INJECTION_REJECTED,
            HoleResolutionReason.SCHEMA_MISMATCH,
            HoleResolutionReason.AUTHORITY_FLOW_REJECTED,
            HoleResolutionReason.EFFECT_FLOW_REJECTED,
            HoleResolutionReason.PROVIDER_NOT_ALLOWED,
        }:
            action = HoleResolutionAction.REFUSE
            reason = last_reason
        else:
            action = HoleResolutionAction.FALLBACK
            reason = last_reason if last_reason is not None else HoleResolutionReason.FALLBACK_REQUIRED
        return self._resolution(
            request,
            action=action,
            reason=reason,
            compiled=compiled,
            provider_class=last_provider,
            attempts_used=attempts_used,
        )

    def last_candidate(self, request: HoleRequest) -> HoleCandidate | None:
        cached = self._candidates_by_request.get(request.content_id)
        if cached is not None:
            return cached
        try:
            compiled = self.compile_context(request)
        except HoleResolutionError:
            return None
        return self._candidates.get(compiled.evidence_fingerprint)


for _artifact_type in (HoleRequest, HoleCandidate, HoleResolution, HoleValidationReceipt):
    ARTIFACT_TYPES_BY_SCHEMA[_artifact_type.SCHEMA] = _artifact_type


__all__ = [
    "ALLOWED_HOLE_TYPES",
    "BYTES_PER_TOKEN",
    "DETERMINISTIC_PROVIDER_CLASSES",
    "HOLE_CALLER",
    "HOLE_RESOLVER_REVISION",
    "HOLE_STAGE",
    "HOLE_VALIDATOR_REVISION",
    "LOCAL_HOLE_ROUTE_ORDER",
    "MAX_HOLE_ATTEMPTS",
    "MAX_HOLE_CONTEXT_BYTES",
    "MAX_HOLE_TOKENS",
    "MODEL_PROVIDER_CLASSES",
    "PROVIDER_CLASS_TO_MODEL_ROUTE",
    "PROVIDER_ROUTE_ORDER",
    "CompiledHoleContext",
    "HoleAttempt",
    "HoleBoundError",
    "HoleCandidate",
    "HoleContextReference",
    "HoleProvider",
    "HoleProviderOutcome",
    "HoleProviderResult",
    "HoleRequest",
    "HoleResolution",
    "HoleResolutionAction",
    "HoleResolutionError",
    "HoleResolutionReason",
    "HoleResolutionValidator",
    "HoleResolver",
    "HoleTypeError",
    "HoleValidationError",
    "HoleValidationReceipt",
    "ProviderCapacitySnapshot",
    "default_hole_context_compiler",
    "evidence_fingerprint",
    "model_route_for_provider_class",
    "provider_port_claims_authority",
]
