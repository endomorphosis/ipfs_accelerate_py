"""Typed procedure-hole routing and independent candidate validation.

Only allowed typed holes may call approved providers.  Provider outputs remain
candidates until independently validated.  Identical failure or a retry with
no new evidence suppresses another provider call.  Hole resolution cannot
grant authority, change policy, omit tests or proofs, promote, complete, or
suppress validation.

Context selection reuses ``ContextCompiler``.  Remote-model admission reuses
the existing provider-route APIs.  This module owns hole routing and
validation only; it does not decide authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Protocol

from ..context.context_compiler import (
    ContextCompilationError,
    ContextCompiler,
    RequiredContextOverflowError,
)
from ..context.context_contracts import (
    ContextBudget,
    ContextContractError,
    ContextReference,
)
from ..entrypoints.provider_route import (
    ProviderRouteEvaluation,
    ProviderRouteError,
    evaluate_preferred_route,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    FORBIDDEN_HOLE_TYPES,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    HoleCandidate as HoleCandidateArtifact,
    HoleRequest as HoleRequestArtifact,
    HoleResolution as HoleResolutionArtifact,
    HoleType,
    HoleValidationReceipt as HoleValidationReceiptArtifact,
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


RESOLVER_REVISION: Final[str] = "HoleResolver@1"
VALIDATOR_REVISION: Final[str] = "HoleResolutionValidator@1"
HOLE_RESOLUTION_STAGE: Final[str] = "typed-hole"
HOLE_RESOLVER_CALLER: Final[str] = "procedure-compiler.hole-resolver"
MAX_HOLE_ATTEMPTS: Final[int] = 4
MAX_CONTEXT_BUDGET_BYTES: Final[int] = 1_048_576
MAX_OUTPUT_BYTES: Final[int] = 65_536

PROVIDER_ROUTE_ORDER: Final[tuple[ProviderClass, ...]] = (
    ProviderClass.EXACT_CACHE,
    ProviderClass.DECLARATIVE_RULE,
    ProviderClass.DETERMINISTIC_CLASSIFIER,
    ProviderClass.LOCAL_SMALL_MODEL,
    ProviderClass.REMOTE_STANDARD_MODEL,
    ProviderClass.REMOTE_STRONG_MODEL,
    ProviderClass.HUMAN,
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
        ProviderClass.HUMAN,
    }
)
REMOTE_PROVIDER_CLASSES: Final[frozenset[ProviderClass]] = frozenset(
    {
        ProviderClass.REMOTE_STANDARD_MODEL,
        ProviderClass.REMOTE_STRONG_MODEL,
    }
)
ALLOWED_HOLE_EFFECTS: Final[frozenset[EffectClass]] = frozenset(
    {EffectClass.OBSERVE, EffectClass.MODEL_REQUEST}
)
_INJECTION_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "authority_decision",
        "policy_decision",
        "confirmation",
        "trusted_key",
        "trusted_key_selection",
        "test_omission",
        "proof_acceptance",
        "release_promotion",
        "task_completion",
        "unbounded_shell",
        "unbounded_shell_command",
        "arbitrary_shell",
        "arbitrary_python",
        "disable_validation",
        "modify_authority",
        "modify_authority_policy",
        "modify_trusted_keys",
        "claim_completion",
        "selected_provider",
        "provider_class",
        "callback",
        "shell_command",
        "python_source",
    }
).union(item.lower().replace("-", "_") for item in FORBIDDEN_HOLE_TYPES)


class HoleResolutionError(ProcedureContractError):
    """A typed hole request, route, or validation is unsafe or malformed."""


class HoleTypeError(HoleResolutionError):
    """The hole type is forbidden, unknown, or not allowed to call a provider."""


class HoleProviderError(HoleResolutionError):
    """A provider class, bound, or route admission is not approved."""


class HoleContextError(HoleResolutionError):
    """Compiled hole context is stale, over budget, or unbound."""


class HoleValidationError(HoleResolutionError):
    """Independent hole validation could not be admitted."""


class HoleAction(str, Enum):
    CANDIDATE = "candidate"
    SUPPRESS = "suppress"
    REFUSE = "refuse"
    FALLBACK = "fallback"


class HoleReason(str, Enum):
    ROUTED_CANDIDATE = "routed-candidate"
    IDENTICAL_FAILURE = "identical-failure"
    NO_NEW_EVIDENCE = "no-new-evidence"
    FORBIDDEN_HOLE_TYPE = "forbidden-hole-type"
    UNKNOWN_HOLE_TYPE = "unknown-hole-type"
    PROVIDER_NOT_ALLOWED = "provider-not-allowed"
    PROVIDER_ROUTE_UNAVAILABLE = "provider-route-unavailable"
    CONTEXT_BUDGET_EXCEEDED = "context-budget-exceeded"
    ATTEMPT_BOUND = "attempt-bound"
    STALE_CONTEXT = "stale-context"
    INJECTION = "injection"
    EFFECT_ESCALATION = "effect-escalation"
    AUTHORITY_FLOW = "authority-flow"
    SCHEMA_MISMATCH = "schema-mismatch"
    VALIDATION_REQUIRED = "validation-required"
    FALLBACK = "fallback"
    PROMPT_SELECTED_PROVIDER = "prompt-selected-provider"
    MODEL_BEFORE_DETERMINISTIC_ROUTE = "model-before-deterministic-route"
    CANDIDATE_TIER_REQUIRED = "candidate-tier-required"


class HoleProviderStatus(str, Enum):
    CANDIDATE = "candidate"
    MISS = "miss"
    FAILURE = "failure"


class HoleValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFUSED = "refused"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise HoleResolutionError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _hole(value: Any) -> ProcedureHole:
    return _nested(value, ProcedureHole, "hole")


def allowed_hole_types() -> tuple[HoleType, ...]:
    return tuple(HoleType)


def forbidden_hole_types() -> frozenset[str]:
    return FORBIDDEN_HOLE_TYPES


def assert_allowed_hole_type(value: Any) -> HoleType:
    """Admit only the closed HoleType vocabulary; forbidden names fail closed."""

    if isinstance(value, HoleType):
        if value.value in FORBIDDEN_HOLE_TYPES:
            raise HoleTypeError("forbidden hole type cannot call a provider")
        return value
    if type(value) is not str:
        raise HoleTypeError("hole_type must be a string")
    if value in FORBIDDEN_HOLE_TYPES:
        raise HoleTypeError("forbidden hole type cannot call a provider")
    try:
        hole_type = HoleType(value)
    except ValueError as exc:
        raise HoleTypeError("unknown hole type cannot call a provider") from exc
    if hole_type.value in FORBIDDEN_HOLE_TYPES:
        raise HoleTypeError("forbidden hole type cannot call a provider")
    return hole_type


def _normalize_marker(value: str) -> str:
    return value.lower().replace("-", "_")


def _injection_hit(key: str) -> bool:
    normalized = _normalize_marker(key)
    if _unsafe_key(key) or normalized in _INJECTION_MARKERS:
        return True
    return any(marker in normalized for marker in _INJECTION_MARKERS)


def scan_injection(value: Any, field_name: str = "payload") -> str | None:
    """Return the first injection marker, or None when the payload is closed."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or _injection_hit(raw_key):
                return f"{field_name} contains a forbidden injection field"
            nested = scan_injection(item, field_name)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            nested = scan_injection(item, field_name)
            if nested is not None:
                return nested
        return None
    if type(value) is str:
        normalized = _normalize_marker(value)
        if normalized in _INJECTION_MARKERS or normalized in {
            _normalize_marker(item) for item in FORBIDDEN_HOLE_TYPES
        }:
            return f"{field_name} contains a forbidden injection value"
    return None


def _payload_bytes(value: Any) -> int:
    return len(canonical_json_bytes(_freeze(value, "payload")))


def _context_reference(value: Any) -> ContextReference:
    if isinstance(value, ContextReference):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return ContextReference.from_dict(value)
        return ContextReference(
            reference_id=_identifier(value.get("reference_id", ""), "reference_id"),
            kind=_text(value.get("kind", "hole-evidence"), "kind"),
            referenced_content_id=_identifier(
                value.get("referenced_content_id", value.get("content_id", "")),
                "referenced_content_id",
                required=False,
            ),
            repository_id=_text(
                value.get("repository_id", ""), "repository_id", required=False
            ),
            tree_id=_text(value.get("tree_id", ""), "tree_id", required=False),
            summary=_text(value.get("summary", ""), "summary", required=False),
            byte_count=_nonnegative_int(value.get("byte_count", 0), "byte_count"),
            token_count=_nonnegative_int(value.get("token_count", 0), "token_count"),
        )
    raise HoleContextError("context references must be ContextReference records")


def _utf8_tokenizer(text: str) -> int:
    encoded = text.encode("utf-8")
    return max(1, (len(encoded) + 3) // 4)


def default_hole_context_compiler(context_budget_bytes: int) -> ContextCompiler:
    """Build a provider-aware compiler whose hole evidence stays in budget."""

    budget_bytes = _positive_int(
        context_budget_bytes, "context_budget_bytes", maximum=MAX_CONTEXT_BUDGET_BYTES
    )
    # Hole evidence is still bounded by context_budget_bytes.  The compiler
    # token window must be large enough for the invariant core even when the
    # hole's evidence budget is small.
    token_ceiling = max(2_048, min(8_192, max(1, budget_bytes // 4)))
    item_bytes = min(16_384, max(4_096, budget_bytes))
    text_bytes = min(item_bytes, max(1_024, budget_bytes))
    return ContextCompiler(
        ContextBudget(
            max_input_tokens=token_ceiling,
            reserved_output_tokens=64,
            reserved_tool_tokens=32,
            max_items=32,
            max_item_bytes=item_bytes,
            max_serialized_bytes=262_144,
            max_text_bytes=text_bytes,
        ),
        tokenizer=_utf8_tokenizer,
    )


@dataclass(frozen=True)
class CompiledHoleContext:
    """Replayable ContextCompiler result bound to one hole request."""

    capsule_id: str
    receipt_cid: str
    repository_id: str
    tree_id: str
    input_tokens: int
    evidence_bytes: int
    evidence_cids: tuple[str, ...]
    reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("capsule_id", "receipt_cid", "repository_id", "tree_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "input_tokens", _nonnegative_int(self.input_tokens, "input_tokens")
        )
        object.__setattr__(
            self,
            "evidence_bytes",
            _nonnegative_int(self.evidence_bytes, "evidence_bytes"),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(
            self,
            "reference_ids",
            _strings(self.reference_ids, "reference_ids", identifiers=True),
        )


@dataclass(frozen=True)
class HoleProviderResult:
    """Closed provider proposal.  A candidate is never a validated fill."""

    status: HoleProviderStatus
    output: Mapping[str, Any] = field(default_factory=dict)
    output_schema_ref: str = ""
    failure_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, HoleProviderStatus, "status"))
        object.__setattr__(self, "output", _freeze(self.output, "output"))
        if not isinstance(self.output, Mapping):
            raise HoleProviderError("provider output must be a mapping")
        object.__setattr__(
            self,
            "output_schema_ref",
            _identifier(self.output_schema_ref, "output_schema_ref", required=False),
        )
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        if self.status is HoleProviderStatus.FAILURE and not self.failure_signature:
            raise HoleProviderError("provider failure requires a failure signature")


class HoleProvider(Protocol):
    """Approved hole provider.  Implementations return proposals only."""

    provider_class: ProviderClass

    def propose(
        self, request: "HoleRequest", compiled: CompiledHoleContext
    ) -> HoleProviderResult:
        """Return a candidate, miss, or failure. Never a validated fill."""
        raise HoleProviderError("hole providers must implement propose")


@dataclass(frozen=True)
class HoleAttemptRecord:
    """One recorded provider call used to suppress identical retries."""

    hole_id: str
    input_cid: str
    context_cid: str
    provider_class: ProviderClass
    failure_signature: str
    evidence_cids: tuple[str, ...]
    model_called: bool

    def __post_init__(self) -> None:
        for name in ("hole_id", "input_cid", "context_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "provider_class",
            _enum(self.provider_class, ProviderClass, "provider_class"),
        )
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(self, "model_called", _bool(self.model_called, "model_called"))


class HoleAttemptLog:
    """In-memory attempt history.  Identical failure needs new evidence."""

    def __init__(self) -> None:
        self._records: list[HoleAttemptRecord] = []

    def records(self) -> tuple[HoleAttemptRecord, ...]:
        return tuple(self._records)

    def record(self, item: HoleAttemptRecord) -> None:
        self._records.append(item)

    def matching(
        self,
        *,
        hole_id: str,
        input_cid: str,
        provider_class: ProviderClass | None = None,
    ) -> tuple[HoleAttemptRecord, ...]:
        return tuple(
            item
            for item in self._records
            if item.hole_id == hole_id
            and item.input_cid == input_cid
            and (provider_class is None or item.provider_class is provider_class)
        )


def _suppression_reason(
    history: Sequence[HoleAttemptRecord],
    *,
    evidence_cids: Sequence[str],
    provider_class: ProviderClass,
) -> HoleReason | None:
    current_evidence = set(evidence_cids)
    for item in history:
        new_evidence = current_evidence - set(item.evidence_cids)
        if new_evidence:
            continue
        if item.provider_class is provider_class and item.failure_signature:
            return HoleReason.IDENTICAL_FAILURE
        if item.model_called and provider_class in MODEL_PROVIDER_CLASSES:
            return HoleReason.NO_NEW_EVIDENCE
    return None


@dataclass(frozen=True)
class HoleRequest(CanonicalContract):
    """Typed request to fill one residual procedure hole."""

    SCHEMA: ClassVar[str] = _schema_name("TypedHoleRequest")

    bindings: ArtifactBindings
    hole: ProcedureHole
    request_id: str
    input_payload: Mapping[str, Any] = field(default_factory=dict)
    context_reference_ids: tuple[str, ...] = ()
    evidence_cids: tuple[str, ...] = ()
    attempt_index: int = 1
    step_id: str = ""
    context_tree_id: str = ""
    expected_context_cid: str = ""
    preferred_provider_class: ProviderClass | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "hole", _hole(self.hole))
        assert_allowed_hole_type(self.hole.hole_type)
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        payload = _freeze(self.input_payload, "input_payload")
        if not isinstance(payload, Mapping):
            raise HoleResolutionError("input_payload must be a mapping")
        injection = scan_injection(payload, "input_payload")
        if injection is not None:
            raise HoleResolutionError(injection)
        object.__setattr__(self, "input_payload", payload)
        object.__setattr__(
            self,
            "context_reference_ids",
            _strings(self.context_reference_ids, "context_reference_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(
            self,
            "attempt_index",
            _positive_int(self.attempt_index, "attempt_index", maximum=MAX_HOLE_ATTEMPTS),
        )
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id", required=False))
        object.__setattr__(
            self,
            "context_tree_id",
            _identifier(self.context_tree_id, "context_tree_id", required=False),
        )
        object.__setattr__(
            self,
            "expected_context_cid",
            _identifier(self.expected_context_cid, "expected_context_cid", required=False),
        )
        preferred = self.preferred_provider_class
        if preferred is not None:
            preferred = _enum(preferred, ProviderClass, "preferred_provider_class")
            object.__setattr__(self, "preferred_provider_class", preferred)
        if not set(self.hole.effect_classes).issubset(ALLOWED_HOLE_EFFECTS):
            raise HoleResolutionError("hole effect classes exceed observe/model-request")
        if not self.hole.validation_observation_ids:
            raise HoleResolutionError("typed holes require validation observations")
        if self.attempt_index > self.hole.maximum_attempts:
            raise HoleResolutionError("attempt_index exceeds the hole attempt bound")
        if self.context_tree_id and self.context_tree_id != self.bindings.tree_id:
            raise HoleContextError("stale context cannot be used for a typed hole")
        _bounded(self, "HoleRequest")

    @property
    def hole_id(self) -> str:
        return self.hole.hole_id

    @property
    def hole_type(self) -> HoleType:
        return self.hole.hole_type

    @property
    def input_cid(self) -> str:
        return _request_input_cid(self)

    @property
    def allowed_provider_classes(self) -> tuple[ProviderClass, ...]:
        return self.hole.allowed_provider_classes

    @property
    def can_authorize(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "hole": self.hole,
            "request_id": self.request_id,
            "input_payload": dict(self.input_payload),
            "context_reference_ids": self.context_reference_ids,
            "evidence_cids": self.evidence_cids,
            "attempt_index": self.attempt_index,
            "step_id": self.step_id,
            "context_tree_id": self.context_tree_id,
            "expected_context_cid": self.expected_context_cid,
            "preferred_provider_class": (
                None
                if self.preferred_provider_class is None
                else self.preferred_provider_class.value
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleRequest:
        fields = (
            "bindings",
            "hole",
            "request_id",
            "input_payload",
            "context_reference_ids",
            "evidence_cids",
            "attempt_index",
            "step_id",
            "context_tree_id",
            "expected_context_cid",
            "preferred_provider_class",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "hole" in values:
            values["hole"] = _hole(values["hole"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record

    def to_artifact(self, *, emitted_at_ms: int = 0) -> HoleRequestArtifact:
        return HoleRequestArtifact(
            bindings=self.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.hole.hole_id,
            reference_cids=self.evidence_cids,
            labels=(self.hole.hole_type.value, "typed-hole"),
            facts={
                "request_id": self.request_id,
                "hole_id": self.hole.hole_id,
                "hole_type": self.hole.hole_type.value,
                "attempt_index": self.attempt_index,
                "allowed_provider_classes": tuple(
                    item.value for item in self.hole.allowed_provider_classes
                ),
                "context_budget_bytes": self.hole.context_budget_bytes,
                "maximum_attempts": self.hole.maximum_attempts,
                "validation_observation_ids": self.hole.validation_observation_ids,
                "fallback_step_id": self.hole.fallback_step_id,
                "can_authorize": False,
            },
            created_at_ms=emitted_at_ms,
        )


def _request_input_cid(request: HoleRequest) -> str:
    return content_identity(
        {
            "hole_id": request.hole.hole_id,
            "input_payload": dict(request.input_payload),
        }
    )


@dataclass(frozen=True)
class HoleCandidate(CanonicalContract):
    """Provider output at candidate tier.  Validation is a separate step."""

    SCHEMA: ClassVar[str] = _schema_name("TypedHoleCandidate")

    bindings: ArtifactBindings
    request_cid: str
    hole_id: str
    hole_type: HoleType
    provider_class: ProviderClass
    output_schema_ref: str
    output: Mapping[str, Any]
    compiled_context_cid: str = ""
    state: ArtifactState = ArtifactState.CANDIDATE
    resolver_revision: str = RESOLVER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("request_cid", "hole_id", "output_schema_ref"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "hole_type", assert_allowed_hole_type(self.hole_type))
        object.__setattr__(
            self, "provider_class", _enum(self.provider_class, ProviderClass, "provider_class")
        )
        output = _freeze(self.output, "output")
        if not isinstance(output, Mapping):
            raise HoleResolutionError("candidate output must be a mapping")
        injection = scan_injection(output, "output")
        if injection is not None:
            raise HoleResolutionError(injection)
        if _payload_bytes(output) > MAX_OUTPUT_BYTES:
            raise HoleResolutionError("candidate output exceeds its byte bound")
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "compiled_context_cid",
            _identifier(self.compiled_context_cid, "compiled_context_cid", required=False),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleResolutionError("hole outputs remain candidates until validation")
        object.__setattr__(
            self, "resolver_revision", _identifier(self.resolver_revision, "resolver_revision")
        )
        if self.resolver_revision != RESOLVER_REVISION:
            raise HoleResolutionError("hole candidate resolver revision is not current")
        _bounded(self, "HoleCandidate")

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    @property
    def can_suppress_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "request_cid": self.request_cid,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "provider_class": self.provider_class.value,
            "output_schema_ref": self.output_schema_ref,
            "output": dict(self.output),
            "compiled_context_cid": self.compiled_context_cid,
            "state": self.state.value,
            "resolver_revision": self.resolver_revision,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleCandidate:
        fields = (
            "bindings",
            "request_cid",
            "hole_id",
            "hole_type",
            "provider_class",
            "output_schema_ref",
            "output",
            "compiled_context_cid",
            "state",
            "resolver_revision",
            "can_authorize",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if values.pop("can_authorize", False):
            raise HoleResolutionError("hole candidates cannot authorize")
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record

    def to_artifact(self, *, emitted_at_ms: int = 0) -> HoleCandidateArtifact:
        return HoleCandidateArtifact(
            bindings=self.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.hole_id,
            reference_cids=(self.request_cid, self.compiled_context_cid)
            if self.compiled_context_cid
            else (self.request_cid,),
            labels=(self.hole_type.value, self.provider_class.value, "candidate"),
            facts={
                "request_cid": self.request_cid,
                "hole_id": self.hole_id,
                "provider_class": self.provider_class.value,
                "output_schema_ref": self.output_schema_ref,
                "can_authorize": False,
                "can_promote": False,
                "can_establish_completion": False,
            },
            created_at_ms=emitted_at_ms,
        )


@dataclass(frozen=True)
class HoleResolution(CanonicalContract):
    """Routed result of one hole-resolution attempt."""

    SCHEMA: ClassVar[str] = _schema_name("TypedHoleResolution")

    bindings: ArtifactBindings
    request_cid: str
    hole_id: str
    action: HoleAction
    reason_code: HoleReason
    attempted_provider_classes: tuple[ProviderClass, ...] = ()
    called_provider_class: ProviderClass | None = None
    candidate: HoleCandidate | None = None
    fallback_step_id: str = ""
    compiled_context_cid: str = ""
    provider_called: bool = False
    resolver_revision: str = RESOLVER_REVISION
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "request_cid", _identifier(self.request_cid, "request_cid"))
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "action", _enum(self.action, HoleAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, HoleReason, "reason_code")
        )
        object.__setattr__(
            self,
            "attempted_provider_classes",
            _enums(
                self.attempted_provider_classes,
                ProviderClass,
                "attempted_provider_classes",
                limit=len(ProviderClass),
            ),
        )
        called = self.called_provider_class
        if called is not None:
            called = _enum(called, ProviderClass, "called_provider_class")
            object.__setattr__(self, "called_provider_class", called)
        if self.candidate is not None:
            object.__setattr__(
                self, "candidate", _nested(self.candidate, HoleCandidate, "candidate")
            )
        object.__setattr__(
            self,
            "fallback_step_id",
            _identifier(self.fallback_step_id, "fallback_step_id", required=False),
        )
        object.__setattr__(
            self,
            "compiled_context_cid",
            _identifier(self.compiled_context_cid, "compiled_context_cid", required=False),
        )
        object.__setattr__(self, "provider_called", _bool(self.provider_called, "provider_called"))
        object.__setattr__(
            self, "resolver_revision", _identifier(self.resolver_revision, "resolver_revision")
        )
        if self.resolver_revision != RESOLVER_REVISION:
            raise HoleResolutionError("hole resolution resolver revision is not current")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleResolutionError("hole resolutions remain candidates until validation")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise HoleResolutionError("hole resolutions cannot authorize")
        if self.action is HoleAction.CANDIDATE:
            if self.candidate is None or not self.provider_called:
                raise HoleResolutionError("a candidate resolution requires a provider output")
            if self.reason_code is not HoleReason.ROUTED_CANDIDATE:
                raise HoleResolutionError("a candidate resolution must be labeled routed-candidate")
        elif self.action is HoleAction.SUPPRESS and self.provider_called:
            raise HoleResolutionError("a suppressed resolution cannot call a provider")
        elif self.action is HoleAction.FALLBACK and not self.fallback_step_id:
            raise HoleResolutionError("fallback resolution requires a fallback step")
        _bounded(self, "HoleResolution")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    @property
    def can_suppress_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "request_cid": self.request_cid,
            "hole_id": self.hole_id,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "attempted_provider_classes": tuple(
                item.value for item in self.attempted_provider_classes
            ),
            "called_provider_class": (
                None if self.called_provider_class is None else self.called_provider_class.value
            ),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "fallback_step_id": self.fallback_step_id,
            "compiled_context_cid": self.compiled_context_cid,
            "provider_called": self.provider_called,
            "resolver_revision": self.resolver_revision,
            "state": self.state.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleResolution:
        fields = (
            "bindings",
            "request_cid",
            "hole_id",
            "action",
            "reason_code",
            "attempted_provider_classes",
            "called_provider_class",
            "candidate",
            "fallback_step_id",
            "compiled_context_cid",
            "provider_called",
            "resolver_revision",
            "state",
            "can_authorize",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if values.get("can_authorize"):
            raise HoleResolutionError("hole resolutions cannot authorize")
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if values.get("candidate") is not None:
            values["candidate"] = _nested(values["candidate"], HoleCandidate, "candidate")
        record = cls(**values)
        _verify_identity(payload, record)
        return record

    def to_artifact(self, *, emitted_at_ms: int = 0) -> HoleResolutionArtifact:
        references = [self.request_cid]
        if self.compiled_context_cid:
            references.append(self.compiled_context_cid)
        if self.candidate is not None:
            references.append(self.candidate.content_id)
        return HoleResolutionArtifact(
            bindings=self.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.hole_id,
            reference_cids=tuple(references),
            labels=(self.action.value, self.reason_code.value),
            facts={
                "action": self.action.value,
                "reason_code": self.reason_code.value,
                "provider_called": self.provider_called,
                "fallback_step_id": self.fallback_step_id,
                "can_authorize": False,
                "can_promote": False,
                "can_suppress_validation": False,
            },
            created_at_ms=emitted_at_ms,
        )


@dataclass(frozen=True)
class IndependentHoleObservation:
    """Externally admitted observation required to validate a hole fill."""

    observation_id: str
    producer_id: str
    admitted: bool
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self, "evidence_cid", _identifier(self.evidence_cid, "evidence_cid", required=False)
        )


@dataclass(frozen=True)
class HoleValidationReceipt(CanonicalContract):
    """Independent validation of a hole candidate.  Never a promotion."""

    SCHEMA: ClassVar[str] = _schema_name("TypedHoleValidationReceipt")

    bindings: ArtifactBindings
    candidate_cid: str
    hole_id: str
    status: HoleValidationStatus
    observation_ids: tuple[str, ...]
    validator_id: str
    provider_id: str = ""
    reason_code: HoleReason | str = ""
    independent: bool = True
    state: ArtifactState = ArtifactState.CANDIDATE
    validator_revision: str = VALIDATOR_REVISION
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "candidate_cid", _identifier(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(
            self, "status", _enum(self.status, HoleValidationStatus, "status")
        )
        object.__setattr__(
            self,
            "observation_ids",
            _strings(self.observation_ids, "observation_ids", identifiers=True),
        )
        object.__setattr__(self, "validator_id", _identifier(self.validator_id, "validator_id"))
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, "provider_id", required=False)
        )
        reason = self.reason_code
        if reason in (None, ""):
            object.__setattr__(self, "reason_code", "")
        elif isinstance(reason, HoleReason):
            object.__setattr__(self, "reason_code", reason.value)
        else:
            object.__setattr__(self, "reason_code", _identifier(reason, "reason_code"))
        object.__setattr__(self, "independent", _bool(self.independent, "independent"))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise HoleValidationError("validated hole outputs remain candidates")
        object.__setattr__(
            self,
            "validator_revision",
            _identifier(self.validator_revision, "validator_revision"),
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise HoleValidationError("hole validation revision is not current")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            raise HoleValidationError("hole validation cannot authorize")
        if not self.independent:
            raise HoleValidationError("hole validation must be independent of the provider")
        if self.provider_id and self.provider_id == self.validator_id:
            raise HoleValidationError("hole validation cannot be produced by the provider")
        _bounded(self, "HoleValidationReceipt")

    @property
    def accepted(self) -> bool:
        return self.status is HoleValidationStatus.ACCEPTED

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    @property
    def can_suppress_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "candidate_cid": self.candidate_cid,
            "hole_id": self.hole_id,
            "status": self.status.value,
            "observation_ids": self.observation_ids,
            "validator_id": self.validator_id,
            "provider_id": self.provider_id,
            "reason_code": self.reason_code,
            "independent": True,
            "state": self.state.value,
            "validator_revision": self.validator_revision,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HoleValidationReceipt:
        fields = (
            "bindings",
            "candidate_cid",
            "hole_id",
            "status",
            "observation_ids",
            "validator_id",
            "provider_id",
            "reason_code",
            "independent",
            "state",
            "validator_revision",
            "can_authorize",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if values.get("can_authorize"):
            raise HoleValidationError("hole validation cannot authorize")
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record

    def to_artifact(self, *, emitted_at_ms: int = 0) -> HoleValidationReceiptArtifact:
        return HoleValidationReceiptArtifact(
            bindings=self.bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.candidate_cid,
            reference_cids=(self.candidate_cid, *self.observation_ids),
            labels=(self.status.value, "independent-validation"),
            facts={
                "hole_id": self.hole_id,
                "accepted": self.accepted,
                "independent": True,
                "can_authorize": False,
                "can_promote": False,
                "can_suppress_validation": False,
            },
            created_at_ms=emitted_at_ms,
        )


class HoleResolutionValidator:
    """Admit or reject a hole candidate from independent observations."""

    revision: ClassVar[str] = VALIDATOR_REVISION
    validator_id: str = "hole-resolution-validator"

    def __init__(self, *, validator_id: str = "hole-resolution-validator") -> None:
        self.validator_id = _identifier(validator_id, "validator_id")

    def validate(
        self,
        candidate: HoleCandidate,
        *,
        hole: ProcedureHole,
        observations: Sequence[IndependentHoleObservation | Mapping[str, Any]] = (),
        provider_id: str = "",
    ) -> HoleValidationReceipt:
        if not isinstance(candidate, HoleCandidate):
            raise HoleValidationError("validation requires a HoleCandidate")
        hole = _hole(hole)
        if candidate.hole_id != hole.hole_id:
            raise HoleValidationError("candidate does not bind the declared hole")
        if candidate.output_schema_ref != hole.output_schema_ref:
            return self._receipt(
                candidate,
                HoleValidationStatus.REJECTED,
                hole.validation_observation_ids,
                provider_id,
                HoleReason.SCHEMA_MISMATCH,
            )
        if candidate.state is not ArtifactState.CANDIDATE:
            return self._receipt(
                candidate,
                HoleValidationStatus.REFUSED,
                hole.validation_observation_ids,
                provider_id,
                HoleReason.CANDIDATE_TIER_REQUIRED,
            )
        injection = scan_injection(candidate.output, "output")
        if injection is not None:
            return self._receipt(
                candidate,
                HoleValidationStatus.REFUSED,
                hole.validation_observation_ids,
                provider_id,
                HoleReason.INJECTION,
            )
        records = tuple(
            item
            if isinstance(item, IndependentHoleObservation)
            else IndependentHoleObservation(**item)
            for item in observations
        )
        admitted = {
            item.observation_id: item
            for item in records
            if item.admitted and item.producer_id != (provider_id or candidate.provider_class.value)
        }
        missing = tuple(
            observation_id
            for observation_id in hole.validation_observation_ids
            if observation_id not in admitted
        )
        if missing:
            return self._receipt(
                candidate,
                HoleValidationStatus.REFUSED,
                hole.validation_observation_ids,
                provider_id,
                HoleReason.VALIDATION_REQUIRED,
            )
        return self._receipt(
            candidate,
            HoleValidationStatus.ACCEPTED,
            hole.validation_observation_ids,
            provider_id,
            "",
        )

    def _receipt(
        self,
        candidate: HoleCandidate,
        status: HoleValidationStatus,
        observation_ids: Sequence[str],
        provider_id: str,
        reason: HoleReason | str,
    ) -> HoleValidationReceipt:
        return HoleValidationReceipt(
            bindings=candidate.bindings,
            candidate_cid=candidate.content_id,
            hole_id=candidate.hole_id,
            status=status,
            observation_ids=tuple(observation_ids),
            validator_id=self.validator_id,
            provider_id=provider_id,
            reason_code=reason,
        )


class HoleResolver:
    """Route an allowed typed hole through approved providers in bound order."""

    revision: ClassVar[str] = RESOLVER_REVISION

    def __init__(
        self,
        *,
        compiler: ContextCompiler | None = None,
        attempt_log: HoleAttemptLog | None = None,
        provider_route_evaluate: Callable[..., ProviderRouteEvaluation] | None = None,
        validator_id: str = "hole-resolution-validator",
    ) -> None:
        self._compiler = compiler
        self._log = attempt_log or HoleAttemptLog()
        self._provider_route_evaluate = provider_route_evaluate or evaluate_preferred_route
        self._validator_id = _identifier(validator_id, "validator_id")

    @property
    def attempt_log(self) -> HoleAttemptLog:
        return self._log

    def resolve(
        self,
        request: HoleRequest,
        *,
        providers: Sequence[HoleProvider] = (),
        context_references: Sequence[ContextReference | Mapping[str, Any]] = (),
        current_tree_id: str = "",
        current_repository_id: str = "",
        provider_route_healthy: bool = True,
    ) -> HoleResolution:
        if not isinstance(request, HoleRequest):
            raise HoleResolutionError("request must be a HoleRequest")
        assert_allowed_hole_type(request.hole.hole_type)
        if request.preferred_provider_class is not None:
            return self._finish(
                request,
                HoleAction.REFUSE,
                HoleReason.PROMPT_SELECTED_PROVIDER,
            )
        if current_tree_id and current_tree_id != request.bindings.tree_id:
            return self._finish(request, HoleAction.REFUSE, HoleReason.STALE_CONTEXT)
        if current_repository_id and current_repository_id != request.bindings.repository_id:
            return self._finish(request, HoleAction.REFUSE, HoleReason.STALE_CONTEXT)
        if request.attempt_index > request.hole.maximum_attempts:
            return self._finish(request, HoleAction.REFUSE, HoleReason.ATTEMPT_BOUND)
        if _payload_bytes(request.input_payload) > request.hole.context_budget_bytes:
            return self._finish(request, HoleAction.REFUSE, HoleReason.CONTEXT_BUDGET_EXCEEDED)

        try:
            compiled = self._compile_context(request, context_references)
        except (HoleContextError, ContextContractError) as exc:
            message = str(exc).lower()
            reason = (
                HoleReason.STALE_CONTEXT
                if "stale" in message or "does not match" in message
                else HoleReason.CONTEXT_BUDGET_EXCEEDED
            )
            return self._finish(request, HoleAction.REFUSE, reason)

        input_cid = _request_input_cid(request)
        allowed = tuple(
            item
            for item in PROVIDER_ROUTE_ORDER
            if item in request.hole.allowed_provider_classes
        )
        if not allowed:
            return self._finish(
                request,
                HoleAction.REFUSE,
                HoleReason.PROVIDER_NOT_ALLOWED,
                compiled_context_cid=compiled.receipt_cid,
            )
        by_class: dict[ProviderClass, HoleProvider] = {}
        for provider in providers:
            provider_class = _enum(provider.provider_class, ProviderClass, "provider_class")
            if provider_class in by_class:
                continue
            by_class[provider_class] = provider

        attempted: list[ProviderClass] = []
        history = self._log.matching(
            hole_id=request.hole.hole_id,
            input_cid=input_cid,
        )
        model_calls = 0
        for provider_class in allowed:
            if provider_class in MODEL_PROVIDER_CLASSES and model_calls >= 1:
                # One model call per resolve(); further models need a later attempt.
                break
            suppression = _suppression_reason(
                history,
                evidence_cids=request.evidence_cids,
                provider_class=provider_class,
            )
            if suppression is HoleReason.IDENTICAL_FAILURE or (
                suppression is HoleReason.NO_NEW_EVIDENCE
                and provider_class in MODEL_PROVIDER_CLASSES
            ):
                return self._finish(
                    request,
                    HoleAction.SUPPRESS,
                    suppression,
                    attempted=tuple(attempted),
                    compiled_context_cid=compiled.receipt_cid,
                )
            if provider_class not in by_class:
                attempted.append(provider_class)
                continue
            if provider_class in REMOTE_PROVIDER_CLASSES:
                try:
                    route = self._provider_route_evaluate(
                        preferred_healthy=provider_route_healthy
                    )
                except ProviderRouteError:
                    attempted.append(provider_class)
                    continue
                if not route.admitted:
                    attempted.append(provider_class)
                    if all(
                        item in REMOTE_PROVIDER_CLASSES or item in MODEL_PROVIDER_CLASSES
                        for item in allowed
                    ) and provider_class == allowed[-1]:
                        return self._finish(
                            request,
                            HoleAction.FALLBACK,
                            HoleReason.PROVIDER_ROUTE_UNAVAILABLE,
                            attempted=tuple(attempted),
                            compiled_context_cid=compiled.receipt_cid,
                            fallback_step_id=request.hole.fallback_step_id,
                        )
                    continue
            attempted.append(provider_class)
            result = by_class[provider_class].propose(request, compiled)
            if result.status is HoleProviderStatus.MISS:
                continue
            if result.status is HoleProviderStatus.FAILURE:
                self._log.record(
                    HoleAttemptRecord(
                        hole_id=request.hole.hole_id,
                        input_cid=input_cid,
                        context_cid=compiled.receipt_cid,
                        provider_class=provider_class,
                        failure_signature=result.failure_signature,
                        evidence_cids=request.evidence_cids,
                        model_called=provider_class in MODEL_PROVIDER_CLASSES,
                    )
                )
                if provider_class in MODEL_PROVIDER_CLASSES:
                    model_calls += 1
                continue
            if result.output_schema_ref and result.output_schema_ref != request.hole.output_schema_ref:
                return self._finish(
                    request,
                    HoleAction.REFUSE,
                    HoleReason.SCHEMA_MISMATCH,
                    attempted=tuple(attempted),
                    compiled_context_cid=compiled.receipt_cid,
                    called=provider_class,
                    provider_called=True,
                )
            injection = scan_injection(result.output, "output")
            if injection is not None:
                return self._finish(
                    request,
                    HoleAction.REFUSE,
                    HoleReason.INJECTION,
                    attempted=tuple(attempted),
                    compiled_context_cid=compiled.receipt_cid,
                    called=provider_class,
                    provider_called=True,
                )
            if provider_class in MODEL_PROVIDER_CLASSES:
                model_calls += 1
                self._log.record(
                    HoleAttemptRecord(
                        hole_id=request.hole.hole_id,
                        input_cid=input_cid,
                        context_cid=compiled.receipt_cid,
                        provider_class=provider_class,
                        failure_signature="",
                        evidence_cids=request.evidence_cids,
                        model_called=True,
                    )
                )
            candidate = HoleCandidate(
                bindings=request.bindings,
                request_cid=request.content_id,
                hole_id=request.hole.hole_id,
                hole_type=request.hole.hole_type,
                provider_class=provider_class,
                output_schema_ref=request.hole.output_schema_ref,
                output=result.output,
                compiled_context_cid=compiled.receipt_cid,
            )
            return self._finish(
                request,
                HoleAction.CANDIDATE,
                HoleReason.ROUTED_CANDIDATE,
                attempted=tuple(attempted),
                compiled_context_cid=compiled.receipt_cid,
                called=provider_class,
                provider_called=True,
                candidate=candidate,
            )
        return self._finish(
            request,
            HoleAction.FALLBACK,
            HoleReason.FALLBACK,
            attempted=tuple(attempted),
            compiled_context_cid=compiled.receipt_cid,
            fallback_step_id=request.hole.fallback_step_id,
        )

    def _compile_context(
        self,
        request: HoleRequest,
        context_references: Sequence[ContextReference | Mapping[str, Any]],
    ) -> CompiledHoleContext:
        references = tuple(_context_reference(item) for item in context_references)
        for item in references:
            if item.tree_id and item.tree_id != request.bindings.tree_id:
                raise HoleContextError("stale context cannot be used for a typed hole")
            if item.repository_id and item.repository_id != request.bindings.repository_id:
                raise HoleContextError("stale context cannot be used for a typed hole")
        if request.context_reference_ids:
            present = {item.reference_id for item in references}
            missing = [item for item in request.context_reference_ids if item not in present]
            if missing:
                raise HoleContextError("stale context cannot be used for a typed hole")
        evidence_bytes = 0
        evidence_cids: list[str] = []
        for item in references:
            size = max(item.byte_count, len(item.canonical_bytes()))
            evidence_bytes += size
            if item.referenced_content_id:
                evidence_cids.append(item.referenced_content_id)
        if evidence_bytes > request.hole.context_budget_bytes:
            raise HoleContextError("hole context exceeds its byte budget")
        compiler = self._compiler or default_hole_context_compiler(
            request.hole.context_budget_bytes
        )
        try:
            compiled = compiler.compile(
                repository_id=request.bindings.repository_id,
                tree_id=request.bindings.tree_id,
                objective_id=request.bindings.objective_id,
                objective_revision=request.bindings.contract_revision,
                policy_id=request.bindings.policy_revision,
                policy_revision=request.bindings.policy_revision,
                caller=HOLE_RESOLVER_CALLER,
                stage=HOLE_RESOLUTION_STAGE,
                goal={
                    "task_id": request.bindings.task_id,
                    "hole_id": request.hole.hole_id,
                    "hole_type": request.hole.hole_type.value,
                },
                authority={
                    "mode": "proposal",
                    "requirement_ids": list(request.hole.authority_requirement_ids),
                },
                scope={
                    "effect_classes": [item.value for item in request.hole.effect_classes],
                    "paths": [],
                },
                acceptance={
                    "criteria": [
                        "outputs remain candidates until validation",
                        "identical failure suppresses another call",
                    ]
                },
                evidence=references,
            )
        except (RequiredContextOverflowError, ContextCompilationError, ContextContractError) as exc:
            message = str(exc).lower()
            if "stale" in message or "does not match" in message:
                raise HoleContextError("stale context cannot be used for a typed hole") from exc
            raise HoleContextError("hole context exceeds its byte budget") from exc
        if compiled.capsule.tree_id != request.bindings.tree_id:
            raise HoleContextError("stale context cannot be used for a typed hole")
        if compiled.capsule.repository_id != request.bindings.repository_id:
            raise HoleContextError("stale context cannot be used for a typed hole")
        receipt_cid = compiled.receipt.content_id
        if request.expected_context_cid and request.expected_context_cid != receipt_cid:
            raise HoleContextError("stale context cannot be used for a typed hole")
        return CompiledHoleContext(
            capsule_id=compiled.capsule.capsule_id,
            receipt_cid=receipt_cid,
            repository_id=compiled.capsule.repository_id,
            tree_id=compiled.capsule.tree_id,
            input_tokens=compiled.capsule.input_tokens,
            evidence_bytes=evidence_bytes,
            evidence_cids=tuple(evidence_cids),
            reference_ids=tuple(item.reference_id for item in compiled.capsule.evidence),
        )

    def _finish(
        self,
        request: HoleRequest,
        action: HoleAction,
        reason: HoleReason,
        *,
        attempted: tuple[ProviderClass, ...] = (),
        compiled_context_cid: str = "",
        called: ProviderClass | None = None,
        provider_called: bool = False,
        candidate: HoleCandidate | None = None,
        fallback_step_id: str = "",
    ) -> HoleResolution:
        return HoleResolution(
            bindings=request.bindings,
            request_cid=request.content_id,
            hole_id=request.hole.hole_id,
            action=action,
            reason_code=reason,
            attempted_provider_classes=attempted,
            called_provider_class=called,
            candidate=candidate,
            fallback_step_id=fallback_step_id or (
                request.hole.fallback_step_id if action is HoleAction.FALLBACK else ""
            ),
            compiled_context_cid=compiled_context_cid,
            provider_called=provider_called,
        )


def route_provider_classes(
    allowed: Sequence[ProviderClass],
) -> tuple[ProviderClass, ...]:
    """Return allowed providers in the sealed deterministic-to-remote order."""

    allowed_set = set(_enums(allowed, ProviderClass, "allowed", limit=len(ProviderClass)))
    return tuple(item for item in PROVIDER_ROUTE_ORDER if item in allowed_set)


__all__ = [
    "ALLOWED_HOLE_EFFECTS",
    "DETERMINISTIC_PROVIDER_CLASSES",
    "HOLE_RESOLUTION_STAGE",
    "HOLE_RESOLVER_CALLER",
    "MAX_HOLE_ATTEMPTS",
    "MODEL_PROVIDER_CLASSES",
    "PROVIDER_ROUTE_ORDER",
    "REMOTE_PROVIDER_CLASSES",
    "RESOLVER_REVISION",
    "VALIDATOR_REVISION",
    "CompiledHoleContext",
    "HoleAction",
    "HoleAttemptLog",
    "HoleAttemptRecord",
    "HoleCandidate",
    "HoleContextError",
    "HoleProvider",
    "HoleProviderError",
    "HoleProviderResult",
    "HoleProviderStatus",
    "HoleReason",
    "HoleRequest",
    "HoleResolution",
    "HoleResolutionError",
    "HoleResolutionValidator",
    "HoleResolver",
    "HoleTypeError",
    "HoleValidationError",
    "HoleValidationReceipt",
    "HoleValidationStatus",
    "IndependentHoleObservation",
    "allowed_hole_types",
    "assert_allowed_hole_type",
    "default_hole_context_compiler",
    "forbidden_hole_types",
    "route_provider_classes",
    "scan_injection",
]


# Keep the generic P0 envelopes registered; richer typed-hole schemas are
# additional decoder targets used by this module.
ARTIFACT_TYPES_BY_SCHEMA[HoleRequest.SCHEMA] = HoleRequest
ARTIFACT_TYPES_BY_SCHEMA[HoleCandidate.SCHEMA] = HoleCandidate
ARTIFACT_TYPES_BY_SCHEMA[HoleResolution.SCHEMA] = HoleResolution
ARTIFACT_TYPES_BY_SCHEMA[HoleValidationReceipt.SCHEMA] = HoleValidationReceipt
