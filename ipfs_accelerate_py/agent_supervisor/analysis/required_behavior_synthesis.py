"""Independent synthesis of RequiredBehaviorContract@1 for new support types.

When change propagation needs a new class, method, data structure, provider,
factory, schema, adapter, or serializer, this module defines the required
behavior *before* placement or code generation.

Authority rules (fail-closed):

* Evidence precedence is independent of any candidate implementation or LLM
  proposal.  Rank is fixed and never inverted.
* Implementation observations are non-authoritative hypotheses only; they
  cannot admit a behavior contract or carry proof authority.
* Conflicting same-rank evidence, or insufficient authoritative evidence for
  the requested kind, yields a typed :class:`BehaviorGap` and never an
  implementation request.
* The synthesizer returns the canonical RPR-022
  :class:`RequiredBehaviorContract` and does not redefine it.

Clause families cover fields/variants/generics/invariants/defaults,
constructors/factories/totality, methods/state-machine/transitions/idempotence,
ownership/lifetime/mutation/concurrency/cache/disposal,
serialization/persistence/versioning/migrations/equality/hash,
errors/cancellation/effects/capabilities/auth/trust/privacy/resources/
degradation, and compatibility/tests/telemetry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    ChangePropagationAuthorityError,
    ChangePropagationError,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

BEHAVIOR_EVIDENCE_ATOM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/behavior-evidence-atom@1"
)
BEHAVIOR_CLAUSE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/behavior-clause-binding@1"
)
BEHAVIOR_GAP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/behavior-gap@1"
)
REQUIRED_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/required-behavior-synthesis-receipt@1"
)
PRODUCER_ID: Final[str] = "required-behavior-synthesis@1"

MAX_EVIDENCE_ATOMS: Final[int] = 512
MAX_CLAUSE_BINDINGS: Final[int] = 256
MAX_REF_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 1_024
MAX_RECORD_BYTES: Final[int] = 262_144
CONTRACT_VERSION: Final[int] = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RequiredBehaviorSynthesisError(ValueError):
    """Malformed or unsafe behavior-synthesis input."""


class RequiredBehaviorSynthesisAuthorityError(RequiredBehaviorSynthesisError):
    """Root, identity, or authority promotion failure."""


class RequiredBehaviorSynthesisBoundsError(RequiredBehaviorSynthesisError):
    """A synthesis budget was exceeded."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class BehaviorClauseFamily(str, Enum):
    """Every dimension a support-type behavior contract may define."""

    FIELDS = "fields"
    VARIANTS = "variants"
    GENERICS = "generics"
    INVARIANTS = "invariants"
    DEFAULTS = "defaults"
    CONSTRUCTORS = "constructors"
    FACTORIES = "factories"
    TOTALITY = "totality"
    METHODS = "methods"
    STATE_MACHINE = "state_machine"
    TRANSITIONS = "transitions"
    IDEMPOTENCE = "idempotence"
    OWNERSHIP = "ownership"
    LIFETIME = "lifetime"
    MUTATION = "mutation"
    CONCURRENCY = "concurrency"
    CACHE = "cache"
    DISPOSAL = "disposal"
    SERIALIZATION = "serialization"
    PERSISTENCE = "persistence"
    VERSIONING = "versioning"
    MIGRATIONS = "migrations"
    EQUALITY = "equality"
    HASH = "hash"
    ERRORS = "errors"
    CANCELLATION = "cancellation"
    EFFECTS = "effects"
    CAPABILITIES = "capabilities"
    AUTHORIZATION = "authorization"
    TRUST = "trust"
    PRIVACY = "privacy"
    RESOURCES = "resources"
    DEGRADATION = "degradation"
    COMPATIBILITY = "compatibility"
    TESTS = "tests"
    TELEMETRY = "telemetry"


class BehaviorGapKind(str, Enum):
    """Closed reasons a behavior contract cannot be admitted."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    IMPLEMENTATION_ONLY = "implementation_only"
    UNSUPPORTED_CLAUSE = "unsupported_clause"
    ROOT_MISMATCH = "root_mismatch"
    KIND_REQUIREMENT_UNMET = "kind_requirement_unmet"
    STALE_OR_CROSS_ROOT = "stale_or_cross_root"


class SynthesisDisposition(str, Enum):
    """Outcome of one synthesis attempt."""

    ADMITTED = "admitted"
    BEHAVIOR_GAP = "behavior_gap"
    ABSTAINED = "abstained"


# Fixed precedence rank: lower number wins.  Implementation is never first.
PRECEDENCE_RANK: Final[Mapping[BehaviorEvidencePrecedence, int]] = {
    BehaviorEvidencePrecedence.REVIEWED_IDL: 0,
    BehaviorEvidencePrecedence.NORMATIVE_SPEC: 1,
    BehaviorEvidencePrecedence.CALLER_POSTCONDITION: 2,
    BehaviorEvidencePrecedence.CALLEE_PRECONDITION: 3,
    BehaviorEvidencePrecedence.DATA_INVARIANT: 4,
    BehaviorEvidencePrecedence.MIGRATION_MANIFEST: 5,
    BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP: 6,
    BehaviorEvidencePrecedence.HISTORY: 7,
    BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS: 8,
}

# Acceptance-policy groups (documentation / alias resolution).
PRECEDENCE_GROUPS: Final[tuple[tuple[str, tuple[BehaviorEvidencePrecedence, ...]], ...]] = (
    (
        "reviewed_idl_schema_public_stub",
        (BehaviorEvidencePrecedence.REVIEWED_IDL,),
    ),
    (
        "normative_spec_conformance_test",
        (BehaviorEvidencePrecedence.NORMATIVE_SPEC,),
    ),
    (
        "caller_postcondition_callee_precondition",
        (
            BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
            BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
        ),
    ),
    (
        "data_invariant_migration_architecture_history",
        (
            BehaviorEvidencePrecedence.DATA_INVARIANT,
            BehaviorEvidencePrecedence.MIGRATION_MANIFEST,
            BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
            BehaviorEvidencePrecedence.HISTORY,
        ),
    ),
    (
        "non_authoritative_observation",
        (BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,),
    ),
)

_PRECEDENCE_ALIASES: Final[Mapping[str, BehaviorEvidencePrecedence]] = {
    "reviewed_idl": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "idl": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "schema": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "public_stub": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "stub": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "reviewed_schema": BehaviorEvidencePrecedence.REVIEWED_IDL,
    "normative_spec": BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    "spec": BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    "conformance_test": BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    "conformance": BehaviorEvidencePrecedence.NORMATIVE_SPEC,
    "caller_postcondition": BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
    "caller": BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
    "postcondition": BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
    "callee_precondition": BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
    "callee": BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
    "precondition": BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
    "data_invariant": BehaviorEvidencePrecedence.DATA_INVARIANT,
    "invariant": BehaviorEvidencePrecedence.DATA_INVARIANT,
    "migration_manifest": BehaviorEvidencePrecedence.MIGRATION_MANIFEST,
    "migration": BehaviorEvidencePrecedence.MIGRATION_MANIFEST,
    "architecture_ownership": BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
    "ownership": BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
    "architecture": BehaviorEvidencePrecedence.ARCHITECTURE_OWNERSHIP,
    "history": BehaviorEvidencePrecedence.HISTORY,
    "lineage": BehaviorEvidencePrecedence.HISTORY,
    "implementation_hypothesis": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    "implementation": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    "observation": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    "hypothesis": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    "llm": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
    "candidate": BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
}

# Map clause families onto RequiredBehaviorContract@1 reference buckets.
_STRUCTURAL_FAMILIES: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.FIELDS,
        BehaviorClauseFamily.VARIANTS,
        BehaviorClauseFamily.GENERICS,
        BehaviorClauseFamily.INVARIANTS,
        BehaviorClauseFamily.DEFAULTS,
        BehaviorClauseFamily.CONSTRUCTORS,
        BehaviorClauseFamily.FACTORIES,
        BehaviorClauseFamily.TOTALITY,
        BehaviorClauseFamily.METHODS,
        BehaviorClauseFamily.STATE_MACHINE,
        BehaviorClauseFamily.TRANSITIONS,
        BehaviorClauseFamily.IDEMPOTENCE,
    }
)

_FIELD_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.FIELDS,
        BehaviorClauseFamily.VARIANTS,
        BehaviorClauseFamily.GENERICS,
        BehaviorClauseFamily.DEFAULTS,
        BehaviorClauseFamily.SERIALIZATION,
        BehaviorClauseFamily.PERSISTENCE,
        BehaviorClauseFamily.VERSIONING,
        BehaviorClauseFamily.EQUALITY,
        BehaviorClauseFamily.HASH,
    }
)
_CONSTRUCTOR_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.CONSTRUCTORS,
        BehaviorClauseFamily.FACTORIES,
        BehaviorClauseFamily.TOTALITY,
    }
)
_METHOD_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.METHODS,
        BehaviorClauseFamily.IDEMPOTENCE,
        BehaviorClauseFamily.COMPATIBILITY,
        BehaviorClauseFamily.TESTS,
        BehaviorClauseFamily.TELEMETRY,
        BehaviorClauseFamily.MIGRATIONS,
    }
)
_INVARIANT_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {BehaviorClauseFamily.INVARIANTS}
)
_TRANSITION_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.STATE_MACHINE,
        BehaviorClauseFamily.TRANSITIONS,
    }
)
_EFFECT_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.EFFECTS,
        BehaviorClauseFamily.CANCELLATION,
        BehaviorClauseFamily.DEGRADATION,
        BehaviorClauseFamily.ERRORS,
    }
)
_CAPABILITY_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.CAPABILITIES,
        BehaviorClauseFamily.TRUST,
        BehaviorClauseFamily.PRIVACY,
    }
)
_AUTHORIZATION_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {BehaviorClauseFamily.AUTHORIZATION}
)
_RESOURCE_BUCKET: Final[frozenset[BehaviorClauseFamily]] = frozenset(
    {
        BehaviorClauseFamily.RESOURCES,
        BehaviorClauseFamily.OWNERSHIP,
        BehaviorClauseFamily.LIFETIME,
        BehaviorClauseFamily.MUTATION,
        BehaviorClauseFamily.CONCURRENCY,
        BehaviorClauseFamily.CACHE,
        BehaviorClauseFamily.DISPOSAL,
    }
)

# Minimum structural families required to admit each BehaviorKind.
_KIND_REQUIRED_FAMILIES: Final[
    Mapping[BehaviorKind, frozenset[BehaviorClauseFamily]]
] = {
    BehaviorKind.CLASS: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.INVARIANTS,
        }
    ),
    BehaviorKind.METHOD: frozenset({BehaviorClauseFamily.METHODS}),
    BehaviorKind.DATA_STRUCTURE: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.INVARIANTS,
            BehaviorClauseFamily.VARIANTS,
        }
    ),
    BehaviorKind.FACTORY: frozenset(
        {
            BehaviorClauseFamily.FACTORIES,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.TOTALITY,
        }
    ),
    BehaviorKind.SCHEMA: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.INVARIANTS,
            BehaviorClauseFamily.DEFAULTS,
            BehaviorClauseFamily.SERIALIZATION,
        }
    ),
    BehaviorKind.STATE_TRANSITION: frozenset(
        {
            BehaviorClauseFamily.STATE_MACHINE,
            BehaviorClauseFamily.TRANSITIONS,
        }
    ),
    BehaviorKind.PROVIDER: frozenset(
        {
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.FACTORIES,
        }
    ),
    BehaviorKind.ADAPTER: frozenset(
        {
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.COMPATIBILITY,
        }
    ),
    BehaviorKind.SERIALIZER: frozenset(
        {
            BehaviorClauseFamily.SERIALIZATION,
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.FIELDS,
        }
    ),
}

# For kinds that accept any-of a set, at least one must be present.
_KIND_ANY_OF: Final[Mapping[BehaviorKind, frozenset[BehaviorClauseFamily]]] = {
    BehaviorKind.CLASS: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.INVARIANTS,
        }
    ),
    BehaviorKind.METHOD: frozenset({BehaviorClauseFamily.METHODS}),
    BehaviorKind.DATA_STRUCTURE: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.INVARIANTS,
            BehaviorClauseFamily.VARIANTS,
        }
    ),
    BehaviorKind.FACTORY: frozenset(
        {
            BehaviorClauseFamily.FACTORIES,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.TOTALITY,
        }
    ),
    BehaviorKind.SCHEMA: frozenset(
        {
            BehaviorClauseFamily.FIELDS,
            BehaviorClauseFamily.INVARIANTS,
            BehaviorClauseFamily.DEFAULTS,
        }
    ),
    BehaviorKind.STATE_TRANSITION: frozenset(
        {
            BehaviorClauseFamily.STATE_MACHINE,
            BehaviorClauseFamily.TRANSITIONS,
        }
    ),
    BehaviorKind.PROVIDER: frozenset(
        {
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.CONSTRUCTORS,
            BehaviorClauseFamily.FACTORIES,
        }
    ),
    BehaviorKind.ADAPTER: frozenset(
        {
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.COMPATIBILITY,
        }
    ),
    BehaviorKind.SERIALIZER: frozenset(
        {
            BehaviorClauseFamily.SERIALIZATION,
            BehaviorClauseFamily.METHODS,
            BehaviorClauseFamily.FIELDS,
        }
    ),
}

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_body",
        "source_text",
        "source_code",
        "body",
        "content",
        "contents",
        "text",
        "code",
        "raw",
        "raw_text",
        "ast",
        "embedding",
        "model_output",
        "completion",
        "prompt",
        "snippet",
    }
)
_SECRET_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "password",
        "api_key",
        "access_token",
        "refresh_token",
        "private_key",
        "credential",
        "session_token",
        "private_witness",
        "private_premise",
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise RequiredBehaviorSynthesisError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise RequiredBehaviorSynthesisError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise RequiredBehaviorSynthesisError(f"{name} must not be empty")
    if len(text.encode("utf-8")) > limit:
        raise RequiredBehaviorSynthesisBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name)
    if any(ch.isspace() for ch in text):
        raise RequiredBehaviorSynthesisError(f"{name} must be a compact identifier")
    if len(text.encode("utf-8")) > MAX_REF_BYTES:
        raise RequiredBehaviorSynthesisBoundsError(f"{name} exceeds its byte bound")
    return text


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise RequiredBehaviorSynthesisError(
            f"{name} must be one of: {choices}"
        ) from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RequiredBehaviorSynthesisError(f"{name} must be a boolean")
    return value


def _ids(values: Iterable[Any], name: str, *, limit: int = MAX_CLAUSE_BINDINGS) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values or ():
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            result.append(text)
    if len(result) > limit:
        raise RequiredBehaviorSynthesisBoundsError(f"{name} exceeds its item bound")
    return tuple(sorted(result))


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, float):
        raise RequiredBehaviorSynthesisError(f"{name} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RequiredBehaviorSynthesisError(f"{name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or normalized in _SECRET_MARKERS:
                raise RequiredBehaviorSynthesisError(
                    f"{name} may not contain source bodies or secrets"
                )
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, name)
    elif isinstance(value, (bytes, bytearray)):
        raise RequiredBehaviorSynthesisError(f"{name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(_canonical_json(payload).encode("utf-8")) > MAX_RECORD_BYTES:
        raise RequiredBehaviorSynthesisBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise RequiredBehaviorSynthesisAuthorityError(
                "stored content identity does not match the canonical record"
            )


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return PropagationAuthorityRoots.from_dict(value)
    raise RequiredBehaviorSynthesisError("roots must be PropagationAuthorityRoots")


def _roots_match(left: PropagationAuthorityRoots, right: PropagationAuthorityRoots) -> bool:
    return left.content_id == right.content_id


def _value_digest(value_ref: str, statement_ref: str = "") -> str:
    """Digest the *semantic claim* for conflict detection.

    ``statement_ref`` is provenance metadata and must not create false
    conflicts when two atoms assert the same ``value_ref`` from different
    sources (e.g. a requirement clause and a reviewed schema clause).
    """
    del statement_ref  # retained for call-site compatibility / audit only
    payload = {"value_ref": value_ref}
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"digest:sha256:{digest}"


def coerce_precedence(value: Any) -> BehaviorEvidencePrecedence:
    """Normalize precedence enums, values, and acceptance aliases."""
    if isinstance(value, BehaviorEvidencePrecedence):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _PRECEDENCE_ALIASES:
            return _PRECEDENCE_ALIASES[key]
        try:
            return BehaviorEvidencePrecedence(key)
        except ValueError as exc:
            raise RequiredBehaviorSynthesisError(
                f"unknown evidence precedence: {value!r}"
            ) from exc
    raise RequiredBehaviorSynthesisError("evidence precedence must be a string or enum")


def coerce_clause_family(value: Any) -> BehaviorClauseFamily:
    if isinstance(value, BehaviorClauseFamily):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return BehaviorClauseFamily(key)
        except ValueError as exc:
            raise RequiredBehaviorSynthesisError(
                f"unknown behavior clause family: {value!r}"
            ) from exc
    raise RequiredBehaviorSynthesisError("clause family must be a string or enum")


def precedence_rank(precedence: BehaviorEvidencePrecedence) -> int:
    try:
        return PRECEDENCE_RANK[precedence]
    except KeyError as exc:
        raise RequiredBehaviorSynthesisError(
            f"precedence has no rank: {precedence!r}"
        ) from exc


def is_authoritative(precedence: BehaviorEvidencePrecedence) -> bool:
    return precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS


def all_clause_families() -> tuple[BehaviorClauseFamily, ...]:
    return tuple(BehaviorClauseFamily)


def all_precedence_levels() -> tuple[BehaviorEvidencePrecedence, ...]:
    return tuple(
        sorted(PRECEDENCE_RANK.keys(), key=lambda item: PRECEDENCE_RANK[item])
    )


# ---------------------------------------------------------------------------
# Evidence atoms and bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorEvidenceAtom(CanonicalContract):
    """One independently sourced statement about a single clause family.

    Atoms are nomination-quality evidence until the synthesizer admits a
    contract.  Implementation hypotheses set ``authoritative=False``.
    """

    SCHEMA: ClassVar[str] = BEHAVIOR_EVIDENCE_ATOM_SCHEMA

    roots: PropagationAuthorityRoots
    evidence_id: str
    precedence: BehaviorEvidencePrecedence
    family: BehaviorClauseFamily
    clause_ref: str
    value_ref: str
    subject_symbol_id: str
    statement_ref: str = ""
    assumption: bool = False
    unsupported: bool = False
    authoritative: bool = True
    proof_ref: str = ""
    source_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "precedence",
            coerce_precedence(self.precedence),
        )
        object.__setattr__(
            self, "family", coerce_clause_family(self.family)
        )
        object.__setattr__(
            self, "clause_ref", _identifier(self.clause_ref, "clause_ref")
        )
        object.__setattr__(
            self, "value_ref", _identifier(self.value_ref, "value_ref")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self,
            "statement_ref",
            _text(self.statement_ref, "statement_ref", required=False),
        )
        object.__setattr__(self, "assumption", _bool(self.assumption, "assumption"))
        object.__setattr__(self, "unsupported", _bool(self.unsupported, "unsupported"))
        object.__setattr__(
            self, "authoritative", _bool(self.authoritative, "authoritative")
        )
        object.__setattr__(
            self, "proof_ref", _text(self.proof_ref, "proof_ref", required=False)
        )
        object.__setattr__(
            self, "source_path", _text(self.source_path, "source_path", required=False)
        )
        if (
            self.precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and self.authoritative
        ):
            raise RequiredBehaviorSynthesisAuthorityError(
                "implementation hypotheses cannot be marked authoritative"
            )
        if (
            self.precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and not self.authoritative
            and not self.assumption
            and not self.unsupported
        ):
            # Non-hypothesis evidence is authoritative unless explicitly marked
            # as assumption or unsupported observation.
            object.__setattr__(self, "authoritative", True)
        if self.unsupported and self.authoritative:
            raise RequiredBehaviorSynthesisAuthorityError(
                "unsupported clauses cannot claim authoritative status"
            )
        if self.proof_ref and self.precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS:
            raise RequiredBehaviorSynthesisAuthorityError(
                "implementation hypotheses cannot carry proof refs"
            )
        _bounded(self, "behavior evidence atom")

    @property
    def value_digest(self) -> str:
        return _value_digest(self.value_ref, self.statement_ref)

    @property
    def rank(self) -> int:
        return precedence_rank(self.precedence)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "evidence_id": self.evidence_id,
            "precedence": self.precedence.value,
            "family": self.family.value,
            "clause_ref": self.clause_ref,
            "value_ref": self.value_ref,
            "subject_symbol_id": self.subject_symbol_id,
            "statement_ref": self.statement_ref,
            "assumption": self.assumption,
            "unsupported": self.unsupported,
            "authoritative": self.authoritative,
            "proof_ref": self.proof_ref,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorEvidenceAtom":
        if not isinstance(payload, Mapping):
            raise RequiredBehaviorSynthesisError("evidence atom payload must be a mapping")
        if payload.get("schema") not in (None, cls.SCHEMA):
            raise RequiredBehaviorSynthesisError("evidence atom has an unsupported schema")
        fields = (
            "roots",
            "evidence_id",
            "precedence",
            "family",
            "clause_ref",
            "value_ref",
            "subject_symbol_id",
            "statement_ref",
            "assumption",
            "unsupported",
            "authoritative",
            "proof_ref",
            "source_path",
        )
        values = {name: payload[name] for name in fields if name in payload}
        values["roots"] = _roots(values["roots"])
        atom = cls(**values)
        _verify_identity(payload, atom)
        return atom

    @classmethod
    def from_mapping(
        cls,
        roots: PropagationAuthorityRoots,
        payload: Mapping[str, Any],
        *,
        default_subject: str = "",
    ) -> "BehaviorEvidenceAtom":
        """Build an atom from a compact recipe mapping (fixture-friendly)."""
        if not isinstance(payload, Mapping):
            raise RequiredBehaviorSynthesisError("evidence mapping must be a mapping")
        _assert_body_free(payload, "evidence mapping")
        subject = str(
            payload.get("subject_symbol_id")
            or payload.get("subject")
            or default_subject
            or ""
        ).strip()
        evidence_id = str(
            payload.get("evidence_id") or payload.get("id") or ""
        ).strip()
        if not evidence_id:
            family = coerce_clause_family(payload.get("family") or payload.get("clause_family"))
            clause_ref = str(payload.get("clause_ref") or payload.get("clause") or family.value)
            evidence_id = f"evidence:{family.value}:{clause_ref}"
        precedence_raw = (
            payload.get("precedence")
            or payload.get("source")
            or payload.get("rank")
            or BehaviorEvidencePrecedence.REVIEWED_IDL
        )
        precedence = coerce_precedence(precedence_raw)
        family = coerce_clause_family(
            payload.get("family") or payload.get("clause_family") or "fields"
        )
        clause_ref = str(
            payload.get("clause_ref") or payload.get("clause") or f"clause:{family.value}"
        ).strip()
        value_ref = str(
            payload.get("value_ref")
            or payload.get("value")
            or payload.get("statement")
            or clause_ref
        ).strip()
        return cls(
            roots=roots,
            evidence_id=evidence_id,
            precedence=precedence,
            family=family,
            clause_ref=clause_ref,
            value_ref=value_ref,
            subject_symbol_id=subject,
            statement_ref=str(payload.get("statement_ref") or ""),
            assumption=bool(payload.get("assumption", False)),
            unsupported=bool(payload.get("unsupported", False)),
            authoritative=bool(
                payload.get(
                    "authoritative",
                    precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
                )
            ),
            proof_ref=str(payload.get("proof_ref") or ""),
            source_path=str(payload.get("source_path") or payload.get("path") or ""),
        )


@dataclass(frozen=True)
class BehaviorClauseBinding(CanonicalContract):
    """Winning (or gap-contributing) binding of one clause family."""

    SCHEMA: ClassVar[str] = BEHAVIOR_CLAUSE_BINDING_SCHEMA

    family: BehaviorClauseFamily
    clause_ref: str
    evidence_id: str
    precedence: BehaviorEvidencePrecedence
    value_ref: str
    value_digest: str
    assumption: bool = False
    unsupported: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", coerce_clause_family(self.family))
        object.__setattr__(
            self, "clause_ref", _identifier(self.clause_ref, "clause_ref")
        )
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self, "precedence", coerce_precedence(self.precedence)
        )
        object.__setattr__(
            self, "value_ref", _identifier(self.value_ref, "value_ref")
        )
        object.__setattr__(
            self, "value_digest", _identifier(self.value_digest, "value_digest")
        )
        object.__setattr__(self, "assumption", _bool(self.assumption, "assumption"))
        object.__setattr__(self, "unsupported", _bool(self.unsupported, "unsupported"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "family": self.family.value,
            "clause_ref": self.clause_ref,
            "evidence_id": self.evidence_id,
            "precedence": self.precedence.value,
            "value_ref": self.value_ref,
            "value_digest": self.value_digest,
            "assumption": self.assumption,
            "unsupported": self.unsupported,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorClauseBinding":
        if not isinstance(payload, Mapping):
            raise RequiredBehaviorSynthesisError("clause binding payload must be a mapping")
        fields = (
            "family",
            "clause_ref",
            "evidence_id",
            "precedence",
            "value_ref",
            "value_digest",
            "assumption",
            "unsupported",
        )
        values = {name: payload[name] for name in fields if name in payload}
        return cls(**values)

    @classmethod
    def from_atom(cls, atom: BehaviorEvidenceAtom) -> "BehaviorClauseBinding":
        return cls(
            family=atom.family,
            clause_ref=atom.clause_ref,
            evidence_id=atom.evidence_id,
            precedence=atom.precedence,
            value_ref=atom.value_ref,
            value_digest=atom.value_digest,
            assumption=atom.assumption,
            unsupported=atom.unsupported,
        )


# ---------------------------------------------------------------------------
# Behavior gap and synthesis receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorGap(CanonicalContract):
    """Typed refusal to admit required behavior; never requests implementation."""

    SCHEMA: ClassVar[str] = BEHAVIOR_GAP_SCHEMA

    roots: PropagationAuthorityRoots
    gap_id: str
    kind: BehaviorGapKind
    subject_symbol_id: str
    requirement_id: str
    reason: str
    missing_families: tuple[str, ...] = ()
    conflicting_evidence_ids: tuple[str, ...] = ()
    unsupported_clauses: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    implementation_request: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "gap_id", _identifier(self.gap_id, "gap_id"))
        object.__setattr__(self, "kind", _enum(self.kind, BehaviorGapKind, "kind"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(
            self, "requirement_id", _identifier(self.requirement_id, "requirement_id")
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "missing_families", _ids(self.missing_families, "missing_families")
        )
        object.__setattr__(
            self,
            "conflicting_evidence_ids",
            _ids(self.conflicting_evidence_ids, "conflicting_evidence_ids"),
        )
        object.__setattr__(
            self,
            "unsupported_clauses",
            _ids(self.unsupported_clauses, "unsupported_clauses"),
        )
        object.__setattr__(self, "assumptions", _ids(self.assumptions, "assumptions"))
        object.__setattr__(self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids"))
        object.__setattr__(
            self,
            "implementation_request",
            _bool(self.implementation_request, "implementation_request"),
        )
        # Gaps never authorize implementation work.
        if self.implementation_request:
            raise RequiredBehaviorSynthesisAuthorityError(
                "behavior gaps cannot request implementation"
            )
        _bounded(self, "behavior gap")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "gap_id": self.gap_id,
            "kind": self.kind.value,
            "subject_symbol_id": self.subject_symbol_id,
            "requirement_id": self.requirement_id,
            "reason": self.reason,
            "missing_families": list(self.missing_families),
            "conflicting_evidence_ids": list(self.conflicting_evidence_ids),
            "unsupported_clauses": list(self.unsupported_clauses),
            "assumptions": list(self.assumptions),
            "evidence_ids": list(self.evidence_ids),
            "implementation_request": self.implementation_request,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorGap":
        if not isinstance(payload, Mapping) or payload.get("schema") not in (
            None,
            cls.SCHEMA,
        ):
            raise RequiredBehaviorSynthesisError("behavior gap has an unsupported schema")
        fields = (
            "roots",
            "gap_id",
            "kind",
            "subject_symbol_id",
            "requirement_id",
            "reason",
            "missing_families",
            "conflicting_evidence_ids",
            "unsupported_clauses",
            "assumptions",
            "evidence_ids",
            "implementation_request",
        )
        values = {name: payload[name] for name in fields if name in payload}
        values["roots"] = _roots(values["roots"])
        gap = cls(**values)
        _verify_identity(payload, gap)
        return gap


@dataclass(frozen=True)
class RequiredBehaviorSynthesisReceipt(CanonicalContract):
    """Complete, content-addressed outcome of required-behavior synthesis."""

    SCHEMA: ClassVar[str] = REQUIRED_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    receipt_id: str
    requirement_id: str
    subject_symbol_id: str
    kind: BehaviorKind
    disposition: SynthesisDisposition
    evidence_precedence: BehaviorEvidencePrecedence
    clause_bindings: tuple[BehaviorClauseBinding, ...]
    assumptions: tuple[str, ...] = ()
    unsupported_clauses: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    implementation_request: bool = False
    contract: RequiredBehaviorContract | None = None
    gap: BehaviorGap | None = None
    contract_delta_id: str = ""
    value_provenance_id: str = ""
    memory_facet_ref: str = ""
    program_contract_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "requirement_id", _identifier(self.requirement_id, "requirement_id")
        )
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(self, "kind", _enum(self.kind, BehaviorKind, "kind"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SynthesisDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "evidence_precedence",
            coerce_precedence(self.evidence_precedence),
        )
        bindings = tuple(self.clause_bindings or ())
        if not all(isinstance(item, BehaviorClauseBinding) for item in bindings):
            raise RequiredBehaviorSynthesisError(
                "clause_bindings must contain BehaviorClauseBinding values"
            )
        if len(bindings) > MAX_CLAUSE_BINDINGS:
            raise RequiredBehaviorSynthesisBoundsError(
                "clause_bindings exceeds its item bound"
            )
        # Deterministic order by family then evidence id.
        bindings = tuple(
            sorted(bindings, key=lambda item: (item.family.value, item.evidence_id))
        )
        object.__setattr__(self, "clause_bindings", bindings)
        object.__setattr__(self, "assumptions", _ids(self.assumptions, "assumptions"))
        object.__setattr__(
            self,
            "unsupported_clauses",
            _ids(self.unsupported_clauses, "unsupported_clauses"),
        )
        object.__setattr__(self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "implementation_request",
            _bool(self.implementation_request, "implementation_request"),
        )
        object.__setattr__(
            self,
            "contract_delta_id",
            _text(self.contract_delta_id, "contract_delta_id", required=False),
        )
        object.__setattr__(
            self,
            "value_provenance_id",
            _text(self.value_provenance_id, "value_provenance_id", required=False),
        )
        object.__setattr__(
            self,
            "memory_facet_ref",
            _text(self.memory_facet_ref, "memory_facet_ref", required=False),
        )
        object.__setattr__(
            self,
            "program_contract_ref",
            _text(self.program_contract_ref, "program_contract_ref", required=False),
        )
        if self.contract is not None and not isinstance(
            self.contract, RequiredBehaviorContract
        ):
            raise RequiredBehaviorSynthesisError(
                "contract must be RequiredBehaviorContract or None"
            )
        if self.gap is not None and not isinstance(self.gap, BehaviorGap):
            raise RequiredBehaviorSynthesisError("gap must be BehaviorGap or None")

        if self.disposition is SynthesisDisposition.ADMITTED:
            if self.contract is None:
                raise RequiredBehaviorSynthesisError(
                    "admitted synthesis requires a RequiredBehaviorContract"
                )
            if self.gap is not None:
                raise RequiredBehaviorSynthesisError(
                    "admitted synthesis cannot carry a behavior gap"
                )
            if not _roots_match(self.roots, self.contract.roots):
                raise RequiredBehaviorSynthesisAuthorityError(
                    "contract roots must match synthesis roots"
                )
            if self.contract.subject_symbol_id != self.subject_symbol_id:
                raise RequiredBehaviorSynthesisAuthorityError(
                    "contract subject must match synthesis subject"
                )
            if self.contract.implementation_hypothesis:
                raise RequiredBehaviorSynthesisAuthorityError(
                    "admitted contracts cannot be implementation hypotheses"
                )
        elif self.disposition is SynthesisDisposition.BEHAVIOR_GAP:
            if self.gap is None:
                raise RequiredBehaviorSynthesisError(
                    "behavior_gap disposition requires a BehaviorGap"
                )
            if self.contract is not None:
                raise RequiredBehaviorSynthesisError(
                    "behavior gaps cannot carry an admitted contract"
                )
            if self.implementation_request:
                raise RequiredBehaviorSynthesisAuthorityError(
                    "behavior gaps cannot request implementation"
                )
            if not _roots_match(self.roots, self.gap.roots):
                raise RequiredBehaviorSynthesisAuthorityError(
                    "gap roots must match synthesis roots"
                )
        elif self.disposition is SynthesisDisposition.ABSTAINED:
            if self.contract is not None:
                raise RequiredBehaviorSynthesisError(
                    "abstained synthesis cannot carry a contract"
                )
            if self.implementation_request:
                raise RequiredBehaviorSynthesisAuthorityError(
                    "abstained synthesis cannot request implementation"
                )

        if (
            self.evidence_precedence
            is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            and self.disposition is SynthesisDisposition.ADMITTED
        ):
            raise RequiredBehaviorSynthesisAuthorityError(
                "implementation hypotheses cannot admit required behavior"
            )
        _bounded(self, "required behavior synthesis receipt")

    @property
    def admitted(self) -> bool:
        return self.disposition is SynthesisDisposition.ADMITTED

    @property
    def has_gap(self) -> bool:
        return self.disposition is SynthesisDisposition.BEHAVIOR_GAP

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "requirement_id": self.requirement_id,
            "subject_symbol_id": self.subject_symbol_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "evidence_precedence": self.evidence_precedence.value,
            "clause_bindings": [item.to_dict() for item in self.clause_bindings],
            "assumptions": list(self.assumptions),
            "unsupported_clauses": list(self.unsupported_clauses),
            "evidence_ids": list(self.evidence_ids),
            "proof_refs": list(self.proof_refs),
            "producer_id": self.producer_id,
            "implementation_request": self.implementation_request,
            "contract": None if self.contract is None else self.contract.to_dict(),
            "gap": None if self.gap is None else self.gap.to_dict(),
            "contract_delta_id": self.contract_delta_id,
            "value_provenance_id": self.value_provenance_id,
            "memory_facet_ref": self.memory_facet_ref,
            "program_contract_ref": self.program_contract_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiredBehaviorSynthesisReceipt":
        if not isinstance(payload, Mapping) or payload.get("schema") not in (
            None,
            cls.SCHEMA,
        ):
            raise RequiredBehaviorSynthesisError(
                "synthesis receipt has an unsupported schema"
            )
        roots = _roots(payload["roots"])
        bindings = tuple(
            BehaviorClauseBinding.from_dict(item)
            for item in (payload.get("clause_bindings") or ())
        )
        contract_payload = payload.get("contract")
        contract = (
            None
            if contract_payload in (None, {})
            else RequiredBehaviorContract.from_dict(contract_payload)
        )
        gap_payload = payload.get("gap")
        gap = None if gap_payload in (None, {}) else BehaviorGap.from_dict(gap_payload)
        receipt = cls(
            roots=roots,
            receipt_id=payload["receipt_id"],
            requirement_id=payload["requirement_id"],
            subject_symbol_id=payload["subject_symbol_id"],
            kind=payload["kind"],
            disposition=payload["disposition"],
            evidence_precedence=payload["evidence_precedence"],
            clause_bindings=bindings,
            assumptions=tuple(payload.get("assumptions") or ()),
            unsupported_clauses=tuple(payload.get("unsupported_clauses") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            proof_refs=tuple(payload.get("proof_refs") or ()),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
            implementation_request=bool(payload.get("implementation_request", False)),
            contract=contract,
            gap=gap,
            contract_delta_id=str(payload.get("contract_delta_id") or ""),
            value_provenance_id=str(payload.get("value_provenance_id") or ""),
            memory_facet_ref=str(payload.get("memory_facet_ref") or ""),
            program_contract_ref=str(payload.get("program_contract_ref") or ""),
        )
        _verify_identity(payload, receipt)
        return receipt


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


def _normalize_atoms(
    roots: PropagationAuthorityRoots,
    evidence: Sequence[BehaviorEvidenceAtom | Mapping[str, Any]],
    *,
    subject_symbol_id: str,
) -> tuple[BehaviorEvidenceAtom, ...]:
    if len(evidence) > MAX_EVIDENCE_ATOMS:
        raise RequiredBehaviorSynthesisBoundsError(
            "evidence set exceeds MAX_EVIDENCE_ATOMS"
        )
    atoms: list[BehaviorEvidenceAtom] = []
    seen_ids: set[str] = set()
    for item in evidence:
        if isinstance(item, BehaviorEvidenceAtom):
            atom = item
        elif isinstance(item, Mapping):
            atom = BehaviorEvidenceAtom.from_mapping(
                roots, item, default_subject=subject_symbol_id
            )
        else:
            raise RequiredBehaviorSynthesisError(
                "evidence items must be BehaviorEvidenceAtom or mappings"
            )
        if not _roots_match(atom.roots, roots):
            raise RequiredBehaviorSynthesisAuthorityError(
                "evidence atom roots must match synthesizer roots"
            )
        if atom.subject_symbol_id != subject_symbol_id:
            raise RequiredBehaviorSynthesisAuthorityError(
                "evidence atom subject_symbol_id must match synthesis subject"
            )
        if atom.evidence_id in seen_ids:
            # Deterministic dedupe by id: keep first occurrence.
            continue
        seen_ids.add(atom.evidence_id)
        atoms.append(atom)
    atoms.sort(key=lambda item: (item.family.value, item.rank, item.evidence_id))
    return tuple(atoms)


def _atoms_from_requirement(
    roots: PropagationAuthorityRoots,
    requirement: MissingInputRequirement,
    subject_symbol_id: str,
) -> tuple[BehaviorEvidenceAtom, ...]:
    """Lift requirement-local clauses into non-structural evidence atoms.

    These are callee-precondition strength facts already bound to the missing
    input; they never invent structural type shape alone.
    """
    atoms: list[BehaviorEvidenceAtom] = []
    for family, refs in (
        (BehaviorClauseFamily.CAPABILITIES, requirement.capability_refs),
        (BehaviorClauseFamily.AUTHORIZATION, requirement.authorization_refs),
        (BehaviorClauseFamily.RESOURCES, requirement.resource_refs),
        (BehaviorClauseFamily.EFFECTS, requirement.effect_refs),
        (BehaviorClauseFamily.ERRORS, requirement.allowed_error_refs),
        (BehaviorClauseFamily.OWNERSHIP, requirement.ownership_refs),
        (
            BehaviorClauseFamily.CONSTRUCTORS,
            requirement.construction_precondition_refs,
        ),
        (
            BehaviorClauseFamily.INVARIANTS,
            requirement.result_postcondition_refs,
        ),
    ):
        for index, ref in enumerate(refs):
            atoms.append(
                BehaviorEvidenceAtom(
                    roots=roots,
                    evidence_id=f"evidence:requirement:{family.value}:{index}:{ref}",
                    precedence=BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
                    family=family,
                    clause_ref=ref,
                    value_ref=ref,
                    subject_symbol_id=subject_symbol_id,
                    statement_ref=f"missing-input:{requirement.requirement_id}",
                    authoritative=True,
                    proof_ref=requirement.proof_refs[0] if requirement.proof_refs else "",
                )
            )
    # The missing type *name* is recorded as an assumption, not as a field
    # definition.  Admitting a new support type still requires independent
    # structural evidence (fields, constructors, methods, …).
    atoms.append(
        BehaviorEvidenceAtom(
            roots=roots,
            evidence_id=f"evidence:requirement:type:{requirement.type_ref}",
            precedence=BehaviorEvidencePrecedence.CALLEE_PRECONDITION,
            family=BehaviorClauseFamily.FIELDS,
            clause_ref=f"type:{requirement.type_ref}",
            value_ref=requirement.type_ref,
            subject_symbol_id=subject_symbol_id,
            statement_ref=requirement.information_content_ref,
            assumption=True,
            authoritative=False,
        )
    )
    return tuple(atoms)


def _atoms_from_contract_delta(
    roots: PropagationAuthorityRoots,
    delta: ProgramContractDelta,
    subject_symbol_id: str,
) -> tuple[BehaviorEvidenceAtom, ...]:
    if not _roots_match(delta.roots, roots):
        raise RequiredBehaviorSynthesisAuthorityError(
            "program contract delta roots must match synthesizer roots"
        )
    atoms: list[BehaviorEvidenceAtom] = []
    for clause in delta.clauses:
        # Breaking/added parameter clauses contribute caller postconditions.
        family = BehaviorClauseFamily.METHODS
        kind_name = clause.kind.value if hasattr(clause.kind, "value") else str(clause.kind)
        if "parameter" in kind_name or "input" in kind_name:
            family = BehaviorClauseFamily.FIELDS
        elif "return" in kind_name or "output" in kind_name:
            family = BehaviorClauseFamily.METHODS
        elif "error" in kind_name:
            family = BehaviorClauseFamily.ERRORS
        elif "effect" in kind_name:
            family = BehaviorClauseFamily.EFFECTS
        elif "capability" in kind_name:
            family = BehaviorClauseFamily.CAPABILITIES
        elif "auth" in kind_name:
            family = BehaviorClauseFamily.AUTHORIZATION
        elif "state" in kind_name or "lifecycle" in kind_name:
            family = BehaviorClauseFamily.STATE_MACHINE
        atoms.append(
            BehaviorEvidenceAtom(
                roots=roots,
                evidence_id=f"evidence:delta:{clause.clause_id}",
                precedence=BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
                family=family,
                clause_ref=clause.clause_id,
                value_ref=clause.after_contract_ref or clause.clause_id,
                subject_symbol_id=subject_symbol_id,
                statement_ref=clause.reason or clause.before_contract_ref,
                authoritative=True,
                proof_ref=delta.proof_refs[0] if delta.proof_refs else "",
            )
        )
    return tuple(atoms)


def _select_winners(
    atoms: Sequence[BehaviorEvidenceAtom],
) -> tuple[
    dict[BehaviorClauseFamily, BehaviorEvidenceAtom],
    list[tuple[BehaviorClauseFamily, tuple[BehaviorEvidenceAtom, ...]]],
    list[BehaviorEvidenceAtom],
    list[BehaviorEvidenceAtom],
]:
    """Select highest-precedence winners per family; report same-rank conflicts.

    Returns ``(winners, conflicts, assumptions, unsupported)``.
    """
    by_family: dict[BehaviorClauseFamily, list[BehaviorEvidenceAtom]] = {}
    assumptions: list[BehaviorEvidenceAtom] = []
    unsupported: list[BehaviorEvidenceAtom] = []
    for atom in atoms:
        if atom.unsupported:
            unsupported.append(atom)
            continue
        if atom.assumption:
            assumptions.append(atom)
            # Assumptions still participate as non-winning context only when no
            # stronger evidence exists; they never override authoritative facts.
        by_family.setdefault(atom.family, []).append(atom)

    winners: dict[BehaviorClauseFamily, BehaviorEvidenceAtom] = {}
    conflicts: list[tuple[BehaviorClauseFamily, tuple[BehaviorEvidenceAtom, ...]]] = []

    for family, group in by_family.items():
        # Prefer authoritative non-assumption atoms; assumptions fill only when
        # nothing stronger is present for documentation, not admission.
        ranked = sorted(group, key=lambda item: (item.rank, item.evidence_id))
        best_rank = ranked[0].rank
        top = [item for item in ranked if item.rank == best_rank]
        # Among top rank, prefer non-assumption authoritative.
        preferred = [
            item
            for item in top
            if not item.assumption
            and item.precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
        ]
        if not preferred:
            preferred = [
                item
                for item in top
                if item.precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            ]
        if not preferred:
            preferred = list(top)

        digests = {item.value_digest for item in preferred}
        if len(digests) > 1:
            conflicts.append((family, tuple(preferred)))
            continue
        # Stable winner: lowest evidence_id among preferred.
        winners[family] = min(preferred, key=lambda item: item.evidence_id)

    return winners, conflicts, assumptions, unsupported


def _bucket_refs(
    winners: Mapping[BehaviorClauseFamily, BehaviorEvidenceAtom],
) -> dict[str, tuple[str, ...]]:
    field_refs: list[str] = []
    constructor_refs: list[str] = []
    method_refs: list[str] = []
    invariant_refs: list[str] = []
    state_transition_refs: list[str] = []
    effect_refs: list[str] = []
    capability_refs: list[str] = []
    authorization_refs: list[str] = []
    resource_refs: list[str] = []
    proof_refs: list[str] = []

    for family, atom in winners.items():
        if atom.unsupported or atom.assumption:
            continue
        if atom.precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS:
            continue
        ref = atom.clause_ref
        if family in _FIELD_BUCKET:
            field_refs.append(ref)
        if family in _CONSTRUCTOR_BUCKET:
            constructor_refs.append(ref)
        if family in _METHOD_BUCKET:
            method_refs.append(ref)
        if family in _INVARIANT_BUCKET:
            invariant_refs.append(ref)
        if family in _TRANSITION_BUCKET:
            state_transition_refs.append(ref)
        if family in _EFFECT_BUCKET:
            effect_refs.append(ref)
        if family in _CAPABILITY_BUCKET:
            capability_refs.append(ref)
        if family in _AUTHORIZATION_BUCKET:
            authorization_refs.append(ref)
        if family in _RESOURCE_BUCKET:
            resource_refs.append(ref)
        if atom.proof_ref:
            proof_refs.append(atom.proof_ref)

    return {
        "field_refs": tuple(sorted(set(field_refs))),
        "constructor_refs": tuple(sorted(set(constructor_refs))),
        "method_refs": tuple(sorted(set(method_refs))),
        "invariant_refs": tuple(sorted(set(invariant_refs))),
        "state_transition_refs": tuple(sorted(set(state_transition_refs))),
        "effect_refs": tuple(sorted(set(effect_refs))),
        "capability_refs": tuple(sorted(set(capability_refs))),
        "authorization_refs": tuple(sorted(set(authorization_refs))),
        "resource_refs": tuple(sorted(set(resource_refs))),
        "proof_refs": tuple(sorted(set(proof_refs))),
    }


def _authoritative_structural_families(
    winners: Mapping[BehaviorClauseFamily, BehaviorEvidenceAtom],
) -> frozenset[BehaviorClauseFamily]:
    present: set[BehaviorClauseFamily] = set()
    for family, atom in winners.items():
        if family not in _STRUCTURAL_FAMILIES and family not in (
            BehaviorClauseFamily.SERIALIZATION,
            BehaviorClauseFamily.COMPATIBILITY,
        ):
            continue
        if (
            atom.precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
            or atom.assumption
            or atom.unsupported
            or not atom.authoritative
        ):
            continue
        present.add(family)
    return frozenset(present)


def _best_precedence(
    winners: Mapping[BehaviorClauseFamily, BehaviorEvidenceAtom],
) -> BehaviorEvidencePrecedence:
    if not winners:
        return BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    authoritative = [
        atom
        for atom in winners.values()
        if atom.precedence is not BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
        and not atom.assumption
        and not atom.unsupported
    ]
    if not authoritative:
        return BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
    return min(authoritative, key=lambda item: item.rank).precedence


def _receipt_id(
    roots: PropagationAuthorityRoots,
    requirement_id: str,
    subject_symbol_id: str,
    kind: BehaviorKind,
    disposition: SynthesisDisposition,
    evidence_ids: Sequence[str],
) -> str:
    payload = {
        "producer": PRODUCER_ID,
        "roots": roots.content_id,
        "requirement_id": requirement_id,
        "subject_symbol_id": subject_symbol_id,
        "kind": kind.value,
        "disposition": disposition.value,
        "evidence_ids": list(evidence_ids),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"behavior-synthesis:{digest}"


def _gap_id(
    roots: PropagationAuthorityRoots,
    requirement_id: str,
    subject_symbol_id: str,
    kind: BehaviorGapKind,
    evidence_ids: Sequence[str],
) -> str:
    payload = {
        "roots": roots.content_id,
        "requirement_id": requirement_id,
        "subject_symbol_id": subject_symbol_id,
        "kind": kind.value,
        "evidence_ids": list(evidence_ids),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"behavior-gap:{digest}"


def _optional_root_id(value: Any, attr_names: Sequence[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    for name in attr_names:
        attr = getattr(value, name, None)
        if isinstance(attr, str) and attr.strip():
            return attr.strip()
    content = getattr(value, "content_id", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _check_optional_roots(
    roots: PropagationAuthorityRoots,
    *,
    value_provenance: Any,
    program_contract: Any,
    memory_facet: Any,
) -> None:
    """Fail closed when optional interfaces carry mismatched tree/root identity."""
    for label, obj in (
        ("value_provenance", value_provenance),
        ("program_contract", program_contract),
        ("memory_facet", memory_facet),
    ):
        if obj is None:
            continue
        obj_roots = getattr(obj, "roots", None)
        if obj_roots is None:
            continue
        # PropagationAuthorityRoots: content_id match.
        if isinstance(obj_roots, PropagationAuthorityRoots):
            if not _roots_match(obj_roots, roots):
                raise RequiredBehaviorSynthesisAuthorityError(
                    f"{label} roots must match synthesizer roots"
                )
            continue
        # ProgramGraphRoots / AuthorityRoots style: compare tree_id when present.
        tree_id = getattr(obj_roots, "tree_id", None) or getattr(
            obj_roots, "candidate_tree_id", None
        )
        if isinstance(tree_id, str) and tree_id.strip():
            if tree_id.strip() not in {
                roots.base_tree_id,
                roots.candidate_tree_id,
            }:
                raise RequiredBehaviorSynthesisAuthorityError(
                    f"{label} tree_id is stale relative to propagation roots"
                )
        repository_id = getattr(obj_roots, "repository_id", None)
        if isinstance(repository_id, str) and repository_id.strip():
            if repository_id.strip() != roots.repository_id:
                raise RequiredBehaviorSynthesisAuthorityError(
                    f"{label} repository_id must match synthesizer roots"
                )


class RequiredBehaviorSynthesizer:
    """Synthesize :class:`RequiredBehaviorContract` under explicit precedence.

    The synthesizer never invents behavior from candidate implementations or
    model output.  Gaps are first-class and never authorize implementation.
    """

    def __init__(self, roots: PropagationAuthorityRoots) -> None:
        self.roots = _roots(roots)

    def synthesize(
        self,
        requirement: MissingInputRequirement,
        *,
        kind: BehaviorKind | str,
        subject_symbol_id: str = "",
        evidence: Sequence[BehaviorEvidenceAtom | Mapping[str, Any]] = (),
        contract_delta: ProgramContractDelta | None = None,
        value_provenance: Any = None,
        program_contract: Any = None,
        memory_facet: Any = None,
        include_requirement_atoms: bool = True,
        placement_decision_ref: str = "",
    ) -> RequiredBehaviorSynthesisReceipt:
        if not isinstance(requirement, MissingInputRequirement):
            raise RequiredBehaviorSynthesisError(
                "requirement must be a typed MissingInputRequirement"
            )
        if not _roots_match(requirement.roots, self.roots):
            raise RequiredBehaviorSynthesisAuthorityError(
                "requirement roots must match synthesizer roots"
            )
        behavior_kind = _enum(kind, BehaviorKind, "kind")
        subject = _identifier(
            subject_symbol_id or requirement.type_ref or requirement.parameter_name,
            "subject_symbol_id",
        )
        _check_optional_roots(
            self.roots,
            value_provenance=value_provenance,
            program_contract=program_contract,
            memory_facet=memory_facet,
        )

        atoms: list[BehaviorEvidenceAtom] = []
        if include_requirement_atoms:
            atoms.extend(_atoms_from_requirement(self.roots, requirement, subject))
        if contract_delta is not None:
            if not isinstance(contract_delta, ProgramContractDelta):
                raise RequiredBehaviorSynthesisError(
                    "contract_delta must be ProgramContractDelta"
                )
            atoms.extend(
                _atoms_from_contract_delta(self.roots, contract_delta, subject)
            )
        atoms.extend(
            _normalize_atoms(self.roots, evidence, subject_symbol_id=subject)
        )

        if not atoms:
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.INSUFFICIENT_EVIDENCE,
                reason="no independent behavior evidence was supplied",
                missing=tuple(item.value for item in _KIND_ANY_OF[behavior_kind]),
                evidence_ids=(),
                assumptions=(),
                unsupported=(),
                winners={},
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        winners, conflicts, assumption_atoms, unsupported_atoms = _select_winners(atoms)
        evidence_ids = tuple(sorted({atom.evidence_id for atom in atoms}))
        assumptions = tuple(
            sorted({atom.clause_ref for atom in assumption_atoms} | {a.clause_ref for a in winners.values() if a.assumption})
        )
        unsupported = tuple(
            sorted({atom.clause_ref for atom in unsupported_atoms})
        )
        bindings = tuple(
            BehaviorClauseBinding.from_atom(atom) for atom in winners.values()
        )

        if conflicts:
            conflict_ids = tuple(
                sorted(
                    {
                        atom.evidence_id
                        for _, group in conflicts
                        for atom in group
                    }
                )
            )
            conflict_families = tuple(sorted(family.value for family, _ in conflicts))
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.CONFLICTING_EVIDENCE,
                reason=(
                    "same-rank independent evidence conflicts for families: "
                    + ",".join(conflict_families)
                ),
                missing=(),
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                conflicting_evidence_ids=conflict_ids,
                bindings=bindings,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        if unsupported_atoms and not winners:
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.UNSUPPORTED_CLAUSE,
                reason="all supplied clauses are unsupported for synthesis",
                missing=tuple(item.value for item in _KIND_ANY_OF[behavior_kind]),
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        structural = _authoritative_structural_families(winners)
        required_any = _KIND_ANY_OF[behavior_kind]
        if not (structural & required_any):
            # Distinguish pure implementation-hypothesis evidence.
            only_hypothesis = all(
                atom.precedence is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
                for atom in winners.values()
            ) and bool(winners)
            gap_kind = (
                BehaviorGapKind.IMPLEMENTATION_ONLY
                if only_hypothesis
                else BehaviorGapKind.KIND_REQUIREMENT_UNMET
                if winners
                else BehaviorGapKind.INSUFFICIENT_EVIDENCE
            )
            missing = tuple(
                sorted(family.value for family in required_any if family not in structural)
            )
            reason = {
                BehaviorGapKind.IMPLEMENTATION_ONLY: (
                    "only implementation hypotheses present; independent "
                    "evidence is required to admit behavior"
                ),
                BehaviorGapKind.KIND_REQUIREMENT_UNMET: (
                    f"authoritative structural evidence for {behavior_kind.value} "
                    "is incomplete"
                ),
                BehaviorGapKind.INSUFFICIENT_EVIDENCE: (
                    "insufficient authoritative evidence for required behavior"
                ),
            }[gap_kind]
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=gap_kind,
                reason=reason,
                missing=missing,
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                bindings=bindings,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        buckets = _bucket_refs(winners)
        if not (
            buckets["field_refs"]
            or buckets["constructor_refs"]
            or buckets["method_refs"]
            or buckets["invariant_refs"]
            or buckets["state_transition_refs"]
        ):
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.INSUFFICIENT_EVIDENCE,
                reason="no structural clause refs survived authoritative selection",
                missing=tuple(item.value for item in required_any),
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                bindings=bindings,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        best = _best_precedence(winners)
        if best is BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS:
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.IMPLEMENTATION_ONLY,
                reason="winning precedence is implementation hypothesis only",
                missing=tuple(item.value for item in required_any),
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                bindings=bindings,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        proof_refs = list(buckets["proof_refs"])
        proof_refs.extend(requirement.proof_refs)
        if contract_delta is not None:
            proof_refs.extend(contract_delta.proof_refs)
        proof_refs = sorted(set(proof_refs))

        try:
            contract = RequiredBehaviorContract(
                roots=self.roots,
                behavior_id=(
                    f"behavior:{behavior_kind.value}:{subject}:"
                    f"{requirement.requirement_id}"
                ),
                kind=behavior_kind,
                subject_symbol_id=subject,
                evidence_precedence=best,
                field_refs=buckets["field_refs"],
                constructor_refs=buckets["constructor_refs"],
                method_refs=buckets["method_refs"],
                invariant_refs=buckets["invariant_refs"],
                state_transition_refs=buckets["state_transition_refs"],
                effect_refs=buckets["effect_refs"],
                capability_refs=buckets["capability_refs"],
                authorization_refs=buckets["authorization_refs"],
                resource_refs=buckets["resource_refs"],
                proof_refs=tuple(proof_refs),
                placement_decision_ref=_text(
                    placement_decision_ref, "placement_decision_ref", required=False
                ),
                implementation_hypothesis=False,
            )
        except (ChangePropagationError, ChangePropagationAuthorityError) as exc:
            return self._gap_receipt(
                requirement=requirement,
                subject=subject,
                kind=behavior_kind,
                gap_kind=BehaviorGapKind.INSUFFICIENT_EVIDENCE,
                reason=f"required behavior contract rejected: {exc}",
                missing=tuple(item.value for item in required_any),
                evidence_ids=evidence_ids,
                assumptions=assumptions,
                unsupported=unsupported,
                winners=winners,
                bindings=bindings,
                value_provenance=value_provenance,
                program_contract=program_contract,
                memory_facet=memory_facet,
                contract_delta=contract_delta,
            )

        disposition = SynthesisDisposition.ADMITTED
        receipt_id = _receipt_id(
            self.roots,
            requirement.requirement_id,
            subject,
            behavior_kind,
            disposition,
            evidence_ids,
        )
        return RequiredBehaviorSynthesisReceipt(
            roots=self.roots,
            receipt_id=receipt_id,
            requirement_id=requirement.requirement_id,
            subject_symbol_id=subject,
            kind=behavior_kind,
            disposition=disposition,
            evidence_precedence=best,
            clause_bindings=bindings,
            assumptions=assumptions,
            unsupported_clauses=unsupported,
            evidence_ids=evidence_ids,
            proof_refs=tuple(proof_refs),
            producer_id=PRODUCER_ID,
            implementation_request=False,
            contract=contract,
            gap=None,
            contract_delta_id=(
                contract_delta.content_id if contract_delta is not None else ""
            ),
            value_provenance_id=_optional_root_id(
                value_provenance, ("graph_id", "content_id", "identity")
            ),
            memory_facet_ref=_optional_root_id(
                memory_facet, ("content_id", "facet_id", "identity")
            ),
            program_contract_ref=_optional_root_id(
                program_contract, ("content_id", "contract_id", "identity")
            ),
        )

    def _gap_receipt(
        self,
        *,
        requirement: MissingInputRequirement,
        subject: str,
        kind: BehaviorKind,
        gap_kind: BehaviorGapKind,
        reason: str,
        missing: Sequence[str],
        evidence_ids: Sequence[str],
        assumptions: Sequence[str],
        unsupported: Sequence[str],
        winners: Mapping[BehaviorClauseFamily, BehaviorEvidenceAtom],
        conflicting_evidence_ids: Sequence[str] = (),
        bindings: Sequence[BehaviorClauseBinding] = (),
        value_provenance: Any = None,
        program_contract: Any = None,
        memory_facet: Any = None,
        contract_delta: ProgramContractDelta | None = None,
    ) -> RequiredBehaviorSynthesisReceipt:
        evidence_ids_t = tuple(sorted(set(evidence_ids)))
        gap = BehaviorGap(
            roots=self.roots,
            gap_id=_gap_id(
                self.roots,
                requirement.requirement_id,
                subject,
                gap_kind,
                evidence_ids_t,
            ),
            kind=gap_kind,
            subject_symbol_id=subject,
            requirement_id=requirement.requirement_id,
            reason=reason,
            missing_families=tuple(missing),
            conflicting_evidence_ids=tuple(conflicting_evidence_ids),
            unsupported_clauses=tuple(unsupported),
            assumptions=tuple(assumptions),
            evidence_ids=evidence_ids_t,
            implementation_request=False,
        )
        disposition = SynthesisDisposition.BEHAVIOR_GAP
        best = _best_precedence(winners) if winners else (
            BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS
        )
        if not bindings and winners:
            bindings = tuple(
                BehaviorClauseBinding.from_atom(atom) for atom in winners.values()
            )
        receipt_id = _receipt_id(
            self.roots,
            requirement.requirement_id,
            subject,
            kind,
            disposition,
            evidence_ids_t,
        )
        return RequiredBehaviorSynthesisReceipt(
            roots=self.roots,
            receipt_id=receipt_id,
            requirement_id=requirement.requirement_id,
            subject_symbol_id=subject,
            kind=kind,
            disposition=disposition,
            evidence_precedence=best,
            clause_bindings=tuple(bindings),
            assumptions=tuple(assumptions),
            unsupported_clauses=tuple(unsupported),
            evidence_ids=evidence_ids_t,
            proof_refs=(),
            producer_id=PRODUCER_ID,
            implementation_request=False,
            contract=None,
            gap=gap,
            contract_delta_id=(
                contract_delta.content_id if contract_delta is not None else ""
            ),
            value_provenance_id=_optional_root_id(
                value_provenance, ("graph_id", "content_id", "identity")
            ),
            memory_facet_ref=_optional_root_id(
                memory_facet, ("content_id", "facet_id", "identity")
            ),
            program_contract_ref=_optional_root_id(
                program_contract, ("content_id", "contract_id", "identity")
            ),
        )


def synthesize_required_behavior(
    roots: PropagationAuthorityRoots,
    requirement: MissingInputRequirement,
    **kwargs: Any,
) -> RequiredBehaviorSynthesisReceipt:
    """Stateless convenience entry point for required-behavior synthesis."""
    return RequiredBehaviorSynthesizer(roots).synthesize(requirement, **kwargs)


__all__ = (
    "BEHAVIOR_EVIDENCE_ATOM_SCHEMA",
    "BEHAVIOR_CLAUSE_BINDING_SCHEMA",
    "BEHAVIOR_GAP_SCHEMA",
    "REQUIRED_BEHAVIOR_SYNTHESIS_RECEIPT_SCHEMA",
    "PRODUCER_ID",
    "PRECEDENCE_RANK",
    "PRECEDENCE_GROUPS",
    "MAX_EVIDENCE_ATOMS",
    "BehaviorClauseFamily",
    "BehaviorGapKind",
    "SynthesisDisposition",
    "BehaviorEvidenceAtom",
    "BehaviorClauseBinding",
    "BehaviorGap",
    "RequiredBehaviorSynthesisReceipt",
    "RequiredBehaviorSynthesizer",
    "RequiredBehaviorSynthesisError",
    "RequiredBehaviorSynthesisAuthorityError",
    "RequiredBehaviorSynthesisBoundsError",
    # Re-exports of canonical RPR-022 contracts / enums (not redefined).
    "RequiredBehaviorContract",
    "BehaviorEvidencePrecedence",
    "BehaviorKind",
    "synthesize_required_behavior",
    "coerce_precedence",
    "coerce_clause_family",
    "precedence_rank",
    "is_authoritative",
    "all_clause_families",
    "all_precedence_levels",
)
