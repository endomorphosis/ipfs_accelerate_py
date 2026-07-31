"""Independent content-addressed program premise corpus (LPR-005).

``ProgramLogicPremiseCorpusBuilder`` projects referenced evidence into a
supervisor-owned, body-free theorem corpus.  The lazy integration layer may
project that record into a Hammer ``CorpusManifest``; this module never
imports optional ``ipfs_datasets_py`` / Hammer types eagerly.

Authority lattice (fail-closed):

* Reviewed contracts, normative specs, and explicitly reviewed conformance
  tests are **expectation** classes and may carry ``expectation_authority``.
* Candidate implementation, comments, runtime witnesses, history, vector/KG
  analogues, and model material are **hypotheses only** — never expectation
  or semantic authority.
* Static type/effect/dataflow/graph facts may be structural premises without
  claiming logical closure.
* ``CorpusManifest`` / corpus identity establish structural and identity
  integrity, not arbitrary logical consistency.
* Suspected authoritative contradiction emits bounded consistency
  obligations.  Only an independently replayed unsat core or native conflict
  proof creates a :class:`PremiseConflictReceipt`; unknown consistency
  abstains without claiming a minimal conflict.

Incremental rebuild with the same current premises and tombstones equals a
clean rebuild (shared content identity), including retained tombstones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .program_logic_prediction_contracts import (
    PROGRAM_LOGIC_PREDICTION_VERSION,
    ProgramLogicAuthorityError,
    ProgramLogicAuthorityRoots,
    ProgramLogicPredictionBoundsError,
    ProgramLogicPredictionError,
    SourceAuthorityClass,
    SourceRouteKind,
)


# ---------------------------------------------------------------------------
# Schemas / bounds
# ---------------------------------------------------------------------------

PROGRAM_LOGIC_PREMISE_CORPUS_VERSION: Final[int] = 1

PROGRAM_LOGIC_PREMISE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-premise@1"
)
PROGRAM_LOGIC_PREMISE_CORPUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-premise-corpus@1"
)
PREMISE_CONFLICT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-conflict-receipt@1"
)
PREMISE_CONSISTENCY_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-consistency-obligation@1"
)
PREMISE_TOMBSTONE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-tombstone@1"
)
PREMISE_FEATURE_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-feature-set@1"
)
PREMISE_LICENSE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-license-policy@1"
)
PREMISE_SPAN_DIGEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-span-digest@1"
)
PREMISE_DEPENDENCY_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/premise-dependency-edge@1"
)
LAZY_CORPUS_MANIFEST_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lazy-corpus-manifest-projection@1"
)

MAX_PREMISE_BYTES: Final[int] = 65_536
MAX_CORPUS_BYTES: Final[int] = 1_048_576
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_PREMISE_COUNT: Final[int] = 10_000
MAX_EDGE_COUNT: Final[int] = 32_768
MAX_SPAN_OFFSET: Final[int] = 2**63 - 1
MAX_STATEMENT_DIGEST_BYTES: Final[int] = 128
MAX_STATEMENT_REF_BYTES: Final[int] = 512

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
        "unlowered_directive",
        "raw_directive",
        "directive_body",
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

_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "api_key=",
    "apikey=",
    "password=",
    "secret=",
    "private_key",
    "authorization:",
    "bearer ",
    "-----begin",
    "client_secret=",
)

_TOMBSTONE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "path_deleted",
        "blob_changed",
        "premise_removed",
        "authority_revoked",
        "superseded",
    }
)

_EXPORT_POLICIES: Final[frozenset[str]] = frozenset(
    {
        "internal",
        "exportable",
        "redacted_export",
        "never_export",
    }
)

_REDACTION_POLICIES: Final[frozenset[str]] = frozenset(
    {
        "none",
        "span_only",
        "identifiers_only",
        "full_redact",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProgramLogicPremiseCorpusError(ContractValidationError):
    """Base class for premise-corpus failures."""


class ProgramLogicPremiseCorpusBoundsError(ProgramLogicPremiseCorpusError):
    """A premise or corpus exceeded declared compactness bounds."""


class ForgedPremiseIdentityError(ProgramLogicPremiseCorpusError):
    """A stored content identity did not match the canonical preimage."""


class DuplicatePremiseIdentityError(ProgramLogicPremiseCorpusError):
    """The same premise identity was bound to conflicting statements."""


class PremiseDerivationCycleError(ProgramLogicPremiseCorpusError):
    """Premise dependency edges form a derivation cycle."""


class PremiseSelfValidationError(ProgramLogicPremiseCorpusError):
    """A premise attempted to validate or expect itself."""


class PremiseAuthorityError(ProgramLogicPremiseCorpusError):
    """Authority, expectation, or semantic flags violated the lattice."""


class PremiseStructuralConflictError(ProgramLogicPremiseCorpusError):
    """Structural/identity integrity failed (distinct from logical unsat)."""


class PremiseConflictProofError(ProgramLogicPremiseCorpusError):
    """A conflict receipt lacked an independently replayable proof binding."""


class PremiseUnloweredDirectiveError(ProgramLogicPremiseCorpusError):
    """Unlowered or raw directive material was supplied as a premise."""


# ---------------------------------------------------------------------------
# Closed taxonomies
# ---------------------------------------------------------------------------


class PremiseSourceClass(str, Enum):
    """Closed source classes projected into the premise corpus."""

    # Expectation classes (may carry expectation_authority under policy).
    REVIEWED_CONTRACT = "reviewed_contract"
    NORMATIVE_SPEC = "normative_spec"
    REVIEWED_CONFORMANCE_TEST = "reviewed_conformance_test"
    # Static structural facts (not logical closure).
    TYPE_AND_EFFECT_FACTS = "type_and_effect_facts"
    VALUE_PROVENANCE = "value_provenance"
    PROGRAM_GRAPH = "program_graph"
    SCHEMA_PROTOCOL = "schema_protocol"
    LOCAL_STATIC = "local_static"
    # Hypotheses only (never expectation or semantic authority).
    CANDIDATE_IMPLEMENTATION = "candidate_implementation"
    COMMENT = "comment"
    RUNTIME_WITNESS = "runtime_witness"
    HISTORY = "history"
    VECTOR_ANALOGUE = "vector_analogue"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MODEL_HYPOTHESIS = "model_hypothesis"
    THEOREM_CORPUS = "theorem_corpus"
    GIT_LINEAGE = "git_lineage"


class PremiseAuthority(str, Enum):
    """Independent premise authority lattice; never conflated with proof status."""

    EXPECTATION = "expectation"
    STATIC_FACT = "static_fact"
    HYPOTHESIS = "hypothesis"
    NONE = "none"


class ConsistencyDisposition(str, Enum):
    """Closed corpus-level consistency outcomes.

    Structural integrity is separate from arbitrary logical consistency.
    Unknown consistency abstains; only a replayed conflict proof elevates
    to :attr:`LOGICAL_CONFLICT_PROVED`.
    """

    UNKNOWN = "unknown"
    STRUCTURAL_INTEGRITY_OK = "structural_integrity_ok"
    STRUCTURAL_CONFLICT = "structural_conflict"
    SUSPECTED_AUTHORITATIVE_CONTRADICTION = "suspected_authoritative_contradiction"
    CONSISTENCY_OBLIGATION_EMITTED = "consistency_obligation_emitted"
    LOGICAL_CONFLICT_PROVED = "logical_conflict_proved"


class ConflictProofKind(str, Enum):
    """Closed kinds of independently replayable conflict proofs."""

    UNSAT_CORE = "unsat_core"
    NATIVE_CONFLICT_PROOF = "native_conflict_proof"


class PremiseEdgeKind(str, Enum):
    """Closed dependency edge kinds between premises."""

    DERIVES_FROM = "derives_from"
    REFINES = "refines"
    INVALIDATES = "invalidates"
    ASSUMES = "assumes"
    TRANSLATES = "translates"
    CONFLICTS_WITH = "conflicts_with"


_EXPECTATION_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.REVIEWED_CONTRACT,
        PremiseSourceClass.NORMATIVE_SPEC,
        PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
    }
)

_STATIC_FACT_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
        PremiseSourceClass.VALUE_PROVENANCE,
        PremiseSourceClass.PROGRAM_GRAPH,
        PremiseSourceClass.SCHEMA_PROTOCOL,
        PremiseSourceClass.LOCAL_STATIC,
    }
)

_HYPOTHESIS_SOURCE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
        PremiseSourceClass.COMMENT,
        PremiseSourceClass.RUNTIME_WITNESS,
        PremiseSourceClass.HISTORY,
        PremiseSourceClass.VECTOR_ANALOGUE,
        PremiseSourceClass.KNOWLEDGE_GRAPH,
        PremiseSourceClass.MODEL_HYPOTHESIS,
        PremiseSourceClass.THEOREM_CORPUS,
        PremiseSourceClass.GIT_LINEAGE,
    }
)

_SOURCE_CLASS_TO_ROUTE: Final[dict[PremiseSourceClass, SourceRouteKind]] = {
    PremiseSourceClass.REVIEWED_CONTRACT: SourceRouteKind.REVIEWED_CONTRACT,
    PremiseSourceClass.NORMATIVE_SPEC: SourceRouteKind.NORMATIVE_SPEC,
    PremiseSourceClass.REVIEWED_CONFORMANCE_TEST: SourceRouteKind.REVIEWED_TEST,
    PremiseSourceClass.TYPE_AND_EFFECT_FACTS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.VALUE_PROVENANCE: SourceRouteKind.DATAFLOW,
    PremiseSourceClass.PROGRAM_GRAPH: SourceRouteKind.GRAPH,
    PremiseSourceClass.SCHEMA_PROTOCOL: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.LOCAL_STATIC: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.CANDIDATE_IMPLEMENTATION: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.COMMENT: SourceRouteKind.HISTORY,
    PremiseSourceClass.RUNTIME_WITNESS: SourceRouteKind.RUNTIME_WITNESS,
    PremiseSourceClass.HISTORY: SourceRouteKind.HISTORY,
    PremiseSourceClass.VECTOR_ANALOGUE: SourceRouteKind.VECTOR,
    PremiseSourceClass.KNOWLEDGE_GRAPH: SourceRouteKind.KNOWLEDGE_GRAPH,
    PremiseSourceClass.MODEL_HYPOTHESIS: SourceRouteKind.LLM,
    PremiseSourceClass.THEOREM_CORPUS: SourceRouteKind.LOCAL_STATIC,
    PremiseSourceClass.GIT_LINEAGE: SourceRouteKind.HISTORY,
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ProgramLogicPremiseCorpusError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ProgramLogicPremiseCorpusError(f"{field_name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ProgramLogicPremiseCorpusBoundsError(
            f"{field_name} exceeds its byte bound"
        )
    _assert_no_secret_text(value, field_name)
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise ProgramLogicPremiseCorpusError(
            f"{field_name} must be an opaque compact identifier"
        )
    return value


def _bounded_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramLogicPremiseCorpusError(f"{field_name} must be a finite integer")
    if value < minimum or value > MAX_SPAN_OFFSET:
        raise ProgramLogicPremiseCorpusBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise PremiseAuthorityError(f"{field_name} must be a relative repository path")
    return candidate.as_posix()


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ProgramLogicPremiseCorpusError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _ids(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise ProgramLogicPremiseCorpusError(
            f"{field_name} must be a sequence of identifiers"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise ProgramLogicPremiseCorpusBoundsError(f"{field_name} exceeds its item bound")
    result_list: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = _identifier(value, field_name)
        if item not in seen:
            seen.add(item)
            result_list.append(item)
    result = tuple(result_list if preserve_order else sorted(result_list))
    if required and not result:
        raise ProgramLogicPremiseCorpusError(f"{field_name} must not be empty")
    return result


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramLogicPremiseCorpusError(f"{field_name} must be a boolean")
    return value


def _assert_no_secret_text(value: str, field_name: str) -> None:
    lowered = value.lower()
    for marker in _SECRET_VALUE_MARKERS:
        if marker in lowered:
            raise ProgramLogicPremiseCorpusError(
                f"{field_name} may not contain secret material"
            )


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise ProgramLogicPremiseCorpusError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramLogicPremiseCorpusError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ProgramLogicPremiseCorpusError(
                    f"{field_name} may not contain source bodies"
                )
            if normalized in _SECRET_KEY_MARKERS:
                raise ProgramLogicPremiseCorpusError(
                    f"{field_name} may not contain secret material"
                )
            if "unlowered" in normalized:
                raise PremiseUnloweredDirectiveError(
                    f"{field_name} may not contain unlowered directives"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ProgramLogicPremiseCorpusError(
            f"{field_name} may not contain binary bodies"
        )
    elif isinstance(value, str):
        _assert_no_secret_text(value, field_name)
        lowered = value.lower()
        if "unlowered_directive" in lowered or "#unlowered" in lowered:
            raise PremiseUnloweredDirectiveError(
                f"{field_name} may not contain unlowered directives"
            )


def _bounded(record: CanonicalContract, name: str, *, limit: int = MAX_PREMISE_BYTES) -> None:
    _assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > limit:
        raise ProgramLogicPremiseCorpusBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ForgedPremiseIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ProgramLogicPremiseCorpusError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (
        None,
        PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
        PROGRAM_LOGIC_PREDICTION_VERSION,
    ):
        raise ProgramLogicPremiseCorpusError(
            f"{name} has an unsupported contract version"
        )
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ProgramLogicPremiseCorpusError(f"{name} contains unsupported fields")
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
    raise ProgramLogicPremiseCorpusError("roots must be ProgramLogicAuthorityRoots")


def _default_authority(source_class: PremiseSourceClass) -> PremiseAuthority:
    if source_class in _EXPECTATION_SOURCE_CLASSES:
        return PremiseAuthority.EXPECTATION
    if source_class in _STATIC_FACT_SOURCE_CLASSES:
        return PremiseAuthority.STATIC_FACT
    if source_class in _HYPOTHESIS_SOURCE_CLASSES:
        return PremiseAuthority.HYPOTHESIS
    return PremiseAuthority.NONE


def _statement_digest(statement_ref: str, extra: str = "") -> str:
    material = f"{statement_ref}\0{extra}".encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


# ---------------------------------------------------------------------------
# Nested records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremiseFeatureSet(CanonicalContract):
    """Bounded symbol / type / effect / import feature bindings."""

    SCHEMA: ClassVar[str] = PREMISE_FEATURE_SET_SCHEMA

    symbol_feature_refs: tuple[str, ...] = ()
    type_feature_refs: tuple[str, ...] = ()
    effect_feature_refs: tuple[str, ...] = ()
    import_feature_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol_feature_refs",
            _ids(self.symbol_feature_refs, "symbol_feature_refs"),
        )
        object.__setattr__(
            self,
            "type_feature_refs",
            _ids(self.type_feature_refs, "type_feature_refs"),
        )
        object.__setattr__(
            self,
            "effect_feature_refs",
            _ids(self.effect_feature_refs, "effect_feature_refs"),
        )
        object.__setattr__(
            self,
            "import_feature_refs",
            _ids(self.import_feature_refs, "import_feature_refs"),
        )
        _bounded(self, "premise feature set")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "symbol_feature_refs": list(self.symbol_feature_refs),
            "type_feature_refs": list(self.type_feature_refs),
            "effect_feature_refs": list(self.effect_feature_refs),
            "import_feature_refs": list(self.import_feature_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseFeatureSet":
        fields = (
            "symbol_feature_refs",
            "type_feature_refs",
            "effect_feature_refs",
            "import_feature_refs",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "premise feature set"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PremiseSpanDigest(CanonicalContract):
    """Source-span digest without retaining source bodies."""

    SCHEMA: ClassVar[str] = PREMISE_SPAN_DIGEST_SCHEMA

    path: str
    start_offset: int
    end_offset: int
    content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        start = _bounded_int(self.start_offset, "start_offset")
        end = _bounded_int(self.end_offset, "end_offset")
        if end < start:
            raise ProgramLogicPremiseCorpusError(
                "end_offset must be greater than or equal to start_offset"
            )
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)
        digest = _text(
            self.content_digest,
            "content_digest",
            required=True,
            limit=MAX_STATEMENT_DIGEST_BYTES,
        )
        if not digest.startswith("sha256:") and not digest.startswith("b"):
            raise ProgramLogicPremiseCorpusError(
                "content_digest must be a sha256 or content-addressed digest"
            )
        object.__setattr__(self, "content_digest", digest)
        _bounded(self, "premise span digest")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "path": self.path,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseSpanDigest":
        fields = ("path", "start_offset", "end_offset", "content_digest")
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "premise span digest"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PremiseLicensePolicy(CanonicalContract):
    """License, redaction, and export policy bound to a premise."""

    SCHEMA: ClassVar[str] = PREMISE_LICENSE_POLICY_SCHEMA

    license_id: str
    redaction_policy: str = "none"
    export_policy: str = "internal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "license_id", _identifier(self.license_id, "license_id")
        )
        redaction = _text(self.redaction_policy, "redaction_policy", required=True)
        if redaction not in _REDACTION_POLICIES:
            raise ProgramLogicPremiseCorpusError(
                "redaction_policy must be a closed redaction policy"
            )
        object.__setattr__(self, "redaction_policy", redaction)
        export = _text(self.export_policy, "export_policy", required=True)
        if export not in _EXPORT_POLICIES:
            raise ProgramLogicPremiseCorpusError(
                "export_policy must be a closed export policy"
            )
        object.__setattr__(self, "export_policy", export)
        _bounded(self, "premise license policy")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "license_id": self.license_id,
            "redaction_policy": self.redaction_policy,
            "export_policy": self.export_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseLicensePolicy":
        fields = ("license_id", "redaction_policy", "export_policy")
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, fields, "premise license policy")
        )
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PremiseDependencyEdge(CanonicalContract):
    """One directed dependency / invalidation / conflict edge."""

    SCHEMA: ClassVar[str] = PREMISE_DEPENDENCY_EDGE_SCHEMA

    from_premise_id: str
    to_premise_id: str
    kind: PremiseEdgeKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_premise_id",
            _identifier(self.from_premise_id, "from_premise_id"),
        )
        object.__setattr__(
            self, "to_premise_id", _identifier(self.to_premise_id, "to_premise_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, PremiseEdgeKind, "kind"))
        if self.from_premise_id == self.to_premise_id:
            raise PremiseSelfValidationError(
                "premise dependency edges cannot be self-referential"
            )
        _bounded(self, "premise dependency edge")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "from_premise_id": self.from_premise_id,
            "to_premise_id": self.to_premise_id,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseDependencyEdge":
        fields = ("from_premise_id", "to_premise_id", "kind")
        value = cls(
            **_decode_fields(payload, cls.SCHEMA, fields, "premise dependency edge")
        )
        _verify_identity(payload, value)
        return value


def _feature_set(value: Any) -> PremiseFeatureSet:
    if isinstance(value, PremiseFeatureSet):
        return value
    if value is None:
        return PremiseFeatureSet()
    if isinstance(value, Mapping):
        return (
            PremiseFeatureSet.from_dict(value)
            if "schema" in value
            else PremiseFeatureSet(**value)
        )
    raise ProgramLogicPremiseCorpusError("features must be PremiseFeatureSet")


def _span_digest(value: Any) -> PremiseSpanDigest | None:
    if value is None or value == "":
        return None
    if isinstance(value, PremiseSpanDigest):
        return value
    if isinstance(value, Mapping):
        return (
            PremiseSpanDigest.from_dict(value)
            if "schema" in value
            else PremiseSpanDigest(**value)
        )
    raise ProgramLogicPremiseCorpusError("span must be PremiseSpanDigest")


def _license_policy(value: Any) -> PremiseLicensePolicy:
    if isinstance(value, PremiseLicensePolicy):
        return value
    if isinstance(value, Mapping):
        return (
            PremiseLicensePolicy.from_dict(value)
            if "schema" in value
            else PremiseLicensePolicy(**value)
        )
    raise ProgramLogicPremiseCorpusError("license_policy must be PremiseLicensePolicy")


def _dependency_edges(values: Any) -> tuple[PremiseDependencyEdge, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ProgramLogicPremiseCorpusError("dependency_edges must be a sequence")
    if len(raw) > MAX_EDGE_COUNT:
        raise ProgramLogicPremiseCorpusBoundsError(
            "dependency_edges exceeds its item bound"
        )
    items: list[PremiseDependencyEdge] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, PremiseDependencyEdge):
            edge = item
        elif isinstance(item, Mapping):
            edge = (
                PremiseDependencyEdge.from_dict(item)
                if "schema" in item
                else PremiseDependencyEdge(**item)
            )
        else:
            raise ProgramLogicPremiseCorpusError(
                "dependency_edges entries must be PremiseDependencyEdge"
            )
        if edge.content_id not in seen:
            seen.add(edge.content_id)
            items.append(edge)
    return tuple(sorted(items, key=lambda edge: edge.content_id))


# ---------------------------------------------------------------------------
# ProgramLogicPremise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramLogicPremise(CanonicalContract):
    """One content-addressed program premise (body-free, authority-explicit)."""

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_PREMISE_SCHEMA

    roots: ProgramLogicAuthorityRoots
    premise_id: str
    source_class: PremiseSourceClass
    statement_ref: str
    statement_digest: str
    lowering_ref: str
    authority: PremiseAuthority = PremiseAuthority.NONE
    source_precedence: int = 0
    expectation_authority: bool = False
    semantic_authority: bool = False
    features: PremiseFeatureSet | None = None
    span: PremiseSpanDigest | None = None
    dependency_edges: tuple[PremiseDependencyEdge, ...] = ()
    translation_refs: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    invalidator_refs: tuple[str, ...] = ()
    contract_identity: str = ""
    graph_identity: str = ""
    tree_identity: str = ""
    license_policy: PremiseLicensePolicy | None = None
    source_route: SourceRouteKind | None = None
    source_authority_class: SourceAuthorityClass = SourceAuthorityClass.NONE
    conflicts_with: tuple[str, ...] = ()
    self_validation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "premise_id", _identifier(self.premise_id, "premise_id")
        )
        object.__setattr__(
            self,
            "source_class",
            _enum(self.source_class, PremiseSourceClass, "source_class"),
        )
        object.__setattr__(
            self,
            "statement_ref",
            _text(
                self.statement_ref,
                "statement_ref",
                required=True,
                limit=MAX_STATEMENT_REF_BYTES,
            ),
        )
        digest = _text(
            self.statement_digest,
            "statement_digest",
            required=True,
            limit=MAX_STATEMENT_DIGEST_BYTES,
        )
        if not digest.startswith("sha256:") and not digest.startswith("b"):
            raise ProgramLogicPremiseCorpusError(
                "statement_digest must be a sha256 or content-addressed digest"
            )
        object.__setattr__(self, "statement_digest", digest)
        object.__setattr__(
            self, "lowering_ref", _identifier(self.lowering_ref, "lowering_ref")
        )

        authority = (
            _enum(self.authority, PremiseAuthority, "authority")
            if self.authority is not None
            else _default_authority(self.source_class)
        )
        # Reconcile authority with source class (fail-closed lattice).
        expected = _default_authority(self.source_class)
        if authority is PremiseAuthority.NONE:
            authority = expected
        if (
            expected is PremiseAuthority.HYPOTHESIS
            and authority is PremiseAuthority.EXPECTATION
        ):
            raise PremiseAuthorityError(
                "hypothesis source classes cannot claim expectation authority class"
            )
        if (
            expected is PremiseAuthority.EXPECTATION
            and authority is PremiseAuthority.HYPOTHESIS
        ):
            # Explicit demotion of a reviewed source is allowed only via static_fact.
            raise PremiseAuthorityError(
                "expectation source classes cannot be demoted to hypothesis"
            )
        object.__setattr__(self, "authority", authority)

        object.__setattr__(
            self,
            "source_precedence",
            _bounded_int(self.source_precedence, "source_precedence"),
        )
        object.__setattr__(
            self,
            "expectation_authority",
            _bool(self.expectation_authority, "expectation_authority"),
        )
        # semantic_authority is always false for premises in this corpus.
        if self.semantic_authority is not False:
            raise PremiseAuthorityError(
                "premises cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

        if self.expectation_authority:
            if self.source_class not in _EXPECTATION_SOURCE_CLASSES:
                raise PremiseAuthorityError(
                    "only reviewed contracts/specs/conformance tests may carry "
                    "expectation_authority"
                )
            if self.authority is not PremiseAuthority.EXPECTATION:
                raise PremiseAuthorityError(
                    "expectation_authority requires PremiseAuthority.EXPECTATION"
                )

        if self.authority is PremiseAuthority.HYPOTHESIS and self.expectation_authority:
            raise PremiseAuthorityError(
                "hypothesis premises cannot carry expectation_authority"
            )

        object.__setattr__(self, "features", _feature_set(self.features))
        object.__setattr__(self, "span", _span_digest(self.span))
        object.__setattr__(
            self, "dependency_edges", _dependency_edges(self.dependency_edges)
        )
        object.__setattr__(
            self, "translation_refs", _ids(self.translation_refs, "translation_refs")
        )
        object.__setattr__(
            self, "assumption_refs", _ids(self.assumption_refs, "assumption_refs")
        )
        object.__setattr__(
            self, "invalidator_refs", _ids(self.invalidator_refs, "invalidator_refs")
        )
        object.__setattr__(
            self,
            "contract_identity",
            _text(self.contract_identity, "contract_identity"),
        )
        object.__setattr__(
            self, "graph_identity", _text(self.graph_identity, "graph_identity")
        )
        tree = _text(self.tree_identity, "tree_identity")
        if not tree:
            tree = self.roots.tree_id
        object.__setattr__(self, "tree_identity", tree)
        if self.tree_identity != self.roots.tree_id:
            raise PremiseStructuralConflictError(
                "tree_identity must match roots.tree_id"
            )

        if self.license_policy is None:
            policy = PremiseLicensePolicy(license_id="license:unspecified")
        else:
            policy = _license_policy(self.license_policy)
        object.__setattr__(self, "license_policy", policy)

        if self.source_route is None:
            route = _SOURCE_CLASS_TO_ROUTE[self.source_class]
        else:
            route = _enum(self.source_route, SourceRouteKind, "source_route")
        object.__setattr__(self, "source_route", route)

        src_auth = _enum(
            self.source_authority_class,
            SourceAuthorityClass,
            "source_authority_class",
        )
        if self.expectation_authority and src_auth not in (
            SourceAuthorityClass.AUTHORITATIVE,
            SourceAuthorityClass.CONFORMANCE,
        ):
            # Align lattice: expectation_authority implies authoritative/conformance.
            src_auth = (
                SourceAuthorityClass.CONFORMANCE
                if self.source_class is PremiseSourceClass.REVIEWED_CONFORMANCE_TEST
                else SourceAuthorityClass.AUTHORITATIVE
            )
        if self.authority is PremiseAuthority.HYPOTHESIS and src_auth in (
            SourceAuthorityClass.AUTHORITATIVE,
            SourceAuthorityClass.CONFORMANCE,
        ):
            raise PremiseAuthorityError(
                "hypothesis premises cannot carry authoritative source_authority_class"
            )
        if src_auth is SourceAuthorityClass.NONE and self.authority is PremiseAuthority.HYPOTHESIS:
            src_auth = SourceAuthorityClass.NOMINATING
        if src_auth is SourceAuthorityClass.NONE and self.authority is PremiseAuthority.STATIC_FACT:
            src_auth = SourceAuthorityClass.DIAGNOSTIC
        object.__setattr__(self, "source_authority_class", src_auth)

        object.__setattr__(
            self, "conflicts_with", _ids(self.conflicts_with, "conflicts_with")
        )
        object.__setattr__(
            self, "self_validation", _bool(self.self_validation, "self_validation")
        )
        if self.self_validation:
            raise PremiseSelfValidationError(
                "premises that self-validate are rejected"
            )

        # Self-reference via dependency edges or assumption of own identity.
        for edge in self.dependency_edges:
            if (
                edge.from_premise_id != self.premise_id
                and edge.to_premise_id != self.premise_id
            ):
                # Edges attached to a premise must involve it.
                raise PremiseStructuralConflictError(
                    "dependency_edges on a premise must reference that premise_id"
                )
            if edge.from_premise_id == edge.to_premise_id:
                raise PremiseSelfValidationError(
                    "self-referential dependency edges are rejected"
                )
        if self.premise_id in self.assumption_refs:
            raise PremiseSelfValidationError(
                "a premise cannot assume its own identity"
            )
        if self.premise_id in self.invalidator_refs:
            raise PremiseSelfValidationError(
                "a premise cannot invalidate itself as its own authority"
            )
        if self.premise_id in self.conflicts_with:
            raise PremiseSelfValidationError(
                "a premise cannot list itself in conflicts_with"
            )

        _bounded(self, "program logic premise")

    def _payload(self) -> dict[str, Any]:
        assert self.features is not None
        assert self.license_policy is not None
        assert self.source_route is not None
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "roots": self.roots.to_dict(),
            "premise_id": self.premise_id,
            "source_class": self.source_class.value,
            "statement_ref": self.statement_ref,
            "statement_digest": self.statement_digest,
            "lowering_ref": self.lowering_ref,
            "authority": self.authority.value,
            "source_precedence": self.source_precedence,
            "expectation_authority": self.expectation_authority,
            "semantic_authority": False,
            "features": self.features.to_dict(),
            "span": self.span.to_dict() if self.span is not None else None,
            "dependency_edges": [edge.to_dict() for edge in self.dependency_edges],
            "translation_refs": list(self.translation_refs),
            "assumption_refs": list(self.assumption_refs),
            "invalidator_refs": list(self.invalidator_refs),
            "contract_identity": self.contract_identity,
            "graph_identity": self.graph_identity,
            "tree_identity": self.tree_identity,
            "license_policy": self.license_policy.to_dict(),
            "source_route": self.source_route.value,
            "source_authority_class": self.source_authority_class.value,
            "conflicts_with": list(self.conflicts_with),
            "self_validation": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicPremise":
        fields = (
            "roots",
            "premise_id",
            "source_class",
            "statement_ref",
            "statement_digest",
            "lowering_ref",
            "authority",
            "source_precedence",
            "expectation_authority",
            "semantic_authority",
            "features",
            "span",
            "dependency_edges",
            "translation_refs",
            "assumption_refs",
            "invalidator_refs",
            "contract_identity",
            "graph_identity",
            "tree_identity",
            "license_policy",
            "source_route",
            "source_authority_class",
            "conflicts_with",
            "self_validation",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "program logic premise")
        values["roots"] = _roots(values["roots"])
        if "features" in values:
            values["features"] = _feature_set(values["features"])
        if "span" in values:
            values["span"] = _span_digest(values["span"])
        if "dependency_edges" in values:
            values["dependency_edges"] = _dependency_edges(values["dependency_edges"])
        if "license_policy" in values:
            values["license_policy"] = _license_policy(values["license_policy"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Tombstones / obligations / conflict receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremiseTombstone(CanonicalContract):
    """Retained removal record so incremental rebuild equals clean rebuild."""

    SCHEMA: ClassVar[str] = PREMISE_TOMBSTONE_SCHEMA

    premise_id: str
    statement_digest: str
    reason: str
    tree_identity: str
    superseding_premise_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "premise_id", _identifier(self.premise_id, "premise_id")
        )
        digest = _text(
            self.statement_digest,
            "statement_digest",
            required=True,
            limit=MAX_STATEMENT_DIGEST_BYTES,
        )
        object.__setattr__(self, "statement_digest", digest)
        reason = _text(self.reason, "reason", required=True)
        if reason not in _TOMBSTONE_REASONS:
            raise ProgramLogicPremiseCorpusError(
                "unsupported premise tombstone reason"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self, "tree_identity", _identifier(self.tree_identity, "tree_identity")
        )
        object.__setattr__(
            self,
            "superseding_premise_id",
            _text(self.superseding_premise_id, "superseding_premise_id"),
        )
        _bounded(self, "premise tombstone")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "premise_id": self.premise_id,
            "statement_digest": self.statement_digest,
            "reason": self.reason,
            "tree_identity": self.tree_identity,
            "superseding_premise_id": self.superseding_premise_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseTombstone":
        fields = (
            "premise_id",
            "statement_digest",
            "reason",
            "tree_identity",
            "superseding_premise_id",
        )
        value = cls(**_decode_fields(payload, cls.SCHEMA, fields, "premise tombstone"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PremiseConsistencyObligation(CanonicalContract):
    """Bounded obligation emitted when authoritative contradiction is suspected.

    Does **not** claim a logical conflict.  Only a later independently
    replayed unsat core / native proof may mint a conflict receipt.
    """

    SCHEMA: ClassVar[str] = PREMISE_CONSISTENCY_OBLIGATION_SCHEMA

    roots: ProgramLogicAuthorityRoots
    obligation_id: str
    premise_ids: tuple[str, ...]
    reason_code: str
    bound_refs: tuple[str, ...] = ()
    disposition: ConsistencyDisposition = (
        ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "obligation_id", _identifier(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True, limit=64),
        )
        object.__setattr__(
            self, "reason_code", _identifier(self.reason_code, "reason_code")
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        disposition = _enum(
            self.disposition, ConsistencyDisposition, "disposition"
        )
        if disposition not in (
            ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED,
            ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
        ):
            raise ProgramLogicPremiseCorpusError(
                "consistency obligations cannot claim logical conflict proof"
            )
        object.__setattr__(self, "disposition", disposition)
        _bounded(self, "premise consistency obligation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "roots": self.roots.to_dict(),
            "obligation_id": self.obligation_id,
            "premise_ids": list(self.premise_ids),
            "reason_code": self.reason_code,
            "bound_refs": list(self.bound_refs),
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseConsistencyObligation":
        fields = (
            "roots",
            "obligation_id",
            "premise_ids",
            "reason_code",
            "bound_refs",
            "disposition",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "premise consistency obligation"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class PremiseConflictReceipt(CanonicalContract):
    """Logical conflict receipt from an independently replayed proof only.

    Structural integrity failures never mint this receipt.  Unknown
    consistency abstains without creating one.
    """

    SCHEMA: ClassVar[str] = PREMISE_CONFLICT_RECEIPT_SCHEMA

    roots: ProgramLogicAuthorityRoots
    receipt_id: str
    premise_ids: tuple[str, ...]
    proof_kind: ConflictProofKind
    proof_artifact_ref: str
    replay_receipt_ref: str
    translator_id: str
    toolchain_id: str
    independently_replayed: bool = True
    unsat_core_refs: tuple[str, ...] = ()
    native_conflict_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self,
            "premise_ids",
            _ids(self.premise_ids, "premise_ids", required=True, limit=64),
        )
        object.__setattr__(
            self, "proof_kind", _enum(self.proof_kind, ConflictProofKind, "proof_kind")
        )
        object.__setattr__(
            self,
            "proof_artifact_ref",
            _identifier(self.proof_artifact_ref, "proof_artifact_ref"),
        )
        object.__setattr__(
            self,
            "replay_receipt_ref",
            _identifier(self.replay_receipt_ref, "replay_receipt_ref"),
        )
        object.__setattr__(
            self, "translator_id", _identifier(self.translator_id, "translator_id")
        )
        object.__setattr__(
            self, "toolchain_id", _identifier(self.toolchain_id, "toolchain_id")
        )
        if self.independently_replayed is not True:
            raise PremiseConflictProofError(
                "conflict receipts require independently_replayed=True"
            )
        object.__setattr__(self, "independently_replayed", True)
        object.__setattr__(
            self, "unsat_core_refs", _ids(self.unsat_core_refs, "unsat_core_refs")
        )
        object.__setattr__(
            self,
            "native_conflict_refs",
            _ids(self.native_conflict_refs, "native_conflict_refs"),
        )
        if self.proof_kind is ConflictProofKind.UNSAT_CORE and not self.unsat_core_refs:
            raise PremiseConflictProofError(
                "unsat_core conflict receipts require unsat_core_refs"
            )
        if (
            self.proof_kind is ConflictProofKind.NATIVE_CONFLICT_PROOF
            and not self.native_conflict_refs
        ):
            raise PremiseConflictProofError(
                "native conflict receipts require native_conflict_refs"
            )
        if self.translator_id != self.roots.translator_id:
            raise PremiseStructuralConflictError(
                "conflict receipt translator_id must match roots"
            )
        if self.toolchain_id != self.roots.toolchain_id:
            raise PremiseStructuralConflictError(
                "conflict receipt toolchain_id must match roots"
            )
        _bounded(self, "premise conflict receipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "roots": self.roots.to_dict(),
            "receipt_id": self.receipt_id,
            "premise_ids": list(self.premise_ids),
            "proof_kind": self.proof_kind.value,
            "proof_artifact_ref": self.proof_artifact_ref,
            "replay_receipt_ref": self.replay_receipt_ref,
            "translator_id": self.translator_id,
            "toolchain_id": self.toolchain_id,
            "independently_replayed": True,
            "unsat_core_refs": list(self.unsat_core_refs),
            "native_conflict_refs": list(self.native_conflict_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PremiseConflictReceipt":
        fields = (
            "roots",
            "receipt_id",
            "premise_ids",
            "proof_kind",
            "proof_artifact_ref",
            "replay_receipt_ref",
            "translator_id",
            "toolchain_id",
            "independently_replayed",
            "unsat_core_refs",
            "native_conflict_refs",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "premise conflict receipt"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _detect_derivation_cycles(
    premises: Sequence[ProgramLogicPremise],
    extra_edges: Sequence[PremiseDependencyEdge] = (),
) -> None:
    adjacency: dict[str, set[str]] = {}
    premise_ids = {item.premise_id for item in premises}
    for premise in premises:
        adjacency.setdefault(premise.premise_id, set())
        for edge in premise.dependency_edges:
            if edge.kind in (
                PremiseEdgeKind.DERIVES_FROM,
                PremiseEdgeKind.REFINES,
                PremiseEdgeKind.ASSUMES,
                PremiseEdgeKind.TRANSLATES,
            ):
                # Edge A -derives_from-> B means A depends on B.
                if edge.from_premise_id in premise_ids:
                    adjacency.setdefault(edge.from_premise_id, set()).add(
                        edge.to_premise_id
                    )
    for edge in extra_edges:
        if edge.kind in (
            PremiseEdgeKind.DERIVES_FROM,
            PremiseEdgeKind.REFINES,
            PremiseEdgeKind.ASSUMES,
            PremiseEdgeKind.TRANSLATES,
        ):
            if edge.from_premise_id in premise_ids:
                adjacency.setdefault(edge.from_premise_id, set()).add(edge.to_premise_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise PremiseDerivationCycleError(
                "premise derivation graph contains a cycle"
            )
        visiting.add(node)
        for dep in adjacency.get(node, ()):
            if dep in premise_ids or dep in adjacency:
                visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in list(adjacency):
        visit(node)


def _assert_no_duplicate_identities(
    premises: Sequence[ProgramLogicPremise],
) -> None:
    by_id: dict[str, ProgramLogicPremise] = {}
    for premise in premises:
        existing = by_id.get(premise.premise_id)
        if existing is None:
            by_id[premise.premise_id] = premise
            continue
        if existing.statement_digest != premise.statement_digest:
            raise DuplicatePremiseIdentityError(
                "duplicate premise identity with conflicting statement digest"
            )
        if existing.content_id != premise.content_id:
            raise DuplicatePremiseIdentityError(
                "duplicate premise identity with conflicting content identity"
            )


@dataclass(frozen=True)
class ProgramLogicPremiseCorpus(CanonicalContract):
    """Immutable content-addressed program premise corpus snapshot."""

    SCHEMA: ClassVar[str] = PROGRAM_LOGIC_PREMISE_CORPUS_SCHEMA

    roots: ProgramLogicAuthorityRoots
    premises: tuple[ProgramLogicPremise, ...] = ()
    tombstones: tuple[PremiseTombstone, ...] = ()
    consistency_obligations: tuple[PremiseConsistencyObligation, ...] = ()
    conflict_receipts: tuple[PremiseConflictReceipt, ...] = ()
    consistency_disposition: ConsistencyDisposition = ConsistencyDisposition.UNKNOWN
    producer_ref: str = "program-logic-premise-corpus-builder@1"
    graph_identity: str = ""
    index_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))

        if self.premises is None:
            premises_raw: Sequence[Any] = ()
        elif isinstance(self.premises, Sequence) and not isinstance(
            self.premises, (str, bytes, bytearray)
        ):
            premises_raw = self.premises
        else:
            raise ProgramLogicPremiseCorpusError("premises must be a sequence")
        if len(premises_raw) > MAX_PREMISE_COUNT:
            raise ProgramLogicPremiseCorpusBoundsError(
                "premises exceeds its item bound"
            )
        premises: list[ProgramLogicPremise] = []
        for item in premises_raw:
            if isinstance(item, ProgramLogicPremise):
                premise = item
            elif isinstance(item, Mapping):
                premise = (
                    ProgramLogicPremise.from_dict(item)
                    if "schema" in item
                    else ProgramLogicPremise(**item)
                )
            else:
                raise ProgramLogicPremiseCorpusError(
                    "premises must contain ProgramLogicPremise values"
                )
            if premise.roots.content_id != self.roots.content_id:
                raise PremiseStructuralConflictError(
                    "premise roots must match corpus roots"
                )
            premises.append(premise)
        _assert_no_duplicate_identities(premises)
        # Deduplicate identical re-ingestions by premise_id (same digest).
        by_id = {item.premise_id: item for item in premises}
        premises_tuple = tuple(
            sorted(by_id.values(), key=lambda item: item.premise_id)
        )
        object.__setattr__(self, "premises", premises_tuple)
        _detect_derivation_cycles(premises_tuple)

        tombstones = _decode_tombstones(self.tombstones)
        object.__setattr__(self, "tombstones", tombstones)
        # Active premises cannot also be tombstoned under the same identity.
        tombstone_ids = {item.premise_id for item in tombstones}
        live_ids = {item.premise_id for item in premises_tuple}
        if tombstone_ids & live_ids:
            raise PremiseStructuralConflictError(
                "a premise cannot be live and tombstoned under the same identity"
            )

        obligations = _decode_obligations(self.consistency_obligations, self.roots)
        object.__setattr__(self, "consistency_obligations", obligations)

        receipts = _decode_conflict_receipts(self.conflict_receipts, self.roots)
        object.__setattr__(self, "conflict_receipts", receipts)

        disposition = _enum(
            self.consistency_disposition,
            ConsistencyDisposition,
            "consistency_disposition",
        )
        # Derive disposition fail-closed from structural state + receipts.
        if receipts:
            disposition = ConsistencyDisposition.LOGICAL_CONFLICT_PROVED
        elif obligations and disposition in (
            ConsistencyDisposition.UNKNOWN,
            ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK,
        ):
            disposition = ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED
        elif disposition is ConsistencyDisposition.LOGICAL_CONFLICT_PROVED and not receipts:
            raise PremiseConflictProofError(
                "LOGICAL_CONFLICT_PROVED requires a conflict receipt"
            )
        elif (
            disposition is ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
            and not receipts
            and not obligations
        ):
            disposition = ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
        elif disposition is ConsistencyDisposition.UNKNOWN and premises_tuple and not obligations:
            # Presence of premises with no obligations/receipts still leaves
            # *logical* consistency unknown; structural integrity is OK.
            disposition = ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
        object.__setattr__(self, "consistency_disposition", disposition)

        object.__setattr__(
            self, "producer_ref", _identifier(self.producer_ref, "producer_ref")
        )
        graph = _text(self.graph_identity, "graph_identity")
        if not graph:
            graph = self.roots.graph_id
        object.__setattr__(self, "graph_identity", graph)
        index = _text(self.index_identity, "index_identity")
        if not index:
            index = self.roots.index_id
        object.__setattr__(self, "index_identity", index)

        _bounded(self, "program logic premise corpus", limit=MAX_CORPUS_BYTES)

    @property
    def corpus_id(self) -> str:
        return self.content_id

    @property
    def revision(self) -> str:
        """Content-addressed revision used by lazy CorpusManifest projection."""
        return self.content_id

    def premise_by_id(self, premise_id: str) -> ProgramLogicPremise | None:
        for premise in self.premises:
            if premise.premise_id == premise_id:
                return premise
        return None

    def expectation_premises(self) -> tuple[ProgramLogicPremise, ...]:
        return tuple(item for item in self.premises if item.expectation_authority)

    def hypothesis_premises(self) -> tuple[ProgramLogicPremise, ...]:
        return tuple(
            item
            for item in self.premises
            if item.authority is PremiseAuthority.HYPOTHESIS
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
            "roots": self.roots.to_dict(),
            "premises": [item.to_dict() for item in self.premises],
            "tombstones": [item.to_dict() for item in self.tombstones],
            "consistency_obligations": [
                item.to_dict() for item in self.consistency_obligations
            ],
            "conflict_receipts": [item.to_dict() for item in self.conflict_receipts],
            "consistency_disposition": self.consistency_disposition.value,
            "producer_ref": self.producer_ref,
            "graph_identity": self.graph_identity,
            "index_identity": self.index_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramLogicPremiseCorpus":
        fields = (
            "roots",
            "premises",
            "tombstones",
            "consistency_obligations",
            "conflict_receipts",
            "consistency_disposition",
            "producer_ref",
            "graph_identity",
            "index_identity",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "program logic premise corpus"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


def _decode_tombstones(values: Any) -> tuple[PremiseTombstone, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ProgramLogicPremiseCorpusError("tombstones must be a sequence")
    if len(raw) > MAX_PREMISE_COUNT:
        raise ProgramLogicPremiseCorpusBoundsError(
            "tombstones exceeds its item bound"
        )
    items: list[PremiseTombstone] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, PremiseTombstone):
            tombstone = item
        elif isinstance(item, Mapping):
            tombstone = (
                PremiseTombstone.from_dict(item)
                if "schema" in item
                else PremiseTombstone(**item)
            )
        else:
            raise ProgramLogicPremiseCorpusError(
                "tombstones must contain PremiseTombstone values"
            )
        if tombstone.content_id not in seen:
            seen.add(tombstone.content_id)
            items.append(tombstone)
    return tuple(sorted(items, key=lambda item: (item.premise_id, item.content_id)))


def _decode_obligations(
    values: Any, roots: ProgramLogicAuthorityRoots
) -> tuple[PremiseConsistencyObligation, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ProgramLogicPremiseCorpusError(
            "consistency_obligations must be a sequence"
        )
    items: list[PremiseConsistencyObligation] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, PremiseConsistencyObligation):
            obligation = item
        elif isinstance(item, Mapping):
            obligation = (
                PremiseConsistencyObligation.from_dict(item)
                if "schema" in item
                else PremiseConsistencyObligation(**item)
            )
        else:
            raise ProgramLogicPremiseCorpusError(
                "consistency_obligations must contain PremiseConsistencyObligation"
            )
        if obligation.roots.content_id != roots.content_id:
            raise PremiseStructuralConflictError(
                "consistency obligation roots must match corpus roots"
            )
        if obligation.content_id not in seen:
            seen.add(obligation.content_id)
            items.append(obligation)
    return tuple(sorted(items, key=lambda item: item.obligation_id))


def _decode_conflict_receipts(
    values: Any, roots: ProgramLogicAuthorityRoots
) -> tuple[PremiseConflictReceipt, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ProgramLogicPremiseCorpusError("conflict_receipts must be a sequence")
    items: list[PremiseConflictReceipt] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, PremiseConflictReceipt):
            receipt = item
        elif isinstance(item, Mapping):
            receipt = (
                PremiseConflictReceipt.from_dict(item)
                if "schema" in item
                else PremiseConflictReceipt(**item)
            )
        else:
            raise ProgramLogicPremiseCorpusError(
                "conflict_receipts must contain PremiseConflictReceipt"
            )
        if receipt.roots.content_id != roots.content_id:
            raise PremiseStructuralConflictError(
                "conflict receipt roots must match corpus roots"
            )
        if receipt.content_id not in seen:
            seen.add(receipt.content_id)
            items.append(receipt)
    return tuple(sorted(items, key=lambda item: item.receipt_id))


# ---------------------------------------------------------------------------
# Lazy CorpusManifest projection (no datasets import)
# ---------------------------------------------------------------------------


def project_lazy_corpus_manifest(
    corpus: ProgramLogicPremiseCorpus,
) -> dict[str, Any]:
    """Project a supervisor corpus into a Hammer-shaped manifest payload.

    Returns a plain ``dict`` suitable for a later optional
    ``CorpusManifest.from_dict`` call.  Never imports ``ipfs_datasets_py``.
    Structural/identity fields only — does not claim logical consistency.
    """

    if not isinstance(corpus, ProgramLogicPremiseCorpus):
        raise ProgramLogicPremiseCorpusError(
            "project_lazy_corpus_manifest requires ProgramLogicPremiseCorpus"
        )
    theorems: list[dict[str, Any]] = []
    for premise in corpus.premises:
        theorems.append(
            {
                "theorem_id": premise.premise_id,
                "statement_digest": premise.statement_digest,
                "statement_ref": premise.statement_ref,
                "source_class": premise.source_class.value,
                "expectation_authority": premise.expectation_authority,
                "semantic_authority": False,
                "lowering_ref": premise.lowering_ref,
                "license_id": (
                    premise.license_policy.license_id
                    if premise.license_policy is not None
                    else "license:unspecified"
                ),
                "content_digest": premise.content_id,
                "tree_identity": premise.tree_identity,
                "translation_refs": list(premise.translation_refs),
            }
        )
    payload = {
        "schema": LAZY_CORPUS_MANIFEST_PROJECTION_SCHEMA,
        "contract_version": PROGRAM_LOGIC_PREMISE_CORPUS_VERSION,
        "corpus_revision": corpus.revision,
        "roots": {
            "repository_id": corpus.roots.repository_id,
            "tree_id": corpus.roots.tree_id,
            "corpus_id": corpus.roots.corpus_id,
            "translator_id": corpus.roots.translator_id,
            "toolchain_id": corpus.roots.toolchain_id,
            "policy_id": corpus.roots.policy_id,
        },
        "theorems": theorems,
        "tombstones": [
            {
                "theorem_id": item.premise_id,
                "statement_digest": item.statement_digest,
                "reason": item.reason,
            }
            for item in corpus.tombstones
        ],
        "consistency_disposition": corpus.consistency_disposition.value,
        "logical_consistency_claimed": False,
        "structural_integrity_only": True,
        "producer_ref": corpus.producer_ref,
    }
    _assert_body_free(payload, "lazy corpus manifest projection")
    return payload


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class _EvidenceDraft:
    """Internal mutable draft used by the builder before admission."""

    premise_id: str
    source_class: PremiseSourceClass
    statement_ref: str
    statement_digest: str
    lowering_ref: str
    expectation_authority: bool = False
    source_precedence: int = 0
    features: PremiseFeatureSet | None = None
    span: PremiseSpanDigest | None = None
    dependency_edges: tuple[PremiseDependencyEdge, ...] = ()
    translation_refs: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    invalidator_refs: tuple[str, ...] = ()
    contract_identity: str = ""
    graph_identity: str = ""
    license_policy: PremiseLicensePolicy | None = None
    conflicts_with: tuple[str, ...] = ()
    self_validation: bool = False
    authority: PremiseAuthority | None = None


class ProgramLogicPremiseCorpusBuilder:
    """Build a supervisor-owned content-addressed program premise corpus.

    Evidence is admitted by explicit source class.  Reviewed contracts/specs/
    conformance tests become expectation premises; candidate implementation,
    comments, runtime, history, vector/KG, and model material become
    non-authoritative hypotheses.
    """

    def __init__(self, roots: ProgramLogicAuthorityRoots) -> None:
        self._roots = _roots(roots)
        self._drafts: dict[str, _EvidenceDraft] = {}
        self._tombstones: dict[str, PremiseTombstone] = {}
        self._conflict_receipts: list[PremiseConflictReceipt] = []
        self._explicit_obligations: list[PremiseConsistencyObligation] = []
        self._previous: ProgramLogicPremiseCorpus | None = None
        self._producer_ref = "program-logic-premise-corpus-builder@1"

    @property
    def roots(self) -> ProgramLogicAuthorityRoots:
        return self._roots

    def with_previous(
        self, previous: ProgramLogicPremiseCorpus | Mapping[str, Any] | None
    ) -> "ProgramLogicPremiseCorpusBuilder":
        """Attach a previous corpus for incremental rebuild / tombstone retention."""
        if previous is None:
            self._previous = None
            return self
        if isinstance(previous, ProgramLogicPremiseCorpus):
            corpus = previous
        elif isinstance(previous, Mapping):
            corpus = ProgramLogicPremiseCorpus.from_dict(previous)
        else:
            raise ProgramLogicPremiseCorpusError(
                "previous must be ProgramLogicPremiseCorpus"
            )
        if corpus.roots.content_id != self._roots.content_id:
            # Allow tree transitions only when repository matches; otherwise fail.
            if corpus.roots.repository_id != self._roots.repository_id:
                raise PremiseStructuralConflictError(
                    "previous corpus repository_id must match builder roots"
                )
        self._previous = corpus
        # Retain prior tombstones under the new tree identity when reusing.
        for tombstone in corpus.tombstones:
            retained = PremiseTombstone(
                premise_id=tombstone.premise_id,
                statement_digest=tombstone.statement_digest,
                reason=tombstone.reason,
                tree_identity=self._roots.tree_id,
                superseding_premise_id=tombstone.superseding_premise_id,
            )
            self._tombstones[retained.premise_id] = retained
        return self

    def add_expectation(
        self,
        *,
        premise_id: str,
        source_class: PremiseSourceClass | str,
        statement_ref: str,
        lowering_ref: str,
        statement_digest: str | None = None,
        source_precedence: int = 100,
        features: PremiseFeatureSet | Mapping[str, Any] | None = None,
        span: PremiseSpanDigest | Mapping[str, Any] | None = None,
        dependency_edges: Sequence[PremiseDependencyEdge | Mapping[str, Any]] = (),
        translation_refs: Sequence[str] = (),
        assumption_refs: Sequence[str] = (),
        invalidator_refs: Sequence[str] = (),
        contract_identity: str = "",
        graph_identity: str = "",
        license_policy: PremiseLicensePolicy | Mapping[str, Any] | None = None,
        conflicts_with: Sequence[str] = (),
    ) -> "ProgramLogicPremiseCorpusBuilder":
        """Admit a reviewed contract, normative spec, or conformance test."""
        source = _enum(source_class, PremiseSourceClass, "source_class")
        if source not in _EXPECTATION_SOURCE_CLASSES:
            raise PremiseAuthorityError(
                "add_expectation requires a reviewed contract/spec/conformance class"
            )
        return self._add_draft(
            premise_id=premise_id,
            source_class=source,
            statement_ref=statement_ref,
            lowering_ref=lowering_ref,
            statement_digest=statement_digest,
            expectation_authority=True,
            source_precedence=source_precedence,
            features=features,
            span=span,
            dependency_edges=dependency_edges,
            translation_refs=translation_refs,
            assumption_refs=assumption_refs,
            invalidator_refs=invalidator_refs,
            contract_identity=contract_identity,
            graph_identity=graph_identity,
            license_policy=license_policy,
            conflicts_with=conflicts_with,
            authority=PremiseAuthority.EXPECTATION,
        )

    def add_static_fact(
        self,
        *,
        premise_id: str,
        source_class: PremiseSourceClass | str,
        statement_ref: str,
        lowering_ref: str,
        statement_digest: str | None = None,
        source_precedence: int = 50,
        features: PremiseFeatureSet | Mapping[str, Any] | None = None,
        span: PremiseSpanDigest | Mapping[str, Any] | None = None,
        dependency_edges: Sequence[PremiseDependencyEdge | Mapping[str, Any]] = (),
        translation_refs: Sequence[str] = (),
        assumption_refs: Sequence[str] = (),
        invalidator_refs: Sequence[str] = (),
        contract_identity: str = "",
        graph_identity: str = "",
        license_policy: PremiseLicensePolicy | Mapping[str, Any] | None = None,
        conflicts_with: Sequence[str] = (),
    ) -> "ProgramLogicPremiseCorpusBuilder":
        """Admit a structural type/effect/dataflow/graph fact (not a proof)."""
        source = _enum(source_class, PremiseSourceClass, "source_class")
        if source not in _STATIC_FACT_SOURCE_CLASSES:
            raise PremiseAuthorityError(
                "add_static_fact requires a static fact source class"
            )
        return self._add_draft(
            premise_id=premise_id,
            source_class=source,
            statement_ref=statement_ref,
            lowering_ref=lowering_ref,
            statement_digest=statement_digest,
            expectation_authority=False,
            source_precedence=source_precedence,
            features=features,
            span=span,
            dependency_edges=dependency_edges,
            translation_refs=translation_refs,
            assumption_refs=assumption_refs,
            invalidator_refs=invalidator_refs,
            contract_identity=contract_identity,
            graph_identity=graph_identity,
            license_policy=license_policy,
            conflicts_with=conflicts_with,
            authority=PremiseAuthority.STATIC_FACT,
        )

    def add_hypothesis(
        self,
        *,
        premise_id: str,
        source_class: PremiseSourceClass | str,
        statement_ref: str,
        lowering_ref: str,
        statement_digest: str | None = None,
        source_precedence: int = 10,
        features: PremiseFeatureSet | Mapping[str, Any] | None = None,
        span: PremiseSpanDigest | Mapping[str, Any] | None = None,
        dependency_edges: Sequence[PremiseDependencyEdge | Mapping[str, Any]] = (),
        translation_refs: Sequence[str] = (),
        assumption_refs: Sequence[str] = (),
        invalidator_refs: Sequence[str] = (),
        contract_identity: str = "",
        graph_identity: str = "",
        license_policy: PremiseLicensePolicy | Mapping[str, Any] | None = None,
        conflicts_with: Sequence[str] = (),
    ) -> "ProgramLogicPremiseCorpusBuilder":
        """Admit non-authoritative hypothesis material (never expectation)."""
        source = _enum(source_class, PremiseSourceClass, "source_class")
        if source not in _HYPOTHESIS_SOURCE_CLASSES:
            raise PremiseAuthorityError(
                "add_hypothesis requires a hypothesis source class"
            )
        return self._add_draft(
            premise_id=premise_id,
            source_class=source,
            statement_ref=statement_ref,
            lowering_ref=lowering_ref,
            statement_digest=statement_digest,
            expectation_authority=False,
            source_precedence=source_precedence,
            features=features,
            span=span,
            dependency_edges=dependency_edges,
            translation_refs=translation_refs,
            assumption_refs=assumption_refs,
            invalidator_refs=invalidator_refs,
            contract_identity=contract_identity,
            graph_identity=graph_identity,
            license_policy=license_policy,
            conflicts_with=conflicts_with,
            authority=PremiseAuthority.HYPOTHESIS,
        )

    def add_tombstone(
        self,
        *,
        premise_id: str,
        statement_digest: str,
        reason: str = "premise_removed",
        superseding_premise_id: str = "",
    ) -> "ProgramLogicPremiseCorpusBuilder":
        tombstone = PremiseTombstone(
            premise_id=premise_id,
            statement_digest=statement_digest,
            reason=reason,
            tree_identity=self._roots.tree_id,
            superseding_premise_id=superseding_premise_id,
        )
        self._tombstones[tombstone.premise_id] = tombstone
        self._drafts.pop(premise_id, None)
        return self

    def add_conflict_receipt(
        self, receipt: PremiseConflictReceipt | Mapping[str, Any]
    ) -> "ProgramLogicPremiseCorpusBuilder":
        """Attach an independently replayed conflict proof receipt."""
        if isinstance(receipt, PremiseConflictReceipt):
            value = receipt
        elif isinstance(receipt, Mapping):
            value = (
                PremiseConflictReceipt.from_dict(receipt)
                if "schema" in receipt
                else PremiseConflictReceipt(**receipt)
            )
        else:
            raise ProgramLogicPremiseCorpusError(
                "conflict receipt must be PremiseConflictReceipt"
            )
        if value.roots.content_id != self._roots.content_id:
            raise PremiseStructuralConflictError(
                "conflict receipt roots must match builder roots"
            )
        self._conflict_receipts.append(value)
        return self

    def _add_draft(
        self,
        *,
        premise_id: str,
        source_class: PremiseSourceClass,
        statement_ref: str,
        lowering_ref: str,
        statement_digest: str | None,
        expectation_authority: bool,
        source_precedence: int,
        features: PremiseFeatureSet | Mapping[str, Any] | None,
        span: PremiseSpanDigest | Mapping[str, Any] | None,
        dependency_edges: Sequence[PremiseDependencyEdge | Mapping[str, Any]],
        translation_refs: Sequence[str],
        assumption_refs: Sequence[str],
        invalidator_refs: Sequence[str],
        contract_identity: str,
        graph_identity: str,
        license_policy: PremiseLicensePolicy | Mapping[str, Any] | None,
        conflicts_with: Sequence[str],
        authority: PremiseAuthority,
        self_validation: bool = False,
    ) -> "ProgramLogicPremiseCorpusBuilder":
        premise_id = _identifier(premise_id, "premise_id")
        statement_ref = _text(
            statement_ref, "statement_ref", required=True, limit=MAX_STATEMENT_REF_BYTES
        )
        lowering_ref = _identifier(lowering_ref, "lowering_ref")
        if statement_digest is None:
            digest = _statement_digest(statement_ref, premise_id)
        else:
            digest = _text(
                statement_digest,
                "statement_digest",
                required=True,
                limit=MAX_STATEMENT_DIGEST_BYTES,
            )
        draft = _EvidenceDraft(
            premise_id=premise_id,
            source_class=source_class,
            statement_ref=statement_ref,
            statement_digest=digest,
            lowering_ref=lowering_ref,
            expectation_authority=expectation_authority,
            source_precedence=source_precedence,
            features=_feature_set(features) if features is not None else PremiseFeatureSet(),
            span=_span_digest(span),
            dependency_edges=_dependency_edges(dependency_edges),
            translation_refs=_ids(translation_refs, "translation_refs"),
            assumption_refs=_ids(assumption_refs, "assumption_refs"),
            invalidator_refs=_ids(invalidator_refs, "invalidator_refs"),
            contract_identity=_text(contract_identity, "contract_identity"),
            graph_identity=_text(graph_identity, "graph_identity"),
            license_policy=(
                _license_policy(license_policy)
                if license_policy is not None
                else PremiseLicensePolicy(license_id="license:unspecified")
            ),
            conflicts_with=_ids(conflicts_with, "conflicts_with"),
            self_validation=self_validation,
            authority=authority,
        )
        existing = self._drafts.get(premise_id)
        if existing is not None and existing.statement_digest != draft.statement_digest:
            raise DuplicatePremiseIdentityError(
                "duplicate premise identity with conflicting statement digest"
            )
        self._drafts[premise_id] = draft
        # Live admission supersedes any tombstone for the same id.
        self._tombstones.pop(premise_id, None)
        return self

    def _materialize_premises(self) -> tuple[ProgramLogicPremise, ...]:
        premises: list[ProgramLogicPremise] = []
        for draft in self._drafts.values():
            premises.append(
                ProgramLogicPremise(
                    roots=self._roots,
                    premise_id=draft.premise_id,
                    source_class=draft.source_class,
                    statement_ref=draft.statement_ref,
                    statement_digest=draft.statement_digest,
                    lowering_ref=draft.lowering_ref,
                    authority=draft.authority or _default_authority(draft.source_class),
                    source_precedence=draft.source_precedence,
                    expectation_authority=draft.expectation_authority,
                    semantic_authority=False,
                    features=draft.features,
                    span=draft.span,
                    dependency_edges=draft.dependency_edges,
                    translation_refs=draft.translation_refs,
                    assumption_refs=draft.assumption_refs,
                    invalidator_refs=draft.invalidator_refs,
                    contract_identity=draft.contract_identity,
                    graph_identity=draft.graph_identity or self._roots.graph_id,
                    tree_identity=self._roots.tree_id,
                    license_policy=draft.license_policy,
                    conflicts_with=draft.conflicts_with,
                    self_validation=draft.self_validation,
                )
            )
        return tuple(sorted(premises, key=lambda item: item.premise_id))

    def _emit_consistency_obligations(
        self, premises: Sequence[ProgramLogicPremise]
    ) -> tuple[PremiseConsistencyObligation, ...]:
        obligations: list[PremiseConsistencyObligation] = list(self._explicit_obligations)
        by_id = {item.premise_id: item for item in premises}
        seen_pairs: set[tuple[str, str]] = set()

        def emit(a: str, b: str, reason: str) -> None:
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            obligation_id = (
                "obligation:consistency:"
                + hashlib.sha256(f"{key[0]}|{key[1]}|{reason}".encode()).hexdigest()[:16]
            )
            obligations.append(
                PremiseConsistencyObligation(
                    roots=self._roots,
                    obligation_id=obligation_id,
                    premise_ids=key,
                    reason_code=reason,
                    bound_refs=(
                        by_id[key[0]].statement_digest,
                        by_id[key[1]].statement_digest,
                    ),
                    disposition=(
                        ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION
                    ),
                )
            )

        # Authoritative pairwise conflicts declared on expectation premises.
        for premise in premises:
            if not premise.expectation_authority:
                continue
            for other_id in premise.conflicts_with:
                other = by_id.get(other_id)
                if other is None:
                    continue
                if other.expectation_authority:
                    emit(
                        premise.premise_id,
                        other.premise_id,
                        "suspected_authoritative_contradiction",
                    )

        # Symmetric CONFLICTS_WITH edges between expectation premises.
        for premise in premises:
            for edge in premise.dependency_edges:
                if edge.kind is not PremiseEdgeKind.CONFLICTS_WITH:
                    continue
                a = by_id.get(edge.from_premise_id)
                b = by_id.get(edge.to_premise_id)
                if a is None or b is None:
                    continue
                if a.expectation_authority and b.expectation_authority:
                    emit(
                        a.premise_id,
                        b.premise_id,
                        "suspected_authoritative_contradiction",
                    )

        # Deduplicate by obligation_id.
        unique = {item.obligation_id: item for item in obligations}
        return tuple(sorted(unique.values(), key=lambda item: item.obligation_id))

    def _tombstones_from_previous(
        self, premises: Sequence[ProgramLogicPremise]
    ) -> tuple[PremiseTombstone, ...]:
        live_ids = {item.premise_id for item in premises}
        tombstones = dict(self._tombstones)
        if self._previous is not None:
            for old in self._previous.premises:
                if old.premise_id in live_ids:
                    continue
                if old.premise_id in tombstones:
                    continue
                tombstones[old.premise_id] = PremiseTombstone(
                    premise_id=old.premise_id,
                    statement_digest=old.statement_digest,
                    reason="premise_removed",
                    tree_identity=self._roots.tree_id,
                )
        return tuple(
            sorted(
                tombstones.values(),
                key=lambda item: (item.premise_id, item.content_id),
            )
        )

    def build(self) -> ProgramLogicPremiseCorpus:
        """Materialize the immutable corpus, obligations, and disposition."""
        premises = self._materialize_premises()
        _assert_no_duplicate_identities(premises)
        _detect_derivation_cycles(premises)
        tombstones = self._tombstones_from_previous(premises)
        # Ensure no live/tombstone collision after prior retention.
        live_ids = {item.premise_id for item in premises}
        tombstones = tuple(
            item for item in tombstones if item.premise_id not in live_ids
        )
        obligations = self._emit_consistency_obligations(premises)
        receipts = tuple(
            sorted(self._conflict_receipts, key=lambda item: item.receipt_id)
        )
        if receipts:
            disposition = ConsistencyDisposition.LOGICAL_CONFLICT_PROVED
        elif obligations:
            disposition = ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED
        elif premises:
            disposition = ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
        else:
            disposition = ConsistencyDisposition.UNKNOWN
        return ProgramLogicPremiseCorpus(
            roots=self._roots,
            premises=premises,
            tombstones=tombstones,
            consistency_obligations=obligations,
            conflict_receipts=receipts,
            consistency_disposition=disposition,
            producer_ref=self._producer_ref,
            graph_identity=self._roots.graph_id,
            index_identity=self._roots.index_id,
        )


def build_program_logic_premise_corpus(
    roots: ProgramLogicAuthorityRoots,
    *,
    expectations: Sequence[Mapping[str, Any]] = (),
    static_facts: Sequence[Mapping[str, Any]] = (),
    hypotheses: Sequence[Mapping[str, Any]] = (),
    tombstones: Sequence[Mapping[str, Any] | PremiseTombstone] = (),
    conflict_receipts: Sequence[Mapping[str, Any] | PremiseConflictReceipt] = (),
    previous: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
) -> ProgramLogicPremiseCorpus:
    """Convenience builder entry point for recipe-style corpus construction."""
    builder = ProgramLogicPremiseCorpusBuilder(roots)
    if previous is not None:
        builder.with_previous(previous)
    for item in expectations:
        builder.add_expectation(**dict(item))
    for item in static_facts:
        builder.add_static_fact(**dict(item))
    for item in hypotheses:
        builder.add_hypothesis(**dict(item))
    for item in tombstones:
        if isinstance(item, PremiseTombstone):
            builder.add_tombstone(
                premise_id=item.premise_id,
                statement_digest=item.statement_digest,
                reason=item.reason,
                superseding_premise_id=item.superseding_premise_id,
            )
        else:
            builder.add_tombstone(**dict(item))
    for item in conflict_receipts:
        builder.add_conflict_receipt(item)
    return builder.build()


def is_expectation_source_class(source_class: PremiseSourceClass | str) -> bool:
    source = _enum(source_class, PremiseSourceClass, "source_class")
    return source in _EXPECTATION_SOURCE_CLASSES


def is_hypothesis_source_class(source_class: PremiseSourceClass | str) -> bool:
    source = _enum(source_class, PremiseSourceClass, "source_class")
    return source in _HYPOTHESIS_SOURCE_CLASSES


__all__ = [
    "PROGRAM_LOGIC_PREMISE_CORPUS_VERSION",
    "PROGRAM_LOGIC_PREMISE_SCHEMA",
    "PROGRAM_LOGIC_PREMISE_CORPUS_SCHEMA",
    "PREMISE_CONFLICT_RECEIPT_SCHEMA",
    "PREMISE_CONSISTENCY_OBLIGATION_SCHEMA",
    "PREMISE_TOMBSTONE_SCHEMA",
    "LAZY_CORPUS_MANIFEST_PROJECTION_SCHEMA",
    "ProgramLogicPremiseCorpusError",
    "ProgramLogicPremiseCorpusBoundsError",
    "ForgedPremiseIdentityError",
    "DuplicatePremiseIdentityError",
    "PremiseDerivationCycleError",
    "PremiseSelfValidationError",
    "PremiseAuthorityError",
    "PremiseStructuralConflictError",
    "PremiseConflictProofError",
    "PremiseUnloweredDirectiveError",
    "PremiseSourceClass",
    "PremiseAuthority",
    "ConsistencyDisposition",
    "ConflictProofKind",
    "PremiseEdgeKind",
    "PremiseFeatureSet",
    "PremiseSpanDigest",
    "PremiseLicensePolicy",
    "PremiseDependencyEdge",
    "ProgramLogicPremise",
    "PremiseTombstone",
    "PremiseConsistencyObligation",
    "PremiseConflictReceipt",
    "ProgramLogicPremiseCorpus",
    "ProgramLogicPremiseCorpusBuilder",
    "project_lazy_corpus_manifest",
    "build_program_logic_premise_corpus",
    "is_expectation_source_class",
    "is_hypothesis_source_class",
]
