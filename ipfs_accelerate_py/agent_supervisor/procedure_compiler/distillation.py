"""Residual-hole distillation corpus admission and compact local resolvers.

Validated hole resolutions become bounded corpus rows.  This module owns
admission, not authority: it never promotes a resolver, skips validation, or
inlines prompts, transcripts, or source bodies.  Large payloads stay behind
content references.  Accepted and rejected examples are labeled only from
independent validation/proof/outcome bindings, and partitions stay disjoint.

Exact-cache, declarative-rule, deterministic-classifier, and small-local
resolvers consume only training rows and remain proposal producers.  Held-out
rows cannot train a resolver, confidence is not authority, and every route
still requires independent downstream validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, NoReturn

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    HoleType,
    ProcedureContractError,
    ProcedureSafetyError,
    ProviderClass,
    _bounded,
    _decode_fields,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
)
from .hole_resolution import (
    HOLE_VALIDATOR_REVISION,
    LOCAL_HOLE_ROUTE_ORDER,
    CompiledHoleContext,
    HoleCandidate,
    HoleContextReference,
    HoleProviderOutcome,
    HoleProviderResult,
    HoleRequest,
    HoleResolution,
    HoleResolutionAction,
    HoleResolutionValidator,
    HoleResolver,
    HoleValidationReceipt,
    ProviderCapacitySnapshot,
    default_hole_context_compiler,
    evidence_fingerprint,
)


BUILDER_REVISION: Final[str] = "DistillationCorpusBuilder@1"
EVALUATOR_REVISION: Final[str] = "DistillationEvaluation@1"
MAX_CORPUS_ROWS: Final[int] = MAX_ITEMS
MAX_FEATURE_VALUE_BYTES: Final[int] = 256

REQUIRED_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "validation_cid",
    "proof_cid",
    "outcome",
)
REQUIRED_PARTITIONS: Final[tuple[str, ...]] = (
    "training",
    "development",
    "held-out",
    "negative",
    "boundary",
    "adversarial",
)
REQUIRED_PRIVACY_CLASSES: Final[tuple[str, ...]] = (
    "no-secrets",
    "no-credentials",
    "no-private-prompts",
    "no-chain-of-thought",
    "no-source-bodies",
    "no-model-transcripts",
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
_PROMPT_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "assistant_prompt",
        "model_prompt",
        "private_prompt",
        "prompt",
        "prompt_body",
        "system_prompt",
        "user_prompt",
    }
)
_PRIVACY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "credential",
        "model_transcript",
        "password",
        "private_key",
        "secret",
        "source_body",
        "transcript",
    }
)
_REJECTED_PARTITIONS: Final[frozenset[str]] = frozenset({"negative", "adversarial"})
_COMPACT_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "selected",
    "template_id",
    "failure_class",
    "test_name",
    "lemma_name",
    "schema_ref",
)


class DistillationError(ProcedureContractError):
    """A distillation example, corpus row, or admission is unsafe."""


class DistillationAdmissionError(DistillationError):
    """An example was refused fail-closed and must not enter the corpus."""

    def __init__(self, message: str, reason_code: "DistillationReason") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class DistillationLabel(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class CorpusPartition(str, Enum):
    TRAINING = "training"
    DEVELOPMENT = "development"
    HELD_OUT = "held-out"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"


class DistillationReferenceKind(str, Enum):
    CONTEXT = "context"
    CANDIDATE = "candidate"
    VALIDATION = "validation"
    PROOF = "proof"
    COUNTEREXAMPLE = "counterexample"
    FEATURE = "feature"
    RESOLUTION = "resolution"


class DistillationReason(str, Enum):
    ADMITTED = "admitted"
    PROMPT_REJECTED = "prompt-rejected"
    STALE_EXAMPLE = "stale-example"
    UNVERIFIED_EXAMPLE = "unverified-example"
    MISLABELED_EXAMPLE = "mislabeled-example"
    PRIVATE_EXAMPLE = "private-example"
    PARTITION_LEAKAGE = "partition-leakage"
    MISSING_VALIDATION = "missing-validation"
    MISSING_PROOF = "missing-proof"
    MISSING_OUTCOME = "missing-outcome"
    MISSING_COUNTEREXAMPLE = "missing-counterexample"
    INCOMPLETE_PROVENANCE = "incomplete-provenance"
    BINDING_MISMATCH = "binding-mismatch"
    DUPLICATE_EXAMPLE = "duplicate-example"
    CORPUS_UNBOUNDED = "corpus-unbounded"
    AUTHORITY_REJECTED = "authority-rejected"
    CANDIDATE_TIER_REQUIRED = "candidate-tier-required"
    INCOMPLETE_CACHE_KEY = "incomplete-cache-key"
    HELD_OUT_TRAINING_REJECTED = "held-out-training-rejected"
    CONFIDENCE_AUTHORITY_REJECTED = "confidence-authority-rejected"


class DistillationAdmissionAction(str, Enum):
    ADMIT = "admit"
    REJECT = "reject"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise DistillationError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _refuse(reason: DistillationReason, message: str) -> NoReturn:
    raise DistillationAdmissionError(message, reason)


def _normalized_marker(value: str) -> str:
    return value.lower().replace("-", "_")


def _marker_hit(value: str, markers: frozenset[str]) -> bool:
    normalized = _normalized_marker(value)
    return any(marker in normalized for marker in markers)


def _scan_forbidden(value: Any) -> DistillationReason | None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                return DistillationReason.PRIVATE_EXAMPLE
            if _unsafe_key(raw_key) or _marker_hit(raw_key, _PRIVACY_MARKERS):
                return DistillationReason.PRIVATE_EXAMPLE
            if _marker_hit(raw_key, _PROMPT_MARKERS):
                return DistillationReason.PROMPT_REJECTED
            nested = _scan_forbidden(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            nested = _scan_forbidden(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, str):
        if _marker_hit(value, _PROMPT_MARKERS):
            return DistillationReason.PROMPT_REJECTED
        if _marker_hit(value, _PRIVACY_MARKERS):
            return DistillationReason.PRIVATE_EXAMPLE
    return None


def _reject_privacy(value: Any, field_name: str) -> None:
    reason = _scan_forbidden(value)
    if reason is DistillationReason.PROMPT_REJECTED:
        _refuse(reason, f"{field_name} contains a prompt body")
    if reason is DistillationReason.PRIVATE_EXAMPLE:
        _refuse(reason, f"{field_name} contains a private or secret field")


def _compact_feature_value(value: Any, field_name: str) -> Any:
    if type(value) is bool:
        return value
    if type(value) is int:
        return _nonnegative_int(value, field_name)
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_FEATURE_VALUE_BYTES:
            _refuse(
                DistillationReason.CORPUS_UNBOUNDED,
                f"{field_name} exceeds the compact feature bound",
            )
        return _identifier(value, field_name)
    _refuse(
        DistillationReason.CORPUS_UNBOUNDED,
        f"{field_name} must be a compact identifier or scalar",
    )


def _typed_features(value: Any, field_name: str = "typed_features") -> Mapping[str, Any]:
    frozen = _freeze(value if value is not None else {}, field_name)
    if not isinstance(frozen, Mapping):
        raise DistillationError(f"{field_name} must be a mapping")
    _reject_privacy(frozen, field_name)
    compact: dict[str, Any] = {}
    for raw_key, item in frozen.items():
        key = _identifier(raw_key, field_name)
        compact[key] = _compact_feature_value(item, f"{field_name}.{key}")
    return MappingProxyType(compact)


def _default_features(request: HoleRequest, candidate: HoleCandidate) -> Mapping[str, Any]:
    features: dict[str, Any] = {
        "hole_type": request.hole_type.value,
        "input_schema_ref": request.input_schema_ref,
        "output_schema_ref": request.output_schema_ref,
        "provider_class": candidate.provider_class.value,
    }
    for key in _COMPACT_OUTPUT_KEYS:
        item = candidate.output.get(key)
        if type(item) is str:
            try:
                features[key] = _identifier(item, key)
            except ProcedureContractError:
                continue
    return _typed_features(features)


def _label_from_accepted(accepted: bool) -> DistillationLabel:
    return DistillationLabel.ACCEPTED if accepted else DistillationLabel.REJECTED


@dataclass(frozen=True)
class DistillationContentReference:
    """Compact external body handle.  The referenced bytes never enter the row."""

    reference_id: str
    content_id: str
    kind: DistillationReferenceKind = DistillationReferenceKind.CONTEXT
    tree_id: str = ""
    byte_count: int = 0
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "reference_id")
        )
        object.__setattr__(self, "content_id", _identifier(self.content_id, "content_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, DistillationReferenceKind, "kind")
        )
        object.__setattr__(
            self, "tree_id", _identifier(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", required=False)
        )
        _reject_privacy(
            {
                "reference_id": self.reference_id,
                "content_id": self.content_id,
                "kind": self.kind.value,
                "summary": self.summary,
            },
            "content_references",
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "content_id": self.content_id,
            "kind": self.kind.value,
            "tree_id": self.tree_id,
            "byte_count": self.byte_count,
            "summary": self.summary,
        }

    @classmethod
    def from_record(
        cls, payload: Mapping[str, Any] | HoleContextReference | DistillationContentReference
    ) -> DistillationContentReference:
        if isinstance(payload, DistillationContentReference):
            return payload
        if isinstance(payload, HoleContextReference):
            return cls(
                reference_id=payload.reference_id,
                content_id=payload.content_id,
                kind=DistillationReferenceKind.CONTEXT,
                tree_id=payload.tree_id,
                byte_count=payload.byte_count,
                summary=payload.summary,
            )
        if not isinstance(payload, Mapping):
            raise DistillationError("content reference must be a mapping")
        return cls(
            reference_id=payload.get("reference_id", ""),
            content_id=payload.get("content_id", payload.get("referenced_content_id", "")),
            kind=payload.get("kind", DistillationReferenceKind.CONTEXT),
            tree_id=payload.get("tree_id", ""),
            byte_count=payload.get("byte_count", 0),
            summary=payload.get("summary", ""),
        )


def _content_references(values: Any) -> tuple[DistillationContentReference, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise DistillationError("content_references must be a sequence")
    if len(raw) > MAX_ITEMS:
        _refuse(
            DistillationReason.CORPUS_UNBOUNDED,
            "content_references exceeds its item bound",
        )
    result: list[DistillationContentReference] = []
    seen: set[str] = set()
    for item in raw:
        record = DistillationContentReference.from_record(item)
        if record.reference_id in seen:
            raise DistillationError("content_references contains a duplicate reference_id")
        seen.add(record.reference_id)
        result.append(record)
    return tuple(result)


@dataclass(frozen=True)
class DistillationCorpusRow:
    """Compact manifest row: provenance and counterexamples, never large bodies."""

    example_id: str
    example_cid: str
    partition: CorpusPartition
    label: DistillationLabel
    request_cid: str
    candidate_cid: str
    resolution_cid: str
    validation_cid: str
    proof_cid: str
    outcome: DistillationLabel
    counterexample_cids: tuple[str, ...]
    family_id: str
    language: str
    framework: str
    hole_id: str
    hole_type: HoleType

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "example_cid",
            "request_cid",
            "candidate_cid",
            "resolution_cid",
            "validation_cid",
            "proof_cid",
            "family_id",
            "language",
            "framework",
            "hole_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "partition", _enum(self.partition, CorpusPartition, "partition")
        )
        object.__setattr__(self, "label", _enum(self.label, DistillationLabel, "label"))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, DistillationLabel, "outcome")
        )
        object.__setattr__(self, "hole_type", _enum(self.hole_type, HoleType, "hole_type"))
        object.__setattr__(
            self,
            "counterexample_cids",
            _strings(
                self.counterexample_cids,
                "counterexample_cids",
                identifiers=True,
                required=True,
            ),
        )
        if self.label is not self.outcome:
            _refuse(
                DistillationReason.MISLABELED_EXAMPLE,
                "corpus row label does not match the bound outcome",
            )
        if not self.validation_cid or not self.proof_cid:
            _refuse(
                DistillationReason.INCOMPLETE_PROVENANCE,
                "corpus row omitted validation or proof provenance",
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "example_cid": self.example_cid,
            "partition": self.partition.value,
            "label": self.label.value,
            "request_cid": self.request_cid,
            "candidate_cid": self.candidate_cid,
            "resolution_cid": self.resolution_cid,
            "validation_cid": self.validation_cid,
            "proof_cid": self.proof_cid,
            "outcome": self.outcome.value,
            "counterexample_cids": self.counterexample_cids,
            "family_id": self.family_id,
            "language": self.language,
            "framework": self.framework,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | DistillationCorpusRow) -> DistillationCorpusRow:
        if isinstance(payload, DistillationCorpusRow):
            return payload
        if not isinstance(payload, Mapping):
            raise DistillationError("corpus row must be a mapping")
        return cls(
            example_id=payload.get("example_id", ""),
            example_cid=payload.get("example_cid", ""),
            partition=payload.get("partition", ""),
            label=payload.get("label", ""),
            request_cid=payload.get("request_cid", ""),
            candidate_cid=payload.get("candidate_cid", ""),
            resolution_cid=payload.get("resolution_cid", ""),
            validation_cid=payload.get("validation_cid", ""),
            proof_cid=payload.get("proof_cid", ""),
            outcome=payload.get("outcome", ""),
            counterexample_cids=payload.get("counterexample_cids", ()),
            family_id=payload.get("family_id", ""),
            language=payload.get("language", ""),
            framework=payload.get("framework", ""),
            hole_id=payload.get("hole_id", ""),
            hole_type=payload.get("hole_type", ""),
        )


def _corpus_rows(values: Any) -> tuple[DistillationCorpusRow, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise DistillationError("rows must be a sequence")
    if len(raw) > MAX_CORPUS_ROWS:
        _refuse(DistillationReason.CORPUS_UNBOUNDED, "corpus rows exceed the row bound")
    return tuple(DistillationCorpusRow.from_record(item) for item in raw)


def _leakage_keys(row: DistillationCorpusRow | DistillationExample) -> frozenset[tuple[str, str]]:
    return frozenset(
        {
            ("example", row.example_id),
            ("request", row.request_cid),
            ("candidate", row.candidate_cid),
            ("validation", row.validation_cid),
        }
    )


def _detect_partition_leakage(
    rows: Sequence[DistillationCorpusRow | DistillationExample],
) -> tuple[str, ...]:
    owners: dict[tuple[str, str], CorpusPartition] = {}
    leaked: list[str] = []
    for row in rows:
        partition = row.partition
        for key in _leakage_keys(row):
            previous = owners.get(key)
            if previous is not None and previous is not partition:
                leaked.append(row.example_id)
                break
            owners[key] = partition
    return tuple(dict.fromkeys(leaked))


def _partition_map(rows: Sequence[DistillationCorpusRow]) -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {name: [] for name in REQUIRED_PARTITIONS}
    for row in rows:
        grouped[row.partition.value].append(row.example_cid)
    return MappingProxyType({name: tuple(items) for name, items in grouped.items()})


def _unwrap_generic_envelope(payload: Mapping[str, Any], schema: str) -> Mapping[str, Any]:
    body = dict(payload)
    keys = set(body).difference({"schema", "contract_version", "content_id", "cid"})
    if keys and keys <= _GENERIC_ENVELOPE_FIELDS and "facts" in body:
        facts = body.get("facts")
        if not isinstance(facts, Mapping):
            raise DistillationError("generic distillation facts must be a mapping")
        merged = {
            "schema": schema,
            "contract_version": body.get("contract_version", PROCEDURE_CONTRACT_VERSION),
            "bindings": body.get("bindings"),
            "state": body.get("state", ArtifactState.CANDIDATE.value),
            **dict(facts),
        }
        if "example_id" not in merged and body.get("subject_cid"):
            merged["example_id"] = body.get("subject_cid")
        if "corpus_id" not in merged and body.get("subject_cid"):
            merged["corpus_id"] = body.get("subject_cid")
        return merged
    return body


@dataclass(frozen=True)
class DistillationExample(CanonicalContract):
    """Independently validated accepted or rejected residual-hole example."""

    SCHEMA: ClassVar[str] = _schema_name("DistillationExample")

    bindings: ArtifactBindings
    example_id: str
    hole_id: str
    hole_type: HoleType
    label: DistillationLabel
    partition: CorpusPartition
    request_cid: str
    candidate_cid: str
    resolution_cid: str
    validation_cid: str
    proof_cid: str
    outcome: DistillationLabel
    counterexample_cids: tuple[str, ...]
    family_id: str
    language: str
    framework: str
    typed_features: Mapping[str, Any] = field(default_factory=dict)
    content_references: tuple[DistillationContentReference, ...] = ()
    provider_class: str = ""
    output_digest: str = ""
    evidence_fingerprint: str = ""
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    builder_revision: str = BUILDER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "example_id",
            "hole_id",
            "request_cid",
            "candidate_cid",
            "resolution_cid",
            "validation_cid",
            "proof_cid",
            "family_id",
            "language",
            "framework",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "hole_type", _enum(self.hole_type, HoleType, "hole_type"))
        object.__setattr__(self, "label", _enum(self.label, DistillationLabel, "label"))
        object.__setattr__(
            self, "partition", _enum(self.partition, CorpusPartition, "partition")
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, DistillationLabel, "outcome")
        )
        object.__setattr__(
            self,
            "counterexample_cids",
            _strings(
                self.counterexample_cids,
                "counterexample_cids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(self, "typed_features", _typed_features(self.typed_features))
        object.__setattr__(
            self, "content_references", _content_references(self.content_references)
        )
        object.__setattr__(
            self,
            "provider_class",
            _identifier(self.provider_class, "provider_class", required=False),
        )
        object.__setattr__(
            self,
            "output_digest",
            _identifier(self.output_digest, "output_digest", required=False),
        )
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _identifier(self.evidence_fingerprint, "evidence_fingerprint", required=False),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                DistillationReason.CANDIDATE_TIER_REQUIRED,
                "distillation examples remain candidate-tier",
            )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                DistillationReason.AUTHORITY_REJECTED,
                "distillation examples cannot authorize",
            )
        object.__setattr__(
            self, "builder_revision", _identifier(self.builder_revision, "builder_revision")
        )
        if self.builder_revision != BUILDER_REVISION:
            raise DistillationError("distillation builder revision is not current")
        if self.label is not self.outcome:
            _refuse(
                DistillationReason.MISLABELED_EXAMPLE,
                "example label does not match the bound validation outcome",
            )
        if self.partition.value in _REJECTED_PARTITIONS and self.label is DistillationLabel.ACCEPTED:
            _refuse(
                DistillationReason.MISLABELED_EXAMPLE,
                "negative and adversarial partitions cannot carry accepted labels",
            )
        if not self.validation_cid:
            _refuse(DistillationReason.MISSING_VALIDATION, "example omitted validation provenance")
        if not self.proof_cid:
            _refuse(DistillationReason.MISSING_PROOF, "example omitted proof provenance")
        if not self.outcome:
            _refuse(DistillationReason.MISSING_OUTCOME, "example omitted outcome provenance")
        if not self.counterexample_cids:
            _refuse(
                DistillationReason.MISSING_COUNTEREXAMPLE,
                "example omitted required counterexamples",
            )
        _reject_privacy(self.typed_features, "typed_features")
        _bounded(self, "DistillationExample")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_skip_validation(self) -> bool:
        return False

    def to_row(self) -> DistillationCorpusRow:
        return DistillationCorpusRow(
            example_id=self.example_id,
            example_cid=self.content_id,
            partition=self.partition,
            label=self.label,
            request_cid=self.request_cid,
            candidate_cid=self.candidate_cid,
            resolution_cid=self.resolution_cid,
            validation_cid=self.validation_cid,
            proof_cid=self.proof_cid,
            outcome=self.outcome,
            counterexample_cids=self.counterexample_cids,
            family_id=self.family_id,
            language=self.language,
            framework=self.framework,
            hole_id=self.hole_id,
            hole_type=self.hole_type,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "example_id": self.example_id,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "label": self.label.value,
            "partition": self.partition.value,
            "request_cid": self.request_cid,
            "candidate_cid": self.candidate_cid,
            "resolution_cid": self.resolution_cid,
            "validation_cid": self.validation_cid,
            "proof_cid": self.proof_cid,
            "outcome": self.outcome.value,
            "counterexample_cids": self.counterexample_cids,
            "family_id": self.family_id,
            "language": self.language,
            "framework": self.framework,
            "typed_features": dict(self.typed_features),
            "content_references": tuple(item.to_record() for item in self.content_references),
            "provider_class": self.provider_class,
            "output_digest": self.output_digest,
            "evidence_fingerprint": self.evidence_fingerprint,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "builder_revision": BUILDER_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationExample:
        if not isinstance(payload, Mapping):
            raise DistillationError("DistillationExample payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "example_id",
            "hole_id",
            "hole_type",
            "label",
            "partition",
            "request_cid",
            "candidate_cid",
            "resolution_cid",
            "validation_cid",
            "proof_cid",
            "outcome",
            "counterexample_cids",
            "family_id",
            "language",
            "framework",
            "typed_features",
            "content_references",
            "provider_class",
            "output_digest",
            "evidence_fingerprint",
            "state",
            "can_authorize",
            "builder_revision",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class DistillationCorpus(CanonicalContract):
    """Compact partition manifest of admitted residual-hole examples."""

    SCHEMA: ClassVar[str] = _schema_name("DistillationCorpus")

    bindings: ArtifactBindings
    corpus_id: str
    rows: tuple[DistillationCorpusRow, ...]
    example_cids: tuple[str, ...] = ()
    partition_example_cids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    family_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    accepted_count: int = 0
    rejected_count: int = 0
    disjoint: bool = True
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    builder_revision: str = BUILDER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "corpus_id", _identifier(self.corpus_id, "corpus_id"))
        rows = _corpus_rows(self.rows)
        object.__setattr__(self, "rows", rows)
        if not rows:
            raise DistillationError("distillation corpus must contain at least one row")
        leaked = _detect_partition_leakage(rows)
        if leaked:
            _refuse(
                DistillationReason.PARTITION_LEAKAGE,
                "corpus partitions are not disjoint",
            )
        derived_cids = tuple(row.example_cid for row in rows)
        supplied = self.example_cids or derived_cids
        object.__setattr__(
            self,
            "example_cids",
            _strings(supplied, "example_cids", identifiers=True, required=True),
        )
        if self.example_cids != derived_cids:
            raise DistillationError("example_cids must match the compact row order")
        partition_map = self.partition_example_cids or _partition_map(rows)
        frozen_partitions = _freeze(dict(partition_map), "partition_example_cids")
        if not isinstance(frozen_partitions, Mapping):
            raise DistillationError("partition_example_cids must be a mapping")
        if set(frozen_partitions) != set(REQUIRED_PARTITIONS):
            raise DistillationError("corpus must declare the closed partition vocabulary")
        normalized: dict[str, tuple[str, ...]] = {}
        for name in REQUIRED_PARTITIONS:
            normalized[name] = _strings(
                frozen_partitions[name],
                "partition_example_cids",
                identifiers=True,
            )
        expected = _partition_map(rows)
        if any(normalized[name] != expected[name] for name in REQUIRED_PARTITIONS):
            raise DistillationError("partition manifest does not match admitted rows")
        object.__setattr__(self, "partition_example_cids", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "family_ids",
            _strings(
                self.family_ids or tuple(row.family_id for row in rows),
                "family_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "languages",
            _strings(
                self.languages or tuple(row.language for row in rows),
                "languages",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "frameworks",
            _strings(
                self.frameworks or tuple(row.framework for row in rows),
                "frameworks",
                identifiers=True,
                required=True,
            ),
        )
        accepted = sum(1 for row in rows if row.label is DistillationLabel.ACCEPTED)
        rejected = sum(1 for row in rows if row.label is DistillationLabel.REJECTED)
        supplied_accepted = self.accepted_count
        supplied_rejected = self.rejected_count
        if supplied_accepted not in {0, accepted} or supplied_rejected not in {0, rejected}:
            raise DistillationError("corpus label counts do not match admitted rows")
        object.__setattr__(self, "accepted_count", accepted)
        object.__setattr__(self, "rejected_count", rejected)
        object.__setattr__(self, "disjoint", _bool(self.disjoint, "disjoint"))
        if not self.disjoint:
            _refuse(
                DistillationReason.PARTITION_LEAKAGE,
                "a distillation corpus cannot claim overlapping partitions",
            )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                DistillationReason.CANDIDATE_TIER_REQUIRED,
                "distillation corpora remain candidate-tier",
            )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                DistillationReason.AUTHORITY_REJECTED,
                "distillation corpora cannot authorize",
            )
        object.__setattr__(
            self, "builder_revision", _identifier(self.builder_revision, "builder_revision")
        )
        if self.builder_revision != BUILDER_REVISION:
            raise DistillationError("distillation builder revision is not current")
        _bounded(self, "DistillationCorpus")

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
            "corpus_id": self.corpus_id,
            "rows": tuple(item.to_record() for item in self.rows),
            "example_cids": self.example_cids,
            "partition_example_cids": {
                name: self.partition_example_cids[name] for name in REQUIRED_PARTITIONS
            },
            "family_ids": self.family_ids,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "disjoint": True,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "builder_revision": BUILDER_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationCorpus:
        if not isinstance(payload, Mapping):
            raise DistillationError("DistillationCorpus payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "corpus_id",
            "rows",
            "example_cids",
            "partition_example_cids",
            "family_ids",
            "languages",
            "frameworks",
            "accepted_count",
            "rejected_count",
            "disjoint",
            "state",
            "can_authorize",
            "builder_revision",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class DistillationEvaluation(CanonicalContract):
    """Independent disjointness and provenance check of one admitted corpus."""

    SCHEMA: ClassVar[str] = _schema_name("DistillationEvaluation")

    bindings: ArtifactBindings
    corpus_cid: str
    disjoint: bool
    complete_provenance: bool
    admitted_count: int
    rejected_admission_count: int
    partition_counts: Mapping[str, int]
    leakage_example_ids: tuple[str, ...] = ()
    reason_code: DistillationReason = DistillationReason.ADMITTED
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    evaluator_revision: str = EVALUATOR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "corpus_cid", _identifier(self.corpus_cid, "corpus_cid"))
        object.__setattr__(self, "disjoint", _bool(self.disjoint, "disjoint"))
        object.__setattr__(
            self,
            "complete_provenance",
            _bool(self.complete_provenance, "complete_provenance"),
        )
        object.__setattr__(
            self,
            "admitted_count",
            _nonnegative_int(self.admitted_count, "admitted_count", maximum=MAX_CORPUS_ROWS),
        )
        object.__setattr__(
            self,
            "rejected_admission_count",
            _nonnegative_int(self.rejected_admission_count, "rejected_admission_count"),
        )
        counts = _freeze(self.partition_counts, "partition_counts")
        if not isinstance(counts, Mapping):
            raise DistillationError("partition_counts must be a mapping")
        normalized: dict[str, int] = {}
        for name in REQUIRED_PARTITIONS:
            normalized[name] = _nonnegative_int(
                counts.get(name, 0), f"partition_counts.{name}", maximum=MAX_CORPUS_ROWS
            )
        object.__setattr__(self, "partition_counts", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "leakage_example_ids",
            _strings(self.leakage_example_ids, "leakage_example_ids", identifiers=True),
        )
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, DistillationReason, "reason_code")
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                DistillationReason.CANDIDATE_TIER_REQUIRED,
                "distillation evaluations remain candidate-tier",
            )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                DistillationReason.AUTHORITY_REJECTED,
                "distillation evaluations cannot authorize",
            )
        object.__setattr__(
            self,
            "evaluator_revision",
            _identifier(self.evaluator_revision, "evaluator_revision"),
        )
        if self.evaluator_revision != EVALUATOR_REVISION:
            raise DistillationError("distillation evaluator revision is not current")
        if self.leakage_example_ids:
            if self.disjoint:
                raise DistillationError("leaked evaluation cannot claim disjoint partitions")
            if self.reason_code is DistillationReason.ADMITTED:
                raise DistillationError("leaked evaluation cannot claim admission")
        elif self.reason_code is DistillationReason.ADMITTED and (
            not self.disjoint or not self.complete_provenance
        ):
            raise DistillationError("successful evaluation must be disjoint with complete provenance")
        _bounded(self, "DistillationEvaluation")

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
            "corpus_cid": self.corpus_cid,
            "disjoint": self.disjoint,
            "complete_provenance": self.complete_provenance,
            "admitted_count": self.admitted_count,
            "rejected_admission_count": self.rejected_admission_count,
            "partition_counts": dict(self.partition_counts),
            "leakage_example_ids": self.leakage_example_ids,
            "reason_code": self.reason_code.value,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "evaluator_revision": EVALUATOR_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationEvaluation:
        if not isinstance(payload, Mapping):
            raise DistillationError("DistillationEvaluation payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "corpus_cid",
            "disjoint",
            "complete_provenance",
            "admitted_count",
            "rejected_admission_count",
            "partition_counts",
            "leakage_example_ids",
            "reason_code",
            "state",
            "can_authorize",
            "evaluator_revision",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


def _proof_bound(
    proof_cid: str,
    request: HoleRequest,
    receipt: HoleValidationReceipt,
    references: Sequence[DistillationContentReference],
) -> bool:
    referenced = {item.content_id for item in references}
    return proof_cid in (
        set(request.validation_observation_ids)
        | set(receipt.observation_ids)
        | referenced
    )


def _assemble_references(
    request: HoleRequest,
    candidate: HoleCandidate,
    resolution: HoleResolution,
    receipt: HoleValidationReceipt,
    proof_cid: str,
    counterexample_cids: Sequence[str],
    extra: Sequence[DistillationContentReference | Mapping[str, Any] | HoleContextReference] = (),
) -> tuple[DistillationContentReference, ...]:
    records: list[DistillationContentReference] = []
    for item in request.context_references:
        records.append(DistillationContentReference.from_record(item))
    records.extend(
        [
            DistillationContentReference(
                reference_id="candidate",
                content_id=candidate.content_id,
                kind=DistillationReferenceKind.CANDIDATE,
                tree_id=request.bindings.tree_id,
            ),
            DistillationContentReference(
                reference_id="resolution",
                content_id=resolution.content_id,
                kind=DistillationReferenceKind.RESOLUTION,
                tree_id=request.bindings.tree_id,
            ),
            DistillationContentReference(
                reference_id="validation",
                content_id=receipt.content_id,
                kind=DistillationReferenceKind.VALIDATION,
                tree_id=request.bindings.tree_id,
            ),
            DistillationContentReference(
                reference_id="proof",
                content_id=proof_cid,
                kind=DistillationReferenceKind.PROOF,
                tree_id=request.bindings.tree_id,
            ),
        ]
    )
    for index, cid in enumerate(counterexample_cids):
        records.append(
            DistillationContentReference(
                reference_id=f"counterexample.{index + 1}",
                content_id=cid,
                kind=DistillationReferenceKind.COUNTEREXAMPLE,
                tree_id=request.bindings.tree_id,
            )
        )
    for item in extra:
        records.append(DistillationContentReference.from_record(item))
    seen: set[str] = set()
    unique: list[DistillationContentReference] = []
    for item in records:
        if item.reference_id in seen:
            continue
        seen.add(item.reference_id)
        unique.append(item)
    return tuple(unique)


class DistillationCorpusBuilder:
    """Admit independently validated hole examples into disjoint compact rows."""

    revision: ClassVar[str] = BUILDER_REVISION

    def __init__(
        self,
        *,
        bindings: ArtifactBindings,
        corpus_id: str,
        current_tree_id: str,
        validator: HoleResolutionValidator | None = None,
    ) -> None:
        self._bindings = _bindings(bindings)
        self._corpus_id = _identifier(corpus_id, "corpus_id")
        self._current_tree_id = _identifier(current_tree_id, "current_tree_id")
        self._validator = validator or HoleResolutionValidator(
            current_tree_id=self._current_tree_id
        )
        self._examples: list[DistillationExample] = []
        self._rejected_count = 0
        self._evaluation: DistillationEvaluation | None = None

    @property
    def examples(self) -> tuple[DistillationExample, ...]:
        return tuple(self._examples)

    @property
    def rejected_admission_count(self) -> int:
        return self._rejected_count

    @property
    def evaluation(self) -> DistillationEvaluation | None:
        return self._evaluation

    def admit(
        self,
        *,
        request: HoleRequest,
        candidate: HoleCandidate,
        resolution: HoleResolution,
        receipt: HoleValidationReceipt,
        partition: CorpusPartition | str,
        proof_cid: str,
        counterexample_cids: Sequence[str],
        family_id: str,
        language: str,
        framework: str,
        label: DistillationLabel | str | None = None,
        example_id: str = "",
        typed_features: Mapping[str, Any] | None = None,
        content_references: Sequence[
            DistillationContentReference | Mapping[str, Any] | HoleContextReference
        ] = (),
    ) -> DistillationExample:
        try:
            example = self._admit_validated(
                request=request,
                candidate=candidate,
                resolution=resolution,
                receipt=receipt,
                partition=partition,
                proof_cid=proof_cid,
                counterexample_cids=counterexample_cids,
                family_id=family_id,
                language=language,
                framework=framework,
                label=label,
                example_id=example_id,
                typed_features=typed_features,
                content_references=content_references,
            )
        except DistillationAdmissionError:
            self._rejected_count += 1
            raise
        self._examples.append(example)
        self._evaluation = None
        return example

    def build(self) -> DistillationCorpus:
        if not self._examples:
            _refuse(
                DistillationReason.INCOMPLETE_PROVENANCE,
                "distillation corpus has no independently validated examples",
            )
        leaked = _detect_partition_leakage(self._examples)
        if leaked:
            _refuse(
                DistillationReason.PARTITION_LEAKAGE,
                "admitted examples leak across partitions",
            )
        rows = tuple(item.to_row() for item in self._examples)
        corpus = DistillationCorpus(
            bindings=self._bindings,
            corpus_id=self._corpus_id,
            rows=rows,
        )
        counts = {
            name: len(corpus.partition_example_cids[name]) for name in REQUIRED_PARTITIONS
        }
        self._evaluation = DistillationEvaluation(
            bindings=self._bindings,
            corpus_cid=corpus.content_id,
            disjoint=True,
            complete_provenance=True,
            admitted_count=len(rows),
            rejected_admission_count=self._rejected_count,
            partition_counts=counts,
            leakage_example_ids=(),
            reason_code=DistillationReason.ADMITTED,
        )
        return corpus

    def _admit_validated(
        self,
        *,
        request: HoleRequest,
        candidate: HoleCandidate,
        resolution: HoleResolution,
        receipt: HoleValidationReceipt,
        partition: CorpusPartition | str,
        proof_cid: str,
        counterexample_cids: Sequence[str],
        family_id: str,
        language: str,
        framework: str,
        label: DistillationLabel | str | None,
        example_id: str,
        typed_features: Mapping[str, Any] | None,
        content_references: Sequence[
            DistillationContentReference | Mapping[str, Any] | HoleContextReference
        ],
    ) -> DistillationExample:
        if not isinstance(request, HoleRequest):
            _refuse(DistillationReason.UNVERIFIED_EXAMPLE, "admission requires a HoleRequest")
        if not isinstance(candidate, HoleCandidate):
            _refuse(DistillationReason.UNVERIFIED_EXAMPLE, "admission requires a HoleCandidate")
        if not isinstance(resolution, HoleResolution):
            _refuse(DistillationReason.UNVERIFIED_EXAMPLE, "admission requires a HoleResolution")
        if not isinstance(receipt, HoleValidationReceipt):
            _refuse(
                DistillationReason.MISSING_VALIDATION,
                "admission requires an independent HoleValidationReceipt",
            )
        if len(self._examples) >= MAX_CORPUS_ROWS:
            _refuse(DistillationReason.CORPUS_UNBOUNDED, "corpus exceeds its row bound")

        normalized_partition = _enum(partition, CorpusPartition, "partition")
        proof = _identifier(proof_cid, "proof_cid")
        if not counterexample_cids:
            _refuse(
                DistillationReason.MISSING_COUNTEREXAMPLE,
                "example omitted required counterexamples",
            )
        counters = _strings(
            counterexample_cids, "counterexample_cids", identifiers=True, required=True
        )
        family = _identifier(family_id, "family_id")
        lang = _identifier(language, "language")
        frame = _identifier(framework, "framework")
        assigned_id = (
            _identifier(example_id, "example_id")
            if example_id
            else _identifier(
                f"ex.{normalized_partition.value}.{request.hole_id}.{len(self._examples) + 1}",
                "example_id",
            )
        )

        try:
            _reject_privacy(request.input_payload, "input_payload")
            _reject_privacy(candidate.output, "candidate.output")
            _reject_privacy(dict(typed_features or {}), "typed_features")
            _reject_privacy({"proof_cid": proof, "example_id": assigned_id}, "provenance")
        except DistillationAdmissionError:
            raise
        except ProcedureSafetyError as exc:
            _refuse(DistillationReason.PRIVATE_EXAMPLE, str(exc))

        if request.bindings != self._bindings:
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "hole request bindings do not match the corpus bindings",
            )
        if (
            candidate.bindings != request.bindings
            or resolution.bindings != request.bindings
            or receipt.bindings != request.bindings
        ):
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "hole artifacts are not bound to the same authority roots",
            )
        if (
            candidate.request_cid != request.content_id
            or resolution.request_cid != request.content_id
            or receipt.request_cid != request.content_id
        ):
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "hole artifacts are not bound to the same request",
            )
        if (
            candidate.hole_id != request.hole_id
            or resolution.hole_id != request.hole_id
            or receipt.hole_id != request.hole_id
            or candidate.hole_type is not request.hole_type
        ):
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "hole artifacts do not share the typed hole identity",
            )
        if resolution.candidate_cid != candidate.content_id:
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "resolution is not bound to the hole candidate",
            )
        if receipt.candidate_cid != candidate.content_id:
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "validation receipt is not bound to the hole candidate",
            )

        tree_id = self._current_tree_id
        if tree_id != request.bindings.tree_id:
            _refuse(DistillationReason.STALE_EXAMPLE, "example is bound to a stale tree")
        stale = self._validator.check_freshness(request, current_tree_id=tree_id)
        if stale is not None:
            _refuse(DistillationReason.STALE_EXAMPLE, "example context is stale")
        for item in request.context_references:
            if item.tree_id and item.tree_id != tree_id:
                _refuse(DistillationReason.STALE_EXAMPLE, "content reference tree is stale")

        if receipt.validator_revision != HOLE_VALIDATOR_REVISION:
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "validation receipt is not from the current independent validator",
            )
        independent = self._validator.validate_candidate(
            request,
            candidate,
            current_tree_id=tree_id,
            observations=receipt.observation_ids,
        )
        if (
            independent.accepted != receipt.accepted
            or independent.reason_code is not receipt.reason_code
            or independent.content_id != receipt.content_id
        ):
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "supplied validation receipt is not independently reproducible",
            )
        if resolution.action is not HoleResolutionAction.PROPOSE:
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "only independently routed hole proposals can enter the corpus",
            )
        if candidate.state is not ArtifactState.CANDIDATE or candidate.validated:
            _refuse(
                DistillationReason.CANDIDATE_TIER_REQUIRED,
                "self-validated or promoted hole outputs cannot enter the corpus",
            )

        observed_label = _label_from_accepted(independent.accepted)
        declared_label = (
            observed_label if label is None else _enum(label, DistillationLabel, "label")
        )
        if declared_label is not observed_label:
            _refuse(
                DistillationReason.MISLABELED_EXAMPLE,
                "corpus label does not match the independent validation outcome",
            )
        if (
            normalized_partition.value in _REJECTED_PARTITIONS
            and declared_label is DistillationLabel.ACCEPTED
        ):
            _refuse(
                DistillationReason.MISLABELED_EXAMPLE,
                "negative and adversarial partitions require rejected labels",
            )

        try:
            features = (
                _default_features(request, candidate)
                if typed_features is None
                else _typed_features(typed_features)
            )
        except DistillationAdmissionError:
            raise
        except ProcedureSafetyError as exc:
            _refuse(DistillationReason.PRIVATE_EXAMPLE, str(exc))
        extra_refs = _content_references(content_references)
        if not _proof_bound(proof, request, receipt, extra_refs):
            _refuse(
                DistillationReason.MISSING_PROOF,
                "proof identity is not bound to validation observations or content references",
            )
        references = _assemble_references(
            request,
            candidate,
            resolution,
            receipt,
            proof,
            counters,
            extra=extra_refs,
        )

        example = DistillationExample(
            bindings=request.bindings,
            example_id=assigned_id,
            hole_id=request.hole_id,
            hole_type=request.hole_type,
            label=declared_label,
            partition=normalized_partition,
            request_cid=request.content_id,
            candidate_cid=candidate.content_id,
            resolution_cid=resolution.content_id,
            validation_cid=receipt.content_id,
            proof_cid=proof,
            outcome=observed_label,
            counterexample_cids=counters,
            family_id=family,
            language=lang,
            framework=frame,
            typed_features=features,
            content_references=references,
            provider_class=candidate.provider_class.value,
            output_digest=candidate.output_digest,
            evidence_fingerprint=candidate.evidence_fingerprint,
        )
        leaked = _detect_partition_leakage((*self._examples, example))
        if leaked:
            _refuse(
                DistillationReason.PARTITION_LEAKAGE,
                "example identity overlaps another partition",
            )
        if any(item.example_id == example.example_id for item in self._examples):
            _refuse(DistillationReason.DUPLICATE_EXAMPLE, "example_id is already admitted")
        if any(item.candidate_cid == example.candidate_cid for item in self._examples):
            _refuse(
                DistillationReason.DUPLICATE_EXAMPLE,
                "candidate is already admitted to this corpus",
            )
        return example


EXACT_HOLE_CACHE_REVISION: Final[str] = "ExactHoleCache@1"
DECLARATIVE_HOLE_RULE_REVISION: Final[str] = "DeclarativeHoleRule@1"
DETERMINISTIC_HOLE_CLASSIFIER_REVISION: Final[str] = "DeterministicHoleClassifier@1"
LOCAL_HOLE_RESOLVER_REVISION: Final[str] = "LocalHoleResolver@1"
MAX_RESOLVER_ENTRIES: Final[int] = MAX_ITEMS
MAX_CONFIDENCE_MILLIS: Final[int] = 1000
TRAINABLE_PARTITIONS: Final[frozenset[CorpusPartition]] = frozenset(
    {CorpusPartition.TRAINING}
)
REQUIRED_CACHE_KEY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_commit",
    "tree_id",
    "objective_id",
    "task_id",
    "contract_revision",
    "policy_revision",
    "environment_id",
    "hole_type",
    "input_schema_ref",
    "output_schema_ref",
    "input_payload",
    "context_reference_ids",
    "context_content_ids",
    "evidence_fingerprint",
    "validation_observation_ids",
)


class DeclarativeHoleRuleKind(str, Enum):
    SELECT_SINGLETON = "select-singleton"
    EXACT_OUTPUT = "exact-output"
    CLOSED_MAP = "closed-map"


def _miss(code: str) -> HoleProviderResult:
    return HoleProviderResult(outcome=HoleProviderOutcome.MISSED, failure_code=code)


def _propose(
    output: Mapping[str, Any], *, token_count: int = 0
) -> HoleProviderResult:
    return HoleProviderResult(
        outcome=HoleProviderOutcome.PROPOSED,
        output=dict(output),
        token_count=token_count,
    )


def _non_authority_guard(owner: object) -> None:
    if getattr(owner, "can_skip_validation", False):
        _refuse(
            DistillationReason.AUTHORITY_REJECTED,
            "hole resolvers cannot skip validation",
        )
    if getattr(owner, "can_authorize", False):
        _refuse(
            DistillationReason.AUTHORITY_REJECTED,
            "hole resolvers cannot authorize",
        )
    if getattr(owner, "claims_correctness", False) or getattr(
        owner, "can_claim_correctness", False
    ):
        _refuse(
            DistillationReason.CONFIDENCE_AUTHORITY_REJECTED,
            "hole resolvers cannot claim correctness",
        )


def _require_training_partition(partition: CorpusPartition | str) -> CorpusPartition:
    normalized = _enum(partition, CorpusPartition, "partition")
    if normalized not in TRAINABLE_PARTITIONS:
        _refuse(
            DistillationReason.HELD_OUT_TRAINING_REJECTED,
            "held-out and rejected partitions cannot train a hole resolver",
        )
    return normalized


def _require_accepted(label: DistillationLabel | str | bool) -> None:
    if type(label) is bool:
        accepted = label
    else:
        accepted = _enum(label, DistillationLabel, "label") is DistillationLabel.ACCEPTED
    if not accepted:
        _refuse(
            DistillationReason.MISLABELED_EXAMPLE,
            "rejected examples cannot train a hole resolver",
        )


def _output_compatible(request: HoleRequest, output: Mapping[str, Any]) -> bool:
    return HoleResolutionValidator().check_output_schema(request, output) is None


def _compact_output(
    request: HoleRequest,
    output: Mapping[str, Any] | None = None,
    features: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {"schema_ref": request.output_schema_ref}
    source = dict(output or {})
    feature_map = dict(features or {})
    for key in _COMPACT_OUTPUT_KEYS:
        if key == "schema_ref":
            continue
        if key in source and type(source[key]) is str:
            payload[key] = source[key]
        elif key in feature_map and type(feature_map[key]) is str:
            payload[key] = feature_map[key]
    if "schema_ref" in source and type(source["schema_ref"]) is str:
        payload["schema_ref"] = source["schema_ref"]
    _reject_privacy(payload, "resolver.output")
    return MappingProxyType(payload)


def _request_features(request: HoleRequest) -> Mapping[str, Any]:
    features: dict[str, Any] = {
        "hole_type": request.hole_type.value,
        "input_schema_ref": request.input_schema_ref,
        "output_schema_ref": request.output_schema_ref,
    }
    payload = dict(request.input_payload)
    for key, item in payload.items():
        if type(item) is str:
            try:
                features[key] = _identifier(item, key)
            except ProcedureContractError:
                continue
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            features[f"{key}_count"] = len(item)
            if len(item) == 1 and type(item[0]) is str:
                try:
                    features[f"{key}_only"] = _identifier(item[0], key)
                except ProcedureContractError:
                    continue
    return MappingProxyType(features)


def _classifier_signature(request: HoleRequest) -> str:
    return content_identity(
        {
            "hole_type": request.hole_type.value,
            "input_schema_ref": request.input_schema_ref,
            "output_schema_ref": request.output_schema_ref,
            "input_payload": _freeze(dict(request.input_payload), "input_payload"),
        }
    )


def _output_digest(output: Mapping[str, Any]) -> str:
    return content_identity({"output": dict(output)})


def _cache_key_payload(
    request: HoleRequest,
    *,
    compiled: CompiledHoleContext | None = None,
    evidence: str = "",
) -> dict[str, Any]:
    # Receipt CIDs are compiler artifacts, not hole identity.  Exact cache keys
    # bind request, bindings, context bodies, and a receipt-free fingerprint.
    if compiled is not None:
        if compiled.tree_id and compiled.tree_id != request.bindings.tree_id:
            _refuse(
                DistillationReason.STALE_EXAMPLE,
                "compiled hole context tree does not match the request",
            )
        if (
            compiled.repository_id
            and compiled.repository_id != request.bindings.repository_id
        ):
            _refuse(
                DistillationReason.STALE_EXAMPLE,
                "compiled hole context repository does not match the request",
            )
    for item in request.context_references:
        if item.tree_id and item.tree_id != request.bindings.tree_id:
            _refuse(
                DistillationReason.STALE_EXAMPLE,
                "context reference tree does not match the request",
            )
    fingerprint = evidence or evidence_fingerprint(request)
    payload = {
        "repository_id": request.bindings.repository_id,
        "repository_commit": request.bindings.repository_commit,
        "tree_id": request.bindings.tree_id,
        "objective_id": request.bindings.objective_id,
        "task_id": request.bindings.task_id,
        "contract_revision": request.bindings.contract_revision,
        "policy_revision": request.bindings.policy_revision,
        "environment_id": request.bindings.environment_id,
        "hole_type": request.hole_type.value,
        "input_schema_ref": request.input_schema_ref,
        "output_schema_ref": request.output_schema_ref,
        "input_payload": _freeze(dict(request.input_payload), "input_payload"),
        "context_reference_ids": tuple(
            item.reference_id for item in request.context_references
        ),
        "context_content_ids": tuple(
            item.content_id for item in request.context_references
        ),
        "evidence_fingerprint": fingerprint,
        "validation_observation_ids": request.validation_observation_ids,
    }
    missing = tuple(
        name
        for name in REQUIRED_CACHE_KEY_FIELDS
        if payload.get(name) in (None, "", (), {})
    )
    if missing:
        _refuse(
            DistillationReason.INCOMPLETE_CACHE_KEY,
            "exact hole cache key omitted required identity dimensions",
        )
    return payload


@dataclass(frozen=True)
class ExactHoleCacheEntry:
    """Stored previously validated proposal.  Replay remains a candidate."""

    cache_key: str
    output: Mapping[str, Any]
    example_id: str = ""
    candidate_cid: str = ""
    validation_cid: str = ""
    partition: CorpusPartition = CorpusPartition.TRAINING

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_key", _identifier(self.cache_key, "cache_key"))
        object.__setattr__(self, "output", _payload_mapping_proxy(self.output, "output"))
        object.__setattr__(
            self, "example_id", _identifier(self.example_id, "example_id", required=False)
        )
        object.__setattr__(
            self,
            "candidate_cid",
            _identifier(self.candidate_cid, "candidate_cid", required=False),
        )
        object.__setattr__(
            self,
            "validation_cid",
            _identifier(self.validation_cid, "validation_cid", required=False),
        )
        object.__setattr__(
            self, "partition", _enum(self.partition, CorpusPartition, "partition")
        )
        if self.partition not in TRAINABLE_PARTITIONS:
            _refuse(
                DistillationReason.HELD_OUT_TRAINING_REJECTED,
                "exact hole cache cannot store held-out rows",
            )
        if not self.validation_cid:
            _refuse(
                DistillationReason.MISSING_VALIDATION,
                "exact hole cache entries require independent validation provenance",
            )


def _payload_mapping_proxy(value: Any, field_name: str) -> Mapping[str, Any]:
    frozen = _freeze(value if value is not None else {}, field_name)
    if not isinstance(frozen, Mapping):
        raise DistillationError(f"{field_name} must be a mapping")
    _reject_privacy(frozen, field_name)
    return MappingProxyType(dict(frozen))


class ExactHoleCache:
    """Exact-identity proposal producer.  Hits never skip validation."""

    provider_class: ClassVar[ProviderClass] = ProviderClass.EXACT_CACHE
    revision: ClassVar[str] = EXACT_HOLE_CACHE_REVISION

    def __init__(self) -> None:
        self._entries: dict[str, ExactHoleCacheEntry] = {}
        self.calls = 0

    @property
    def can_skip_validation(self) -> bool:
        return False

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def claims_correctness(self) -> bool:
        return False

    def cache_key(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext | None = None,
        *,
        evidence_fingerprint: str = "",
    ) -> str:
        payload = _cache_key_payload(
            request, compiled=compiled, evidence=evidence_fingerprint
        )
        return content_identity(payload)

    def remember(
        self,
        request: HoleRequest,
        output: Mapping[str, Any],
        *,
        compiled: CompiledHoleContext | None = None,
        evidence_fingerprint: str = "",
        partition: CorpusPartition | str = CorpusPartition.TRAINING,
        accepted: bool = True,
        example_id: str = "",
        candidate_cid: str = "",
        validation_cid: str = "",
    ) -> str:
        _non_authority_guard(self)
        _require_accepted(accepted)
        normalized_partition = _require_training_partition(partition)
        if not validation_cid:
            _refuse(
                DistillationReason.MISSING_VALIDATION,
                "exact hole cache entries require independent validation provenance",
            )
        if len(self._entries) >= MAX_RESOLVER_ENTRIES:
            _refuse(DistillationReason.CORPUS_UNBOUNDED, "exact hole cache exceeds its bound")
        compact = dict(_compact_output(request, output))
        if not _output_compatible(request, compact):
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "exact hole cache cannot store a schema-invalid proposal",
            )
        key = self.cache_key(
            request, compiled, evidence_fingerprint=evidence_fingerprint
        )
        self._entries[key] = ExactHoleCacheEntry(
            cache_key=key,
            output=compact,
            example_id=example_id,
            candidate_cid=candidate_cid,
            validation_cid=validation_cid,
            partition=normalized_partition,
        )
        return key

    def ingest_example(
        self,
        example: DistillationExample,
        request: HoleRequest,
        output: Mapping[str, Any],
        *,
        compiled: CompiledHoleContext | None = None,
    ) -> str:
        if example.request_cid != request.content_id:
            _refuse(
                DistillationReason.BINDING_MISMATCH,
                "cache example is not bound to the hole request",
            )
        return self.remember(
            request,
            output,
            compiled=compiled,
            partition=example.partition,
            accepted=example.label is DistillationLabel.ACCEPTED,
            example_id=example.example_id,
            candidate_cid=example.candidate_cid,
            validation_cid=example.validation_cid,
        )

    def lookup(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext | None = None,
    ) -> ExactHoleCacheEntry | None:
        try:
            key = self.cache_key(request, compiled)
        except DistillationAdmissionError:
            return None
        return self._entries.get(key)

    def propose(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        self.calls += 1
        _non_authority_guard(self)
        entry = self.lookup(request, compiled)
        if entry is None:
            return _miss("cache-miss")
        output = dict(entry.output)
        if not _output_compatible(request, output):
            return _miss("cache-incompatible")
        return _propose(output)


@dataclass(frozen=True)
class DeclarativeHoleRule:
    """Closed deterministic feature rule.  A miss never claims correctness."""

    rule_id: str
    hole_type: HoleType
    kind: DeclarativeHoleRuleKind = DeclarativeHoleRuleKind.EXACT_OUTPUT
    match: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    input_field: str = ""
    output_field: str = ""
    mapping: Mapping[str, str] = field(default_factory=dict)

    provider_class: ClassVar[ProviderClass] = ProviderClass.DECLARATIVE_RULE
    revision: ClassVar[str] = DECLARATIVE_HOLE_RULE_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _identifier(self.rule_id, "rule_id"))
        object.__setattr__(self, "hole_type", _enum(self.hole_type, HoleType, "hole_type"))
        object.__setattr__(
            self, "kind", _enum(self.kind, DeclarativeHoleRuleKind, "kind")
        )
        object.__setattr__(self, "match", _payload_mapping_proxy(self.match, "match"))
        object.__setattr__(self, "output", _payload_mapping_proxy(self.output, "output"))
        object.__setattr__(
            self, "input_field", _identifier(self.input_field, "input_field", required=False)
        )
        object.__setattr__(
            self,
            "output_field",
            _identifier(self.output_field, "output_field", required=False),
        )
        frozen_map = _payload_mapping_proxy(self.mapping, "mapping")
        normalized_map: dict[str, str] = {}
        for key, value in frozen_map.items():
            if type(value) is not str:
                raise DistillationError("rule mapping values must be identifiers")
            normalized_map[_identifier(key, "mapping")] = _identifier(value, "mapping")
        object.__setattr__(self, "mapping", MappingProxyType(normalized_map))
        if self.kind is DeclarativeHoleRuleKind.CLOSED_MAP and (
            not self.input_field or not self.output_field or not self.mapping
        ):
            raise DistillationError("closed-map rules require input, output, and mapping")
        _reject_privacy(self.match, "match")
        _reject_privacy(self.output, "output")

    @property
    def can_skip_validation(self) -> bool:
        return False

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def claims_correctness(self) -> bool:
        return False

    def match_output(self, request: HoleRequest) -> Mapping[str, Any] | None:
        if request.hole_type is not self.hole_type:
            return None
        features = _request_features(request)
        for key, expected in self.match.items():
            actual = features.get(key, request.input_payload.get(key))
            if actual != expected:
                return None
        if self.kind is DeclarativeHoleRuleKind.SELECT_SINGLETON:
            values = request.input_payload.get("allowed_values")
            if values is None:
                values = request.input_payload.get("template_ids")
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray, memoryview)
            ):
                return None
            if len(values) != 1 or type(values[0]) is not str:
                return None
            field = self.output_field or {
                HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS: "selected",
                HoleType.PROPOSE_BOUNDED_PATCH: "template_id",
                HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE: "template_id",
            }.get(request.hole_type)
            if not field:
                return None
            output = {"schema_ref": request.output_schema_ref, field: values[0]}
            return output if _output_compatible(request, output) else None
        if self.kind is DeclarativeHoleRuleKind.CLOSED_MAP:
            raw = request.input_payload.get(self.input_field)
            if type(raw) is not str or raw not in self.mapping:
                return None
            output = {
                "schema_ref": request.output_schema_ref,
                self.output_field: self.mapping[raw],
            }
            return output if _output_compatible(request, output) else None
        output = {"schema_ref": request.output_schema_ref, **dict(self.output)}
        return output if _output_compatible(request, output) else None

    def propose(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        _non_authority_guard(self)
        output = self.match_output(request)
        if output is None:
            return _miss("rule-miss")
        return _propose(output)


class _DeclarativeRuleProvider:
    """Unique-match bundle over declarative rules."""

    provider_class: ClassVar[ProviderClass] = ProviderClass.DECLARATIVE_RULE

    def __init__(self, rules: Sequence[DeclarativeHoleRule] = ()) -> None:
        self._rules = tuple(sorted(rules, key=lambda item: item.rule_id))
        self.calls = 0

    @property
    def rules(self) -> tuple[DeclarativeHoleRule, ...]:
        return self._rules

    @property
    def can_skip_validation(self) -> bool:
        return False

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def claims_correctness(self) -> bool:
        return False

    def propose(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        self.calls += 1
        hits: list[Mapping[str, Any]] = []
        for rule in self._rules:
            output = rule.match_output(request)
            if output is not None:
                hits.append(output)
        if len(hits) != 1:
            return _miss("rule-miss" if not hits else "rule-ambiguous")
        return _propose(hits[0])


class DeterministicHoleClassifier:
    """Exact typed-feature classifier trained only on disjoint training rows."""

    provider_class: ClassVar[ProviderClass] = ProviderClass.DETERMINISTIC_CLASSIFIER
    revision: ClassVar[str] = DETERMINISTIC_HOLE_CLASSIFIER_REVISION

    def __init__(self) -> None:
        self._outputs: dict[str, dict[str, Mapping[str, Any]]] = {}
        self.calls = 0

    @property
    def can_skip_validation(self) -> bool:
        return False

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def claims_correctness(self) -> bool:
        return False

    def ingest(
        self,
        request: HoleRequest,
        output: Mapping[str, Any],
        *,
        partition: CorpusPartition | str = CorpusPartition.TRAINING,
        accepted: bool = True,
    ) -> None:
        _non_authority_guard(self)
        _require_accepted(accepted)
        _require_training_partition(partition)
        compact = dict(_compact_output(request, output))
        if not _output_compatible(request, compact):
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "classifier cannot train on a schema-invalid proposal",
            )
        signature = _classifier_signature(request)
        bucket = self._outputs.setdefault(signature, {})
        if len(self._outputs) > MAX_RESOLVER_ENTRIES:
            _refuse(
                DistillationReason.CORPUS_UNBOUNDED,
                "deterministic classifier exceeds its bound",
            )
        bucket[_output_digest(compact)] = MappingProxyType(compact)

    def propose(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        self.calls += 1
        _non_authority_guard(self)
        bucket = self._outputs.get(_classifier_signature(request), {})
        if len(bucket) != 1:
            return _miss("classifier-miss" if not bucket else "classifier-ambiguous")
        output = dict(next(iter(bucket.values())))
        if not _output_compatible(request, output):
            return _miss("classifier-incompatible")
        return _propose(output)


@dataclass(frozen=True)
class HeldOutResolverEvaluation:
    """Held-out counts.  Accuracy never becomes authority or correctness."""

    evaluated_count: int
    proposed_count: int
    missed_count: int
    matched_count: int
    claims_correctness: bool = False
    can_skip_validation: bool = False
    can_authorize: bool = False
    accuracy_is_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "evaluated_count",
            "proposed_count",
            "missed_count",
            "matched_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        object.__setattr__(
            self, "claims_correctness", _bool(self.claims_correctness, "claims_correctness")
        )
        object.__setattr__(
            self,
            "can_skip_validation",
            _bool(self.can_skip_validation, "can_skip_validation"),
        )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        object.__setattr__(
            self,
            "accuracy_is_authority",
            _bool(self.accuracy_is_authority, "accuracy_is_authority"),
        )
        if self.claims_correctness or self.can_skip_validation or self.can_authorize:
            _refuse(
                DistillationReason.CONFIDENCE_AUTHORITY_REJECTED,
                "held-out evaluation cannot claim correctness or skip validation",
            )
        if self.accuracy_is_authority:
            _refuse(
                DistillationReason.CONFIDENCE_AUTHORITY_REJECTED,
                "held-out accuracy is not authority",
            )


class LocalHoleResolver:
    """Small local model plus the exact cache -> rule -> classifier cascade.

    Confidence is a non-authoritative millicount.  Every proposal remains a
    candidate until independent validation.
    """

    provider_class: ClassVar[ProviderClass] = ProviderClass.LOCAL_SMALL_MODEL
    revision: ClassVar[str] = LOCAL_HOLE_RESOLVER_REVISION
    route_order: ClassVar[tuple[ProviderClass, ...]] = LOCAL_HOLE_ROUTE_ORDER

    def __init__(
        self,
        *,
        cache: ExactHoleCache | None = None,
        rules: Sequence[DeclarativeHoleRule] | DeclarativeHoleRule = (),
        classifier: DeterministicHoleClassifier | None = None,
        remote: object | None = None,
    ) -> None:
        self._cache = cache if cache is not None else ExactHoleCache()
        if isinstance(rules, DeclarativeHoleRule):
            rule_items: tuple[DeclarativeHoleRule, ...] = (rules,)
        else:
            rule_items = tuple(rules)
        self._rules = rule_items
        self._rule_port = _DeclarativeRuleProvider(rule_items)
        self._classifier = (
            classifier if classifier is not None else DeterministicHoleClassifier()
        )
        self._remote = remote
        self._outputs: dict[HoleType, dict[str, Mapping[str, Any]]] = {}
        self._counts: dict[HoleType, dict[str, int]] = {}
        self.calls = 0
        self._last_confidence_millis = 0
        _non_authority_guard(self)
        _non_authority_guard(self._cache)
        _non_authority_guard(self._rule_port)
        _non_authority_guard(self._classifier)

    @property
    def cache(self) -> ExactHoleCache:
        return self._cache

    @property
    def rules(self) -> tuple[DeclarativeHoleRule, ...]:
        return self._rules

    @property
    def rule_provider(self) -> _DeclarativeRuleProvider:
        return self._rule_port

    @property
    def classifier(self) -> DeterministicHoleClassifier:
        return self._classifier

    @property
    def last_confidence_millis(self) -> int:
        return self._last_confidence_millis

    @property
    def can_skip_validation(self) -> bool:
        return False

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def claims_correctness(self) -> bool:
        return False

    def provider_ports(self) -> Mapping[ProviderClass, object]:
        ports: dict[ProviderClass, object] = {
            ProviderClass.EXACT_CACHE: self._cache,
            ProviderClass.DECLARATIVE_RULE: self._rule_port,
            ProviderClass.DETERMINISTIC_CLASSIFIER: self._classifier,
            ProviderClass.LOCAL_SMALL_MODEL: self,
        }
        if self._remote is not None:
            ports[ProviderClass.REMOTE_STANDARD_MODEL] = self._remote
        return MappingProxyType(ports)

    def ingest(
        self,
        request: HoleRequest,
        output: Mapping[str, Any],
        *,
        compiled: CompiledHoleContext | None = None,
        partition: CorpusPartition | str = CorpusPartition.TRAINING,
        accepted: bool = True,
        example_id: str = "",
        candidate_cid: str = "",
        validation_cid: str = "",
        remember_cache: bool = False,
    ) -> None:
        _require_accepted(accepted)
        _require_training_partition(partition)
        compact = dict(_compact_output(request, output))
        if not _output_compatible(request, compact):
            _refuse(
                DistillationReason.UNVERIFIED_EXAMPLE,
                "local hole resolver cannot train on a schema-invalid proposal",
            )
        digest = _output_digest(compact)
        bucket = self._outputs.setdefault(request.hole_type, {})
        counts = self._counts.setdefault(request.hole_type, {})
        if len(self._outputs) > MAX_RESOLVER_ENTRIES:
            _refuse(DistillationReason.CORPUS_UNBOUNDED, "local hole resolver exceeds its bound")
        bucket[digest] = MappingProxyType(compact)
        counts[digest] = counts.get(digest, 0) + 1
        self._classifier.ingest(
            request, compact, partition=partition, accepted=accepted
        )
        if remember_cache:
            self._cache.remember(
                request,
                compact,
                compiled=compiled,
                partition=partition,
                accepted=accepted,
                example_id=example_id,
                candidate_cid=candidate_cid,
                validation_cid=validation_cid,
            )

    def _compatible_unique(
        self, request: HoleRequest
    ) -> tuple[Mapping[str, Any] | None, int]:
        bucket = self._outputs.get(request.hole_type, {})
        counts = self._counts.get(request.hole_type, {})
        compatible: list[tuple[str, Mapping[str, Any], int]] = []
        total = 0
        for digest, output in bucket.items():
            weight = counts.get(digest, 0)
            total += weight
            if _output_compatible(request, dict(output)):
                compatible.append((digest, output, weight))
        if len(compatible) != 1 or total < 1:
            return None, 0
        _, output, weight = compatible[0]
        confidence = min(MAX_CONFIDENCE_MILLIS, (weight * MAX_CONFIDENCE_MILLIS) // total)
        return dict(output), confidence

    def propose(
        self,
        request: HoleRequest,
        compiled: CompiledHoleContext,
    ) -> HoleProviderResult:
        self.calls += 1
        _non_authority_guard(self)
        output, confidence = self._compatible_unique(request)
        self._last_confidence_millis = confidence
        if output is None:
            return _miss("local-model-miss")
        return _propose(output)

    def evaluate_held_out(
        self,
        cases: Sequence[tuple[HoleRequest, CompiledHoleContext, Mapping[str, Any]]],
    ) -> HeldOutResolverEvaluation:
        proposed = 0
        missed = 0
        matched = 0
        for request, compiled, expected in cases:
            result = self.propose(request, compiled)
            if result.outcome is HoleProviderOutcome.MISSED:
                missed += 1
                continue
            proposed += 1
            if dict(result.output) == dict(_compact_output(request, expected)):
                matched += 1
        return HeldOutResolverEvaluation(
            evaluated_count=len(tuple(cases)),
            proposed_count=proposed,
            missed_count=missed,
            matched_count=matched,
            claims_correctness=False,
            can_skip_validation=False,
            can_authorize=False,
            accuracy_is_authority=False,
        )

    def to_hole_resolver(
        self,
        *,
        capacity: Sequence[ProviderCapacitySnapshot | Mapping[str, Any]] | None = None,
        context_compiler: object | None = None,
        current_tree_id: str = "",
        remote: object | None = None,
        extra_providers: Mapping[ProviderClass | str, object] | None = None,
    ) -> HoleResolver:
        ports: dict[ProviderClass, object] = dict(self.provider_ports())
        if remote is not None:
            ports[ProviderClass.REMOTE_STANDARD_MODEL] = remote
        for key, port in dict(extra_providers or {}).items():
            ports[_enum(key, ProviderClass, "provider_class")] = port
        classes = tuple(ports)
        snapshots = capacity
        if snapshots is None:
            snapshots = tuple(
                ProviderCapacitySnapshot(
                    provider_class=item,
                    available=True,
                    remaining_calls=4,
                    max_context_bytes=65_536,
                    max_tokens=8_192,
                    provider_id=f"provider.{item.value}",
                )
                for item in classes
            )
        compiler = context_compiler or default_hole_context_compiler()
        return HoleResolver(
            compiler,
            providers=ports,
            capacity=snapshots,
            current_tree_id=current_tree_id,
        )


for _artifact_type in (DistillationExample, DistillationCorpus, DistillationEvaluation):
    ARTIFACT_TYPES_BY_SCHEMA[_artifact_type.SCHEMA] = _artifact_type


__all__ = [
    "BUILDER_REVISION",
    "DECLARATIVE_HOLE_RULE_REVISION",
    "DETERMINISTIC_HOLE_CLASSIFIER_REVISION",
    "EVALUATOR_REVISION",
    "EXACT_HOLE_CACHE_REVISION",
    "LOCAL_HOLE_RESOLVER_REVISION",
    "MAX_CORPUS_ROWS",
    "MAX_RESOLVER_ENTRIES",
    "REQUIRED_CACHE_KEY_FIELDS",
    "REQUIRED_PARTITIONS",
    "REQUIRED_PRIVACY_CLASSES",
    "REQUIRED_PROVENANCE_FIELDS",
    "TRAINABLE_PARTITIONS",
    "CorpusPartition",
    "DeclarativeHoleRule",
    "DeclarativeHoleRuleKind",
    "DeterministicHoleClassifier",
    "DistillationAdmissionAction",
    "DistillationAdmissionError",
    "DistillationContentReference",
    "DistillationCorpus",
    "DistillationCorpusBuilder",
    "DistillationCorpusRow",
    "DistillationError",
    "DistillationEvaluation",
    "DistillationExample",
    "DistillationLabel",
    "DistillationReason",
    "DistillationReferenceKind",
    "ExactHoleCache",
    "ExactHoleCacheEntry",
    "HeldOutResolverEvaluation",
    "LocalHoleResolver",
]
