"""Residual-hole distillation corpus admission and compact manifests.

Validated hole resolutions become bounded corpus rows.  This module owns
admission, not authority: it never promotes a resolver, skips validation, or
inlines prompts, transcripts, or source bodies.  Large payloads stay behind
content references.  Accepted and rejected examples are labeled only from
independent validation/proof/outcome bindings, and partitions stay disjoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, NoReturn

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    HoleType,
    ProcedureContractError,
    ProcedureSafetyError,
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
    HoleCandidate,
    HoleContextReference,
    HoleRequest,
    HoleResolution,
    HoleResolutionAction,
    HoleResolutionValidator,
    HoleValidationReceipt,
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


for _artifact_type in (DistillationExample, DistillationCorpus, DistillationEvaluation):
    ARTIFACT_TYPES_BY_SCHEMA[_artifact_type.SCHEMA] = _artifact_type


__all__ = [
    "BUILDER_REVISION",
    "EVALUATOR_REVISION",
    "MAX_CORPUS_ROWS",
    "REQUIRED_PARTITIONS",
    "REQUIRED_PRIVACY_CLASSES",
    "REQUIRED_PROVENANCE_FIELDS",
    "CorpusPartition",
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
]
