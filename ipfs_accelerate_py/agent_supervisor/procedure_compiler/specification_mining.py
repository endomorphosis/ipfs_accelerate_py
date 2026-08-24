"""Candidate specification mining from admitted, bounded sources.

The miner proposes declarative pre/post/invariant/frame/effect/resource/order/
idempotency/rollback/freshness properties.  It is not an authority, verifier,
or promotion path:

- every source kind keeps its original provenance and evidence tier
- frequency, absence, and passing tests cannot upgrade candidate status
- conflicting evidence yields a counterexample and a refusal
- emitted wire artifacts stay in ``candidate`` or ``rejected`` state
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    MAX_ITEMS,
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    EpisodeKind,
    ExecutionTrajectory,
    IdempotencyClass,
    ProcedureContractError,
    ProcedureSpec,
    TaskFamily,
    TrajectoryTerminalStatus,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _strings,
    _text,
    canonical_json_bytes,
)
from .contracts import (
    InvariantCandidate as InvariantCandidateArtifact,
)
from .contracts import (
    SpecificationCandidate as SpecificationCandidateArtifact,
)
from .contracts import (
    SpecificationCounterexample as SpecificationCounterexampleArtifact,
)
from .contracts import (
    SpecificationEvidence as SpecificationEvidenceArtifact,
)
from .contracts import (
    SpecificationMiningReceipt as SpecificationMiningReceiptArtifact,
)

MINER_REVISION: Final[str] = "specification-miner@1"
MAX_SOURCES: Final[int] = MAX_ITEMS
MAX_NOMINATIONS: Final[int] = MAX_ITEMS
MAX_OCCURRENCES: Final[int] = 1_000_000
REQUIRED_PROPERTY_KINDS: Final[tuple[str, ...]] = (
    "precondition",
    "postcondition",
    "invariant",
    "frame",
    "effect",
    "resource",
    "order",
    "idempotency",
    "rollback",
    "freshness",
)


class SpecificationMiningError(ProcedureContractError):
    """Admitted sources could not be mined into bounded candidate properties."""


class PropertyKind(str, Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    FRAME = "frame"
    EFFECT = "effect"
    RESOURCE = "resource"
    ORDER = "order"
    IDEMPOTENCY = "idempotency"
    ROLLBACK = "rollback"
    FRESHNESS = "freshness"


class SourceKind(str, Enum):
    TYPE = "type"
    OPERATION_CONTRACT = "operation_contract"
    TEST = "test"
    PROOF_OBLIGATION = "proof_obligation"
    RUNTIME_CHECK = "runtime_check"
    ADMITTED_TRACE = "admitted_trace"
    REJECTED_TRACE = "rejected_trace"
    FAILURE_SIGNATURE = "failure_signature"
    MUTANT = "mutant"
    AUTHORITATIVE_DOCUMENTATION = "authoritative_documentation"


class EvidenceTier(str, Enum):
    """Closed nomination-only tiers.  Mining cannot mint verification."""

    NOMINATION = "nomination"
    DOCUMENTATION_NOMINATION = "documentation_nomination"
    MUTANT_OBSERVATION = "mutant_observation"
    FAILURE_SIGNATURE = "failure_signature"
    TRACE_OBSERVATION = "trace_observation"
    TEST_OBSERVATION = "test_observation"
    RUNTIME_OBSERVATION = "runtime_observation"
    OPERATION_CONTRACT = "operation_contract"
    TYPE_DECLARATION = "type_declaration"
    PROOF_OBLIGATION = "proof_obligation"


class CandidateStatus(str, Enum):
    CANDIDATE = "candidate"
    REFUSED = "refused"


_TIER_RANK: Final[Mapping[EvidenceTier, int]] = MappingProxyType(
    {
        EvidenceTier.NOMINATION: 0,
        EvidenceTier.DOCUMENTATION_NOMINATION: 1,
        EvidenceTier.MUTANT_OBSERVATION: 2,
        EvidenceTier.FAILURE_SIGNATURE: 3,
        EvidenceTier.TRACE_OBSERVATION: 4,
        EvidenceTier.TEST_OBSERVATION: 5,
        EvidenceTier.RUNTIME_OBSERVATION: 6,
        EvidenceTier.OPERATION_CONTRACT: 7,
        EvidenceTier.TYPE_DECLARATION: 8,
        EvidenceTier.PROOF_OBLIGATION: 9,
    }
)

SOURCE_TIER_CEILING: Final[Mapping[SourceKind, EvidenceTier]] = MappingProxyType(
    {
        SourceKind.TYPE: EvidenceTier.TYPE_DECLARATION,
        SourceKind.OPERATION_CONTRACT: EvidenceTier.OPERATION_CONTRACT,
        SourceKind.TEST: EvidenceTier.TEST_OBSERVATION,
        SourceKind.PROOF_OBLIGATION: EvidenceTier.PROOF_OBLIGATION,
        SourceKind.RUNTIME_CHECK: EvidenceTier.RUNTIME_OBSERVATION,
        SourceKind.ADMITTED_TRACE: EvidenceTier.TRACE_OBSERVATION,
        SourceKind.REJECTED_TRACE: EvidenceTier.TRACE_OBSERVATION,
        SourceKind.FAILURE_SIGNATURE: EvidenceTier.FAILURE_SIGNATURE,
        SourceKind.MUTANT: EvidenceTier.MUTANT_OBSERVATION,
        SourceKind.AUTHORITATIVE_DOCUMENTATION: EvidenceTier.DOCUMENTATION_NOMINATION,
    }
)

DEFAULT_SOURCE_TIER: Final[Mapping[SourceKind, EvidenceTier]] = SOURCE_TIER_CEILING


def _property_kind(value: Any) -> PropertyKind:
    return _enum(value, PropertyKind, "property_kind")


def _source_kind(value: Any) -> SourceKind:
    return _enum(value, SourceKind, "source_kind")


def _evidence_tier(value: Any) -> EvidenceTier:
    return _enum(value, EvidenceTier, "evidence_tier")


def _candidate_status(value: Any) -> CandidateStatus:
    return _enum(value, CandidateStatus, "candidate_status")


def _operator(value: Any) -> ConditionOperator:
    return _enum(value, ConditionOperator, "operator")


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SpecificationMiningError(f"{field_name} must be a boolean")
    return value


def _weaker_tier(left: EvidenceTier, right: EvidenceTier) -> EvidenceTier:
    if _TIER_RANK[left] <= _TIER_RANK[right]:
        return left
    return right


def _enforce_tier_ceiling(source_kind: SourceKind, evidence_tier: EvidenceTier) -> EvidenceTier:
    ceiling = SOURCE_TIER_CEILING[source_kind]
    if _TIER_RANK[evidence_tier] > _TIER_RANK[ceiling]:
        raise SpecificationMiningError(
            "source evidence tier exceeds the ceiling for its source kind"
        )
    return evidence_tier


def _canonical_operand(value: Any) -> Any:
    try:
        return _freeze(value, "operand")
    except ProcedureContractError as exc:
        raise SpecificationMiningError(str(exc)) from exc


def _claim(
    property_kind: PropertyKind,
    property_id: str,
    binding: str,
    operator: ConditionOperator,
    operand: Any,
    required: bool,
) -> tuple[Any, ...]:
    return (
        property_kind.value,
        property_id,
        binding,
        operator.value,
        canonical_json_bytes({"operand": _canonical_operand(operand)}).decode("utf-8"),
        required,
    )


@dataclass(frozen=True)
class PropertyNomination:
    """One bounded property proposed by an admitted source."""

    property_kind: PropertyKind
    property_id: str
    binding: str
    operator: ConditionOperator
    operand: Any = None
    evidence_cid: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_kind", _property_kind(self.property_kind))
        object.__setattr__(
            self, "property_id", _identifier(self.property_id, "property_id")
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(self, "operator", _operator(self.operator))
        object.__setattr__(self, "operand", _canonical_operand(self.operand))
        object.__setattr__(
            self,
            "evidence_cid",
            _identifier(self.evidence_cid, "evidence_cid", required=False),
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))

    @property
    def claim(self) -> tuple[Any, ...]:
        return _claim(
            self.property_kind,
            self.property_id,
            self.binding,
            self.operator,
            self.operand,
            self.required,
        )

    @property
    def merge_key(self) -> tuple[str, str]:
        return (self.property_kind.value, self.property_id)


@dataclass(frozen=True)
class SourceProvenance:
    """Exact source-kind provenance retained on every candidate."""

    source_kind: SourceKind
    evidence_tier: EvidenceTier
    provenance_cid: str
    artifact_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", _source_kind(self.source_kind))
        object.__setattr__(self, "evidence_tier", _evidence_tier(self.evidence_tier))
        _enforce_tier_ceiling(self.source_kind, self.evidence_tier)
        object.__setattr__(
            self, "provenance_cid", _identifier(self.provenance_cid, "provenance_cid")
        )
        object.__setattr__(
            self, "artifact_cid", _identifier(self.artifact_cid, "artifact_cid")
        )

    def to_facts(self) -> dict[str, str]:
        return {
            "source_kind": self.source_kind.value,
            "evidence_tier": self.evidence_tier.value,
            "provenance_cid": self.provenance_cid,
            "artifact_cid": self.artifact_cid,
        }

    @classmethod
    def from_facts(cls, payload: Mapping[str, Any]) -> SourceProvenance:
        if not isinstance(payload, Mapping):
            raise SpecificationMiningError("source provenance must be a mapping")
        return cls(
            source_kind=payload.get("source_kind", ""),
            evidence_tier=payload.get("evidence_tier", ""),
            provenance_cid=payload.get("provenance_cid", ""),
            artifact_cid=payload.get("artifact_cid", ""),
        )


@dataclass(frozen=True)
class AdmittedSource:
    """Closed admitted-source schema consumed by the miner."""

    bindings: ArtifactBindings
    source_id: str
    source_kind: SourceKind
    evidence_tier: EvidenceTier
    provenance_cid: str
    artifact_cid: str
    nominations: tuple[PropertyNomination, ...]
    admitted: bool = True
    passing: bool = False
    occurrence_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "source_kind", _source_kind(self.source_kind))
        object.__setattr__(self, "evidence_tier", _evidence_tier(self.evidence_tier))
        _enforce_tier_ceiling(self.source_kind, self.evidence_tier)
        object.__setattr__(
            self, "provenance_cid", _identifier(self.provenance_cid, "provenance_cid")
        )
        object.__setattr__(
            self, "artifact_cid", _identifier(self.artifact_cid, "artifact_cid")
        )
        if not isinstance(self.nominations, Sequence) or isinstance(
            self.nominations, (str, bytes, bytearray, memoryview)
        ):
            raise SpecificationMiningError("nominations must be a sequence")
        if len(self.nominations) > MAX_NOMINATIONS:
            raise SpecificationMiningError("nominations exceeds its item bound")
        object.__setattr__(
            self,
            "nominations",
            tuple(
                item
                if isinstance(item, PropertyNomination)
                else PropertyNomination(**item)
                for item in self.nominations
            ),
        )
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(self, "passing", _bool(self.passing, "passing"))
        object.__setattr__(
            self,
            "occurrence_count",
            _positive_int(self.occurrence_count, "occurrence_count", maximum=MAX_OCCURRENCES),
        )

    @property
    def provenance(self) -> SourceProvenance:
        return SourceProvenance(
            source_kind=self.source_kind,
            evidence_tier=self.evidence_tier,
            provenance_cid=self.provenance_cid,
            artifact_cid=self.artifact_cid,
        )


MiningSource = AdmittedSource | ProcedureSpec | ExecutionTrajectory | TaskFamily


@dataclass(frozen=True)
class SpecificationCandidate:
    """Mined candidate property with exact evidence references.

    Status is ``candidate`` until conflicting evidence forces ``refused``.
    Independent validation is a later obligation and is never performed here.
    """

    property_kind: PropertyKind
    property_id: str
    binding: str
    operator: ConditionOperator
    operand: Any
    required: bool
    status: CandidateStatus
    evidence_tier: EvidenceTier
    evidence_cids: tuple[str, ...]
    source_provenances: tuple[SourceProvenance, ...]
    supporting_occurrences: int = 1
    passing_test_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_kind", _property_kind(self.property_kind))
        object.__setattr__(
            self, "property_id", _identifier(self.property_id, "property_id")
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(self, "operator", _operator(self.operator))
        object.__setattr__(self, "operand", _canonical_operand(self.operand))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "status", _candidate_status(self.status))
        object.__setattr__(self, "evidence_tier", _evidence_tier(self.evidence_tier))
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True, required=True),
        )
        if not isinstance(self.source_provenances, Sequence) or isinstance(
            self.source_provenances, (str, bytes, bytearray, memoryview)
        ):
            raise SpecificationMiningError("source_provenances must be a sequence")
        if not self.source_provenances:
            raise SpecificationMiningError("source_provenances must not be empty")
        if len(self.source_provenances) > MAX_ITEMS:
            raise SpecificationMiningError("source_provenances exceeds its item bound")
        provenances = tuple(
            item
            if isinstance(item, SourceProvenance)
            else SourceProvenance.from_facts(item)
            for item in self.source_provenances
        )
        object.__setattr__(self, "source_provenances", provenances)
        object.__setattr__(
            self,
            "supporting_occurrences",
            _positive_int(
                self.supporting_occurrences,
                "supporting_occurrences",
                maximum=MAX_OCCURRENCES,
            ),
        )
        object.__setattr__(
            self,
            "passing_test_count",
            _nonnegative_int(self.passing_test_count, "passing_test_count"),
        )
        if self.status is CandidateStatus.CANDIDATE:
            for provenance in provenances:
                if _TIER_RANK[self.evidence_tier] > _TIER_RANK[provenance.evidence_tier]:
                    raise SpecificationMiningError(
                        "candidate evidence tier cannot exceed retained source tiers"
                    )

    @property
    def merge_key(self) -> tuple[str, str]:
        return (self.property_kind.value, self.property_id)

    @property
    def claim(self) -> tuple[Any, ...]:
        return _claim(
            self.property_kind,
            self.property_id,
            self.binding,
            self.operator,
            self.operand,
            self.required,
        )

    @property
    def source_kinds(self) -> tuple[SourceKind, ...]:
        kinds: list[SourceKind] = []
        for provenance in self.source_provenances:
            if provenance.source_kind not in kinds:
                kinds.append(provenance.source_kind)
        return tuple(kinds)

    @property
    def provenance_cids(self) -> tuple[str, ...]:
        return tuple(item.provenance_cid for item in self.source_provenances)

    def to_facts(self) -> dict[str, Any]:
        return {
            "property_kind": self.property_kind.value,
            "property_id": self.property_id,
            "binding": self.binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "required": self.required,
            "candidate_status": self.status.value,
            "evidence_tier": self.evidence_tier.value,
            "evidence_cids": self.evidence_cids,
            "source_provenances": tuple(item.to_facts() for item in self.source_provenances),
            "supporting_occurrences": self.supporting_occurrences,
            "passing_test_count": self.passing_test_count,
        }

    def to_artifact(
        self,
        bindings: ArtifactBindings,
        *,
        emitted_at_ms: int = 0,
    ) -> SpecificationCandidateArtifact:
        state = (
            ArtifactState.CANDIDATE
            if self.status is CandidateStatus.CANDIDATE
            else ArtifactState.REJECTED
        )
        return SpecificationCandidateArtifact(
            bindings=bindings,
            state=state,
            subject_cid=self.property_id,
            reference_cids=self.evidence_cids,
            labels=(self.property_kind.value, self.status.value),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )

    def to_invariant_artifact(
        self,
        bindings: ArtifactBindings,
        *,
        emitted_at_ms: int = 0,
    ) -> InvariantCandidateArtifact:
        if self.property_kind is not PropertyKind.INVARIANT:
            raise SpecificationMiningError("invariant artifacts require invariant properties")
        state = (
            ArtifactState.CANDIDATE
            if self.status is CandidateStatus.CANDIDATE
            else ArtifactState.REJECTED
        )
        return InvariantCandidateArtifact(
            bindings=bindings,
            state=state,
            subject_cid=self.property_id,
            reference_cids=self.evidence_cids,
            labels=(self.property_kind.value, self.status.value),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )

    @classmethod
    def from_artifact(cls, artifact: SpecificationCandidateArtifact) -> SpecificationCandidate:
        if not isinstance(artifact, SpecificationCandidateArtifact):
            raise SpecificationMiningError("artifact must be SpecificationCandidate")
        facts = artifact.facts
        return cls(
            property_kind=facts["property_kind"],
            property_id=facts["property_id"],
            binding=facts["binding"],
            operator=facts["operator"],
            operand=facts.get("operand"),
            required=facts["required"],
            status=facts["candidate_status"],
            evidence_tier=facts["evidence_tier"],
            evidence_cids=facts["evidence_cids"],
            source_provenances=tuple(
                SourceProvenance.from_facts(item) for item in facts["source_provenances"]
            ),
            supporting_occurrences=facts["supporting_occurrences"],
            passing_test_count=facts["passing_test_count"],
        )


@dataclass(frozen=True)
class SpecificationCounterexample:
    """Immutable disagreement between two admitted nominations."""

    property_kind: PropertyKind
    property_id: str
    conflict_class: str
    left_claim: tuple[Any, ...]
    right_claim: tuple[Any, ...]
    left_evidence_cid: str
    right_evidence_cid: str
    left_provenance_cid: str
    right_provenance_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_kind", _property_kind(self.property_kind))
        object.__setattr__(
            self, "property_id", _identifier(self.property_id, "property_id")
        )
        object.__setattr__(
            self, "conflict_class", _identifier(self.conflict_class, "conflict_class")
        )
        object.__setattr__(self, "left_claim", tuple(_freeze(self.left_claim, "left_claim")))
        object.__setattr__(
            self, "right_claim", tuple(_freeze(self.right_claim, "right_claim"))
        )
        for name in (
            "left_evidence_cid",
            "right_evidence_cid",
            "left_provenance_cid",
            "right_provenance_cid",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))

    def to_artifact(
        self,
        bindings: ArtifactBindings,
        *,
        emitted_at_ms: int = 0,
    ) -> SpecificationCounterexampleArtifact:
        return SpecificationCounterexampleArtifact(
            bindings=bindings,
            state=ArtifactState.REJECTED,
            subject_cid=self.property_id,
            reference_cids=(self.left_evidence_cid, self.right_evidence_cid),
            labels=(self.property_kind.value, self.conflict_class),
            facts={
                "property_kind": self.property_kind.value,
                "property_id": self.property_id,
                "conflict_class": self.conflict_class,
                "left_claim": self.left_claim,
                "right_claim": self.right_claim,
                "left_evidence_cid": self.left_evidence_cid,
                "right_evidence_cid": self.right_evidence_cid,
                "left_provenance_cid": self.left_provenance_cid,
                "right_provenance_cid": self.right_provenance_cid,
            },
            created_at_ms=emitted_at_ms,
        )


@dataclass(frozen=True)
class SpecificationMiningResult:
    """Deterministic mining output: candidates, refusals, and wire artifacts."""

    bindings: ArtifactBindings
    candidates: tuple[SpecificationCandidate, ...]
    refused: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    evidence_artifacts: tuple[SpecificationEvidenceArtifact, ...]
    candidate_artifacts: tuple[SpecificationCandidateArtifact, ...]
    counterexample_artifacts: tuple[SpecificationCounterexampleArtifact, ...]
    receipt: SpecificationMiningReceiptArtifact
    retained_source_kinds: tuple[SourceKind, ...]
    retained_evidence_tiers: tuple[EvidenceTier, ...]

    @property
    def upgraded_count(self) -> int:
        return 0


def _nomination(
    *,
    property_kind: PropertyKind,
    property_id: str,
    binding: str,
    operator: ConditionOperator,
    operand: Any = None,
    evidence_cid: str,
    required: bool = True,
) -> PropertyNomination:
    return PropertyNomination(
        property_kind=property_kind,
        property_id=property_id,
        binding=binding,
        operator=operator,
        operand=operand,
        evidence_cid=evidence_cid,
        required=required,
    )


def _source(
    *,
    bindings: ArtifactBindings,
    source_id: str,
    source_kind: SourceKind,
    provenance_cid: str,
    artifact_cid: str,
    nominations: Sequence[PropertyNomination],
    evidence_tier: EvidenceTier | None = None,
    passing: bool = False,
) -> AdmittedSource:
    return AdmittedSource(
        bindings=bindings,
        source_id=source_id,
        source_kind=source_kind,
        evidence_tier=evidence_tier or DEFAULT_SOURCE_TIER[source_kind],
        provenance_cid=provenance_cid,
        artifact_cid=artifact_cid,
        nominations=tuple(nominations),
        passing=passing,
    )


def project_procedure_spec(spec: ProcedureSpec) -> tuple[AdmittedSource, ...]:
    """Project a typed ProcedureSpec into closed admitted sources."""

    if not isinstance(spec, ProcedureSpec):
        raise SpecificationMiningError("procedure must be ProcedureSpec")
    bindings = spec.bindings
    provenance = spec.content_id
    type_noms: list[PropertyNomination] = []
    for condition, kind in (
        *((item, PropertyKind.PRECONDITION) for item in spec.preconditions),
        *((item, PropertyKind.POSTCONDITION) for item in spec.postconditions),
        *((item, PropertyKind.INVARIANT) for item in spec.invariants),
    ):
        type_noms.append(
            _nomination(
                property_kind=kind,
                property_id=condition.condition_id,
                binding=condition.binding,
                operator=condition.operator,
                operand=condition.operand,
                evidence_cid=condition.evidence_type or provenance,
                required=condition.required,
            )
        )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.FRAME,
            property_id="frame.declared-reads",
            binding="procedure.declared_reads",
            operator=ConditionOperator.SUBSET_OF,
            operand=spec.declared_reads,
            evidence_cid=provenance,
        )
    )
    effect_targets = tuple(
        target for effect in spec.declared_effects for target in effect.targets
    )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.FRAME,
            property_id="frame.declared-effect-targets",
            binding="procedure.declared_effects.targets",
            operator=ConditionOperator.SUBSET_OF,
            operand=effect_targets,
            evidence_cid=provenance,
        )
    )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.FRAME,
            property_id="frame.scope-paths",
            binding="procedure.scope_paths",
            operator=ConditionOperator.SUBSET_OF,
            operand=spec.scope_paths,
            evidence_cid=provenance,
        )
    )
    for effect in spec.declared_effects:
        type_noms.append(
            _nomination(
                property_kind=PropertyKind.EFFECT,
                property_id=effect.effect_id,
                binding="procedure.declared_effects",
                operator=ConditionOperator.EXISTS,
                operand=effect.effect_class.value,
                evidence_cid=provenance,
            )
        )
    resources = spec.resources
    if resources is None:
        raise SpecificationMiningError("procedure resources envelope is required")
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.RESOURCE,
            property_id="resource.envelope",
            binding="procedure.resources",
            operator=ConditionOperator.EQUALS,
            operand={
                "wall_time_ms": resources.wall_time_ms,
                "cpu_time_ms": resources.cpu_time_ms,
                "memory_bytes": resources.memory_bytes,
                "disk_bytes": resources.disk_bytes,
                "model_token_limit": resources.model_token_limit,
                "model_call_limit": resources.model_call_limit,
                "subprocess_limit": resources.subprocess_limit,
                "network_request_limit": resources.network_request_limit,
            },
            evidence_cid=provenance,
        )
    )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.ORDER,
            property_id="order.step-sequence",
            binding="procedure.steps",
            operator=ConditionOperator.EQUALS,
            operand=tuple(step.step_id for step in spec.steps),
            evidence_cid=provenance,
        )
    )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.ORDER,
            property_id="order.entry-step",
            binding="procedure.entry_step_id",
            operator=ConditionOperator.EQUALS,
            operand=spec.entry_step_id,
            evidence_cid=provenance,
        )
    )
    for step in spec.steps:
        type_noms.append(
            _nomination(
                property_kind=PropertyKind.IDEMPOTENCY,
                property_id=f"idempotency.{step.step_id}",
                binding=step.step_id,
                operator=ConditionOperator.EQUALS,
                operand=step.idempotency.value
                if isinstance(step.idempotency, IdempotencyClass)
                else step.idempotency,
                evidence_cid=step.operation_contract,
            )
        )
    for rollback in spec.rollback:
        type_noms.append(
            _nomination(
                property_kind=PropertyKind.ROLLBACK,
                property_id=rollback.rollback_id,
                binding="procedure.rollback",
                operator=ConditionOperator.CID_EQUALS,
                operand=rollback.exact_target_cid,
                evidence_cid=provenance,
            )
        )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.FRESHNESS,
            property_id="freshness.tree",
            binding="bindings.tree_id",
            operator=ConditionOperator.CURRENT,
            operand=spec.bindings.tree_id,
            evidence_cid=provenance,
        )
    )
    type_noms.append(
        _nomination(
            property_kind=PropertyKind.FRESHNESS,
            property_id="freshness.environment",
            binding="bindings.environment_id",
            operator=ConditionOperator.CURRENT,
            operand=spec.bindings.environment_id,
            evidence_cid=provenance,
        )
    )
    sources = [
        _source(
            bindings=bindings,
            source_id=f"type.{spec.name}",
            source_kind=SourceKind.TYPE,
            provenance_cid=provenance,
            artifact_cid=provenance,
            nominations=type_noms,
        )
    ]
    operation_noms = [
        _nomination(
            property_kind=PropertyKind.IDEMPOTENCY,
            property_id=f"idempotency.{step.step_id}",
            binding=step.step_id,
            operator=ConditionOperator.EQUALS,
            operand=step.idempotency.value
            if isinstance(step.idempotency, IdempotencyClass)
            else step.idempotency,
            evidence_cid=step.operation_contract,
        )
        for step in spec.steps
    ]
    sources.append(
        _source(
            bindings=bindings,
            source_id=f"operation-contract.{spec.name}",
            source_kind=SourceKind.OPERATION_CONTRACT,
            provenance_cid=provenance,
            artifact_cid=provenance,
            nominations=operation_noms,
        )
    )
    validation = spec.validation
    if validation is not None and validation.required_test_contracts:
        typed_posts = {item.condition_id: item for item in spec.postconditions}
        test_noms = []
        for contract in validation.required_test_contracts:
            typed = typed_posts.get("postcondition.tests-admitted")
            test_noms.append(
                _nomination(
                    property_kind=PropertyKind.POSTCONDITION,
                    property_id="postcondition.tests-admitted",
                    binding=typed.binding if typed is not None else "local:test-result",
                    operator=typed.operator if typed is not None else ConditionOperator.ADMITTED,
                    operand=typed.operand if typed is not None else None,
                    evidence_cid=contract,
                    required=typed.required if typed is not None else True,
                )
            )
        sources.append(
            _source(
                bindings=bindings,
                source_id=f"test.{spec.name}",
                source_kind=SourceKind.TEST,
                provenance_cid=provenance,
                artifact_cid=provenance,
                nominations=test_noms,
                passing=True,
            )
        )
    if validation is not None and validation.required_proof_contracts:
        proof_noms = [
            _nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id=f"postcondition.proof-admitted.{contract}",
                binding="local:proof-result",
                operator=ConditionOperator.ADMITTED,
                operand=contract,
                evidence_cid=contract,
            )
            for contract in validation.required_proof_contracts
        ]
        sources.append(
            _source(
                bindings=bindings,
                source_id=f"proof.{spec.name}",
                source_kind=SourceKind.PROOF_OBLIGATION,
                provenance_cid=provenance,
                artifact_cid=provenance,
                nominations=proof_noms,
            )
        )
    if spec.observations:
        runtime_noms = [
            _nomination(
                property_kind=PropertyKind.INVARIANT
                if observation.operator is ConditionOperator.CURRENT
                else PropertyKind.POSTCONDITION,
                property_id=observation.observation_id,
                binding=observation.output_binding,
                operator=observation.operator,
                operand=observation.operand,
                evidence_cid=observation.producer_contract,
            )
            for observation in spec.observations
        ]
        sources.append(
            _source(
                bindings=bindings,
                source_id=f"runtime.{spec.name}",
                source_kind=SourceKind.RUNTIME_CHECK,
                provenance_cid=provenance,
                artifact_cid=provenance,
                nominations=runtime_noms,
            )
        )
    return tuple(sources)


def project_trajectory(trajectory: ExecutionTrajectory) -> AdmittedSource:
    """Project one admitted or rejected trajectory into a closed source."""

    if not isinstance(trajectory, ExecutionTrajectory):
        raise SpecificationMiningError("trajectory must be ExecutionTrajectory")
    provenance = trajectory.content_id
    rejected = trajectory.outcome.status is TrajectoryTerminalStatus.REJECTED or (
        trajectory.source_episode_kind is EpisodeKind.REJECTED_TASK_RECORD
    )
    source_kind = SourceKind.REJECTED_TRACE if rejected else SourceKind.ADMITTED_TRACE
    noms: list[PropertyNomination] = [
        _nomination(
            property_kind=PropertyKind.PRECONDITION,
            property_id="precondition.initial-state",
            binding="trajectory.initial_abstract_state_cid",
            operator=ConditionOperator.EXISTS,
            operand=trajectory.initial_abstract_state_cid,
            evidence_cid=provenance,
        ),
        _nomination(
            property_kind=PropertyKind.ORDER,
            property_id="order.trace-step-sequence",
            binding="trajectory.steps",
            operator=ConditionOperator.EQUALS,
            operand=tuple(step.operation.value for step in trajectory.steps),
            evidence_cid=provenance,
        ),
        _nomination(
            property_kind=PropertyKind.FRAME,
            property_id="frame.state-chain",
            binding="trajectory.steps",
            operator=ConditionOperator.EQUALS,
            operand=tuple(
                (step.initial_state_cid, step.terminal_state_cid) for step in trajectory.steps
            ),
            evidence_cid=provenance,
        ),
        _nomination(
            property_kind=PropertyKind.FRESHNESS,
            property_id="freshness.tree",
            binding="bindings.tree_id",
            operator=ConditionOperator.CURRENT,
            operand=trajectory.bindings.tree_id,
            evidence_cid=provenance,
        ),
        _nomination(
            property_kind=PropertyKind.RESOURCE,
            property_id="resource.observed-cost",
            binding="trajectory.cost",
            operator=ConditionOperator.EQUALS,
            operand={
                "total_tokens": trajectory.total_tokens,
                "total_latency_ms": trajectory.total_latency_ms,
                "total_cost_units": trajectory.total_cost_units,
            },
            evidence_cid=provenance,
        ),
    ]
    effect_ids = tuple(
        dict.fromkeys(effect_id for step in trajectory.steps for effect_id in step.effect_ids)
    )
    for effect_id in effect_ids:
        noms.append(
            _nomination(
                property_kind=PropertyKind.EFFECT,
                property_id=f"effect.{effect_id}",
                binding="trajectory.steps.effect_ids",
                operator=ConditionOperator.EXISTS,
                operand=effect_id,
                evidence_cid=provenance,
            )
        )
    if rejected:
        noms.append(
            _nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.accepted-criteria",
                binding="trajectory.outcome",
                operator=ConditionOperator.NOT_EXISTS,
                operand=trajectory.objective_criterion_ids,
                evidence_cid=provenance,
            )
        )
    else:
        noms.append(
            _nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.accepted-criteria",
                binding="trajectory.outcome",
                operator=ConditionOperator.EQUALS,
                operand=trajectory.outcome.accepted_criterion_ids,
                evidence_cid=provenance,
            )
        )
        noms.append(
            _nomination(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.validation-coverage",
                binding="trajectory.outcome.validation_receipt_cids",
                operator=ConditionOperator.SUBSET_OF,
                operand=tuple(
                    receipt
                    for step in trajectory.steps
                    for receipt in step.validation_receipt_cids
                ),
                evidence_cid=provenance,
            )
        )
    if (
        trajectory.outcome.status is TrajectoryTerminalStatus.ROLLED_BACK
        or trajectory.source_episode_kind is EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT
    ):
        noms.append(
            _nomination(
                property_kind=PropertyKind.ROLLBACK,
                property_id="rollback.successful",
                binding="trajectory.outcome",
                operator=ConditionOperator.EXISTS,
                operand=trajectory.source_episode_cid,
                evidence_cid=provenance,
            )
        )
    return _source(
        bindings=trajectory.bindings,
        source_id=f"{source_kind.value}.{trajectory.source_episode_cid}",
        source_kind=source_kind,
        provenance_cid=provenance,
        artifact_cid=trajectory.source_episode_cid,
        nominations=noms,
    )


def project_task_family(family: TaskFamily) -> tuple[AdmittedSource, ...]:
    """Project task-family shapes and failure signatures into admitted sources."""

    if not isinstance(family, TaskFamily):
        raise SpecificationMiningError("family must be TaskFamily")
    provenance = family.content_id
    shape_noms: list[PropertyNomination] = []
    for item in family.precondition_shape:
        shape_noms.append(
            _nomination(
                property_kind=PropertyKind.PRECONDITION,
                property_id=f"precondition.shape.{item}",
                binding="task_family.precondition_shape",
                operator=ConditionOperator.EXISTS,
                operand=item,
                evidence_cid=provenance,
            )
        )
    for item in family.postcondition_shape:
        shape_noms.append(
            _nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id=f"postcondition.shape.{item}",
                binding="task_family.postcondition_shape",
                operator=ConditionOperator.EXISTS,
                operand=item,
                evidence_cid=provenance,
            )
        )
    for effect in family.effect_classes:
        value = effect.value if isinstance(effect, EffectClass) else effect
        shape_noms.append(
            _nomination(
                property_kind=PropertyKind.EFFECT,
                property_id=f"effect.{value}",
                binding="task_family.effect_classes",
                operator=ConditionOperator.EXISTS,
                operand=value,
                evidence_cid=provenance,
            )
        )
    for item in family.rollback_structure:
        shape_noms.append(
            _nomination(
                property_kind=PropertyKind.ROLLBACK,
                property_id=f"rollback.{item}",
                binding="task_family.rollback_structure",
                operator=ConditionOperator.EXISTS,
                operand=item,
                evidence_cid=provenance,
            )
        )
    sources = [
        _source(
            bindings=family.bindings,
            source_id=f"type.family.{family.name}",
            source_kind=SourceKind.TYPE,
            provenance_cid=provenance,
            artifact_cid=provenance,
            nominations=shape_noms,
        )
    ]
    if family.required_operation_contracts:
        sources.append(
            _source(
                bindings=family.bindings,
                source_id=f"operation-contract.family.{family.name}",
                source_kind=SourceKind.OPERATION_CONTRACT,
                provenance_cid=provenance,
                artifact_cid=provenance,
                nominations=tuple(
                    _nomination(
                        property_kind=PropertyKind.ORDER,
                        property_id=f"operation.{contract}",
                        binding="task_family.required_operation_contracts",
                        operator=ConditionOperator.EXISTS,
                        operand=contract,
                        evidence_cid=contract,
                    )
                    for contract in family.required_operation_contracts
                ),
            )
        )
    if family.failure_signatures:
        sources.append(
            _source(
                bindings=family.bindings,
                source_id=f"failure-signature.{family.name}",
                source_kind=SourceKind.FAILURE_SIGNATURE,
                provenance_cid=provenance,
                artifact_cid=provenance,
                nominations=tuple(
                    _nomination(
                        property_kind=PropertyKind.INVARIANT,
                        property_id=f"invariant.not.{signature}",
                        binding="task_family.failure_signatures",
                        operator=ConditionOperator.NOT_EXISTS,
                        operand=signature,
                        evidence_cid=provenance,
                    )
                    for signature in family.failure_signatures
                ),
            )
        )
    return tuple(sources)


def project_mining_sources(
    sources: Sequence[MiningSource],
) -> tuple[AdmittedSource, ...]:
    """Normalize mixed admitted artifacts into the closed source schema."""

    if not isinstance(sources, Sequence) or isinstance(
        sources, (str, bytes, bytearray, memoryview)
    ):
        raise SpecificationMiningError("sources must be a bounded sequence")
    if len(sources) > MAX_SOURCES:
        raise SpecificationMiningError("sources exceeds its item bound")
    projected: list[AdmittedSource] = []
    for item in sources:
        if isinstance(item, AdmittedSource):
            projected.append(item)
        elif isinstance(item, ProcedureSpec):
            projected.extend(project_procedure_spec(item))
        elif isinstance(item, ExecutionTrajectory):
            projected.append(project_trajectory(item))
        elif isinstance(item, TaskFamily):
            projected.extend(project_task_family(item))
        else:
            raise SpecificationMiningError("unsupported mining source")
    if len(projected) > MAX_SOURCES:
        raise SpecificationMiningError("projected sources exceeds its item bound")
    return tuple(projected)


def _unique_cids(values: Iterable[str]) -> tuple[str, ...]:
    return _strings(tuple(values), "evidence_cids", identifiers=True, required=True)


def _unique_provenances(
    values: Sequence[SourceProvenance],
) -> tuple[SourceProvenance, ...]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[SourceProvenance] = []
    for item in values:
        key = (
            item.source_kind.value,
            item.evidence_tier.value,
            item.provenance_cid,
            item.artifact_cid,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) > MAX_ITEMS:
            raise SpecificationMiningError("source_provenances exceeds its item bound")
    return tuple(result)


@dataclass
class _Accumulator:
    kind: PropertyKind
    property_id: str
    binding: str
    operator: ConditionOperator
    operand: Any
    required: bool
    status: CandidateStatus
    evidence_tier: EvidenceTier
    evidence_cids: list[str] = field(default_factory=list)
    provenances: list[SourceProvenance] = field(default_factory=list)
    supporting_occurrences: int = 0
    passing_test_count: int = 0
    claim: tuple[Any, ...] = ()

    def to_candidate(self) -> SpecificationCandidate:
        return SpecificationCandidate(
            property_kind=self.kind,
            property_id=self.property_id,
            binding=self.binding,
            operator=self.operator,
            operand=self.operand,
            required=self.required,
            status=self.status,
            evidence_tier=self.evidence_tier,
            evidence_cids=_unique_cids(self.evidence_cids),
            source_provenances=_unique_provenances(self.provenances),
            supporting_occurrences=max(self.supporting_occurrences, 1),
            passing_test_count=self.passing_test_count,
        )


class SpecificationMiner:
    """Propose bounded candidate specifications from admitted sources only."""

    def __init__(self, *, miner_revision: str = MINER_REVISION, emitted_at_ms: int = 0) -> None:
        self.miner_revision = _identifier(miner_revision, "miner_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")

    def mine(self, sources: Sequence[MiningSource]) -> SpecificationMiningResult:
        admitted = project_mining_sources(sources)
        if not admitted:
            raise SpecificationMiningError("at least one admitted source is required")
        bindings = admitted[0].bindings
        seen_ids: set[str] = set()
        retained_kinds: list[SourceKind] = []
        retained_tiers: list[EvidenceTier] = []
        accumulators: dict[tuple[str, str], _Accumulator] = {}
        counterexamples: list[SpecificationCounterexample] = []
        evidence_artifacts: list[SpecificationEvidenceArtifact] = []

        for source in admitted:
            if source.bindings != bindings:
                raise SpecificationMiningError("source exact bindings differ")
            if not source.admitted:
                raise SpecificationMiningError("unadmitted sources cannot enter specification mining")
            if source.source_id in seen_ids:
                raise SpecificationMiningError("source_id values must be unique")
            seen_ids.add(source.source_id)
            if source.source_kind not in retained_kinds:
                retained_kinds.append(source.source_kind)
            if source.evidence_tier not in retained_tiers:
                retained_tiers.append(source.evidence_tier)
            property_ids = tuple(item.property_id for item in source.nominations)
            evidence_artifacts.append(
                SpecificationEvidenceArtifact(
                    bindings=bindings,
                    state=ArtifactState.CANDIDATE,
                    subject_cid=source.source_id,
                    reference_cids=(source.provenance_cid, source.artifact_cid),
                    labels=(source.source_kind.value, source.evidence_tier.value),
                    facts={
                        "source_kind": source.source_kind.value,
                        "evidence_tier": source.evidence_tier.value,
                        "provenance_cid": source.provenance_cid,
                        "artifact_cid": source.artifact_cid,
                        "property_ids": property_ids,
                        "occurrence_count": source.occurrence_count,
                        "passing": source.passing,
                    },
                    created_at_ms=self.emitted_at_ms,
                )
            )
            for nomination in source.nominations:
                evidence_cid = nomination.evidence_cid or source.artifact_cid
                key = nomination.merge_key
                existing = accumulators.get(key)
                if existing is None:
                    accumulators[key] = _Accumulator(
                        kind=nomination.property_kind,
                        property_id=nomination.property_id,
                        binding=nomination.binding,
                        operator=nomination.operator,
                        operand=nomination.operand,
                        required=nomination.required,
                        status=CandidateStatus.CANDIDATE,
                        evidence_tier=source.evidence_tier,
                        evidence_cids=[evidence_cid],
                        provenances=[source.provenance],
                        supporting_occurrences=source.occurrence_count,
                        passing_test_count=int(
                            source.passing and source.source_kind is SourceKind.TEST
                        ),
                        claim=nomination.claim,
                    )
                    continue
                already_from_source = any(
                    provenance.provenance_cid == source.provenance_cid
                    and provenance.source_kind is source.source_kind
                    and provenance.artifact_cid == source.artifact_cid
                    for provenance in existing.provenances
                )
                if existing.claim != nomination.claim:
                    existing.status = CandidateStatus.REFUSED
                    if len(counterexamples) < MAX_ITEMS:
                        counterexamples.append(
                            SpecificationCounterexample(
                                property_kind=nomination.property_kind,
                                property_id=nomination.property_id,
                                conflict_class="disagreeing-claim",
                                left_claim=existing.claim,
                                right_claim=nomination.claim,
                                left_evidence_cid=existing.evidence_cids[0],
                                right_evidence_cid=evidence_cid,
                                left_provenance_cid=existing.provenances[0].provenance_cid,
                                right_provenance_cid=source.provenance_cid,
                            )
                        )
                    existing.evidence_cids.append(evidence_cid)
                    if not already_from_source:
                        existing.provenances.append(source.provenance)
                        existing.supporting_occurrences += source.occurrence_count
                    continue
                existing.evidence_cids.append(evidence_cid)
                if already_from_source:
                    continue
                existing.provenances.append(source.provenance)
                existing.supporting_occurrences += source.occurrence_count
                if source.passing and source.source_kind is SourceKind.TEST:
                    existing.passing_test_count += 1
                # Frequency, tests, and extra source kinds cannot raise status or tier.
                existing.evidence_tier = _weaker_tier(
                    existing.evidence_tier, source.evidence_tier
                )

        if len(accumulators) > MAX_ITEMS:
            raise SpecificationMiningError("mined properties exceeds its item bound")

        ordered = tuple(
            accumulators[key].to_candidate()
            for key in sorted(accumulators)
        )
        candidates = tuple(
            item for item in ordered if item.status is CandidateStatus.CANDIDATE
        )
        refused = tuple(item for item in ordered if item.status is CandidateStatus.REFUSED)
        ordered_counterexamples = tuple(
            sorted(
                counterexamples,
                key=lambda item: (item.property_kind.value, item.property_id),
            )
        )
        candidate_artifacts = tuple(
            item.to_artifact(bindings, emitted_at_ms=self.emitted_at_ms)
            for item in candidates
        )
        refused_artifacts = tuple(
            item.to_artifact(bindings, emitted_at_ms=self.emitted_at_ms)
            for item in refused
        )
        counterexample_artifacts = tuple(
            item.to_artifact(bindings, emitted_at_ms=self.emitted_at_ms)
            for item in ordered_counterexamples
        )
        receipt = SpecificationMiningReceiptArtifact(
            bindings=bindings,
            state=ArtifactState.CANDIDATE,
            subject_cid=self.miner_revision,
            reference_cids=tuple(
                artifact.content_id
                for artifact in (*candidate_artifacts, *counterexample_artifacts)
            ),
            labels=(self.miner_revision.replace("@", "-"), "candidate"),
            facts={
                "miner_revision": self.miner_revision,
                "candidate_cids": tuple(item.content_id for item in candidate_artifacts),
                "refused_cids": tuple(item.content_id for item in refused_artifacts),
                "evidence_cids": tuple(item.content_id for item in evidence_artifacts),
                "counterexample_cids": tuple(
                    item.content_id for item in counterexample_artifacts
                ),
                "retained_source_kinds": tuple(item.value for item in retained_kinds),
                "retained_evidence_tiers": tuple(item.value for item in retained_tiers),
                "candidate_count": len(candidates),
                "refused_count": len(refused),
                "counterexample_count": len(ordered_counterexamples),
                "upgraded_count": 0,
                "verified_count": 0,
            },
            created_at_ms=self.emitted_at_ms,
        )
        return SpecificationMiningResult(
            bindings=bindings,
            candidates=candidates,
            refused=refused,
            counterexamples=ordered_counterexamples,
            evidence_artifacts=tuple(evidence_artifacts),
            candidate_artifacts=candidate_artifacts,
            counterexample_artifacts=counterexample_artifacts,
            receipt=receipt,
            retained_source_kinds=tuple(retained_kinds),
            retained_evidence_tiers=tuple(retained_tiers),
        )


__all__ = [
    "DEFAULT_SOURCE_TIER",
    "MAX_NOMINATIONS",
    "MAX_OCCURRENCES",
    "MAX_SOURCES",
    "MINER_REVISION",
    "REQUIRED_PROPERTY_KINDS",
    "SOURCE_TIER_CEILING",
    "AdmittedSource",
    "MiningSource",
    "CandidateStatus",
    "EvidenceTier",
    "InvariantCandidateArtifact",
    "PropertyKind",
    "PropertyNomination",
    "SourceKind",
    "SourceProvenance",
    "SpecificationCandidate",
    "SpecificationCandidateArtifact",
    "SpecificationCounterexample",
    "SpecificationCounterexampleArtifact",
    "SpecificationEvidenceArtifact",
    "SpecificationMiner",
    "SpecificationMiningError",
    "SpecificationMiningReceiptArtifact",
    "SpecificationMiningResult",
    "project_mining_sources",
    "project_procedure_spec",
    "project_task_family",
    "project_trajectory",
]
