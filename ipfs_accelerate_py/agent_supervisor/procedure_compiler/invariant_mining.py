"""Candidate-tier invariant mining from admitted specification sources.

Invariant mining is a specialized view of specification mining.  It proposes
only invariant properties, retains exact evidence references, and never
promotes a candidate because of frequency, absence, or passing tests.
Conflicting operands become counterexamples.  Independent non-vacuity and
assurance validation remain a later tranche.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    _bounded,
    _decode_fields,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _records,
    _schema_name,
    _text,
    _verify_identity,
)
from .specification_mining import (
    MAX_SUPPORTING_COUNT,
    GroupedClaim,
    PropertyKind,
    SpecificationCounterexample,
    SpecificationEvidence,
    SpecificationMiningError,
    SpecificationSource,
    SpecificationSourceKind,
    SpecificationTier,
    claims_to_candidates,
    conflicts_to_counterexamples,
    group_source_claims,
    validate_specification_sources,
)


INVARIANT_SOURCE_KINDS: Final[frozenset[SpecificationSourceKind]] = frozenset(
    {
        SpecificationSourceKind.TYPES,
        SpecificationSourceKind.TEST,
        SpecificationSourceKind.PROOF_OBLIGATION,
        SpecificationSourceKind.RUNTIME_CHECK,
        SpecificationSourceKind.ADMITTED_TRACE,
        SpecificationSourceKind.REJECTED_TRACE,
        SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION,
        SpecificationSourceKind.OPERATION_CONTRACT,
    }
)


class InvariantMiningError(SpecificationMiningError):
    """Admitted sources could not be mined into candidate invariants."""


@dataclass(frozen=True)
class InvariantCandidate(CanonicalContract):
    """One mined invariant that remains candidate-tier."""

    SCHEMA: ClassVar[str] = _schema_name("InvariantCandidate")

    bindings: ArtifactBindings
    property_id: str
    condition_id: str
    binding: str
    operator: ConditionOperator
    operand: Any
    evidence: tuple[SpecificationEvidence, ...]
    source_kinds: tuple[SpecificationSourceKind, ...]
    supporting_count: int = 1
    passing_test_count: int = 0
    state: ArtifactState = ArtifactState.CANDIDATE
    evidence_producer: str = ""
    evidence_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings")
        )
        object.__setattr__(self, "property_id", _identifier(self.property_id, "property_id"))
        object.__setattr__(
            self, "condition_id", _identifier(self.condition_id, "condition_id")
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(
            self, "operator", _enum(self.operator, ConditionOperator, "operator")
        )
        object.__setattr__(self, "operand", _freeze(self.operand, "operand"))
        object.__setattr__(
            self,
            "evidence",
            _records(self.evidence, SpecificationEvidence, "evidence", required=True),
        )
        kinds: list[SpecificationSourceKind] = []
        if not isinstance(self.source_kinds, Sequence) or isinstance(
            self.source_kinds, (str, bytes, bytearray, memoryview)
        ):
            raise InvariantMiningError("source_kinds must be a sequence")
        for value in self.source_kinds:
            item = _enum(value, SpecificationSourceKind, "source_kinds")
            if item not in kinds:
                kinds.append(item)
        if not kinds:
            raise InvariantMiningError("source_kinds must not be empty")
        object.__setattr__(self, "source_kinds", tuple(kinds))
        object.__setattr__(
            self,
            "supporting_count",
            _nonnegative_int(
                self.supporting_count, "supporting_count", maximum=MAX_SUPPORTING_COUNT
            ),
        )
        object.__setattr__(
            self,
            "passing_test_count",
            _nonnegative_int(
                self.passing_test_count,
                "passing_test_count",
                maximum=MAX_SUPPORTING_COUNT,
            ),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            raise InvariantMiningError("mined invariants remain candidate-tier")
        object.__setattr__(
            self,
            "evidence_producer",
            _identifier(self.evidence_producer, "evidence_producer", required=False),
        )
        object.__setattr__(
            self,
            "evidence_type",
            _identifier(self.evidence_type, "evidence_type", required=False),
        )
        if self.property_id != self.condition_id:
            raise InvariantMiningError("invariant condition_id must equal property_id")
        evidence_kinds = tuple(dict.fromkeys(item.source_kind for item in self.evidence))
        if evidence_kinds != self.source_kinds:
            raise InvariantMiningError("source_kinds must retain exact evidence provenance")
        for item in self.evidence:
            if item.property_id != self.property_id:
                raise InvariantMiningError("evidence does not reference this invariant")
            if item.bindings != self.bindings:
                raise InvariantMiningError("evidence exact bindings differ")
            if item.tier not in {SpecificationTier.NOMINATION, SpecificationTier.CANDIDATE}:
                raise InvariantMiningError("evidence left the candidate tier")
        _bounded(self, "InvariantCandidate")

    @property
    def evidence_cids(self) -> tuple[str, ...]:
        return tuple(item.content_id for item in self.evidence)

    @property
    def property_kind(self) -> PropertyKind:
        return PropertyKind.INVARIANT

    @property
    def remains_candidate(self) -> bool:
        return self.state is ArtifactState.CANDIDATE

    @property
    def tier(self) -> SpecificationTier:
        if self.evidence and all(
            item.tier is SpecificationTier.NOMINATION for item in self.evidence
        ):
            return SpecificationTier.NOMINATION
        return SpecificationTier.CANDIDATE

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "property_id": self.property_id,
            "condition_id": self.condition_id,
            "binding": self.binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "evidence": self.evidence,
            "source_kinds": tuple(kind.value for kind in self.source_kinds),
            "supporting_count": self.supporting_count,
            "passing_test_count": self.passing_test_count,
            "state": self.state.value,
            "evidence_producer": self.evidence_producer,
            "evidence_type": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InvariantCandidate:
        fields = (
            "bindings",
            "property_id",
            "condition_id",
            "binding",
            "operator",
            "operand",
            "evidence",
            "source_kinds",
            "supporting_count",
            "passing_test_count",
            "state",
            "evidence_producer",
            "evidence_type",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _nested(values["bindings"], ArtifactBindings, "bindings")
        if "evidence" in values:
            values["evidence"] = _records(
                values["evidence"], SpecificationEvidence, "evidence"
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class InvariantMiningResult:
    candidates: tuple[InvariantCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    specification_candidates: tuple[Any, ...] = ()

    @property
    def refused_property_ids(self) -> tuple[str, ...]:
        return tuple(item.property_id for item in self.counterexamples)


def _claim_to_invariant(claim: GroupedClaim, bindings: ArtifactBindings) -> InvariantCandidate:
    producer = ""
    evidence_type = ""
    if claim.evidence:
        producer = claim.evidence[0].producer_id
        evidence_type = claim.evidence[0].evidence_type
    return InvariantCandidate(
        bindings=bindings,
        property_id=claim.property_id,
        condition_id=claim.property_id,
        binding=claim.binding,
        operator=claim.operator,
        operand=claim.operand,
        evidence=claim.evidence,
        source_kinds=claim.source_kinds,
        supporting_count=claim.supporting_count,
        passing_test_count=claim.passing_test_count,
        state=ArtifactState.CANDIDATE,
        evidence_producer=producer,
        evidence_type=evidence_type,
    )


class InvariantMiner:
    """Propose candidate invariants without discharging or promoting them."""

    def mine(
        self,
        sources: Sequence[SpecificationSource],
    ) -> InvariantMiningResult:
        validated = validate_specification_sources(sources)
        claims, conflicts = group_source_claims(
            validated, kinds=frozenset({PropertyKind.INVARIANT})
        )
        bindings = validated[0].bindings
        candidates = tuple(_claim_to_invariant(claim, bindings) for claim in claims)
        counterexamples = conflicts_to_counterexamples(conflicts, bindings)
        specification_candidates = claims_to_candidates(claims, bindings)
        return InvariantMiningResult(
            candidates=candidates,
            counterexamples=counterexamples,
            specification_candidates=specification_candidates,
        )


__all__ = [
    "INVARIANT_SOURCE_KINDS",
    "InvariantCandidate",
    "InvariantMiner",
    "InvariantMiningError",
    "InvariantMiningResult",
]
