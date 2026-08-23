"""Candidate-tier specification mining from admitted hermetic sources.

The miner reads independently admitted source artifacts and emits bounded
precondition, postcondition, invariant, frame, effect, resource, order,
idempotency, rollback, and freshness candidates.  Every candidate keeps exact
evidence references and remains ``candidate``; frequency, absence, and passing
tests cannot promote a property.  Conflicting operands produce a
counterexample and refuse the property.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, canonical_json_bytes
from .contracts import (
    MAX_ITEMS,
    MAX_MAPPING_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    IdempotencyClass,
    ProcedureBoundsError,
    ProcedureContractError,
    _bounded,
    _decode_fields,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _records,
    _schema_name,
    _strings,
    _text,
    _verify_identity,
)


MAX_MINING_SOURCES: Final[int] = MAX_ITEMS
MAX_PROPOSALS_PER_SOURCE: Final[int] = 32
MAX_SUPPORTING_COUNT: Final[int] = 2**31 - 1
_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "wall_time_ms",
    "cpu_time_ms",
    "memory_bytes",
    "disk_bytes",
    "model_token_limit",
    "model_call_limit",
    "subprocess_limit",
    "network_request_limit",
)
_RESOURCE_MAXIMA: Final[Mapping[str, int]] = MappingProxyType(
    {
        "wall_time_ms": 86_400_000,
        "cpu_time_ms": 86_400_000,
        "memory_bytes": 1 << 50,
        "disk_bytes": 1 << 50,
        "model_token_limit": 10_000_000,
        "model_call_limit": 1_024,
        "subprocess_limit": 1_024,
        "network_request_limit": 1_024,
    }
)


class SpecificationMiningError(ProcedureContractError):
    """Admitted sources could not be mined into bounded candidate properties."""


class SpecificationSourceKind(str, Enum):
    TYPES = "types"
    OPERATION_CONTRACT = "operation_contract"
    TEST = "test"
    PROOF_OBLIGATION = "proof_obligation"
    RUNTIME_CHECK = "runtime_check"
    ADMITTED_TRACE = "admitted_trace"
    REJECTED_TRACE = "rejected_trace"
    FAILURE_SIGNATURE = "failure_signature"
    MUTANT = "mutant"
    AUTHORITATIVE_DOCUMENTATION = "authoritative_documentation"


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


class SpecificationTier(str, Enum):
    NOMINATION = "nomination"
    CANDIDATE = "candidate"


_PROMOTING_STATES: Final[frozenset[ArtifactState]] = frozenset(
    {
        ArtifactState.VERIFIED,
        ArtifactState.PROMOTED,
        ArtifactState.SHADOW,
        ArtifactState.DEVELOPMENT,
    }
)


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _frozen_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    frozen = _freeze(value, field_name)
    if not isinstance(frozen, Mapping):
        raise SpecificationMiningError(f"{field_name} must be a mapping")
    return frozen


def _claim_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_freeze(value, "claim"))


def _property_id(kind: PropertyKind, binding: str, operator: ConditionOperator) -> str:
    raw = f"{kind.value}.{binding}.{operator.value}"
    try:
        return _identifier(raw, "property_id")
    except ProcedureContractError:
        digest = _identifier(
            f"{kind.value}.{canonical_json_bytes(raw).hex()[:32]}",
            "property_id",
        )
        return digest


def _tier_for_source(source: SpecificationSource) -> SpecificationTier:
    if (
        source.source_kind is SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION
        and not source.authoritative
    ):
        return SpecificationTier.NOMINATION
    return SpecificationTier.CANDIDATE


def _normalize_paths(values: Any, field_name: str) -> tuple[str, ...]:
    return _strings(values, field_name, paths=True, preserve_order=False)


def _normalize_operand(kind: PropertyKind, operand: Any) -> Any:
    if kind is PropertyKind.FRAME:
        return _normalize_paths(operand if operand is not None else (), "operand")
    if kind is PropertyKind.EFFECT:
        return _enum(operand, EffectClass, "operand").value
    if kind is PropertyKind.IDEMPOTENCY:
        return _enum(operand, IdempotencyClass, "operand").value
    if kind is PropertyKind.ORDER:
        return _strings(operand if operand is not None else (), "operand", identifiers=True)
    if kind is PropertyKind.RESOURCE:
        if not isinstance(operand, Mapping):
            raise SpecificationMiningError("resource operand must be a mapping")
        if len(operand) > MAX_MAPPING_ITEMS:
            raise ProcedureBoundsError("resource operand exceeds its mapping bound")
        normalized: dict[str, int] = {}
        for raw_key, raw_value in operand.items():
            key = _identifier(raw_key, "operand")
            if key not in _RESOURCE_MAXIMA:
                raise SpecificationMiningError("resource operand names an unknown bound")
            normalized[key] = _nonnegative_int(
                raw_value, "operand", maximum=_RESOURCE_MAXIMA[key]
            )
        if not normalized:
            raise SpecificationMiningError("resource operand must not be empty")
        return _freeze(normalized, "operand")
    if kind is PropertyKind.ROLLBACK:
        if isinstance(operand, Mapping):
            target = _identifier(operand.get("exact_target_cid", ""), "operand")
            extras = {
                key: operand[key]
                for key in operand
                if key != "exact_target_cid"
            }
            if extras:
                _freeze(extras, "operand")
            return target
        return _identifier(operand, "operand")
    if kind is PropertyKind.FRESHNESS:
        if not isinstance(operand, Mapping):
            raise SpecificationMiningError("freshness operand must be a mapping")
        tree_id = _identifier(operand.get("tree_id", ""), "operand")
        commit = _identifier(operand.get("repository_commit", ""), "operand")
        return _freeze({"tree_id": tree_id, "repository_commit": commit}, "operand")
    return _freeze(operand, "operand")


def _normalize_extras(kind: PropertyKind, extras: Any) -> Mapping[str, Any]:
    if extras is None:
        extras = {}
    frozen = _frozen_mapping(extras, "extras")
    if kind is PropertyKind.FRAME and "unchanged_paths" in frozen:
        return MappingProxyType(
            {
                **dict(frozen),
                "unchanged_paths": _normalize_paths(
                    frozen["unchanged_paths"], "extras"
                ),
            }
        )
    if kind is PropertyKind.EFFECT and "targets" in frozen:
        return MappingProxyType(
            {
                **dict(frozen),
                "targets": _strings(frozen["targets"], "extras", paths=True),
            }
        )
    if kind is PropertyKind.ORDER and "step_ids" in frozen:
        return MappingProxyType(
            {
                **dict(frozen),
                "step_ids": _strings(frozen["step_ids"], "extras", identifiers=True),
            }
        )
    if kind is PropertyKind.ROLLBACK and "step_ids" in frozen:
        return MappingProxyType(
            {
                **dict(frozen),
                "step_ids": _strings(frozen["step_ids"], "extras", identifiers=True),
            }
        )
    return frozen


@dataclass(frozen=True)
class PropertyProposal(CanonicalContract):
    """One bounded property claim extracted from an admitted source."""

    SCHEMA: ClassVar[str] = _schema_name("PropertyProposal")

    property_kind: PropertyKind
    binding: str
    operator: ConditionOperator
    operand: Any = None
    occurrence_count: int = 1
    passing_test_count: int = 0
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_kind",
            _enum(self.property_kind, PropertyKind, "property_kind"),
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(
            self, "operator", _enum(self.operator, ConditionOperator, "operator")
        )
        object.__setattr__(
            self, "operand", _normalize_operand(self.property_kind, self.operand)
        )
        object.__setattr__(
            self,
            "occurrence_count",
            _nonnegative_int(
                self.occurrence_count, "occurrence_count", maximum=MAX_SUPPORTING_COUNT
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
        object.__setattr__(
            self, "extras", _normalize_extras(self.property_kind, self.extras)
        )
        _bounded(self, "PropertyProposal")

    @property
    def property_id(self) -> str:
        return _property_id(self.property_kind, self.binding, self.operator)

    @property
    def claim_identity(self) -> bytes:
        return _claim_bytes({"operand": self.operand, "extras": dict(self.extras)})

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "property_kind": self.property_kind.value,
            "binding": self.binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "occurrence_count": self.occurrence_count,
            "passing_test_count": self.passing_test_count,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PropertyProposal:
        fields = (
            "property_kind",
            "binding",
            "operator",
            "operand",
            "occurrence_count",
            "passing_test_count",
            "extras",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PropertyProposal:
        if "schema" in payload:
            return cls.from_dict(payload)
        values = {
            key: payload[key]
            for key in (
                "property_kind",
                "binding",
                "operator",
                "operand",
                "occurrence_count",
                "passing_test_count",
                "extras",
            )
            if key in payload
        }
        return cls(**values)


@dataclass(frozen=True)
class SpecificationSource(CanonicalContract):
    """Closed admitted-source schema consumed by specification mining."""

    SCHEMA: ClassVar[str] = _schema_name("SpecificationSource")

    bindings: ArtifactBindings
    source_kind: SpecificationSourceKind
    source_cid: str
    producer_id: str
    evidence_type: str
    admitted: bool
    authoritative: bool = False
    proposals: tuple[PropertyProposal, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self,
            "source_kind",
            _enum(self.source_kind, SpecificationSourceKind, "source_kind"),
        )
        for name in ("source_cid", "producer_id", "evidence_type"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if type(self.admitted) is not bool:
            raise SpecificationMiningError("admitted must be a boolean")
        if type(self.authoritative) is not bool:
            raise SpecificationMiningError("authoritative must be a boolean")
        if (
            self.authoritative
            and self.source_kind is not SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION
        ):
            raise SpecificationMiningError(
                "authoritative is valid only for authoritative_documentation sources"
            )
        object.__setattr__(
            self,
            "proposals",
            _records(
                self.proposals,
                PropertyProposal,
                "proposals",
                limit=MAX_PROPOSALS_PER_SOURCE,
            ),
        )
        object.__setattr__(self, "facts", _frozen_mapping(self.facts, "facts"))
        if not self.admitted:
            raise SpecificationMiningError("specification sources must be independently admitted")
        _bounded(self, "SpecificationSource")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_kind": self.source_kind.value,
            "source_cid": self.source_cid,
            "producer_id": self.producer_id,
            "evidence_type": self.evidence_type,
            "admitted": self.admitted,
            "authoritative": self.authoritative,
            "proposals": self.proposals,
            "facts": self.facts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpecificationSource:
        fields = (
            "bindings",
            "source_kind",
            "source_cid",
            "producer_id",
            "evidence_type",
            "admitted",
            "authoritative",
            "proposals",
            "facts",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "proposals" in values:
            values["proposals"] = _records(
                values["proposals"], PropertyProposal, "proposals", limit=MAX_PROPOSALS_PER_SOURCE
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class SpecificationEvidence(CanonicalContract):
    """Exact provenance retained for one mined property claim."""

    SCHEMA: ClassVar[str] = _schema_name("SpecificationEvidence")

    bindings: ArtifactBindings
    source_kind: SpecificationSourceKind
    source_cid: str
    producer_id: str
    evidence_type: str
    tier: SpecificationTier
    property_id: str
    admitted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self,
            "source_kind",
            _enum(self.source_kind, SpecificationSourceKind, "source_kind"),
        )
        for name in ("source_cid", "producer_id", "evidence_type", "property_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "tier", _enum(self.tier, SpecificationTier, "tier"))
        if type(self.admitted) is not bool:
            raise SpecificationMiningError("admitted must be a boolean")
        if self.tier not in {SpecificationTier.NOMINATION, SpecificationTier.CANDIDATE}:
            raise SpecificationMiningError("mined evidence cannot leave the candidate tier")
        _bounded(self, "SpecificationEvidence")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_kind": self.source_kind.value,
            "source_cid": self.source_cid,
            "producer_id": self.producer_id,
            "evidence_type": self.evidence_type,
            "tier": self.tier.value,
            "property_id": self.property_id,
            "admitted": self.admitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpecificationEvidence:
        fields = (
            "bindings",
            "source_kind",
            "source_cid",
            "producer_id",
            "evidence_type",
            "tier",
            "property_id",
            "admitted",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


def _reject_promoted_state(state: ArtifactState, artifact_name: str) -> ArtifactState:
    normalized = _enum(state, ArtifactState, "state")
    if normalized in _PROMOTING_STATES:
        raise SpecificationMiningError(
            f"{artifact_name} cannot assert verified or promoted status"
        )
    if normalized is not ArtifactState.CANDIDATE and normalized is not ArtifactState.REJECTED:
        raise SpecificationMiningError(
            f"{artifact_name} must remain candidate-tier until independent validation"
        )
    return normalized


@dataclass(frozen=True)
class SpecificationCandidate(CanonicalContract):
    """One mined property that remains candidate-tier until independent validation."""

    SCHEMA: ClassVar[str] = _schema_name("SpecificationCandidate")

    bindings: ArtifactBindings
    property_id: str
    property_kind: PropertyKind
    binding: str
    operator: ConditionOperator
    operand: Any
    evidence: tuple[SpecificationEvidence, ...]
    source_kinds: tuple[SpecificationSourceKind, ...]
    supporting_count: int = 1
    passing_test_count: int = 0
    extras: Mapping[str, Any] = field(default_factory=dict)
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "property_id", _identifier(self.property_id, "property_id"))
        object.__setattr__(
            self,
            "property_kind",
            _enum(self.property_kind, PropertyKind, "property_kind"),
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(
            self, "operator", _enum(self.operator, ConditionOperator, "operator")
        )
        object.__setattr__(
            self, "operand", _normalize_operand(self.property_kind, self.operand)
        )
        object.__setattr__(
            self,
            "evidence",
            _records(self.evidence, SpecificationEvidence, "evidence", required=True),
        )
        object.__setattr__(
            self,
            "source_kinds",
            _enums_closed(self.source_kinds, "source_kinds"),
        )
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
        object.__setattr__(self, "extras", _normalize_extras(self.property_kind, self.extras))
        object.__setattr__(
            self, "state", _reject_promoted_state(self.state, "SpecificationCandidate")
        )
        if self.state is not ArtifactState.CANDIDATE:
            raise SpecificationMiningError(
                "mined specification properties remain candidate-tier"
            )
        expected = _property_id(self.property_kind, self.binding, self.operator)
        if self.property_id != expected:
            raise SpecificationMiningError("property_id does not match the closed claim key")
        evidence_kinds = tuple(
            dict.fromkeys(item.source_kind for item in self.evidence)
        )
        if evidence_kinds != self.source_kinds:
            raise SpecificationMiningError("source_kinds must retain exact evidence provenance")
        for item in self.evidence:
            if item.property_id != self.property_id:
                raise SpecificationMiningError("evidence does not reference this property")
            if item.bindings != self.bindings:
                raise SpecificationMiningError("evidence exact bindings differ")
            if item.tier not in {SpecificationTier.NOMINATION, SpecificationTier.CANDIDATE}:
                raise SpecificationMiningError("evidence left the candidate tier")
        _bounded(self, "SpecificationCandidate")

    @property
    def evidence_cids(self) -> tuple[str, ...]:
        return tuple(item.content_id for item in self.evidence)

    @property
    def tier(self) -> SpecificationTier:
        if any(item.tier is SpecificationTier.NOMINATION for item in self.evidence):
            if all(item.tier is SpecificationTier.NOMINATION for item in self.evidence):
                return SpecificationTier.NOMINATION
        return SpecificationTier.CANDIDATE

    @property
    def remains_candidate(self) -> bool:
        return self.state is ArtifactState.CANDIDATE

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "property_id": self.property_id,
            "property_kind": self.property_kind.value,
            "binding": self.binding,
            "operator": self.operator.value,
            "operand": self.operand,
            "evidence": self.evidence,
            "source_kinds": tuple(kind.value for kind in self.source_kinds),
            "supporting_count": self.supporting_count,
            "passing_test_count": self.passing_test_count,
            "extras": self.extras,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpecificationCandidate:
        fields = (
            "bindings",
            "property_id",
            "property_kind",
            "binding",
            "operator",
            "operand",
            "evidence",
            "source_kinds",
            "supporting_count",
            "passing_test_count",
            "extras",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "evidence" in values:
            values["evidence"] = _records(
                values["evidence"], SpecificationEvidence, "evidence"
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


def _enums_closed(
    values: Any, field_name: str
) -> tuple[SpecificationSourceKind, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise SpecificationMiningError(f"{field_name} must be a sequence")
    if len(values) > len(SpecificationSourceKind):
        raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
    result: list[SpecificationSourceKind] = []
    for value in values:
        item = _enum(value, SpecificationSourceKind, field_name)
        if item not in result:
            result.append(item)
    if not result:
        raise SpecificationMiningError(f"{field_name} must not be empty")
    return tuple(result)


@dataclass(frozen=True)
class SpecificationCounterexample(CanonicalContract):
    """Conflicting evidence that refuses a mined property."""

    SCHEMA: ClassVar[str] = _schema_name("SpecificationCounterexample")

    bindings: ArtifactBindings
    property_id: str
    property_kind: PropertyKind
    binding: str
    operator: ConditionOperator
    conflicting_source_cids: tuple[str, ...]
    conflicting_operands: tuple[Any, ...]
    evidence_cids: tuple[str, ...]
    violation_class: str = "conflicting_evidence"
    state: ArtifactState = ArtifactState.REJECTED

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "property_id", _identifier(self.property_id, "property_id"))
        object.__setattr__(
            self,
            "property_kind",
            _enum(self.property_kind, PropertyKind, "property_kind"),
        )
        object.__setattr__(self, "binding", _text(self.binding, "binding"))
        object.__setattr__(
            self, "operator", _enum(self.operator, ConditionOperator, "operator")
        )
        object.__setattr__(
            self,
            "conflicting_source_cids",
            _strings(
                self.conflicting_source_cids,
                "conflicting_source_cids",
                identifiers=True,
                required=True,
            ),
        )
        operands = _freeze(self.conflicting_operands, "conflicting_operands")
        if not isinstance(operands, tuple) or len(operands) < 2:
            raise SpecificationMiningError(
                "counterexamples require at least two conflicting operands"
            )
        object.__setattr__(self, "conflicting_operands", operands)
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "violation_class",
            _identifier(self.violation_class, "violation_class"),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state not in {ArtifactState.REJECTED, ArtifactState.CANDIDATE}:
            raise SpecificationMiningError(
                "counterexamples cannot assert verified or promoted status"
            )
        _bounded(self, "SpecificationCounterexample")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "property_id": self.property_id,
            "property_kind": self.property_kind.value,
            "binding": self.binding,
            "operator": self.operator.value,
            "conflicting_source_cids": self.conflicting_source_cids,
            "conflicting_operands": self.conflicting_operands,
            "evidence_cids": self.evidence_cids,
            "violation_class": self.violation_class,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpecificationCounterexample:
        fields = (
            "bindings",
            "property_id",
            "property_kind",
            "binding",
            "operator",
            "conflicting_source_cids",
            "conflicting_operands",
            "evidence_cids",
            "violation_class",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class SpecificationMiningReceipt(CanonicalContract):
    """Durable mining summary: candidates, refusals, and source provenance."""

    SCHEMA: ClassVar[str] = _schema_name("SpecificationMiningReceipt")

    bindings: ArtifactBindings
    source_cids: tuple[str, ...]
    candidate_cids: tuple[str, ...]
    counterexample_cids: tuple[str, ...]
    invariant_candidate_cids: tuple[str, ...] = ()
    refused_property_ids: tuple[str, ...] = ()
    refusal_classes: tuple[str, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "source_cids",
            "candidate_cids",
            "counterexample_cids",
            "invariant_candidate_cids",
            "refused_property_ids",
            "refusal_classes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(
            self, "state", _reject_promoted_state(self.state, "SpecificationMiningReceipt")
        )
        if self.state is not ArtifactState.CANDIDATE:
            raise SpecificationMiningError("mining receipts remain candidate-tier")
        if len(self.refused_property_ids) != len(self.refusal_classes):
            raise SpecificationMiningError("refusal classes must pair refused property ids")
        if not self.source_cids:
            raise SpecificationMiningError("mining receipts require source provenance")
        _bounded(self, "SpecificationMiningReceipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_cids": self.source_cids,
            "candidate_cids": self.candidate_cids,
            "counterexample_cids": self.counterexample_cids,
            "invariant_candidate_cids": self.invariant_candidate_cids,
            "refused_property_ids": self.refused_property_ids,
            "refusal_classes": self.refusal_classes,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpecificationMiningReceipt:
        fields = (
            "bindings",
            "source_cids",
            "candidate_cids",
            "counterexample_cids",
            "invariant_candidate_cids",
            "refused_property_ids",
            "refusal_classes",
            "state",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GroupedClaim:
    property_id: str
    property_kind: PropertyKind
    binding: str
    operator: ConditionOperator
    operand: Any
    extras: Mapping[str, Any]
    evidence: tuple[SpecificationEvidence, ...]
    source_kinds: tuple[SpecificationSourceKind, ...]
    supporting_count: int
    passing_test_count: int
    source_cids: tuple[str, ...]


@dataclass(frozen=True)
class GroupedConflict:
    property_id: str
    property_kind: PropertyKind
    binding: str
    operator: ConditionOperator
    operands: tuple[Any, ...]
    source_cids: tuple[str, ...]
    evidence: tuple[SpecificationEvidence, ...]
    violation_class: str = "conflicting_evidence"


@dataclass(frozen=True)
class SpecificationMiningResult:
    receipt: SpecificationMiningReceipt
    candidates: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    invariant_candidates: tuple[Any, ...] = ()

    @property
    def refused_property_ids(self) -> tuple[str, ...]:
        return self.receipt.refused_property_ids


def _proposal_from_fields(
    *,
    property_kind: PropertyKind | str,
    binding: str,
    operator: ConditionOperator | str,
    operand: Any,
    occurrence_count: int = 1,
    passing_test_count: int = 0,
    extras: Mapping[str, Any] | None = None,
) -> PropertyProposal:
    return PropertyProposal(
        property_kind=property_kind,
        binding=binding,
        operator=operator,
        operand=operand,
        occurrence_count=occurrence_count,
        passing_test_count=passing_test_count,
        extras=dict(extras or {}),
    )


def _fact_int(facts: Mapping[str, Any], name: str, default: int = 0) -> int:
    if name not in facts:
        return default
    return _nonnegative_int(facts[name], name, maximum=MAX_SUPPORTING_COUNT)


def _claim_counts(facts: Mapping[str, Any]) -> tuple[int, int]:
    """Return occurrence and passing-test counts without mixing those signals.

    Absence (``occurrence_count == 0``) yields no claim. Passing tests are
    recorded separately and do not invent a frequency of one.
    """

    passing = _fact_int(facts, "passing_test_count", 0)
    if "occurrence_count" in facts:
        occurrence = _fact_int(facts, "occurrence_count")
    elif passing > 0:
        occurrence = 0
    else:
        occurrence = 1
    return occurrence, passing


def _derive_proposals(source: SpecificationSource) -> tuple[PropertyProposal, ...]:
    if source.proposals:
        return source.proposals
    facts = source.facts
    if "proposals" in facts:
        raw = facts["proposals"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray, memoryview)):
            raise SpecificationMiningError("facts.proposals must be a sequence")
        if len(raw) > MAX_PROPOSALS_PER_SOURCE:
            raise ProcedureBoundsError("facts.proposals exceeds its item bound")
        return tuple(
            PropertyProposal.from_mapping(item) if isinstance(item, Mapping) else item
            for item in raw
        )

    derived: list[PropertyProposal] = []
    occurrence, passing = _claim_counts(facts)
    if occurrence == 0 and passing == 0:
        return ()

    def add(
        kind: PropertyKind,
        binding: str,
        operator: ConditionOperator,
        operand: Any,
        extras: Mapping[str, Any] | None = None,
        passing_tests: int = 0,
    ) -> None:
        derived.append(
            _proposal_from_fields(
                property_kind=kind,
                binding=binding,
                operator=operator,
                operand=operand,
                occurrence_count=occurrence,
                passing_test_count=passing_tests or passing,
                extras=extras,
            )
        )

    if "property_kind" in facts:
        kind = _enum(facts["property_kind"], PropertyKind, "property_kind")
        operator = _enum(
            facts.get("operator", _default_operator(kind).value),
            ConditionOperator,
            "operator",
        )
        extras = facts.get("extras") if isinstance(facts.get("extras"), Mapping) else {}
        extras = dict(extras)
        if kind is PropertyKind.FRAME and "unchanged_paths" in facts and "unchanged_paths" not in extras:
            extras["unchanged_paths"] = facts["unchanged_paths"]
        if kind is PropertyKind.EFFECT and "targets" in facts and "targets" not in extras:
            extras["targets"] = facts["targets"]
        if kind is PropertyKind.ROLLBACK and "step_ids" in facts and "step_ids" not in extras:
            extras["step_ids"] = facts["step_ids"]
        operand = facts.get("operand")
        if operand is None:
            operand = _default_operand(kind, source, facts)
        add(kind, _text(facts.get("binding", _default_binding(kind, facts)), "binding"), operator, operand, extras)

    present = {item.property_kind for item in derived}
    if "effect_class" in facts and PropertyKind.EFFECT not in present:
        add(
            PropertyKind.EFFECT,
            _text(facts.get("effect_id", facts.get("binding", "effect.declared")), "binding"),
            ConditionOperator.EQUALS,
            facts["effect_class"],
            {"targets": facts.get("targets", ())},
        )
    if "idempotency_class" in facts and PropertyKind.IDEMPOTENCY not in present:
        add(
            PropertyKind.IDEMPOTENCY,
            _text(facts.get("step_id", facts.get("binding", "step.idempotency")), "binding"),
            ConditionOperator.EQUALS,
            facts["idempotency_class"],
        )
    if "step_ids" in facts and PropertyKind.ORDER not in present and PropertyKind.ROLLBACK not in present:
        if source.source_kind in {
            SpecificationSourceKind.OPERATION_CONTRACT,
            SpecificationSourceKind.ADMITTED_TRACE,
        } or "property_kind" not in facts:
            add(
                PropertyKind.ORDER,
                _text(facts.get("order_binding", "step.order"), "binding"),
                ConditionOperator.EQUALS,
                facts["step_ids"],
                {"step_ids": facts["step_ids"]},
            )
    resource = {name: facts[name] for name in _RESOURCE_FIELDS if name in facts}
    if resource and PropertyKind.RESOURCE not in present:
        add(
            PropertyKind.RESOURCE,
            _text(facts.get("resource_binding", "resources.envelope"), "binding"),
            ConditionOperator.EQUALS,
            resource,
        )
    if "exact_target_cid" in facts and PropertyKind.ROLLBACK not in present:
        add(
            PropertyKind.ROLLBACK,
            _text(facts.get("rollback_id", facts.get("binding", "rollback.declared")), "binding"),
            ConditionOperator.EQUALS,
            facts["exact_target_cid"],
            {
                "step_ids": facts.get("rollback_step_ids", facts.get("step_ids", ())),
                "trigger_effect_ids": facts.get("trigger_effect_ids", ()),
            },
        )
    if "unchanged_paths" in facts and PropertyKind.FRAME not in present:
        add(
            PropertyKind.FRAME,
            _text(facts.get("binding", "frame.scope"), "binding"),
            ConditionOperator.EQUALS,
            facts["unchanged_paths"],
            {"unchanged_paths": facts["unchanged_paths"]},
        )
    freshness_requested = (
        PropertyKind.FRESHNESS not in present
        and source.source_kind
        in {
            SpecificationSourceKind.RUNTIME_CHECK,
            SpecificationSourceKind.ADMITTED_TRACE,
            SpecificationSourceKind.TYPES,
        }
        and ("tree_id" in facts or "repository_commit" in facts or facts.get("property_kind") == "freshness")
    )
    if freshness_requested:
        add(
            PropertyKind.FRESHNESS,
            _text(facts.get("freshness_binding", "binding:world_state"), "binding"),
            ConditionOperator.CURRENT,
            {
                "tree_id": facts.get("tree_id", source.bindings.tree_id),
                "repository_commit": facts.get(
                    "repository_commit", source.bindings.repository_commit
                ),
            },
        )

    if "property_kind" not in facts:
        present = {item.property_kind for item in derived}
        for item in _default_kind_proposals(source, occurrence, passing):
            if item.property_kind not in present:
                derived.append(item)
                present.add(item.property_kind)
    if not derived:
        if source.source_kind is SpecificationSourceKind.REJECTED_TRACE:
            raise SpecificationMiningError("rejected traces must name the contradicted claim")
        if source.source_kind is SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION:
            raise SpecificationMiningError("documentation sources must name a bounded property")
    return tuple(derived)


def _default_operator(kind: PropertyKind) -> ConditionOperator:
    defaults = {
        PropertyKind.PRECONDITION: ConditionOperator.EXISTS,
        PropertyKind.POSTCONDITION: ConditionOperator.ADMITTED,
        PropertyKind.INVARIANT: ConditionOperator.EQUALS,
        PropertyKind.FRAME: ConditionOperator.EQUALS,
        PropertyKind.EFFECT: ConditionOperator.EQUALS,
        PropertyKind.RESOURCE: ConditionOperator.EQUALS,
        PropertyKind.ORDER: ConditionOperator.EQUALS,
        PropertyKind.IDEMPOTENCY: ConditionOperator.EQUALS,
        PropertyKind.ROLLBACK: ConditionOperator.EQUALS,
        PropertyKind.FRESHNESS: ConditionOperator.CURRENT,
    }
    return defaults[kind]


def _default_binding(kind: PropertyKind, facts: Mapping[str, Any]) -> str:
    defaults = {
        PropertyKind.PRECONDITION: "binding:precondition",
        PropertyKind.POSTCONDITION: "local:test-result",
        PropertyKind.INVARIANT: "binding:invariant",
        PropertyKind.FRAME: "frame.scope",
        PropertyKind.EFFECT: str(facts.get("effect_id", "effect.declared")),
        PropertyKind.RESOURCE: "resources.envelope",
        PropertyKind.ORDER: "step.order",
        PropertyKind.IDEMPOTENCY: str(facts.get("step_id", "step.idempotency")),
        PropertyKind.ROLLBACK: str(facts.get("rollback_id", "rollback.declared")),
        PropertyKind.FRESHNESS: "binding:world_state",
    }
    return defaults[kind]


def _default_operand(
    kind: PropertyKind, source: SpecificationSource, facts: Mapping[str, Any]
) -> Any:
    if kind is PropertyKind.FRAME:
        return facts.get("unchanged_paths", ())
    if kind is PropertyKind.EFFECT:
        return facts.get("effect_class")
    if kind is PropertyKind.IDEMPOTENCY:
        return facts.get("idempotency_class")
    if kind is PropertyKind.ORDER:
        return facts.get("step_ids", ())
    if kind is PropertyKind.RESOURCE:
        return {name: facts[name] for name in _RESOURCE_FIELDS if name in facts}
    if kind is PropertyKind.ROLLBACK:
        return facts.get("exact_target_cid")
    if kind is PropertyKind.FRESHNESS:
        return {
            "tree_id": facts.get("tree_id", source.bindings.tree_id),
            "repository_commit": facts.get(
                "repository_commit", source.bindings.repository_commit
            ),
        }
    return facts.get("operand")


def _default_kind_proposals(
    source: SpecificationSource,
    occurrence: int,
    passing: int,
) -> tuple[PropertyProposal, ...]:
    facts = source.facts
    kind = source.source_kind
    if kind is SpecificationSourceKind.TYPES:
        binding = facts.get("binding")
        if not binding:
            return ()
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.PRECONDITION,
                binding=_text(binding, "binding"),
                operator=ConditionOperator.EXISTS,
                operand=facts.get("operand"),
                occurrence_count=occurrence,
            ),
        )
    if kind is SpecificationSourceKind.TEST:
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.POSTCONDITION,
                binding=_text(facts.get("binding", "local:test-result"), "binding"),
                operator=ConditionOperator.ADMITTED,
                operand=facts.get("operand"),
                occurrence_count=occurrence,
                passing_test_count=passing or _fact_int(facts, "passing_test_count", 1),
            ),
        )
    if kind is SpecificationSourceKind.PROOF_OBLIGATION:
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.INVARIANT,
                binding=_text(facts.get("binding", "binding:invariant"), "binding"),
                operator=_enum(
                    facts.get("operator", ConditionOperator.EQUALS.value),
                    ConditionOperator,
                    "operator",
                ),
                operand=facts.get("operand"),
                occurrence_count=occurrence,
            ),
        )
    if kind is SpecificationSourceKind.RUNTIME_CHECK:
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.INVARIANT,
                binding=_text(facts.get("binding", "binding:runtime-invariant"), "binding"),
                operator=_enum(
                    facts.get("operator", ConditionOperator.EQUALS.value),
                    ConditionOperator,
                    "operator",
                ),
                operand=facts.get("operand"),
                occurrence_count=occurrence,
            ),
        )
    if kind is SpecificationSourceKind.MUTANT:
        paths = facts.get("unchanged_paths")
        if not paths:
            return ()
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.FRAME,
                binding=_text(facts.get("binding", "frame.scope"), "binding"),
                operator=ConditionOperator.EQUALS,
                operand=paths,
                occurrence_count=occurrence,
                extras={"unchanged_paths": paths},
            ),
        )
    if kind is SpecificationSourceKind.FAILURE_SIGNATURE:
        if "exact_target_cid" in facts:
            return (
                _proposal_from_fields(
                    property_kind=PropertyKind.ROLLBACK,
                    binding=_text(facts.get("rollback_id", "rollback.failure"), "binding"),
                    operator=ConditionOperator.EQUALS,
                    operand=facts["exact_target_cid"],
                    occurrence_count=occurrence,
                    extras={
                        "step_ids": facts.get("rollback_step_ids", ()),
                        "trigger_effect_ids": facts.get("trigger_effect_ids", ()),
                    },
                ),
            )
        binding = facts.get("binding")
        if not binding:
            return ()
        return (
            _proposal_from_fields(
                property_kind=PropertyKind.PRECONDITION,
                binding=_text(binding, "binding"),
                operator=_enum(
                    facts.get("operator", ConditionOperator.NOT_EXISTS.value),
                    ConditionOperator,
                    "operator",
                ),
                operand=facts.get("operand"),
                occurrence_count=occurrence,
            ),
        )
    return ()


def extract_proposals(source: SpecificationSource) -> tuple[PropertyProposal, ...]:
    """Return the bounded claims implied by one admitted source."""

    if not isinstance(source, SpecificationSource):
        raise SpecificationMiningError("source must be SpecificationSource")
    proposals = _derive_proposals(source)
    if len(proposals) > MAX_PROPOSALS_PER_SOURCE:
        raise ProcedureBoundsError("source proposals exceed their item bound")
    return proposals


def validate_specification_sources(
    sources: Sequence[SpecificationSource],
) -> tuple[SpecificationSource, ...]:
    if not isinstance(sources, Sequence) or isinstance(
        sources, (str, bytes, bytearray, memoryview)
    ):
        raise SpecificationMiningError("sources must be a bounded sequence")
    if not sources:
        raise SpecificationMiningError("specification mining requires at least one admitted source")
    if len(sources) > MAX_MINING_SOURCES:
        raise ProcedureBoundsError("sources exceed their item bound")
    validated: list[SpecificationSource] = []
    bindings: ArtifactBindings | None = None
    seen_cids: set[str] = set()
    for source in sources:
        if not isinstance(source, SpecificationSource):
            raise SpecificationMiningError("sources must be SpecificationSource records")
        if bindings is None:
            bindings = source.bindings
        elif source.bindings != bindings:
            raise SpecificationMiningError("source exact bindings differ")
        if source.source_cid in seen_cids:
            raise SpecificationMiningError("source provenance contains a duplicate source_cid")
        seen_cids.add(source.source_cid)
        validated.append(source)
    return tuple(validated)


def _evidence_for(
    source: SpecificationSource, proposal: PropertyProposal
) -> SpecificationEvidence:
    return SpecificationEvidence(
        bindings=source.bindings,
        source_kind=source.source_kind,
        source_cid=source.source_cid,
        producer_id=source.producer_id,
        evidence_type=source.evidence_type,
        tier=_tier_for_source(source),
        property_id=proposal.property_id,
        admitted=source.admitted,
    )


def _freshness_violation(
    source: SpecificationSource, proposal: PropertyProposal
) -> str | None:
    if proposal.property_kind is not PropertyKind.FRESHNESS:
        return None
    operand = proposal.operand
    if not isinstance(operand, Mapping):
        return "stale_freshness"
    if (
        operand.get("tree_id") != source.bindings.tree_id
        or operand.get("repository_commit") != source.bindings.repository_commit
    ):
        return "stale_freshness"
    return None


def group_source_claims(
    sources: Sequence[SpecificationSource],
    *,
    kinds: frozenset[PropertyKind] | None = None,
) -> tuple[tuple[GroupedClaim, ...], tuple[GroupedConflict, ...]]:
    """Group extracted claims, refusing conflicting operands."""

    validated = validate_specification_sources(sources)
    grouped: dict[str, list[tuple[PropertyProposal, SpecificationEvidence, SpecificationSource]]] = {}
    stale: list[GroupedConflict] = []
    for source in validated:
        proposals = extract_proposals(source)
        for proposal in proposals:
            if kinds is not None and proposal.property_kind not in kinds:
                continue
            if proposal.occurrence_count == 0 and proposal.passing_test_count == 0:
                continue
            evidence = _evidence_for(source, proposal)
            freshness = _freshness_violation(source, proposal)
            if freshness is not None:
                stale.append(
                    GroupedConflict(
                        property_id=proposal.property_id,
                        property_kind=proposal.property_kind,
                        binding=proposal.binding,
                        operator=proposal.operator,
                        operands=(proposal.operand, _freeze(
                            {
                                "tree_id": source.bindings.tree_id,
                                "repository_commit": source.bindings.repository_commit,
                            },
                            "operand",
                        )),
                        source_cids=(source.source_cid,),
                        evidence=(evidence,),
                        violation_class=freshness,
                    )
                )
                continue
            grouped.setdefault(proposal.property_id, []).append((proposal, evidence, source))

    claims: list[GroupedClaim] = []
    conflicts: list[GroupedConflict] = list(stale)
    for property_id, items in grouped.items():
        identities: dict[bytes, list[tuple[PropertyProposal, SpecificationEvidence, SpecificationSource]]] = {}
        for item in items:
            identities.setdefault(item[0].claim_identity, []).append(item)
        first = items[0][0]
        if len(identities) > 1:
            operands = tuple(group[0][0].operand for group in identities.values())
            evidence = tuple(item[1] for item in items)
            source_cids = tuple(dict.fromkeys(item[2].source_cid for item in items))
            conflicts.append(
                GroupedConflict(
                    property_id=property_id,
                    property_kind=first.property_kind,
                    binding=first.binding,
                    operator=first.operator,
                    operands=operands,
                    source_cids=source_cids,
                    evidence=evidence,
                )
            )
            continue
        agreeing = next(iter(identities.values()))
        evidence = tuple(item[1] for item in agreeing)
        if len(evidence) > MAX_ITEMS:
            raise ProcedureBoundsError("merged evidence exceeds its item bound")
        source_kinds = tuple(dict.fromkeys(item[1].source_kind for item in agreeing))
        supporting = 0
        passing = 0
        for proposal, _, _ in agreeing:
            supporting += proposal.occurrence_count
            passing += proposal.passing_test_count
            if supporting > MAX_SUPPORTING_COUNT or passing > MAX_SUPPORTING_COUNT:
                raise ProcedureBoundsError("supporting counts exceed their numeric bound")
        if supporting == 0:
            # Passing-test-only agreement is not frequency; keep source cardinality.
            supporting = len(agreeing)
        representative = agreeing[0][0]
        claims.append(
            GroupedClaim(
                property_id=property_id,
                property_kind=representative.property_kind,
                binding=representative.binding,
                operator=representative.operator,
                operand=representative.operand,
                extras=representative.extras,
                evidence=evidence,
                source_kinds=source_kinds,
                supporting_count=supporting,
                passing_test_count=passing,
                source_cids=tuple(dict.fromkeys(item[2].source_cid for item in agreeing)),
            )
        )
    claims.sort(key=lambda item: item.property_id)
    conflicts.sort(key=lambda item: item.property_id)
    return tuple(claims), tuple(conflicts)


def claims_to_candidates(
    claims: Sequence[GroupedClaim],
    bindings: ArtifactBindings,
) -> tuple[SpecificationCandidate, ...]:
    return tuple(
        SpecificationCandidate(
            bindings=bindings,
            property_id=claim.property_id,
            property_kind=claim.property_kind,
            binding=claim.binding,
            operator=claim.operator,
            operand=claim.operand,
            evidence=claim.evidence,
            source_kinds=claim.source_kinds,
            supporting_count=claim.supporting_count,
            passing_test_count=claim.passing_test_count,
            extras=claim.extras,
            state=ArtifactState.CANDIDATE,
        )
        for claim in claims
    )


def conflicts_to_counterexamples(
    conflicts: Sequence[GroupedConflict],
    bindings: ArtifactBindings,
) -> tuple[SpecificationCounterexample, ...]:
    return tuple(
        SpecificationCounterexample(
            bindings=bindings,
            property_id=conflict.property_id,
            property_kind=conflict.property_kind,
            binding=conflict.binding,
            operator=conflict.operator,
            conflicting_source_cids=conflict.source_cids,
            conflicting_operands=conflict.operands,
            evidence_cids=tuple(item.content_id for item in conflict.evidence),
            violation_class=conflict.violation_class,
            state=ArtifactState.REJECTED,
        )
        for conflict in conflicts
    )


class SpecificationMiner:
    """Propose bounded candidate properties from admitted specification sources."""

    def mine(
        self,
        sources: Sequence[SpecificationSource],
        *,
        include_invariants: bool = True,
    ) -> SpecificationMiningResult:
        validated = validate_specification_sources(sources)
        claims, conflicts = group_source_claims(validated)
        bindings = validated[0].bindings
        candidates = claims_to_candidates(claims, bindings)
        counterexamples = conflicts_to_counterexamples(conflicts, bindings)
        invariant_candidates: tuple[Any, ...] = ()
        if include_invariants:
            from .invariant_mining import InvariantMiner

            invariant_candidates = InvariantMiner().mine(validated).candidates
        receipt = SpecificationMiningReceipt(
            bindings=bindings,
            source_cids=tuple(source.source_cid for source in validated),
            candidate_cids=tuple(item.content_id for item in candidates),
            counterexample_cids=tuple(item.content_id for item in counterexamples),
            invariant_candidate_cids=tuple(item.content_id for item in invariant_candidates),
            refused_property_ids=tuple(item.property_id for item in counterexamples),
            refusal_classes=tuple(item.violation_class for item in counterexamples),
            state=ArtifactState.CANDIDATE,
        )
        return SpecificationMiningResult(
            receipt=receipt,
            candidates=candidates,
            counterexamples=counterexamples,
            invariant_candidates=invariant_candidates,
        )


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecificationMiningError("specification JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise SpecificationMiningError("specification JSON cannot contain floating point values")


def parse_specification_source(value: Any) -> SpecificationSource:
    if isinstance(value, SpecificationSource):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SpecificationMiningError("specification bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise SpecificationMiningError("specification JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise SpecificationMiningError("specification source must be a mapping or JSON object")
    return SpecificationSource.from_dict(value)


__all__ = [
    "GroupedClaim",
    "GroupedConflict",
    "MAX_MINING_SOURCES",
    "MAX_PROPOSALS_PER_SOURCE",
    "PropertyKind",
    "PropertyProposal",
    "SpecificationCandidate",
    "SpecificationCounterexample",
    "SpecificationEvidence",
    "SpecificationMiner",
    "SpecificationMiningError",
    "SpecificationMiningReceipt",
    "SpecificationMiningResult",
    "SpecificationSource",
    "SpecificationSourceKind",
    "SpecificationTier",
    "claims_to_candidates",
    "conflicts_to_counterexamples",
    "extract_proposals",
    "group_source_claims",
    "parse_specification_source",
    "validate_specification_sources",
]
