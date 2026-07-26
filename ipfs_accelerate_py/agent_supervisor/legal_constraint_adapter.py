"""Deterministic, fail-closed LegalIR applicability compilation.

The shared IR adapter verifies and normalizes LegalIR artifacts.  This module
performs the next, deliberately separate step: it selects provisions using an
exact applicability query and compiles the selected norms, relationships,
assumptions, and proof obligations into a source-bound result.

Semantic retrieval is nomination-only.  It may annotate a result but never
limits the declarations inspected by this compiler and therefore cannot prove
that no provision applies.  Likewise, a legal permission is a constraint
result, not SecurityIR authorization or an execution grant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .decision_contracts import ApplicabilityFact
from .ir_adapters import (
    IRAdapterResult,
    IRAdapterStatus,
    IRNodeKind,
    NormalizedIRArtifact,
    NormalizedIRNode,
    NormalizedResultAuthority,
)
from .ir_registry import IRFamily


LEGAL_CONSTRAINT_ADAPTER_VERSION: Final[int] = 1
# Stable evidence key consumed by the ASI-G340 constraint-compilation goal.
# The receipt below always carries this key, including fail-closed receipts,
# so evidence assemblers cannot confuse an absent legal lane with a successful
# applicability computation.
LEGAL_APPLICABILITY_REQUIREMENT_ID: Final[str] = (
    "legal_constraint_adapter.LEGAL_APPLICABILITY_REQUIREMENT_ID"
)
LEGAL_APPLICABILITY_QUERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legal-applicability-query@1"
)
LEGAL_SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legal-source-binding@1"
)
LEGAL_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legal-constraint@1"
)
LEGAL_PROOF_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legal-proof-obligation@1"
)
LEGAL_COMPILATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/legal-compilation-result@1"
)

_SCOPE_FIELDS: Final[tuple[str, ...]] = (
    "jurisdiction",
    "subject",
    "principal",
    "action",
    "resource",
    "effect",
)
_SUPPORTED_FORMAL_VIEW_KINDS: Final[frozenset[str]] = frozenset(
    {
        "deontic",
        "temporal",
        "frame",
        "first_order",
        "first-order",
        "knowledge_graph",
        "knowledge-graph",
    }
)
_WILDCARDS: Final[frozenset[str]] = frozenset({"*", "any", "all"})
_MAX_ITEMS: Final[int] = 4096
_MAX_TEXT_BYTES: Final[int] = 8192


class LegalConstraintError(ValueError):
    """A LegalIR applicability contract is malformed."""


class LegalModality(str, Enum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    POWER = "power"
    EXCEPTION = "exception"


class LegalApplicabilityOutcome(str, Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    REVIEW_REQUIRED = "review_required"


class LegalCompilationStatus(str, Enum):
    COMPLETE = "complete"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"
    REVIEW_REQUIRED = "review_required"


_MODALITY_ALIASES: Final[Mapping[str, LegalModality]] = MappingProxyType(
    {
        "duty": LegalModality.OBLIGATION,
        "obligatory": LegalModality.OBLIGATION,
        "obligation": LegalModality.OBLIGATION,
        "forbidden": LegalModality.PROHIBITION,
        "prohibited": LegalModality.PROHIBITION,
        "prohibition": LegalModality.PROHIBITION,
        "permission": LegalModality.PERMISSION,
        "permitted": LegalModality.PERMISSION,
        "right": LegalModality.PERMISSION,
        "power": LegalModality.POWER,
        "legal_power": LegalModality.POWER,
        "exception": LegalModality.EXCEPTION,
    }
)


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise LegalConstraintError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise LegalConstraintError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise LegalConstraintError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > maximum:
        raise LegalConstraintError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _strings(
    value: Any,
    name: str,
    *,
    maximum: int = _MAX_ITEMS,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LegalConstraintError(f"{name} must be a sequence")
    if len(value) > maximum:
        raise LegalConstraintError(f"{name} exceeds its count bound")
    result = tuple(_text(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise LegalConstraintError(f"{name} must be unique and sorted")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _content_id(namespace: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _ids(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_text(value, name),)
    return _strings(value, name)


def _integer(value: Any, name: str, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise LegalConstraintError(f"{name} must be an integer")
    if value < 0:
        raise LegalConstraintError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class LegalApplicabilityQuery:
    """Exact facts and pinned root expected by one applicability decision."""

    legal_root_artifact_id: str
    legal_root_cid_v1: str
    legal_root_supervisor_digest: str
    jurisdiction: str
    subject: str
    principal: str
    action: str
    resource: str
    effect: str
    effective_at_ms: int
    applicability_facts: tuple[ApplicabilityFact, ...] = ()
    semantic_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "legal_root_artifact_id",
            "legal_root_cid_v1",
            "legal_root_supervisor_digest",
            *_SCOPE_FIELDS,
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        moment = _integer(self.effective_at_ms, "effective_at_ms")
        assert moment is not None
        object.__setattr__(self, "effective_at_ms", moment)
        facts: list[ApplicabilityFact] = []
        if isinstance(self.applicability_facts, (str, bytes)) or not isinstance(
            self.applicability_facts, Sequence
        ):
            raise LegalConstraintError("applicability_facts must be a sequence")
        for item in self.applicability_facts:
            if isinstance(item, ApplicabilityFact):
                facts.append(item)
            elif isinstance(item, Mapping):
                facts.append(ApplicabilityFact.from_dict(item))
            else:
                raise LegalConstraintError(
                    "applicability_facts must contain ApplicabilityFact records"
                )
        normalized_facts = tuple(sorted(facts, key=lambda item: item.fact_id))
        if len({item.fact_id for item in normalized_facts}) != len(normalized_facts):
            raise LegalConstraintError("applicability fact IDs must be unique")
        if any(
            not item.applies_at(self.effective_at_ms, self.jurisdiction)
            for item in normalized_facts
        ):
            raise LegalConstraintError(
                "applicability facts must match the exact jurisdiction and time"
            )
        object.__setattr__(self, "applicability_facts", normalized_facts)
        object.__setattr__(
            self,
            "semantic_candidate_ids",
            _strings(self.semantic_candidate_ids, "semantic_candidate_ids"),
        )

    @property
    def exact_scope(self) -> Mapping[str, str]:
        return MappingProxyType(
            {name: getattr(self, name) for name in _SCOPE_FIELDS}
        )

    @property
    def fact_ids(self) -> frozenset[str]:
        return frozenset(item.fact_id for item in self.applicability_facts)

    @property
    def facts_by_predicate(self) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
        result: dict[str, list[Mapping[str, Any]]] = {}
        for item in self.applicability_facts:
            result.setdefault(item.predicate, []).append(item.value)
        return MappingProxyType(
            {
                key: tuple(values)
                for key, values in sorted(result.items())
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LEGAL_APPLICABILITY_QUERY_SCHEMA,
            "adapter_version": LEGAL_CONSTRAINT_ADAPTER_VERSION,
            "legal_root_artifact_id": self.legal_root_artifact_id,
            "legal_root_cid_v1": self.legal_root_cid_v1,
            "legal_root_supervisor_digest": self.legal_root_supervisor_digest,
            **dict(self.exact_scope),
            "effective_at_ms": self.effective_at_ms,
            "applicability_facts": [
                item.to_dict() for item in self.applicability_facts
            ],
            "semantic_candidate_ids": list(self.semantic_candidate_ids),
            "semantic_candidates_are_authority": False,
        }
        payload["content_id"] = _content_id("legal-applicability-query", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]


@dataclass(frozen=True)
class LegalSourceBinding:
    provision_id: str
    legal_root_artifact_id: str
    legal_root_cid_v1: str
    legal_root_supervisor_digest: str
    source_references: tuple[Mapping[str, Any], ...]
    provenance_references: tuple[Mapping[str, Any], ...]
    formal_view_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEGAL_SOURCE_BINDING_SCHEMA,
            "adapter_version": LEGAL_CONSTRAINT_ADAPTER_VERSION,
            "provision_id": self.provision_id,
            "legal_root_artifact_id": self.legal_root_artifact_id,
            "legal_root_cid_v1": self.legal_root_cid_v1,
            "legal_root_supervisor_digest": self.legal_root_supervisor_digest,
            "source_references": [_plain(item) for item in self.source_references],
            "provenance_references": [
                _plain(item) for item in self.provenance_references
            ],
            "formal_view_ids": list(self.formal_view_ids),
        }


@dataclass(frozen=True)
class LegalConstraint:
    provision_id: str
    modality: LegalModality | None
    outcome: LegalApplicabilityOutcome
    mandatory: bool
    active: bool
    precedence: int
    effective_from_ms: int | None
    effective_until_ms: int | None
    exact_scope: Mapping[str, tuple[str, ...]]
    exception_to: tuple[str, ...]
    exception_ids: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    supersedes: tuple[str, ...]
    defeated_by: tuple[str, ...]
    proof_obligation_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    source_binding: LegalSourceBinding

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LEGAL_CONSTRAINT_SCHEMA,
            "adapter_version": LEGAL_CONSTRAINT_ADAPTER_VERSION,
            "provision_id": self.provision_id,
            "modality": self.modality.value if self.modality is not None else None,
            "outcome": self.outcome.value,
            "mandatory": self.mandatory,
            "active": self.active,
            "precedence": self.precedence,
            "effective_from_ms": self.effective_from_ms,
            "effective_until_ms": self.effective_until_ms,
            "exact_scope": {
                key: list(value) for key, value in sorted(self.exact_scope.items())
            },
            "exception_to": list(self.exception_to),
            "exception_ids": list(self.exception_ids),
            "conflicts_with": list(self.conflicts_with),
            "supersedes": list(self.supersedes),
            "defeated_by": list(self.defeated_by),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "assumption_ids": list(self.assumption_ids),
            "reason_codes": list(self.reason_codes),
            "source_binding": self.source_binding.to_dict(),
            "grants_security_authorization": False,
            "grants_execution_authority": False,
        }
        payload["content_id"] = _content_id("legal-constraint", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]


@dataclass(frozen=True)
class CompiledLegalProofObligation:
    obligation_id: str
    provision_ids: tuple[str, ...]
    required: bool
    discharged: bool
    source_binding: LegalSourceBinding

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEGAL_PROOF_OBLIGATION_SCHEMA,
            "adapter_version": LEGAL_CONSTRAINT_ADAPTER_VERSION,
            "obligation_id": self.obligation_id,
            "provision_ids": list(self.provision_ids),
            "required": self.required,
            "discharged": self.discharged,
            "source_binding": self.source_binding.to_dict(),
        }


@dataclass(frozen=True)
class LegalCompilationResult:
    status: LegalCompilationStatus
    outcome: LegalApplicabilityOutcome
    query_id: str
    legal_root_artifact_id: str
    legal_root_cid_v1: str
    legal_root_supervisor_digest: str
    constraints: tuple[LegalConstraint, ...]
    selected_formal_view_ids: tuple[str, ...]
    assumptions: tuple[LegalSourceBinding, ...]
    proof_obligations: tuple[CompiledLegalProofObligation, ...]
    reason_codes: tuple[str, ...]
    semantic_candidate_ids: tuple[str, ...]
    authoritative_scan_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.status, LegalCompilationStatus):
            object.__setattr__(
                self, "status", LegalCompilationStatus(str(self.status))
            )
        if not isinstance(self.outcome, LegalApplicabilityOutcome):
            object.__setattr__(
                self,
                "outcome",
                LegalApplicabilityOutcome(str(self.outcome)),
            )

    def _with_outcome(
        self, outcome: LegalApplicabilityOutcome
    ) -> tuple[LegalConstraint, ...]:
        return tuple(item for item in self.constraints if item.outcome is outcome)

    @property
    def applicable(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.APPLICABLE)

    @property
    def inapplicable(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.INAPPLICABLE)

    @property
    def unknown(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.UNKNOWN)

    @property
    def conflicting(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.CONFLICTING)

    @property
    def expired(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.EXPIRED)

    @property
    def superseded(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.SUPERSEDED)

    @property
    def review_required(self) -> tuple[LegalConstraint, ...]:
        return self._with_outcome(LegalApplicabilityOutcome.REVIEW_REQUIRED)

    @property
    def obligations(self) -> tuple[LegalConstraint, ...]:
        return tuple(
            item
            for item in self.applicable
            if item.active and item.modality is LegalModality.OBLIGATION
        )

    @property
    def prohibitions(self) -> tuple[LegalConstraint, ...]:
        return tuple(
            item
            for item in self.applicable
            if item.active and item.modality is LegalModality.PROHIBITION
        )

    @property
    def permissions(self) -> tuple[LegalConstraint, ...]:
        return tuple(
            item
            for item in self.applicable
            if item.active and item.modality is LegalModality.PERMISSION
        )

    @property
    def powers(self) -> tuple[LegalConstraint, ...]:
        return tuple(
            item
            for item in self.applicable
            if item.active and item.modality is LegalModality.POWER
        )

    @property
    def exceptions(self) -> tuple[LegalConstraint, ...]:
        return tuple(
            item
            for item in self.applicable
            if item.active and item.modality is LegalModality.EXCEPTION
        )

    @property
    def legal_constraints_complete(self) -> bool:
        return self.status in {
            LegalCompilationStatus.COMPLETE,
            LegalCompilationStatus.PROHIBITED,
        }

    @property
    def complete(self) -> bool:
        return self.legal_constraints_complete

    @property
    def successful(self) -> bool:
        """The legal applicability computation completed without ambiguity."""

        return self.legal_constraints_complete

    @property
    def accepted(self) -> bool:
        """Whether the legal lane is complete and does not prohibit the action."""

        return self.status is LegalCompilationStatus.COMPLETE

    @property
    def legally_permitted(self) -> bool:
        return (
            self.status is LegalCompilationStatus.COMPLETE
            and not self.prohibitions
            and bool(self.permissions or self.powers)
        )

    @property
    def fail_closed(self) -> bool:
        return (
            self.status is not LegalCompilationStatus.COMPLETE
            or not self.authoritative_scan_complete
        )

    @property
    def grants_security_authorization(self) -> bool:
        return False

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def action_admitted(self) -> bool:
        # Admission belongs to the later execution-permit compiler and requires
        # an exact SecurityIR authorization decision.
        return False

    admits_action = action_admitted

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LEGAL_COMPILATION_RESULT_SCHEMA,
            "adapter_version": LEGAL_CONSTRAINT_ADAPTER_VERSION,
            "requirement_id": LEGAL_APPLICABILITY_REQUIREMENT_ID,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "query_id": self.query_id,
            "legal_root_artifact_id": self.legal_root_artifact_id,
            "legal_root_cid_v1": self.legal_root_cid_v1,
            "legal_root_supervisor_digest": self.legal_root_supervisor_digest,
            "constraints": [item.to_dict() for item in self.constraints],
            "selected_formal_view_ids": list(self.selected_formal_view_ids),
            "assumptions": [item.to_dict() for item in self.assumptions],
            "proof_obligations": [
                item.to_dict() for item in self.proof_obligations
            ],
            "reason_codes": list(self.reason_codes),
            "semantic_candidate_ids": list(self.semantic_candidate_ids),
            "semantic_candidates_are_authority": False,
            "authoritative_scan_complete": self.authoritative_scan_complete,
            "legally_permitted": self.legally_permitted,
            "grants_security_authorization": False,
            "grants_execution_authority": False,
            "action_admitted": False,
        }
        payload["content_id"] = _content_id("legal-compilation-result", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]


def _node_values(node: NormalizedIRNode) -> Mapping[str, Any]:
    attributes = node.attributes
    nested = attributes.get("applicability", attributes.get("scope", {}))
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise LegalConstraintError("provision applicability must be an object")
    result = dict(attributes)
    result.update(nested)
    return result


def _scope(
    values: Mapping[str, Any],
) -> tuple[Mapping[str, tuple[str, ...]], frozenset[str]]:
    universal = frozenset(
        _ids(values.get("universal_fields", values.get("applies_to_all", ())), "universal_fields")
    )
    if not universal.issubset(_SCOPE_FIELDS):
        raise LegalConstraintError("universal_fields contains an unsupported field")
    result: dict[str, tuple[str, ...]] = {}
    for field_name in _SCOPE_FIELDS:
        value = values.get(field_name)
        if value is None:
            value = values.get(f"{field_name}s")
        if value is None:
            continue
        members = _ids(value, field_name)
        if not members or any(item.lower() in _WILDCARDS for item in members):
            raise LegalConstraintError(
                f"{field_name} must contain explicit values, not a wildcard"
            )
        result[field_name] = members
    return MappingProxyType(result), universal


def _modality(node: NormalizedIRNode, values: Mapping[str, Any]) -> LegalModality:
    raw = values.get("modality", values.get("deontic_modality"))
    if raw is None:
        raw = node.declaration_kind
    if isinstance(raw, Mapping):
        raw = raw.get("kind", raw.get("modality"))
    if not isinstance(raw, str):
        raise LegalConstraintError("legal modality must be a string")
    modality = _MODALITY_ALIASES.get(raw.strip().lower())
    if modality is None:
        raise LegalConstraintError(f"unsupported legal modality: {raw}")
    return modality


def _binding(
    artifact: NormalizedIRArtifact,
    node: NormalizedIRNode,
    formal_view_ids: tuple[str, ...] = (),
) -> LegalSourceBinding:
    return LegalSourceBinding(
        provision_id=node.node_id,
        legal_root_artifact_id=artifact.root_artifact_id,
        legal_root_cid_v1=artifact.root_cid_v1,
        legal_root_supervisor_digest=artifact.root_supervisor_digest,
        source_references=node.source_references,
        provenance_references=node.provenance_references,
        formal_view_ids=formal_view_ids,
    )


def _initial_constraint(
    artifact: NormalizedIRArtifact,
    node: NormalizedIRNode,
    query: LegalApplicabilityQuery,
    formal_views: Mapping[str, NormalizedIRNode],
) -> LegalConstraint:
    mandatory = True
    values: Mapping[str, Any] = node.attributes
    try:
        values = _node_values(node)
        raw_mandatory = values.get("mandatory_applicability", True)
        if not isinstance(raw_mandatory, bool):
            raise LegalConstraintError(
                "mandatory_applicability must be a boolean"
            )
        mandatory = raw_mandatory
        exact_scope, universal = _scope(values)
    except LegalConstraintError:
        return LegalConstraint(
            provision_id=node.node_id,
            modality=None,
            outcome=LegalApplicabilityOutcome.REVIEW_REQUIRED,
            mandatory=mandatory,
            active=False,
            precedence=0,
            effective_from_ms=None,
            effective_until_ms=None,
            exact_scope=MappingProxyType({}),
            exception_to=(),
            exception_ids=(),
            conflicts_with=(),
            supersedes=(),
            defeated_by=(),
            proof_obligation_ids=(),
            assumption_ids=(),
            reason_codes=("malformed_applicability",),
            source_binding=_binding(artifact, node),
        )

    start = _integer(
        values.get("effective_from_ms", values.get("effective_from")),
        "effective_from_ms",
    )
    end = _integer(
        values.get("effective_until_ms", values.get("effective_until")),
        "effective_until_ms",
    )
    if start is not None and end is not None and end <= start:
        return LegalConstraint(
            provision_id=node.node_id,
            modality=None,
            outcome=LegalApplicabilityOutcome.REVIEW_REQUIRED,
            mandatory=mandatory,
            active=False,
            precedence=0,
            effective_from_ms=start,
            effective_until_ms=end,
            exact_scope=exact_scope,
            exception_to=(),
            exception_ids=(),
            conflicts_with=(),
            supersedes=(),
            defeated_by=(),
            proof_obligation_ids=(),
            assumption_ids=(),
            reason_codes=("invalid_effective_interval",),
            source_binding=_binding(artifact, node),
        )

    outcome = LegalApplicabilityOutcome.APPLICABLE
    reasons: list[str] = []
    if end is not None and query.effective_at_ms >= end:
        outcome = LegalApplicabilityOutcome.EXPIRED
        reasons.append("effective_interval_expired")
    elif start is not None and query.effective_at_ms < start:
        outcome = LegalApplicabilityOutcome.INAPPLICABLE
        reasons.append("not_yet_effective")
    else:
        missing = [
            name
            for name in _SCOPE_FIELDS
            if name not in exact_scope and name not in universal
        ]
        mismatched = [
            name
            for name, expected in exact_scope.items()
            if query.exact_scope[name] not in expected
        ]
        if mismatched:
            outcome = LegalApplicabilityOutcome.INAPPLICABLE
            reasons.extend(f"{name}_mismatch" for name in mismatched)
        elif missing:
            outcome = LegalApplicabilityOutcome.UNKNOWN
            reasons.extend(f"missing_{name}_selector" for name in missing)

    required_fact_ids = _ids(
        values.get("required_fact_ids", values.get("applicability_fact_ids")),
        "required_fact_ids",
    )
    absent_facts = sorted(set(required_fact_ids).difference(query.fact_ids))
    if (
        absent_facts
        and outcome
        not in {
            LegalApplicabilityOutcome.INAPPLICABLE,
            LegalApplicabilityOutcome.EXPIRED,
        }
    ):
        outcome = LegalApplicabilityOutcome.UNKNOWN
        reasons.append("missing_required_applicability_fact")

    conditions = values.get("conditions", {})
    if conditions is None:
        conditions = {}
    if not isinstance(conditions, Mapping):
        outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
        reasons.append("malformed_applicability_condition")
    elif outcome is LegalApplicabilityOutcome.APPLICABLE:
        for predicate, expected in sorted(conditions.items()):
            actual_values = query.facts_by_predicate.get(str(predicate), ())
            if not actual_values:
                outcome = LegalApplicabilityOutcome.UNKNOWN
                reasons.append("unresolved_applicability_condition")
                break
            if _plain(expected) not in tuple(_plain(item) for item in actual_values):
                outcome = LegalApplicabilityOutcome.INAPPLICABLE
                reasons.append("applicability_condition_false")
                break

    formal_view_ids = _ids(
        values.get("formal_view_ids", values.get("formal_views")),
        "formal_view_ids",
    )
    if outcome in {
        LegalApplicabilityOutcome.APPLICABLE,
        LegalApplicabilityOutcome.UNKNOWN,
    }:
        if (
            not node.grounded
            or not node.review_state.accepted
            or not node.trust_state.accepted
            or node.result_authority
            is not NormalizedResultAuthority.CONSTRAINT_INPUT
        ):
            outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
            reasons.append("provision_not_reviewed_trusted_authority")
        if not node.source_references or not node.provenance_references:
            outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
            reasons.append("missing_trusted_source_or_provenance")
        for view_id in formal_view_ids:
            view = formal_views.get(view_id)
            if view is None:
                outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
                reasons.append("missing_formal_view")
                continue
            view_kind = view.declaration_kind.lower()
            if view_kind not in _SUPPORTED_FORMAL_VIEW_KINDS:
                outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
                reasons.append("unsupported_formal_view")
            if not view.review_state.accepted or not view.trust_state.accepted:
                outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
                reasons.append("formal_view_requires_review")
            if not view.source_references or not view.provenance_references:
                outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
                reasons.append("formal_view_missing_source_or_provenance")

    modality: LegalModality | None = None
    if outcome not in {
        LegalApplicabilityOutcome.INAPPLICABLE,
        LegalApplicabilityOutcome.EXPIRED,
    }:
        try:
            modality = _modality(node, values)
        except LegalConstraintError:
            outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
            reasons.append("unsupported_modality")
    else:
        try:
            modality = _modality(node, values)
        except LegalConstraintError:
            # An exact mismatch or expired interval establishes that this
            # declaration cannot constrain this query without interpreting its
            # modality.
            modality = None

    precedence = _integer(
        values.get("precedence", values.get("priority")), "precedence", default=0
    )
    assert precedence is not None
    return LegalConstraint(
        provision_id=node.node_id,
        modality=modality,
        outcome=outcome,
        mandatory=mandatory,
        active=outcome is LegalApplicabilityOutcome.APPLICABLE,
        precedence=precedence,
        effective_from_ms=start,
        effective_until_ms=end,
        exact_scope=exact_scope,
        exception_to=_ids(values.get("exception_to"), "exception_to"),
        exception_ids=_ids(
            values.get("exception_ids", values.get("exceptions")),
            "exception_ids",
        ),
        conflicts_with=_ids(values.get("conflicts_with"), "conflicts_with"),
        supersedes=_ids(values.get("supersedes"), "supersedes"),
        defeated_by=(),
        proof_obligation_ids=_ids(
            values.get("proof_obligation_ids", values.get("obligation_ids")),
            "proof_obligation_ids",
        ),
        assumption_ids=_ids(values.get("assumption_ids"), "assumption_ids"),
        reason_codes=tuple(sorted(set(reasons or ["exact_scope_and_time_match"]))),
        source_binding=_binding(artifact, node, formal_view_ids),
    )


def _safe_initial_constraint(
    artifact: NormalizedIRArtifact,
    node: NormalizedIRNode,
    query: LegalApplicabilityQuery,
    formal_views: Mapping[str, NormalizedIRNode],
) -> LegalConstraint:
    try:
        return _initial_constraint(artifact, node, query, formal_views)
    except (LegalConstraintError, TypeError, ValueError):
        # Artifact normalization deliberately preserves unfamiliar compact
        # fields for downstream adapters.  A malformed LegalIR-specific field
        # therefore becomes an explicit closed finding, never an exception
        # that could be mistaken for absence.
        return LegalConstraint(
            provision_id=node.node_id,
            modality=None,
            outcome=LegalApplicabilityOutcome.REVIEW_REQUIRED,
            mandatory=True,
            active=False,
            precedence=0,
            effective_from_ms=None,
            effective_until_ms=None,
            exact_scope=MappingProxyType({}),
            exception_to=(),
            exception_ids=(),
            conflicts_with=(),
            supersedes=(),
            defeated_by=(),
            proof_obligation_ids=(),
            assumption_ids=(),
            reason_codes=("malformed_legal_declaration",),
            source_binding=_binding(artifact, node),
        )


def _opposed(left: LegalModality | None, right: LegalModality | None) -> bool:
    pair = frozenset((left, right))
    return pair in {
        frozenset((LegalModality.PROHIBITION, LegalModality.PERMISSION)),
        frozenset((LegalModality.PROHIBITION, LegalModality.POWER)),
        frozenset((LegalModality.PROHIBITION, LegalModality.OBLIGATION)),
    }


def _add_reason(item: LegalConstraint, *reasons: str) -> LegalConstraint:
    return replace(
        item,
        reason_codes=tuple(sorted(set((*item.reason_codes, *reasons)))),
    )


def _resolve_relationships(
    constraints: tuple[LegalConstraint, ...],
) -> tuple[LegalConstraint, ...]:
    by_id = {item.provision_id: item for item in constraints}

    # A referenced exception is mandatory closure, even when retrieval omitted
    # it.  Missing or unresolved exception applicability invalidates the norm.
    for item in tuple(by_id.values()):
        if item.outcome is not LegalApplicabilityOutcome.APPLICABLE:
            continue
        for exception_id in item.exception_ids:
            exception = by_id.get(exception_id)
            if (
                exception is None
                or exception.outcome
                in {
                    LegalApplicabilityOutcome.UNKNOWN,
                    LegalApplicabilityOutcome.CONFLICTING,
                    LegalApplicabilityOutcome.REVIEW_REQUIRED,
                }
            ):
                by_id[item.provision_id] = _add_reason(
                    replace(
                        item,
                        outcome=LegalApplicabilityOutcome.UNKNOWN,
                        active=False,
                    ),
                    "unresolved_exception",
                )
                break
        missing_conflicts = set(item.conflicts_with).difference(by_id)
        if missing_conflicts:
            by_id[item.provision_id] = _add_reason(
                replace(
                    by_id[item.provision_id],
                    outcome=LegalApplicabilityOutcome.UNKNOWN,
                    active=False,
                ),
                "unresolved_conflict_reference",
            )

    # Applicable exceptions defeat, but do not erase, their target.  Both
    # remain visible in the result.
    for exception in tuple(by_id.values()):
        if not (
            exception.outcome is LegalApplicabilityOutcome.APPLICABLE
            and exception.active
            and exception.modality is LegalModality.EXCEPTION
        ):
            continue
        if not exception.exception_to:
            by_id[exception.provision_id] = _add_reason(
                replace(
                    exception,
                    outcome=LegalApplicabilityOutcome.UNKNOWN,
                    active=False,
                ),
                "exception_missing_target",
            )
            continue
        for target_id in exception.exception_to:
            target = by_id.get(target_id)
            if target is None:
                by_id[exception.provision_id] = _add_reason(
                    replace(
                        by_id[exception.provision_id],
                        outcome=LegalApplicabilityOutcome.UNKNOWN,
                        active=False,
                    ),
                    "exception_target_missing",
                )
            elif target.outcome is LegalApplicabilityOutcome.APPLICABLE:
                by_id[target_id] = _add_reason(
                    replace(
                        target,
                        active=False,
                        defeated_by=tuple(
                            sorted(set((*target.defeated_by, exception.provision_id)))
                        ),
                    ),
                    "applicable_exception",
                )

    # Express supersession before conflict checking.
    for winner in tuple(by_id.values()):
        if winner.outcome is not LegalApplicabilityOutcome.APPLICABLE or not winner.active:
            continue
        for target_id in winner.supersedes:
            target = by_id.get(target_id)
            if target is None:
                by_id[winner.provision_id] = _add_reason(
                    replace(
                        by_id[winner.provision_id],
                        outcome=LegalApplicabilityOutcome.UNKNOWN,
                        active=False,
                    ),
                    "superseded_provision_missing",
                )
            elif target.outcome is LegalApplicabilityOutcome.APPLICABLE:
                by_id[target_id] = _add_reason(
                    replace(
                        target,
                        outcome=LegalApplicabilityOutcome.SUPERSEDED,
                        active=False,
                        defeated_by=tuple(
                            sorted(set((*target.defeated_by, winner.provision_id)))
                        ),
                    ),
                    "express_supersession",
                )

    def conflict_pairs() -> tuple[tuple[LegalConstraint, LegalConstraint], ...]:
        active = tuple(
            by_id[item]
            for item in sorted(by_id)
            if by_id[item].outcome is LegalApplicabilityOutcome.APPLICABLE
            and by_id[item].active
        )
        pairs: list[tuple[LegalConstraint, LegalConstraint]] = []
        for offset, left in enumerate(active):
            for right in active[offset + 1 :]:
                explicit = (
                    right.provision_id in left.conflicts_with
                    or left.provision_id in right.conflicts_with
                )
                # Both provisions have already matched this one exact query.
                # Different selector-set representations can still overlap at
                # the queried fact.
                if explicit or _opposed(left.modality, right.modality):
                    pairs.append((left, right))
        return tuple(pairs)

    # Resolve every strict-precedence edge as a set before considering ties.
    # This prevents provision-ID ordering from manufacturing a conflict that
    # a higher-precedence applicable norm resolves.
    defeated: dict[str, set[str]] = {}
    for left, right in conflict_pairs():
        if left.precedence == right.precedence:
            continue
        winner, loser = (
            (left, right)
            if left.precedence > right.precedence
            else (right, left)
        )
        defeated.setdefault(loser.provision_id, set()).add(winner.provision_id)
    for loser_id, winner_ids in sorted(defeated.items()):
        loser = by_id[loser_id]
        by_id[loser_id] = _add_reason(
            replace(
                loser,
                outcome=LegalApplicabilityOutcome.SUPERSEDED,
                active=False,
                defeated_by=tuple(
                    sorted(set((*loser.defeated_by, *winner_ids)))
                ),
            ),
            "higher_precedence_provision",
        )

    tied: dict[str, set[str]] = {}
    for left, right in conflict_pairs():
        if left.precedence != right.precedence:
            continue
        tied.setdefault(left.provision_id, set()).add(right.provision_id)
        tied.setdefault(right.provision_id, set()).add(left.provision_id)
    for provision_id in sorted(tied):
        item = by_id[provision_id]
        by_id[provision_id] = _add_reason(
            replace(
                item,
                outcome=LegalApplicabilityOutcome.CONFLICTING,
                active=False,
            ),
            "unresolved_equal_precedence_conflict",
        )
    return tuple(by_id[item] for item in sorted(by_id))


def _related_ids(node: NormalizedIRNode, keys: tuple[str, ...]) -> tuple[str, ...]:
    values = _node_values(node)
    for key in keys:
        if key in values:
            return _ids(values[key], key)
    return ()


def _compile_dependencies(
    artifact: NormalizedIRArtifact,
    constraints: tuple[LegalConstraint, ...],
) -> tuple[tuple[LegalSourceBinding, ...], tuple[CompiledLegalProofObligation, ...], tuple[str, ...]]:
    selected_ids = {
        item.provision_id
        for item in constraints
        if item.outcome is LegalApplicabilityOutcome.APPLICABLE
    }
    assumption_nodes = {item.node_id: item for item in artifact.assumptions}
    obligation_nodes = {item.node_id: item for item in artifact.obligations}
    required_assumptions = {
        assumption_id
        for item in constraints
        if item.provision_id in selected_ids
        for assumption_id in item.assumption_ids
    }
    required_obligations = {
        obligation_id
        for item in constraints
        if item.provision_id in selected_ids
        for obligation_id in item.proof_obligation_ids
    }
    reasons: set[str] = set()

    assumptions: list[LegalSourceBinding] = []
    for assumption_id in sorted(required_assumptions):
        node = assumption_nodes.get(assumption_id)
        if node is None:
            reasons.add("missing_required_assumption")
            continue
        assumptions.append(_binding(artifact, node))
        if (
            not node.review_state.accepted
            or not node.trust_state.accepted
            or not node.source_references
            or not node.provenance_references
        ):
            reasons.add("untrusted_required_assumption")
    for node in artifact.assumptions:
        provision_ids = _related_ids(
            node, ("provision_ids", "norm_ids", "applies_to")
        )
        if (
            not selected_ids.intersection(provision_ids)
            or node.node_id in {item.provision_id for item in assumptions}
        ):
            continue
        assumptions.append(_binding(artifact, node))
        if (
            not node.review_state.accepted
            or not node.trust_state.accepted
            or not node.source_references
            or not node.provenance_references
        ):
            reasons.add("untrusted_applicable_assumption")
    assumptions.sort(key=lambda item: item.provision_id)

    obligations: list[CompiledLegalProofObligation] = []
    for obligation_id in sorted(required_obligations):
        node = obligation_nodes.get(obligation_id)
        if node is None:
            reasons.add("missing_required_proof_obligation")
            continue
        provision_ids = _related_ids(
            node, ("provision_ids", "norm_ids", "applies_to")
        )
        if not provision_ids:
            provision_ids = tuple(
                sorted(
                    item.provision_id
                    for item in constraints
                    if obligation_id in item.proof_obligation_ids
                )
            )
        raw_discharged = _node_values(node).get("discharged", False)
        if not isinstance(raw_discharged, bool):
            reasons.add("malformed_proof_obligation")
            raw_discharged = False
        obligations.append(
            CompiledLegalProofObligation(
                obligation_id=obligation_id,
                provision_ids=provision_ids,
                required=True,
                discharged=raw_discharged,
                source_binding=_binding(artifact, node),
            )
        )
        if (
            not node.review_state.accepted
            or not node.trust_state.accepted
            or not node.source_references
            or not node.provenance_references
        ):
            reasons.add("missing_proof_obligation_source")

    # Also compile globally declared dependencies that explicitly bind an
    # applicable provision, even when the provision omits the reverse edge.
    for node in artifact.obligations:
        provision_ids = _related_ids(
            node, ("provision_ids", "norm_ids", "applies_to")
        )
        if not selected_ids.intersection(provision_ids):
            continue
        if node.node_id in {item.obligation_id for item in obligations}:
            continue
        raw_required = _node_values(node).get("required", True)
        raw_discharged = _node_values(node).get("discharged", False)
        if not isinstance(raw_required, bool) or not isinstance(raw_discharged, bool):
            reasons.add("malformed_proof_obligation")
            raw_required, raw_discharged = True, False
        obligations.append(
            CompiledLegalProofObligation(
                obligation_id=node.node_id,
                provision_ids=tuple(sorted(set(provision_ids))),
                required=raw_required,
                discharged=raw_discharged,
                source_binding=_binding(artifact, node),
            )
        )
        if (
            not node.review_state.accepted
            or not node.trust_state.accepted
            or not node.source_references
            or not node.provenance_references
        ):
            reasons.add("missing_proof_obligation_source")
    obligations.sort(key=lambda item: item.obligation_id)
    return tuple(assumptions), tuple(obligations), tuple(sorted(reasons))


def _failed_result(
    query: LegalApplicabilityQuery,
    *,
    reason: str,
    root_artifact_id: str = "",
    root_cid_v1: str = "",
    root_digest: str = "",
    outcome: LegalApplicabilityOutcome = LegalApplicabilityOutcome.REVIEW_REQUIRED,
) -> LegalCompilationResult:
    return LegalCompilationResult(
        status=(
            LegalCompilationStatus.UNKNOWN
            if outcome is LegalApplicabilityOutcome.UNKNOWN
            else LegalCompilationStatus.REVIEW_REQUIRED
        ),
        outcome=outcome,
        query_id=query.content_id,
        legal_root_artifact_id=root_artifact_id,
        legal_root_cid_v1=root_cid_v1,
        legal_root_supervisor_digest=root_digest,
        constraints=(),
        selected_formal_view_ids=(),
        assumptions=(),
        proof_obligations=(),
        reason_codes=(reason,),
        semantic_candidate_ids=query.semantic_candidate_ids,
        authoritative_scan_complete=False,
    )


class LegalConstraintAdapter:
    """Compile normalized, pinned LegalIR without importing a legal provider."""

    adapter_id: Final[str] = "supervisor-legal-constraint-adapter@1"

    def compile(
        self,
        artifact: NormalizedIRArtifact | IRAdapterResult | None,
        query: LegalApplicabilityQuery,
    ) -> LegalCompilationResult:
        if not isinstance(query, LegalApplicabilityQuery):
            raise LegalConstraintError("query must be a LegalApplicabilityQuery")
        if artifact is None:
            return _failed_result(query, reason="missing_trusted_legal_source")
        if isinstance(artifact, IRAdapterResult):
            if artifact.status is not IRAdapterStatus.NORMALIZED:
                assert artifact.failure is not None
                return _failed_result(
                    query,
                    reason=f"legal_ir_{artifact.failure.code.value}",
                )
            artifact = artifact.require_artifact()
        if not isinstance(artifact, NormalizedIRArtifact):
            raise LegalConstraintError(
                "artifact must be NormalizedIRArtifact, IRAdapterResult, or None"
            )
        if artifact.family is not IRFamily.LEGAL:
            return _failed_result(
                query,
                reason="unsupported_ir_family",
                root_artifact_id=artifact.root_artifact_id,
                root_cid_v1=artifact.root_cid_v1,
                root_digest=artifact.root_supervisor_digest,
            )
        if (
            artifact.root_artifact_id != query.legal_root_artifact_id
            or artifact.root_cid_v1 != query.legal_root_cid_v1
            or artifact.root_supervisor_digest
            != query.legal_root_supervisor_digest
        ):
            return _failed_result(
                query,
                reason="changed_legal_root",
                root_artifact_id=artifact.root_artifact_id,
                root_cid_v1=artifact.root_cid_v1,
                root_digest=artifact.root_supervisor_digest,
            )
        if (
            not artifact.review_state.accepted
            or not artifact.trust_state.accepted
            or artifact.declared_authority.value
            not in {"authoritative", "verified"}
        ):
            return _failed_result(
                query,
                reason="legal_root_requires_review",
                root_artifact_id=artifact.root_artifact_id,
                root_cid_v1=artifact.root_cid_v1,
                root_digest=artifact.root_supervisor_digest,
            )

        formal_views = {item.node_id: item for item in artifact.formal_views}
        constraints = tuple(
            _safe_initial_constraint(artifact, node, query, formal_views)
            for node in artifact.declarations
            if node.node_kind is IRNodeKind.DECLARATION
        )
        if not constraints:
            return _failed_result(
                query,
                reason="no_authoritative_legal_declarations",
                root_artifact_id=artifact.root_artifact_id,
                root_cid_v1=artifact.root_cid_v1,
                root_digest=artifact.root_supervisor_digest,
                outcome=LegalApplicabilityOutcome.UNKNOWN,
            )
        constraints = _resolve_relationships(constraints)
        assumptions, proof_obligations, dependency_reasons = _compile_dependencies(
            artifact, constraints
        )

        selected_view_ids = tuple(
            sorted(
                {
                    view_id
                    for item in constraints
                    if item.outcome
                    in {
                        LegalApplicabilityOutcome.APPLICABLE,
                        LegalApplicabilityOutcome.CONFLICTING,
                        LegalApplicabilityOutcome.SUPERSEDED,
                    }
                    for view_id in item.source_binding.formal_view_ids
                    if view_id in formal_views
                }
            )
        )
        mandatory_unknown = any(
            item.mandatory
            and item.outcome is LegalApplicabilityOutcome.UNKNOWN
            for item in constraints
        )
        has_review = any(
            item.outcome is LegalApplicabilityOutcome.REVIEW_REQUIRED
            for item in constraints
        ) or bool(dependency_reasons)
        has_conflict = any(
            item.outcome is LegalApplicabilityOutcome.CONFLICTING
            for item in constraints
        )
        has_prohibition = any(
            item.outcome is LegalApplicabilityOutcome.APPLICABLE
            and item.active
            and item.modality is LegalModality.PROHIBITION
            for item in constraints
        )
        reasons = set(dependency_reasons)
        reasons.update(
            reason
            for item in constraints
            if item.outcome
            in {
                LegalApplicabilityOutcome.UNKNOWN,
                LegalApplicabilityOutcome.CONFLICTING,
                LegalApplicabilityOutcome.REVIEW_REQUIRED,
            }
            for reason in item.reason_codes
        )
        if has_review:
            status = LegalCompilationStatus.REVIEW_REQUIRED
            outcome = LegalApplicabilityOutcome.REVIEW_REQUIRED
        elif has_conflict:
            status = LegalCompilationStatus.CONFLICTING
            outcome = LegalApplicabilityOutcome.CONFLICTING
        elif mandatory_unknown:
            status = LegalCompilationStatus.UNKNOWN
            outcome = LegalApplicabilityOutcome.UNKNOWN
        elif has_prohibition:
            status = LegalCompilationStatus.PROHIBITED
            outcome = LegalApplicabilityOutcome.APPLICABLE
            reasons.add("applicable_prohibition")
        else:
            status = LegalCompilationStatus.COMPLETE
            applicable = any(
                item.outcome is LegalApplicabilityOutcome.APPLICABLE
                for item in constraints
            )
            if applicable:
                outcome = LegalApplicabilityOutcome.APPLICABLE
            elif any(
                item.outcome is LegalApplicabilityOutcome.SUPERSEDED
                for item in constraints
            ):
                outcome = LegalApplicabilityOutcome.SUPERSEDED
            elif any(
                item.outcome is LegalApplicabilityOutcome.EXPIRED
                for item in constraints
            ):
                outcome = LegalApplicabilityOutcome.EXPIRED
            else:
                outcome = LegalApplicabilityOutcome.INAPPLICABLE
        return LegalCompilationResult(
            status=status,
            outcome=outcome,
            query_id=query.content_id,
            legal_root_artifact_id=artifact.root_artifact_id,
            legal_root_cid_v1=artifact.root_cid_v1,
            legal_root_supervisor_digest=artifact.root_supervisor_digest,
            constraints=constraints,
            selected_formal_view_ids=selected_view_ids,
            assumptions=assumptions,
            proof_obligations=proof_obligations,
            reason_codes=tuple(sorted(reasons)),
            semantic_candidate_ids=query.semantic_candidate_ids,
            authoritative_scan_complete=True,
        )


def compile_legal_constraints(
    artifact: NormalizedIRArtifact | IRAdapterResult | None,
    query: LegalApplicabilityQuery,
) -> LegalCompilationResult:
    """Compile with the default provider-free LegalIR constraint adapter."""

    return LegalConstraintAdapter().compile(artifact, query)


LegalConstraintRequest = LegalApplicabilityQuery
LegalConstraintResult = LegalCompilationResult
LegalApplicabilityStatus = LegalApplicabilityOutcome


__all__ = [
    "CompiledLegalProofObligation",
    "LEGAL_APPLICABILITY_QUERY_SCHEMA",
    "LEGAL_APPLICABILITY_REQUIREMENT_ID",
    "LEGAL_COMPILATION_RESULT_SCHEMA",
    "LEGAL_CONSTRAINT_ADAPTER_VERSION",
    "LEGAL_CONSTRAINT_SCHEMA",
    "LEGAL_PROOF_OBLIGATION_SCHEMA",
    "LEGAL_SOURCE_BINDING_SCHEMA",
    "LegalApplicabilityOutcome",
    "LegalApplicabilityQuery",
    "LegalApplicabilityStatus",
    "LegalCompilationResult",
    "LegalCompilationStatus",
    "LegalConstraint",
    "LegalConstraintAdapter",
    "LegalConstraintError",
    "LegalConstraintRequest",
    "LegalConstraintResult",
    "LegalModality",
    "LegalSourceBinding",
    "compile_legal_constraints",
]
