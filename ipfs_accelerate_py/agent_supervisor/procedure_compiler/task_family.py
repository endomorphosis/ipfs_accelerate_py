"""Task-family discovery baseline and live boundary validation.

PCPC-010 owns deterministic family discovery: classification uses goal
semantics, precondition, artifact, effect, tool, validation, failure,
postcondition, and rollback shapes.  Titles, embeddings, hints, and proposed
labels are never evidence, and insufficient shapes return unknown.  PCPC-011
owns the live boundary validator and rejection policy: every family must
declare complete boundary dimensions, and overgeneralization or an unsafe
near-match is a critical typed refusal with a persisted counterexample.
The immutable contract helpers continue to reject inconsistent boundaries,
memberships, and already-known counterexamples.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .contracts import (
    MAX_ITEMS,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ExecutionTrajectory,
    FamilyMembershipClass,
    ProcedureContractError,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyCounterexample,
    TaskFamilyMembership,
    _enum,
    _enums,
    _identifier,
    _strings,
    _text,
)


class TaskFamilyContractError(ProcedureContractError):
    """A task-family wire artifact violates its declared safe boundary."""


class TaskFamilyBoundaryError(TaskFamilyContractError):
    """A family boundary check produced a critical typed rejection."""

    def __init__(self, message: str, decision: BoundaryDecision | None = None) -> None:
        super().__init__(message)
        self.decision = decision


class BoundarySeverity(str, Enum):
    """Closed severity for family-boundary decisions."""

    NONE = "none"
    CRITICAL = "critical"


class BoundaryViolationClass(str, Enum):
    """Closed reasons a candidate or merge cannot remain inside a family."""

    INCOMPLETE_BOUNDARY = "incomplete-boundary"
    NEGATIVE_EXAMPLE = "negative-example"
    BOUNDARY_EXAMPLE = "boundary-example"
    UNKNOWN_CASE = "unknown-case"
    MEMBERSHIP_CONTRADICTION = "membership-contradiction"
    RISK_CEILING = "risk-ceiling"
    REPOSITORY = "repository"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    EFFECT = "effect"
    AUTHORITY = "authority-split"
    VALIDATION = "validation-split"
    ROLLBACK = "rollback-split"
    LEGAL = "legal-split"
    SECURITY = "security-split"
    OWNERSHIP = "ownership-split"
    PROOF = "proof-split"
    OVERGENERALIZATION = "overgeneralization"
    UNSAFE_NEAR_MATCH = "unsafe-near-match"
    KNOWN_COUNTEREXAMPLE = "known-counterexample"


REQUIRED_BOUNDARY_DIMENSIONS: Final[tuple[str, ...]] = (
    "positive_member_cids",
    "negative_example_cids",
    "boundary_example_cids",
    "unknown_case_cids",
    "risk_ceiling",
    "permitted_repositories",
    "permitted_languages",
    "permitted_frameworks",
    "permitted_effect_classes",
    "required_operation_contracts",
    "validation_structure",
    "rollback_structure",
    "postcondition_shape",
)

_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}

_DECLARED_VIOLATION: Final[dict[FamilyMembershipClass, BoundaryViolationClass]] = {
    FamilyMembershipClass.NEGATIVE: BoundaryViolationClass.NEGATIVE_EXAMPLE,
    FamilyMembershipClass.BOUNDARY: BoundaryViolationClass.BOUNDARY_EXAMPLE,
    FamilyMembershipClass.UNKNOWN: BoundaryViolationClass.UNKNOWN_CASE,
}

_DECLARED_REASON: Final[dict[FamilyMembershipClass, str]] = {
    FamilyMembershipClass.NEGATIVE: "negative-example-cannot-join-family",
    FamilyMembershipClass.BOUNDARY: "boundary-example-cannot-join-family",
    FamilyMembershipClass.UNKNOWN: "unknown-case-cannot-join-family",
}


def _risk_rank(value: RiskClass) -> int:
    return _RISK_RANK[value]


def _legal_classes(risk: RiskClass) -> frozenset[str]:
    if _risk_rank(risk) >= _risk_rank(RiskClass.PUBLIC_CONTRACT):
        return frozenset({risk.value, "public-contract"})
    return frozenset()


def _security_classes(risk: RiskClass) -> frozenset[str]:
    if risk is RiskClass.AUTHORITY_OR_SECURITY:
        return frozenset({risk.value, "authority-or-security"})
    return frozenset()


@dataclass(frozen=True)
class BoundaryCandidate:
    """Typed features offered when testing family membership or a near-match."""

    example_cid: str
    repository_id: str = ""
    language: str = ""
    framework: str = ""
    risk_class: RiskClass | None = None
    effect_classes: tuple[EffectClass, ...] = ()
    authority_classes: tuple[str, ...] = ()
    validation_classes: tuple[str, ...] = ()
    rollback_classes: tuple[str, ...] = ()
    proof_classes: tuple[str, ...] = ()
    legal_classes: tuple[str, ...] = ()
    security_classes: tuple[str, ...] = ()
    ownership_classes: tuple[str, ...] = ()
    goal_semantics: tuple[str, ...] = ()
    precondition_shape: tuple[str, ...] = ()
    affected_artifact_classes: tuple[str, ...] = ()
    required_operation_contracts: tuple[str, ...] = ()
    failure_signatures: tuple[str, ...] = ()
    proposed_membership: FamilyMembershipClass = FamilyMembershipClass.POSITIVE
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "example_cid", _identifier(self.example_cid, "example_cid")
        )
        for name in ("repository_id", "language", "framework"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, required=False),
            )
        if self.risk_class is not None:
            object.__setattr__(
                self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class")
            )
        object.__setattr__(
            self,
            "effect_classes",
            _enums(
                self.effect_classes,
                EffectClass,
                "effect_classes",
                limit=len(EffectClass),
            ),
        )
        object.__setattr__(
            self,
            "proposed_membership",
            _enum(self.proposed_membership, FamilyMembershipClass, "proposed_membership"),
        )
        for name in (
            "authority_classes",
            "validation_classes",
            "rollback_classes",
            "proof_classes",
            "legal_classes",
            "security_classes",
            "ownership_classes",
            "goal_semantics",
            "precondition_shape",
            "affected_artifact_classes",
            "required_operation_contracts",
            "failure_signatures",
            "evidence_cids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )


@dataclass(frozen=True)
class BoundaryDecision:
    """Typed admit/refuse result for one family-boundary check."""

    admitted: bool
    membership: FamilyMembershipClass
    severity: BoundarySeverity
    reason_code: str
    violation_classes: tuple[BoundaryViolationClass, ...] = ()
    conflicting_authority_classes: tuple[str, ...] = ()
    conflicting_effect_classes: tuple[EffectClass, ...] = ()
    conflicting_validation_classes: tuple[str, ...] = ()
    counterexample: TaskFamilyCounterexample | None = None
    evidence_cids: tuple[str, ...] = ()
    missing_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TaskFamilyContractError("BoundaryDecision.admitted must be a boolean")
        object.__setattr__(
            self, "membership", _enum(self.membership, FamilyMembershipClass, "membership")
        )
        object.__setattr__(
            self, "severity", _enum(self.severity, BoundarySeverity, "severity")
        )
        object.__setattr__(
            self,
            "reason_code",
            _identifier(self.reason_code, "reason_code", required=not self.admitted),
        )
        violations = _enums(
            self.violation_classes,
            BoundaryViolationClass,
            "violation_classes",
            limit=len(BoundaryViolationClass),
        )
        object.__setattr__(self, "violation_classes", violations)
        object.__setattr__(
            self,
            "conflicting_authority_classes",
            _strings(
                self.conflicting_authority_classes,
                "conflicting_authority_classes",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "conflicting_effect_classes",
            _enums(
                self.conflicting_effect_classes,
                EffectClass,
                "conflicting_effect_classes",
                limit=len(EffectClass),
            ),
        )
        object.__setattr__(
            self,
            "conflicting_validation_classes",
            _strings(
                self.conflicting_validation_classes,
                "conflicting_validation_classes",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        object.__setattr__(
            self,
            "missing_dimensions",
            _strings(self.missing_dimensions, "missing_dimensions", identifiers=True),
        )
        if self.counterexample is not None and not isinstance(
            self.counterexample, TaskFamilyCounterexample
        ):
            raise TaskFamilyContractError("counterexample must be TaskFamilyCounterexample")
        if self.admitted:
            if self.severity is not BoundarySeverity.NONE:
                raise TaskFamilyContractError("an admitted boundary decision cannot be critical")
            if self.violation_classes or self.counterexample is not None:
                raise TaskFamilyContractError("an admitted boundary decision cannot carry a refusal")
        elif self.severity is not BoundarySeverity.CRITICAL:
            raise TaskFamilyContractError("a refused boundary decision must be critical")
        elif not self.violation_classes:
            raise TaskFamilyContractError("a refused boundary decision must name a violation class")

    @property
    def is_critical_rejection(self) -> bool:
        return (not self.admitted) and self.severity is BoundarySeverity.CRITICAL


def _declared_membership(family: TaskFamily, example_cid: str) -> FamilyMembershipClass | None:
    boundary = family.boundary
    declared = {
        FamilyMembershipClass.POSITIVE: boundary.positive_member_cids,
        FamilyMembershipClass.NEGATIVE: boundary.negative_example_cids,
        FamilyMembershipClass.BOUNDARY: boundary.boundary_example_cids,
        FamilyMembershipClass.UNKNOWN: boundary.unknown_case_cids,
    }
    matches = [cls for cls, cids in declared.items() if example_cid in cids]
    if len(matches) > 1:
        raise TaskFamilyContractError("family boundary example classes must be disjoint")
    return matches[0] if matches else None


def _incomplete_family_dimensions(family: TaskFamily) -> tuple[str, ...]:
    boundary = family.boundary
    present = {
        "positive_member_cids": boundary.positive_member_cids,
        "negative_example_cids": boundary.negative_example_cids,
        "boundary_example_cids": boundary.boundary_example_cids,
        "unknown_case_cids": boundary.unknown_case_cids,
        "risk_ceiling": (boundary.risk_ceiling.value,),
        "permitted_repositories": boundary.permitted_repositories,
        "permitted_languages": boundary.permitted_languages,
        "permitted_frameworks": boundary.permitted_frameworks,
        "permitted_effect_classes": boundary.permitted_effect_classes,
        "required_operation_contracts": family.required_operation_contracts,
        "validation_structure": family.validation_structure,
        "rollback_structure": family.rollback_structure,
        "postcondition_shape": family.postcondition_shape,
    }
    missing = tuple(name for name in REQUIRED_BOUNDARY_DIMENSIONS if not present[name])
    if family.bindings.repository_id not in boundary.permitted_repositories:
        missing = missing + ("ownership",)
    return missing


def _candidate_authority(candidate: BoundaryCandidate) -> frozenset[str]:
    return frozenset(candidate.authority_classes) | frozenset(
        candidate.required_operation_contracts
    )


def _candidate_ownership(candidate: BoundaryCandidate) -> frozenset[str]:
    ownership = set(candidate.ownership_classes)
    if candidate.repository_id:
        ownership.add(candidate.repository_id)
    return frozenset(ownership)


def _surface_near_match(family: TaskFamily, candidate: BoundaryCandidate) -> bool:
    boundary = family.boundary
    shared_goals = frozenset(candidate.goal_semantics) & frozenset(family.goal_semantics)
    shared_artifacts = frozenset(candidate.affected_artifact_classes) & frozenset(
        family.affected_artifact_classes
    )
    return bool(
        (candidate.language and candidate.language in boundary.permitted_languages)
        or (candidate.repository_id and candidate.repository_id in boundary.permitted_repositories)
        or (candidate.framework and candidate.framework in boundary.permitted_frameworks)
        or shared_goals
        or shared_artifacts
    )


def _coerce_candidate(
    family: TaskFamily,
    candidate: BoundaryCandidate | TaskFamilyMembership,
) -> BoundaryCandidate:
    if isinstance(candidate, BoundaryCandidate):
        return candidate
    if not isinstance(candidate, TaskFamilyMembership):
        raise TaskFamilyContractError("boundary candidate must be typed")
    if candidate.bindings != family.bindings:
        raise TaskFamilyContractError("membership and family exact bindings differ")
    if candidate.task_family_cid != family.content_id:
        raise TaskFamilyContractError("membership does not bind the exact task-family CID")
    return BoundaryCandidate(
        example_cid=candidate.trajectory_cid,
        proposed_membership=candidate.membership,
        evidence_cids=candidate.evidence_cids,
        repository_id=family.bindings.repository_id,
    )


class TaskFamilyBoundaryValidator:
    """Fail-closed family-boundary and negative-example checker.

    The validator never promotes a family and never treats titles, embeddings,
    or a proposed membership label as evidence.  Declared negatives, boundary
    cases, and unknowns stay out of the positive set.  Any material split in
    authority, effects, validation, rollback, legal/security treatment,
    ownership, or proof obligations is a critical overgeneralization.
    """

    def validate_family(self, family: TaskFamily) -> TaskFamily:
        """Require every family to declare the complete boundary dimensions."""

        if not isinstance(family, TaskFamily):
            raise TaskFamilyContractError("family must be TaskFamily")
        missing = _incomplete_family_dimensions(family)
        if missing:
            decision = self._reject(
                family,
                example_cid=family.boundary.positive_member_cids[0]
                if family.boundary.positive_member_cids
                else family.name,
                membership=FamilyMembershipClass.UNKNOWN,
                reason_code="incomplete-boundary",
                violations=(BoundaryViolationClass.INCOMPLETE_BOUNDARY,),
                missing_dimensions=missing,
            )
            raise TaskFamilyBoundaryError(
                "family does not declare complete boundary dimensions",
                decision=decision,
            )
        return family

    def evaluate(
        self,
        family: TaskFamily,
        candidate: BoundaryCandidate | TaskFamilyMembership,
        *,
        counterexamples: Sequence[TaskFamilyCounterexample] = (),
    ) -> BoundaryDecision:
        """Classify one candidate against a family without raising on refusal."""

        if not isinstance(family, TaskFamily):
            raise TaskFamilyContractError("family must be TaskFamily")
        missing = _incomplete_family_dimensions(family)
        if missing:
            example_cid = family.name
            evidence: tuple[str, ...] = ()
            membership = FamilyMembershipClass.UNKNOWN
            if isinstance(candidate, BoundaryCandidate):
                example_cid = candidate.example_cid
                evidence = candidate.evidence_cids
                membership = candidate.proposed_membership
            elif isinstance(candidate, TaskFamilyMembership):
                example_cid = candidate.trajectory_cid
                evidence = candidate.evidence_cids
                membership = candidate.membership
            return self._reject(
                family,
                example_cid=example_cid,
                membership=membership,
                reason_code="incomplete-boundary",
                violations=(BoundaryViolationClass.INCOMPLETE_BOUNDARY,),
                evidence_cids=evidence,
                missing_dimensions=missing,
            )

        known = self._known_counterexample_decision(family, counterexamples)
        if known is not None:
            return known

        normalized = _coerce_candidate(family, candidate)
        declared = _declared_membership(family, normalized.example_cid)
        if declared is not None:
            return self._evaluate_declared(family, normalized, declared)
        if normalized.proposed_membership is FamilyMembershipClass.POSITIVE:
            return self._evaluate_undeclared_positive(family, normalized)
        return self._admit(
            normalized.proposed_membership,
            evidence_cids=normalized.evidence_cids,
        )

    def require(
        self,
        family: TaskFamily,
        candidate: BoundaryCandidate | TaskFamilyMembership,
        *,
        counterexamples: Sequence[TaskFamilyCounterexample] = (),
    ) -> BoundaryDecision:
        """Return an admitted decision or raise a critical typed rejection."""

        decision = self.evaluate(family, candidate, counterexamples=counterexamples)
        if decision.is_critical_rejection:
            raise TaskFamilyBoundaryError(
                self._rejection_message(decision),
                decision=decision,
            )
        return decision

    def evaluate_merge(self, family: TaskFamily, other: TaskFamily) -> BoundaryDecision:
        """Refuse any merge that would widen or split a family's boundary."""

        self.validate_family(family)
        if not isinstance(other, TaskFamily):
            raise TaskFamilyContractError("merged family must be TaskFamily")
        other_missing = _incomplete_family_dimensions(other)
        if other_missing:
            return self._reject(
                family,
                example_cid=other.name,
                membership=FamilyMembershipClass.NEGATIVE,
                reason_code="incomplete-boundary",
                violations=(
                    BoundaryViolationClass.INCOMPLETE_BOUNDARY,
                    BoundaryViolationClass.OVERGENERALIZATION,
                ),
                missing_dimensions=other_missing,
            )
        if family.content_id == other.content_id:
            return self._admit(FamilyMembershipClass.POSITIVE)

        violations: list[BoundaryViolationClass] = [BoundaryViolationClass.OVERGENERALIZATION]
        authority: list[str] = []
        effects: list[EffectClass] = []
        validation: list[str] = []
        if family.required_operation_contracts != other.required_operation_contracts:
            violations.append(BoundaryViolationClass.AUTHORITY)
            authority.extend(
                sorted(
                    set(family.required_operation_contracts).symmetric_difference(
                        other.required_operation_contracts
                    )
                )
            )
        if family.validation_structure != other.validation_structure:
            violations.append(BoundaryViolationClass.VALIDATION)
            validation.extend(
                sorted(
                    set(family.validation_structure).symmetric_difference(other.validation_structure)
                )
            )
        if family.rollback_structure != other.rollback_structure:
            violations.append(BoundaryViolationClass.ROLLBACK)
        if family.postcondition_shape != other.postcondition_shape:
            violations.append(BoundaryViolationClass.PROOF)
        if family.effect_classes != other.effect_classes:
            violations.append(BoundaryViolationClass.EFFECT)
            effects.extend(
                sorted(
                    set(family.effect_classes).symmetric_difference(other.effect_classes),
                    key=lambda item: item.value,
                )
            )
        left_boundary = family.boundary
        right_boundary = other.boundary
        if left_boundary.risk_ceiling is not right_boundary.risk_ceiling:
            violations.append(BoundaryViolationClass.RISK_CEILING)
            higher = (
                left_boundary.risk_ceiling
                if _risk_rank(left_boundary.risk_ceiling) > _risk_rank(right_boundary.risk_ceiling)
                else right_boundary.risk_ceiling
            )
            if higher is RiskClass.PUBLIC_CONTRACT:
                violations.append(BoundaryViolationClass.LEGAL)
            if higher is RiskClass.AUTHORITY_OR_SECURITY:
                violations.append(BoundaryViolationClass.SECURITY)
                violations.append(BoundaryViolationClass.LEGAL)
        if left_boundary.permitted_repositories != right_boundary.permitted_repositories:
            violations.append(BoundaryViolationClass.REPOSITORY)
            violations.append(BoundaryViolationClass.OWNERSHIP)
        if left_boundary.permitted_languages != right_boundary.permitted_languages:
            violations.append(BoundaryViolationClass.LANGUAGE)
        if left_boundary.permitted_frameworks != right_boundary.permitted_frameworks:
            violations.append(BoundaryViolationClass.FRAMEWORK)
        if left_boundary.permitted_effect_classes != right_boundary.permitted_effect_classes:
            violations.append(BoundaryViolationClass.EFFECT)
            effects.extend(
                sorted(
                    set(left_boundary.permitted_effect_classes).symmetric_difference(
                        right_boundary.permitted_effect_classes
                    ),
                    key=lambda item: item.value,
                )
            )
        if family.bindings.policy_revision != other.bindings.policy_revision:
            violations.append(BoundaryViolationClass.AUTHORITY)
            authority.append(other.bindings.policy_revision)
        if family.bindings.repository_id != other.bindings.repository_id:
            violations.append(BoundaryViolationClass.OWNERSHIP)
        if set(other.boundary.positive_member_cids) & set(left_boundary.negative_example_cids):
            violations.append(BoundaryViolationClass.NEGATIVE_EXAMPLE)
        if set(other.boundary.positive_member_cids) & set(left_boundary.boundary_example_cids):
            violations.append(BoundaryViolationClass.BOUNDARY_EXAMPLE)
        if set(other.boundary.positive_member_cids) & set(left_boundary.unknown_case_cids):
            violations.append(BoundaryViolationClass.UNKNOWN_CASE)

        ordered: list[BoundaryViolationClass] = []
        for item in violations:
            if item not in ordered:
                ordered.append(item)
        unique_effects: list[EffectClass] = []
        for item in effects:
            if item not in unique_effects:
                unique_effects.append(item)
        return self._reject(
            family,
            example_cid=other.content_id,
            membership=FamilyMembershipClass.NEGATIVE,
            reason_code="overgeneralization",
            violations=tuple(ordered),
            conflicting_authority_classes=tuple(dict.fromkeys(authority)),
            conflicting_effect_classes=tuple(unique_effects),
            conflicting_validation_classes=tuple(dict.fromkeys(validation)),
        )

    def require_merge(self, family: TaskFamily, other: TaskFamily) -> BoundaryDecision:
        decision = self.evaluate_merge(family, other)
        if decision.is_critical_rejection:
            raise TaskFamilyBoundaryError(
                self._rejection_message(decision),
                decision=decision,
            )
        return decision

    def _evaluate_declared(
        self,
        family: TaskFamily,
        candidate: BoundaryCandidate,
        declared: FamilyMembershipClass,
    ) -> BoundaryDecision:
        if candidate.proposed_membership is not declared:
            if declared is FamilyMembershipClass.POSITIVE:
                return self._reject(
                    family,
                    example_cid=candidate.example_cid,
                    membership=declared,
                    reason_code="membership-contradicts-declared-boundary",
                    violations=(BoundaryViolationClass.MEMBERSHIP_CONTRADICTION,),
                    evidence_cids=candidate.evidence_cids,
                )
            violation = _DECLARED_VIOLATION[declared]
            violations = [violation]
            if candidate.proposed_membership is FamilyMembershipClass.POSITIVE:
                violations.append(BoundaryViolationClass.OVERGENERALIZATION)
                if _surface_near_match(family, candidate):
                    violations.append(BoundaryViolationClass.UNSAFE_NEAR_MATCH)
            return self._reject(
                family,
                example_cid=candidate.example_cid,
                membership=declared,
                reason_code=_DECLARED_REASON.get(
                    declared, "membership-contradicts-declared-boundary"
                ),
                violations=tuple(violations),
                evidence_cids=candidate.evidence_cids,
            )
        if declared is FamilyMembershipClass.POSITIVE:
            structural = self._structural_violations(family, candidate, require_complete=False)
            if structural[0]:
                return self._reject(
                    family,
                    example_cid=candidate.example_cid,
                    membership=FamilyMembershipClass.NEGATIVE,
                    reason_code=structural[1],
                    violations=structural[0],
                    conflicting_authority_classes=structural[2],
                    conflicting_effect_classes=structural[3],
                    conflicting_validation_classes=structural[4],
                    evidence_cids=candidate.evidence_cids,
                )
        return self._admit(declared, evidence_cids=candidate.evidence_cids)

    def _evaluate_undeclared_positive(
        self,
        family: TaskFamily,
        candidate: BoundaryCandidate,
    ) -> BoundaryDecision:
        violations, reason, authority, effects, validation = self._structural_violations(
            family, candidate, require_complete=True
        )
        if not violations:
            return self._admit(
                FamilyMembershipClass.POSITIVE,
                evidence_cids=candidate.evidence_cids,
            )
        if _surface_near_match(family, candidate):
            if BoundaryViolationClass.UNSAFE_NEAR_MATCH not in violations:
                violations = violations + (BoundaryViolationClass.UNSAFE_NEAR_MATCH,)
            if reason in {
                "authority-split",
                "validation-split",
                "rollback-split",
                "proof-split",
                "legal-split",
                "security-split",
                "ownership-split",
                "effect-class-not-permitted",
                "risk-ceiling-exceeded",
            }:
                reason = "unsafe-near-match"
        if BoundaryViolationClass.OVERGENERALIZATION not in violations:
            violations = violations + (BoundaryViolationClass.OVERGENERALIZATION,)
        return self._reject(
            family,
            example_cid=candidate.example_cid,
            membership=FamilyMembershipClass.NEGATIVE,
            reason_code=reason,
            violations=violations,
            conflicting_authority_classes=authority,
            conflicting_effect_classes=effects,
            conflicting_validation_classes=validation,
            evidence_cids=candidate.evidence_cids,
        )

    def _structural_violations(
        self,
        family: TaskFamily,
        candidate: BoundaryCandidate,
        *,
        require_complete: bool,
    ) -> tuple[
        tuple[BoundaryViolationClass, ...],
        str,
        tuple[str, ...],
        tuple[EffectClass, ...],
        tuple[str, ...],
    ]:
        boundary = family.boundary
        violations: list[BoundaryViolationClass] = []
        reason = "overgeneralization"
        authority_conflict: list[str] = []
        effect_conflict: list[EffectClass] = []
        validation_conflict: list[str] = []

        def note(violation: BoundaryViolationClass, code: str) -> None:
            nonlocal reason
            if violation not in violations:
                violations.append(violation)
            if reason == "overgeneralization":
                reason = code

        if require_complete:
            missing_features: list[str] = []
            if not candidate.repository_id:
                missing_features.append("repository")
            if not candidate.language:
                missing_features.append("language")
            if boundary.permitted_frameworks and not candidate.framework:
                missing_features.append("framework")
            if candidate.risk_class is None:
                missing_features.append("risk_ceiling")
            if not candidate.effect_classes:
                missing_features.append("effects")
            if not _candidate_authority(candidate):
                missing_features.append("authority")
            if not candidate.validation_classes:
                missing_features.append("validation")
            if not candidate.rollback_classes:
                missing_features.append("rollback")
            if not candidate.proof_classes:
                missing_features.append("proof")
            if missing_features:
                note(BoundaryViolationClass.INCOMPLETE_BOUNDARY, "incomplete-boundary")
                note(BoundaryViolationClass.OVERGENERALIZATION, "overgeneralization")

        if candidate.repository_id and candidate.repository_id not in boundary.permitted_repositories:
            note(BoundaryViolationClass.REPOSITORY, "repository-not-permitted")
            note(BoundaryViolationClass.OWNERSHIP, "ownership-split")
        if candidate.language and candidate.language not in boundary.permitted_languages:
            note(BoundaryViolationClass.LANGUAGE, "language-not-permitted")
        if candidate.framework and candidate.framework not in boundary.permitted_frameworks:
            note(BoundaryViolationClass.FRAMEWORK, "framework-not-permitted")
        if candidate.risk_class is not None and _risk_rank(candidate.risk_class) > _risk_rank(
            boundary.risk_ceiling
        ):
            note(BoundaryViolationClass.RISK_CEILING, "risk-ceiling-exceeded")
            if candidate.risk_class is RiskClass.PUBLIC_CONTRACT:
                note(BoundaryViolationClass.LEGAL, "legal-split")
            if candidate.risk_class is RiskClass.AUTHORITY_OR_SECURITY:
                note(BoundaryViolationClass.SECURITY, "security-split")
                note(BoundaryViolationClass.LEGAL, "legal-split")

        extra_effects = tuple(
            item for item in candidate.effect_classes if item not in family.effect_classes
        )
        if extra_effects:
            note(BoundaryViolationClass.EFFECT, "effect-class-not-permitted")
            effect_conflict.extend(extra_effects)
        unpermitted_effects = tuple(
            item
            for item in candidate.effect_classes
            if item not in boundary.permitted_effect_classes
        )
        if unpermitted_effects:
            note(BoundaryViolationClass.EFFECT, "effect-class-not-permitted")
            for item in unpermitted_effects:
                if item not in effect_conflict:
                    effect_conflict.append(item)

        family_authority = frozenset(family.required_operation_contracts)
        candidate_authority = _candidate_authority(candidate)
        if candidate_authority and candidate_authority != family_authority:
            note(BoundaryViolationClass.AUTHORITY, "authority-split")
            authority_conflict.extend(sorted(candidate_authority.symmetric_difference(family_authority)))
        if candidate.validation_classes and frozenset(candidate.validation_classes) != frozenset(
            family.validation_structure
        ):
            note(BoundaryViolationClass.VALIDATION, "validation-split")
            validation_conflict.extend(
                sorted(
                    frozenset(candidate.validation_classes).symmetric_difference(
                        family.validation_structure
                    )
                )
            )
        if candidate.rollback_classes and frozenset(candidate.rollback_classes) != frozenset(
            family.rollback_structure
        ):
            note(BoundaryViolationClass.ROLLBACK, "rollback-split")
        if candidate.proof_classes and frozenset(candidate.proof_classes) != frozenset(
            family.postcondition_shape
        ):
            note(BoundaryViolationClass.PROOF, "proof-split")
        if candidate.goal_semantics and frozenset(candidate.goal_semantics) != frozenset(
            family.goal_semantics
        ):
            note(BoundaryViolationClass.OVERGENERALIZATION, "overgeneralization")
        if candidate.precondition_shape and frozenset(candidate.precondition_shape) != frozenset(
            family.precondition_shape
        ):
            note(BoundaryViolationClass.OVERGENERALIZATION, "overgeneralization")
        if candidate.affected_artifact_classes and frozenset(
            candidate.affected_artifact_classes
        ) != frozenset(family.affected_artifact_classes):
            note(BoundaryViolationClass.OVERGENERALIZATION, "overgeneralization")
        if candidate.failure_signatures and frozenset(candidate.failure_signatures) != frozenset(
            family.failure_signatures
        ):
            note(BoundaryViolationClass.OVERGENERALIZATION, "overgeneralization")
        if candidate.required_operation_contracts and frozenset(
            candidate.required_operation_contracts
        ) != frozenset(family.required_operation_contracts):
            note(BoundaryViolationClass.AUTHORITY, "authority-split")
            authority_conflict.extend(
                sorted(
                    frozenset(candidate.required_operation_contracts).symmetric_difference(
                        family.required_operation_contracts
                    )
                )
            )

        ownership = _candidate_ownership(candidate)
        if ownership and not ownership.issubset(boundary.permitted_repositories):
            note(BoundaryViolationClass.OWNERSHIP, "ownership-split")
            note(BoundaryViolationClass.REPOSITORY, "repository-not-permitted")
        family_legal = _legal_classes(boundary.risk_ceiling)
        family_security = _security_classes(boundary.risk_ceiling)
        extra_legal = frozenset(candidate.legal_classes) - family_legal
        extra_security = frozenset(candidate.security_classes) - family_security
        if extra_legal:
            note(BoundaryViolationClass.LEGAL, "legal-split")
        if extra_security:
            note(BoundaryViolationClass.SECURITY, "security-split")
        if candidate.risk_class is not None:
            if _legal_classes(candidate.risk_class) - family_legal:
                note(BoundaryViolationClass.LEGAL, "legal-split")
            if _security_classes(candidate.risk_class) - family_security:
                note(BoundaryViolationClass.SECURITY, "security-split")

        unique_authority = tuple(dict.fromkeys(authority_conflict))
        unique_effects = tuple(dict.fromkeys(effect_conflict))
        unique_validation = tuple(dict.fromkeys(validation_conflict))
        return tuple(violations), reason, unique_authority, unique_effects, unique_validation

    def _known_counterexample_decision(
        self,
        family: TaskFamily,
        counterexamples: Sequence[TaskFamilyCounterexample],
    ) -> BoundaryDecision | None:
        if not isinstance(counterexamples, Sequence) or isinstance(
            counterexamples, (str, bytes, bytearray, memoryview)
        ):
            raise TaskFamilyContractError("counterexamples must be a bounded sequence")
        if len(counterexamples) > 128:
            raise TaskFamilyContractError("counterexamples exceeds its item bound")
        for counterexample in counterexamples:
            if not isinstance(counterexample, TaskFamilyCounterexample):
                raise TaskFamilyContractError("counterexamples must be typed contracts")
            if counterexample.bindings != family.bindings:
                raise TaskFamilyContractError("counterexample exact bindings differ")
            if counterexample.task_family_cid != family.content_id:
                raise TaskFamilyContractError("counterexample does not bind the exact family CID")
            violations = [BoundaryViolationClass.KNOWN_COUNTEREXAMPLE]
            if (
                counterexample.conflicting_authority_classes
                or counterexample.conflicting_effect_classes
                or counterexample.conflicting_validation_classes
            ):
                violations.append(BoundaryViolationClass.OVERGENERALIZATION)
                if counterexample.conflicting_authority_classes:
                    violations.append(BoundaryViolationClass.AUTHORITY)
                if counterexample.conflicting_effect_classes:
                    violations.append(BoundaryViolationClass.EFFECT)
                if counterexample.conflicting_validation_classes:
                    violations.append(BoundaryViolationClass.VALIDATION)
            return self._reject(
                family,
                example_cid=counterexample.example_cid,
                membership=FamilyMembershipClass.NEGATIVE,
                reason_code="known-counterexample",
                violations=tuple(violations),
                conflicting_authority_classes=counterexample.conflicting_authority_classes,
                conflicting_effect_classes=counterexample.conflicting_effect_classes,
                conflicting_validation_classes=counterexample.conflicting_validation_classes,
                counterexample=counterexample,
            )
        return None

    def _admit(
        self,
        membership: FamilyMembershipClass,
        *,
        evidence_cids: tuple[str, ...] = (),
    ) -> BoundaryDecision:
        return BoundaryDecision(
            admitted=True,
            membership=membership,
            severity=BoundarySeverity.NONE,
            reason_code="",
            evidence_cids=evidence_cids,
        )

    def _reject(
        self,
        family: TaskFamily,
        *,
        example_cid: str,
        membership: FamilyMembershipClass,
        reason_code: str,
        violations: Sequence[BoundaryViolationClass],
        conflicting_authority_classes: Sequence[str] = (),
        conflicting_effect_classes: Sequence[EffectClass] = (),
        conflicting_validation_classes: Sequence[str] = (),
        evidence_cids: Sequence[str] = (),
        missing_dimensions: Sequence[str] = (),
        counterexample: TaskFamilyCounterexample | None = None,
    ) -> BoundaryDecision:
        if counterexample is None:
            primary = violations[0] if violations else BoundaryViolationClass.OVERGENERALIZATION
            counterexample = TaskFamilyCounterexample(
                bindings=family.bindings,
                task_family_cid=family.content_id,
                example_cid=example_cid,
                violation_class=primary.value,
                conflicting_authority_classes=tuple(conflicting_authority_classes),
                conflicting_effect_classes=tuple(conflicting_effect_classes),
                conflicting_validation_classes=tuple(conflicting_validation_classes),
            )
        return BoundaryDecision(
            admitted=False,
            membership=membership,
            severity=BoundarySeverity.CRITICAL,
            reason_code=reason_code,
            violation_classes=tuple(violations),
            conflicting_authority_classes=tuple(conflicting_authority_classes),
            conflicting_effect_classes=tuple(conflicting_effect_classes),
            conflicting_validation_classes=tuple(conflicting_validation_classes),
            counterexample=counterexample,
            evidence_cids=tuple(evidence_cids),
            missing_dimensions=tuple(missing_dimensions),
        )

    @staticmethod
    def _rejection_message(decision: BoundaryDecision) -> str:
        if BoundaryViolationClass.INCOMPLETE_BOUNDARY in decision.violation_classes:
            missing = ",".join(decision.missing_dimensions) or "required-dimensions"
            return f"family does not declare complete boundary dimensions: {missing}"
        if BoundaryViolationClass.NEGATIVE_EXAMPLE in decision.violation_classes:
            return "negative example cannot join the family as a positive member"
        if BoundaryViolationClass.BOUNDARY_EXAMPLE in decision.violation_classes:
            return "boundary example cannot join the family as a positive member"
        if BoundaryViolationClass.UNKNOWN_CASE in decision.violation_classes:
            return "unknown case cannot join the family as a positive member"
        if BoundaryViolationClass.UNSAFE_NEAR_MATCH in decision.violation_classes:
            return "unsafe near-match is a critical typed rejection"
        if BoundaryViolationClass.OVERGENERALIZATION in decision.violation_classes:
            return "overgeneralization is a critical typed rejection"
        return f"family boundary refused: {decision.reason_code}"


def validate_task_family_membership(
    membership: TaskFamilyMembership,
    family: TaskFamily,
) -> TaskFamilyMembership:
    """Require membership class to agree with the exact declared example set."""

    if not isinstance(membership, TaskFamilyMembership) or not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("membership and family must use typed contracts")
    if membership.bindings != family.bindings:
        raise TaskFamilyContractError("membership and family exact bindings differ")
    if membership.task_family_cid != family.content_id:
        raise TaskFamilyContractError("membership does not bind the exact task-family CID")
    boundary = family.boundary
    expected = {
        FamilyMembershipClass.POSITIVE: set(boundary.positive_member_cids),
        FamilyMembershipClass.NEGATIVE: set(boundary.negative_example_cids),
        FamilyMembershipClass.BOUNDARY: set(boundary.boundary_example_cids),
        FamilyMembershipClass.UNKNOWN: set(boundary.unknown_case_cids),
    }[membership.membership]
    if membership.trajectory_cid not in expected:
        raise TaskFamilyContractError("membership class contradicts the declared boundary")
    return membership


def validate_task_family_contract(
    family: TaskFamily,
    *,
    counterexamples: Sequence[TaskFamilyCounterexample] = (),
) -> TaskFamily:
    """Reject a family invalidated by a known authority/effect/validation split."""

    if not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("family must be TaskFamily")
    if not isinstance(counterexamples, Sequence) or isinstance(
        counterexamples, (str, bytes, bytearray, memoryview)
    ):
        raise TaskFamilyContractError("counterexamples must be a bounded sequence")
    if len(counterexamples) > 128:
        raise TaskFamilyContractError("counterexamples exceeds its item bound")
    for counterexample in counterexamples:
        if not isinstance(counterexample, TaskFamilyCounterexample):
            raise TaskFamilyContractError("counterexamples must be typed contracts")
        if counterexample.bindings != family.bindings:
            raise TaskFamilyContractError("counterexample exact bindings differ")
        if counterexample.task_family_cid != family.content_id:
            raise TaskFamilyContractError("counterexample does not bind the exact family CID")
        if (
            counterexample.conflicting_authority_classes
            or counterexample.conflicting_effect_classes
            or counterexample.conflicting_validation_classes
        ):
            raise TaskFamilyContractError(
                "known counterexample materially splits authority, effects, or validation"
            )
        raise TaskFamilyContractError("known counterexample invalidates the family boundary")
    return family


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskFamilyContractError("task-family JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise TaskFamilyContractError("task-family JSON cannot contain floating point values")


def _decode_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TaskFamilyContractError("task-family bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            return json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise TaskFamilyContractError("task-family JSON is malformed") from exc
    return value


def parse_task_family(value: Any) -> TaskFamily:
    if isinstance(value, TaskFamily):
        return validate_task_family_contract(value)
    value = _decode_json(value)
    if not isinstance(value, Mapping):
        raise TaskFamilyContractError("task family must be a mapping or JSON object")
    return validate_task_family_contract(TaskFamily.from_dict(value))


def parse_task_family_membership(
    value: Any,
    family: TaskFamily,
) -> TaskFamilyMembership:
    if not isinstance(value, TaskFamilyMembership):
        value = _decode_json(value)
        if not isinstance(value, Mapping):
            raise TaskFamilyContractError("membership must be a mapping or JSON object")
        value = TaskFamilyMembership.from_dict(value)
    return validate_task_family_membership(value, family)


CLASSIFIER_REVISION: Final[str] = "TaskFamilyClassifier@1"
DISCOVERY_REVISION: Final[str] = "TaskFamilyDiscovery@1"

REQUIRED_DISCOVERY_DIMENSIONS: Final[tuple[str, ...]] = (
    "goal_semantics",
    "precondition_shape",
    "affected_artifact_classes",
    "effect_classes",
    "required_operation_contracts",
    "validation_structure",
    "failure_signatures",
    "postcondition_shape",
    "rollback_structure",
)

CLOSED_TASK_FAMILY_NAMES: Final[tuple[str, ...]] = (
    "IMPORT_PURITY_REPAIR",
    "FALSE_SUCCESS_TO_TYPED_OUTCOME",
    "PSEUDO_CID_REPLACEMENT",
    "MUTABLE_DEPENDENCY_PIN",
    "SCHEMA_REGENERATION",
    "API_ADAPTER_MIGRATION",
    "CACHE_KEY_DEPENDENCY_REPAIR",
    "STALE_RECEIPT_INVALIDATION",
    "MISSING_ADMISSION_GATE",
    "CONFIRMATION_BINDING_REPAIR",
    "TEST_SELECTION_REPAIR",
    "PROOF_SELECTION_REPAIR",
    "DOCUMENTATION_CLAIM_NARROWING",
    "MECHANICAL_RENAME",
    "GENERATED_PROJECTION_REFRESH",
    "POST_MERGE_REQUALIFICATION",
    "MERGE_CONFLICT_CLASSIFICATION",
    "PROVIDER_UNAVAILABLE_RECOVERY",
    "CONTEXT_OMISSION_REPAIR",
    "KNOWN_FLAKY_FAILURE",
    "UNKNOWN_TASK_FAMILY",
    "UNSAFE_NEAR_MATCH_TASK",
    "CROSS_REPOSITORY_TRANSFER",
)

_IMPORT_PURITY_SHAPE: Final[dict[str, Any]] = {
    "goal_semantics": ("restore-import-purity",),
    "precondition_shape": ("import-side-effect-observed",),
    "affected_artifact_classes": ("python-source",),
    "effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    "required_operation_contracts": ("approved-patch-template@1", "test-runner@1"),
    "validation_structure": ("focused-tests", "postcondition-check"),
    "failure_signatures": ("import-side-effect",),
    "postcondition_shape": ("import-is-pure",),
    "rollback_structure": ("restore-exact-tree",),
}

_SHAPE_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "goal_semantics": ("goal_semantics", "goal-semantics"),
    "precondition_shape": ("precondition_shape", "precondition-shape"),
    "affected_artifact_classes": (
        "affected_artifact_classes",
        "affected-artifact-classes",
        "artifact_classes",
    ),
    "effect_classes": ("effect_classes", "effect-classes"),
    "required_operation_contracts": (
        "required_operation_contracts",
        "required-operation-contracts",
        "authority_classes",
        "tools",
    ),
    "validation_structure": (
        "validation_structure",
        "validation-structure",
        "validation_classes",
    ),
    "failure_signatures": ("failure_signatures", "failure-signatures"),
    "postcondition_shape": (
        "postcondition_shape",
        "postcondition-shape",
        "proof_classes",
    ),
    "rollback_structure": (
        "rollback_structure",
        "rollback-structure",
        "rollback_classes",
    ),
}


@dataclass(frozen=True)
class TaskFamilyFeatures:
    """Structural shapes used by the deterministic discovery baseline."""

    goal_semantics: tuple[str, ...] = ()
    precondition_shape: tuple[str, ...] = ()
    affected_artifact_classes: tuple[str, ...] = ()
    effect_classes: tuple[EffectClass, ...] = ()
    required_operation_contracts: tuple[str, ...] = ()
    validation_structure: tuple[str, ...] = ()
    failure_signatures: tuple[str, ...] = ()
    postcondition_shape: tuple[str, ...] = ()
    rollback_structure: tuple[str, ...] = ()
    title: str = ""
    embedding_cid: str = ""
    trajectory_cid: str = ""
    evidence_cids: tuple[str, ...] = ()
    non_structural_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self.non_structural_evidence) is not bool:
            raise TaskFamilyContractError("non_structural_evidence must be a boolean")
        object.__setattr__(
            self, "title", _text(self.title, "title", required=False)
        )
        for name in ("embedding_cid", "trajectory_cid"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "effect_classes",
            _enums(
                self.effect_classes,
                EffectClass,
                "effect_classes",
                limit=len(EffectClass),
            ),
        )
        for name in (
            "goal_semantics",
            "precondition_shape",
            "affected_artifact_classes",
            "required_operation_contracts",
            "validation_structure",
            "failure_signatures",
            "postcondition_shape",
            "rollback_structure",
            "evidence_cids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        if self.title or self.embedding_cid:
            object.__setattr__(self, "non_structural_evidence", True)

    @property
    def missing_dimensions(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name in REQUIRED_DISCOVERY_DIMENSIONS:
            if not getattr(self, name):
                missing.append(name)
        return tuple(missing)

    @property
    def sufficient(self) -> bool:
        return not self.missing_dimensions

    @property
    def fingerprint(self) -> tuple[frozenset[Any], ...]:
        return (
            frozenset(self.goal_semantics),
            frozenset(self.precondition_shape),
            frozenset(self.affected_artifact_classes),
            frozenset(item.value for item in self.effect_classes),
            frozenset(self.required_operation_contracts),
            frozenset(self.validation_structure),
            frozenset(self.failure_signatures),
            frozenset(self.postcondition_shape),
            frozenset(self.rollback_structure),
        )


@dataclass(frozen=True)
class TaskFamilyClassification:
    """Deterministic admit-or-unknown result for one discovery check."""

    membership: FamilyMembershipClass
    family_name: str = ""
    family: TaskFamily | None = None
    reason_code: str = ""
    missing_features: tuple[str, ...] = ()
    matched_dimensions: tuple[str, ...] = ()
    evidence_cids: tuple[str, ...] = ()
    classifier_revision: str = CLASSIFIER_REVISION
    trajectory_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "membership", _enum(self.membership, FamilyMembershipClass, "membership")
        )
        object.__setattr__(
            self, "family_name", _identifier(self.family_name, "family_name", required=False)
        )
        object.__setattr__(
            self,
            "reason_code",
            _identifier(self.reason_code, "reason_code", required=False),
        )
        object.__setattr__(
            self,
            "classifier_revision",
            _identifier(self.classifier_revision, "classifier_revision"),
        )
        object.__setattr__(
            self,
            "trajectory_cid",
            _identifier(self.trajectory_cid, "trajectory_cid", required=False),
        )
        object.__setattr__(
            self,
            "missing_features",
            _strings(self.missing_features, "missing_features", identifiers=True),
        )
        object.__setattr__(
            self,
            "matched_dimensions",
            _strings(self.matched_dimensions, "matched_dimensions", identifiers=True),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        if self.family is not None and not isinstance(self.family, TaskFamily):
            raise TaskFamilyContractError("classification family must be TaskFamily")
        if self.family is not None and self.family_name and self.family.name != self.family_name:
            raise TaskFamilyContractError("classification family name disagrees with the family")
        if self.membership is FamilyMembershipClass.POSITIVE:
            if not self.family_name:
                raise TaskFamilyContractError("positive classification must name a family")
            if self.missing_features:
                raise TaskFamilyContractError("positive classification cannot be missing features")

    @property
    def admitted(self) -> bool:
        return self.membership is FamilyMembershipClass.POSITIVE


def _slug(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _generated_family_shape(name: str) -> dict[str, Any]:
    slug = _slug(name)
    effects: tuple[EffectClass, ...]
    if name == "PROOF_SELECTION_REPAIR":
        effects = (EffectClass.PROOF, EffectClass.VALIDATION)
    elif name == "MERGE_CONFLICT_CLASSIFICATION":
        effects = (EffectClass.MERGE, EffectClass.VALIDATION)
    elif name == "PROVIDER_UNAVAILABLE_RECOVERY":
        effects = (EffectClass.ROLLBACK, EffectClass.ESCALATION)
    elif name == "CROSS_REPOSITORY_TRANSFER":
        effects = (EffectClass.OBSERVE, EffectClass.VALIDATION)
    else:
        effects = (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION)
    return {
        "goal_semantics": (f"{slug}-goal",),
        "precondition_shape": (f"{slug}-precondition",),
        "affected_artifact_classes": (f"{slug}-artifact",),
        "effect_classes": effects,
        "required_operation_contracts": (f"{slug}-operation@1",),
        "validation_structure": (f"{slug}-validation",),
        "failure_signatures": (f"{slug}-failure",),
        "postcondition_shape": (f"{slug}-postcondition",),
        "rollback_structure": (f"{slug}-rollback",),
    }


def task_family_features_for(name: str) -> TaskFamilyFeatures:
    """Return the closed baseline shape for one initial family name."""

    normalized = _identifier(name, "family_name")
    if normalized not in CLOSED_TASK_FAMILY_NAMES:
        raise TaskFamilyContractError("family name is outside the closed discovery vocabulary")
    payload = (
        dict(_IMPORT_PURITY_SHAPE)
        if normalized == "IMPORT_PURITY_REPAIR"
        else _generated_family_shape(normalized)
    )
    return TaskFamilyFeatures(**payload)


def task_family_features(value: TaskFamily) -> TaskFamilyFeatures:
    if not isinstance(value, TaskFamily):
        raise TaskFamilyContractError("family must be TaskFamily")
    return TaskFamilyFeatures(
        goal_semantics=value.goal_semantics,
        precondition_shape=value.precondition_shape,
        affected_artifact_classes=value.affected_artifact_classes,
        effect_classes=value.effect_classes,
        required_operation_contracts=value.required_operation_contracts,
        validation_structure=value.validation_structure,
        failure_signatures=value.failure_signatures,
        postcondition_shape=value.postcondition_shape,
        rollback_structure=value.rollback_structure,
    )


def _closed_family_boundary(name: str, features: TaskFamilyFeatures) -> TaskFamilyBoundary:
    slug = _slug(name)
    return TaskFamilyBoundary(
        positive_member_cids=(f"{slug}-positive",),
        negative_example_cids=(f"{slug}-negative",),
        boundary_example_cids=(f"{slug}-boundary",),
        unknown_case_cids=(f"{slug}-unknown",),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
        permitted_repositories=("repo",),
        permitted_languages=("python",),
        permitted_frameworks=("pytest",),
        permitted_effect_classes=features.effect_classes,
    )


def closed_task_families(bindings: ArtifactBindings) -> tuple[TaskFamily, ...]:
    """Materialize the closed initial family catalog against exact bindings."""

    if not isinstance(bindings, ArtifactBindings):
        raise TaskFamilyContractError("bindings must be ArtifactBindings")
    families: list[TaskFamily] = []
    for name in CLOSED_TASK_FAMILY_NAMES:
        features = task_family_features_for(name)
        families.append(
            TaskFamily(
                bindings=bindings,
                name=name,
                goal_semantics=features.goal_semantics,
                precondition_shape=features.precondition_shape,
                affected_artifact_classes=features.affected_artifact_classes,
                effect_classes=features.effect_classes,
                required_operation_contracts=features.required_operation_contracts,
                validation_structure=features.validation_structure,
                failure_signatures=features.failure_signatures,
                postcondition_shape=features.postcondition_shape,
                rollback_structure=features.rollback_structure,
                boundary=_closed_family_boundary(name, features),
                state=ArtifactState.CANDIDATE,
            )
        )
    _require_distinguishable(families)
    return tuple(families)


def _mapping_value(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _has_non_structural_surface(payload: Mapping[str, Any]) -> bool:
    for key in (
        "title",
        "task_title",
        "embedding",
        "embedding_cid",
        "embedding-cid",
        "embedding_vector",
        "task_family_hint",
        "task-family-hint",
        "proposed_membership",
        "proposed-membership",
    ):
        value = payload.get(key)
        if value not in (None, "", (), [], {}):
            return True
    return False


def _features_from_mapping(payload: Mapping[str, Any]) -> TaskFamilyFeatures:
    values: dict[str, Any] = {}
    for field_name, aliases in _SHAPE_FIELD_ALIASES.items():
        raw = _mapping_value(payload, aliases)
        if raw is not None:
            values[field_name] = raw
    title = payload.get("title", payload.get("task_title", ""))
    if title not in (None, ""):
        values["title"] = title
    embedding_cid = payload.get("embedding_cid", payload.get("embedding-cid", ""))
    if isinstance(embedding_cid, str) and embedding_cid:
        values["embedding_cid"] = embedding_cid
    trajectory_cid = payload.get("trajectory_cid", payload.get("example_cid", ""))
    if isinstance(trajectory_cid, str) and trajectory_cid:
        values["trajectory_cid"] = trajectory_cid
    evidence = payload.get("evidence_cids")
    if evidence is not None:
        values["evidence_cids"] = evidence
    values["non_structural_evidence"] = _has_non_structural_surface(payload)
    return TaskFamilyFeatures(**values)


def _features_from_trajectory(trajectory: ExecutionTrajectory) -> TaskFamilyFeatures:
    contracts = tuple(step.operation_contract for step in trajectory.steps)
    effects: list[EffectClass] = []
    for step in trajectory.steps:
        for raw in step.effect_ids:
            try:
                item = _enum(raw, EffectClass, "effect_classes")
            except ProcedureContractError:
                continue
            if item not in effects:
                effects.append(item)
    return TaskFamilyFeatures(
        required_operation_contracts=contracts,
        effect_classes=tuple(effects),
        trajectory_cid=trajectory.content_id,
        evidence_cids=(trajectory.source_episode_cid,),
        non_structural_evidence=bool(trajectory.task_family_hint),
    )


def extract_task_family_features(value: Any) -> TaskFamilyFeatures:
    """Project a typed source onto discovery shapes without using labels."""

    if isinstance(value, TaskFamilyFeatures):
        return value
    if isinstance(value, TaskFamily):
        return task_family_features(value)
    if isinstance(value, BoundaryCandidate):
        return TaskFamilyFeatures(
            goal_semantics=value.goal_semantics,
            precondition_shape=value.precondition_shape,
            affected_artifact_classes=value.affected_artifact_classes,
            effect_classes=value.effect_classes,
            required_operation_contracts=value.required_operation_contracts
            or value.authority_classes,
            validation_structure=value.validation_classes,
            failure_signatures=value.failure_signatures,
            postcondition_shape=value.proof_classes,
            rollback_structure=value.rollback_classes,
            trajectory_cid=value.example_cid,
            evidence_cids=value.evidence_cids,
            non_structural_evidence=False,
        )
    if isinstance(value, ExecutionTrajectory):
        return _features_from_trajectory(value)
    trajectory = getattr(value, "trajectory", None)
    if isinstance(trajectory, ExecutionTrajectory):
        return _features_from_trajectory(trajectory)
    if isinstance(value, Mapping):
        return _features_from_mapping(value)
    raise TaskFamilyContractError("discovery features must be a typed family source")


def _present_dimensions(features: TaskFamilyFeatures) -> tuple[str, ...]:
    return tuple(
        name for name in REQUIRED_DISCOVERY_DIMENSIONS if getattr(features, name)
    )


def _require_distinguishable(families: Sequence[TaskFamily]) -> None:
    seen: dict[tuple[frozenset[Any], ...], str] = {}
    for family in families:
        features = task_family_features(family)
        if not features.sufficient:
            continue
        fingerprint = features.fingerprint
        previous = seen.get(fingerprint)
        if previous is not None and previous != family.name:
            raise TaskFamilyContractError("closed families must be distinguishable")
        seen[fingerprint] = family.name


def _validate_discovery_catalog(families: Sequence[TaskFamily]) -> tuple[TaskFamily, ...]:
    if not isinstance(families, Sequence) or isinstance(
        families, (str, bytes, bytearray, memoryview)
    ):
        raise TaskFamilyContractError("families must be a bounded sequence")
    if len(families) > MAX_ITEMS:
        raise TaskFamilyContractError("families exceeds its item bound")
    catalog: list[TaskFamily] = []
    names: set[str] = set()
    for family in families:
        if not isinstance(family, TaskFamily):
            raise TaskFamilyContractError("families must be typed TaskFamily contracts")
        if family.name in names:
            raise TaskFamilyContractError("discovery catalog family names must be unique")
        names.add(family.name)
        catalog.append(family)
    _require_distinguishable(catalog)
    return tuple(catalog)


def _unknown_classification(
    features: TaskFamilyFeatures,
    *,
    reason_code: str,
    classifier_revision: str,
) -> TaskFamilyClassification:
    return TaskFamilyClassification(
        membership=FamilyMembershipClass.UNKNOWN,
        reason_code=reason_code,
        missing_features=features.missing_dimensions,
        matched_dimensions=_present_dimensions(features),
        evidence_cids=features.evidence_cids,
        classifier_revision=classifier_revision,
        trajectory_cid=features.trajectory_cid,
    )


@dataclass(frozen=True)
class TaskFamilyClassifier:
    """Deterministic structural classifier.  Never promotes a family."""

    revision: str = CLASSIFIER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _identifier(self.revision, "classifier_revision"))

    def classify(
        self,
        source: Any,
        families: Sequence[TaskFamily] | None = None,
    ) -> TaskFamilyClassification:
        features = extract_task_family_features(source)
        named: list[tuple[str, TaskFamilyFeatures, TaskFamily | None]] = []
        if families is None:
            for name in CLOSED_TASK_FAMILY_NAMES:
                named.append((name, task_family_features_for(name), None))
        else:
            for family in _validate_discovery_catalog(families):
                named.append((family.name, task_family_features(family), family))

        if not features.sufficient:
            reason = "insufficient-evidence"
            if features.non_structural_evidence and not _present_dimensions(features):
                reason = "title-or-embedding-only"
            return _unknown_classification(
                features, reason_code=reason, classifier_revision=self.revision
            )

        matches = [
            (name, family)
            for name, shape, family in named
            if shape.sufficient and shape.fingerprint == features.fingerprint
        ]
        if len(matches) == 1:
            name, family = matches[0]
            return TaskFamilyClassification(
                membership=FamilyMembershipClass.POSITIVE,
                family_name=name,
                family=family,
                reason_code="exact-structural-match",
                matched_dimensions=REQUIRED_DISCOVERY_DIMENSIONS,
                evidence_cids=features.evidence_cids,
                classifier_revision=self.revision,
                trajectory_cid=features.trajectory_cid,
            )
        reason = "ambiguous-family-match" if len(matches) > 1 else "no-family-match"
        return _unknown_classification(
            features, reason_code=reason, classifier_revision=self.revision
        )

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        raise TaskFamilyContractError("task-family classifier cannot promote or mutate families")


@dataclass(frozen=True)
class TaskFamilyDiscovery:
    """Closed-family discovery over normalized structural features."""

    families: tuple[TaskFamily, ...] = ()
    classifier: TaskFamilyClassifier = field(default_factory=TaskFamilyClassifier)
    revision: str = DISCOVERY_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _identifier(self.revision, "discovery_revision"))
        if not isinstance(self.classifier, TaskFamilyClassifier):
            raise TaskFamilyContractError("classifier must be TaskFamilyClassifier")
        object.__setattr__(self, "families", _validate_discovery_catalog(self.families))

    @classmethod
    def from_bindings(
        cls,
        bindings: ArtifactBindings,
        *,
        classifier: TaskFamilyClassifier | None = None,
    ) -> TaskFamilyDiscovery:
        return cls(
            families=closed_task_families(bindings),
            classifier=classifier or TaskFamilyClassifier(),
        )

    @property
    def catalog(self) -> tuple[TaskFamily, ...]:
        return self.families

    def classify(self, source: Any) -> TaskFamilyClassification:
        families = self.families if self.families else None
        return self.classifier.classify(source, families)

    def discover(self, source: Any) -> TaskFamilyClassification:
        return self.classify(source)

    def membership(
        self,
        source: Any,
        *,
        bindings: ArtifactBindings | None = None,
        trajectory_cid: str = "",
        evidence_cids: Sequence[str] = (),
    ) -> TaskFamilyMembership:
        decision = self.classify(source)
        family = decision.family
        if family is None:
            raise TaskFamilyContractError("unknown classification cannot bind a family membership")
        cid = trajectory_cid or decision.trajectory_cid
        if not cid:
            raise TaskFamilyContractError("membership requires a trajectory CID")
        evidence = tuple(evidence_cids) or decision.evidence_cids or (self.classifier.revision,)
        record_bindings = bindings or family.bindings
        return TaskFamilyMembership(
            bindings=record_bindings,
            task_family_cid=family.content_id,
            trajectory_cid=cid,
            membership=decision.membership,
            evidence_cids=evidence,
            classifier_revision=decision.classifier_revision,
        )

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        raise TaskFamilyContractError("task-family discovery cannot promote or mutate families")


def classify_task_family(
    source: Any,
    families: Sequence[TaskFamily] | None = None,
    *,
    classifier: TaskFamilyClassifier | None = None,
) -> TaskFamilyClassification:
    return (classifier or TaskFamilyClassifier()).classify(source, families)


def discover_task_family(
    source: Any,
    families: Sequence[TaskFamily] | None = None,
    *,
    bindings: ArtifactBindings | None = None,
) -> TaskFamilyClassification:
    if families is None and bindings is not None:
        discovery = TaskFamilyDiscovery.from_bindings(bindings)
    else:
        discovery = TaskFamilyDiscovery(families=tuple(families or ()))
    return discovery.discover(source)


__all__ = [
    "CLASSIFIER_REVISION",
    "CLOSED_TASK_FAMILY_NAMES",
    "DISCOVERY_REVISION",
    "REQUIRED_BOUNDARY_DIMENSIONS",
    "REQUIRED_DISCOVERY_DIMENSIONS",
    "BoundaryCandidate",
    "BoundaryDecision",
    "BoundarySeverity",
    "BoundaryViolationClass",
    "FamilyMembershipClass",
    "TaskFamily",
    "TaskFamilyBoundary",
    "TaskFamilyBoundaryError",
    "TaskFamilyBoundaryValidator",
    "TaskFamilyClassification",
    "TaskFamilyClassifier",
    "TaskFamilyContractError",
    "TaskFamilyCounterexample",
    "TaskFamilyDiscovery",
    "TaskFamilyFeatures",
    "TaskFamilyMembership",
    "classify_task_family",
    "closed_task_families",
    "discover_task_family",
    "extract_task_family_features",
    "parse_task_family",
    "parse_task_family_membership",
    "task_family_features",
    "task_family_features_for",
    "validate_task_family_contract",
    "validate_task_family_membership",
]
