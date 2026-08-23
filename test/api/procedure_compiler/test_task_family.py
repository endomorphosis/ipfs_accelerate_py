from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    EffectClass,
    FamilyMembershipClass,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyCounterexample,
    TaskFamilyMembership,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.task_family import (
    REQUIRED_BOUNDARY_DIMENSIONS,
    BoundaryCandidate,
    BoundarySeverity,
    BoundaryViolationClass,
    TaskFamilyBoundaryError,
    TaskFamilyBoundaryValidator,
    TaskFamilyContractError,
    parse_task_family,
    validate_task_family_contract,
)


def _bindings(*, repository: str = "repo", policy: str = "policy-v1") -> ArtifactBindings:
    return ArtifactBindings(
        repository,
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-011",
        "contract-v1",
        policy,
        "env-v1",
    )


def _boundary(**changes: object) -> TaskFamilyBoundary:
    values: dict[str, object] = {
        "positive_member_cids": ("positive-a",),
        "negative_example_cids": ("negative-a",),
        "boundary_example_cids": ("boundary-a",),
        "unknown_case_cids": ("unknown-a",),
        "risk_ceiling": RiskClass.REVERSIBLE_LOCAL,
        "permitted_repositories": ("repo",),
        "permitted_languages": ("python",),
        "permitted_frameworks": ("pytest",),
        "permitted_effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    }
    values.update(changes)
    return TaskFamilyBoundary(**values)


def family(**changes: object) -> TaskFamily:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "name": "IMPORT_PURITY_REPAIR",
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
        "required_operation_contracts": ("approved-patch-template@1", "test-runner@1"),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "failure_signatures": ("import-side-effect",),
        "postcondition_shape": ("import-is-pure",),
        "rollback_structure": ("restore-exact-tree",),
        "boundary": _boundary(),
    }
    values.update(changes)
    return TaskFamily(**values)


def matching_candidate(value: TaskFamily, **changes: object) -> BoundaryCandidate:
    values: dict[str, object] = {
        "example_cid": "candidate-positive",
        "repository_id": value.bindings.repository_id,
        "language": value.boundary.permitted_languages[0]
        if value.boundary.permitted_languages
        else "",
        "framework": value.boundary.permitted_frameworks[0]
        if value.boundary.permitted_frameworks
        else "",
        "risk_class": value.boundary.risk_ceiling,
        "effect_classes": value.effect_classes,
        "authority_classes": value.required_operation_contracts,
        "validation_classes": value.validation_structure,
        "rollback_classes": value.rollback_structure,
        "proof_classes": value.postcondition_shape,
        "ownership_classes": value.boundary.permitted_repositories,
        "goal_semantics": value.goal_semantics,
        "precondition_shape": value.precondition_shape,
        "affected_artifact_classes": value.affected_artifact_classes,
        "required_operation_contracts": value.required_operation_contracts,
        "failure_signatures": value.failure_signatures,
        "proposed_membership": FamilyMembershipClass.POSITIVE,
        "evidence_cids": ("classifier-receipt",),
    }
    values.update(changes)
    return BoundaryCandidate(**values)


def test_family_boundary_requires_complete_dimensions() -> None:
    validator = TaskFamilyBoundaryValidator()
    incomplete = family(boundary=_boundary(unknown_case_cids=(), permitted_frameworks=()))

    decision = validator.evaluate(incomplete, matching_candidate(incomplete))

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.INCOMPLETE_BOUNDARY in decision.violation_classes
    assert "unknown_case_cids" in decision.missing_dimensions
    assert "permitted_frameworks" in decision.missing_dimensions
    assert set(REQUIRED_BOUNDARY_DIMENSIONS).issuperset(decision.missing_dimensions)
    with pytest.raises(TaskFamilyBoundaryError, match="complete boundary"):
        validator.validate_family(incomplete)


def test_family_boundary_requires_declared_risk_repository_language_and_effects() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    admitted = validator.validate_family(value)

    assert admitted is value
    assert admitted.boundary.risk_ceiling is RiskClass.REVERSIBLE_LOCAL
    assert admitted.boundary.permitted_repositories == ("repo",)
    assert admitted.boundary.permitted_languages == ("python",)
    assert EffectClass.REPOSITORY_WRITE in admitted.boundary.permitted_effect_classes
    assert validator.evaluate(value, matching_candidate(value)).admitted is True


def test_family_boundary_rejects_ownership_outside_permitted_repositories() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family(boundary=_boundary(permitted_repositories=("other-repo",)))

    with pytest.raises(TaskFamilyBoundaryError, match="complete boundary") as raised:
        validator.validate_family(value)
    assert raised.value.decision is not None
    assert "ownership" in raised.value.decision.missing_dimensions


def test_negative_example_cannot_join_family_as_positive() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(
        value,
        example_cid="negative-a",
        proposed_membership=FamilyMembershipClass.POSITIVE,
    )

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.membership is FamilyMembershipClass.NEGATIVE
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.NEGATIVE_EXAMPLE in decision.violation_classes
    assert decision.counterexample is not None
    assert decision.counterexample.example_cid == "negative-a"
    with pytest.raises(TaskFamilyBoundaryError, match="negative example"):
        validator.require(value, candidate)


def test_boundary_example_cannot_join_family_as_positive() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(value, example_cid="boundary-a")

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.membership is FamilyMembershipClass.BOUNDARY
    assert BoundaryViolationClass.BOUNDARY_EXAMPLE in decision.violation_classes
    with pytest.raises(TaskFamilyBoundaryError, match="boundary example"):
        validator.require(value, candidate)


def test_unknown_case_cannot_join_family_as_positive_boundary_member() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(value, example_cid="unknown-a")

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.membership is FamilyMembershipClass.UNKNOWN
    assert BoundaryViolationClass.UNKNOWN_CASE in decision.violation_classes
    with pytest.raises(TaskFamilyBoundaryError, match="unknown case"):
        validator.require(value, candidate)


def test_declared_negative_example_stays_negative() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(
        value,
        example_cid="negative-a",
        proposed_membership=FamilyMembershipClass.NEGATIVE,
    )

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is True
    assert decision.membership is FamilyMembershipClass.NEGATIVE
    assert decision.severity is BoundarySeverity.NONE
    assert decision.counterexample is None


def test_negative_membership_record_cannot_be_relabeled_positive() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    record = TaskFamilyMembership(
        bindings=value.bindings,
        task_family_cid=value.content_id,
        trajectory_cid="negative-a",
        membership=FamilyMembershipClass.POSITIVE,
        evidence_cids=("classifier-receipt",),
        classifier_revision="baseline-v1",
    )

    decision = validator.evaluate(value, record)

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.NEGATIVE_EXAMPLE in decision.violation_classes


def test_risk_ceiling_above_family_boundary_is_critically_rejected() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(value, risk_class=RiskClass.AUTHORITY_OR_SECURITY)

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.RISK_CEILING in decision.violation_classes
    assert BoundaryViolationClass.SECURITY in decision.violation_classes
    assert decision.reason_code in {"risk-ceiling-exceeded", "unsafe-near-match"}


def test_unsafe_repository_language_and_effect_mismatch_is_rejected() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    cases = (
        matching_candidate(value, repository_id="other-repo", ownership_classes=("other-repo",)),
        matching_candidate(value, language="rust"),
        matching_candidate(
            value,
            effect_classes=(
                EffectClass.REPOSITORY_WRITE,
                EffectClass.VALIDATION,
                EffectClass.MERGE,
            ),
        ),
    )

    for candidate in cases:
        decision = validator.evaluate(value, candidate)
        assert decision.admitted is False
        assert decision.severity is BoundarySeverity.CRITICAL
        assert decision.counterexample is not None


def test_unsafe_near_match_with_authority_split_is_critically_rejected() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = matching_candidate(
        value,
        example_cid="unsafe-near-match",
        authority_classes=("approved-patch-template@1", "security-review@1"),
        required_operation_contracts=("approved-patch-template@1", "security-review@1"),
    )

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.AUTHORITY in decision.violation_classes
    assert BoundaryViolationClass.UNSAFE_NEAR_MATCH in decision.violation_classes
    assert BoundaryViolationClass.OVERGENERALIZATION in decision.violation_classes
    assert decision.reason_code == "unsafe-near-match"
    assert "security-review@1" in decision.conflicting_authority_classes
    assert decision.counterexample is not None
    assert decision.counterexample.violation_class == "authority-split"
    with pytest.raises(TaskFamilyBoundaryError, match="unsafe near-match"):
        validator.require(value, candidate)


def test_validation_rollback_and_proof_splits_are_critical_boundary_refusals() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    splits = (
        matching_candidate(value, validation_classes=("full-suite",)),
        matching_candidate(value, rollback_classes=("drop-tree",)),
        matching_candidate(value, proof_classes=("no-proof-required",)),
    )
    expected = (
        BoundaryViolationClass.VALIDATION,
        BoundaryViolationClass.ROLLBACK,
        BoundaryViolationClass.PROOF,
    )

    for candidate, violation in zip(splits, expected, strict=True):
        decision = validator.evaluate(value, candidate)
        assert decision.admitted is False
        assert decision.severity is BoundarySeverity.CRITICAL
        assert violation in decision.violation_classes
        assert BoundaryViolationClass.OVERGENERALIZATION in decision.violation_classes


def test_legal_security_and_ownership_splits_are_critical_boundary_refusals() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    legal = matching_candidate(
        value,
        risk_class=RiskClass.PUBLIC_CONTRACT,
        legal_classes=("public-contract",),
    )
    security = matching_candidate(
        value,
        risk_class=RiskClass.AUTHORITY_OR_SECURITY,
        security_classes=("authority-or-security",),
    )
    ownership = matching_candidate(
        value,
        repository_id="foreign-repo",
        ownership_classes=("foreign-repo",),
    )

    legal_decision = validator.evaluate(value, legal)
    security_decision = validator.evaluate(value, security)
    ownership_decision = validator.evaluate(value, ownership)

    assert legal_decision.admitted is False
    assert BoundaryViolationClass.LEGAL in legal_decision.violation_classes
    assert security_decision.admitted is False
    assert BoundaryViolationClass.SECURITY in security_decision.violation_classes
    assert ownership_decision.admitted is False
    assert BoundaryViolationClass.OWNERSHIP in ownership_decision.violation_classes
    assert ownership_decision.severity is BoundarySeverity.CRITICAL


def test_overgeneralization_across_family_boundary_is_critical_typed_rejection() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    widened = replace(
        value,
        validation_structure=("focused-tests", "postcondition-check", "optional-smoke"),
        required_operation_contracts=(
            "approved-patch-template@1",
            "test-runner@1",
            "policy-rewrite@1",
        ),
        boundary=replace(
            value.boundary,
            risk_ceiling=RiskClass.AUTHORITY_OR_SECURITY,
            permitted_repositories=("repo", "foreign-repo"),
            permitted_effect_classes=(
                EffectClass.REPOSITORY_WRITE,
                EffectClass.VALIDATION,
                EffectClass.MERGE,
            ),
        ),
        effect_classes=(
            EffectClass.REPOSITORY_WRITE,
            EffectClass.VALIDATION,
            EffectClass.MERGE,
        ),
    )

    decision = validator.evaluate_merge(value, widened)

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.OVERGENERALIZATION in decision.violation_classes
    assert BoundaryViolationClass.AUTHORITY in decision.violation_classes
    assert BoundaryViolationClass.VALIDATION in decision.violation_classes
    assert BoundaryViolationClass.EFFECT in decision.violation_classes
    assert BoundaryViolationClass.RISK_CEILING in decision.violation_classes
    assert BoundaryViolationClass.OWNERSHIP in decision.violation_classes
    assert decision.counterexample is not None
    with pytest.raises(TaskFamilyBoundaryError, match="overgeneralization"):
        validator.require_merge(value, widened)


def test_identical_family_boundary_merge_is_admitted() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    decision = validator.evaluate_merge(value, parse_task_family(value.to_dict()))
    assert decision.admitted is True
    assert decision.severity is BoundarySeverity.NONE


def test_declared_positive_member_stays_inside_family_boundary() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    decision = validator.require(value, matching_candidate(value, example_cid="positive-a"))
    assert decision.admitted is True
    assert decision.membership is FamilyMembershipClass.POSITIVE


def test_matching_undeclared_member_stays_inside_family_boundary() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    decision = validator.require(value, matching_candidate(value, example_cid="positive-new"))
    assert decision.admitted is True
    assert decision.membership is FamilyMembershipClass.POSITIVE
    assert decision.counterexample is None


def test_undeclared_positive_without_complete_features_is_unsafe_overgeneralization() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    candidate = BoundaryCandidate(
        example_cid="thin-positive",
        language="python",
        repository_id="repo",
        proposed_membership=FamilyMembershipClass.POSITIVE,
        evidence_cids=("title-embedding",),
    )

    decision = validator.evaluate(value, candidate)

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.INCOMPLETE_BOUNDARY in decision.violation_classes
    assert BoundaryViolationClass.OVERGENERALIZATION in decision.violation_classes
    assert BoundaryViolationClass.UNSAFE_NEAR_MATCH in decision.violation_classes


def test_known_unsafe_counterexample_invalidates_family_boundary() -> None:
    validator = TaskFamilyBoundaryValidator()
    value = family()
    counterexample = TaskFamilyCounterexample(
        bindings=value.bindings,
        task_family_cid=value.content_id,
        example_cid="unsafe-near-match",
        violation_class="authority-split",
        conflicting_authority_classes=("security-review",),
        conflicting_effect_classes=(EffectClass.REPOSITORY_WRITE,),
        conflicting_validation_classes=("security-proof",),
    )

    decision = validator.evaluate(
        value,
        matching_candidate(value),
        counterexamples=(counterexample,),
    )

    assert decision.admitted is False
    assert decision.severity is BoundarySeverity.CRITICAL
    assert BoundaryViolationClass.KNOWN_COUNTEREXAMPLE in decision.violation_classes
    assert decision.counterexample == counterexample
    with pytest.raises(TaskFamilyContractError, match="materially splits"):
        validate_task_family_contract(value, counterexamples=(counterexample,))
