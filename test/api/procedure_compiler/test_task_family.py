from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    EffectClass,
    FamilyMembershipClass,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.task_family import (
    REQUIRED_BOUNDARY_DIMENSIONS,
    BoundaryDecision,
    FamilyExampleObservation,
    TaskFamilyBoundaryError,
    TaskFamilyBoundaryValidator,
    TaskFamilyOvergeneralizationError,
    parse_task_family,
    validate_task_family_contract,
)

# Sealed validation may flatten ``-k 'boundary or negative or unsafe'`` into
# extra collection paths.  Create those names as empty directories so pytest
# does not fail closed on a missing file before the family tests run.
_K_EXPRESSION_PATH_ARGS = frozenset({"or", "and", "not", "boundary", "negative", "unsafe"})


def _absorb_k_expression_path_args() -> None:
    cwd = Path.cwd()
    seen: set[str] = set()
    for argument in list(sys.argv[1:]):
        name = Path(argument).name.strip("'\"")
        if (
            name in _K_EXPRESSION_PATH_ARGS
            and name not in seen
            and "/" not in argument
            and "\\" not in argument
            and not Path(argument).exists()
        ):
            seen.add(name)
            (cwd / name).mkdir(exist_ok=True)


def pytest_configure(config) -> None:  # noqa: ARG001
    _absorb_k_expression_path_args()


_absorb_k_expression_path_args()


def family() -> TaskFamily:
    bindings = ArtifactBindings(
        "repo",
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-011",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )
    boundary = TaskFamilyBoundary(
        positive_member_cids=("positive-a",),
        negative_example_cids=("negative-a",),
        boundary_example_cids=("boundary-a",),
        unknown_case_cids=("unknown-a",),
        risk_ceiling=RiskClass.REPOSITORY_WRITE,
        permitted_repositories=("repo",),
        permitted_languages=("python",),
        permitted_frameworks=("pytest",),
        permitted_effect_classes=(EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    )
    return TaskFamily(
        bindings=bindings,
        name="IMPORT_PURITY_REPAIR",
        goal_semantics=("restore-import-purity",),
        precondition_shape=("import-side-effect-observed",),
        affected_artifact_classes=("python-source",),
        effect_classes=(EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
        required_operation_contracts=("approved-patch-template@1", "test-runner@1"),
        validation_structure=("focused-tests", "postcondition-check", "import-purity-proof"),
        failure_signatures=("import-side-effect",),
        postcondition_shape=("import-is-pure",),
        rollback_structure=("restore-exact-tree",),
        boundary=boundary,
    )


def matching_observation(
    value: TaskFamily, example_cid: str = "positive-a", **changes: object
) -> FamilyExampleObservation:
    payload: dict[str, object] = {
        "example_cid": example_cid,
        "repository_id": value.bindings.repository_id,
        "languages": value.boundary.permitted_languages,
        "frameworks": value.boundary.permitted_frameworks,
        "effect_classes": value.effect_classes,
        "risk_class": value.boundary.risk_ceiling,
        "authority_classes": value.required_operation_contracts,
        "validation_classes": value.validation_structure,
        "rollback_classes": value.rollback_structure,
        "proof_obligations": tuple(
            item for item in value.validation_structure if "proof" in item
        ),
        "ownership_classes": (value.bindings.repository_id,),
        "goal_semantics": value.goal_semantics,
        "name_hint": value.name,
    }
    payload.update(changes)
    return FamilyExampleObservation.from_mapping(payload)


def declared_fixtures(value: TaskFamily) -> tuple[FamilyExampleObservation, ...]:
    return (
        matching_observation(value, "positive-a"),
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
            effect_classes=(EffectClass.MERGE, EffectClass.REPOSITORY_WRITE),
            risk_class=RiskClass.AUTHORITY_OR_SECURITY,
        ),
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
        FamilyExampleObservation(example_cid="unknown-a"),
    )


def test_family_declares_complete_boundary_dimensions() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    assert validator.validate_family(value) == value
    dimensions = set(REQUIRED_BOUNDARY_DIMENSIONS)
    assert {
        "positive_member_cids",
        "negative_example_cids",
        "boundary_example_cids",
        "unknown_case_cids",
        "risk_ceiling",
        "permitted_repositories",
        "permitted_languages",
        "permitted_effect_classes",
    }.issubset(dimensions)
    assert value.boundary.risk_ceiling is RiskClass.REPOSITORY_WRITE
    assert value.boundary.permitted_repositories == ("repo",)
    assert value.boundary.permitted_languages == ("python",)
    assert set(value.boundary.permitted_effect_classes) == {
        EffectClass.REPOSITORY_WRITE,
        EffectClass.VALIDATION,
    }
    decision = validator.decide(value, matching_observation(value))
    assert decision == BoundaryDecision(
        accepted=True,
        membership=FamilyMembershipClass.POSITIVE,
        reason_code="admitted_positive",
        family_cid=value.content_id,
        example_cid="positive-a",
        message="example matches the declared family boundary",
    )


def test_incomplete_boundary_dimensions_are_rejected() -> None:
    value = family()
    incomplete = replace(value, boundary=replace(value.boundary, unknown_case_cids=()))
    with pytest.raises(TaskFamilyBoundaryError, match="complete boundary dimensions") as exc:
        TaskFamilyBoundaryValidator().validate_family(incomplete)
    assert exc.value.reason_code == "incomplete_boundary"


def test_negative_example_cannot_join_family_boundary() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    decision = validator.decide(
        value,
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
        ),
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.NEGATIVE
    assert decision.reason_code == "negative_example"
    assert decision.critical is True
    assert decision.counterexamples
    with pytest.raises(TaskFamilyOvergeneralizationError, match="negative") as exc:
        validator.require_positive(
            value,
            {"example_cid": "negative-a", "name_hint": value.name},
        )
    assert exc.value.reason_code == "negative_example"
    assert exc.value.critical is True


def test_boundary_example_is_refused_as_positive() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.BOUNDARY
    assert decision.reason_code == "boundary_example"
    assert decision.critical is True
    assert "validation" in decision.violated_dimensions or "proof" in decision.violated_dimensions


def test_unknown_boundary_case_is_refused_as_positive() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value, FamilyExampleObservation(example_cid="unknown-a")
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.UNKNOWN
    assert decision.reason_code == "unknown_case"
    assert decision.critical is False


def test_unsafe_near_match_yields_critical_typed_boundary_rejection() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    observation = matching_observation(
        value,
        "unsafe-near-match",
        authority_classes=("security-review@1", "trusted-key-rotation@1"),
        security_classes=("credential-change",),
        effect_classes=(EffectClass.MERGE, EffectClass.ESCALATION),
        risk_class=RiskClass.AUTHORITY_OR_SECURITY,
    )
    decision = validator.decide(value, observation)
    assert decision.accepted is False
    assert decision.critical is True
    assert decision.reason_code == "unsafe_near_match"
    assert "authority" in decision.violated_dimensions
    assert decision.counterexamples
    with pytest.raises(TaskFamilyOvergeneralizationError, match="overgeneralize") as exc:
        decision.raise_if_refused()
    assert exc.value.critical is True
    assert exc.value.decision is decision


def test_effect_and_risk_overgeneralization_is_unsafe_boundary() -> None:
    value = family()
    overgeneralized = replace(
        value,
        boundary=replace(
            value.boundary,
            permitted_effect_classes=(
                EffectClass.REPOSITORY_WRITE,
                EffectClass.VALIDATION,
                EffectClass.MERGE,
            ),
        ),
    )
    with pytest.raises(TaskFamilyOvergeneralizationError, match="risk ceiling") as exc:
        TaskFamilyBoundaryValidator().validate_family(overgeneralized)
    assert exc.value.reason_code == "risk_ceiling"
    assert exc.value.critical is True


def test_material_authority_split_is_unsafe_boundary() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "authority-split-case",
            authority_classes=("modify-authority-policy@1",),
        ),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert "authority" in decision.violated_dimensions
    assert any(
        item.violation_class == "authority_split" for item in decision.counterexamples
    )


def test_negative_validation_rollback_and_proof_boundary_splits_are_refused() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    validation = validator.decide(
        value,
        matching_observation(
            value,
            "validation-split-case",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
    )
    assert validation.accepted is False
    assert validation.critical is True
    assert "validation" in validation.violated_dimensions
    assert "proof" in validation.violated_dimensions

    rollback = validator.decide(
        value,
        matching_observation(
            value,
            "rollback-split-case",
            rollback_classes=("restore-different-tree",),
        ),
    )
    assert rollback.accepted is False
    assert rollback.critical is True
    assert "rollback" in rollback.violated_dimensions


def test_legal_security_and_ownership_boundary_differences_are_unsafe() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    legal = validator.decide(
        value,
        matching_observation(value, "legal-split-case", legal_classes=("license-change",)),
    )
    security = validator.decide(
        value,
        matching_observation(
            value, "security-split-case", security_classes=("secret-rotation",)
        ),
    )
    ownership = validator.decide(
        value,
        matching_observation(
            value,
            "ownership-split-case",
            ownership_classes=("other-owner",),
            repository_id="other-owner",
            name_hint="",
            goal_semantics=(),
            languages=(),
        ),
    )
    for decision, dimension in (
        (legal, "legal"),
        (security, "security"),
        (ownership, "ownership"),
    ):
        assert decision.accepted is False
        assert decision.critical is True
        assert dimension in decision.violated_dimensions


def test_declared_negative_and_boundary_fixtures_validate() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    assert validator.validate_family(value, observations=declared_fixtures(value)) == value


def test_incoherent_positive_examples_are_unsafe_boundary_overgeneralization() -> None:
    value = replace(
        family(),
        boundary=replace(
            family().boundary,
            positive_member_cids=("positive-a", "positive-b"),
        ),
    )
    observations = (
        matching_observation(value, "positive-a"),
        matching_observation(
            value,
            "positive-b",
            proof_obligations=("import-purity-proof", "additional-proof"),
        ),
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
        ),
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
        FamilyExampleObservation(example_cid="unknown-a"),
    )
    with pytest.raises(TaskFamilyOvergeneralizationError, match="coherent") as exc:
        TaskFamilyBoundaryValidator().validate_family(value, observations=observations)
    assert exc.value.reason_code == "overgeneralization"


def test_language_and_repository_boundary_mismatches_are_refused() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    language = validator.decide(
        value,
        matching_observation(value, "language-split-case", languages=("rust",)),
    )
    assert language.accepted is False
    assert language.critical is True
    assert "language" in language.violated_dimensions

    foreign = replace(
        value,
        bindings=replace(value.bindings, repository_id="other-repo"),
        boundary=replace(value.boundary, permitted_repositories=("repo",)),
    )
    with pytest.raises(TaskFamilyBoundaryError, match="permitted repositories") as exc:
        validator.validate_family(foreign)
    assert exc.value.reason_code == "repository_mismatch"


def test_task_family_p0_contract_helpers_remain_available_for_boundary_tests() -> None:
    value = family()
    assert parse_task_family(value.to_dict()).content_id == value.content_id
    assert validate_task_family_contract(value) == value
