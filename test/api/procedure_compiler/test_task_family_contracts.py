from __future__ import annotations

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
    TaskFamilyContractError,
    parse_task_family,
    parse_task_family_membership,
    validate_task_family_contract,
)


def family() -> TaskFamily:
    bindings = ArtifactBindings(
        "repo",
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-003",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )
    boundary = TaskFamilyBoundary(
        positive_member_cids=("positive-a",),
        negative_example_cids=("negative-a",),
        boundary_example_cids=("boundary-a",),
        unknown_case_cids=("unknown-a",),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
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
        validation_structure=("focused-tests", "postcondition-check"),
        failure_signatures=("import-side-effect",),
        postcondition_shape=("import-is-pure",),
        rollback_structure=("restore-exact-tree",),
        boundary=boundary,
    )


def test_task_family_wire_parser_round_trip() -> None:
    value = family()
    assert parse_task_family(value.to_dict()) == value
    assert parse_task_family(value.to_json()).content_id == value.content_id


@pytest.mark.parametrize(
    ("trajectory_cid", "membership"),
    [
        ("positive-a", FamilyMembershipClass.POSITIVE),
        ("negative-a", FamilyMembershipClass.NEGATIVE),
        ("boundary-a", FamilyMembershipClass.BOUNDARY),
        ("unknown-a", FamilyMembershipClass.UNKNOWN),
    ],
)
def test_membership_must_match_exact_declared_boundary(
    trajectory_cid: str, membership: FamilyMembershipClass
) -> None:
    value = family()
    record = TaskFamilyMembership(
        bindings=value.bindings,
        task_family_cid=value.content_id,
        trajectory_cid=trajectory_cid,
        membership=membership,
        evidence_cids=("classifier-receipt",),
        classifier_revision="baseline-v1",
    )
    assert parse_task_family_membership(record.to_dict(), value) == record


def test_membership_cannot_relabel_a_negative_as_positive() -> None:
    value = family()
    record = TaskFamilyMembership(
        bindings=value.bindings,
        task_family_cid=value.content_id,
        trajectory_cid="negative-a",
        membership=FamilyMembershipClass.POSITIVE,
        evidence_cids=("classifier-receipt",),
        classifier_revision="baseline-v1",
    )
    with pytest.raises(TaskFamilyContractError, match="contradicts"):
        parse_task_family_membership(record, value)


def test_known_material_counterexample_invalidates_family_boundary() -> None:
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
    with pytest.raises(TaskFamilyContractError, match="materially splits"):
        validate_task_family_contract(value, counterexamples=(counterexample,))


def test_task_family_module_does_not_offer_discovery_or_clustering() -> None:
    import ipfs_accelerate_py.agent_supervisor.procedure_compiler.task_family as module

    assert not hasattr(module, "discover_task_families")
    assert not hasattr(module, "TaskFamilyDiscoverer")
