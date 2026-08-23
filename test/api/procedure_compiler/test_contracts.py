from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler import contracts
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    BoundedArtifact,
    EffectClass,
    ExecutionTrajectory,
    ProcedureContractError,
    ProcedureEffect,
    ProcedureIdentityError,
    ProcedureInvocation,
    ProcedureParameter,
    ProcedureSafetyError,
    ProcedureStep,
    RiskClass,
    StepOperation,
    TaskFamilyBoundary,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
    ValueType,
    canonical_json_bytes,
    parse_procedure_artifact,
)

from ipfs_accelerate_py.agent_supervisor import procedure_compiler


def bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo-main",
        repository_commit="abc123",
        tree_id="tree-abc123",
        objective_id="PCPC-G000",
        task_id="PCPC-004",
        contract_revision="procedure-contracts-v1",
        policy_revision="authority-policy-v1",
        environment_id="python312-linux-lock1",
    )


def test_exact_bindings_round_trip_and_canonical_identity() -> None:
    value = bindings()
    decoded = ArtifactBindings.from_json(value.to_json())

    assert decoded == value
    assert decoded.content_id == value.content_id
    assert canonical_json_bytes(decoded.to_dict()) == decoded.canonical_bytes()
    with pytest.raises(FrozenInstanceError):
        decoded.tree_id = "other"  # type: ignore[misc]


def test_unknown_normative_field_and_forged_identity_are_rejected() -> None:
    payload = bindings().to_dict()
    payload["claim_completion"] = True
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        ArtifactBindings.from_dict(payload)

    payload = bindings().to_dict()
    payload["content_id"] = "forged"
    with pytest.raises(ProcedureIdentityError):
        ArtifactBindings.from_dict(payload)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"ratio": 0.5},
        {"callback": "not-even-code"},
        {"value": lambda: None},
        {"absolute_path": "/etc/passwd"},
        {"nested": {"api_key": "redacted"}},
    ],
)
def test_generic_artifacts_reject_float_executable_secret_and_absolute_path(unsafe: object) -> None:
    with pytest.raises(ProcedureContractError):
        contracts.SpecificationCandidate(bindings=bindings(), facts=unsafe)


def test_generic_artifact_is_bounded_and_uses_closed_state() -> None:
    with pytest.raises(ProcedureContractError):
        contracts.ExperimentPlan(bindings=bindings(), state="invented")
    with pytest.raises(ProcedureContractError):
        contracts.ExperimentPlan(bindings=bindings(), facts={f"k{i}": i for i in range(65)})

    record = contracts.ExperimentPlan(
        bindings=bindings(),
        state=ArtifactState.SHADOW,
        subject_cid="experiment-cid",
        reference_cids=("evidence-a",),
        facts={"maximum_executions": 2, "dry_run": True},
    )
    decoded = parse_procedure_artifact(record.to_dict())
    assert isinstance(decoded, contracts.ExperimentPlan)
    assert decoded == record
    assert isinstance(decoded, BoundedArtifact)


def test_all_closed_artifact_vocabulary_names_are_models() -> None:
    names = {
        "RepositoryWorldState",
        "AbstractRepositoryState",
        "WorldStateDelta",
        "TransitionObservation",
        "TransitionModel",
        "TransitionPrediction",
        "PredictionCalibration",
        "ExecutionTrajectory",
        "TrajectoryStep",
        "TrajectoryOutcome",
        "TrajectoryNormalizationReceipt",
        "TaskFamily",
        "TaskFamilyMembership",
        "TaskFamilyBoundary",
        "TaskFamilyCounterexample",
        "ProcedureSpec",
        "ProcedureVersion",
        "ProcedureParameter",
        "ProcedureLocal",
        "ProcedureStep",
        "ProcedureBranch",
        "ProcedureLoop",
        "ProcedureHole",
        "ProcedureEffect",
        "ProcedureObservation",
        "ProcedurePrecondition",
        "ProcedureInvariant",
        "ProcedurePostcondition",
        "ProcedureRollback",
        "ProcedureFallback",
        "ProcedureResourceEnvelope",
        "ProcedureAuthorityEnvelope",
        "ProcedureValidationPlan",
        "ProcedureCandidate",
        "ProcedureSynthesisPlan",
        "ProcedureSynthesisCounterexample",
        "ProcedureVerificationResult",
        "ProcedureCertificate",
        "ProcedureInvocation",
        "ProcedureInvocationReceipt",
        "ProcedureExecutionTrace",
        "ProcedureOutcome",
        "ProcedureFailure",
        "ProcedureRecoveryPlan",
        "SpecificationCandidate",
        "SpecificationEvidence",
        "SpecificationCounterexample",
        "SpecificationMiningReceipt",
        "InvariantCandidate",
        "InvariantValidationReceipt",
        "NonVacuityReceipt",
        "AntiUnificationPattern",
        "GeneralizationBoundary",
        "GeneralizationCounterexample",
        "ProcedureRegistry",
        "ProcedureRegistryRevision",
        "ProcedurePromotionReceipt",
        "ProcedureRollbackReceipt",
        "ProcedureDeprecationReceipt",
        "ProcedureDriftReport",
        "HoleRequest",
        "HoleCandidate",
        "HoleResolution",
        "HoleValidationReceipt",
        "DistillationCorpus",
        "DistillationExample",
        "DistillationEvaluation",
        "LocalDecisionModelArtifact",
        "GeneratedToolSpec",
        "GeneratedToolCandidate",
        "GeneratedToolCertificate",
        "GeneratedToolInvocationReceipt",
        "ExperimentPlan",
        "ExperimentObservation",
        "ExperimentEvaluation",
        "ProcedureCompilerRunReceipt",
        "ProcedureCompilerReleaseReceipt",
    }
    for name in names:
        model = getattr(procedure_compiler, name)
        assert issubclass(model, contracts.CanonicalContract)
        assert model.SCHEMA.endswith("@1")


def test_step_vocabulary_is_exact_and_forbidden_categories_fail_closed() -> None:
    assert set(contracts.ALLOWED_STEP_OPERATIONS) == {item.value for item in StepOperation}
    assert {
        "ARBITRARY_SHELL",
        "ARBITRARY_PYTHON",
        "ARBITRARY_NETWORK_REQUEST",
        "ARBITRARY_FILESYSTEM_PATH",
        "DISABLE_VALIDATION",
        "MODIFY_AUTHORITY_POLICY",
        "MODIFY_TRUSTED_KEYS",
        "CLAIM_COMPLETION",
    } == set(contracts.FORBIDDEN_STEP_OPERATIONS)
    with pytest.raises(ProcedureContractError):
        ProcedureStep("bad", "ARBITRARY_SHELL", "shell@1")


def test_step_contract_binds_retry_authority_effect_timeout_and_evidence() -> None:
    step = ProcedureStep(
        step_id="tests",
        operation=StepOperation.RUN_SELECTED_TESTS,
        operation_contract="test-runner@1",
        input_bindings={"selector": "parameter:selector"},
        output_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=("authority.execute-tests",),
        timeout_ms=30_000,
        evidence_outputs=("observation.tests",),
    )
    assert ProcedureStep.from_dict(step.to_dict()) == step
    assert step.operation is StepOperation.RUN_SELECTED_TESTS


def test_paths_are_repository_relative() -> None:
    with pytest.raises(ProcedureSafetyError):
        ProcedureEffect("write", EffectClass.REPOSITORY_WRITE, targets=("/tmp/escape.py",))
    with pytest.raises(ProcedureSafetyError):
        ProcedureEffect("write", EffectClass.REPOSITORY_WRITE, targets=("src/../escape.py",))


def test_parameter_values_are_closed_and_immutable() -> None:
    parameter = ProcedureParameter(
        name="outcome",
        value_type=ValueType.ENUM,
        allowed_values=("unavailable", "simulated"),
    )
    assert parameter.allowed_values == ("unavailable", "simulated")
    with pytest.raises(ProcedureContractError):
        ProcedureParameter(name="open", value_type=ValueType.ENUM)


def test_task_family_boundary_requires_disjoint_negative_and_positive_examples() -> None:
    with pytest.raises(ProcedureContractError, match="disjoint"):
        TaskFamilyBoundary(
            positive_member_cids=("episode-a",),
            negative_example_cids=("episode-a",),
            boundary_example_cids=("episode-b",),
            unknown_case_cids=(),
            risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
            permitted_repositories=("repo-main",),
            permitted_languages=("python",),
            permitted_frameworks=(),
            permitted_effect_classes=(EffectClass.REPOSITORY_WRITE,),
        )


def test_accepted_trajectory_requires_independently_admitted_source_and_validation() -> None:
    outcome = TrajectoryOutcome(
        status=TrajectoryTerminalStatus.ACCEPTED,
        accepted_criterion_ids=("criterion-1",),
        validation_receipt_cids=("test-receipt",),
        proof_receipt_cids=(),
    )
    step = TrajectoryStep(
        sequence=0,
        operation=StepOperation.RUN_SELECTED_TESTS,
        operation_contract="test-runner@1",
        initial_state_cid="state-before",
        terminal_state_cid="state-after",
        observation_cids=("observation",),
        effect_ids=("validation",),
        validation_receipt_cids=("test-receipt",),
    )
    trajectory = ExecutionTrajectory(
        bindings=bindings(),
        source_episode_cid="accepted-receipt",
        source_episode_kind=contracts.EpisodeKind.ACCEPTED_TASK_RECEIPT,
        initial_abstract_state_cid="state-before",
        terminal_abstract_state_cid="state-after",
        objective_criterion_ids=("criterion-1",),
        task_family_hint="IMPORT_PURITY_REPAIR",
        steps=(step,),
        outcome=outcome,
        total_cost_units=1,
        total_tokens=0,
        total_latency_ms=10,
        human_interventions=0,
    )
    assert ExecutionTrajectory.from_dict(trajectory.to_dict()) == trajectory


def test_invocation_binds_scope_lease_fence_and_idempotency() -> None:
    invocation = ProcedureInvocation(
        bindings=bindings(),
        procedure_cid="procedure-cid",
        certificate_cid="certificate-cid",
        registry_revision="registry-r1",
        parameters={"selector": "focused"},
        requested_scope=("ipfs_accelerate_py/agent_supervisor",),
        authority_receipt_cids=("authority-receipt",),
        idempotency_key="invocation-1",
        dry_run=True,
        requested_at_ms=1,
        lease_id="lease-1",
        fencing_token=3,
    )
    assert ProcedureInvocation.from_dict(invocation.to_dict()) == invocation
    with pytest.raises(ProcedureContractError, match="bound together"):
        ProcedureInvocation(
            bindings=bindings(),
            procedure_cid="procedure-cid",
            certificate_cid="certificate-cid",
            registry_revision="registry-r1",
            parameters={},
            requested_scope=("src",),
            authority_receipt_cids=(),
            idempotency_key="invocation-2",
            dry_run=True,
            requested_at_ms=1,
            lease_id="lease-1",
        )
