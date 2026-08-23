from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureCandidate,
    ProcedureEffect,
    ProcedureLocal,
    ProcedureObservation,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    RiskClass,
    StepOperation,
    TaskFamily,
    TaskFamilyBoundary,
    ValueType,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.verifier import (
    REQUIRED_EVIDENCE_KINDS,
    REQUIRED_VERIFICATION_LAYERS,
    VERIFIER_REVISION,
    AdmittedReceipt,
    IndependentEvidence,
    ProcedureVerificationError,
    ProcedureVerifier,
    VerificationLayer,
    VerificationPolicy,
    VerificationReasonCode,
    VerificationStatus,
    verify_procedure,
)


ISSUER_ID = "procedure-certificate-issuer@1"
EVIDENCE_PRODUCER = "independent-assurance-campaign@1"


def bindings(**changes: str) -> ArtifactBindings:
    values = {
        "repository_id": "repo-main",
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G000",
        "task_id": "PCPC-017",
        "contract_revision": "procedure-contracts-v1",
        "policy_revision": "authority-policy-v1",
        "environment_id": "python312-linux-lock1",
    }
    values.update(changes)
    return ArtifactBindings(**values)


def valid_spec(*, name: str = "focused-validation-procedure") -> ProcedureSpec:
    precondition = ProcedurePrecondition(
        condition_id="precondition.current-tree",
        binding="binding:tree_id",
        operator=ConditionOperator.CURRENT,
        evidence_producer="tree-verifier@1",
        evidence_type="current-tree-receipt@1",
    )
    postcondition = ProcedurePostcondition(
        condition_id="postcondition.tests-admitted",
        binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_producer="postcondition-checker@1",
        evidence_type="postcondition-receipt@1",
    )
    test_observation = ProcedureObservation(
        observation_id="observation.tests",
        producer_contract="test-runner@1",
        output_binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_type="test-receipt@1",
    )
    post_observation = ProcedureObservation(
        observation_id="observation.postcondition",
        producer_contract="postcondition-checker@1",
        output_binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
        evidence_type="postcondition-receipt@1",
    )
    validation_effect = ProcedureEffect(
        effect_id="effect.validation", effect_class=EffectClass.VALIDATION
    )
    receipt_effect = ProcedureEffect(
        effect_id="effect.receipt", effect_class=EffectClass.RECEIPT_EMIT
    )
    read = ProcedureStep(
        step_id="read",
        operation=StepOperation.READ_STATE,
        operation_contract="state-reader@1",
        output_bindings={"state": "local:state"},
        next_step_id="tests",
    )
    tests = ProcedureStep(
        step_id="tests",
        operation=StepOperation.RUN_SELECTED_TESTS,
        operation_contract="test-runner@1",
        input_bindings={"state": "local:state"},
        output_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("observation.tests",),
        next_step_id="postcondition",
    )
    check = ProcedureStep(
        step_id="postcondition",
        operation=StepOperation.CHECK_POSTCONDITION,
        operation_contract="postcondition-checker@1",
        input_bindings={"result": "local:test-result"},
        declared_effect_ids=("effect.validation",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("observation.postcondition",),
        next_step_id="receipt",
    )
    receipt = ProcedureStep(
        step_id="receipt",
        operation=StepOperation.EMIT_RECEIPT,
        operation_contract="receipt-emitter@1",
        declared_effect_ids=("effect.receipt",),
        required_authority_ids=("authority.execute",),
        evidence_outputs=("receipt.execution",),
    )
    steps = (read, tests, check, receipt)
    authority = ProcedureAuthorityEnvelope(
        authority_policy_revision="authority-policy-v1",
        requirement_ids=("authority.execute",),
        required_capability_ids=("capability.tests",),
        allowed_operations=tuple(step.operation for step in steps),
        risk_ceiling=RiskClass.OBSERVATION_ONLY,
    )
    resources = ProcedureResourceEnvelope(
        wall_time_ms=60_000,
        cpu_time_ms=60_000,
        memory_bytes=128_000_000,
        disk_bytes=128_000_000,
        model_token_limit=0,
        model_call_limit=0,
        subprocess_limit=4,
    )
    validation = ProcedureValidationPlan(
        required_step_ids=("tests", "postcondition"),
        required_observation_ids=("observation.tests", "observation.postcondition"),
        required_test_contracts=("focused-tests@1",),
        required_proof_contracts=("proof-runner@1",),
    )
    return ProcedureSpec(
        bindings=bindings(),
        name=name,
        version=ProcedureVersion(major=1),
        task_family_id="IMPORT_PURITY_REPAIR",
        entry_step_id="read",
        locals=(
            ProcedureLocal("state", ValueType.STRUCTURED),
            ProcedureLocal("test-result", ValueType.STRUCTURED),
        ),
        preconditions=(precondition,),
        declared_reads=("ipfs_accelerate_py/agent_supervisor/example.py",),
        declared_effects=(validation_effect, receipt_effect),
        steps=steps,
        postconditions=(postcondition,),
        observations=(test_observation, post_observation),
        validation=validation,
        authority=authority,
        resources=resources,
        terminal_step_ids=("receipt",),
        scope_paths=("ipfs_accelerate_py/agent_supervisor",),
        provenance_cids=("accepted-trajectory-cid",),
    )


def family_for(spec: ProcedureSpec, **changes: object) -> TaskFamily:
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "name": spec.task_family_id,
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": tuple(effect.effect_class for effect in spec.declared_effects),
        "required_operation_contracts": tuple(
            step.operation_contract for step in spec.steps
        ),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "failure_signatures": ("import-side-effect",),
        "postcondition_shape": ("import-is-pure",),
        "rollback_structure": ("restore-exact-tree",),
        "boundary": TaskFamilyBoundary(
            positive_member_cids=("positive-a",),
            negative_example_cids=("negative-a",),
            boundary_example_cids=("boundary-a",),
            unknown_case_cids=("unknown-a",),
            risk_ceiling=spec.authority.risk_ceiling,
            permitted_repositories=(spec.bindings.repository_id,),
            permitted_languages=("python",),
            permitted_frameworks=("stdlib",),
            permitted_effect_classes=tuple(
                effect.effect_class for effect in spec.declared_effects
            ),
        ),
    }
    values.update(changes)
    return TaskFamily(**values)


def candidate_for(spec: ProcedureSpec, **changes: object) -> ProcedureCandidate:
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "procedure": spec,
        "synthesis_plan_cid": "synthesis-plan-1",
        "source_episode_cids": ("episode-1",),
        "counterexample_set_cid": "counterexamples-1",
        "state": ArtifactState.CANDIDATE,
    }
    values.update(changes)
    return ProcedureCandidate(**values)


def receipt(
    cid: str,
    kind: str,
    *,
    contract_id: str = "",
    bound: ArtifactBindings | None = None,
    producer_id: str = EVIDENCE_PRODUCER,
    observed_at_ms: int = 50,
    expires_at_ms: int = 10_000,
) -> AdmittedReceipt:
    return AdmittedReceipt(
        receipt_cid=cid,
        kind=kind,
        producer_id=producer_id,
        bindings=bound or bindings(),
        observed_at_ms=observed_at_ms,
        expires_at_ms=expires_at_ms,
        contract_id=contract_id,
    )


def evidence_for(
    spec: ProcedureSpec,
    *,
    family: TaskFamily | None = None,
    producer_id: str = EVIDENCE_PRODUCER,
    include_receipts: bool = True,
    **changes: object,
) -> IndependentEvidence:
    family = family or family_for(spec)
    values: dict[str, object] = {
        "producer_id": producer_id,
        "task_family": family,
        "source_episode_cids": ("episode-1", "episode-2"),
        "specification_cids": ("specification-1",),
        "counterexample_set_cid": "counterexamples-1",
        "proof_receipt_cids": ("proof-1",),
        "test_receipt_cids": ("test-1",),
        "adversarial_assurance_cids": ("assurance-1",),
        "held_out_evaluation_cid": "held-out-1",
        "shadow_evaluation_cid": "shadow-1",
        "repository_families": (spec.bindings.repository_id,),
        "supported_language_classes": ("python",),
        "supported_framework_classes": ("stdlib",),
        "known_limitations": ("shadow-evaluation-is-not-promotion",),
        "observed_at_ms": 50,
        "receipts": (
            (
                receipt("proof-1", "proof", contract_id="proof-runner@1"),
                receipt("test-1", "test", contract_id="focused-tests@1"),
                receipt("assurance-1", "adversarial"),
                receipt("held-out-1", "held_out"),
                receipt("shadow-1", "shadow"),
            )
            if include_receipts
            else ()
        ),
    }
    values.update(changes)
    return IndependentEvidence(**values)


def policy_for(spec: ProcedureSpec, **changes: object) -> VerificationPolicy:
    values: dict[str, object] = {
        "revision": "verification-1",
        "bindings": spec.bindings,
        "operation_catalog_revision": "catalog-1",
        "effect_policy_revision": "effects-1",
        "authority_policy_revision": spec.bindings.policy_revision,
        "required_test_contracts": ("focused-tests@1",),
        "required_proof_contracts": ("proof-runner@1",),
        "require_adversarial": True,
        "require_held_out": True,
        "require_shadow": True,
        "confirmation_required": False,
        "max_risk_ceiling": RiskClass.REPOSITORY_WRITE,
        "review_horizon_ms": 9_000,
    }
    values.update(changes)
    return VerificationPolicy(**values)


def test_independently_verified_candidate_passes_every_required_layer() -> None:
    spec = valid_spec()
    candidate = candidate_for(spec)
    evidence = evidence_for(spec)
    policy = policy_for(spec)

    result = ProcedureVerifier().verify(candidate, evidence, policy, now_ms=100)

    assert result.status is VerificationStatus.ACCEPTED
    assert result.accepted
    assert result.reason_code is VerificationReasonCode.ACCEPTED
    assert result.candidate_cid == candidate.content_id
    assert result.procedure_cid == spec.content_id
    assert result.producer_id == EVIDENCE_PRODUCER
    assert result.policy_revision == "verification-1"
    assert tuple(item.layer.value for item in result.layers) == REQUIRED_VERIFICATION_LAYERS
    assert all(item.accepted for item in result.layers)
    assert result.artifact.state is ArtifactState.VERIFIED
    assert result.artifact.subject_cid == candidate.content_id
    assert result.artifact.facts["verifier_revision"] == VERIFIER_REVISION
    assert set(result.artifact.facts["layers"].values()) == {True}
    assert set(REQUIRED_EVIDENCE_KINDS).issubset(set(policy.required_evidence_kinds))
    assert "promote" not in result.artifact.facts


def test_verify_procedure_helper_matches_class() -> None:
    spec = valid_spec()
    result = verify_procedure(
        candidate_for(spec), evidence_for(spec), policy_for(spec), now_ms=100
    )
    assert result.accepted


def test_rejected_candidate_cannot_be_verified() -> None:
    spec = valid_spec()
    result = ProcedureVerifier().verify(
        candidate_for(spec, state=ArtifactState.REJECTED),
        evidence_for(spec),
        policy_for(spec),
        now_ms=100,
    )
    assert not result.accepted
    assert result.reason_code is VerificationReasonCode.CANDIDATE_REJECTED
    assert result.artifact.state is ArtifactState.REJECTED
    assert all(not item.accepted for item in result.layers)


def test_self_produced_evidence_is_not_independent() -> None:
    spec = valid_spec()
    candidate = candidate_for(spec)
    result = ProcedureVerifier().verify(
        candidate,
        evidence_for(spec, producer_id=candidate.content_id, include_receipts=False),
        policy_for(spec),
        now_ms=100,
    )
    assert not result.accepted
    assert result.reason_code is VerificationReasonCode.SELF_CERTIFICATION


def test_structural_layer_rejects_unreachable_control_flow() -> None:
    spec = valid_spec()
    extra = ProcedureStep(
        step_id="orphan",
        operation=StepOperation.READ_STATE,
        operation_contract="state-reader@1",
        output_bindings={"state": "local:state"},
    )
    broken = replace(spec, steps=spec.steps + (extra,))
    result = ProcedureVerifier().verify(
        candidate_for(broken), evidence_for(broken), policy_for(broken), now_ms=100
    )
    assert not result.accepted
    assert result.outcome(VerificationLayer.STRUCTURAL).accepted is False
    assert (
        result.outcome(VerificationLayer.STRUCTURAL).reason_code
        == VerificationReasonCode.STRUCTURAL_UNSAFE.value
    )


def test_authority_layer_rejects_stale_policy_and_risk_escalation() -> None:
    spec = valid_spec()
    stale_policy = policy_for(
        spec,
        bindings=bindings(policy_revision="authority-policy-v1", tree_id="tree-other"),
        authority_policy_revision="authority-policy-v1",
    )
    stale = ProcedureVerifier().verify(
        candidate_for(spec), evidence_for(spec), stale_policy, now_ms=100
    )
    assert not stale.accepted
    assert stale.outcome(VerificationLayer.AUTHORITY).reason_code in {
        VerificationReasonCode.STALE_BINDINGS.value,
        VerificationReasonCode.STALE_POLICY.value,
    }

    high_risk = replace(
        spec,
        authority=replace(spec.authority, risk_ceiling=RiskClass.AUTHORITY_OR_SECURITY),
    )
    family = family_for(
        high_risk,
        boundary=replace(
            family_for(spec).boundary, risk_ceiling=RiskClass.OBSERVATION_ONLY
        ),
    )
    risk = ProcedureVerifier().verify(
        candidate_for(high_risk),
        evidence_for(high_risk, family=family),
        policy_for(high_risk, max_risk_ceiling=RiskClass.OBSERVATION_ONLY),
        now_ms=100,
    )
    assert not risk.accepted
    assert (
        risk.outcome(VerificationLayer.AUTHORITY).reason_code
        == VerificationReasonCode.AUTHORITY_UNSAFE.value
    )


def test_effect_layer_rejects_family_boundary_escape() -> None:
    spec = valid_spec()
    family = family_for(
        spec,
        effect_classes=(EffectClass.VALIDATION,),
        boundary=replace(
            family_for(spec).boundary,
            permitted_effect_classes=(EffectClass.VALIDATION,),
        ),
    )
    result = ProcedureVerifier().verify(
        candidate_for(spec), evidence_for(spec, family=family), policy_for(spec), now_ms=100
    )
    assert not result.accepted
    assert (
        result.outcome(VerificationLayer.EFFECT).reason_code
        == VerificationReasonCode.EFFECT_UNSAFE.value
    )


def test_dataflow_layer_rejects_uninitialized_local() -> None:
    spec = valid_spec()
    read, tests, check, receipt = spec.steps
    tests = replace(tests, input_bindings={"state": "local:test-result"})
    broken = replace(spec, steps=(read, tests, check, receipt))
    result = ProcedureVerifier().verify(
        candidate_for(broken), evidence_for(broken), policy_for(broken), now_ms=100
    )
    assert not result.accepted
    assert (
        result.outcome(VerificationLayer.DATAFLOW).reason_code
        == VerificationReasonCode.DATAFLOW_UNSAFE.value
    )


def test_temporal_layer_rejects_stale_or_future_evidence() -> None:
    spec = valid_spec()
    future = ProcedureVerifier().verify(
        candidate_for(spec),
        evidence_for(spec, observed_at_ms=500, include_receipts=False),
        policy_for(spec),
        now_ms=100,
    )
    assert not future.accepted
    assert (
        future.outcome(VerificationLayer.TEMPORAL).reason_code
        == VerificationReasonCode.TEMPORAL_UNSAFE.value
    )

    expired = ProcedureVerifier().verify(
        candidate_for(spec),
        evidence_for(
            spec,
            receipts=(
                receipt("proof-1", "proof", contract_id="proof-runner@1", expires_at_ms=20),
                receipt("test-1", "test", contract_id="focused-tests@1"),
                receipt("assurance-1", "adversarial"),
                receipt("held-out-1", "held_out"),
                receipt("shadow-1", "shadow"),
            ),
        ),
        policy_for(spec),
        now_ms=100,
    )
    assert not expired.accepted
    assert expired.outcome(VerificationLayer.TEMPORAL).reason_code in {
        VerificationReasonCode.STALE_BINDINGS.value,
        VerificationReasonCode.TEMPORAL_UNSAFE.value,
    }


def test_semantic_layer_rejects_family_and_counterexample_mismatch() -> None:
    spec = valid_spec()
    other = family_for(spec, name="OTHER_FAMILY")
    family_mismatch = ProcedureVerifier().verify(
        candidate_for(spec),
        evidence_for(spec, family=other),
        policy_for(spec),
        now_ms=100,
    )
    assert not family_mismatch.accepted
    assert (
        family_mismatch.outcome(VerificationLayer.SEMANTIC).reason_code
        == VerificationReasonCode.SEMANTIC_UNSAFE.value
    )

    cex = ProcedureVerifier().verify(
        candidate_for(spec, counterexample_set_cid="counterexamples-other"),
        evidence_for(spec),
        policy_for(spec),
        now_ms=100,
    )
    assert not cex.accepted
    assert (
        cex.outcome(VerificationLayer.SEMANTIC).reason_code
        == VerificationReasonCode.SEMANTIC_UNSAFE.value
    )


def test_validation_layer_rejects_weaker_or_incomplete_evidence() -> None:
    spec = valid_spec()
    weaker_plan = replace(spec.validation, required_test_contracts=(), required_proof_contracts=())
    weaker = replace(spec, validation=weaker_plan)
    weakened = ProcedureVerifier().verify(
        candidate_for(weaker), evidence_for(weaker), policy_for(weaker), now_ms=100
    )
    assert not weakened.accepted
    assert (
        weakened.outcome(VerificationLayer.VALIDATION).reason_code
        == VerificationReasonCode.VALIDATION_WEAKENED.value
    )

    uncovered = evidence_for(
        spec,
        receipts=(
            receipt("test-1", "test", contract_id="focused-tests@1"),
            receipt("assurance-1", "adversarial"),
            receipt("held-out-1", "held_out"),
            receipt("shadow-1", "shadow"),
        ),
    )
    uncovered_result = ProcedureVerifier().verify(
        candidate_for(spec), uncovered, policy_for(spec), now_ms=100
    )
    assert not uncovered_result.accepted
    assert uncovered_result.outcome(VerificationLayer.VALIDATION).reason_code in {
        VerificationReasonCode.VALIDATION_INCOMPLETE.value,
        VerificationReasonCode.VALIDATION_WEAKENED.value,
    }


def test_validation_rejects_self_identity_used_as_evidence() -> None:
    spec = valid_spec()
    candidate = candidate_for(spec)
    result = ProcedureVerifier().verify(
        candidate,
        evidence_for(
            spec,
            include_receipts=False,
            proof_receipt_cids=(candidate.content_id,),
        ),
        policy_for(spec),
        now_ms=100,
    )
    assert not result.accepted
    assert result.reason_code is VerificationReasonCode.SELF_CERTIFICATION


def test_stale_receipt_bindings_fail_validation() -> None:
    spec = valid_spec()
    stale = evidence_for(
        spec,
        receipts=(
            receipt(
                "proof-1",
                "proof",
                contract_id="proof-runner@1",
                bound=bindings(tree_id="tree-old"),
            ),
            receipt("test-1", "test", contract_id="focused-tests@1"),
            receipt("assurance-1", "adversarial"),
            receipt("held-out-1", "held_out"),
            receipt("shadow-1", "shadow"),
        ),
    )
    result = ProcedureVerifier().verify(
        candidate_for(spec), stale, policy_for(spec), now_ms=100
    )
    assert not result.accepted
    assert (
        result.outcome(VerificationLayer.VALIDATION).reason_code
        == VerificationReasonCode.STALE_BINDINGS.value
    )


def test_policy_cannot_drop_required_layers_or_evidence_kinds() -> None:
    spec = valid_spec()
    with pytest.raises(ProcedureVerificationError, match="omits a required layer"):
        policy_for(spec, required_layers=("structural", "validation"))
    with pytest.raises(ProcedureVerificationError, match="omits a required evidence kind"):
        policy_for(spec, required_evidence_kinds=("proof", "test"))


def test_malformed_inputs_fail_closed() -> None:
    spec = valid_spec()
    verifier = ProcedureVerifier()
    with pytest.raises(ProcedureVerificationError, match="candidate must be"):
        verifier.verify(spec, evidence_for(spec), policy_for(spec), now_ms=100)  # type: ignore[arg-type]
    with pytest.raises(ProcedureVerificationError, match="evidence must be"):
        verifier.verify(candidate_for(spec), spec, policy_for(spec), now_ms=100)  # type: ignore[arg-type]
    with pytest.raises(ProcedureVerificationError, match="policy must be"):
        verifier.verify(candidate_for(spec), evidence_for(spec), spec, now_ms=100)  # type: ignore[arg-type]


def test_forbidden_self_producer_names_are_rejected_at_construction() -> None:
    spec = valid_spec()
    with pytest.raises(ProcedureVerificationError, match="not independent"):
        evidence_for(spec, producer_id="self")
