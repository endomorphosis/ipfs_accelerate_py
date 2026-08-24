from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureCertificate,
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
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.transfer import (
    GATE_REVISION,
    INSUFFICIENT_SIMILARITY_SIGNALS,
    REQUIRED_COMPATIBILITY_DIMENSIONS,
    UNSAFE_TRANSFER_COUNT,
    GeneralizationBoundaryEvaluator,
    HeldOutRepositoryResult,
    ProcedureTransferGate,
    SimilaritySignals,
    TargetRepository,
    TransferAction,
    TransferDecision,
    TransferDimension,
    TransferReason,
    TransferRefusalError,
    TransferRequest,
    evaluate_transfer,
)


SOURCE_REPOSITORY = "repo-main"
TARGET_REPOSITORY = "repo-held-out"
REPOSITORY_FAMILY = "python-repair-family"
LANGUAGE = "python"
FRAMEWORK = "stdlib"
SCOPE = "ipfs_accelerate_py/agent_supervisor"


def _bindings(*, repository: str = SOURCE_REPOSITORY) -> ArtifactBindings:
    return ArtifactBindings(
        repository_id=repository,
        repository_commit="commit-1",
        tree_id="tree-1",
        objective_id="PCPC-G000",
        task_id="PCPC-025",
        contract_revision="contract-1",
        policy_revision="policy-1",
        environment_id="environment-1",
    )


def _spec() -> ProcedureSpec:
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
    return ProcedureSpec(
        bindings=_bindings(),
        name="focused-validation-procedure",
        version=ProcedureVersion(major=1),
        task_family_id="IMPORT_PURITY_REPAIR",
        entry_step_id="read",
        locals=(
            ProcedureLocal("state", ValueType.STRUCTURED),
            ProcedureLocal("test-result", ValueType.STRUCTURED),
        ),
        preconditions=(precondition,),
        declared_reads=(SCOPE + "/example.py",),
        declared_effects=(validation_effect, receipt_effect),
        steps=steps,
        postconditions=(postcondition,),
        observations=(test_observation, post_observation),
        validation=ProcedureValidationPlan(
            required_step_ids=("tests", "postcondition"),
            required_observation_ids=("observation.tests", "observation.postcondition"),
            required_test_contracts=("focused-tests@1",),
            required_proof_contracts=("proof-runner@1",),
        ),
        authority=ProcedureAuthorityEnvelope(
            authority_policy_revision="policy-1",
            requirement_ids=("authority.execute",),
            required_capability_ids=("capability.tests",),
            allowed_operations=tuple(step.operation for step in steps),
            risk_ceiling=RiskClass.OBSERVATION_ONLY,
        ),
        resources=ProcedureResourceEnvelope(
            wall_time_ms=60_000,
            cpu_time_ms=60_000,
            memory_bytes=128_000_000,
            disk_bytes=128_000_000,
            model_token_limit=0,
            model_call_limit=0,
            subprocess_limit=4,
        ),
        terminal_step_ids=("receipt",),
        scope_paths=(SCOPE,),
        provenance_cids=("accepted-trajectory-cid",),
    )


def _family(spec: ProcedureSpec, **changes: object) -> TaskFamily:
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "name": spec.task_family_id,
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": tuple(effect.effect_class for effect in spec.declared_effects),
        "required_operation_contracts": tuple(step.operation_contract for step in spec.steps),
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
            permitted_repositories=(SOURCE_REPOSITORY, TARGET_REPOSITORY),
            permitted_languages=(LANGUAGE,),
            permitted_frameworks=(FRAMEWORK,),
            permitted_effect_classes=tuple(
                effect.effect_class for effect in spec.declared_effects
            ),
        ),
    }
    values.update(changes)
    return TaskFamily(**values)


def _certificate(spec: ProcedureSpec, **changes: object) -> ProcedureCertificate:
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "procedure_cid": spec.content_id,
        "procedure_version": spec.version,
        "task_family_cid": spec.task_family_id,
        "source_episode_cids": ("episode-1",),
        "specification_cids": ("specification-1",),
        "counterexample_set_cid": "counterexamples-1",
        "operation_catalog_revision": "catalog-1",
        "effect_policy_revision": "effects-1",
        "authority_policy_revision": spec.bindings.policy_revision,
        "verification_policy_revision": "verification-1",
        "repository_families": (REPOSITORY_FAMILY,),
        "supported_language_classes": (LANGUAGE,),
        "supported_framework_classes": (FRAMEWORK,),
        "risk_ceiling": spec.authority.risk_ceiling,
        "proof_receipt_cids": ("proof-1",),
        "test_receipt_cids": ("test-1",),
        "adversarial_assurance_cids": ("assurance-1",),
        "held_out_evaluation_cid": "source-held-out-1",
        "shadow_evaluation_cid": "shadow-1",
        "known_limitations": (),
        "issuer": "independent-issuer",
        "signature": "independently-verified-test-signature",
        "issued_at_ms": 1,
        "expires_at_ms": 10_000,
        "state": ArtifactState.VERIFIED,
    }
    values.update(changes)
    return ProcedureCertificate(**values)


def _target(spec: ProcedureSpec, **changes: object) -> TargetRepository:
    values: dict[str, object] = {
        "repository_id": TARGET_REPOSITORY,
        "tree_id": "target-tree-1",
        "repository_family": REPOSITORY_FAMILY,
        "language_classes": (LANGUAGE,),
        "framework_classes": (FRAMEWORK,),
        "permitted_operations": tuple(step.operation for step in spec.steps),
        "permitted_operation_contracts": tuple(step.operation_contract for step in spec.steps),
        "permitted_effect_classes": tuple(
            effect.effect_class for effect in spec.declared_effects
        ),
        "authority_policy_revision": spec.authority.authority_policy_revision,
        "authority_requirement_ids": spec.authority.requirement_ids,
        "required_capability_ids": spec.authority.required_capability_ids,
        "validation_contracts": (
            *spec.validation.required_test_contracts,
            *spec.validation.required_proof_contracts,
        ),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "path_prefixes": spec.scope_paths,
        "risk_ceiling": spec.authority.risk_ceiling,
        "operation_catalog_revision": "catalog-1",
        "effect_policy_revision": "effects-1",
        "authorized": True,
        "production": False,
        "policy_mutable": False,
        "name": "import-purity-repair",
        "maintainer_id": "shared-maintainer",
        "description": "repair import purity",
        "embedding_id": "embedding-near-match",
    }
    values.update(changes)
    return TargetRepository(**values)


def _held_out(**changes: object) -> HeldOutRepositoryResult:
    values: dict[str, object] = {
        "evaluation_cid": "target-held-out-1",
        "repository_id": TARGET_REPOSITORY,
        "tree_id": "target-tree-1",
        "passed": True,
        "read_only": True,
        "disposable": True,
        "production": False,
        "policy_mutable": False,
        "authorized": True,
        "mutate": False,
        "scope_paths": (SCOPE,),
        "observed_postcondition_count": 1,
        "observed_validation_count": 1,
    }
    values.update(changes)
    return HeldOutRepositoryResult(**values)


def _similar() -> SimilaritySignals:
    return SimilaritySignals(
        name_similar=True,
        embedding_similar=True,
        description_similar=True,
        language_similar=True,
        maintainer_similar=True,
    )


def _request(**changes: object) -> TransferRequest:
    spec = changes.pop("procedure", None) or _spec()
    family = changes.pop("family", None) or _family(spec)
    certificate = changes.pop("certificate", None) or _certificate(spec)
    target = changes.pop("target", None) or _target(spec)
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "procedure": spec,
        "certificate": certificate,
        "family": family,
        "target": target,
        "held_out": _held_out(),
        "similarity": SimilaritySignals(),
        "mutate_target": False,
        "experiment_authorizes": False,
    }
    values.update(changes)
    return TransferRequest(**values)


def _walk_has_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_walk_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_walk_has_float(item) for item in value)
    return False


def test_compatible_transfer_receives_bounded_candidate_eligibility() -> None:
    gate = ProcedureTransferGate()
    decision = gate.evaluate(_request())

    assert decision.action is TransferAction.ELIGIBLE
    assert decision.reason_code is TransferReason.COMPATIBLE
    assert decision.eligible is True
    assert decision.eligibility_state is ArtifactState.CANDIDATE
    assert decision.eligibility_state is not ArtifactState.PROMOTED
    assert decision.eligibility_state is not ArtifactState.VERIFIED
    assert decision.changed_assumptions == ()
    assert tuple(decision.compatible_dimensions) == REQUIRED_COMPATIBILITY_DIMENSIONS
    assert decision.held_out_passed is True
    assert decision.held_out_evaluation_cid == "target-held-out-1"
    assert decision.unsafe_transfer_count == UNSAFE_TRANSFER_COUNT == 0
    assert gate.unsafe_transfer_count == 0
    assert decision.can_mutate_target is False
    assert decision.can_authorize is False
    assert decision.can_promote is False
    assert decision.can_grant_authority is False
    assert decision.can_establish_proof is False
    assert decision.can_establish_postcondition is False
    assert decision.can_establish_completion is False
    assert decision.similarity_used_as_evidence is False
    assert decision.gate_revision == GATE_REVISION
    assert not _walk_has_float(decision.to_dict())

    decoded = TransferDecision.from_dict(decision.to_dict())
    assert decoded == decision
    assert parse_procedure_artifact(decision.to_dict()) == decision

    required = gate.require(_request())
    assert required.eligible is True
    assert evaluate_transfer(_request()).eligible is True


@pytest.mark.parametrize(
    ("change", "reason", "dimension"),
    (
        (
            lambda spec: {"target": _target(spec, permitted_operations=(StepOperation.ESCALATE,))},
            TransferReason.OPERATION_INCOMPATIBLE,
            TransferDimension.OPERATION,
        ),
        (
            lambda spec: {
                "target": _target(spec, permitted_effect_classes=(EffectClass.MERGE,))
            },
            TransferReason.EFFECT_INCOMPATIBLE,
            TransferDimension.EFFECT,
        ),
        (
            lambda spec: {
                "target": _target(spec, authority_policy_revision="other-policy")
            },
            TransferReason.AUTHORITY_INCOMPATIBLE,
            TransferDimension.AUTHORITY,
        ),
        (
            lambda spec: {"target": _target(spec, language_classes=("rust",))},
            TransferReason.LANGUAGE_INCOMPATIBLE,
            TransferDimension.LANGUAGE,
        ),
        (
            lambda spec: {"target": _target(spec, framework_classes=("pytest",))},
            TransferReason.FRAMEWORK_INCOMPATIBLE,
            TransferDimension.FRAMEWORK,
        ),
        (
            lambda spec: {
                "target": _target(spec, validation_contracts=("other-tests@1",))
            },
            TransferReason.VALIDATION_INCOMPATIBLE,
            TransferDimension.VALIDATION,
        ),
        (
            lambda spec: {
                "family": _family(
                    spec,
                    boundary=replace(
                        _family(spec).boundary,
                        permitted_repositories=(SOURCE_REPOSITORY,),
                    ),
                )
            },
            TransferReason.FAMILY_INCOMPATIBLE,
            TransferDimension.FAMILY,
        ),
        (
            lambda spec: {"target": _target(spec, path_prefixes=("lib",))},
            TransferReason.PATH_INCOMPATIBLE,
            TransferDimension.PATH,
        ),
        (
            lambda spec: {"held_out": None},
            TransferReason.HELD_OUT_MISSING,
            TransferDimension.HELD_OUT,
        ),
        (
            lambda spec: {"held_out": _held_out(passed=False)},
            TransferReason.HELD_OUT_FAILED,
            TransferDimension.HELD_OUT,
        ),
        (
            lambda spec: {
                "held_out": _held_out(repository_id=SOURCE_REPOSITORY, tree_id="tree-1")
            },
            TransferReason.HELD_OUT_REPOSITORY_MISMATCH,
            TransferDimension.HELD_OUT,
        ),
    ),
)
def test_every_changed_assumption_returns_typed_refusal(change, reason, dimension) -> None:
    spec = _spec()
    decision = ProcedureTransferGate().evaluate(_request(procedure=spec, **change(spec)))

    assert decision.action is TransferAction.REFUSE
    assert decision.eligible is False
    assert decision.reason_code is reason
    assert reason in decision.reason_codes
    assert dimension.value in decision.changed_assumptions
    assert decision.eligibility_state is ArtifactState.REJECTED
    assert decision.unsafe_transfer_count == 0
    assert decision.can_promote is False
    assert decision.can_mutate_target is False


def test_similar_names_embeddings_language_and_maintainer_never_suffice() -> None:
    spec = _spec()
    similar = _similar()
    assert set(INSUFFICIENT_SIMILARITY_SIGNALS) == {
        "name",
        "embedding",
        "description",
        "language",
        "maintainer",
    }
    assert similar.any_similar is True
    assert similar.asserted_signals == INSUFFICIENT_SIMILARITY_SIGNALS

    path_changed = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            target=_target(spec, path_prefixes=("vendor/other",)),
            similarity=similar,
        )
    )
    assert path_changed.action is TransferAction.REFUSE
    assert path_changed.reason_code is TransferReason.PATH_INCOMPATIBLE
    assert TransferReason.SIMILARITY_INSUFFICIENT in path_changed.reason_codes
    assert path_changed.similarity_signals == INSUFFICIENT_SIMILARITY_SIGNALS
    assert path_changed.similarity_used_as_evidence is False
    assert path_changed.eligible is False

    held_out_missing = ProcedureTransferGate().evaluate(
        _request(procedure=spec, held_out=None, similarity=similar)
    )
    assert held_out_missing.reason_code is TransferReason.HELD_OUT_MISSING
    assert held_out_missing.eligible is False

    compatible_with_similarity = ProcedureTransferGate().evaluate(
        _request(procedure=spec, similarity=similar)
    )
    assert compatible_with_similarity.eligible is True
    assert compatible_with_similarity.similarity_used_as_evidence is False
    assert compatible_with_similarity.similarity_signals == INSUFFICIENT_SIMILARITY_SIGNALS


def test_unsafe_transfer_count_remains_zero_for_mutation_and_refusals() -> None:
    gate = ProcedureTransferGate()
    spec = _spec()
    attempts = (
        _request(procedure=spec, mutate_target=True),
        _request(procedure=spec, experiment_authorizes=True),
        _request(procedure=spec, target=_target(spec, production=True)),
        _request(procedure=spec, target=_target(spec, policy_mutable=True)),
        _request(procedure=spec, target=_target(spec, authorized=False)),
        _request(procedure=spec, held_out=_held_out(mutate=True)),
        _request(procedure=spec, held_out=_held_out(production=True)),
        _request(procedure=spec, held_out=_held_out(read_only=False)),
        _request(procedure=spec, held_out=_held_out(disposable=False)),
        _request(procedure=spec, target=_target(spec, path_prefixes=("other",))),
    )
    reasons = []
    for request in attempts:
        decision = gate.evaluate(request)
        reasons.append(decision.reason_code)
        assert decision.action is TransferAction.REFUSE
        assert decision.unsafe_transfer_count == 0
        assert decision.can_mutate_target is False
        assert decision.eligible is False
    assert TransferReason.CROSS_REPOSITORY_MUTATION in reasons
    assert TransferReason.EXPERIMENT_CANNOT_AUTHORIZE in reasons
    assert TransferReason.PRODUCTION_MUTATION in reasons
    assert TransferReason.POLICY_MUTATION in reasons
    assert TransferReason.TARGET_NOT_AUTHORIZED in reasons
    assert TransferReason.UNSAFE_FIXTURE in reasons
    assert gate.unsafe_transfer_count == 0
    assert UNSAFE_TRANSFER_COUNT == 0

    with pytest.raises(TransferRefusalError, match="cross-repository-mutation") as raised:
        gate.require(_request(procedure=spec, mutate_target=True))
    assert raised.value.decision.unsafe_transfer_count == 0


def test_generalization_boundary_evaluator_admits_permitted_target_and_refuses_splits() -> None:
    spec = _spec()
    family = _family(spec)
    evaluator = GeneralizationBoundaryEvaluator()
    admitted = evaluator.evaluate(family, _target(spec), procedure=spec, certificate=_certificate(spec))
    assert admitted.admitted is True
    assert admitted.reason_code is TransferReason.COMPATIBLE
    assert admitted.changed_dimensions == ()
    assert admitted.artifact is not None
    assert admitted.artifact.state is ArtifactState.CANDIDATE
    assert admitted.artifact.facts["can_authorize"] is False
    assert parse_procedure_artifact(admitted.artifact.to_dict()) == admitted.artifact

    outside = evaluator.evaluate(
        family,
        _target(spec, repository_id="unpermitted-repo", language_classes=("rust",)),
        procedure=spec,
        certificate=_certificate(spec),
    )
    assert outside.admitted is False
    assert TransferDimension.FAMILY in outside.changed_dimensions
    assert TransferDimension.LANGUAGE in outside.changed_dimensions
    assert TransferReason.FAMILY_INCOMPATIBLE in outside.reason_codes
    assert TransferReason.LANGUAGE_INCOMPATIBLE in outside.reason_codes
    assert outside.artifact.state is ArtifactState.REJECTED

    incomplete = evaluator.evaluate(
        _family(spec, boundary=replace(_family(spec).boundary, unknown_case_cids=())),
        _target(spec),
        procedure=spec,
    )
    assert incomplete.admitted is False
    assert incomplete.reason_code is TransferReason.INCOMPLETE_BOUNDARY
    assert "unknown_case_cids" in incomplete.missing_dimensions


def test_held_out_must_be_nonvacuous_read_only_and_target_bound() -> None:
    spec = _spec()
    vacuous = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            held_out=_held_out(observed_validation_count=0, observed_postcondition_count=0),
        )
    )
    assert vacuous.reason_code is TransferReason.HELD_OUT_FAILED
    assert TransferDimension.HELD_OUT.value in vacuous.changed_assumptions

    source_held_out = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            held_out=_held_out(
                evaluation_cid="source-held-out-1",
                repository_id=SOURCE_REPOSITORY,
                tree_id="tree-1",
            ),
        )
    )
    assert source_held_out.reason_code is TransferReason.HELD_OUT_REPOSITORY_MISMATCH
    assert source_held_out.eligible is False

    writable = ProcedureTransferGate().evaluate(
        _request(procedure=spec, held_out=_held_out(read_only=False, disposable=True))
    )
    assert writable.reason_code is TransferReason.UNSAFE_FIXTURE


def test_multiple_changed_assumptions_are_all_reported() -> None:
    spec = _spec()
    decision = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            target=_target(
                spec,
                permitted_operations=(StepOperation.ESCALATE,),
                language_classes=("rust",),
                path_prefixes=("vendor",),
            ),
            held_out=None,
            similarity=_similar(),
        )
    )
    assert decision.action is TransferAction.REFUSE
    assert TransferDimension.OPERATION.value in decision.changed_assumptions
    assert TransferDimension.LANGUAGE.value in decision.changed_assumptions
    assert TransferDimension.PATH.value in decision.changed_assumptions
    assert TransferDimension.HELD_OUT.value in decision.changed_assumptions
    assert TransferReason.OPERATION_INCOMPATIBLE in decision.reason_codes
    assert TransferReason.LANGUAGE_INCOMPATIBLE in decision.reason_codes
    assert TransferReason.PATH_INCOMPATIBLE in decision.reason_codes
    assert TransferReason.HELD_OUT_MISSING in decision.reason_codes
    assert TransferReason.SIMILARITY_INSUFFICIENT in decision.reason_codes
    assert decision.unsafe_transfer_count == 0


def test_certificate_or_policy_binding_mismatch_is_refused() -> None:
    spec = _spec()
    mismatched = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            certificate=_certificate(spec, procedure_cid="other-procedure"),
        )
    )
    assert mismatched.reason_code is TransferReason.BINDING_MISMATCH
    assert mismatched.eligible is False

    weaker_validation = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            target=_target(spec, validation_structure=("focused-tests",)),
        )
    )
    assert weaker_validation.reason_code is TransferReason.VALIDATION_INCOMPATIBLE

    higher_risk = ProcedureTransferGate().evaluate(
        _request(
            procedure=spec,
            certificate=_certificate(spec, risk_ceiling=RiskClass.REPOSITORY_WRITE),
        )
    )
    assert higher_risk.reason_code is TransferReason.RISK_CEILING
    assert TransferDimension.AUTHORITY.value in higher_risk.changed_assumptions
