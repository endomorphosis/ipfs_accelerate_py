from __future__ import annotations

import copy

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    EpisodeKind,
    ExecutionTrajectory,
    FamilyMembershipClass,
    HoleType,
    IdempotencyClass,
    ProcedureAuthorityEnvelope,
    ProcedureEffect,
    ProcedureInvariant,
    ProcedureLocal,
    ProcedureObservation,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    RiskClass,
    StepOperation,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyMembership,
    TraceEventStatus,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
    ValueType,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.invariant_mining import (
    InvariantMiner,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.specification_mining import (
    REQUIRED_PROPERTY_KINDS,
    SOURCE_TIER_CEILING,
    AdmittedSource,
    CandidateStatus,
    EvidenceTier,
    PropertyKind,
    PropertyNomination,
    SourceKind,
    SpecificationCandidate,
    SpecificationMiner,
    SpecificationMiningError,
    project_procedure_spec,
)


def bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo-main",
        repository_commit="abc123",
        tree_id="tree-abc123",
        objective_id="PCPC-G000",
        task_id="PCPC-013",
        contract_revision="procedure-contracts-v1",
        policy_revision="authority-policy-v1",
        environment_id="python312-linux-lock1",
    )


def procedure_spec() -> ProcedureSpec:
    precondition = ProcedurePrecondition(
        condition_id="precondition.current-tree",
        binding="binding:tree_id",
        operator=ConditionOperator.CURRENT,
        evidence_producer="tree-verifier@1",
        evidence_type="current-tree-receipt@1",
    )
    invariant = ProcedureInvariant(
        condition_id="invariant.scope",
        binding="procedure.scope_paths",
        operator=ConditionOperator.SUBSET_OF,
        operand=("ipfs_accelerate_py/agent_supervisor",),
        evidence_producer="scope-checker@1",
        evidence_type="scope-receipt@1",
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
    write_effect = ProcedureEffect(
        effect_id="effect.write",
        effect_class=EffectClass.REPOSITORY_WRITE,
        targets=("ipfs_accelerate_py/agent_supervisor/example.py",),
        reversible=True,
    )
    validation_effect = ProcedureEffect(
        effect_id="effect.validation", effect_class=EffectClass.VALIDATION
    )
    receipt_effect = ProcedureEffect(
        effect_id="effect.receipt", effect_class=EffectClass.RECEIPT_EMIT
    )
    rollback_effect = ProcedureEffect(
        effect_id="effect.rollback", effect_class=EffectClass.ROLLBACK
    )
    read = ProcedureStep(
        step_id="read",
        operation=StepOperation.READ_STATE,
        operation_contract="state-reader@1",
        output_bindings={"state": "local:state"},
        next_step_id="patch",
        idempotency=IdempotencyClass.PURE,
    )
    patch = ProcedureStep(
        step_id="patch",
        operation=StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,
        operation_contract="approved-patch-template@1",
        input_bindings={"state": "local:state"},
        declared_effect_ids=("effect.write",),
        required_authority_ids=("authority.execute",),
        next_step_id="tests",
        idempotency=IdempotencyClass.IDEMPOTENCY_KEY_REQUIRED,
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
    rollback = ProcedureRollback(
        rollback_id="rollback.restore-tree",
        trigger_effect_ids=("effect.write",),
        step_ids=("patch",),
        verification_observation_ids=("observation.tests",),
        exact_target_cid="tree-abc123",
    )
    authority = ProcedureAuthorityEnvelope(
        authority_policy_revision="authority-policy-v1",
        requirement_ids=("authority.execute",),
        required_capability_ids=("capability.tests",),
        allowed_operations=tuple(step.operation for step in (read, patch, tests, check, receipt)),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
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
        required_proof_contracts=("scope-proof@1",),
    )
    return ProcedureSpec(
        bindings=bindings(),
        name="focused-validation-procedure",
        version=ProcedureVersion(major=1),
        task_family_id="IMPORT_PURITY_REPAIR",
        entry_step_id="read",
        locals=(
            ProcedureLocal("state", ValueType.STRUCTURED),
            ProcedureLocal("test-result", ValueType.STRUCTURED),
        ),
        preconditions=(precondition,),
        declared_reads=("ipfs_accelerate_py/agent_supervisor/example.py",),
        declared_effects=(write_effect, validation_effect, receipt_effect, rollback_effect),
        steps=(read, patch, tests, check, receipt),
        invariants=(invariant,),
        postconditions=(postcondition,),
        observations=(test_observation, post_observation),
        validation=validation,
        rollback=(rollback,),
        authority=authority,
        resources=resources,
        terminal_step_ids=("receipt",),
        scope_paths=("ipfs_accelerate_py/agent_supervisor",),
        provenance_cids=("accepted-trajectory-cid",),
    )


def trajectory(
    *,
    source_episode_cid: str = "accepted-receipt",
    source_episode_kind: EpisodeKind = EpisodeKind.ACCEPTED_TASK_RECEIPT,
    status: TrajectoryTerminalStatus = TrajectoryTerminalStatus.ACCEPTED,
    accepted_criterion_ids: tuple[str, ...] = ("criterion-a",),
    rejection_reason_code: str = "",
    terminal_state_cid: str = "state-2",
) -> ExecutionTrajectory:
    steps = (
        TrajectoryStep(
            sequence=0,
            operation=StepOperation.REQUEST_TYPED_MODEL_HOLE,
            operation_contract="typed-hole-service@1",
            initial_state_cid="state-0",
            terminal_state_cid="state-1",
            observation_cids=("hole-observation",),
            effect_ids=("model-request",),
            validation_receipt_cids=("hole-validation",),
            hole_type=HoleType.CLASSIFY_FAILURE.value,
            model_calls=1,
            input_tokens=10,
            output_tokens=2,
            latency_ms=20,
            status=TraceEventStatus.SUCCEEDED,
        ),
        TrajectoryStep(
            sequence=1,
            operation=StepOperation.RUN_SELECTED_TESTS,
            operation_contract="test-runner@1",
            initial_state_cid="state-1",
            terminal_state_cid=terminal_state_cid,
            observation_cids=("test-observation",),
            effect_ids=("validation",),
            validation_receipt_cids=("test-receipt",),
            latency_ms=30,
            status=TraceEventStatus.SUCCEEDED,
        ),
    )
    return ExecutionTrajectory(
        bindings=bindings(),
        source_episode_cid=source_episode_cid,
        source_episode_kind=source_episode_kind,
        initial_abstract_state_cid="state-0",
        terminal_abstract_state_cid=terminal_state_cid,
        objective_criterion_ids=("criterion-a", "criterion-b"),
        task_family_hint="IMPORT_PURITY_REPAIR",
        steps=steps,
        outcome=TrajectoryOutcome(
            status=status,
            accepted_criterion_ids=accepted_criterion_ids,
            validation_receipt_cids=("hole-validation", "test-receipt"),
            proof_receipt_cids=(),
            rejection_reason_code=rejection_reason_code,
        ),
        total_cost_units=3,
        total_tokens=12,
        total_latency_ms=55,
        human_interventions=0,
    )


def task_family() -> TaskFamily:
    boundary = TaskFamilyBoundary(
        positive_member_cids=("positive-a",),
        negative_example_cids=("negative-a",),
        boundary_example_cids=("boundary-a",),
        unknown_case_cids=("unknown-a",),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
        permitted_repositories=("repo-main",),
        permitted_languages=("python",),
        permitted_frameworks=("pytest",),
        permitted_effect_classes=(EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    )
    return TaskFamily(
        bindings=bindings(),
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


def nomination(
    *,
    property_kind: PropertyKind = PropertyKind.PRECONDITION,
    property_id: str = "precondition.current-tree",
    binding: str = "binding:tree_id",
    operator: ConditionOperator = ConditionOperator.CURRENT,
    operand: object = "tree-abc123",
    evidence_cid: str = "evidence-a",
) -> PropertyNomination:
    return PropertyNomination(
        property_kind=property_kind,
        property_id=property_id,
        binding=binding,
        operator=operator,
        operand=operand,
        evidence_cid=evidence_cid,
    )


def source(
    *,
    source_id: str,
    source_kind: SourceKind,
    nominations: tuple[PropertyNomination, ...] = (),
    occurrence_count: int = 1,
    passing: bool = False,
    admitted: bool = True,
    provenance_cid: str | None = None,
    artifact_cid: str | None = None,
    evidence_tier: EvidenceTier | None = None,
) -> AdmittedSource:
    return AdmittedSource(
        bindings=bindings(),
        source_id=source_id,
        source_kind=source_kind,
        evidence_tier=evidence_tier or SOURCE_TIER_CEILING[source_kind],
        provenance_cid=provenance_cid or f"prov-{source_id}",
        artifact_cid=artifact_cid or f"art-{source_id}",
        nominations=nominations or (nomination(evidence_cid=f"ev-{source_id}"),),
        admitted=admitted,
        passing=passing,
        occurrence_count=occurrence_count,
    )


def kinds_of(result) -> set[str]:
    return {item.property_kind.value for item in result.candidates}


def test_procedure_projection_proposes_all_required_property_kinds() -> None:
    spec = procedure_spec()
    result = SpecificationMiner().mine((spec,))
    assert kinds_of(result) >= set(REQUIRED_PROPERTY_KINDS)
    by_kind = {kind: [] for kind in PropertyKind}
    for candidate in result.candidates:
        by_kind[candidate.property_kind].append(candidate)
        assert candidate.status is CandidateStatus.CANDIDATE
        assert candidate.evidence_cids
        assert candidate.source_provenances
        artifact = candidate.to_artifact(result.bindings)
        assert artifact.state is ArtifactState.CANDIDATE
        assert artifact.reference_cids == candidate.evidence_cids
        decoded = parse_procedure_artifact(artifact.to_dict())
        assert decoded == artifact
        assert SpecificationCandidate.from_artifact(decoded).property_id == candidate.property_id
    assert by_kind[PropertyKind.PRECONDITION]
    assert by_kind[PropertyKind.POSTCONDITION]
    assert by_kind[PropertyKind.INVARIANT]
    assert by_kind[PropertyKind.FRAME]
    assert by_kind[PropertyKind.EFFECT]
    assert by_kind[PropertyKind.RESOURCE]
    assert by_kind[PropertyKind.ORDER]
    assert by_kind[PropertyKind.IDEMPOTENCY]
    assert by_kind[PropertyKind.ROLLBACK]
    assert by_kind[PropertyKind.FRESHNESS]
    assert result.receipt.state is ArtifactState.CANDIDATE
    assert result.receipt.facts["upgraded_count"] == 0
    assert result.receipt.facts["verified_count"] == 0
    assert result.upgraded_count == 0


def test_every_source_kind_retains_provenance_and_tier() -> None:
    spec = procedure_spec()
    family = task_family()
    admitted = trajectory()
    extra = (
        source(
            source_id="docs",
            source_kind=SourceKind.AUTHORITATIVE_DOCUMENTATION,
            nominations=(
                nomination(
                    property_kind=PropertyKind.PRECONDITION,
                    property_id="precondition.current-tree",
                    binding="binding:tree_id",
                    operator=ConditionOperator.CURRENT,
                    operand=None,
                    evidence_cid="docs-pre",
                ),
            ),
        ),
        source(
            source_id="mutant-agreeing",
            source_kind=SourceKind.MUTANT,
            nominations=(
                nomination(
                    property_kind=PropertyKind.INVARIANT,
                    property_id="invariant.scope",
                    binding="procedure.scope_paths",
                    operator=ConditionOperator.SUBSET_OF,
                    operand=("ipfs_accelerate_py/agent_supervisor",),
                    evidence_cid="mutant-scope",
                ),
            ),
        ),
        source(
            source_id="rejected-unique",
            source_kind=SourceKind.REJECTED_TRACE,
            nominations=(
                nomination(
                    property_kind=PropertyKind.ROLLBACK,
                    property_id="rollback.rejected-path",
                    binding="trajectory.outcome",
                    operator=ConditionOperator.EXISTS,
                    operand="rejected-receipt",
                    evidence_cid="rejected-unique-ev",
                ),
            ),
        ),
    )
    result = SpecificationMiner().mine((spec, family, admitted, *extra))
    retained = set(result.retained_source_kinds)
    assert retained == set(SourceKind)
    by_kind = {kind: [] for kind in SourceKind}
    for candidate in result.candidates:
        for provenance in candidate.source_provenances:
            by_kind[provenance.source_kind].append(provenance)
            assert provenance.evidence_tier is SOURCE_TIER_CEILING[provenance.source_kind] or (
                provenance.evidence_tier is candidate.evidence_tier
            )
            assert provenance.provenance_cid
            assert provenance.artifact_cid
            assert provenance.evidence_tier is not None
    for kind in SourceKind:
        assert by_kind[kind], f"missing retained provenance for {kind.value}"
    pre = next(
        item
        for item in result.candidates
        if item.property_id == "precondition.current-tree"
    )
    pre_kinds = {item.source_kind for item in pre.source_provenances}
    assert SourceKind.TYPE in pre_kinds
    assert SourceKind.AUTHORITATIVE_DOCUMENTATION in pre_kinds
    doc_tier = next(
        item.evidence_tier
        for item in pre.source_provenances
        if item.source_kind is SourceKind.AUTHORITATIVE_DOCUMENTATION
    )
    assert doc_tier is EvidenceTier.DOCUMENTATION_NOMINATION
    assert pre.status is CandidateStatus.CANDIDATE


def test_frequency_absence_and_passing_tests_do_not_upgrade_candidate_status() -> None:
    frequent = source(
        source_id="freq",
        source_kind=SourceKind.TEST,
        passing=True,
        occurrence_count=500,
        nominations=(
            nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.tests-admitted",
                binding="local:test-result",
                operator=ConditionOperator.ADMITTED,
                operand=None,
                evidence_cid="test-pass",
            ),
        ),
    )
    typed = source(
        source_id="type",
        source_kind=SourceKind.TYPE,
        nominations=(
            nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.tests-admitted",
                binding="local:test-result",
                operator=ConditionOperator.ADMITTED,
                operand=None,
                evidence_cid="type-post",
            ),
        ),
    )
    result = SpecificationMiner().mine((typed, frequent))
    candidate = result.candidates[0]
    assert candidate.status is CandidateStatus.CANDIDATE
    assert candidate.supporting_occurrences == 501
    assert candidate.passing_test_count == 1
    assert candidate.evidence_tier is EvidenceTier.TEST_OBSERVATION
    assert ArtifactState.VERIFIED.value not in {
        item.state.value for item in result.candidate_artifacts
    }
    assert result.receipt.facts["upgraded_count"] == 0

    absent_proof = SpecificationMiner().mine((typed,))
    assert SourceKind.PROOF_OBLIGATION not in absent_proof.retained_source_kinds
    assert absent_proof.candidates[0].status is CandidateStatus.CANDIDATE
    assert absent_proof.receipt.facts["verified_count"] == 0


def test_conflicting_evidence_yields_counterexample_and_refusal() -> None:
    left = source(
        source_id="admitted-trace",
        source_kind=SourceKind.ADMITTED_TRACE,
        nominations=(
            nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.accepted-criteria",
                binding="trajectory.outcome",
                operator=ConditionOperator.EQUALS,
                operand=("criterion-a",),
                evidence_cid="accepted-ev",
            ),
        ),
    )
    right = source(
        source_id="rejected-trace",
        source_kind=SourceKind.REJECTED_TRACE,
        nominations=(
            nomination(
                property_kind=PropertyKind.POSTCONDITION,
                property_id="postcondition.accepted-criteria",
                binding="trajectory.outcome",
                operator=ConditionOperator.NOT_EXISTS,
                operand=("criterion-a", "criterion-b"),
                evidence_cid="rejected-ev",
            ),
        ),
    )
    result = SpecificationMiner().mine((left, right))
    assert result.candidates == ()
    assert len(result.refused) == 1
    assert result.refused[0].status is CandidateStatus.REFUSED
    assert result.refused[0].to_artifact(result.bindings).state is ArtifactState.REJECTED
    assert len(result.counterexamples) == 1
    counter = result.counterexamples[0]
    assert counter.conflict_class == "disagreeing-claim"
    assert counter.left_evidence_cid == "accepted-ev"
    assert counter.right_evidence_cid == "rejected-ev"
    artifact = result.counterexample_artifacts[0]
    assert artifact.state is ArtifactState.REJECTED
    assert parse_procedure_artifact(artifact.to_dict()) == artifact
    assert "accepted-ev" in artifact.reference_cids
    assert "rejected-ev" in artifact.reference_cids


def test_admitted_and_rejected_trajectories_conflict_on_acceptance() -> None:
    accepted = trajectory()
    rejected = trajectory(
        source_episode_cid="rejected-receipt",
        source_episode_kind=EpisodeKind.REJECTED_TASK_RECORD,
        status=TrajectoryTerminalStatus.REJECTED,
        accepted_criterion_ids=(),
        rejection_reason_code="boundary-split",
    )
    result = SpecificationMiner().mine((accepted, rejected))
    refused_posts = [
        item for item in result.refused if item.property_id == "postcondition.accepted-criteria"
    ]
    assert refused_posts
    assert any(item.property_id == "postcondition.accepted-criteria" for item in result.counterexamples)


def test_unadmitted_sources_and_binding_drift_are_refused() -> None:
    miner = SpecificationMiner()
    with pytest.raises(SpecificationMiningError, match="unadmitted"):
        miner.mine((source(source_id="ghost", source_kind=SourceKind.TYPE, admitted=False),))

    drifted = AdmittedSource(
        bindings=ArtifactBindings(
            repository_id="repo-main",
            repository_commit="other",
            tree_id="tree-other",
            objective_id="PCPC-G000",
            task_id="PCPC-013",
            contract_revision="procedure-contracts-v1",
            policy_revision="authority-policy-v1",
            environment_id="python312-linux-lock1",
        ),
        source_id="drift",
        source_kind=SourceKind.TYPE,
        evidence_tier=EvidenceTier.TYPE_DECLARATION,
        provenance_cid="prov-drift",
        artifact_cid="art-drift",
        nominations=(nomination(),),
    )
    with pytest.raises(SpecificationMiningError, match="bindings"):
        miner.mine((source(source_id="ok", source_kind=SourceKind.TYPE), drifted))

    with pytest.raises(SpecificationMiningError, match="ceiling"):
        source(
            source_id="overclaim",
            source_kind=SourceKind.TEST,
            evidence_tier=EvidenceTier.PROOF_OBLIGATION,
        )


def test_unsafe_and_unbounded_nominations_are_rejected() -> None:
    with pytest.raises(SpecificationMiningError, match="floating"):
        nomination(operand=0.5)
    with pytest.raises(SpecificationMiningError, match="unsafe"):
        nomination(operand="/etc/passwd")
    with pytest.raises(SpecificationMiningError, match="item bound"):
        AdmittedSource(
            bindings=bindings(),
            source_id="too-many",
            source_kind=SourceKind.TYPE,
            evidence_tier=EvidenceTier.TYPE_DECLARATION,
            provenance_cid="prov-too-many",
            artifact_cid="art-too-many",
            nominations=tuple(
                nomination(property_id=f"precondition.n{index}", evidence_cid=f"ev-{index}")
                for index in range(129)
            ),
        )


def test_task_family_and_membership_remain_external_authorities() -> None:
    family = task_family()
    membership = TaskFamilyMembership(
        bindings=family.bindings,
        task_family_cid=family.content_id,
        trajectory_cid="positive-a",
        membership=FamilyMembershipClass.POSITIVE,
        evidence_cids=("classifier-receipt",),
        classifier_revision="baseline-v1",
    )
    result = SpecificationMiner().mine((family,))
    assert any(item.property_kind is PropertyKind.ROLLBACK for item in result.candidates)
    assert any(
        item.property_id == "invariant.not.import-side-effect" for item in result.candidates
    )
    assert membership.membership is FamilyMembershipClass.POSITIVE
    failure = next(
        item
        for item in result.candidates
        if item.property_id == "invariant.not.import-side-effect"
    )
    assert failure.source_kinds == (SourceKind.FAILURE_SIGNATURE,)
    assert failure.evidence_tier is EvidenceTier.FAILURE_SIGNATURE


def test_invariant_miner_emits_candidate_invariants_with_exact_evidence() -> None:
    spec = procedure_spec()
    family = task_family()
    result = InvariantMiner().mine((spec, family, trajectory()))
    assert result.candidates
    assert all(item.property_kind is PropertyKind.INVARIANT for item in result.candidates)
    assert all(item.status is CandidateStatus.CANDIDATE for item in result.candidates)
    assert {item.property_id for item in result.candidates} >= {
        "invariant.scope",
        "invariant.scope-respected",
        "invariant.tree-current",
        "invariant.not.import-side-effect",
        "invariant.effect-ceiling",
        "invariant.validation-coverage",
        "invariant.contiguous-state-chain",
    }
    for artifact in result.invariant_artifacts:
        assert artifact.state is ArtifactState.CANDIDATE
        assert artifact.reference_cids
        assert artifact.facts["candidate_status"] == "candidate"
        assert parse_procedure_artifact(artifact.to_dict()) == artifact
    assert result.upgraded_count == 0


def test_invariant_conflict_is_refused_without_status_upgrade() -> None:
    left = source(
        source_id="inv-left",
        source_kind=SourceKind.TYPE,
        nominations=(
            nomination(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.scope",
                binding="procedure.scope_paths",
                operator=ConditionOperator.SUBSET_OF,
                operand=("ipfs_accelerate_py/agent_supervisor",),
                evidence_cid="inv-left-ev",
            ),
        ),
    )
    right = source(
        source_id="inv-right",
        source_kind=SourceKind.MUTANT,
        nominations=(
            nomination(
                property_kind=PropertyKind.INVARIANT,
                property_id="invariant.scope",
                binding="procedure.scope_paths",
                operator=ConditionOperator.NOT_EXISTS,
                operand=("ipfs_accelerate_py/agent_supervisor",),
                evidence_cid="inv-right-ev",
            ),
        ),
    )
    result = InvariantMiner().mine((left, right))
    assert result.candidates == ()
    assert result.refused[0].status is CandidateStatus.REFUSED
    assert result.counterexamples[0].left_evidence_cid == "inv-left-ev"


def test_projected_sources_are_deterministic_and_bounded() -> None:
    spec = procedure_spec()
    first = project_procedure_spec(spec)
    second = project_procedure_spec(spec)
    assert first == second
    kinds = {item.source_kind for item in first}
    assert SourceKind.TYPE in kinds
    assert SourceKind.OPERATION_CONTRACT in kinds
    assert SourceKind.TEST in kinds
    assert SourceKind.PROOF_OBLIGATION in kinds
    assert SourceKind.RUNTIME_CHECK in kinds
    miner = SpecificationMiner(emitted_at_ms=7)
    left = miner.mine((spec,))
    right = miner.mine((spec,))
    assert left.receipt.content_id == right.receipt.content_id
    assert left.candidate_artifacts[0].created_at_ms == 7


def test_wire_receipt_binds_candidate_and_evidence_identities() -> None:
    result = SpecificationMiner().mine((procedure_spec(),))
    receipt = parse_procedure_artifact(result.receipt.to_dict())
    assert receipt.facts["candidate_cids"] == tuple(
        item.content_id for item in result.candidate_artifacts
    )
    assert receipt.facts["evidence_cids"] == tuple(
        item.content_id for item in result.evidence_artifacts
    )
    assert result.receipt.facts["upgraded_count"] == 0
    assert result.receipt.facts["verified_count"] == 0
    payload = copy.deepcopy(receipt.to_dict())
    payload.pop("content_id", None)
    payload.pop("cid", None)
    payload["facts"]["upgraded_count"] = 1
    forged = parse_procedure_artifact(payload)
    assert forged.facts["upgraded_count"] == 1
    assert forged.content_id != receipt.content_id
