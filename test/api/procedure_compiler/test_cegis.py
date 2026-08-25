from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.cegis import (
    CEGIS_REVISION,
    GENERATION_PRIORITY,
    CegisError,
    CounterexampleKind,
    CounterexampleSet,
    ModelSketch,
    NarrowingConstraints,
    ProcedureCegis,
    ProcedureSynthesisPlan,
    SkipReason,
    SynthesisCounterexample,
    SynthesisRequest,
    SynthesisSourceKind,
    SynthesisStatus,
    SynthesisStopReason,
    ValidationFinding,
    VerifiedProcedureSeed,
    enumerate_procedure_variants,
    insert_closed_validation_step,
    replay_hits,
    structural_overflow,
    synthesize_procedure,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    HoleType,
    ProcedureAuthorityEnvelope,
    ProcedureBranch,
    ProcedureEffect,
    ProcedureHole,
    ProcedureLocal,
    ProcedureLoop,
    ProcedureObservation,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    ProviderClass,
    RiskClass,
    StepOperation,
    ValueType,
    parse_procedure_artifact,
)


def bindings(**changes: str) -> ArtifactBindings:
    values = {
        "repository_id": "repo-main",
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G000",
        "task_id": "PCPC-015",
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


def spec_without_selected_tests(*, name: str = "postcondition-only-procedure") -> ProcedureSpec:
    spec = valid_spec(name=name)
    read, _tests, check, receipt = spec.steps
    read = replace(read, next_step_id="postcondition")
    check = replace(check, input_bindings={"state": "local:state"})
    steps = (read, check, receipt)
    post_observation = replace(
        spec.observations[1],
        output_binding="local:state",
    )
    return replace(
        spec,
        steps=steps,
        observations=(post_observation,),
        validation=replace(
            spec.validation,
            required_step_ids=("postcondition",),
            required_observation_ids=("observation.postcondition",),
            required_test_contracts=(),
        ),
        authority=replace(
            spec.authority,
            allowed_operations=tuple(step.operation for step in steps),
        ),
        locals=(ProcedureLocal("state", ValueType.STRUCTURED),),
        postconditions=(
            replace(
                spec.postconditions[0],
                binding="binding:tree_id",
                operator=ConditionOperator.CURRENT,
                evidence_producer="tree-verifier@1",
                evidence_type="current-tree-receipt@1",
            ),
        ),
    )


def plan(**changes: object) -> ProcedureSynthesisPlan:
    values: dict[str, object] = {
        "bindings": bindings(),
        "task_family_id": "IMPORT_PURITY_REPAIR",
        "max_candidates": 16,
        "max_steps": 16,
        "max_branches": 8,
        "max_holes": 8,
        "max_loops": 4,
        "max_model_calls": 4,
        "max_tokens": 4_096,
        "max_validation": 16,
        "max_proof": 8,
        "max_wall_time_ms": 60_000,
    }
    values.update(changes)
    return ProcedureSynthesisPlan(**values)


def named(spec: ProcedureSpec, name: str) -> ProcedureSpec:
    return replace(spec, name=name)


def reject_sources(
    *kinds: SynthesisSourceKind,
    obligation: str = "seeded-adversarial",
    constraints: NarrowingConstraints | None = None,
) -> object:
    blocked = frozenset(kinds)

    def _validate(candidate: object, _counterexamples: object) -> ValidationFinding | None:
        if candidate.source_kind in blocked:  # type: ignore[attr-defined]
            return ValidationFinding(
                kind=CounterexampleKind.ADVERSARIAL,
                obligation=obligation,
                witness={"source_kind": candidate.source_kind.value},  # type: ignore[attr-defined]
                constraints=constraints or NarrowingConstraints(),
            )
        return None

    return _validate


def require_operation(operation: StepOperation, *, kind: CounterexampleKind) -> object:
    def _validate(candidate: object, _counterexamples: object) -> ValidationFinding | None:
        present = {step.operation for step in candidate.procedure.steps}  # type: ignore[attr-defined]
        if operation in present:
            return None
        return ValidationFinding(
            kind=kind,
            obligation=f"missing-{operation.value.lower().replace('_', '-')}",
            witness={"required_operations": (operation.value,)},
            constraints=NarrowingConstraints(required_operations=(operation,)),
        )

    return _validate


def extra_hole(spec: ProcedureSpec, hole_id: str) -> ProcedureHole:
    return ProcedureHole(
        hole_id=hole_id,
        hole_type=HoleType.CLASSIFY_FAILURE,
        input_schema_ref="schema.failure-in",
        output_schema_ref="schema.failure-out",
        allowed_provider_classes=(ProviderClass.DECLARATIVE_RULE,),
        context_budget_bytes=1024,
        authority_requirement_ids=("authority.execute",),
        effect_classes=(EffectClass.OBSERVE,),
        validation_observation_ids=("observation.tests",),
        fallback_step_id="receipt",
        maximum_attempts=1,
    )


class ManualClock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def test_generation_follows_required_priority_order() -> None:
    base = valid_spec()
    sources = {
        SynthesisSourceKind.EXISTING_VERIFIED: named(base, "verified-procedure"),
        SynthesisSourceKind.BUILTIN_TEMPLATE: named(base, "template-procedure"),
        SynthesisSourceKind.ANTI_UNIFIED_PATTERN: named(base, "pattern-procedure"),
        SynthesisSourceKind.ENUMERATIVE: named(base, "enumerative-seed"),
        SynthesisSourceKind.MODEL_SKETCH: named(base, "model-procedure"),
        SynthesisSourceKind.HUMAN: named(base, "human-procedure"),
    }
    request = SynthesisRequest(
        plan=plan(),
        verified_procedures=(
            VerifiedProcedureSeed(
                procedure=sources[SynthesisSourceKind.EXISTING_VERIFIED],
                certificate_cid="certificate.verified",
            ),
        ),
        templates=(sources[SynthesisSourceKind.BUILTIN_TEMPLATE],),
        anti_unified=(sources[SynthesisSourceKind.ANTI_UNIFIED_PATTERN],),
        enumerative_seeds=(sources[SynthesisSourceKind.ENUMERATIVE],),
        model_sketches=(
            ModelSketch(
                procedure=sources[SynthesisSourceKind.MODEL_SKETCH],
                token_cost=8,
                model_calls=1,
            ),
        ),
        human_candidates=(sources[SynthesisSourceKind.HUMAN],),
    )
    generated = ProcedureCegis().generate_candidates(request)
    ordered_kinds: list[SynthesisSourceKind] = []
    for item in generated:
        if item.source_kind not in ordered_kinds:
            ordered_kinds.append(item.source_kind)
    assert tuple(ordered_kinds) == GENERATION_PRIORITY
    assert generated[0].procedure.name == "verified-procedure"
    result = ProcedureCegis().synthesize(request)
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].source_kind is SynthesisSourceKind.EXISTING_VERIFIED
    assert result.surviving_candidates[0].procedure.state is ArtifactState.CANDIDATE
    assert result.candidate_artifacts[0].state is ArtifactState.CANDIDATE


def test_lower_priority_sources_are_reached_only_after_higher_priority_failures() -> None:
    base = valid_spec()
    request = SynthesisRequest(
        plan=plan(),
        verified_procedures=(
            VerifiedProcedureSeed(
                procedure=named(base, "verified-procedure"),
                certificate_cid="certificate.verified",
            ),
        ),
        templates=(named(base, "template-procedure"),),
        anti_unified=(named(base, "pattern-procedure"),),
        model_sketches=(
            ModelSketch(procedure=named(base, "model-procedure"), token_cost=4, model_calls=1),
        ),
        human_candidates=(named(base, "human-procedure"),),
        validator=reject_sources(
            SynthesisSourceKind.EXISTING_VERIFIED,
            SynthesisSourceKind.BUILTIN_TEMPLATE,
            SynthesisSourceKind.ANTI_UNIFIED_PATTERN,
            SynthesisSourceKind.MODEL_SKETCH,
        ),
    )
    result = ProcedureCegis().synthesize(request)
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].source_kind is SynthesisSourceKind.HUMAN
    assert tuple(result.considered_source_kinds) == (
        SynthesisSourceKind.EXISTING_VERIFIED,
        SynthesisSourceKind.BUILTIN_TEMPLATE,
        SynthesisSourceKind.ANTI_UNIFIED_PATTERN,
        SynthesisSourceKind.MODEL_SKETCH,
        SynthesisSourceKind.HUMAN,
    )
    assert [item.source_kind for item in result.rejected_candidates] == [
        SynthesisSourceKind.EXISTING_VERIFIED,
        SynthesisSourceKind.BUILTIN_TEMPLATE,
        SynthesisSourceKind.ANTI_UNIFIED_PATTERN,
        SynthesisSourceKind.MODEL_SKETCH,
    ]


def test_replay_counterexamples_narrow_remaining_candidates() -> None:
    weak = spec_without_selected_tests(name="weak-procedure")
    strong = valid_spec(name="strong-procedure")
    prior = SynthesisCounterexample(
        kind=CounterexampleKind.REPLAY,
        obligation="missing-run-selected-tests",
        candidate_id="historical-candidate",
        counterexample_set_cid=CounterexampleSet().content_id,
        witness={"required_operations": (StepOperation.RUN_SELECTED_TESTS.value,)},
        constraints=NarrowingConstraints(
            required_operations=(StepOperation.RUN_SELECTED_TESTS,)
        ),
    )
    assert replay_hits(weak, prior)
    assert not replay_hits(strong, prior)
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            verified_procedures=(
                VerifiedProcedureSeed(procedure=weak, certificate_cid="certificate.weak"),
            ),
            templates=(strong,),
            initial_counterexamples=(prior,),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.name == "strong-procedure"
    assert result.skipped_pairs[0].reason is SkipReason.NARROWED
    assert result.skipped_pairs[0].candidate_id == weak.content_id
    assert result.rejected_candidates[0].procedure.name == "weak-procedure"


def test_adversarial_counterexamples_refine_enumerative_candidates() -> None:
    seed = valid_spec(name="enumerative-seed")
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            enumerative_seeds=(seed,),
            validator=require_operation(
                StepOperation.RUN_STATIC_ANALYSIS, kind=CounterexampleKind.ADVERSARIAL
            ),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    survivor = result.surviving_candidates[0]
    assert survivor.source_kind is SynthesisSourceKind.ENUMERATIVE
    assert StepOperation.RUN_STATIC_ANALYSIS in {step.operation for step in survivor.procedure.steps}
    assert result.counterexamples[0].kind is CounterexampleKind.ADVERSARIAL
    assert result.counterexamples[0].constraints.required_operations == (
        StepOperation.RUN_STATIC_ANALYSIS,
    )
    assert any(item.reason is SkipReason.NARROWED for item in result.skipped_pairs) or (
        result.rejected_candidates[0].procedure.name.startswith("enumerative.identity")
    )


def test_repeated_candidate_set_pairs_are_skipped() -> None:
    spec = valid_spec(name="duplicate-procedure")
    prior = SynthesisCounterexample(
        kind=CounterexampleKind.REPLAY,
        obligation="missing-run-proof",
        candidate_id="historical-candidate",
        counterexample_set_cid=CounterexampleSet().content_id,
        constraints=NarrowingConstraints(required_operations=(StepOperation.RUN_PROOF,)),
    )
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(spec, spec),
            initial_counterexamples=(prior,),
        )
    )
    assert result.status is SynthesisStatus.INCOMPLETE
    assert result.stop_reason is SynthesisStopReason.NO_ADMISSIBLE_CANDIDATE
    reasons = tuple(item.reason for item in result.skipped_pairs)
    assert SkipReason.NARROWED in reasons
    assert SkipReason.REPEATED_PAIR in reasons
    assert result.skipped_pairs[0].pair_key == result.skipped_pairs[1].pair_key
    assert result.usage.candidates_tried == 0
    assert result.usage.candidates_skipped >= 2


def test_current_constraint_set_refutes_without_re_evaluation() -> None:
    spec = valid_spec(name="already-refuted")
    prior = SynthesisCounterexample(
        kind=CounterexampleKind.REPLAY,
        obligation="missing-run-proof",
        candidate_id="historical-candidate",
        counterexample_set_cid=CounterexampleSet().content_id,
        constraints=NarrowingConstraints(required_operations=(StepOperation.RUN_PROOF,)),
    )
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(spec,),
            initial_counterexamples=(prior,),
        )
    )
    assert result.skipped_pairs[0].reason is SkipReason.NARROWED
    assert result.usage.unique_pairs_evaluated == 0
    assert result.usage.candidates_tried == 0
    assert result.stop_reason is SynthesisStopReason.NO_ADMISSIBLE_CANDIDATE


@pytest.mark.parametrize(
    ("bound_changes", "request_builder", "reason"),
    [
        (
            {"max_candidates": 1},
            lambda spec: {
                "templates": (named(spec, "first"), named(spec, "second")),
                "validator": reject_sources(SynthesisSourceKind.BUILTIN_TEMPLATE),
            },
            SynthesisStopReason.CANDIDATE_BUDGET_EXHAUSTED,
        ),
        (
            {"max_validation": 1},
            lambda spec: {
                "templates": (named(spec, "first"), named(spec, "second")),
                "validator": reject_sources(SynthesisSourceKind.BUILTIN_TEMPLATE),
            },
            SynthesisStopReason.VALIDATION_BUDGET_EXHAUSTED,
        ),
        (
            {"max_model_calls": 0, "max_tokens": 4_096},
            lambda spec: {
                "model_sketches": (
                    ModelSketch(procedure=named(spec, "model-a"), token_cost=1, model_calls=1),
                )
            },
            SynthesisStopReason.MODEL_CALL_BUDGET_EXHAUSTED,
        ),
        (
            {"max_tokens": 0, "max_model_calls": 4},
            lambda spec: {
                "model_sketches": (
                    ModelSketch(procedure=named(spec, "model-a"), token_cost=8, model_calls=1),
                )
            },
            SynthesisStopReason.TOKEN_BUDGET_EXHAUSTED,
        ),
        (
            {"max_proof": 0},
            lambda spec: {
                "templates": (
                    insert_closed_validation_step(
                        named(spec, "needs-proof"),
                        StepOperation.RUN_PROOF,
                        "proof-runner@1",
                        effect_id="effect.proof",
                        effect_class=EffectClass.PROOF,
                        step_id="run-proof",
                    ),
                )
            },
            SynthesisStopReason.PROOF_BUDGET_EXHAUSTED,
        ),
        (
            {"max_steps": 3},
            lambda spec: {"templates": (spec,)},
            SynthesisStopReason.STEP_BOUND_EXHAUSTED,
        ),
        (
            {"max_branches": 0},
            lambda spec: {
                "templates": (
                    replace(
                        named(spec, "branched"),
                        branches=(
                            ProcedureBranch(
                                "branch-a",
                                "observation.tests",
                                "tests",
                                "postcondition",
                            ),
                        ),
                    ),
                )
            },
            SynthesisStopReason.BRANCH_BOUND_EXHAUSTED,
        ),
        (
            {"max_holes": 0},
            lambda spec: {
                "templates": (
                    replace(
                        named(spec, "holed"),
                        holes=(extra_hole(spec, "hole.classify"),),
                    ),
                )
            },
            SynthesisStopReason.HOLE_BOUND_EXHAUSTED,
        ),
        (
            {"max_loops": 0},
            lambda spec: {
                "templates": (
                    replace(
                        named(spec, "looped"),
                        loops=(
                            ProcedureLoop(
                                "loop-a",
                                "observation.tests",
                                "tests",
                                "postcondition",
                                2,
                            ),
                        ),
                    ),
                )
            },
            SynthesisStopReason.LOOP_BOUND_EXHAUSTED,
        ),
    ],
)
def test_bound_exhaustion_is_typed_incomplete(
    bound_changes: dict[str, int],
    request_builder: object,
    reason: SynthesisStopReason,
) -> None:
    spec = valid_spec()
    payload = request_builder(spec)  # type: ignore[operator]
    result = ProcedureCegis().synthesize(
        SynthesisRequest(plan=plan(**bound_changes), **payload)
    )
    assert result.status is SynthesisStatus.INCOMPLETE
    assert result.incomplete is True
    assert result.stop_reason is reason
    assert result.surviving_candidates == ()
    assert result.completeness_claimed is False


def test_wall_budget_exhaustion_is_typed_incomplete() -> None:
    clock = ManualClock()
    spec = valid_spec()

    def _advance(candidate: object, _counterexamples: object) -> ValidationFinding:
        clock.value = 50
        return ValidationFinding(
            kind=CounterexampleKind.ADVERSARIAL,
            obligation="clock-advanced",
            witness={"elapsed": 50},
        )

    result = ProcedureCegis(clock_ms=clock).synthesize(
        SynthesisRequest(
            plan=plan(max_wall_time_ms=10),
            templates=(named(spec, "first"), named(spec, "second")),
            validator=_advance,
        )
    )
    assert result.status is SynthesisStatus.INCOMPLETE
    assert result.stop_reason is SynthesisStopReason.WALL_BUDGET_EXHAUSTED
    assert result.usage.wall_time_ms >= 10
    assert result.usage.candidates_tried == 1


def test_duplicate_counterexamples_do_not_change_set_identity() -> None:
    spec = valid_spec(name="refuted")
    finding = ValidationFinding(
        kind=CounterexampleKind.ADVERSARIAL,
        obligation="stable-obligation",
        constraints=NarrowingConstraints(
            required_operations=(StepOperation.RUN_STATIC_ANALYSIS,)
        ),
    )
    first = ProcedureCegis().synthesize(
        SynthesisRequest(plan=plan(), templates=(spec,), validator=lambda *_: finding)
    )
    repeated = first.counterexamples[0]
    combined = CounterexampleSet(first.counterexamples).add(repeated)
    assert combined.content_id == first.counterexample_set_cid
    assert len(combined.members) == 1


def test_plan_and_counterexample_artifacts_round_trip() -> None:
    spec = valid_spec()
    result = ProcedureCegis(emitted_at_ms=11).synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(spec,),
            validator=require_operation(
                StepOperation.RUN_STATIC_ANALYSIS, kind=CounterexampleKind.ADVERSARIAL
            ),
            enumerative_seeds=(named(spec, "enumerative-seed"),),
        )
    )
    plan_artifact = parse_procedure_artifact(result.plan_artifact.to_dict())
    restored_plan = ProcedureSynthesisPlan.from_artifact(plan_artifact)
    assert restored_plan.task_family_id == result.plan.task_family_id
    assert restored_plan.max_candidates == result.plan.max_candidates
    assert plan_artifact.facts["synthesizer_revision"] == CEGIS_REVISION
    assert plan_artifact.state is ArtifactState.CANDIDATE
    decoded_ce = parse_procedure_artifact(result.counterexample_artifacts[0].to_dict())
    restored_ce = SynthesisCounterexample.from_artifact(decoded_ce)
    assert restored_ce.counterexample_id == result.counterexamples[0].counterexample_id
    assert restored_ce.kind is CounterexampleKind.ADVERSARIAL


def test_surviving_candidates_remain_unpromoted_and_do_not_claim_completeness() -> None:
    result = synthesize_procedure(
        SynthesisRequest(plan=plan(), templates=(valid_spec(),))
    )
    assert result.converged is True
    assert result.stop_reason is SynthesisStopReason.CONVERGED
    survivor = result.surviving_candidates[0]
    assert survivor.procedure.state is ArtifactState.CANDIDATE
    assert result.candidate_artifacts[0].state is ArtifactState.CANDIDATE
    assert result.plan_artifact.state is ArtifactState.CANDIDATE
    assert result.completeness_claimed is False
    with pytest.raises(CegisError, match="completeness"):
        replace(result, completeness_claimed=True)


def test_model_sketches_must_be_declarative_procedure_specs() -> None:
    with pytest.raises(CegisError, match="ProcedureSpec"):
        ModelSketch(procedure="print('code')", token_cost=1)  # type: ignore[arg-type]


def test_enumerative_variants_stay_within_plan_bounds() -> None:
    seed = valid_spec()
    bounded = plan(max_steps=len(seed.steps), max_proof=0)
    variants = enumerate_procedure_variants(seed, bounded)
    assert variants
    assert all(structural_overflow(item, bounded) is None for item in variants)
    assert all(StepOperation.RUN_PROOF not in {step.operation for step in item.steps} for item in variants)
    expanded = plan(max_steps=len(seed.steps) + 1, max_proof=0)
    wider = enumerate_procedure_variants(seed, expanded)
    assert any(
        StepOperation.RUN_STATIC_ANALYSIS in {step.operation for step in item.steps} for item in wider
    )


def test_builtin_template_flag_uses_the_validated_scaffold() -> None:
    spec = valid_spec(name="scaffold-procedure")
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(include_builtin_templates=True),
            enumerative_seeds=(spec,),
            validator=reject_sources(SynthesisSourceKind.ENUMERATIVE),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].source_kind is SynthesisSourceKind.BUILTIN_TEMPLATE
    assert result.surviving_candidates[0].procedure.name == "builtin-template.focused-validation"
