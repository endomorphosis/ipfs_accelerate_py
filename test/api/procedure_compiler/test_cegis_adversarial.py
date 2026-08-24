from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.cegis import (
    ASSURANCE_CAMPAIGN_API_INTERFACE_PIN,
    ASSURANCE_COUNTEREXAMPLE_ADAPTER_REVISION,
    EXECUTE_MUTATION_CAMPAIGN_COMMAND,
    REQUIRED_ASSURANCE_ATTACK_CLASSES,
    ZERO_CRITICAL_ESCAPED_MUTANTS,
    AssuranceAttackClass,
    AssuranceCounterexampleAdapter,
    AssuranceSeed,
    CounterexampleKind,
    EscapedMutantGate,
    ProcedureCegis,
    ProcedureSynthesisPlan,
    SkipReason,
    SynthesisRequest,
    SynthesisStatus,
    default_assurance_seeds,
    insert_closed_validation_step,
    inspect_assurance_seed,
    narrowing_for_attack,
    synthesize_procedure,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    HoleType,
    ProcedureAuthorityEnvelope,
    ProcedureEffect,
    ProcedureHole,
    ProcedureLocal,
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
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.invariant_mining import (
    AssuranceApiStatus,
)


def bindings(**changes: str) -> ArtifactBindings:
    values = {
        "repository_id": "repo-main",
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
        "objective_id": "PCPC-G000",
        "task_id": "PCPC-016",
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


def injection_hole(spec: ProcedureSpec) -> ProcedureHole:
    return ProcedureHole(
        hole_id="hole.prompt",
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


def named(spec: ProcedureSpec, name: str) -> ProcedureSpec:
    return replace(spec, name=name)


def with_provenance(spec: ProcedureSpec, *cids: str) -> ProcedureSpec:
    ordered: list[str] = []
    for item in (*spec.provenance_cids, *cids):
        if item and item not in ordered:
            ordered.append(item)
    return replace(spec, provenance_cids=tuple(ordered))


def insert_defense(spec: ProcedureSpec, operation: StepOperation) -> ProcedureSpec:
    recipes = {
        StepOperation.RUN_ADVERSARIAL_ASSURANCE: (
            "adversarial-assurance@1",
            "effect.validation",
            EffectClass.VALIDATION,
            "run-adversarial-assurance",
        ),
        StepOperation.CHECK_SCOPE: (
            "scope-checker@1",
            "effect.validation",
            EffectClass.VALIDATION,
            "check-scope",
        ),
        StepOperation.RUN_SELECTED_TESTS: (
            "test-runner@1",
            "effect.validation",
            EffectClass.VALIDATION,
            "run-selected-tests",
        ),
    }
    contract, effect_id, effect_class, step_id = recipes[operation]
    updated = insert_closed_validation_step(
        spec,
        operation,
        contract,
        effect_id=effect_id,
        effect_class=effect_class,
        step_id=step_id,
    )
    return named(updated, spec.name)


def spec_with_assurance_defenses(*, name: str = "assurance-hardened-procedure") -> ProcedureSpec:
    spec = named(valid_spec(), name)
    spec = insert_defense(spec, StepOperation.RUN_ADVERSARIAL_ASSURANCE)
    spec = insert_defense(spec, StepOperation.CHECK_SCOPE)
    return named(spec, name)


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


def seed_for(attack_class: AssuranceAttackClass, **changes: object) -> AssuranceSeed:
    base = next(
        item for item in default_assurance_seeds() if item.attack_class is attack_class
    )
    if not changes:
        return base
    return replace(base, **changes)  # type: ignore[arg-type]


WEAK_BUILDERS: dict[AssuranceAttackClass, object] = {
    AssuranceAttackClass.CRITICAL_SEEDED_MUTANT: lambda: named(
        valid_spec(), "weak-critical-mutant"
    ),
    AssuranceAttackClass.PROMPT_INJECTION: lambda: with_provenance(
        named(valid_spec(), "weak-prompt-injection"),
        "ignore-previous-instructions",
    ),
    AssuranceAttackClass.VALIDATION_WEAKENING: lambda: spec_without_selected_tests(
        name="weak-validation"
    ),
    AssuranceAttackClass.SCOPE_ESCAPE: lambda: named(valid_spec(), "weak-scope-escape"),
    AssuranceAttackClass.SELF_PROMOTION: lambda: with_provenance(
        named(valid_spec(), "weak-self-promotion"),
        "self-promote",
    ),
    AssuranceAttackClass.FORGED_EVIDENCE: lambda: with_provenance(
        named(valid_spec(), "weak-forged-evidence"),
        "forged-evidence",
    ),
    AssuranceAttackClass.UNSAFE_FAMILY_MERGE: lambda: with_provenance(
        named(valid_spec(), "weak-unsafe-family"),
        "UNSAFE_CROSS_FAMILY",
    ),
}


STRONG_BUILDERS: dict[AssuranceAttackClass, object] = {
    AssuranceAttackClass.CRITICAL_SEEDED_MUTANT: lambda: insert_defense(
        named(valid_spec(), "strong-critical-mutant"),
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
    ),
    AssuranceAttackClass.PROMPT_INJECTION: lambda: named(
        valid_spec(), "strong-prompt-injection"
    ),
    AssuranceAttackClass.VALIDATION_WEAKENING: lambda: named(
        valid_spec(), "strong-validation"
    ),
    AssuranceAttackClass.SCOPE_ESCAPE: lambda: insert_defense(
        named(valid_spec(), "strong-scope-escape"),
        StepOperation.CHECK_SCOPE,
    ),
    AssuranceAttackClass.SELF_PROMOTION: lambda: named(
        valid_spec(), "strong-self-promotion"
    ),
    AssuranceAttackClass.FORGED_EVIDENCE: lambda: named(
        valid_spec(), "strong-forged-evidence"
    ),
    AssuranceAttackClass.UNSAFE_FAMILY_MERGE: lambda: named(
        valid_spec(), "strong-family-boundary"
    ),
}


class RecordingCampaignApi:
    def __init__(
        self,
        *,
        reports: tuple[dict[str, object], ...] | None = None,
        result_cid: str = "aae-campaign-result",
        finding_cids: tuple[str, ...] = ("aae-campaign-receipt",),
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.reports = reports
        self.result_cid = result_cid
        self.finding_cids = finding_cids

    @property
    def interface_id(self) -> str:
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def probe_api(self, name: str) -> dict[str, object]:
        return {"command": name, "available": True, "status": "available"}

    def execute_mutation_campaign(
        self, plan_payload: object, policy: object, **kwargs: object
    ) -> dict[str, object]:
        reports = self.reports
        if reports is None:
            raw = kwargs.get("precomputed_reports") or ()
            reports = tuple(item for item in raw if isinstance(item, dict))  # type: ignore[misc]
        self.calls.append(
            {
                "plan": plan_payload,
                "policy": policy,
                "kwargs": kwargs,
                "reports": reports,
            }
        )
        survivors = sum(
            1
            for item in reports
            if "surviv" in str(item.get("terminal_status", "")).lower()
        )
        killed = sum(
            1
            for item in reports
            if "kill" in str(item.get("terminal_status", "")).lower()
        )
        return {
            "result_cid": self.result_cid,
            "finding_cids": self.finding_cids,
            "candidate_reports": reports,
            "survivor_count": survivors,
            "killed_count": killed,
            "precise_nonclaims": ("completeness-beyond-tested-obligations",),
        }


class UnavailableCampaignApi:
    @property
    def interface_id(self) -> str:
        return ASSURANCE_CAMPAIGN_API_INTERFACE_PIN

    def probe_api(self, name: str) -> dict[str, object]:
        return {
            "command": name,
            "available": False,
            "status": "typed_unavailable",
            "reason_code": "assurance_api_unavailable",
        }


def adapter_for(
    api: object | None = None,
    *,
    seeds: tuple[AssuranceSeed, ...] | None = None,
) -> AssuranceCounterexampleAdapter:
    campaign = api if api is not None else RecordingCampaignApi()
    kwargs: dict[str, object] = {"emitted_at_ms": 11}
    if seeds is not None:
        kwargs["seeds"] = seeds
    return AssuranceCounterexampleAdapter(campaign, **kwargs)  # type: ignore[arg-type]


def test_default_seeds_cover_required_attack_classes() -> None:
    seeds = default_assurance_seeds()
    assert tuple(item.attack_class for item in seeds) == REQUIRED_ASSURANCE_ATTACK_CLASSES
    assert all(item.critical for item in seeds)
    assert all(item.expected_killed for item in seeds)


def test_adapter_binds_existing_campaign_interface() -> None:
    api = RecordingCampaignApi()
    adapter = adapter_for(api, seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),))
    probe = adapter.probe()
    assert adapter.interface_id == ASSURANCE_CAMPAIGN_API_INTERFACE_PIN
    assert adapter.adapter_revision == ASSURANCE_COUNTEREXAMPLE_ADAPTER_REVISION
    assert probe["available"] is True
    assert probe["command"] == EXECUTE_MUTATION_CAMPAIGN_COMMAND
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(named(valid_spec(), "weak"),),
            assurance_adapter=adapter,
        )
    )
    assert api.calls
    assert api.calls[0]["plan"]["completeness_claimed"] is False
    assert api.calls[0]["plan"]["promotion_permitted"] is False
    assert api.calls[0]["policy"]["tree_id"] == "tree-abc123"
    assert result.counterexamples[0].kind is CounterexampleKind.ADVERSARIAL


@pytest.mark.parametrize("attack_class", REQUIRED_ASSURANCE_ATTACK_CLASSES)
def test_each_attack_class_rejects_and_persists_typed_counterexample(
    attack_class: AssuranceAttackClass,
) -> None:
    weak = WEAK_BUILDERS[attack_class]()
    strong = STRONG_BUILDERS[attack_class]()
    attack_seed = seed_for(attack_class)
    assert inspect_assurance_seed(weak, attack_seed) is True
    assert inspect_assurance_seed(strong, attack_seed) is False
    api = RecordingCampaignApi()
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak, strong),
            assurance_adapter=adapter_for(api, seeds=(attack_seed,)),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.name == strong.name
    assert result.rejected_candidates[0].procedure.name == weak.name
    assert result.rejected_candidates[0].procedure.state is ArtifactState.CANDIDATE
    matching = [
        item
        for item in result.counterexamples
        if item.obligation == attack_class.value
    ]
    assert matching
    assert matching[0].kind is CounterexampleKind.ADVERSARIAL
    assert matching[0].witness["attack_class"] == attack_class.value
    assert matching[0].witness["critical"] is True
    assert "aae-campaign-result" in matching[0].evidence_cids
    expected_ops = narrowing_for_attack(attack_class).required_operations
    if expected_ops:
        assert matching[0].constraints.required_operations == expected_ops
        assert any(item.reason is SkipReason.NARROWED for item in result.skipped_pairs) or (
            expected_ops[0] in {step.operation for step in strong.steps}
        )


def test_every_assurance_failure_narrows_remaining_candidates() -> None:
    weak = named(valid_spec(), "missing-adversarial")
    strong = insert_defense(
        named(valid_spec(), "has-adversarial"),
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
    )
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak, strong),
            assurance_seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),),
            assurance_adapter=adapter_for(
                seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),)
            ),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.name == "has-adversarial"
    assert StepOperation.RUN_ADVERSARIAL_ASSURANCE in {
        step.operation for step in result.surviving_candidates[0].procedure.steps
    }
    assert result.counterexamples[0].constraints.required_operations == (
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
    )
    assert result.rejected_candidates[0].procedure.name == "missing-adversarial"
    assert any(item.reason is SkipReason.NARROWED for item in result.skipped_pairs) or (
        StepOperation.RUN_ADVERSARIAL_ASSURANCE
        in {step.operation for step in result.surviving_candidates[0].procedure.steps}
    )


def test_strong_candidate_survives_all_required_attack_classes() -> None:
    weak = named(valid_spec(), "unhardened")
    strong = spec_with_assurance_defenses()
    api = RecordingCampaignApi()
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak, strong),
            assurance_adapter=adapter_for(api),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    survivor = result.surviving_candidates[0]
    assert survivor.procedure.name == "assurance-hardened-procedure"
    operations = {step.operation for step in survivor.procedure.steps}
    assert StepOperation.RUN_ADVERSARIAL_ASSURANCE in operations
    assert StepOperation.CHECK_SCOPE in operations
    hit_classes = {
        item.witness.get("attack_class")
        for item in result.counterexamples
        if item.kind is CounterexampleKind.ADVERSARIAL
    }
    assert AssuranceAttackClass.CRITICAL_SEEDED_MUTANT.value in hit_classes
    assert AssuranceAttackClass.SCOPE_ESCAPE.value in hit_classes
    assert result.escaped_mutant_gate is not None
    assert result.escaped_mutant_gate.blocked is False
    assert result.escaped_mutant_gate.escaped_critical_mutant_ids == ()
    assert result.promotion_requirements == (ZERO_CRITICAL_ESCAPED_MUTANTS,)
    assert survivor.procedure.state is ArtifactState.CANDIDATE
    assert result.candidate_artifacts[0].state is ArtifactState.CANDIDATE


def test_escaped_mutant_gate_is_immutable_later_promotion_requirement() -> None:
    strong = spec_with_assurance_defenses()
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(strong,),
            assurance_adapter=adapter_for(),
        )
    )
    gate = result.escaped_mutant_gate
    assert gate is not None
    assert gate.requirement_id == ZERO_CRITICAL_ESCAPED_MUTANTS
    assert gate.waivable is False
    assert gate.promotion_permitted is False
    assert gate.completeness_claimed is False
    assert gate.later_promotion_blocked is True
    assert gate.blocked is False
    assert gate.campaign_confirmed is True
    assert result.completeness_claimed is False
    with pytest.raises(Exception, match="waived"):
        replace(gate, waivable=True)
    with pytest.raises(Exception, match="promotion"):
        replace(gate, promotion_permitted=True)
    with pytest.raises(Exception, match="completeness"):
        replace(gate, completeness_claimed=True)


def test_escaped_mutant_gate_cannot_be_constructed_as_waivable() -> None:
    with pytest.raises(Exception, match="waived"):
        EscapedMutantGate(
            candidate_id="candidate.a",
            counterexample_set_cid="set.a",
            critical_seed_ids=("seed.critical-seeded-mutant",),
            escaped_critical_mutant_ids=(),
            waivable=True,
        )
    with pytest.raises(Exception, match="immutable"):
        EscapedMutantGate(
            candidate_id="candidate.a",
            counterexample_set_cid="set.a",
            critical_seed_ids=("seed.critical-seeded-mutant",),
            escaped_critical_mutant_ids=(),
            requirement_id="optional-mutant-budget",
        )


def test_critical_escaped_mutant_rejects_candidate() -> None:
    weak = named(valid_spec(), "escaped-mutant-host")
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak,),
            assurance_adapter=adapter_for(
                seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),)
            ),
        )
    )
    assert result.status is SynthesisStatus.INCOMPLETE
    assert result.surviving_candidates == ()
    assert result.rejected_candidates[0].procedure.name == "escaped-mutant-host"
    assert result.counterexamples[0].obligation == "critical-seeded-mutant"
    gate = result.escaped_mutant_gate
    assert gate is not None
    assert gate.blocked is True
    assert "seed.critical-seeded-mutant" in gate.escaped_critical_mutant_ids
    assert result.promotion_requirements == (ZERO_CRITICAL_ESCAPED_MUTANTS,)
    assert all(
        artifact.state is ArtifactState.REJECTED for artifact in result.candidate_artifacts
    )


def test_unavailable_api_is_typed_and_does_not_hide_local_failures() -> None:
    weak = named(valid_spec(), "local-failure")
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak,),
            assurance_adapter=adapter_for(
                UnavailableCampaignApi(),
                seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),),
            ),
        )
    )
    assert result.status is SynthesisStatus.INCOMPLETE
    assert result.surviving_candidates == ()
    assert result.counterexamples[0].kind is CounterexampleKind.ADVERSARIAL
    assert result.escaped_mutant_gate is not None
    assert result.escaped_mutant_gate.blocked is True
    assert result.escaped_mutant_gate.campaign_confirmed is False
    assert result.escaped_mutant_gate.promotion_permitted is False


def test_unavailable_api_does_not_permit_promotion_on_local_pass() -> None:
    strong = spec_with_assurance_defenses()
    adapter = adapter_for(UnavailableCampaignApi())
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(strong,),
            assurance_adapter=adapter,
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.state is ArtifactState.CANDIDATE
    probe = adapter.probe()
    assert probe["available"] is False
    assert probe["status"] == AssuranceApiStatus.TYPED_UNAVAILABLE.value
    gate = result.escaped_mutant_gate
    assert gate is not None
    assert gate.campaign_confirmed is False
    assert gate.promotion_permitted is False
    assert gate.requirement_id == ZERO_CRITICAL_ESCAPED_MUTANTS


def test_unvalidated_typed_hole_is_prompt_injection() -> None:
    weak = replace(
        named(valid_spec(), "hole-injection"),
        holes=(injection_hole(valid_spec()),),
    )
    strong = named(valid_spec(), "no-hole")
    attack_seed = seed_for(AssuranceAttackClass.PROMPT_INJECTION)
    assert inspect_assurance_seed(weak, attack_seed) is True
    assert inspect_assurance_seed(strong, attack_seed) is False
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(weak, strong),
            assurance_adapter=adapter_for(seeds=(attack_seed,)),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.name == "no-hole"
    assert result.counterexamples[0].obligation == "prompt-injection"


def test_enumerative_refinement_inserts_adversarial_assurance() -> None:
    seed = valid_spec(name="enumerative-seed")
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            enumerative_seeds=(seed,),
            assurance_adapter=adapter_for(
                seeds=(seed_for(AssuranceAttackClass.CRITICAL_SEEDED_MUTANT),)
            ),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    survivor = result.surviving_candidates[0]
    assert StepOperation.RUN_ADVERSARIAL_ASSURANCE in {
        step.operation for step in survivor.procedure.steps
    }
    assert result.counterexamples[0].kind is CounterexampleKind.ADVERSARIAL
    assert result.escaped_mutant_gate is not None
    assert result.escaped_mutant_gate.blocked is False


def test_campaign_receipts_are_persisted_only_as_candidate_counterexamples() -> None:
    weak = named(valid_spec(), "receipt-host")
    strong = spec_with_assurance_defenses(name="receipt-survivor")
    result = synthesize_procedure(
        SynthesisRequest(
            plan=plan(),
            templates=(weak, strong),
            assurance_adapter=adapter_for(),
        )
    )
    assert result.assurance_receipt_cids
    assert "aae-campaign-result" in result.assurance_receipt_cids
    assert result.counterexample_artifacts
    assert all(item.state is ArtifactState.REJECTED for item in result.counterexample_artifacts)
    assert all(
        artifact.state in {ArtifactState.CANDIDATE, ArtifactState.REJECTED}
        for artifact in result.candidate_artifacts
    )
    assert all(
        artifact.state not in {ArtifactState.VERIFIED, ArtifactState.PROMOTED}
        for artifact in result.candidate_artifacts
    )


def test_surviving_candidates_remain_unpromoted_after_assurance() -> None:
    result = ProcedureCegis().synthesize(
        SynthesisRequest(
            plan=plan(),
            templates=(spec_with_assurance_defenses(),),
            assurance_adapter=adapter_for(),
        )
    )
    assert result.status is SynthesisStatus.CONVERGED
    assert result.surviving_candidates[0].procedure.state is ArtifactState.CANDIDATE
    assert result.plan_artifact.state is ArtifactState.CANDIDATE
    assert result.completeness_claimed is False
    assert result.escaped_mutant_gate is not None
    assert result.escaped_mutant_gate.promotion_permitted is False
