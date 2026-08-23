from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ConditionOperator,
    EffectClass,
    ProcedureEffect,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    RiskClass,
    StepOperation,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.planner_adapter import (
    ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE,
    ADAPTIVE_PLANNER_MODULE,
    HAMMER_TRACE_SCHEMA_NAME,
    REQUIRED_PLANNER_ORDER,
    AdaptivePlannerCompatibility,
    CompositionReason,
    MatchDisposition,
    PlannerCompatibilityStatus,
    PlannerDispatchStatus,
    PlannerStage,
    PlanningBoundary,
    PlanningNeedKind,
    PlanningRequest,
    ProcedureCompositionValidator,
    ProcedureOperator,
    ProcedurePlannerAdapter,
    probe_adaptive_planner,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.procedure_ir import (
    parse_procedure_spec,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.runtime import (
    compiler_capabilities,
)


def _load_verifier_helpers():
    path = Path(__file__).with_name("test_verifier.py")
    spec = importlib.util.spec_from_file_location("_pcpc019_verifier_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verifier test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_verifier_helpers()
bindings = _helpers.bindings
valid_spec = _helpers.valid_spec


def _condition(*, kind: str, condition_id: str, binding: str, operator: ConditionOperator):
    cls = ProcedurePrecondition if kind == "pre" else ProcedurePostcondition
    return cls(
        condition_id=condition_id,
        binding=binding,
        operator=operator,
        evidence_producer="tree-verifier@1",
        evidence_type="current-tree-receipt@1",
    )


def _operator(
    spec=None,
    *,
    need_kind: PlanningNeedKind = PlanningNeedKind.TASK,
    satisfied_ids: tuple[str, ...] = ("task.import-purity",),
    verified: bool = True,
    promoted: bool = True,
    language_classes: tuple[str, ...] = ("python",),
    **changes,
) -> ProcedureOperator:
    values = {
        "procedure": spec or valid_spec(),
        "need_kind": need_kind,
        "satisfied_ids": satisfied_ids,
        "verified": verified,
        "promoted": promoted,
        "language_classes": language_classes,
        "framework_classes": ("stdlib",),
        "repository_families": ("ipfs-accelerate",),
        "certificate_cid": "certificate-1",
    }
    values.update(changes)
    return ProcedureOperator(**values)


def _request(
    *,
    need_kind: PlanningNeedKind = PlanningNeedKind.TASK,
    need_id: str = "task.import-purity",
    criterion_ids: tuple[str, ...] = (),
    allowed_effect_classes: tuple[EffectClass, ...] = (),
    risk_ceiling: RiskClass = RiskClass.OBSERVATION_ONLY,
    resource_budget: ProcedureResourceEnvelope | None = None,
    **binding_changes,
) -> PlanningRequest:
    spec = valid_spec()
    bound = spec.bindings if not binding_changes else replace(spec.bindings, **binding_changes)
    return PlanningRequest(
        boundary=PlanningBoundary(
            bindings=bound,
            task_family_id=spec.task_family_id,
            need_kind=need_kind,
            need_id=need_id,
            criterion_ids=criterion_ids,
            allowed_effect_classes=allowed_effect_classes,
            risk_ceiling=risk_ceiling,
            required_capability_ids=spec.authority.required_capability_ids,
        ),
        resource_budget=resource_budget,
    )


def _ready_post() -> ProcedurePostcondition:
    return _condition(
        kind="post",
        condition_id="postcondition.state-ready",
        binding="local:state",
        operator=ConditionOperator.ADMITTED,
    )


def _ready_pre() -> ProcedurePrecondition:
    return _condition(
        kind="pre",
        condition_id="precondition.state-ready",
        binding="local:state",
        operator=ConditionOperator.ADMITTED,
    )


def _tree_pre() -> ProcedurePrecondition:
    return _condition(
        kind="pre",
        condition_id="precondition.current-tree",
        binding="binding:tree_id",
        operator=ConditionOperator.CURRENT,
    )


def _tests_post() -> ProcedurePostcondition:
    return _condition(
        kind="post",
        condition_id="postcondition.tests-admitted",
        binding="local:test-result",
        operator=ConditionOperator.ADMITTED,
    )


def _write_effect() -> ProcedureEffect:
    return ProcedureEffect(
        effect_id="effect.write",
        effect_class=EffectClass.REPOSITORY_WRITE,
        targets=("ipfs_accelerate_py/agent_supervisor/example.py",),
        reversible=True,
    )


def _rollback_for_write() -> ProcedureRollback:
    return ProcedureRollback(
        rollback_id="rollback.write",
        trigger_effect_ids=("effect.write",),
        step_ids=("receipt",),
        verification_observation_ids=("observation.tests",),
        exact_target_cid="tree-abc123",
    )


def _prepare_spec(*, name: str):
    spec = valid_spec(name=name)
    return replace(
        spec,
        preconditions=(_tree_pre(),),
        postconditions=(_ready_post(),),
    )


def _finish_spec(*, name: str, **changes):
    spec = valid_spec(name=name)
    values = {
        "preconditions": (_ready_pre(),),
        "postconditions": (_tests_post(),),
    }
    values.update(changes)
    return replace(spec, **values)


def _qualified_probe() -> AdaptivePlannerCompatibility:
    return AdaptivePlannerCompatibility(
        status=PlannerCompatibilityStatus.QUALIFIED,
        reason_code="adaptive_planner_qualified",
        diagnostic="",
        planner_version=2,
    )


def test_required_planner_order_is_exact() -> None:
    assert tuple(item.value for item in REQUIRED_PLANNER_ORDER) == (
        "exact_verified_procedure",
        "composable_verified_procedures",
        "deterministic_baseline",
        "bounded_local_synthesis",
        "small_local_model",
        "standard_remote_model",
        "strong_remote_model",
        "human_escalation",
    )
    adapter = ProcedurePlannerAdapter()
    assert adapter.planner_order is REQUIRED_PLANNER_ORDER


def test_current_adaptive_planner_import_qualifies_or_is_typed_unavailable() -> None:
    result = probe_adaptive_planner()
    if result.qualified:
        assert result.status is PlannerCompatibilityStatus.QUALIFIED
        assert result.planner_version == 2
        assert result.reason_code == "adaptive_planner_qualified"
        return
    assert result.status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE
    assert result.reason_code == ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE
    assert HAMMER_TRACE_SCHEMA_NAME in result.diagnostic
    assert "NameError" in result.diagnostic


def test_match_and_compose_do_not_import_adaptive_planner() -> None:
    sys.modules.pop(ADAPTIVE_PLANNER_MODULE, None)
    operator = _operator()
    adapter = ProcedurePlannerAdapter((operator,))
    matches = adapter.match(_request())
    assert matches[0].disposition is MatchDisposition.EXACT
    decision = adapter.compose((operator,))
    assert decision.accepted
    assert ADAPTIVE_PLANNER_MODULE not in sys.modules


def test_incompatible_dispatch_is_typed_unavailable_before_procedure_use() -> None:
    operator = _operator()
    adapter = ProcedurePlannerAdapter((operator,))
    result = adapter.dispatch(_request())
    live = probe_adaptive_planner()
    if live.qualified:
        assert result.status is PlannerDispatchStatus.CANDIDATES
        assert result.selected_stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
        assert result.candidates[0].operators[0].procedure_id == operator.procedure_id
        return
    assert result.status is PlannerDispatchStatus.TYPED_UNAVAILABLE
    assert result.typed_unavailable
    assert result.reason_code == ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE
    assert result.candidates == ()
    matches = adapter.match(_request())
    assert matches[0].exact


def test_other_runtime_remains_usable_when_planner_is_unavailable() -> None:
    capabilities = compiler_capabilities()
    assert capabilities["parse_and_validate"] is True
    assert capabilities["deterministic_invoke"] is True
    spec = valid_spec()
    parsed = parse_procedure_spec(spec)
    assert parsed.name == spec.name
    assert parsed.content_id == spec.content_id
    adapter = ProcedurePlannerAdapter((_operator(spec),))
    assert adapter.match(_request())[0].accepted


def test_exact_match_requires_compatible_boundary() -> None:
    operator = _operator()
    adapter = ProcedurePlannerAdapter((operator,))
    exact = adapter.match(_request())
    assert exact[0].disposition is MatchDisposition.EXACT
    assert exact[0].reason_code == "exact-compatible-boundary"
    drifted = adapter.match(_request(tree_id="tree-other"))
    assert drifted[0].disposition is MatchDisposition.INCOMPATIBLE
    assert drifted[0].reason_code == "incompatible-boundary"
    family_miss = adapter.match(
        PlanningRequest(
            boundary=PlanningBoundary(
                bindings=bindings(),
                task_family_id="OTHER_FAMILY",
                need_kind=PlanningNeedKind.TASK,
                need_id="task.import-purity",
                risk_ceiling=RiskClass.OBSERVATION_ONLY,
                required_capability_ids=("capability.tests",),
            )
        )
    )
    assert family_miss[0].reason_code == "task-family-mismatch"


def test_partial_criterion_does_not_claim_the_task() -> None:
    criterion = _operator(
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.tests",),
    )
    adapter = ProcedurePlannerAdapter((criterion,))
    partial = adapter.match(
        _request(
            need_kind=PlanningNeedKind.TASK,
            need_id="task.import-purity",
            criterion_ids=("criterion.tests", "criterion.proof"),
        )
    )
    assert partial[0].disposition is MatchDisposition.PARTIAL_CRITERION
    assert partial[0].satisfied_ids == ("criterion.tests",)
    assert criterion.claims_task is False
    exact_criterion = adapter.match(
        _request(need_kind=PlanningNeedKind.CRITERION, need_id="criterion.tests")
    )
    assert exact_criterion[0].disposition is MatchDisposition.EXACT
    overclaim = ProcedurePlannerAdapter(
        (
            _operator(
                need_kind=PlanningNeedKind.TASK,
                satisfied_ids=("task.import-purity",),
            ),
        )
    ).match(_request(need_kind=PlanningNeedKind.CRITERION, need_id="criterion.tests"))
    assert overclaim[0].disposition is MatchDisposition.OVERCLAIM


def test_qualified_planner_uses_required_order_and_exact_procedures_only() -> None:
    exact = _operator(valid_spec(name="exact-procedure"))
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    finish = _operator(
        _finish_spec(name="finish-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    adapter = ProcedurePlannerAdapter(
        (exact, prepare, finish),
        planner_probe=_qualified_probe,
    )
    result = adapter.dispatch(_request())
    assert result.status is PlannerDispatchStatus.CANDIDATES
    assert result.planner_order == REQUIRED_PLANNER_ORDER
    assert result.selected_stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
    assert [item.stage for item in result.candidates] == [
        PlannerStage.EXACT_VERIFIED_PROCEDURE
    ]
    assert all(item.stage in REQUIRED_PLANNER_ORDER[:2] for item in result.candidates)
    assert result.candidates[0].operators[0].procedure_id == "exact-procedure"

    composable_only = ProcedurePlannerAdapter(
        (prepare, finish),
        planner_probe=_qualified_probe,
    )
    composed = composable_only.dispatch(
        _request(
            need_kind=PlanningNeedKind.TASK,
            need_id="task.import-purity",
            criterion_ids=("criterion.prepare", "criterion.finish"),
        )
    )
    assert composed.selected_stage is PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES
    assert composed.candidates[0].operators[0].procedure_id == "prepare-procedure"
    assert composed.candidates[0].operators[1].procedure_id == "finish-procedure"
    assert composed.candidates[0].composition is not None
    assert composed.candidates[0].composition.accepted


def test_composition_requires_exact_post_to_pre_entailment() -> None:
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    finish = _operator(
        _finish_spec(name="finish-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    validator = ProcedureCompositionValidator()
    accepted = validator.validate((prepare, finish))
    assert accepted.accepted
    assert accepted.reason is CompositionReason.ACCEPTED
    assert accepted.entailment[0].exact
    assert accepted.entailment[0].missing_condition_ids == ()

    mismatched = _operator(
        replace(valid_spec(name="mismatched-procedure"), preconditions=(_tree_pre(),)),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    refused = validator.validate((prepare, mismatched))
    assert refused.accepted is False
    assert refused.reason is CompositionReason.ENTAILMENT_FAILURE
    assert refused.entailment[0].missing_condition_ids == ("precondition.current-tree",)


def test_composition_rejects_incompatible_effects_authority_and_environment() -> None:
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    other_env = _operator(
        replace(
            _finish_spec(name="env-procedure"),
            bindings=replace(valid_spec().bindings, environment_id="other-lock"),
        ),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    other_policy = _operator(
        replace(
            _finish_spec(name="policy-procedure"),
            authority=replace(
                valid_spec().authority,
                authority_policy_revision="other-policy",
            ),
        ),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    validator = ProcedureCompositionValidator()
    env = validator.validate((prepare, other_env))
    assert env.reason is CompositionReason.ENVIRONMENT_INCOMPATIBLE
    policy = validator.validate((prepare, other_policy))
    assert policy.reason is CompositionReason.AUTHORITY_INCOMPATIBLE
    claimed = validator.validate(
        (
            prepare,
            _operator(
                _finish_spec(name="finish-procedure"),
                need_kind=PlanningNeedKind.CRITERION,
                satisfied_ids=("criterion.finish",),
            ),
        ),
        claimed_effect_classes=(
            EffectClass.VALIDATION,
            EffectClass.RECEIPT_EMIT,
            EffectClass.MERGE,
        ),
    )
    assert claimed.reason is CompositionReason.EFFECT_INCOMPATIBLE


def test_composition_budget_is_additive_and_bounded() -> None:
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    finish = _operator(
        _finish_spec(name="finish-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    validator = ProcedureCompositionValidator()
    tight = ProcedureResourceEnvelope(
        wall_time_ms=60_000,
        cpu_time_ms=60_000,
        memory_bytes=128_000_000,
        disk_bytes=128_000_000,
        model_token_limit=0,
        model_call_limit=0,
        subprocess_limit=4,
    )
    overflow = validator.validate((prepare, finish), resource_budget=tight)
    assert overflow.reason is CompositionReason.BUDGET_OVERFLOW
    assert overflow.composed_resources["wall_time_ms"] == 120_000
    room = ProcedureResourceEnvelope(
        wall_time_ms=120_000,
        cpu_time_ms=120_000,
        memory_bytes=256_000_000,
        disk_bytes=256_000_000,
        model_token_limit=0,
        model_call_limit=0,
        subprocess_limit=8,
    )
    accepted = validator.validate((prepare, finish), resource_budget=room)
    assert accepted.accepted
    assert accepted.composed_resources["cpu_time_ms"] == 120_000


def test_composition_requires_defined_rollback_and_complete_validation() -> None:
    write_spec = replace(
        _finish_spec(name="write-procedure"),
        declared_effects=valid_spec().declared_effects + (_write_effect(),),
        authority=replace(
            valid_spec().authority,
            allowed_operations=valid_spec().authority.allowed_operations
            + (StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,),
            risk_ceiling=RiskClass.REPOSITORY_WRITE,
        ),
    )
    missing_rollback = _operator(
        write_spec,
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    validator = ProcedureCompositionValidator()
    refused = validator.validate((prepare, missing_rollback))
    assert refused.reason is CompositionReason.ROLLBACK_UNDEFINED

    with_rollback = _operator(
        replace(write_spec, rollback=(_rollback_for_write(),)),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    accepted = validator.validate((prepare, with_rollback))
    assert accepted.accepted
    assert accepted.composed_rollback == (_rollback_for_write(),)
    assert "focused-tests@1" in accepted.composed_validation.required_test_contracts
    assert "proof-runner@1" in accepted.composed_validation.required_proof_contracts
    assert accepted.composed_validation.required_observation_ids


def test_composition_rejects_cycles_and_hidden_effect_escalation() -> None:
    prepare = _operator(
        _prepare_spec(name="prepare-procedure"),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.prepare",),
    )
    validator = ProcedureCompositionValidator()
    cyclic = validator.validate((prepare, prepare))
    assert cyclic.reason is CompositionReason.CYCLE

    write = _operator(
        replace(
            _finish_spec(name="write-procedure"),
            declared_effects=valid_spec().declared_effects + (_write_effect(),),
            rollback=(_rollback_for_write(),),
            authority=replace(
                valid_spec().authority,
                allowed_operations=valid_spec().authority.allowed_operations
                + (StepOperation.APPLY_APPROVED_PATCH_TEMPLATE,),
                risk_ceiling=RiskClass.REPOSITORY_WRITE,
            ),
        ),
        need_kind=PlanningNeedKind.CRITERION,
        satisfied_ids=("criterion.finish",),
    )
    hidden_risk = validator.validate(
        (prepare, write),
        claimed_risk_ceiling=RiskClass.OBSERVATION_ONLY,
    )
    assert hidden_risk.reason is CompositionReason.HIDDEN_EFFECT_ESCALATION
    hidden_effects = validator.validate(
        (prepare, write),
        claimed_effect_classes=(EffectClass.VALIDATION, EffectClass.RECEIPT_EMIT),
    )
    assert hidden_effects.reason is CompositionReason.HIDDEN_EFFECT_ESCALATION


def test_unusable_operator_is_not_matched_or_composed() -> None:
    shadow = _operator(verified=True, promoted=False)
    adapter = ProcedurePlannerAdapter((shadow,))
    match = adapter.match(_request())
    assert match[0].disposition is MatchDisposition.INCOMPATIBLE
    assert match[0].reason_code == "operator-not-usable"
    decision = adapter.compose((shadow,))
    assert decision.reason is CompositionReason.UNVERIFIED
