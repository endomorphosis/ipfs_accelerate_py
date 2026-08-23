from __future__ import annotations

import ast
import importlib.util
import inspect
import types
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ConditionOperator,
    EffectClass,
    ProcedureEffect,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureRollback,
    ProcedureValidationPlan,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.planner_adapter import (
    ADAPTIVE_PLANNER_MODULE,
    ADAPTIVE_PLANNER_SYMBOL,
    HAMMER_TRACE_IMPORT_SIGNATURE,
    HAMMER_TRACE_REASON_CODE,
    REQUIRED_PLANNER_ORDER,
    AdaptivePlannerCompatibilityStatus,
    CatalogEntry,
    CompositionReason,
    EntailmentEvidence,
    InMemoryProcedureCatalog,
    PlannerAdapterStatus,
    PlannerStage,
    PlanningRequest,
    ProcedureCompositionValidator,
    ProcedureOperator,
    ProcedurePlannerAdapter,
    RejectionReason,
    SatisfactionKind,
    probe_adaptive_planner,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.runtime import (
    compiler_capabilities,
)


def _load_registry_helpers():
    path = Path(__file__).with_name("test_registry.py")
    spec = importlib.util.spec_from_file_location("_pcpc019_registry_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registry test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_registry_helpers()
issue_for = _helpers.issue_for
make_registry = _helpers.make_registry
promote_head = _helpers.promote_head
register_spec = _helpers.register_spec
valid_spec = _helpers.valid_spec


class _QualifiedPlanner:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def plan(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("adapter must not reinterpret AdaptivePlanner.plan")


def _available_importer(name: str) -> types.SimpleNamespace:
    assert name == ADAPTIVE_PLANNER_MODULE
    return types.SimpleNamespace(**{ADAPTIVE_PLANNER_SYMBOL: _QualifiedPlanner})


def _hammer_importer(name: str) -> None:
    assert name == ADAPTIVE_PLANNER_MODULE
    raise NameError("name 'HAMMER_TRACE_SCHEMA' is not defined")


def _missing_importer(name: str) -> None:
    raise ImportError("adaptive planner module is missing")


def _handoff_post():
    return ProcedurePostcondition(
        condition_id="postcondition.handoff",
        binding="binding:handoff",
        operator=ConditionOperator.EQUALS,
        operand="ready",
        evidence_producer="handoff-producer@1",
        evidence_type="handoff-receipt@1",
    )


def _handoff_pre():
    return ProcedurePrecondition(
        condition_id="precondition.handoff",
        binding="binding:handoff",
        operator=ConditionOperator.EQUALS,
        operand="ready",
        evidence_producer="handoff-producer@1",
        evidence_type="handoff-receipt@1",
    )


def _rollback(spec, rollback_id: str) -> ProcedureRollback:
    return ProcedureRollback(
        rollback_id=rollback_id,
        trigger_effect_ids=(spec.declared_effects[0].effect_id,),
        step_ids=(spec.entry_step_id,),
        verification_observation_ids=(spec.observations[0].observation_id,),
        exact_target_cid=spec.bindings.tree_id,
    )


def _with_rollback(spec, rollback_id: str):
    return replace(spec, rollback=(_rollback(spec, rollback_id),))


def _predecessor(name: str = "procedure-a"):
    spec = valid_spec(name=name)
    return _with_rollback(
        replace(spec, postconditions=spec.postconditions + (_handoff_post(),)),
        "rollback-a",
    )


def _successor(name: str = "procedure-b"):
    spec = valid_spec(name=name)
    return _with_rollback(replace(spec, preconditions=(_handoff_pre(),)), "rollback-b")


def _operator(spec, **changes: object) -> ProcedureOperator:
    values = {
        "revision_id": spec.name + "-revision",
        "certificate_cid": spec.name + "-certificate",
        "satisfaction_kind": SatisfactionKind.TASK,
        "satisfaction_id": spec.name,
    }
    values.update(changes)
    return ProcedureOperator.from_spec(spec, **values)


def _request(spec, **changes: object) -> PlanningRequest:
    values = {
        "bindings": spec.bindings,
        "task_family_id": spec.task_family_id,
        "task_id": spec.bindings.task_id,
        "capability_ids": spec.authority.required_capability_ids,
        "max_risk": RiskClass.REPOSITORY_WRITE,
        "language_classes": ("python",),
        "framework_classes": ("stdlib",),
        "repository_families": (spec.bindings.repository_id,),
    }
    values.update(changes)
    return PlanningRequest(**values)


def _promote(registry, spec):
    _issued, _candidate, certificate, _context = issue_for(spec)
    registered = register_spec(registry, spec, certificate)
    return promote_head(registry, registered, spec)


def _adapter_with(spec, *, catalog: InMemoryProcedureCatalog, **adapter_changes):
    _issued, _candidate, certificate, context = issue_for(spec)
    registry = make_registry(context)
    _promote(registry, spec)
    values = {
        "registry": registry,
        "catalog": catalog,
        "importer": _available_importer,
    }
    values.update(adapter_changes)
    return ProcedurePlannerAdapter(**values), registry


def test_required_planner_order_is_closed_and_exact() -> None:
    assert REQUIRED_PLANNER_ORDER == (
        "exact_verified_procedure",
        "composable_verified_procedures",
        "deterministic_baseline",
        "bounded_local_synthesis",
        "small_local_model",
        "standard_remote_model",
        "strong_remote_model",
        "human_escalation",
    )


def test_adapter_source_does_not_import_adaptive_planner() -> None:
    from ipfs_accelerate_py.agent_supervisor.procedure_compiler import planner_adapter

    tree = ast.parse(inspect.getsource(planner_adapter))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "adaptive_planner" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "adaptive_planner" not in alias.name
    assert ADAPTIVE_PLANNER_SYMBOL not in vars(planner_adapter)


def test_live_adaptive_planner_import_is_available_or_typed_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = valid_spec()
    _issued, _candidate, certificate, context = issue_for(spec)
    registry = make_registry(context)
    _promote(registry, spec)
    catalog = InMemoryProcedureCatalog(
        {
            spec.content_id: CatalogEntry(
                spec=spec,
                satisfaction_kind=SatisfactionKind.TASK,
                satisfaction_id=spec.bindings.task_id,
            )
        }
    )

    def boom(*_args: object, **_kwargs: object):
        raise AssertionError("registry must not be consulted before qualification")

    adapter = ProcedurePlannerAdapter(registry, catalog)
    compatibility = adapter.qualify()
    if compatibility.available:
        decision = adapter.plan(_request(spec))
        assert decision.status is PlannerAdapterStatus.AVAILABLE
        assert decision.registry_consulted is True
        assert decision.selected is not None
        assert decision.selected.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
        assert decision.procedures_dispatched is True
        return
    monkeypatch.setattr(registry, "filter", boom)
    decision = adapter.plan(_request(spec))
    assert decision.status is PlannerAdapterStatus.TYPED_UNAVAILABLE
    assert (
        decision.compatibility.status is AdaptivePlannerCompatibilityStatus.INCOMPATIBLE
    )
    assert decision.reason_code == HAMMER_TRACE_REASON_CODE
    assert decision.compatibility.signature == HAMMER_TRACE_IMPORT_SIGNATURE
    assert decision.registry_consulted is False
    assert decision.procedures_dispatched is False
    assert decision.selected is None
    assert decision.candidates == ()


def test_incompatibility_keeps_other_runtime_usable() -> None:
    spec = valid_spec()
    _issued, _candidate, certificate, context = issue_for(spec)
    registry = make_registry(context)
    promoted = _promote(registry, spec)
    adapter = ProcedurePlannerAdapter(
        registry,
        InMemoryProcedureCatalog({spec.content_id: spec}),
        importer=_hammer_importer,
    )
    decision = adapter.plan(_request(spec))
    assert decision.status is PlannerAdapterStatus.TYPED_UNAVAILABLE
    assert decision.procedures_dispatched is False
    capabilities = compiler_capabilities()
    assert capabilities["parse_and_validate"] is True
    assert capabilities["deterministic_invoke"] is True
    assert capabilities["promote"] is False
    status = registry.status()
    assert status["procedure_count"] == 1
    assert registry.get(spec.name).revision_id == promoted.revision.revision_id


def test_hammer_trace_probe_is_typed_incompatible_before_dispatch() -> None:
    compatibility = probe_adaptive_planner(importer=_hammer_importer)
    assert compatibility.status is AdaptivePlannerCompatibilityStatus.INCOMPATIBLE
    assert compatibility.reason_code == HAMMER_TRACE_REASON_CODE
    assert compatibility.available is False


def test_missing_planner_is_typed_unavailable() -> None:
    compatibility = probe_adaptive_planner(importer=_missing_importer)
    assert compatibility.status is AdaptivePlannerCompatibilityStatus.TYPED_UNAVAILABLE
    assert compatibility.reason_code == "adaptive_planner_import_failed"


def test_qualified_planner_uses_required_order_and_prefers_exact_procedure() -> None:
    exact_spec = valid_spec(name="exact-procedure")
    first = _predecessor("compose-a")
    second = _successor("compose-b")
    _issued, _candidate, certificate, context = issue_for(exact_spec)
    registry = make_registry(context)
    _promote(registry, exact_spec)
    _promote(registry, first)
    _promote(registry, second)
    catalog = InMemoryProcedureCatalog(
        {
            exact_spec.content_id: CatalogEntry(
                spec=exact_spec,
                satisfaction_kind=SatisfactionKind.TASK,
                satisfaction_id=exact_spec.bindings.task_id,
            ),
            first.content_id: CatalogEntry(spec=first),
            second.content_id: CatalogEntry(spec=second),
        }
    )
    adapter = ProcedurePlannerAdapter(registry, catalog, importer=_available_importer)
    decision = adapter.plan(_request(exact_spec))
    assert decision.status is PlannerAdapterStatus.AVAILABLE
    assert tuple(item.stage.value for item in decision.candidates) == REQUIRED_PLANNER_ORDER
    assert decision.selected is not None
    assert decision.selected.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
    assert decision.selected.operators[0].procedure_cid == exact_spec.content_id
    assert decision.procedures_dispatched is True
    composable = next(
        item
        for item in decision.candidates
        if item.stage is PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES
    )
    assert composable.applicable is True
    baseline = next(
        item
        for item in decision.candidates
        if item.stage is PlannerStage.DETERMINISTIC_BASELINE
    )
    assert baseline.applicable is True
    assert baseline.uses_procedure is False


def test_exact_match_requires_compatible_bindings_and_identity() -> None:
    spec = valid_spec()
    catalog = InMemoryProcedureCatalog(
        {
            spec.content_id: CatalogEntry(
                spec=spec,
                satisfaction_kind=SatisfactionKind.TASK,
                satisfaction_id=spec.bindings.task_id,
            )
        }
    )
    adapter, _registry = _adapter_with(spec, catalog=catalog)
    decision = adapter.plan(_request(spec, procedure_cid=spec.content_id))
    assert decision.selected is not None
    assert decision.selected.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
    assert decision.selected.operators[0].procedure_cid == spec.content_id
    assert decision.selected.claims_task is True


def test_partial_criterion_matches_without_claiming_the_task() -> None:
    spec = valid_spec(name="criterion-procedure")
    catalog = InMemoryProcedureCatalog(
        {
            spec.content_id: CatalogEntry(
                spec=spec,
                satisfaction_kind=SatisfactionKind.CRITERION,
                satisfaction_id="criterion.focused-tests",
            )
        }
    )
    adapter, _registry = _adapter_with(spec, catalog=catalog)
    criterion = adapter.plan(
        _request(
            spec,
            task_id="",
            criterion_id="criterion.focused-tests",
        )
    )
    assert criterion.selected is not None
    assert criterion.selected.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE
    assert criterion.selected.claims_task is False
    assert criterion.selected.operators[0].satisfaction_kind is SatisfactionKind.CRITERION

    task = adapter.plan(_request(spec))
    assert task.selected is not None
    assert task.selected.stage is PlannerStage.DETERMINISTIC_BASELINE
    assert task.procedures_dispatched is False
    assert any(
        item.reason_code is RejectionReason.SATISFACTION_KIND_MISMATCH
        for item in task.rejected
    )


def test_incompatible_boundary_does_not_use_the_procedure() -> None:
    spec = valid_spec()
    catalog = InMemoryProcedureCatalog(
        {
            spec.content_id: CatalogEntry(
                spec=spec,
                satisfaction_kind=SatisfactionKind.TASK,
                satisfaction_id=spec.bindings.task_id,
            )
        }
    )
    adapter, _registry = _adapter_with(spec, catalog=catalog)
    request = _request(
        spec,
        bindings=replace(spec.bindings, repository_commit="commit-other"),
    )
    decision = adapter.plan(request)
    assert decision.selected is not None
    assert decision.selected.stage is PlannerStage.DETERMINISTIC_BASELINE
    assert decision.procedures_dispatched is False
    assert any(
        item.reason_code is RejectionReason.INCOMPATIBLE_BOUNDARY
        for item in decision.rejected
    )


def test_qualified_planner_selects_composable_verified_procedures() -> None:
    first = _predecessor()
    second = _successor()
    _issued, _candidate, certificate, context = issue_for(first)
    registry = make_registry(context)
    _promote(registry, first)
    _promote(registry, second)
    catalog = InMemoryProcedureCatalog({first.content_id: first, second.content_id: second})
    adapter = ProcedurePlannerAdapter(registry, catalog, importer=_available_importer)
    decision = adapter.plan(_request(first))
    assert decision.selected is not None
    assert decision.selected.stage is PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES
    assert tuple(item.procedure_cid for item in decision.selected.operators) == (
        first.content_id,
        second.content_id,
    )
    assert decision.selected.composition is not None
    assert decision.selected.composition.accepted is True
    assert decision.selected.composition.composed_rollback_ids == (
        "rollback-b",
        "rollback-a",
    )


def test_composition_requires_exact_post_to_pre_entailment() -> None:
    validator = ProcedureCompositionValidator()
    first = _operator(_with_rollback(valid_spec(name="left"), "rollback-a"))
    second = _operator(_with_rollback(valid_spec(name="right"), "rollback-b"))
    request = _request(first.spec)
    decision = validator.validate((first, second), request)
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.MISSING_ENTAILMENT

    evidence = EntailmentEvidence(
        predecessor_cid=first.procedure_cid,
        successor_cid=second.procedure_cid,
        postcondition_id=first.spec.postconditions[0].condition_id,
        precondition_id=second.spec.preconditions[0].condition_id,
        evidence_cid="entailment-1",
        producer="entailment-producer@1",
        evidence_type="entailment-receipt@1",
    )
    admitted = validator.validate(
        (first, second),
        _request(first.spec, entailment_evidence=(evidence,)),
    )
    assert admitted.accepted is True
    assert admitted.reason_code is CompositionReason.ACCEPTED


def test_composition_rejects_incompatible_effects_and_authority() -> None:
    validator = ProcedureCompositionValidator()
    first = _operator(_predecessor("auth-a"))
    other_authority = replace(
        _successor("auth-b").authority,
        authority_policy_revision="authority-policy-other",
        risk_ceiling=RiskClass.AUTHORITY_OR_SECURITY,
    )
    second_spec = replace(_successor("auth-b"), authority=other_authority)
    second = _operator(second_spec)
    decision = validator.validate((first, second), _request(first.spec))
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.INCOMPATIBLE_AUTHORITY

    write = ProcedureEffect(
        effect_id="effect.write",
        effect_class=EffectClass.REPOSITORY_WRITE,
        targets=("ipfs_accelerate_py/agent_supervisor/example.py",),
        reversible=True,
    )
    write_spec = replace(
        _successor("effect-b"),
        declared_effects=_successor("effect-b").declared_effects + (write,),
        authority=replace(
            _successor("effect-b").authority,
            risk_ceiling=RiskClass.REPOSITORY_WRITE,
        ),
    )
    request = _request(
        first.spec,
        allowed_effect_classes=(EffectClass.VALIDATION, EffectClass.RECEIPT_EMIT),
    )
    effects = validator.validate((first, _operator(write_spec)), request)
    assert effects.accepted is False
    assert effects.reason_code in {
        CompositionReason.INCOMPATIBLE_EFFECTS,
        CompositionReason.HIDDEN_EFFECT_ESCALATION,
    }


def test_composition_rejects_environment_budget_rollback_and_validation() -> None:
    validator = ProcedureCompositionValidator()
    first = _operator(_predecessor("env-a"))
    other_env = replace(
        _successor("env-b"),
        bindings=replace(_successor("env-b").bindings, environment_id="other-environment"),
    )
    environment = validator.validate((first, _operator(other_env)), _request(first.spec))
    assert environment.reason_code is CompositionReason.INCOMPATIBLE_ENVIRONMENT

    budget = validator.validate(
        (_operator(_predecessor("budget-a")), _operator(_successor("budget-b"))),
        _request(first.spec, resource_budget=_predecessor("budget-a").resources),
    )
    assert budget.reason_code is CompositionReason.BUDGET_OVERFLOW

    no_rollback = validator.validate(
        (
            _operator(
                replace(
                    valid_spec(name="no-rollback-a"),
                    postconditions=valid_spec(name="no-rollback-a").postconditions
                    + (_handoff_post(),),
                )
            ),
            _operator(replace(valid_spec(name="no-rollback-b"), preconditions=(_handoff_pre(),))),
        ),
        _request(first.spec),
    )
    assert no_rollback.reason_code is CompositionReason.MISSING_COMPOSED_ROLLBACK

    incomplete = replace(
        _successor("incomplete-b"),
        validation=ProcedureValidationPlan(
            required_step_ids=_successor("incomplete-b").validation.required_step_ids,
            required_observation_ids=_successor(
                "incomplete-b"
            ).validation.required_observation_ids,
        ),
    )
    validation = validator.validate(
        (_operator(_predecessor("incomplete-a")), _operator(incomplete)),
        _request(first.spec),
    )
    assert validation.reason_code is CompositionReason.INCOMPLETE_VALIDATION


def test_composition_rejects_cycles_and_hidden_effect_escalation() -> None:
    validator = ProcedureCompositionValidator()
    first = _operator(_predecessor("cycle-a"))
    second = _operator(_successor("cycle-b"))
    cycled = validator.validate((first, second, first), _request(first.spec))
    assert cycled.accepted is False
    assert cycled.reason_code is CompositionReason.COMPOSITION_CYCLE

    write = ProcedureEffect(
        effect_id="effect.write",
        effect_class=EffectClass.REPOSITORY_WRITE,
        targets=("ipfs_accelerate_py/agent_supervisor/example.py",),
        reversible=True,
    )
    escalated = replace(
        _successor("escalate-b"),
        declared_effects=_successor("escalate-b").declared_effects + (write,),
        authority=replace(
            _successor("escalate-b").authority,
            risk_ceiling=RiskClass.REPOSITORY_WRITE,
        ),
    )
    hidden = validator.validate(
        (_operator(_predecessor("escalate-a")), _operator(escalated)),
        _request(first.spec),
    )
    assert hidden.accepted is False
    assert hidden.reason_code is CompositionReason.HIDDEN_EFFECT_ESCALATION


def test_qualified_probe_constructs_planner_without_calling_plan() -> None:
    compatibility = probe_adaptive_planner(importer=_available_importer)
    assert compatibility.available is True
    assert compatibility.planner_cls is _QualifiedPlanner
    assert compatibility.planner_cls() is not None
    with pytest.raises(AssertionError, match="must not reinterpret"):
        compatibility.planner_cls().plan()
