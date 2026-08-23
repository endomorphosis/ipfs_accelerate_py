from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactState,
    ConditionOperator,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureCertificate,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureValidationPlan,
    RiskClass,
    StepOperation,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.planner_adapter import (
    ADAPTER_REVISION,
    ADAPTIVE_PLANNER_HAMMER_BLOCKER,
    ADAPTIVE_PLANNER_MODULE,
    COMPOSITION_VALIDATOR_REVISION,
    HAMMER_TRACE_SCHEMA_SIGNATURE,
    OPERATOR_REVISION,
    PLANNER_OPERATOR_ORDER,
    REQUIRED_COMPOSITION_DIMENSIONS,
    REQUIRED_MATCH_DIMENSIONS,
    CompositionAction,
    CompositionReason,
    CompositionRequest,
    EntailmentEvidence,
    PlannerCompatibility,
    PlannerCompatibilityStatus,
    PlannerDispatchAction,
    PlannerDispatchReason,
    PlannerDispatchRequest,
    PlannerMatchAction,
    PlannerMatchReason,
    PlannerMatchRequest,
    PlannerOperatorKind,
    ProcedureClaimScope,
    ProcedureCompositionValidator,
    ProcedureOperator,
    ProcedurePlannerAdapter,
    compose_procedure_operators,
    match_procedure_operator,
    probe_adaptive_planner_compatibility,
    qualified_planner_compatibility,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.runtime import (
    compiler_capabilities,
)


LANGUAGE = "python"
FRAMEWORK = "stdlib"
REPOSITORY_FAMILY = "python-repair-family"
CRITERION_ID = "criterion.import-is-pure"


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
        "held_out_evaluation_cid": "held-out-1",
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


def _operator(
    spec: ProcedureSpec | None = None,
    *,
    claim_scope: ProcedureClaimScope = ProcedureClaimScope.TASK,
    claim_id: str = "task.import-purity",
    **certificate_changes: object,
) -> ProcedureOperator:
    spec = spec or valid_spec()
    return ProcedureOperator(
        bindings=spec.bindings,
        procedure=spec,
        certificate=_certificate(spec, **certificate_changes),
        claim_scope=claim_scope,
        claim_id=claim_id,
    )


def _match_request(
    spec: ProcedureSpec,
    *,
    claim_scope: ProcedureClaimScope = ProcedureClaimScope.TASK,
    claim_id: str = "task.import-purity",
    **changes: object,
) -> PlannerMatchRequest:
    values: dict[str, object] = {
        "bindings": spec.bindings,
        "task_family_id": spec.task_family_id,
        "claim_scope": claim_scope,
        "claim_id": claim_id,
        "language_classes": (LANGUAGE,),
        "framework_classes": (FRAMEWORK,),
        "effect_classes": tuple(item.effect_class for item in spec.declared_effects),
        "authority_policy_revision": spec.authority.authority_policy_revision,
        "authority_requirement_ids": spec.authority.requirement_ids,
        "required_capability_ids": spec.authority.required_capability_ids,
        "validation_contracts": (
            *spec.validation.required_test_contracts,
            *spec.validation.required_proof_contracts,
        ),
        "risk_ceiling": spec.authority.risk_ceiling,
        "scope_paths": spec.scope_paths,
        "repository_families": (REPOSITORY_FAMILY,),
    }
    values.update(changes)
    return PlannerMatchRequest(**values)


def _bridge_condition(*, condition_id: str, kind: str):
    payload = dict(
        condition_id=condition_id,
        binding="local:bridge-state",
        operator=ConditionOperator.ADMITTED,
        evidence_producer="bridge-checker@1",
        evidence_type="bridge-receipt@1",
    )
    if kind == "pre":
        return ProcedurePrecondition(**payload)
    return ProcedurePostcondition(**payload)


def _rollback() -> ProcedureRollback:
    return ProcedureRollback(
        rollback_id="rollback.restore",
        trigger_effect_ids=("effect.validation",),
        step_ids=("read",),
        verification_observation_ids=("observation.tests",),
        exact_target_cid="rollback-target-1",
    )


def _composable_spec(name: str, *, role: str) -> ProcedureSpec:
    spec = valid_spec(name=name)
    if role == "predecessor":
        postconditions = (
            _bridge_condition(condition_id="postcondition.bridge", kind="post"),
            *spec.postconditions,
        )
        preconditions = spec.preconditions
    else:
        preconditions = (
            _bridge_condition(condition_id="precondition.bridge", kind="pre"),
        )
        postconditions = spec.postconditions
    return replace(
        spec,
        preconditions=preconditions,
        postconditions=postconditions,
        rollback=(_rollback(),),
        resources=ProcedureResourceEnvelope(
            wall_time_ms=30_000,
            cpu_time_ms=30_000,
            memory_bytes=64_000_000,
            disk_bytes=64_000_000,
            model_token_limit=0,
            model_call_limit=0,
            subprocess_limit=2,
        ),
    )


def _entailment(predecessor: ProcedureOperator, successor: ProcedureOperator) -> EntailmentEvidence:
    return EntailmentEvidence(
        predecessor_procedure_cid=predecessor.procedure_cid,
        successor_procedure_cid=successor.procedure_cid,
        predecessor_postcondition_id="postcondition.bridge",
        successor_precondition_id="precondition.bridge",
        evidence_cid="entailment-bridge-1",
    )


def _union_effects(*operators: ProcedureOperator) -> tuple[EffectClass, ...]:
    result: list[EffectClass] = []
    for operator in operators:
        for effect in operator.procedure.declared_effects:
            if effect.effect_class not in result:
                result.append(effect.effect_class)
    return tuple(result)


def _union_authority(*operators: ProcedureOperator) -> ProcedureAuthorityEnvelope:
    first = operators[0].procedure.authority
    requirements: list[str] = []
    capabilities: list[str] = []
    operations: list[StepOperation] = []
    for operator in operators:
        authority = operator.procedure.authority
        for item in authority.requirement_ids:
            if item not in requirements:
                requirements.append(item)
        for item in authority.required_capability_ids:
            if item not in capabilities:
                capabilities.append(item)
        for item in authority.allowed_operations:
            if item not in operations:
                operations.append(item)
    return ProcedureAuthorityEnvelope(
        authority_policy_revision=first.authority_policy_revision,
        requirement_ids=tuple(requirements),
        required_capability_ids=tuple(capabilities),
        allowed_operations=tuple(operations),
        risk_ceiling=first.risk_ceiling,
        confirmation_required=any(
            operator.procedure.authority.confirmation_required for operator in operators
        ),
    )


def _sum_resources(*operators: ProcedureOperator) -> ProcedureResourceEnvelope:
    totals = {
        "wall_time_ms": 0,
        "cpu_time_ms": 0,
        "memory_bytes": 0,
        "disk_bytes": 0,
        "model_token_limit": 0,
        "model_call_limit": 0,
        "subprocess_limit": 0,
        "network_request_limit": 0,
    }
    for operator in operators:
        resources = operator.procedure.resources
        for name in totals:
            totals[name] += int(getattr(resources, name))
    return ProcedureResourceEnvelope(**totals)


def _union_validation(*operators: ProcedureOperator) -> ProcedureValidationPlan:
    steps: list[str] = []
    observations: list[str] = []
    tests: list[str] = []
    proofs: list[str] = []
    post_merge: list[str] = []
    for operator in operators:
        plan = operator.procedure.validation
        for item in plan.required_step_ids:
            if item not in steps:
                steps.append(item)
        for item in plan.required_observation_ids:
            if item not in observations:
                observations.append(item)
        for item in plan.required_test_contracts:
            if item not in tests:
                tests.append(item)
        for item in plan.required_proof_contracts:
            if item not in proofs:
                proofs.append(item)
        for item in plan.post_merge_validation_contracts:
            if item not in post_merge:
                post_merge.append(item)
    return ProcedureValidationPlan(
        required_step_ids=tuple(steps),
        required_observation_ids=tuple(observations),
        required_test_contracts=tuple(tests),
        required_proof_contracts=tuple(proofs),
        post_merge_validation_contracts=tuple(post_merge),
    )


def _composed_rollback(*operators: ProcedureOperator) -> tuple[ProcedureRollback, ...]:
    effects: list[str] = []
    steps: list[str] = []
    observations: list[str] = []
    for operator in operators:
        for rollback in operator.procedure.rollback:
            for item in rollback.trigger_effect_ids:
                if item not in effects:
                    effects.append(item)
            for item in rollback.step_ids:
                if item not in steps:
                    steps.append(item)
            for item in rollback.verification_observation_ids:
                if item not in observations:
                    observations.append(item)
    return (
        ProcedureRollback(
            rollback_id="rollback.composed",
            trigger_effect_ids=tuple(effects),
            step_ids=tuple(steps),
            verification_observation_ids=tuple(observations),
            exact_target_cid="rollback-target-composed",
        ),
    )


def _composition_request(
    *operators: ProcedureOperator,
    entailment: tuple[EntailmentEvidence, ...] | None = None,
    **changes: object,
) -> CompositionRequest:
    values: dict[str, object] = {
        "operators": operators,
        "entailment": entailment
        if entailment is not None
        else (_entailment(operators[0], operators[1]),),
        "composed_effects": _union_effects(*operators),
        "composed_authority": _union_authority(*operators),
        "composed_resources": _sum_resources(*operators),
        "composed_validation": _union_validation(*operators),
        "composed_rollback": _composed_rollback(*operators),
    }
    values.update(changes)
    return CompositionRequest(**values)


def _qualified_adapter() -> ProcedurePlannerAdapter:
    return ProcedurePlannerAdapter(compatibility_probe=qualified_planner_compatibility)


def _assert_other_runtime_usable() -> None:
    capabilities = compiler_capabilities()
    assert capabilities["parse_and_validate"] is True
    assert capabilities["deterministic_invoke"] is True
    spec = valid_spec()
    assert spec.content_id
    from ipfs_accelerate_py.agent_supervisor.procedure_compiler.registry import (
        ProcedureRegistry,
        REGISTRY_REVISION,
    )

    assert REGISTRY_REVISION == "ProcedureRegistry@1"
    assert ProcedureRegistry is not None


def test_adapter_import_does_not_bind_adaptive_planner() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.procedure_compiler.planner_adapter"
    )
    assert not hasattr(module, "AdaptivePlanner")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "from ..planning.adaptive_planner import" not in source
    assert "from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import" not in source
    _assert_other_runtime_usable()


def test_live_adaptive_planner_probe_qualifies_or_is_typed_unavailable() -> None:
    compatibility = probe_adaptive_planner_compatibility()
    assert compatibility.other_runtime_usable is True
    assert compatibility.module_name == ADAPTIVE_PLANNER_MODULE
    if compatibility.qualified:
        assert compatibility.status is PlannerCompatibilityStatus.QUALIFIED
        assert compatibility.planner_class_present is True
        assert compatibility.blocker == ""
        assert "AdaptivePlanner" in sys.modules.get(ADAPTIVE_PLANNER_MODULE).__dict__
    else:
        assert compatibility.status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE
        assert compatibility.reason_code == ADAPTIVE_PLANNER_HAMMER_BLOCKER
        assert compatibility.blocker == ADAPTIVE_PLANNER_HAMMER_BLOCKER
        assert compatibility.diagnostic == HAMMER_TRACE_SCHEMA_SIGNATURE
        assert HAMMER_TRACE_SCHEMA_SIGNATURE in compatibility.diagnostic
    _assert_other_runtime_usable()


def test_typed_unavailable_probe_blocks_dispatch_and_keeps_runtime_usable() -> None:
    spec = valid_spec()
    operator = _operator(spec)
    compatibility = PlannerCompatibility(
        status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
        reason_code=ADAPTIVE_PLANNER_HAMMER_BLOCKER,
        diagnostic=HAMMER_TRACE_SCHEMA_SIGNATURE,
        planner_class_present=False,
        blocker=ADAPTIVE_PLANNER_HAMMER_BLOCKER,
    )
    adapter = ProcedurePlannerAdapter(compatibility_probe=lambda: compatibility)
    decision = adapter.plan(
        PlannerDispatchRequest(
            match=_match_request(spec),
            operators=(operator,),
            composition=_composition_request(*_composed_operators()),
        )
    )
    assert decision.action is PlannerDispatchAction.UNAVAILABLE
    assert decision.reason_code is PlannerDispatchReason.ADAPTIVE_PLANNER_INCOMPATIBLE
    assert decision.compatibility_status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE
    assert decision.dispatched is False
    assert decision.procedure_cids == ()
    assert decision.selected_kind == ""
    assert decision.considered_kinds == ()
    assert decision.compatibility_reason_code == ADAPTIVE_PLANNER_HAMMER_BLOCKER
    assert decision.diagnostic == HAMMER_TRACE_SCHEMA_SIGNATURE
    assert decision.blocker == ADAPTIVE_PLANNER_HAMMER_BLOCKER
    assert decision.other_runtime_usable is True
    assert parse_procedure_artifact(decision.to_dict()) == decision
    _assert_other_runtime_usable()


def test_incompatible_import_blocks_procedure_dispatch_and_keeps_runtime_usable() -> None:
    spec = valid_spec()
    operator = _operator(spec)
    live = ProcedurePlannerAdapter()
    decision = live.plan(
        PlannerDispatchRequest(
            match=_match_request(spec),
            operators=(operator,),
        )
    )
    compatibility = live.probe_compatibility()
    if compatibility.qualified:
        assert decision.action is PlannerDispatchAction.CANDIDATES
        assert decision.dispatched is True
        assert decision.procedure_cids == (operator.procedure_cid,)
        assert decision.selected_kind == PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE.value
    else:
        assert decision.action is PlannerDispatchAction.UNAVAILABLE
        assert decision.reason_code is PlannerDispatchReason.ADAPTIVE_PLANNER_INCOMPATIBLE
        assert decision.dispatched is False
        assert decision.procedure_cids == ()
        assert decision.selected_kind == ""
        assert decision.compatibility_reason_code == ADAPTIVE_PLANNER_HAMMER_BLOCKER
        assert decision.diagnostic == HAMMER_TRACE_SCHEMA_SIGNATURE
        assert decision.blocker == ADAPTIVE_PLANNER_HAMMER_BLOCKER
    assert decision.other_runtime_usable is True
    _assert_other_runtime_usable()


def test_required_planner_order_is_closed_and_stable() -> None:
    assert PLANNER_OPERATOR_ORDER == (
        PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE,
        PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES,
        PlannerOperatorKind.DETERMINISTIC_BASELINE,
        PlannerOperatorKind.BOUNDED_LOCAL_SYNTHESIS,
        PlannerOperatorKind.SMALL_LOCAL_MODEL,
        PlannerOperatorKind.STANDARD_REMOTE_MODEL,
        PlannerOperatorKind.STRONG_REMOTE_MODEL,
        PlannerOperatorKind.HUMAN_ESCALATION,
    )
    assert _qualified_adapter().planner_operator_order == PLANNER_OPERATOR_ORDER


def _composed_operators() -> tuple[ProcedureOperator, ProcedureOperator]:
    predecessor = _operator(
        _composable_spec("procedure-a", role="predecessor"),
        claim_scope=ProcedureClaimScope.REPAIR_SUFFIX,
        claim_id="repair.suffix",
    )
    successor = _operator(
        _composable_spec("procedure-b", role="successor"),
        claim_scope=ProcedureClaimScope.REPAIR_SUFFIX,
        claim_id="repair.suffix",
    )
    return predecessor, successor


def test_qualified_planner_selects_exact_verified_procedure_before_composition() -> None:
    predecessor, successor = _composed_operators()
    exact = _operator(valid_spec())
    request = PlannerDispatchRequest(
        match=_match_request(exact.procedure),
        operators=(exact, predecessor, successor),
        composition=_composition_request(predecessor, successor),
    )
    decision = _qualified_adapter().plan(request)
    assert decision.action is PlannerDispatchAction.CANDIDATES
    assert decision.reason_code is PlannerDispatchReason.QUALIFIED_ORDER
    assert decision.selected_kind == PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE.value
    assert decision.procedure_cids == (exact.procedure_cid,)
    assert decision.dispatched is True
    assert tuple(decision.considered_kinds) == tuple(
        item.value for item in PLANNER_OPERATOR_ORDER
    )
    assert PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES.value in decision.considered_kinds
    assert parse_procedure_artifact(decision.to_dict()) == decision


def test_qualified_planner_uses_composition_only_when_no_exact_match() -> None:
    predecessor, successor = _composed_operators()
    request = PlannerDispatchRequest(
        match=_match_request(valid_spec()),
        operators=(),
        composition=_composition_request(predecessor, successor),
    )
    decision = _qualified_adapter().plan(request)
    assert decision.selected_kind == PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES.value
    assert decision.procedure_cids == (predecessor.procedure_cid, successor.procedure_cid)
    assert decision.dispatched is True


def test_qualified_planner_does_not_emit_procedures_for_later_ranks() -> None:
    unmatched = _operator(valid_spec(name="unrelated-procedure"))
    request = PlannerDispatchRequest(
        match=_match_request(valid_spec(), task_family_id="UNRELATED_FAMILY"),
        operators=(unmatched,),
    )
    decision = _qualified_adapter().plan(request)
    assert decision.selected_kind == PlannerOperatorKind.DETERMINISTIC_BASELINE.value
    assert decision.procedure_cids == ()
    assert decision.dispatched is False
    assert decision.considered_kinds[0] == PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE.value
    assert decision.considered_kinds[1] == PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES.value
    assert PlannerOperatorKind.STANDARD_REMOTE_MODEL.value in decision.considered_kinds
    assert decision.selected_kind != PlannerOperatorKind.STANDARD_REMOTE_MODEL.value
    assert decision.selected_kind != PlannerOperatorKind.STRONG_REMOTE_MODEL.value


def test_exact_match_requires_every_compatible_boundary() -> None:
    spec = valid_spec()
    operator = _operator(spec)
    decision = match_procedure_operator(_match_request(spec), operator)
    assert decision.action is PlannerMatchAction.MATCH
    assert decision.reason_code is PlannerMatchReason.EXACT_COMPATIBLE
    assert decision.matched is True
    assert decision.claims_task is True
    assert tuple(decision.compatible_dimensions) == REQUIRED_MATCH_DIMENSIONS or set(
        decision.compatible_dimensions
    ) == set(REQUIRED_MATCH_DIMENSIONS)
    assert decision.incompatible_dimensions == ()
    assert parse_procedure_artifact(decision.to_dict()) == decision
    assert parse_procedure_artifact(operator.to_dict()) == operator
    assert operator.can_authorize is False
    assert operator.can_promote is False
    assert operator.operator_revision == OPERATOR_REVISION
    assert operator.verified is True


@pytest.mark.parametrize(
    "scope,claim_id",
    (
        (ProcedureClaimScope.CRITERION, CRITERION_ID),
        (ProcedureClaimScope.SUBGOAL, "subgoal.restore-import"),
        (ProcedureClaimScope.REPAIR_SUFFIX, "repair.suffix"),
        (ProcedureClaimScope.VALIDATION_STAGE, "validation.focused-tests"),
    ),
)
def test_partial_criterion_matches_without_claiming_the_task(
    scope: ProcedureClaimScope, claim_id: str
) -> None:
    spec = valid_spec()
    operator = _operator(spec, claim_scope=scope, claim_id=claim_id)
    decision = match_procedure_operator(
        _match_request(spec, claim_scope=scope, claim_id=claim_id),
        operator,
    )
    assert decision.matched is True
    assert decision.reason_code is PlannerMatchReason.PARTIAL_CRITERION
    assert decision.claim_scope is scope
    assert decision.claims_task is False
    task_decision = match_procedure_operator(_match_request(spec), operator)
    assert task_decision.matched is False
    assert task_decision.claims_task is False
    assert "claim" in task_decision.incompatible_dimensions
    assert task_decision.reason_code is PlannerMatchReason.CLAIM_INCOMPATIBLE


@pytest.mark.parametrize(
    ("change", "dimension", "reason"),
    (
        (
            lambda spec: {"task_family_id": "OTHER_FAMILY"},
            "task_family",
            PlannerMatchReason.TASK_FAMILY_INCOMPATIBLE,
        ),
        (
            lambda spec: {"bindings": replace(spec.bindings, repository_id="repo-other")},
            "repository",
            PlannerMatchReason.REPOSITORY_INCOMPATIBLE,
        ),
        (
            lambda spec: {"bindings": replace(spec.bindings, tree_id="tree-other")},
            "tree",
            PlannerMatchReason.TREE_INCOMPATIBLE,
        ),
        (
            lambda spec: {"bindings": replace(spec.bindings, policy_revision="policy-other")},
            "policy",
            PlannerMatchReason.POLICY_INCOMPATIBLE,
        ),
        (
            lambda spec: {
                "bindings": replace(spec.bindings, environment_id="environment-other")
            },
            "environment",
            PlannerMatchReason.ENVIRONMENT_INCOMPATIBLE,
        ),
        (
            lambda spec: {"language_classes": ("rust",)},
            "language",
            PlannerMatchReason.LANGUAGE_INCOMPATIBLE,
        ),
        (
            lambda spec: {"framework_classes": ("django",)},
            "framework",
            PlannerMatchReason.FRAMEWORK_INCOMPATIBLE,
        ),
        (
            lambda spec: {"effect_classes": (EffectClass.MERGE,)},
            "effect",
            PlannerMatchReason.EFFECT_INCOMPATIBLE,
        ),
        (
            lambda spec: {"authority_requirement_ids": ("authority.other",)},
            "authority",
            PlannerMatchReason.AUTHORITY_INCOMPATIBLE,
        ),
        (
            lambda spec: {"validation_contracts": ("other-tests@1",)},
            "validation",
            PlannerMatchReason.VALIDATION_INCOMPATIBLE,
        ),
        (
            lambda spec: {"risk_ceiling": RiskClass.AUTHORITY_OR_SECURITY},
            "risk",
            PlannerMatchReason.RISK_INCOMPATIBLE,
        ),
        (
            lambda spec: {"scope_paths": ("docs/architecture",)},
            "scope",
            PlannerMatchReason.SCOPE_INCOMPATIBLE,
        ),
    ),
)
def test_partial_or_near_boundary_is_rejected(change, dimension, reason) -> None:
    spec = valid_spec()
    operator = _operator(spec)
    request = _match_request(spec, **change(spec))
    decision = match_procedure_operator(request, operator)
    assert decision.matched is False
    assert decision.action is PlannerMatchAction.REJECT
    assert dimension in decision.incompatible_dimensions
    if dimension == "task_family":
        assert decision.reason_code is reason
    elif dimension in {"repository", "tree", "policy", "environment"}:
        assert decision.reason_code in {reason, PlannerMatchReason.NEAR_MATCH_REJECTED}
        if decision.reason_code is PlannerMatchReason.NEAR_MATCH_REJECTED:
            assert "task_family" in decision.compatible_dimensions
    else:
        assert decision.reason_code in {reason, PlannerMatchReason.NEAR_MATCH_REJECTED}


def test_unverified_certificate_cannot_match() -> None:
    spec = valid_spec()
    operator = _operator(spec, state=ArtifactState.REJECTED)
    decision = match_procedure_operator(_match_request(spec), operator)
    assert decision.matched is False
    assert decision.reason_code is PlannerMatchReason.CERTIFICATE_UNVERIFIED
    assert "certificate" in decision.incompatible_dimensions


def test_compatible_composition_requires_every_declared_dimension() -> None:
    predecessor, successor = _composed_operators()
    decision = compose_procedure_operators(_composition_request(predecessor, successor))
    assert decision.action is CompositionAction.ACCEPT
    assert decision.accepted is True
    assert decision.reason_code is CompositionReason.COMPATIBLE
    assert set(decision.compatible_dimensions) == set(REQUIRED_COMPOSITION_DIMENSIONS)
    assert decision.incompatible_dimensions == ()
    assert decision.validator_revision == COMPOSITION_VALIDATOR_REVISION
    assert parse_procedure_artifact(decision.to_dict()) == decision


def test_missing_or_inexact_entailment_is_rejected() -> None:
    predecessor, successor = _composed_operators()
    missing = compose_procedure_operators(
        _composition_request(predecessor, successor, entailment=())
    )
    assert missing.accepted is False
    assert missing.reason_code is CompositionReason.ENTAILMENT_MISSING
    assert "entailment" in missing.incompatible_dimensions

    inexact = compose_procedure_operators(
        _composition_request(
            predecessor,
            successor,
            entailment=(
                EntailmentEvidence(
                    predecessor_procedure_cid=predecessor.procedure_cid,
                    successor_procedure_cid=successor.procedure_cid,
                    predecessor_postcondition_id="postcondition.tests-admitted",
                    successor_precondition_id="precondition.bridge",
                    evidence_cid="entailment-wrong-1",
                ),
            ),
        )
    )
    assert inexact.accepted is False
    assert inexact.reason_code is CompositionReason.ENTAILMENT_INEXACT


def test_hidden_effect_escalation_is_rejected() -> None:
    predecessor, successor = _composed_operators()
    decision = compose_procedure_operators(
        _composition_request(
            predecessor,
            successor,
            composed_effects=(EffectClass.VALIDATION, EffectClass.RECEIPT_EMIT, EffectClass.MERGE),
        )
    )
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.HIDDEN_EFFECT_ESCALATION
    assert "effect" in decision.incompatible_dimensions


def test_omitted_component_effect_is_incompatible_not_hidden_escalation() -> None:
    predecessor, successor = _composed_operators()
    decision = compose_procedure_operators(
        _composition_request(
            predecessor,
            successor,
            composed_effects=(EffectClass.VALIDATION,),
        )
    )
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.EFFECT_INCOMPATIBLE
    assert "effect" in decision.incompatible_dimensions


def test_invalid_composition_does_not_dispatch_procedures() -> None:
    predecessor, successor = _composed_operators()
    request = PlannerDispatchRequest(
        match=_match_request(valid_spec()),
        operators=(),
        composition=_composition_request(predecessor, successor, entailment=()),
    )
    decision = _qualified_adapter().plan(request)
    assert decision.selected_kind == PlannerOperatorKind.DETERMINISTIC_BASELINE.value
    assert decision.procedure_cids == ()
    assert decision.dispatched is False
    assert decision.action is PlannerDispatchAction.CANDIDATES
    assert decision.other_runtime_usable is True


def test_hidden_authority_escalation_is_rejected() -> None:
    predecessor, successor = _composed_operators()
    authority = replace(
        _union_authority(predecessor, successor),
        confirmation_required=True,
        risk_ceiling=RiskClass.AUTHORITY_OR_SECURITY,
    )
    decision = compose_procedure_operators(
        _composition_request(predecessor, successor, composed_authority=authority)
    )
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.HIDDEN_AUTHORITY_ESCALATION
    assert "authority" in decision.incompatible_dimensions


def test_budget_must_be_additive_and_bounded() -> None:
    predecessor, successor = _composed_operators()
    short = replace(_sum_resources(predecessor, successor), wall_time_ms=1)
    decision = compose_procedure_operators(
        _composition_request(predecessor, successor, composed_resources=short)
    )
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.BUDGET_INCOMPATIBLE
    assert "budget" in decision.incompatible_dimensions


def test_incomplete_rollback_or_validation_is_rejected() -> None:
    predecessor, successor = _composed_operators()
    rollback = compose_procedure_operators(
        _composition_request(predecessor, successor, composed_rollback=())
    )
    assert rollback.accepted is False
    assert rollback.reason_code is CompositionReason.ROLLBACK_INCOMPLETE

    validation = compose_procedure_operators(
        _composition_request(
            predecessor,
            successor,
            composed_validation=ProcedureValidationPlan(
                required_step_ids=("tests",),
                required_observation_ids=("observation.tests",),
            ),
        )
    )
    assert validation.accepted is False
    assert validation.reason_code is CompositionReason.VALIDATION_INCOMPLETE


def test_composition_cycle_is_rejected() -> None:
    predecessor, _successor = _composed_operators()
    decision = ProcedureCompositionValidator().validate(
        _composition_request(
            predecessor,
            predecessor,
            entailment=(
                EntailmentEvidence(
                    predecessor_procedure_cid=predecessor.procedure_cid,
                    successor_procedure_cid=predecessor.procedure_cid,
                    predecessor_postcondition_id="postcondition.bridge",
                    successor_precondition_id="precondition.current-tree",
                    evidence_cid="cycle-1",
                ),
            ),
        )
    )
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.CYCLE_REJECTED
    assert "acyclicity" in decision.incompatible_dimensions


def test_environment_mismatch_rejects_composition() -> None:
    predecessor, _successor = _composed_operators()
    shifted = replace(
        _composable_spec("procedure-b", role="successor"),
        bindings=replace(valid_spec().bindings, environment_id="environment-other"),
    )
    successor = ProcedureOperator(
        bindings=shifted.bindings,
        procedure=shifted,
        certificate=_certificate(
            shifted,
            authority_policy_revision=shifted.bindings.policy_revision,
        ),
        claim_scope=ProcedureClaimScope.TASK,
        claim_id="task.import-purity",
    )
    decision = compose_procedure_operators(_composition_request(predecessor, successor))
    assert decision.accepted is False
    assert decision.reason_code is CompositionReason.ENVIRONMENT_INCOMPATIBLE
    assert "environment" in decision.incompatible_dimensions


def test_operator_and_decisions_round_trip_without_authority() -> None:
    spec = valid_spec()
    operator = _operator(spec)
    assert operator.to_dict()["claim_scope"] == "task"
    assert "can_authorize" not in operator.to_dict()
    again = ProcedureOperator.from_dict(operator.to_dict())
    assert again == operator
    assert again.can_complete is False
    adapter = _qualified_adapter()
    match = adapter.match(_match_request(spec), operator)
    assert match.adapter_revision == ADAPTER_REVISION
    assert parse_procedure_artifact(match.to_dict()) == match
    live = ProcedurePlannerAdapter()
    unavailable = live.plan(
        PlannerDispatchRequest(match=_match_request(spec), operators=(operator,))
    )
    if unavailable.action is PlannerDispatchAction.UNAVAILABLE:
        assert parse_procedure_artifact(unavailable.to_dict()) == unavailable
        assert unavailable.other_runtime_usable is True
        assert unavailable.procedure_cids == ()
