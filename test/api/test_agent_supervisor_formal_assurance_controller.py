"""FACP-052: bounded supervisor controller synthesis and validation."""

from __future__ import annotations

import importlib

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_assurance_orchestrator import (
    PROHIBITED_ASSURANCE_STAGES,
    EscalationStage,
)
from ipfs_accelerate_py.agent_supervisor.runtime.formal_assurance_controller import (
    ANALYZER_VERSION,
    BUNDLE,
    CONTROL_EFFECTS,
    CORE_SCHEMA,
    GOAL_ID,
    INTERFACE,
    NORMATIVE_PINS,
    POLICY_SCHEMA,
    REACTIVE_EVIDENCE,
    RESULT_SCHEMA,
    SCHEMA,
    SPEC_SCHEMA,
    TASK_ID,
    TOOLCHAIN_ID,
    AuthorityClass,
    ControlEffect,
    ControllerBounds,
    ControllerError,
    ControllerMode,
    ControllerObservation,
    ControllerPolicy,
    ControllerSpec,
    ControllerTransition,
    ControllerVerdict,
    EvidenceClass,
    FormalAssuranceController,
    HardProperty,
    HardPropertyId,
    ReversibilityClass,
    SoftObjectiveId,
    UnrealizableCore,
    build_default_grammar,
    check_hard_properties,
    default_hard_properties,
    default_realizable_spec,
    explain_unrealizable,
    synthesize_controller,
    synthesize_or_validate,
    validate_controller,
)


def _spec(**overrides: object) -> ControllerSpec:
    base = default_realizable_spec().to_dict()
    base.update(overrides)
    return ControllerSpec.from_mapping(base)


def test_evidence_envelope_is_stable() -> None:
    assert TASK_ID == "FACP-052"
    assert GOAL_ID == "FACP-G720"
    assert BUNDLE == "facp/synthesis/controller"
    assert SCHEMA == "facp/supervisor-controller@1"
    assert REACTIVE_EVIDENCE == "facp/reactive-controller@1"
    assert CORE_SCHEMA == "facp/unrealizable-core@1"
    assert SPEC_SCHEMA == "facp/controller-spec@1"
    assert POLICY_SCHEMA == "facp/controller-policy@1"
    assert RESULT_SCHEMA == "facp/controller-result@1"
    assert INTERFACE == "FormalAssuranceController@1"
    assert ANALYZER_VERSION
    assert TOOLCHAIN_ID.endswith(ANALYZER_VERSION)
    assert "norm:eak-typestate" in NORMATIVE_PINS
    assert "norm:terminal-safety-statement" in NORMATIVE_PINS
    assert "norm:promotion-predicates" in NORMATIVE_PINS
    assert "norm:non-implications" in NORMATIVE_PINS
    assert "llm" in PROHIBITED_ASSURANCE_STAGES
    assert "route_provider" in CONTROL_EFFECTS
    assert "retry" in CONTROL_EFFECTS
    assert "shutdown" in CONTROL_EFFECTS


def test_cold_import_is_hermetic() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.runtime.formal_assurance_controller"
    )
    assert module.TASK_ID == "FACP-052"
    assert callable(module.synthesize_controller)
    assert callable(module.validate_controller)


def test_synthesize_realizable_default_spec() -> None:
    result = synthesize_controller(default_realizable_spec())
    assert result.verdict is ControllerVerdict.REALIZED
    assert result.realized is True
    assert result.policy is not None
    assert result.policy.schema == POLICY_SCHEMA
    assert result.policy.initial_mode == "idle"
    assert result.unrealizable_core is None
    assert SCHEMA in result.evidence
    assert REACTIVE_EVIDENCE in result.evidence
    assert result.toolchain == TOOLCHAIN_ID
    assert set(result.assumptions) >= {"assumption:deps-healthy"}
    assert all(result.hard_property_results.values())
    payload = result.to_dict()
    assert payload["task_id"] == TASK_ID
    assert payload["goal_id"] == GOAL_ID
    assert payload["bundle"] == BUNDLE
    assert payload["interface"] == INTERFACE
    assert payload["policy"]["bounds"]["max_retries"] == 2
    assert SoftObjectiveId.MINIMIZE_COST.value in result.soft_scores


def test_validate_matching_policy() -> None:
    synthesized = synthesize_controller(default_realizable_spec())
    assert synthesized.policy is not None
    validated = validate_controller(default_realizable_spec(), synthesized.policy)
    assert validated.verdict is ControllerVerdict.REALIZED
    assert validated.policy is not None
    assert set(validated.policy.discharged_properties) == set(
        synthesized.hard_property_results
    )


def test_validate_rejects_weakened_hard_properties() -> None:
    synthesized = synthesize_controller(default_realizable_spec())
    assert synthesized.policy is not None
    weakened = ControllerPolicy(
        policy_id="provisional",
        initial_mode=synthesized.policy.initial_mode,
        transitions=synthesized.policy.transitions,
        bounds=synthesized.policy.bounds,
        hard_properties=(
            HardProperty(
                HardPropertyId.BOUNDED_RETRY,
                "retries <= 2",
                bound=2,
            ),
        ),
        soft_objectives=synthesized.policy.soft_objectives,
        assumptions=synthesized.policy.assumptions,
    )
    result = validate_controller(default_realizable_spec(), weakened)
    assert result.verdict is ControllerVerdict.REJECTED
    assert result.hard_property_results[HardPropertyId.NO_WEAKEN_HARD_SAFETY.value] is False


def test_waivable_hard_property_is_rejected() -> None:
    with pytest.raises(ControllerError) as exc:
        HardProperty(
            HardPropertyId.BOUNDED_RETRY,
            "retries <= 2",
            bound=2,
            waivable=True,
        )
    assert exc.value.code.value == "weakened_hard_property"


def test_bounded_retry_accepts_within_budget_and_rejects_overflow() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None
    policy = synthesized.policy

    ok = controller.step(
        policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            lease_held=True,
            retry_count=0,
            reversibility=ReversibilityClass.REVERSIBLE.value,
        ),
        proposed_effect=ControlEffect.RETRY,
    )
    assert ok.accepted is True
    assert ok.next_mode == "leased"

    overflow = controller.step(
        policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            lease_held=True,
            retry_count=policy.bounds.max_retries,
            reversibility=ReversibilityClass.REVERSIBLE.value,
        ),
        proposed_effect=ControlEffect.RETRY,
    )
    assert overflow.accepted is False
    assert overflow.code == "bound_exceeded"
    assert overflow.invariant == HardPropertyId.BOUNDED_RETRY.value


def test_bounded_parallelism_rejects_overflow() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    verdict = controller.step(
        synthesized.policy,
        mode="leased",
        observation=ControllerObservation(
            mode="leased",
            lease_held=True,
            parallel_count=synthesized.policy.bounds.max_parallel + 1,
        ),
        proposed_effect=ControlEffect.ROUTE_PROVIDER,
    )
    assert verdict.accepted is False
    assert verdict.code == "bound_exceeded"
    assert verdict.invariant == HardPropertyId.BOUNDED_PARALLELISM.value


def test_no_blind_unknown_irreversible_retry() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    verdict = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            lease_held=True,
            retry_count=0,
            reversibility=ReversibilityClass.IRREVERSIBLE.value,
            unknown_pending=True,
            effect_count=1,
        ),
        proposed_effect=ControlEffect.RETRY,
    )
    assert verdict.accepted is False
    assert (
        verdict.invariant
        == HardPropertyId.NO_BLIND_UNKNOWN_IRREVERSIBLE_RETRY.value
    )


def test_compensatable_unknown_may_compensate() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    verdict = controller.step(
        synthesized.policy,
        mode="unknown_pending",
        observation=ControllerObservation(
            mode="unknown_pending",
            lease_held=True,
            reversibility=ReversibilityClass.COMPENSATABLE.value,
            unknown_pending=True,
        ),
        proposed_effect=ControlEffect.COMPENSATE,
    )
    assert verdict.accepted is True
    assert verdict.next_mode == "compensating"


def test_fallback_preserves_authority_and_evidence_class() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    preserved = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            lease_held=True,
            authority_class=AuthorityClass.KERNEL_ADMITTED.value,
            evidence_class=EvidenceClass.HERMETIC.value,
            fallback_authority_class=AuthorityClass.PROPOSAL_ONLY.value,
            fallback_evidence_class=EvidenceClass.SIMULATED.value,
            parallel_count=1,
        ),
        proposed_effect=ControlEffect.FALLBACK_PROVIDER,
    )
    assert preserved.accepted is True

    promoted = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            lease_held=True,
            authority_class=AuthorityClass.PROPOSAL_ONLY.value,
            evidence_class=EvidenceClass.FIXTURE.value,
            fallback_authority_class=AuthorityClass.LIVE_OBSERVED.value,
            fallback_evidence_class=EvidenceClass.LIVE.value,
            parallel_count=1,
        ),
        proposed_effect=ControlEffect.FALLBACK_PROVIDER,
    )
    assert promoted.accepted is False
    assert promoted.code == "fallback_promotion"


def test_unrealizable_unbounded_retry_returns_explanatory_core() -> None:
    result = synthesize_controller(
        _spec(allow_unbounded_retry=True, spec_id="spec:unbounded-retry")
    )
    assert result.verdict is ControllerVerdict.UNREALIZABLE
    assert result.unrealizable is True
    assert result.unrealizable_core is not None
    assert result.unrealizable_core.schema == CORE_SCHEMA
    assert CORE_SCHEMA in result.evidence
    assert "spec.allow_unbounded_retry" in result.unrealizable_core.conflicting_requirements
    assert (
        HardPropertyId.BOUNDED_RETRY.value
        in result.unrealizable_core.conflicting_properties
    )
    assert "BoundedRetry" in result.unrealizable_core.explanation


def test_unrealizable_fallback_authority_promotion_core() -> None:
    core = explain_unrealizable(
        _spec(
            require_fallback_authority_promotion=True,
            spec_id="spec:fallback-auth",
        )
    )
    assert isinstance(core, UnrealizableCore)
    assert HardPropertyId.FALLBACK_PRESERVES_AUTHORITY.value in core.conflicting_properties
    assert "FallbackPreservesAuthority" in core.explanation


def test_unrealizable_required_and_forbidden_effect_overlap() -> None:
    result = synthesize_controller(
        _spec(
            required_effects=("retry",),
            forbidden_effects=("retry",),
            spec_id="spec:overlap",
        )
    )
    assert result.verdict is ControllerVerdict.UNREALIZABLE
    assert result.unrealizable_core is not None
    assert any(
        "retry" in item for item in result.unrealizable_core.conflicting_requirements
    )


def test_proof_escalation_monotone_and_llm_forbidden() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    ok = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            proof_stage=EscalationStage.SMT.value,
            proposed_proof_stage=EscalationStage.ALLOY.value,
        ),
        proposed_effect=ControlEffect.ESCALATE_PROOF,
    )
    assert ok.accepted is True
    assert ok.next_mode == "escalating"

    weaker = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            proof_stage=EscalationStage.LEAN.value,
            proposed_proof_stage=EscalationStage.SMT.value,
        ),
        proposed_effect=ControlEffect.ESCALATE_PROOF,
    )
    assert weaker.accepted is False
    assert weaker.invariant == HardPropertyId.PROOF_ESCALATION_MONOTONE.value

    llm = controller.step(
        synthesized.policy,
        mode="failed",
        observation=ControllerObservation(
            mode="failed",
            proof_stage=EscalationStage.SMT.value,
            proposed_proof_stage="llm",
        ),
        proposed_effect=ControlEffect.ESCALATE_PROOF,
    )
    assert llm.accepted is False
    assert llm.invariant == HardPropertyId.PROOF_ESCALATION_MONOTONE.value


def test_human_gate_required_for_irreversible_route() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    denied = controller.step(
        synthesized.policy,
        mode="human_gated",
        observation=ControllerObservation(
            mode="human_gated",
            lease_held=True,
            confirmation_present=False,
            reversibility=ReversibilityClass.IRREVERSIBLE.value,
            parallel_count=1,
        ),
        proposed_effect=ControlEffect.ROUTE_PROVIDER,
    )
    assert denied.accepted is False
    assert denied.invariant == HardPropertyId.HUMAN_GATE_BEFORE_IRREVERSIBLE.value

    allowed = controller.step(
        synthesized.policy,
        mode="human_gated",
        observation=ControllerObservation(
            mode="human_gated",
            lease_held=True,
            confirmation_present=True,
            confirmation_spent=False,
            reversibility=ReversibilityClass.IRREVERSIBLE.value,
            parallel_count=1,
        ),
        proposed_effect=ControlEffect.ROUTE_PROVIDER,
    )
    assert allowed.accepted is True


def test_safe_shutdown_blocks_effectful_actions() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    shutdown = controller.step(
        synthesized.policy,
        mode="idle",
        observation=ControllerObservation(mode="idle"),
        proposed_effect=ControlEffect.SHUTDOWN,
    )
    assert shutdown.accepted is True
    assert shutdown.next_mode == "shutting_down"

    blocked = controller.step(
        synthesized.policy,
        mode="leased",
        observation=ControllerObservation(
            mode="leased",
            lease_held=True,
            shutdown_latched=True,
            parallel_count=1,
        ),
        proposed_effect=ControlEffect.ROUTE_PROVIDER,
    )
    assert blocked.accepted is False
    assert blocked.code == "shutdown_latched"
    assert blocked.invariant == HardPropertyId.SAFE_SHUTDOWN.value


def test_soft_objectives_do_not_override_hard_fallback_rules() -> None:
    # Even when a cheaper/live provider is desirable, authority promotion fails.
    result = synthesize_controller(
        _spec(
            require_fallback_evidence_promotion=True,
            spec_id="spec:soft-vs-hard",
        )
    )
    assert result.verdict is ControllerVerdict.UNREALIZABLE
    assert result.unrealizable_core is not None
    assert (
        HardPropertyId.FALLBACK_PRESERVES_EVIDENCE_CLASS.value
        in result.unrealizable_core.conflicting_properties
    )


def test_no_success_without_observation() -> None:
    controller = FormalAssuranceController()
    synthesized = controller.synthesize(default_realizable_spec())
    assert synthesized.policy is not None

    verdict = controller.step(
        synthesized.policy,
        mode="awaiting_observation",
        observation=ControllerObservation(
            mode="awaiting_observation",
            lease_held=True,
            observed=False,
        ),
        proposed_effect=ControlEffect.SEAL_RECEIPT,
    )
    assert verdict.accepted is False
    assert verdict.invariant == HardPropertyId.NO_SUCCESS_WITHOUT_OBSERVATION.value

    sealed = controller.step(
        synthesized.policy,
        mode="awaiting_observation",
        observation=ControllerObservation(
            mode="awaiting_observation",
            lease_held=True,
            observed=True,
        ),
        proposed_effect=ControlEffect.SEAL_RECEIPT,
    )
    assert sealed.accepted is True
    assert sealed.next_mode == "terminal_success"


def test_unknown_effect_and_field_rejected() -> None:
    with pytest.raises(ControllerError):
        ControllerSpec.from_mapping(
            {
                "spec_id": "spec:bad-effect",
                "required_effects": ["launch_missiles"],
            }
        )
    with pytest.raises(ControllerError):
        ControllerObservation.from_mapping({"mode": "idle", "extra_field": 1})


def test_synthesize_or_validate_modes() -> None:
    synthesized = synthesize_or_validate(
        mode=ControllerMode.SYNTHESIZE,
        spec=default_realizable_spec(),
    )
    assert synthesized.realized is True
    assert synthesized.policy is not None

    validated = synthesize_or_validate(
        mode="validate",
        spec=default_realizable_spec(),
        policy=synthesized.policy,
    )
    assert validated.realized is True

    missing = synthesize_or_validate(
        mode=ControllerMode.VALIDATE,
        spec=default_realizable_spec(),
        policy=None,
    )
    assert missing.verdict is ControllerVerdict.INVALID_SPEC


def test_check_hard_properties_on_default_grammar() -> None:
    bounds = ControllerBounds()
    policy = ControllerPolicy(
        policy_id="provisional",
        initial_mode="idle",
        transitions=build_default_grammar(bounds),
        bounds=bounds,
        hard_properties=default_hard_properties(bounds),
    )
    results = check_hard_properties(policy)
    assert all(results.values())
    assert HardPropertyId.BOUNDED_RETRY.value in results
    assert HardPropertyId.SAFE_SHUTDOWN.value in results


def test_controller_facade_explain_and_check() -> None:
    controller = FormalAssuranceController(bounds=ControllerBounds(max_retries=1))
    core = controller.explain_unrealizable(
        _spec(allow_unbounded_retry=True, spec_id="spec:facade")
    )
    assert core.schema == CORE_SCHEMA
    synthesized = controller.synthesize(default_realizable_spec(bounds=controller.bounds))
    assert synthesized.policy is not None
    assert synthesized.policy.bounds.max_retries == 1
    checked = controller.check_hard_properties(synthesized.policy)
    assert all(checked.values())
