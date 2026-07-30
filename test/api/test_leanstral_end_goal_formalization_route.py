"""Evidence for FormalizedGoalDevelopmentRoute@1 (FVT-G025 / FVT-024).

Proves:

* prose cannot bypass formalization before Leanstral goal development;
* the untrusted provider receives only immutable selected goal, formula,
  assumption, vocabulary, and template identifiers;
* Leanstral cannot create/mutate formulas, source, assumptions, proof,
  commands, admission, or completion through this route; and
* timeout / unavailable / malformed responses fall back deterministically
  without stalling the supervisor.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_development_contracts import (
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
)
from ipfs_accelerate_py.agent_supervisor.proof.end_goal_development import (
    FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE,
    EndGoalDevelopmentError,
    FormalizationGateReason,
    FormalizedGoalDevelopmentRequest,
    FormalizedGoalDevelopmentResult,
    FormalizedGoalDevelopmentRoute,
    FormalizedGoalIdentifiers,
    FormalizedRouteStatus,
    build_formalized_leanstral_invocation,
    create_formalized_goal_development_route,
    extract_formalized_identifiers,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    CancellationToken,
    ProofProviderError,
    ProviderFailureCode,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_goal_development import (
    GoalDevelopmentFallbackReason,
    GoalDevelopmentResultStatus,
    GoalDevelopmentTemplate,
    LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
    LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA,
    LeanstralGoalDevelopmentProvider,
    LeanstralGoalDevelopmentProviderConfig,
)
from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
    AmbiguityStatus,
    AssumptionBinding,
    AssumptionClass,
    AuthorityCeiling,
    EndGoalInterpretation,
    EndGoalSpec,
    FormalGoal,
    PhraseProvenance,
    PropertyClass,
    QuantifierKind,
    ResourceBounds,
    SourceSpanBinding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _source(**overrides: Any) -> SourceSpanBinding:
    payload = {
        "tree_id": "tree:repo@abc",
        "source_ref_ids": ("source:lease.py",),
        "span_ids": ("span:claim",),
        "ast_scope_ids": ("scope:package", "scope:tests"),
        "snapshot_id": "snap:1",
    }
    payload.update(overrides)
    return SourceSpanBinding(**payload)


def _bounds(**overrides: Any) -> ResourceBounds:
    payload = {
        "wall_time_ms": 5_000,
        "memory_bytes": 64 * 1024 * 1024,
        "max_steps": 32,
        "max_depth": 8,
        "max_nodes": 64,
        "max_candidates": 16,
        "network_allowed": False,
    }
    payload.update(overrides)
    return ResourceBounds(**payload)


def _assumption(
    assumption_id: str = "assumption:token-order",
    *,
    assumption_class: AssumptionClass = AssumptionClass.MUST_PROVE,
) -> AssumptionBinding:
    return AssumptionBinding(
        assumption_id=assumption_id,
        assumption_class=assumption_class,
        kind="semantic",
        statement="tokens are totally ordered",
        source=_source(),
        authority=AuthorityCeiling.NONE,
        reviewable=True,
    )


def _interpretation(
    interpretation_id: str = "interp:exists-ready",
    *,
    property_class: PropertyClass = PropertyClass.EXISTENTIAL_REACHABILITY,
    selected: bool = True,
) -> EndGoalInterpretation:
    return EndGoalInterpretation(
        interpretation_id=interpretation_id,
        controlled_english="Some execution reaches ready.",
        property_class=property_class,
        quantifiers=(QuantifierKind.EXISTS, QuantifierKind.EVENTUALLY),
        current_state={"phase": "init"},
        target_state={"phase": "ready"},
        environment={"scheduler": "fair"},
        semantic_diff={"vs_invariant": "does not require all executions"},
        unresolved_fields=(),
        selected=selected,
    )


def _end_goal(**overrides: Any) -> EndGoalSpec:
    payload: dict[str, Any] = {
        "goal_id": "goal:lease-ready",
        "root_goal_id": "goal:lease-ready",
        "caller_text": "the system reaches ready",
        "source": _source(),
        "property_class": PropertyClass.EXISTENTIAL_REACHABILITY,
        "quantifiers": (QuantifierKind.EXISTS, QuantifierKind.EVENTUALLY),
        "actors": ("scheduler", "worker"),
        "state_variables": ("phase", "owner"),
        "current_state": {"phase": "init"},
        "target_state": {"phase": "ready"},
        "transitions": ("claim", "release"),
        "environment": {"network": "async"},
        "interference": {"preempt": True},
        "assumptions": (
            _assumption(),
            _assumption(
                "assumption:fair-scheduler",
                assumption_class=AssumptionClass.TRUSTED,
            ),
        ),
        "logic_family": "temporal.ltl",
        "provider_ids": ("provider:z3",),
        "assurance_target": AuthorityCeiling.BOUNDED,
        "bounds": _bounds(),
        "provenance": (
            PhraseProvenance(
                phrase="reaches ready",
                clause_id="clause:target-ready",
                source_ref_ids=("source:prompt",),
                span_ids=("span:prompt-1",),
                start_offset=11,
                end_offset=24,
            ),
        ),
        "interpretations": (_interpretation("interp:exists-ready", selected=True),),
        "ambiguity_status": AmbiguityStatus.RESOLVED,
        "unsupported_semantics": (),
        "translation_loss": (),
        "acceptance_evidence": ("evidence:tests", "evidence:review"),
        "expected_receipt_classes": ("proof-receipt", "counterexample"),
        "status": "confirmed",
        "authority": AuthorityCeiling.DECLARATIVE,
        "proof_claimed": False,
        "completion_claimed": False,
    }
    payload.update(overrides)
    return EndGoalSpec(**payload)


def _formal_goal(end_goal: EndGoalSpec | None = None, **overrides: Any) -> FormalGoal:
    goal = end_goal or _end_goal()
    payload = {
        "formal_goal_id": "formal:lease-ready",
        "end_goal": goal,
        "selected_interpretation_id": "interp:exists-ready",
        "confirmation_receipt_id": "receipt:confirm-1",
        "status": "confirmed",
        "authority": AuthorityCeiling.DECLARATIVE,
        "proof_claimed": False,
        "completion_claimed": False,
    }
    payload.update(overrides)
    return FormalGoal(**payload)


def _policy(**changes: Any) -> GoalDevelopmentPolicy:
    values = {
        "mode": GoalDevelopmentMode.SHADOW,
        "max_depth": 3,
        "max_breadth": 3,
        "max_proposals": 8,
        "max_bytes": 32_768,
        "max_tokens": 2_048,
    }
    values.update(changes)
    return GoalDevelopmentPolicy(**values)


def _templates() -> tuple[GoalDevelopmentTemplate, ...]:
    return (
        GoalDevelopmentTemplate(
            template_id="template:implementation@1",
            satisfaction_formula_id="formula:reviewed-implementation",
            evidence_requirement_ids=("evidence:tests",),
            assurance_ids=("assurance:typed",),
            resource_ids=("resource:codex",),
            scope_ids=("scope:package",),
            validation_check_ids=("check:pytest",),
        ),
        GoalDevelopmentTemplate(
            template_id="template:validation@1",
            satisfaction_formula_id="formula:reviewed-validation",
            evidence_requirement_ids=("evidence:review",),
            assurance_ids=("assurance:review",),
            resource_ids=("resource:reviewer",),
            scope_ids=("scope:tests",),
            validation_check_ids=("check:review",),
        ),
    )


def _route_request(**overrides: Any) -> FormalizedGoalDevelopmentRequest:
    values: dict[str, Any] = {
        "formal_goal": _formal_goal(),
        "policy": _policy(),
        "templates": _templates(),
        "resource_budget": ResourceBudget(
            wall_time_ms=5_000,
            model_token_limit=1_024,
            max_output_bytes=64 * 1024,
        ),
        "network_allowed": False,
    }
    values.update(overrides)
    return FormalizedGoalDevelopmentRequest(**values)


def _proposal(
    proposal_id: str,
    *,
    parent_id: str = "goal:lease-ready",
    template_id: str = "template:implementation@1",
    depends_on: tuple[str, ...] = (),
) -> dict[str, Any]:
    if template_id == "template:implementation@1":
        evidence = ["evidence:tests"]
        assurance = ["assurance:typed"]
        resources = ["resource:codex"]
        scopes = ["scope:package"]
        checks = ["check:pytest"]
    else:
        evidence = ["evidence:review"]
        assurance = ["assurance:review"]
        resources = ["resource:reviewer"]
        scopes = ["scope:tests"]
        checks = ["check:review"]
    return {
        "proposal_id": proposal_id,
        "parent_id": parent_id,
        "template_id": template_id,
        "title": f"Develop {proposal_id}",
        "evidence_requirement_ids": evidence,
        "assurance_ids": assurance,
        "resource_ids": resources,
        "scope_ids": scopes,
        "validation_check_ids": checks,
        "depends_on": list(depends_on),
    }


def _provider_output(request_id: str, proposals: list[dict[str, Any]] | None = None) -> str:
    if proposals is None:
        proposals = [
            _proposal("subgoal:implementation"),
            _proposal(
                "subgoal:validation",
                parent_id="subgoal:implementation",
                template_id="template:validation@1",
                depends_on=("subgoal:implementation",),
            ),
        ]
    return json.dumps(
        {
            "schema": LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA,
            "operation": LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
            "request_id": request_id,
            "proposals": proposals,
        },
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Interface and formalization gate
# ---------------------------------------------------------------------------


def test_route_interface_and_capability_boundary() -> None:
    route = create_formalized_goal_development_route(
        llm_generate=lambda *_a, **_k: ""
    )
    capability = route.capabilities()

    assert route.interface == FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE
    assert capability["interface"] == FORMALIZED_GOAL_DEVELOPMENT_ROUTE_INTERFACE
    assert capability["requires_confirmed_formal_goal"] is True
    assert capability["prose_bypass_allowed"] is False
    assert capability["exposes_only_immutable_identifiers"] is True
    assert capability["can_create_formulas"] is False
    assert capability["can_mutate_assumptions"] is False
    assert capability["can_mutate_source"] is False
    assert capability["can_execute_commands"] is False
    assert capability["can_claim_admission"] is False
    assert capability["can_claim_completion"] is False
    assert capability["can_claim_proof"] is False
    assert capability["operation"] == LEANSTRAL_GOAL_DEVELOPMENT_OPERATION


@pytest.mark.parametrize(
    "prose_payload",
    [
        "the system reaches ready",
        {"prose": "the system reaches ready"},
        {"caller_text": "the system reaches ready", "goal_id": "goal:x"},
        {
            "prompt": "prove the system ready",
            "natural_language": "ready state",
        },
        {
            "goal_id": "goal:draft",
            "caller_text": "draft prose goal",
            "status": "draft",
            "source": {"tree_id": "tree:x"},
        },
    ],
)
def test_prose_cannot_bypass_formalization(prose_payload: Any) -> None:
    calls: list[Any] = []

    def generate(*_args: Any, **_kwargs: Any) -> str:
        calls.append(True)
        return "unused"

    route = FormalizedGoalDevelopmentRoute(llm_generate=generate)
    if isinstance(prose_payload, str):
        request = _route_request(formal_goal=prose_payload)
    elif "prose" in prose_payload and len(prose_payload) == 1:
        request = _route_request(formal_goal=None, prose=prose_payload["prose"])
    else:
        request = _route_request(formal_goal=prose_payload)

    result = route.develop(request)

    assert result.status is FormalizedRouteStatus.REJECTED
    assert result.gate_reason is FormalizationGateReason.PROSE_BYPASS
    assert result.provider_result is None
    assert result["authoritative"] is False
    assert result["admitted"] is False
    assert result["complete"] is False
    assert calls == []


def test_missing_formal_goal_is_rejected_without_model_call() -> None:
    calls: list[Any] = []
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: calls.append(True) or ""
    )

    result = route.develop(_route_request(formal_goal=None))

    assert result.status is FormalizedRouteStatus.REJECTED
    assert result.gate_reason is FormalizationGateReason.MISSING_FORMAL_GOAL
    assert calls == []


def test_unconfirmed_formal_goal_is_rejected() -> None:
    calls: list[Any] = []
    formal = _formal_goal(status="draft")
    # FormalGoal constructor may reject non-confirmed status depending on
    # contract version; if construction succeeds, the route must still gate.
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: calls.append(True) or ""
    )
    result = route.develop(_route_request(formal_goal=formal))

    assert result.rejected
    assert result.gate_reason in {
        FormalizationGateReason.UNCONFIRMED,
        FormalizationGateReason.INVALID_FORMAL_GOAL,
    }
    assert calls == []


def test_ambiguity_requiring_selection_cannot_reach_leanstral() -> None:
    calls: list[Any] = []
    with pytest.raises(Exception):
        # FormalGoal itself fails closed on REQUIRES_SELECTION.
        _formal_goal(
            end_goal=_end_goal(
                interpretations=(
                    _interpretation("interp:a", selected=False),
                    _interpretation(
                        "interp:b",
                        property_class=PropertyClass.INVARIANCE,
                        selected=False,
                    ),
                ),
                ambiguity_status=AmbiguityStatus.REQUIRES_SELECTION,
            )
        )

    # Also reject a raw mapping that tries to smuggle unresolved ambiguity.
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: calls.append(True) or ""
    )
    smuggled = {
        "formal_goal_id": "formal:ambiguous",
        "selected_interpretation_id": "interp:a",
        "status": "confirmed",
        "confirmation_receipt_id": "receipt:x",
        "end_goal": {
            "goal_id": "goal:ambiguous",
            "root_goal_id": "goal:ambiguous",
            "caller_text": "ready?",
            "source": {
                "tree_id": "tree:repo",
                "source_ref_ids": ["source:a"],
                "span_ids": ["span:a"],
                "ast_scope_ids": ["scope:package"],
            },
            "property_class": "existential_reachability",
            "assumptions": [],
            "acceptance_evidence": ["evidence:tests"],
            "status": "confirmed",
            "ambiguity_status": "requires_selection",
            "interpretations": [
                {
                    "interpretation_id": "interp:a",
                    "controlled_english": "A",
                    "property_class": "existential_reachability",
                    "selected": False,
                },
                {
                    "interpretation_id": "interp:b",
                    "controlled_english": "B",
                    "property_class": "invariance",
                    "selected": False,
                },
            ],
            "proof_claimed": False,
            "completion_claimed": False,
        },
        "proof_claimed": False,
        "completion_claimed": False,
    }
    result = route.develop(_route_request(formal_goal=smuggled))
    assert result.rejected
    assert result.gate_reason in {
        FormalizationGateReason.AMBIGUITY_UNRESOLVED,
        FormalizationGateReason.INVALID_FORMAL_GOAL,
        FormalizationGateReason.UNCONFIRMED,
    }
    assert calls == []


def test_missing_templates_rejected_before_model() -> None:
    calls: list[Any] = []
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: calls.append(True) or ""
    )
    result = route.develop(_route_request(templates=()))
    assert result.gate_reason is FormalizationGateReason.MISSING_TEMPLATES
    assert calls == []


# ---------------------------------------------------------------------------
# Identifier projection and provider exposure
# ---------------------------------------------------------------------------


def test_extract_identifiers_are_immutable_and_prose_free() -> None:
    request = _route_request()
    identifiers = extract_formalized_identifiers(request)

    assert isinstance(identifiers, FormalizedGoalIdentifiers)
    assert identifiers.formal_goal_id == "formal:lease-ready"
    assert identifiers.root_goal_id == "goal:lease-ready"
    assert identifiers.satisfaction_formula_id.startswith("formula:")
    assert "assumption:token-order" in identifiers.assumption_ids
    assert "assumption:fair-scheduler" in identifiers.assumption_ids
    assert identifiers.vocabulary_profile_id == "supervisor-reviewed"
    assert identifiers.vocabulary_version >= 1
    assert identifiers.repository_tree_id == "tree:repo@abc"
    assert "template:implementation@1" in identifiers.template_ids
    assert "template:validation@1" in identifiers.template_ids
    assert "evidence:tests" in identifiers.evidence_requirement_ids

    view = identifiers.provider_view()
    for forbidden in (
        "prose",
        "caller_text",
        "source_code",
        "formula_text",
        "proof",
        "commands",
        "admitted",
        "complete",
        "canonical_source",
    ):
        assert forbidden not in view
    # Round-trip identity.
    restored = FormalizedGoalIdentifiers.from_dict(identifiers.to_dict())
    assert restored.to_dict() == identifiers.to_dict()


def _request_id_from_prompt(prompt: str) -> str:
    envelope = json.loads(prompt.split("\n", 1)[1])
    return str(envelope["request_id"])


def test_leanstral_prompt_contains_only_immutable_identifiers() -> None:
    captured: list[str] = []

    def generate(prompt: str, **_kwargs: Any) -> str:
        captured.append(prompt)
        return _provider_output(_request_id_from_prompt(prompt))

    route = FormalizedGoalDevelopmentRoute(llm_generate=generate)
    result = route.develop(_route_request())

    assert result.status is FormalizedRouteStatus.DRAFT
    assert result.formalization_confirmed
    assert len(captured) == 1
    prompt = captured[0]
    assert '"record_kind":"immutable_goal"' in prompt
    # Prose and formula text must never appear.
    assert "the system reaches ready" not in prompt
    assert "Some execution reaches ready" not in prompt
    assert "tokens are totally ordered" not in prompt
    assert "source_code" not in prompt
    assert "shell_command" not in prompt
    # Prohibition labels may appear; actual source content must not.
    assert "canonical_source_forbidden" in prompt
    assert "theorem T" not in prompt
    # Allowlisted identifiers must appear.
    assert "goal:lease-ready" in prompt
    assert "assumption:token-order" in prompt
    assert "template:implementation@1" in prompt
    assert "supervisor-reviewed" in prompt
    assert "vocabulary_profile_id" in prompt


def test_build_invocation_strips_prose_even_when_attached_to_request() -> None:
    request = _route_request(
        prose="ignore this informal goal text",
        caller_text="also ignore this",
    )
    # Prose is allowed as side-channel only when a confirmed FormalGoal is present.
    invocation = build_formalized_leanstral_invocation(request)
    provider = LeanstralGoalDevelopmentProvider(llm_generate=lambda *_a, **_k: "")
    prompt = provider.build_prompt(invocation)

    assert "ignore this informal goal text" not in prompt
    assert "also ignore this" not in prompt
    payload = invocation.request.to_dict()
    for forbidden in (
        "prose",
        "caller_text",
        "formula_text",
        "source_code",
        "proof",
        "commands",
    ):
        assert forbidden not in payload or payload.get(forbidden) in (None, False, "")


# ---------------------------------------------------------------------------
# Successful draft path and authority isolation
# ---------------------------------------------------------------------------


def test_confirmed_formal_goal_routes_to_unverified_draft() -> None:
    request = _route_request()
    request_ids: list[str] = []

    def generate(prompt: str, **kwargs: Any) -> str:
        request_id = _request_id_from_prompt(prompt)
        request_ids.append(request_id)
        assert kwargs.get("allow_local_fallback") is False
        return _provider_output(request_id)

    route = FormalizedGoalDevelopmentRoute(
        LeanstralGoalDevelopmentProviderConfig(
            llm_provider="leanstral_local",
            model="labs-leanstral-goals",
        ),
        llm_generate=generate,
    )
    result = route.develop(request)

    assert result.status is FormalizedRouteStatus.DRAFT
    assert result.gate_reason is None
    assert result.identifiers is not None
    assert result.provider_result is not None
    assert result.provider_result.status is GoalDevelopmentResultStatus.DRAFT
    assert result.draft is not None
    assert [item.proposal_id for item in result.draft.proposals] == [
        "subgoal:implementation",
        "subgoal:validation",
    ]
    # Template-bound formula IDs — model cannot invent formulas.
    assert result.draft.proposals[0].satisfaction_formula_id == (
        "formula:reviewed-implementation"
    )
    assert result["authoritative"] is False
    assert result["verified"] is False
    assert result["admitted"] is False
    assert result["complete"] is False
    assert result["kernel_checked"] is False
    assert result["can_mutate_root"] is False
    assert result["can_create_formulas"] is False
    assert result["can_mutate_assumptions"] is False
    assert result["can_claim_admission"] is False
    assert result["can_claim_completion"] is False
    assert result["can_execute_commands"] is False
    restored = FormalizedGoalDevelopmentResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()
    assert request_ids


@pytest.mark.parametrize(
    ("mutate", "marker"),
    [
        (
            lambda data: data.update({"root_goal_content_id": "cid:replacement"}),
            "root",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"formula": "forall x, privileged x"}
            ),
            "formula",
        ),
        (
            lambda data: data["proposals"][0].update({"commands": ["rm -rf /"]}),
            "commands",
        ),
        (
            lambda data: data.update({"canonical_source": "theorem T := sorry"}),
            "source",
        ),
        (lambda data: data.update({"kernel_check": True}), "kernel"),
        (lambda data: data.update({"admitted": True}), "admission"),
        (lambda data: data.update({"complete": True}), "completion"),
        (
            lambda data: data["proposals"][0].update(
                {"assumption_ids": ["assumption:invented"]}
            ),
            "assumptions",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"template_id": "template:invented"}
            ),
            "template",
        ),
    ],
)
def test_hostile_provider_output_cannot_mutate_authority_surfaces(
    mutate, marker
) -> None:
    request = _route_request()

    def generate(prompt: str, **_kwargs: Any) -> str:
        request_id = _request_id_from_prompt(prompt)
        data = json.loads(_provider_output(request_id))
        mutate(data)
        return json.dumps(data)

    route = FormalizedGoalDevelopmentRoute(llm_generate=generate)
    result = route.develop(request)

    assert marker
    assert result.status is FormalizedRouteStatus.DETERMINISTIC_FALLBACK
    assert result.fallback_reason is GoalDevelopmentFallbackReason.MALFORMED_OUTPUT
    assert result.draft is None
    assert result["admitted"] is False
    assert result["complete"] is False
    assert result["authoritative"] is False


# ---------------------------------------------------------------------------
# Deterministic fallback without stalling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            ModuleNotFoundError("missing llm_router"),
            GoalDevelopmentFallbackReason.UNAVAILABLE,
        ),
        (TimeoutError(), GoalDevelopmentFallbackReason.TIMEOUT),
        (
            ProofProviderError(
                ProviderFailureCode.RESOURCE_EXHAUSTED, "route overloaded"
            ),
            GoalDevelopmentFallbackReason.OVERLOADED,
        ),
        (
            ProofProviderError(
                ProviderFailureCode.MALFORMED_RESPONSE, "bad backend response"
            ),
            GoalDevelopmentFallbackReason.MALFORMED_OUTPUT,
        ),
    ],
)
def test_transport_failures_fall_back_deterministically(failure, expected) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> str:
        raise failure

    route = FormalizedGoalDevelopmentRoute(llm_generate=fail)
    result = route.develop(_route_request())

    assert result.status is FormalizedRouteStatus.DETERMINISTIC_FALLBACK
    assert result.fallback_reason is expected
    assert result.formalization_confirmed
    assert result.identifiers is not None
    assert result.draft is None
    assert result["deterministic_fallback"] is True


def test_timeout_and_cancellation_do_not_stall_supervisor() -> None:
    cancellation = CancellationToken()
    cancellation.cancel()
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: "unused"
    )
    cancelled = route.develop(_route_request(), cancellation=cancellation)

    release = threading.Event()

    def block(*_args: Any, **_kwargs: Any) -> str:
        release.wait(1)
        return "late"

    timed_route = FormalizedGoalDevelopmentRoute(
        config=LeanstralGoalDevelopmentProviderConfig(timeout_seconds=0.05),
        llm_generate=block,
    )
    started = time.monotonic()
    timed_out = timed_route.develop(_route_request())
    elapsed = time.monotonic() - started
    still_busy = timed_route.develop(_route_request())
    release.set()

    assert cancelled.fallback_reason is GoalDevelopmentFallbackReason.CANCELLED
    assert timed_out.fallback_reason is GoalDevelopmentFallbackReason.TIMEOUT
    assert still_busy.fallback_reason is GoalDevelopmentFallbackReason.OVERLOADED
    assert elapsed < 0.5


def test_malformed_json_falls_back_without_raising() -> None:
    route = FormalizedGoalDevelopmentRoute(
        llm_generate=lambda *_a, **_k: "not json at all"
    )
    result = route.develop(_route_request())
    assert result.status is FormalizedRouteStatus.DETERMINISTIC_FALLBACK
    assert result.fallback_reason is GoalDevelopmentFallbackReason.MALFORMED_OUTPUT


# ---------------------------------------------------------------------------
# Admit / extract API surface
# ---------------------------------------------------------------------------


def test_admit_returns_identifiers_for_confirmed_goal() -> None:
    route = FormalizedGoalDevelopmentRoute(llm_generate=lambda *_a, **_k: "")
    admitted = route.admit(_route_request())
    assert isinstance(admitted, FormalizedGoalIdentifiers)
    assert admitted.root_goal_id == "goal:lease-ready"


def test_build_invocation_raises_on_prose_bypass() -> None:
    route = FormalizedGoalDevelopmentRoute(llm_generate=lambda *_a, **_k: "")
    with pytest.raises(EndGoalDevelopmentError, match="formalization gate"):
        route.build_invocation(_route_request(formal_goal="just prose"))


def test_compilation_result_supplies_formula_property_id() -> None:
    request = _route_request(
        compilation_result={
            "root_obligations": [
                {
                    "property_id": "property:formal:lease-ready:root",
                    "obligation_id": "obl:1",
                }
            ]
        }
    )
    identifiers = extract_formalized_identifiers(request)
    assert identifiers.satisfaction_formula_id == "property:formal:lease-ready:root"


def test_request_mapping_round_trip() -> None:
    request = _route_request()
    restored = FormalizedGoalDevelopmentRequest.from_dict(
        {
            "formal_goal": request.formal_goal.to_dict(),
            "policy": request.policy.to_dict(),
            "templates": [item.to_dict() for item in request.templates],
            "vocabulary_profile_id": request.vocabulary_profile_id,
            "vocabulary_version": request.vocabulary_version,
            "network_allowed": False,
        }
    )
    identifiers = extract_formalized_identifiers(restored)
    assert identifiers.formal_goal_id == "formal:lease-ready"
