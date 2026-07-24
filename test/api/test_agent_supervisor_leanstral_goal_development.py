"""Tests for the capability-isolated Leanstral goal-development provider."""

from __future__ import annotations

import json
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_provider import (
    CancellationToken,
    ProofProviderError,
    ProviderFailureCode,
)
from ipfs_accelerate_py.agent_supervisor.goal_development_contracts import (
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentRequest,
)
from ipfs_accelerate_py.agent_supervisor.leanstral_goal_development import (
    ASTGraphRAGReferenceRecord,
    CapabilityRecord,
    EvidenceGapRecord,
    GoalDevelopmentContext,
    GoalDevelopmentFallbackReason,
    GoalDevelopmentProviderResult,
    GoalDevelopmentResultStatus,
    GoalDevelopmentTemplate,
    ImmutableGoalRecord,
    LEANSTRAL_GOAL_DEVELOPMENT_OPERATION,
    LEANSTRAL_GOAL_DEVELOPMENT_OUTPUT_SCHEMA,
    LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID,
    LeanstralGoalDevelopmentInvocation,
    LeanstralGoalDevelopmentProvider,
    LeanstralGoalDevelopmentProviderConfig,
    PriorCounterexampleRecord,
    ReusableReceiptRecord,
)
from ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider import (
    LEANSTRAL_MODEL_RESOURCE_CLASS,
    LEAN_KERNEL_RESOURCE_CLASS,
)


def _policy(**changes):
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


def _request(policy=None):
    policy = policy or _policy()
    return GoalDevelopmentRequest(
        root_goal_id="goal:root",
        root_goal_content_id="cid:immutable-root",
        satisfaction_formula_id="formula:root",
        assumption_ids=("assumption:reviewed",),
        evidence_requirement_ids=("evidence:tests", "evidence:review"),
        vocabulary_profile_id="vocabulary:reviewed",
        vocabulary_version=1,
        repository_tree_id="tree:fixed",
        scope_ids=("scope:package", "scope:tests"),
        policy_digest=policy.policy_digest,
        mode=policy.mode,
    )


def _context(request=None):
    request = request or _request()
    return GoalDevelopmentContext(
        goal=ImmutableGoalRecord(
            goal_id=request.root_goal_id,
            content_id=request.root_goal_content_id,
            satisfaction_formula_id=request.satisfaction_formula_id,
        ),
        templates=(
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
        ),
        evidence_gaps=(
            EvidenceGapRecord("gap:tests", ("evidence:tests",)),
        ),
        code_references=(
            ASTGraphRAGReferenceRecord(
                "ast:symbol-index",
                "ast",
                request.repository_tree_id,
                ("scope:package",),
                ("symbol:provider",),
            ),
            ASTGraphRAGReferenceRecord(
                "graphrag:neighbors",
                "graphrag",
                request.repository_tree_id,
                ("scope:tests",),
            ),
        ),
        capabilities=(
            CapabilityRecord(
                "capability:pytest",
                ("resource:codex",),
                ("check:pytest",),
            ),
        ),
        prior_counterexamples=(
            PriorCounterexampleRecord(
                "counterexample:cycle",
                "cyclic_dependency",
                ("old:a", "old:b"),
                ("check:review",),
            ),
        ),
        reusable_receipts=(
            ReusableReceiptRecord(
                "receipt:old-tests",
                ("evidence:tests",),
                "assurance:typed",
                ("scope:package",),
            ),
        ),
    )


def _invocation(**changes):
    policy = changes.pop("policy", _policy())
    request = changes.pop("request", _request(policy))
    values = {
        "request": request,
        "policy": policy,
        "context": _context(request),
        "resource_budget": ResourceBudget(
            wall_time_ms=5_000,
            model_token_limit=1_024,
            max_output_bytes=64 * 1024,
        ),
    }
    values.update(changes)
    return LeanstralGoalDevelopmentInvocation(**values)


def _proposal(
    proposal_id,
    *,
    parent_id="goal:root",
    template_id="template:implementation@1",
    depends_on=(),
):
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


def _output(invocation=None, proposals=None):
    invocation = invocation or _invocation()
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
            "request_id": invocation.request.request_id,
            "proposals": proposals,
        },
        separators=(",", ":"),
    )


def test_capability_is_a_separate_draft_only_operation() -> None:
    provider = LeanstralGoalDevelopmentProvider(llm_generate=lambda *_a, **_k: "")

    capability = provider.capabilities()
    payload = capability.to_dict()

    assert capability.provider_id == LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID
    assert capability.supports("goal_development.v1")
    assert not capability.supports("prove")
    assert payload["operations"] == ["goal_development.v1"]
    assert payload["proof_operations"] == []
    assert payload["kernel_check_supported"] is False
    assert payload["command_execution_supported"] is False
    assert payload["resource_classes"] == {
        "model_inference": LEANSTRAL_MODEL_RESOURCE_CLASS,
        "kernel_check": LEAN_KERNEL_RESOURCE_CLASS,
    }


def test_bounded_context_contains_all_reference_kinds_without_source() -> None:
    provider = LeanstralGoalDevelopmentProvider(llm_generate=lambda *_a, **_k: "")
    prompt = provider.build_prompt(_invocation())

    assert '"record_kind":"immutable_goal"' in prompt
    assert '"record_kind":"evidence_gap"' in prompt
    assert '"reference_kind":"ast"' in prompt
    assert '"reference_kind":"graphrag"' in prompt
    assert '"record_kind":"capability"' in prompt
    assert '"record_kind":"prior_counterexample"' in prompt
    assert '"record_kind":"reusable_receipt"' in prompt
    assert "canonical_source" in prompt  # prohibition, never source content
    assert "source_code" not in prompt
    assert "shell_command" not in prompt


def test_route_is_pinned_and_valid_output_becomes_unverified_contract() -> None:
    invocation = _invocation()
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return _output(invocation)

    provider = LeanstralGoalDevelopmentProvider(
        LeanstralGoalDevelopmentProviderConfig(
            llm_provider="leanstral_local",
            model="labs-leanstral-goals",
            timeout_seconds=30,
            max_new_tokens=1_500,
        ),
        llm_generate=generate,
    )

    result = provider.develop(invocation)

    assert result.status is GoalDevelopmentResultStatus.DRAFT
    assert result.used_fallback is False
    assert result.draft is not None
    assert result.draft.request == invocation.request
    assert result.draft.policy == invocation.policy
    assert [item.proposal_id for item in result.draft.proposals] == [
        "subgoal:implementation",
        "subgoal:validation",
    ]
    assert result.draft.proposals[0].satisfaction_formula_id == (
        "formula:reviewed-implementation"
    )
    assert result.draft.assurance is AssuranceLevel.UNVERIFIED
    assert result["authoritative"] is False
    assert result["kernel_checked"] is False
    assert result["can_execute_commands"] is False
    assert calls[0][1]["provider"] == "leanstral_local"
    assert calls[0][1]["model_name"] == "labs-leanstral-goals"
    assert calls[0][1]["timeout"] == 5.0
    assert calls[0][1]["max_new_tokens"] == 1_024
    assert calls[0][1]["allow_local_fallback"] is False
    assert calls[0][1]["disable_model_retry"] is True
    restored = GoalDevelopmentProviderResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()
    assert LeanstralGoalDevelopmentInvocation.from_dict(
        invocation.to_dict()
    ).to_dict() == invocation.to_dict()


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
            lambda data: data["proposals"][0].update(
                {"commands": ["rm -rf /"]}
            ),
            "commands",
        ),
        (
            lambda data: data.update({"canonical_source": "theorem T := sorry"}),
            "source",
        ),
        (lambda data: data.update({"kernel_check": True}), "kernel"),
        (lambda data: data.update({"surprise": True}), "unknown"),
        (
            lambda data: data["proposals"][0].update(
                {"template_id": "template:invented"}
            ),
            "template",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"evidence_requirement_ids": ["evidence:invented"]}
            ),
            "evidence",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"assurance_ids": ["assurance:kernel"]}
            ),
            "assurance",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"resource_ids": ["resource:root-shell"]}
            ),
            "resource",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"scope_ids": ["scope:outside"]}
            ),
            "scope",
        ),
        (
            lambda data: data["proposals"][0].update(
                {"validation_check_ids": ["check:arbitrary-command"]}
            ),
            "check",
        ),
    ],
)
def test_hostile_or_non_allowlisted_output_uses_malformed_fallback(
    mutate, marker
) -> None:
    invocation = _invocation()
    data = json.loads(_output(invocation))
    mutate(data)
    provider = LeanstralGoalDevelopmentProvider(
        llm_generate=lambda *_a, **_k: json.dumps(data)
    )

    result = provider.develop(invocation)

    assert marker  # readable parametrization IDs in failure output
    assert result.used_fallback
    assert result.fallback_reason is GoalDevelopmentFallbackReason.MALFORMED_OUTPUT
    assert result.draft is None


def test_parent_and_dependency_cycles_are_rejected() -> None:
    invocation = _invocation()
    parent_cycle = [
        _proposal("subgoal:a", parent_id="subgoal:b"),
        _proposal("subgoal:b", parent_id="subgoal:a"),
    ]
    dependency_cycle = [
        _proposal("subgoal:a", depends_on=("subgoal:b",)),
        _proposal("subgoal:b", depends_on=("subgoal:a",)),
    ]

    for proposals in (parent_cycle, dependency_cycle):
        provider = LeanstralGoalDevelopmentProvider(
            llm_generate=lambda *_a, _proposals=proposals, **_k: _output(
                invocation, _proposals
            )
        )
        result = provider.develop(invocation)
        assert result.fallback_reason is GoalDevelopmentFallbackReason.MALFORMED_OUTPUT


@pytest.mark.parametrize(
    "bad_output",
    [
        "not json",
        "[]",
        '{"schema":"x","schema":"y"}',
        '{"schema":NaN}',
        "",
    ],
)
def test_strict_json_and_schema_fail_closed(bad_output) -> None:
    result = LeanstralGoalDevelopmentProvider(
        llm_generate=lambda *_a, **_k: bad_output
    ).develop(_invocation())

    assert result.fallback_reason is GoalDevelopmentFallbackReason.MALFORMED_OUTPUT


def test_excessive_output_and_context_route_to_overload_fallback() -> None:
    invocation = _invocation()
    oversized_output = LeanstralGoalDevelopmentProvider(
        LeanstralGoalDevelopmentProviderConfig(max_output_bytes=8),
        llm_generate=lambda *_a, **_k: _output(invocation),
    ).develop(invocation)
    oversized_context = LeanstralGoalDevelopmentProvider(
        LeanstralGoalDevelopmentProviderConfig(max_context_bytes=32),
        llm_generate=lambda *_a, **_k: _output(invocation),
    ).develop(invocation)

    assert oversized_output.fallback_reason is GoalDevelopmentFallbackReason.OVERLOADED
    assert oversized_context.fallback_reason is GoalDevelopmentFallbackReason.OVERLOADED


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
def test_expected_transport_failures_return_explicit_fallback(failure, expected) -> None:
    def fail(*_args, **_kwargs):
        raise failure

    result = LeanstralGoalDevelopmentProvider(llm_generate=fail).develop(
        _invocation()
    )

    assert result.status is GoalDevelopmentResultStatus.DETERMINISTIC_FALLBACK
    assert result.deterministic_fallback
    assert result.fallback_reason is expected
    assert result["result_id"] == result.result_id


def test_cancellation_and_hard_timeout_do_not_stall_caller() -> None:
    cancellation = CancellationToken()
    cancellation.cancel()
    cancelled = LeanstralGoalDevelopmentProvider(
        llm_generate=lambda *_a, **_k: "unused"
    ).develop(_invocation(), cancellation=cancellation)

    release = threading.Event()

    def block(*_args, **_kwargs):
        release.wait(1)
        return "late"

    provider = LeanstralGoalDevelopmentProvider(
        LeanstralGoalDevelopmentProviderConfig(timeout_seconds=0.05),
        llm_generate=block,
    )
    started = time.monotonic()
    timed_out = provider.develop(_invocation())
    elapsed = time.monotonic() - started
    still_busy = provider.develop(_invocation())
    release.set()

    assert cancelled.fallback_reason is GoalDevelopmentFallbackReason.CANCELLED
    assert timed_out.fallback_reason is GoalDevelopmentFallbackReason.TIMEOUT
    assert still_busy.fallback_reason is GoalDevelopmentFallbackReason.OVERLOADED
    assert elapsed < 0.5


def test_busy_provider_returns_overload_without_waiting() -> None:
    entered = threading.Event()
    release = threading.Event()
    invocation = _invocation()

    def block(*_args, **_kwargs):
        entered.set()
        release.wait(1)
        return _output(invocation)

    provider = LeanstralGoalDevelopmentProvider(
        LeanstralGoalDevelopmentProviderConfig(
            max_concurrent_requests=1,
            timeout_seconds=2,
        ),
        llm_generate=block,
    )
    holder = threading.Thread(target=lambda: provider.develop(invocation))
    holder.start()
    assert entered.wait(0.5)

    started = time.monotonic()
    overloaded = provider.develop(invocation)
    elapsed = time.monotonic() - started
    release.set()
    holder.join(1)

    assert overloaded.fallback_reason is GoalDevelopmentFallbackReason.OVERLOADED
    assert elapsed < 0.2


def test_context_rejects_source_injection_and_changed_root_before_model_call() -> None:
    request = _request()
    context = _context(request).to_dict()
    context["code_references"][0]["canonical_source"] = "malicious source"
    calls = []
    provider = LeanstralGoalDevelopmentProvider(
        llm_generate=lambda *_a, **_k: calls.append(True) or ""
    )

    with pytest.raises(Exception, match="unknown fields"):
        provider.develop(request, policy=_policy(), context=context)

    changed = _context(request).to_dict()
    changed["goal"]["content_id"] = "cid:changed"
    with pytest.raises(Exception, match="immutable request root"):
        provider.develop(request, policy=_policy(), context=changed)
    assert calls == []
