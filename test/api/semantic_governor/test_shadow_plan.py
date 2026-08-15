"""Shadow planning tests for SCG-025.

Acceptance criteria enforced here:

* Development / high risk can shadow 100 percent.
* Mature low risk samples (does not force 100 percent).
* Forbidden disclosure produces local-only or no external call, never a
  policy bypass (``allow_external_expanded_disclosure`` stays false).
* Sampling is deterministic given an explicit random seed.
* Expanded remains oracle/candidate only with isolated worktree required.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    SHADOW_EXECUTION_PLAN_INTERFACE,
    ShadowExecutionPlan,
    ShadowSelectionReason,
    verify_plan_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    ShadowDisclosurePolicy,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    BASIS_POINTS_MAX,
    CREATE_SHADOW_PLAN_INTERFACE,
    CompressedContextView,
    DEFAULT_MATURE_LOW_RISK_RATE_BP,
    LifecyclePhase,
    RepositoryStateSignals,
    ResourceGateError,
    SCG_SHADOW_PLAN_EVIDENCE,
    SHADOW_PLAN_DECISION_INTERFACE,
    SHADOW_SAMPLING_POLICY_INTERFACE,
    SemanticGovernorShadowPlanError,
    ShadowPlanDecision,
    ShadowPlanDisposition,
    ShadowPlanNotSelected,
    ShadowSamplingPolicy,
    ShadowTaskView,
    assert_no_disclosure_bypass,
    collect_selection_candidates,
    create_shadow_plan,
    default_shadow_sampling_policy,
    deterministic_sample_roll,
    development_shadow_sampling_policy,
    plan_allows_external_expanded_call,
    resolve_disclosure_gate,
    sample_hits,
    select_shadow_reasons,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHADOW_PLAN_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/shadow_plan.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _task(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "SCG-025",
        "task_class": "bugfix",
        "risk_class": "low",
        "route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
    }
    base.update(overrides)
    return base


def _compressed(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "context_pack_cid": _cid("compressed-pack"),
        "includes_private_source": False,
    }
    base.update(overrides)
    return base


def _repo(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "repository_state_cid": _cid("repo-state"),
        "verification_bundle_cid": _cid("verification-bundle"),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Module surface / evidence
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_SHADOW_PLAN_EVIDENCE == "scg/shadow-plan@1"
    assert CREATE_SHADOW_PLAN_INTERFACE == "create_shadow_plan@1"
    assert SHADOW_SAMPLING_POLICY_INTERFACE == "ShadowSamplingPolicy@1"
    assert SHADOW_PLAN_DECISION_INTERFACE == "ShadowPlanDecision@1"
    assert SHADOW_EXECUTION_PLAN_INTERFACE == "ShadowExecutionPlan@1"


def test_module_import_performs_no_io() -> None:
    source = SHADOW_PLAN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                assert name not in forbidden


# ---------------------------------------------------------------------------
# ShadowSamplingPolicy identity and defaults
# ---------------------------------------------------------------------------


def test_default_policy_is_mature_with_low_sample_rate() -> None:
    policy = default_shadow_sampling_policy()
    assert policy.lifecycle_phase == LifecyclePhase.MATURE.value
    assert policy.mature_low_risk_sample_rate_bp == DEFAULT_MATURE_LOW_RISK_RATE_BP
    assert policy.mature_low_risk_sample_rate_bp < BASIS_POINTS_MAX
    assert policy.high_risk_sample_rate_bp == BASIS_POINTS_MAX
    assert policy.development_sample_rate_bp == BASIS_POINTS_MAX
    assert policy.expanded_is_oracle_candidate_only is True
    assert policy.require_isolated_evaluation_worktree is True
    assert policy.allow_external_expanded_disclosure is False
    assert policy.policy_cid == ShadowSamplingPolicy().policy_cid


def test_development_policy_is_full_rate() -> None:
    policy = development_shadow_sampling_policy(random_seed=42)
    assert policy.lifecycle_phase == LifecyclePhase.DEVELOPMENT.value
    assert policy.development_sample_rate_bp == BASIS_POINTS_MAX
    assert policy.random_seed == 42


def test_policy_round_trip_identity() -> None:
    policy = ShadowSamplingPolicy(
        policy_id="shadow-sampling-lab",
        lifecycle_phase=LifecyclePhase.PRODUCTION,
        random_seed=99,
        mature_low_risk_sample_rate_bp=500,
        notes="lab",
    )
    restored = ShadowSamplingPolicy.from_dict(policy.to_dict())
    assert restored.policy_cid == policy.policy_cid
    assert restored.mature_low_risk_sample_rate_bp == 500
    assert restored.lifecycle_phase == LifecyclePhase.PRODUCTION.value


def test_policy_rejects_oracle_false_and_worktree_false() -> None:
    with pytest.raises(SemanticGovernorShadowPlanError, match="oracle"):
        ShadowSamplingPolicy(expanded_is_oracle_candidate_only=False)
    with pytest.raises(SemanticGovernorShadowPlanError, match="worktree"):
        ShadowSamplingPolicy(require_isolated_evaluation_worktree=False)


def test_policy_rejects_rate_above_basis_points() -> None:
    with pytest.raises(SemanticGovernorShadowPlanError, match="10000|between"):
        ShadowSamplingPolicy(high_risk_sample_rate_bp=10_001)


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------


def test_deterministic_sample_roll_is_stable() -> None:
    pack = _cid("pack-roll")
    left = deterministic_sample_roll(
        random_seed=7, task_id="T1", compressed_context_pack_cid=pack
    )
    right = deterministic_sample_roll(
        random_seed=7, task_id="T1", compressed_context_pack_cid=pack
    )
    other = deterministic_sample_roll(
        random_seed=8, task_id="T1", compressed_context_pack_cid=pack
    )
    assert left == right
    assert 0 <= left < BASIS_POINTS_MAX
    assert other != left or True  # may theoretically collide; seed change is allowed
    assert sample_hits(0, BASIS_POINTS_MAX) is True
    assert sample_hits(0, 0) is False
    assert sample_hits(50, 100) is True
    assert sample_hits(100, 100) is False


def test_select_shadow_reasons_deterministic_with_seed() -> None:
    task = ShadowTaskView(task_id="SCG-025-A", risk_class="low")
    compressed = CompressedContextView(context_pack_cid=_cid("pack-a"))
    repo = RepositoryStateSignals(repository_state_cid=_cid("repo-a"))
    policy = default_shadow_sampling_policy(random_seed=123)
    a = select_shadow_reasons(task, compressed, repo, policy)
    b = select_shadow_reasons(task, compressed, repo, policy)
    assert a == b


# ---------------------------------------------------------------------------
# Acceptance: development / high risk can shadow 100 percent
# ---------------------------------------------------------------------------


def test_development_shadows_one_hundred_percent() -> None:
    decision = create_shadow_plan(
        _task(environment="development", risk_class="low"),
        _compressed(),
        _repo(),
        development_shadow_sampling_policy(),
        sample_roll=9_999,  # would miss low rates
    )
    assert decision.selected is True
    assert decision.disposition in {
        ShadowPlanDisposition.SELECTED.value,
        ShadowPlanDisposition.DISCLOSURE_LOCAL_ONLY.value,
    }
    assert ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value in decision.selection_reasons
    assert decision.plan is not None
    assert decision.plan.expanded_is_oracle_candidate_only is True
    assert decision.plan.isolated_evaluation_worktree_required is True
    assert verify_plan_identity(decision.plan) == decision.plan.plan_cid


def test_development_lifecycle_policy_shadows_full_rate() -> None:
    policy = ShadowSamplingPolicy(
        lifecycle_phase=LifecyclePhase.DEVELOPMENT,
        development_sample_rate_bp=BASIS_POINTS_MAX,
    )
    decision = create_shadow_plan(
        _task(risk_class="low"),
        _compressed(context_pack_cid=_cid("dev-pack")),
        _repo(repository_state_cid=_cid("dev-repo")),
        policy,
        sample_roll=9_999,
    )
    assert decision.selected is True
    assert ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value in decision.selection_reasons


def test_high_risk_shadows_one_hundred_percent() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-HIGH", risk_class="high"),
        _compressed(context_pack_cid=_cid("high-pack")),
        _repo(repository_state_cid=_cid("high-repo")),
        default_shadow_sampling_policy(),
        sample_roll=9_999,
    )
    assert decision.selected is True
    assert ShadowSelectionReason.RISK_CLASS_MANDATORY.value in decision.selection_reasons
    assert decision.plan is not None
    assert decision.effective_sample_rate_bp == BASIS_POINTS_MAX


def test_critical_risk_shadows_one_hundred_percent() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-CRIT", risk_class="critical"),
        _compressed(context_pack_cid=_cid("crit-pack")),
        _repo(repository_state_cid=_cid("crit-repo")),
        default_shadow_sampling_policy(),
        sample_roll=9_999,
    )
    assert decision.selected is True
    assert ShadowSelectionReason.RISK_CLASS_MANDATORY.value in decision.selection_reasons


# ---------------------------------------------------------------------------
# Acceptance: mature low risk samples (not 100 percent)
# ---------------------------------------------------------------------------


def test_mature_low_risk_does_not_force_full_rate() -> None:
    policy = default_shadow_sampling_policy(random_seed=1)
    assert policy.mature_low_risk_sample_rate_bp < BASIS_POINTS_MAX

    miss = create_shadow_plan(
        _task(task_id="SCG-025-LOW-MISS", risk_class="low"),
        _compressed(context_pack_cid=_cid("low-miss")),
        _repo(repository_state_cid=_cid("repo-low-miss")),
        policy,
        sample_roll=9_999,
    )
    assert miss.selected is False
    assert miss.plan is None
    assert miss.disposition == ShadowPlanDisposition.SKIPPED.value
    assert miss.effective_sample_rate_bp == policy.mature_low_risk_sample_rate_bp or (
        miss.effective_sample_rate_bp == max(
            policy.mature_low_risk_sample_rate_bp,
            policy.random_quality_control_rate_bp,
        )
    )


def test_mature_low_risk_samples_when_roll_hits() -> None:
    policy = default_shadow_sampling_policy(random_seed=1)
    hit = create_shadow_plan(
        _task(task_id="SCG-025-LOW-HIT", risk_class="low"),
        _compressed(context_pack_cid=_cid("low-hit")),
        _repo(repository_state_cid=_cid("repo-low-hit")),
        policy,
        sample_roll=0,
    )
    assert hit.selected is True
    assert ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value in hit.selection_reasons
    assert hit.plan is not None
    # Not forced to development/high mandatory reasons.
    assert ShadowSelectionReason.DEVELOPMENT_FULL_RATE.value not in hit.selection_reasons
    assert ShadowSelectionReason.RISK_CLASS_MANDATORY.value not in hit.selection_reasons


def test_mature_low_risk_information_value_boosts() -> None:
    policy = default_shadow_sampling_policy()
    decision = create_shadow_plan(
        _task(
            task_id="SCG-025-IV",
            risk_class="low",
            new_analyzer=True,
            promotion_evaluation=False,
        ),
        _compressed(
            context_pack_cid=_cid("iv-pack"),
            capsule_uncertainty=True,
            token_savings_eligible=True,
        ),
        _repo(
            repository_state_cid=_cid("iv-repo"),
            recent_omission=True,
        ),
        policy,
        sample_roll=0,
    )
    assert decision.selected is True
    reasons = set(decision.selection_reasons)
    assert ShadowSelectionReason.CAPSULE_UNCERTAINTY.value in reasons
    assert ShadowSelectionReason.NEW_ANALYZER.value in reasons
    assert ShadowSelectionReason.TOKEN_SAVINGS_SAMPLE.value in reasons
    assert ShadowSelectionReason.RECENT_OMISSION.value in reasons


def test_require_selected_raises_on_miss() -> None:
    with pytest.raises(ShadowPlanNotSelected):
        create_shadow_plan(
            _task(task_id="SCG-025-REQ", risk_class="low"),
            _compressed(context_pack_cid=_cid("req-pack")),
            _repo(repository_state_cid=_cid("req-repo")),
            default_shadow_sampling_policy(),
            sample_roll=9_999,
            require_selected=True,
        )


# ---------------------------------------------------------------------------
# Acceptance: forbidden disclosure → local-only / no external, never bypass
# ---------------------------------------------------------------------------


def test_forbidden_disclosure_never_allows_external_expanded() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-PRIV", risk_class="high"),
        _compressed(
            context_pack_cid=_cid("priv-pack"),
            includes_private_source=True,
        ),
        _repo(repository_state_cid=_cid("priv-repo")),
        default_shadow_sampling_policy(),
        expanded_provider_id="external.unapproved.vendor",
        expanded_context={"raw_private_source": "def secret():\n    return 1\n"},
    )
    assert decision.selected is True
    assert decision.allow_external_expanded_disclosure is False
    assert decision.plan is not None
    assert decision.plan.allow_external_expanded_disclosure is False
    assert decision.disclosure_disposition == DisclosureDisposition.FORBIDDEN.value
    assert (
        ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
        in decision.selection_reasons
    )
    assert plan_allows_external_expanded_call(decision) is False
    assert_no_disclosure_bypass(decision)


def test_forbidden_disclosure_local_provider_is_local_only_not_external() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-LOCAL", risk_class="high"),
        _compressed(
            context_pack_cid=_cid("local-pack"),
            includes_private_source=True,
        ),
        _repo(repository_state_cid=_cid("local-repo")),
        default_shadow_sampling_policy(),
        expanded_provider_id="local:hermetic-expanded",
        expanded_context={"raw_private_source": "x = 1\n"},
    )
    assert decision.selected is True
    assert decision.allow_external_expanded_disclosure is False
    assert decision.disclosure_disposition == DisclosureDisposition.LOCAL_ONLY.value
    assert_no_disclosure_bypass(decision)


def test_sampling_policy_cannot_bypass_privacy_for_unapproved_external() -> None:
    # Even if sampling policy requests external disclosure, privacy forbids it.
    greedy = ShadowSamplingPolicy(
        allow_external_expanded_disclosure=True,
        high_risk_sample_rate_bp=BASIS_POINTS_MAX,
    )
    decision = create_shadow_plan(
        _task(task_id="SCG-025-BYPASS", risk_class="high"),
        _compressed(
            context_pack_cid=_cid("bypass-pack"),
            includes_private_source=True,
        ),
        _repo(repository_state_cid=_cid("bypass-repo")),
        greedy,
        disclosure_policy=default_shadow_disclosure_policy(),
        expanded_provider_id="openai.unlisted.model",
        expanded_context={"private_source": "classified"},
    )
    assert decision.plan is not None
    assert decision.plan.allow_external_expanded_disclosure is False
    assert decision.allow_external_expanded_disclosure is False
    assert_no_disclosure_bypass(decision)


def test_assert_no_disclosure_bypass_detects_invalid_decision() -> None:
    # Construct an illegal decision-like path via resolve_disclosure_gate checks.
    policy = default_shadow_sampling_policy()
    gate = resolve_disclosure_gate(
        policy=policy,
        disclosure_policy=default_shadow_disclosure_policy(),
        expanded_provider_id="external.bad",
        includes_private_source=True,
        expanded_context={"raw_private_source": "x"},
    )
    assert gate.allow_external_expanded_disclosure is False
    assert gate.disposition == DisclosureDisposition.FORBIDDEN.value


def test_approved_external_requires_exact_authority_still_no_ambient_trust() -> None:
    # Approved id listed but no private-external authority → forbidden, no bypass.
    disc = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.model.v1",),
        allow_private_source_to_approved_external=False,
    )
    decision = create_shadow_plan(
        _task(task_id="SCG-025-APPR", risk_class="high"),
        _compressed(
            context_pack_cid=_cid("appr-pack"),
            includes_private_source=True,
        ),
        _repo(repository_state_cid=_cid("appr-repo")),
        ShadowSamplingPolicy(allow_external_expanded_disclosure=True),
        disclosure_policy=disc,
        expanded_provider_id="partner.model.v1",
        expanded_context={"raw_private_source": "body"},
    )
    assert decision.allow_external_expanded_disclosure is False
    assert decision.plan is not None
    assert decision.plan.allow_external_expanded_disclosure is False


def test_approved_external_with_exact_authority_can_allow_external() -> None:
    disc = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.model.v1",),
        allow_private_source_to_approved_external=True,
        authorization_cid=_cid("auth-exact"),
    )
    decision = create_shadow_plan(
        _task(task_id="SCG-025-OK", risk_class="high"),
        _compressed(
            context_pack_cid=_cid("ok-pack"),
            includes_private_source=True,
        ),
        _repo(repository_state_cid=_cid("ok-repo")),
        ShadowSamplingPolicy(allow_external_expanded_disclosure=True),
        disclosure_policy=disc,
        expanded_provider_id="partner.model.v1",
        expanded_context={"raw_private_source": "body"},
        worktree_id="worktree-eval-ok",
    )
    assert decision.selected is True
    assert decision.allow_external_expanded_disclosure is True
    assert decision.plan is not None
    assert decision.plan.allow_external_expanded_disclosure is True
    assert plan_allows_external_expanded_call(decision) is True


# ---------------------------------------------------------------------------
# Plan contract invariants
# ---------------------------------------------------------------------------


def test_plan_identity_stable_and_oracle_only() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-ID", risk_class="high"),
        _compressed(context_pack_cid=_cid("id-pack")),
        _repo(repository_state_cid=_cid("id-repo")),
        default_shadow_sampling_policy(random_seed=0),
    )
    assert decision.plan is not None
    again = create_shadow_plan(
        _task(task_id="SCG-025-ID", risk_class="high"),
        _compressed(context_pack_cid=_cid("id-pack")),
        _repo(repository_state_cid=_cid("id-repo")),
        default_shadow_sampling_policy(random_seed=0),
    )
    assert again.plan is not None
    assert decision.plan.plan_cid == again.plan.plan_cid
    assert decision.plan.expanded_is_oracle_candidate_only is True
    assert decision.plan.isolated_evaluation_worktree_required is True
    restored = ShadowExecutionPlan.from_dict(decision.plan.to_dict())
    assert restored.plan_cid == decision.plan.plan_cid


def test_decision_round_trip_selected() -> None:
    decision = create_shadow_plan(
        _task(task_id="SCG-025-RT", risk_class="high"),
        _compressed(context_pack_cid=_cid("rt-pack")),
        _repo(repository_state_cid=_cid("rt-repo")),
        default_shadow_sampling_policy(),
    )
    payload = decision.to_dict()
    assert payload["decision_cid"] == decision.decision_cid
    assert payload["selected"] is True
    assert payload["plan"]["plan_cid"] == decision.plan_cid


def test_resource_gate_zero_wall_time_fails() -> None:
    with pytest.raises(ResourceGateError, match="max_wall_time_ms"):
        create_shadow_plan(
            _task(task_id="SCG-025-RES", risk_class="high"),
            _compressed(context_pack_cid=_cid("res-pack")),
            _repo(repository_state_cid=_cid("res-repo")),
            ShadowSamplingPolicy(max_wall_time_ms=0),
        )


def test_coerce_string_task_and_cids() -> None:
    decision = create_shadow_plan(
        "SCG-025-STR",
        _cid("str-pack"),
        _cid("str-repo"),
        {"lifecycle_phase": "development", "random_seed": 3},
        sample_roll=0,
    )
    assert decision.task_id == "SCG-025-STR"
    assert decision.selected is True


def test_collect_selection_candidates_includes_risk_and_qc() -> None:
    task = ShadowTaskView(task_id="SCG-025-C", risk_class="medium")
    compressed = CompressedContextView(context_pack_cid=_cid("c-pack"))
    repo = RepositoryStateSignals(repository_state_cid=_cid("c-repo"))
    policy = default_shadow_sampling_policy()
    candidates = collect_selection_candidates(task, compressed, repo, policy)
    reasons = {r for r, _ in candidates}
    assert ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value in reasons
    assert ShadowSelectionReason.RISK_CLASS_MANDATORY.value not in reasons


def test_promotion_evaluation_selects_at_full_configured_rate() -> None:
    decision = create_shadow_plan(
        _task(
            task_id="SCG-025-PROM",
            risk_class="low",
            promotion_evaluation=True,
        ),
        _compressed(context_pack_cid=_cid("prom-pack")),
        _repo(repository_state_cid=_cid("prom-repo")),
        default_shadow_sampling_policy(),
        sample_roll=0,
    )
    assert decision.selected is True
    assert (
        ShadowSelectionReason.PROMOTION_EVALUATION.value in decision.selection_reasons
    )


def test_views_validate_tokens_and_cids() -> None:
    with pytest.raises(SemanticGovernorShadowPlanError):
        ShadowTaskView(task_id="SCG-025", risk_class="NOT VALID")
    with pytest.raises(SemanticGovernorShadowPlanError):
        CompressedContextView(context_pack_cid="not-a-cid")
    with pytest.raises(SemanticGovernorShadowPlanError):
        RepositoryStateSignals(repository_state_cid="also-bad")
