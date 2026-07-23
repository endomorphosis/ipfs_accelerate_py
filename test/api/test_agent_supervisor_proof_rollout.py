from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    ChangedScope,
    FormalVerificationPolicy,
    OverrideReceipt,
    ProofOutcome,
    ProofPolicyRule,
    ProofResultStatus,
    RiskLevel,
    RolloutMode,
    RolloutTransitionReceipt,
    build_proof_rollout_status,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    apply_proof_rollout_projection,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)
TREE = "git-tree:rollout"
PATH = "src/agent_supervisor/lease.py"


def _policy(mode: RolloutMode = RolloutMode.ENFORCEMENT) -> FormalVerificationPolicy:
    return FormalVerificationPolicy(
        name="rollout-security",
        version="11",
        rollout_mode=mode,
        rules=(
            ProofPolicyRule(
                rule_id="lease-safety",
                path_patterns=("src/agent_supervisor/**",),
                minimum_risk=RiskLevel.HIGH,
                invariant_classes=("lease_safety",),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                fallback_validations=("pytest:lease",),
                allow_fallback=True,
            ),
        ),
        canary_percent=25,
        canary_salt="rollout-security-v11",
        minimum_promotion_observations=2,
    )


def _selection(policy: FormalVerificationPolicy):
    return policy.select(
        (
            ChangedScope(
                path=PATH,
                ast_scope_ids=("class:LeaseStore",),
                risk=RiskLevel.CRITICAL,
                invariant_classes=("lease_safety",),
            ),
        ),
        repository_tree_id=TREE,
    )


def test_status_exposes_policy_scope_capability_plan_assurance_and_failures() -> None:
    policy = _policy()
    selection = _selection(policy)
    outcome = ProofOutcome(
        requirement_id=selection.requirements[0].requirement_id,
        status=ProofResultStatus.UNAVAILABLE,
        reason_code="provider_outage",
    )
    decision = policy.evaluate_gate(selection, (outcome,))

    status = build_proof_rollout_status(
        policy,
        selections=(selection,),
        decisions=(decision,),
        capability_health={
            "hammer": {
                "status": "unavailable",
                "healthy": False,
                "reason_code": "provider_outage",
            },
            "lean": {"status": "healthy", "healthy": True, "version": "4.19"},
        },
        active_plans=(
            {
                "plan_id": "plan:active",
                "repository_tree_id": TREE,
                "steps": [{"step_id": "solve"}, {"step_id": "kernel"}],
                "snapshot": {
                    "status": "running",
                    "nodes": [
                        {"state": "running"},
                        {"state": "blocked"},
                    ],
                },
            },
        ),
        generated_at=NOW,
    )

    assert status["rollout_mode"] == "enforcement"
    assert status["blocking"] is True
    assert status["mode_authority"] == "policy"
    assert status["provider_health_can_change_mode"] is False
    assert status["protected_scopes"] == ["src/agent_supervisor/**"]
    assert status["capability_health"][0]["authoritative"] is False
    assert status["active_plans"][0]["failed_step_count"] == 1
    assert status["assurance_counts"]["unverified"] == 1
    assert status["fallbacks"] == ["pytest:lease"]
    assert any(
        item["reason_code"] == "proof_unavailable"
        for item in status["failures"]
    )


def test_provider_outage_cannot_silently_downgrade_enforcement() -> None:
    policy = _policy(RolloutMode.ENFORCEMENT)
    selection = _selection(policy)
    unavailable = ProofOutcome(
        requirement_id=selection.requirements[0].requirement_id,
        status=ProofResultStatus.UNAVAILABLE,
        reason_code="provider_outage",
    )
    decision = policy.evaluate_gate(selection, unavailable)
    status = build_proof_rollout_status(
        policy,
        selections=(selection,),
        decisions=(decision,),
        capability_health={"hammer": {"status": "unavailable"}},
        generated_at=NOW,
    )

    assert decision.allowed is False
    assert status["rollout_mode"] == RolloutMode.ENFORCEMENT.value
    assert status["blocking"] is True


def test_canary_expansion_and_rollback_require_durable_policy_identity() -> None:
    shadow = _policy(RolloutMode.SHADOW)
    promote = RolloutTransitionReceipt(
        policy_id=shadow.policy_id,
        from_mode=RolloutMode.SHADOW,
        target_mode=RolloutMode.CANARY,
        actor="release-manager",
        reason="Reviewed shadow observations passed.",
        issued_at=NOW.isoformat(),
        observation_count=2,
        evidence_receipt_ids=("benchmark:1", "adversarial:1"),
    )
    canary = shadow.transition(promote)
    rollback = RolloutTransitionReceipt(
        policy_id=canary.policy_id,
        from_mode=RolloutMode.CANARY,
        target_mode=RolloutMode.SHADOW,
        actor="incident-commander",
        reason="Rollback on false-block regression.",
        issued_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    rolled_back = canary.transition(rollback)
    status = build_proof_rollout_status(
        canary,
        transitions=(promote, rollback),
        generated_at=NOW + timedelta(minutes=2),
    )

    assert canary.policy_id != shadow.policy_id
    assert rolled_back.policy_id != canary.policy_id
    assert [row["receipt_id"] for row in status["transitions"]] == [
        promote.receipt_id,
        rollback.receipt_id,
    ]


def test_override_diagnostics_are_expiring_scoped_and_do_not_rewrite_verdict() -> None:
    policy = _policy()
    selection = _selection(policy)
    override = OverrideReceipt.create(
        policy_id=policy.policy_id,
        repository_tree_id=TREE,
        paths=(PATH,),
        ast_scope_ids=("class:LeaseStore",),
        invariant_classes=("lease_safety",),
        actor="incident-commander",
        reason="Provider incident INC-365.",
        ticket_id="INC-365",
        ttl_seconds=60,
        now=NOW,
    )
    outcome = ProofOutcome(
        requirement_id=selection.requirements[0].requirement_id,
        status=ProofResultStatus.TIMED_OUT,
    )
    decision = policy.evaluate_gate(
        selection, outcome, override=override, now=NOW
    )
    active = build_proof_rollout_status(
        policy,
        decisions=(decision,),
        overrides=(override,),
        generated_at=NOW,
    )
    expired = build_proof_rollout_status(
        policy,
        overrides=(override,),
        generated_at=NOW + timedelta(seconds=61),
    )

    assert decision.allowed is True
    assert decision.results[0].proof_status is ProofResultStatus.TIMED_OUT
    assert active["overrides"][0]["state"] == "active"
    assert active["overrides"][0]["paths"] == [PATH]
    assert active["overrides"][0]["rewrites_proof_verdict"] is False
    assert expired["overrides"][0]["state"] == "expired"


def test_supervisor_projection_keeps_bounded_rollout_diagnostics() -> None:
    status = build_proof_rollout_status(
        _policy(),
        capability_health={"hammer": {"status": "healthy", "healthy": True}},
        generated_at=NOW,
    )
    projected = apply_proof_rollout_projection(
        {"schema": "supervisor@1", "status": "running"}, status
    )

    assert projected["proof_rollout"]["snapshot_id"] == status.snapshot_id
    assert projected["proof_policy_id"] == status["policy_id"]
    assert projected["proof_rollout_mode"] == "enforcement"
    assert projected["proof_rollout_blocking"] is True
    assert projected["proof_capability_healthy"] is True

    unknown = build_proof_rollout_status(_policy(), generated_at=NOW)
    assert apply_proof_rollout_projection(
        {"schema": "supervisor@1"}, unknown
    )["proof_capability_healthy"] is False

    tampered = status.to_dict()
    tampered["proof_transcript"] = "must never enter supervisor state"
    with pytest.raises(ValueError, match="unsupported proof rollout status fields"):
        apply_proof_rollout_projection({}, tampered)
