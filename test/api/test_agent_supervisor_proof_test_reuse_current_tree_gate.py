"""Final current-tree authority gate coverage for PTR-102."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
    ROOT_EVIDENCE_REQUIREMENTS,
    ROOT_GOAL_ID,
    ProofTestReuseCompletionEvidence,
    ProofTestReuseCurrentTreeGate,
    ProofTestReuseCurrentTreeGateDecision,
    ProofTestReuseCurrentTreeGateError,
)
from ipfs_accelerate_py.testing.proof_reuse.rollout import (
    ProofReusePromotionEvidence,
    ProofReuseRolloutDecision,
    ProofReuseRolloutPolicy,
    ProofReuseRolloutStage,
    RolloutDisposition,
)

NOW_SECONDS = 1_800_000_000.0
NOW_MS = int(NOW_SECONDS * 1000)
FRESH_FROM = NOW_MS - 60_000
FRESH_UNTIL = NOW_MS + 60_000
TASKS = frozenset({"PTR-001", "PTR-102"})
GOALS = frozenset({"PTR-G010", "PTR-G110"})
POPULATIONS = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
ANALYZERS = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)


def _bound_record(**values):
    result = {
        "repository_id": "repository:current",
        "tree_id": "tree:current",
        "commit_id": "commit:current",
        "gitlink_state_cid": "gitlinks:recursive-current",
        "gitlink_closure_complete": True,
        "repository_forest_cid": "forest:current",
        "capability_cid": "capability:current",
        "verifying_key_cid": "key:current",
        "circuit_cid": "circuit:current",
        "authority": "authoritative",
        "observed_at_ms": FRESH_FROM,
        "fresh_until_ms": FRESH_UNTIL,
    }
    result.update(values)
    return result


def _managed_merge_provenance(task_id):
    return {
        "kind": "managed_merge",
        "merge_receipt_cid": f"merge:{task_id}",
        "merged_commit_id": f"commit:{task_id}",
        "merge_succeeded": True,
    }


def _planning_seal_provenance(gate):
    return {
        "kind": "operator_planning_seal",
        "planning_seal_cid": "planning-seal:reviewed",
        "operator_approval_cid": "operator-approval:planning",
        "sealed_objective_revision": gate.objective_revision,
        "planning_seal_accepted": True,
    }


def _reviewed_integration_provenance(gate):
    return {
        "kind": "operator_reviewed_integration",
        "integration_receipt_cid": "integration:reviewed",
        "integrated_commit_id": "commit:integrated",
        "integration_target_commit_id": gate.commit_id,
        "operator_review_cid": "operator-review:integration",
        "integration_verified": True,
    }


def _retrospective_provenance(gate):
    return {
        "kind": "retrospective_integration_verification",
        "integrated_commit_id": "commit:historically-integrated",
        "ancestry_target_commit_id": gate.commit_id,
        "ancestry_receipt_cid": "ancestry:verified",
        "ancestry_verified": True,
        "current_tree_rerun_receipt_cid": "rerun:current-tree",
        "current_tree_rerun_repository_id": gate.repository_id,
        "current_tree_rerun_tree_id": gate.tree_id,
        "current_tree_rerun_commit_id": gate.commit_id,
        "current_tree_rerun_gitlink_state_cid": gate.gitlink_state_cid,
        "current_tree_rerun_repository_forest_cid": gate.repository_forest_cid,
        "current_tree_rerun_policy_cid": gate.policy_cid,
        "current_tree_rerun_capability_cid": gate.capability_cid,
        "current_tree_rerun_verifying_key_cid": gate.verifying_key_cid,
        "current_tree_rerun_circuit_cid": gate.circuit_cid,
        "current_tree_rerun_passed": True,
        "policy_approval_cid": "policy-approval:retrospective",
        "approved_policy_cid": gate.policy_cid,
        "policy_approved": True,
    }


@pytest.fixture()
def gate():
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:ptr",
        policy_revision="revision:1",
        approved_stages=(
            ProofReuseRolloutStage.OFF,
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
        ),
    )
    return ProofTestReuseCurrentTreeGate(
        repository_id="repository:current",
        tree_id="tree:current",
        commit_id="commit:current",
        gitlink_state_cid="gitlinks:recursive-current",
        repository_forest_cid="forest:current",
        capability_cid="capability:current",
        verifying_key_cid="key:current",
        circuit_cid="circuit:current",
        objective_revision="objective:current",
        rollout_policy=policy,
        required_task_ids=TASKS,
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
        clock=lambda: NOW_SECONDS,
    )


@pytest.fixture()
def valid_packet(gate, monkeypatch):
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.validation."
        "proof_test_reuse_current_tree_gate.verify_benchmark_receipt",
        lambda receipt: receipt.passed,
    )
    task_evidence = [
        _bound_record(
            policy_cid=gate.policy_cid,
            task_id=task_id,
            status="complete",
            task_cid=f"task-cid:{task_id}",
            task_provenance=_managed_merge_provenance(task_id),
            validation_receipt_cid=f"validation:{task_id}",
            validation_disposition="executed",
            evidence_cid=f"evidence:{task_id}",
        )
        for task_id in sorted(TASKS)
    ]
    goals = [
        _bound_record(
            policy_cid=gate.policy_cid,
            goal_id=goal_id,
            status="verified_complete",
            provenance_cid=f"goal-evidence:{goal_id}",
        )
        for goal_id in sorted(GOALS)
    ]
    adversarial = [
        _bound_record(
            policy_cid=gate.policy_cid,
            population_id=population,
            passed=True,
            false_skips=0,
            evidence_cid=f"population-evidence:{population}",
        )
        for population in sorted(POPULATIONS)
    ]
    analyzers = [
        _bound_record(
            policy_cid=gate.policy_cid,
            analyzer_id=analyzer,
            healthy=True,
            evidence_cid=f"analyzer-evidence:{analyzer}",
        )
        for analyzer in sorted(ANALYZERS)
    ]
    benchmark_receipt = ProofReuseBenchmarkReceipt(
        corpus_id="corpus:current",
        false_admissions=0,
        warm_eligible_count=1,
        warm_verified_skips=1,
        warm_skip_bps=10_000,
        passed=True,
    )
    promotion = ProofReusePromotionEvidence(
        observed_at=datetime.fromtimestamp(FRESH_FROM / 1000, tz=UTC),
        repository_id=gate.repository_id,
        tree_id=gate.tree_id,
        policy_id=gate.rollout_policy.policy_id,
        policy_revision=gate.rollout_policy.policy_revision,
        current_stage=ProofReuseRolloutStage.SHADOW,
        target_stage=ProofReuseRolloutStage.READ,
        mutation_false_skips=0,
        degradation_false_skips=0,
        authority_contradictions=0,
        corruption_spike=False,
        stale_keys=0,
        key_health_ok=True,
        revocation_health_ok=True,
        controlled_issuer=True,
        current_tree_gate_passed=None,
        all_repositories_passed=True,
    )
    decision = ProofReuseRolloutDecision(
        current_stage=ProofReuseRolloutStage.SHADOW,
        requested_stage=ProofReuseRolloutStage.READ,
        effective_stage=ProofReuseRolloutStage.READ,
        disposition=RolloutDisposition.PROMOTE,
        gates=(),
        evidence_id=promotion.evidence_id,
        policy_id=gate.rollout_policy.policy_id,
        policy_revision=gate.rollout_policy.policy_revision,
    )
    return {
        "objective_graph": _bound_record(
            policy_cid=gate.policy_cid,
            objective_revision=gate.objective_revision,
            task_ids=sorted(TASKS),
            goal_ids=sorted(GOALS | {ROOT_GOAL_ID}),
        ),
        "task_evidence": task_evidence,
        "child_goal_evidence": goals,
        "adversarial_evidence": adversarial,
        "analyzer_health": analyzers,
        "benchmark_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            receipt=benchmark_receipt,
        ),
        "rollout_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            decision=decision,
            promotion_evidence=promotion,
        ),
    }


def _evaluate(gate, packet):
    return gate.evaluate(**packet)


def _replace_rollout_promotion(packet, promotion, **decision_changes):
    packet["rollout_evidence"]["promotion_evidence"] = promotion
    packet["rollout_evidence"]["decision"] = replace(
        packet["rollout_evidence"]["decision"],
        evidence_id=promotion.evidence_id,
        **decision_changes,
    )


def test_success_emits_only_root_completion_evidence(gate, valid_packet):
    decision = _evaluate(gate, valid_packet)

    assert decision.passed is True
    assert decision.reason_codes == ()
    assert isinstance(decision.completion_evidence, ProofTestReuseCompletionEvidence)
    assert decision.completion_evidence.goal_id == "PTR-G000"
    assert decision.completion_evidence.producing_task_id == "PTR-102"
    assert decision.completion_evidence.repository_forest_cid == "forest:current"
    assert decision.completion_evidence.policy_cid == gate.policy_cid
    assert (
        decision.completion_evidence.satisfied_requirements
        == ROOT_EVIDENCE_REQUIREMENTS
    )
    assert decision.completion_evidence.fresh_until_ms == FRESH_UNTIL
    assert (
        decision.completion_evidence.as_completion_evidence().provenance_cid
        == decision.completion_evidence.evidence_id
    )


@pytest.mark.parametrize(
    "provenance_factory",
    [
        lambda gate: _managed_merge_provenance("PTR-001"),
        _planning_seal_provenance,
        _reviewed_integration_provenance,
        _retrospective_provenance,
    ],
)
def test_closed_task_provenance_union_accepts_each_reviewed_success_path(
    gate, valid_packet, provenance_factory
):
    packet = deepcopy(valid_packet)
    packet["task_evidence"][0]["task_provenance"] = provenance_factory(gate)

    decision = _evaluate(gate, packet)

    assert decision.passed is True
    assert decision.reason_codes == ()


@pytest.mark.parametrize(
    ("provenance", "reason_fragment"),
    [
        (None, "missing_task_provenance"),
        (
            {
                "kind": "quarantine",
                "quarantine_receipt_cid": "quarantine:not-success",
            },
            "unsupported_task_provenance",
        ),
        (
            {
                **_managed_merge_provenance("PTR-001"),
                "merge_succeeded": False,
            },
            "unsuccessful_task_provenance",
        ),
        (
            {
                **_managed_merge_provenance("PTR-001"),
                "unreviewed_extension": "must-fail-closed",
            },
            "malformed_task_provenance",
        ),
    ],
)
def test_task_provenance_union_rejects_missing_unknown_or_unsuccessful_members(
    gate, valid_packet, provenance, reason_fragment
):
    packet = deepcopy(valid_packet)
    packet["task_evidence"][0]["task_provenance"] = provenance

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


def test_quarantine_cannot_be_disguised_as_completed_managed_merge(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    packet["task_evidence"][0]["queue_status"] = "quarantined"

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert any(
        "quarantined_task" in reason for reason in decision.reason_codes
    )


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        (
            lambda provenance: provenance.update(ancestry_verified=False),
            "unsuccessful_task_provenance",
        ),
        (
            lambda provenance: provenance.update(
                current_tree_rerun_passed=False
            ),
            "unsuccessful_task_provenance",
        ),
        (
            lambda provenance: provenance.update(policy_approved=False),
            "unsuccessful_task_provenance",
        ),
        (
            lambda provenance: provenance.update(
                ancestry_target_commit_id="commit:not-current"
            ),
            "retrospective_ancestry_target_mismatch",
        ),
        (
            lambda provenance: provenance.update(
                current_tree_rerun_tree_id="tree:not-current"
            ),
            "retrospective_current_tree_rerun_binding_mismatch",
        ),
        (
            lambda provenance: provenance.update(
                approved_policy_cid="policy:not-approved"
            ),
            "retrospective_policy_approval_mismatch",
        ),
    ],
)
def test_retrospective_integration_requires_ancestry_rerun_and_policy_evidence(
    gate, valid_packet, mutation, reason_fragment
):
    packet = deepcopy(valid_packet)
    provenance = _retrospective_provenance(gate)
    mutation(provenance)
    packet["task_evidence"][0]["task_provenance"] = provenance

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        (lambda packet: packet["task_evidence"].pop(), "missing_task"),
        (
            lambda packet: packet["task_evidence"][0].update(status="in_progress"),
            "open_task",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                fresh_until_ms=NOW_MS
            ),
            "stale_task",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                validation_disposition="skip",
                validation_receipt_cid="ordinary:skip",
            ),
            "ordinary_skip_not_authority",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                authority="simulated"
            ),
            "non_authoritative_task",
        ),
        (
            lambda packet: packet["adversarial_evidence"][0].update(
                false_skips=1
            ),
            "false_skip_detected",
        ),
        (
            lambda packet: packet["analyzer_health"][0].update(healthy=False),
            "analyzer_unhealthy",
        ),
        (
            lambda packet: packet["benchmark_evidence"].update(
                observed_at_ms=NOW_MS - 120_000,
                fresh_until_ms=NOW_MS - 1,
            ),
            "benchmark_stale",
        ),
        (
            lambda packet: packet["benchmark_evidence"].update(
                authority="simulated"
            ),
            "benchmark_non_authoritative",
        ),
    ],
)
def test_gate_fails_closed_without_emitting_evidence(
    gate, valid_packet, mutation, reason_fragment
):
    packet = deepcopy(valid_packet)
    mutation(packet)

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


@pytest.mark.parametrize(
    ("field", "reason_fragment"),
    [
        ("repository_forest_cid", "repository_forest_cid_mismatch"),
        ("policy_cid", "policy_cid_mismatch"),
        ("capability_cid", "capability_cid_mismatch"),
        ("verifying_key_cid", "verifying_key_cid_mismatch"),
        ("circuit_cid", "circuit_cid_mismatch"),
    ],
)
def test_every_identity_is_bound_across_evidence(
    gate, valid_packet, field, reason_fragment
):
    packet = deepcopy(valid_packet)
    packet["child_goal_evidence"][0][field] = "mismatched"

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


def test_historical_or_unverified_benchmark_is_not_authority(
    gate, valid_packet, monkeypatch
):
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.validation."
        "proof_test_reuse_current_tree_gate.verify_benchmark_receipt",
        lambda receipt: False,
    )

    decision = _evaluate(gate, valid_packet)

    assert not decision.passed
    assert "benchmark_not_reverified" in decision.reason_codes
    assert decision.completion_evidence is None


def test_rollout_decision_is_evidence_not_simulated_authority(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    packet["rollout_evidence"]["authority"] = "simulated"

    decision = _evaluate(gate, packet)

    assert not decision.passed
    assert "rollout_non_authoritative" in decision.reason_codes
    assert decision.completion_evidence is None


def test_explicit_false_current_tree_flag_is_valid_pre_default_readiness(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    promotion = replace(
        packet["rollout_evidence"]["promotion_evidence"],
        current_tree_gate_passed=False,
    )
    _replace_rollout_promotion(packet, promotion)

    decision = _evaluate(gate, packet)

    assert decision.passed is True


def test_rollout_cannot_preclaim_the_gate_result(gate, valid_packet):
    packet = deepcopy(valid_packet)
    promotion = replace(
        packet["rollout_evidence"]["promotion_evidence"],
        current_tree_gate_passed=True,
    )
    _replace_rollout_promotion(packet, promotion)

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert "rollout_current_tree_gate_preclaimed" in decision.reason_codes


def test_rollout_readiness_must_be_fresh_by_its_own_timestamp(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    promotion = replace(
        packet["rollout_evidence"]["promotion_evidence"],
        observed_at=datetime.fromtimestamp(
            NOW_SECONDS - gate.rollout_policy.max_evidence_age_seconds - 1,
            tz=UTC,
        ),
    )
    _replace_rollout_promotion(packet, promotion)

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert "rollout_readiness_stale" in decision.reason_codes


def test_eligible_default_decision_is_not_pre_default_readiness(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    promotion = replace(
        packet["rollout_evidence"]["promotion_evidence"],
        target_stage=ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
    )
    _replace_rollout_promotion(
        packet,
        promotion,
        requested_stage=ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
        effective_stage=ProofReuseRolloutStage.ELIGIBLE_DEFAULT,
    )

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.completion_evidence is None
    assert "rollout_readiness_not_pre_default" in decision.reason_codes


def test_failed_decision_cannot_be_forged_with_completion_evidence(
    gate, valid_packet
):
    evidence = _evaluate(gate, valid_packet).completion_evidence
    assert evidence is not None

    with pytest.raises(ProofTestReuseCurrentTreeGateError):
        ProofTestReuseCurrentTreeGateDecision(
            passed=False,
            reason_codes=("forged",),
            evaluated_at_ms=NOW_MS,
            completion_evidence=evidence,
        )

    with pytest.raises(ProofTestReuseCurrentTreeGateError):
        replace(evidence, goal_id="PTR-G110")
