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
    ROOT_GOAL_ID,
    ROOT_EVIDENCE_REQUIREMENTS,
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
            merge_receipt_cid=f"merge:{task_id}",
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
        current_tree_gate_passed=True,
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
