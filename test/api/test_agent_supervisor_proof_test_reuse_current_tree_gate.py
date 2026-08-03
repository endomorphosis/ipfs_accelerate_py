"""Final current-tree authority gate coverage for PTR-122."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
    FINAL_GATE_ACCEPTANCE_CRITERION,
    FINAL_GATE_GOAL_ID,
    FINAL_GATE_SATISFIED_REQUIREMENTS,
    FINAL_GATE_TASK_ID,
    PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
    PRODUCTION_RUNTIME_ACTIVATION_ID,
    PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
    PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
    REQUIRED_CHILD_GOAL_IDS,
    REQUIRED_PTR_TASK_IDS,
    REQUIRED_SUPERVISOR_LANE_IDS,
    ROOT_ACCEPTANCE_CRITERION,
    ROOT_EVIDENCE_REQUIREMENTS,
    ROOT_GOAL_ID,
    ROOT_SATISFIED_REQUIREMENTS,
    RUNTIME_ACTIVATION_REPAIR_EVIDENCE_REQUIREMENT,
    RUNTIME_ACTIVATION_REPAIR_ID,
    RUNTIME_ACTIVATION_REPAIR_TASK_IDS,
    SEALED_PRODUCTION_TASK_COUNT,
    ProofTestReuseCompletionEvidence,
    ProofTestReuseCurrentTreeGate,
    ProofTestReuseCurrentTreeGateDecision,
    ProofTestReuseCurrentTreeGateError,
    ProofTestReusePersistedGateBundle,
    verify_persisted_current_tree_gate_bundle,
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

# Small sealed subset for unit tests; production default is the full board.
TASKS = frozenset(
    {
        "PTR-001",
        "PTR-102",
        "PTR-108",
        "PTR-109",
        "PTR-110",
        "PTR-111",
        "PTR-112",
        "PTR-120",
        "PTR-121",
        "PTR-122",
        "PTR-130",
    }
)
GOALS = frozenset(
    {
        "PTR-G010",
        "PTR-G020",
        "PTR-G030",
        "PTR-G040",
        "PTR-G050",
        "PTR-G060",
        "PTR-G070",
        "PTR-G080",
        "PTR-G090",
        "PTR-G100",
    }
)
POPULATIONS = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
ANALYZERS = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)
LANES = frozenset({"ptr_lane_0", "ptr_lane_1", "ptr_lane_2"})

GIT_TREE = "tree:current"
FOREST = "forest:current"
COMPLETION_TREE = "completion-tree:current"
G110_OBJECTIVE_REVISION = "objective:g110-current"
ROOT_OBJECTIVE_REVISION = "objective:g000-current"
GRAPH_OBJECTIVE_REVISION = "objective:graph-current"


def _bound_record(**values):
    result = {
        "repository_id": "repository:current",
        "tree_id": GIT_TREE,
        "commit_id": "commit:current",
        "gitlink_state_cid": "gitlinks:recursive-current",
        "gitlink_closure_complete": True,
        "repository_forest_cid": FOREST,
        "objective_completion_tree_id": COMPLETION_TREE,
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


def _supervisor_health(gate):
    return _bound_record(
        policy_cid=gate.policy_cid,
        config_cid="config:proof-backed-test-reuse-v1",
        configuration_revision="config:proof-backed-test-reuse-v1",
        lane_count=3,
        all_lanes_healthy=True,
        evidence_cid="supervisor-health:current",
        lanes=[
            {
                "lane_id": lane_id,
                "healthy": True,
                "authority": "authoritative",
                "repository_id": gate.repository_id,
                "tree_id": gate.tree_id,
                "repository_forest_cid": gate.repository_forest_cid,
            }
            for lane_id in sorted(LANES)
        ],
    )


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
        tree_id=GIT_TREE,
        commit_id="commit:current",
        gitlink_state_cid="gitlinks:recursive-current",
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        capability_cid="capability:current",
        verifying_key_cid="key:current",
        circuit_cid="circuit:current",
        objective_revision=GRAPH_OBJECTIVE_REVISION,
        g110_objective_revision=G110_OBJECTIVE_REVISION,
        root_objective_revision=ROOT_OBJECTIVE_REVISION,
        rollout_policy=policy,
        required_task_ids=TASKS,
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
        required_supervisor_lane_ids=LANES,
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
            goal_ids=sorted(GOALS | {ROOT_GOAL_ID, FINAL_GATE_GOAL_ID}),
        ),
        "task_evidence": task_evidence,
        "child_goal_evidence": goals,
        "adversarial_evidence": adversarial,
        "analyzer_health": analyzers,
        "benchmark_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            receipt=benchmark_receipt,
            evidence_cid="benchmark:current",
        ),
        "rollout_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            decision=decision,
            promotion_evidence=promotion,
            evidence_cid="rollout:current",
        ),
        "supervisor_health_evidence": _supervisor_health(gate),
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


def test_production_population_includes_repair_and_closeout_tasks():
    for task_id in (
        "PTR-108",
        "PTR-109",
        "PTR-110",
        "PTR-111",
        "PTR-112",
        "PTR-120",
        "PTR-121",
        "PTR-122",
        "PTR-130",
    ):
        assert task_id in REQUIRED_PTR_TASK_IDS
    # PTR-149 corrects PTR-142's activation evidence and expands the sealed
    # board from 53 to exactly 60 tasks.
    assert len(REQUIRED_PTR_TASK_IDS) == SEALED_PRODUCTION_TASK_COUNT == 60
    assert RUNTIME_ACTIVATION_REPAIR_TASK_IDS <= REQUIRED_PTR_TASK_IDS
    for task_id in sorted(RUNTIME_ACTIVATION_REPAIR_TASK_IDS):
        assert task_id in REQUIRED_PTR_TASK_IDS
    assert "PTR-142" in REQUIRED_PTR_TASK_IDS
    assert PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS <= REQUIRED_PTR_TASK_IDS
    assert PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS == {
        "PTR-143",
        "PTR-144",
        "PTR-145",
        "PTR-146",
        "PTR-147",
        "PTR-148",
        "PTR-149",
    }
    assert FINAL_GATE_TASK_ID == "PTR-122"
    assert FINAL_GATE_GOAL_ID not in REQUIRED_CHILD_GOAL_IDS
    assert REQUIRED_CHILD_GOAL_IDS == {
        "PTR-G010",
        "PTR-G020",
        "PTR-G030",
        "PTR-G040",
        "PTR-G050",
        "PTR-G060",
        "PTR-G070",
        "PTR-G080",
        "PTR-G090",
        "PTR-G100",
    }
    assert REQUIRED_SUPERVISOR_LANE_IDS == {
        "ptr_lane_0",
        "ptr_lane_1",
        "ptr_lane_2",
    }
    assert RUNTIME_ACTIVATION_REPAIR_ID == "runtime-activation"
    assert (
        RUNTIME_ACTIVATION_REPAIR_EVIDENCE_REQUIREMENT
        == "ptr/runtime-activation-repair-evidence@1"
    )
    assert PRODUCTION_RUNTIME_ACTIVATION_ID == "production-runtime-activation"
    assert PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID == "PTR-149"
    assert (
        PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT
        == "ptr/production-runtime-activation-evidence@1"
    )


def test_production_gate_requires_fresh_repair_evidence(monkeypatch):
    """The full 60-task population rejects historical PTR-142 evidence."""

    policy = ProofReuseRolloutPolicy(
        policy_id="policy:ptr",
        policy_revision="revision:1",
        approved_stages=(
            ProofReuseRolloutStage.OFF,
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
        ),
    )
    gate = ProofTestReuseCurrentTreeGate(
        repository_id="repository:current",
        tree_id=GIT_TREE,
        commit_id="commit:current",
        gitlink_state_cid="gitlinks:recursive-current",
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        capability_cid="capability:current",
        verifying_key_cid="key:current",
        circuit_cid="circuit:current",
        objective_revision=GRAPH_OBJECTIVE_REVISION,
        g110_objective_revision=G110_OBJECTIVE_REVISION,
        root_objective_revision=ROOT_OBJECTIVE_REVISION,
        rollout_policy=policy,
        # Production default: exact 60-task set.
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
        required_supervisor_lane_ids=LANES,
        clock=lambda: NOW_SECONDS,
    )
    assert gate.required_task_ids == REQUIRED_PTR_TASK_IDS
    assert len(gate.required_task_ids) == 60

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
            task_provenance=(
                _planning_seal_provenance(gate)
                if task_id == "PTR-000"
                else _reviewed_integration_provenance(gate)
                if task_id in {"PTR-001", "PTR-011"}
                else _retrospective_provenance(gate)
                if task_id == "PTR-041"
                else _managed_merge_provenance(task_id)
            ),
            validation_receipt_cid=f"validation:{task_id}",
            validation_disposition="executed",
            evidence_cid=f"evidence:{task_id}",
        )
        for task_id in sorted(REQUIRED_PTR_TASK_IDS)
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
    packet = {
        "objective_graph": _bound_record(
            policy_cid=gate.policy_cid,
            objective_revision=gate.objective_revision,
            task_ids=sorted(REQUIRED_PTR_TASK_IDS),
            goal_ids=sorted(GOALS | {ROOT_GOAL_ID, FINAL_GATE_GOAL_ID}),
        ),
        "task_evidence": task_evidence,
        "child_goal_evidence": goals,
        "adversarial_evidence": adversarial,
        "analyzer_health": analyzers,
        "benchmark_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            receipt=benchmark_receipt,
            evidence_cid="benchmark:current",
        ),
        "rollout_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            decision=decision,
            promotion_evidence=promotion,
            evidence_cid="rollout:current",
        ),
        "supervisor_health_evidence": _supervisor_health(gate),
    }
    missing = gate.evaluate(**packet)
    assert missing.passed is False
    assert any(
        code.startswith("repair_evidence") for code in missing.reason_codes
    )

    # The formerly accepted PTR-142 packet is now historical only.
    packet["repair_evidence"] = _bound_record(
        policy_cid=gate.policy_cid,
        repair_id=RUNTIME_ACTIVATION_REPAIR_ID,
        repair_task_ids=sorted(RUNTIME_ACTIVATION_REPAIR_TASK_IDS),
        producer_task_id="PTR-142",
        passed=True,
        false_skips=0,
        zero_false_skip_assurance=True,
        activation_e2e_passed=True,
        requirement_id=RUNTIME_ACTIVATION_REPAIR_EVIDENCE_REQUIREMENT,
        evidence_cid="repair:runtime-activation",
    )
    historical = gate.evaluate(**packet)
    assert historical.passed is False
    assert any(
        code in {
            "repair_evidence_id_mismatch",
            "repair_evidence_producer_task_mismatch",
        }
        or code.startswith("repair_evidence_missing_task")
        for code in historical.reason_codes
    )

    packet["repair_evidence"] = _bound_record(
        policy_cid=gate.policy_cid,
        repair_id=PRODUCTION_RUNTIME_ACTIVATION_ID,
        repair_task_ids=sorted(PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS),
        producer_task_id=PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
        passed=True,
        false_skips=0,
        zero_false_skip_assurance=True,
        activation_e2e_passed=True,
        zero_injection_default_path=True,
        three_repository_cold_warm=True,
        real_groth16_certificate=True,
        measured_subprocess_benchmark=True,
        historical_activation_claims_superseded=True,
        sealed_task_count=SEALED_PRODUCTION_TASK_COUNT,
        requirement_id=PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
        evidence_cid="repair:production-runtime-activation",
    )
    admitted = gate.evaluate(**packet)
    assert admitted.passed is True
    assert admitted.final_gate_completion_evidence is not None
    assert admitted.root_completion_evidence is not None


def test_gate_rejects_g110_as_child_premise_configuration():
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:ptr",
        policy_revision="revision:1",
        approved_stages=(ProofReuseRolloutStage.OFF, ProofReuseRolloutStage.READ),
    )
    with pytest.raises(ProofTestReuseCurrentTreeGateError, match="self-reference"):
        ProofTestReuseCurrentTreeGate(
            repository_id="repository:current",
            tree_id=GIT_TREE,
            commit_id="commit:current",
            gitlink_state_cid="gitlinks:recursive-current",
            repository_forest_cid=FOREST,
            objective_completion_tree_id=COMPLETION_TREE,
            capability_cid="capability:current",
            verifying_key_cid="key:current",
            circuit_cid="circuit:current",
            objective_revision=GRAPH_OBJECTIVE_REVISION,
            rollout_policy=policy,
            required_task_ids=TASKS,
            required_child_goal_ids=GOALS | {FINAL_GATE_GOAL_ID},
            required_adversarial_populations=POPULATIONS,
            required_analyzers=ANALYZERS,
            clock=lambda: NOW_SECONDS,
        )


def test_gate_rejects_collapsed_identity_domains():
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:ptr",
        policy_revision="revision:1",
        approved_stages=(ProofReuseRolloutStage.OFF, ProofReuseRolloutStage.READ),
    )
    with pytest.raises(ProofTestReuseCurrentTreeGateError, match="distinct"):
        ProofTestReuseCurrentTreeGate(
            repository_id="repository:current",
            tree_id=GIT_TREE,
            commit_id="commit:current",
            gitlink_state_cid="gitlinks:recursive-current",
            repository_forest_cid=GIT_TREE,
            objective_completion_tree_id=COMPLETION_TREE,
            capability_cid="capability:current",
            verifying_key_cid="key:current",
            circuit_cid="circuit:current",
            objective_revision=GRAPH_OBJECTIVE_REVISION,
            rollout_policy=policy,
            required_task_ids=TASKS,
            required_child_goal_ids=GOALS,
            required_adversarial_populations=POPULATIONS,
            required_analyzers=ANALYZERS,
            clock=lambda: NOW_SECONDS,
        )


def test_success_emits_only_root_completion_evidence(gate, valid_packet):
    decision = _evaluate(gate, valid_packet)

    assert decision.passed is True
    assert decision.reason_codes == ()
    assert decision.final_gate_completion_evidence is not None
    assert decision.root_completion_evidence is not None
    assert decision.completion_evidence is decision.root_completion_evidence

    g110 = decision.final_gate_completion_evidence
    g000 = decision.root_completion_evidence

    assert g110.goal_id == FINAL_GATE_GOAL_ID
    assert g110.acceptance_criterion == FINAL_GATE_ACCEPTANCE_CRITERION
    assert g110.satisfied_requirements == FINAL_GATE_SATISFIED_REQUIREMENTS
    assert g110.producing_task_id == "PTR-122"
    assert g110.objective_revision == G110_OBJECTIVE_REVISION
    assert g110.objective_completion_tree_id == COMPLETION_TREE
    assert g110.repository_forest_cid == FOREST
    assert g110.tree_id == GIT_TREE

    assert g000.goal_id == ROOT_GOAL_ID
    assert g000.acceptance_criterion == ROOT_ACCEPTANCE_CRITERION
    assert g000.satisfied_requirements == ROOT_SATISFIED_REQUIREMENTS
    assert g000.producing_task_id == "PTR-122"
    assert g000.objective_revision == ROOT_OBJECTIVE_REVISION

    # Must not claim other root requirements by implication.
    assert g000.satisfied_requirements == (
        "ptr/cross-repository-current-tree-gate@1",
    )
    assert "ptr/zero-false-authoritative-skip@1" not in g000.satisfied_requirements
    assert "ptr/warm-reuse-benchmark@1" not in g000.satisfied_requirements
    assert "ptr/supervisor-launch-health@1" not in g000.satisfied_requirements
    assert "ptr/final-current-tree-gate@1" not in g000.satisfied_requirements
    assert g110.satisfied_requirements == ("ptr/final-current-tree-gate@1",)
    assert "ptr/cross-repository-current-tree-gate@1" not in g110.satisfied_requirements

    # Declared root requirement catalogue remains documented separately.
    assert ROOT_EVIDENCE_REQUIREMENTS == (
        "ptr/cross-repository-current-tree-gate@1",
        "ptr/zero-false-authoritative-skip@1",
        "ptr/warm-reuse-benchmark@1",
        "ptr/supervisor-launch-health@1",
    )



def test_success_emits_separate_g110_and_g000_evidence(gate, valid_packet):
    """Alias retained for PTR-122 dual-evidence documentation clarity."""
    return test_success_emits_only_root_completion_evidence(gate, valid_packet)


def test_generic_adapter_uses_allowed_producer_channel_and_freshness(
    gate, valid_packet
):
    decision = _evaluate(gate, valid_packet)
    for evidence in (
        decision.final_gate_completion_evidence,
        decision.root_completion_evidence,
    ):
        assert evidence is not None
        projected = evidence.as_completion_evidence()
        assert projected.producer_kind == "task"
        assert projected.producing_task_or_scan == "PTR-122"
        assert projected.producer_channel == evidence.producer_channel
        assert projected.channel_proof_revision == evidence.channel_proof_revision
        assert projected.objective_revision == evidence.objective_revision
        assert projected.provenance_cid == evidence.evidence_id
        assert projected.validation_passed is True
        assert projected.observed_at is not None
        assert projected.fresh_until is not None
        receipt = projected.validation_receipt
        assert isinstance(receipt, dict)
        assert receipt["producer_channel"] == evidence.producer_channel
        assert receipt["channel_proof_revision"] == evidence.channel_proof_revision
        assert isinstance(receipt["channel_proof"], dict)
        assert receipt["channel_proof"]["channel"] == evidence.producer_channel
        assert receipt["channel_proof"]["goal_id"] == evidence.goal_id
        assert projected.metadata["goal_id"] == evidence.goal_id
        assert projected.metadata["evidence_source_policy"]["satisfies"] is True
        # Evidence CID is strict CIDv1 dag-json form.
        assert evidence.evidence_id.startswith("b")
        assert ":" not in evidence.evidence_id


def test_child_goal_self_reference_is_rejected(gate, valid_packet):
    packet = deepcopy(valid_packet)
    packet["child_goal_evidence"].append(
        _bound_record(
            policy_cid=gate.policy_cid,
            goal_id=FINAL_GATE_GOAL_ID,
            status="verified_complete",
            provenance_cid="goal-evidence:g110-self",
        )
    )

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.final_gate_completion_evidence is None
    assert decision.root_completion_evidence is None
    assert any(
        "unexpected_child_goal" in reason or "self_reference" in reason
        for reason in decision.reason_codes
    )


def test_supervisor_health_is_required(gate, valid_packet):
    packet = deepcopy(valid_packet)
    packet["supervisor_health_evidence"] = {}

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert any(
        "supervisor_health" in reason for reason in decision.reason_codes
    )


def test_unhealthy_supervisor_lane_fails_closed(gate, valid_packet):
    packet = deepcopy(valid_packet)
    packet["supervisor_health_evidence"]["lanes"][0]["healthy"] = False
    packet["supervisor_health_evidence"]["all_lanes_healthy"] = True

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert any(
        "supervisor_lane_unhealthy" in reason for reason in decision.reason_codes
    )


def test_stale_supervisor_health_fails_closed(gate, valid_packet):
    packet = deepcopy(valid_packet)
    packet["supervisor_health_evidence"]["observed_at_ms"] = NOW_MS - 120_000
    packet["supervisor_health_evidence"]["fresh_until_ms"] = NOW_MS - 1

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert "supervisor_health_stale" in decision.reason_codes


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
    assert decision.final_gate_completion_evidence is None
    assert decision.root_completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


def test_quarantine_cannot_be_disguised_as_completed_managed_merge(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    packet["task_evidence"][0]["queue_status"] = "quarantined"

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.root_completion_evidence is None
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
    assert decision.root_completion_evidence is None
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
    assert decision.final_gate_completion_evidence is None
    assert decision.root_completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


@pytest.mark.parametrize(
    ("field", "reason_fragment"),
    [
        ("repository_forest_cid", "repository_forest_cid_mismatch"),
        ("policy_cid", "policy_cid_mismatch"),
        ("capability_cid", "capability_cid_mismatch"),
        ("verifying_key_cid", "verifying_key_cid_mismatch"),
        ("circuit_cid", "circuit_cid_mismatch"),
        ("objective_completion_tree_id", "objective_completion_tree_id_mismatch"),
    ],
)
def test_every_identity_is_bound_across_evidence(
    gate, valid_packet, field, reason_fragment
):
    packet = deepcopy(valid_packet)
    packet["child_goal_evidence"][0][field] = "mismatched"

    decision = _evaluate(gate, packet)

    assert decision.passed is False
    assert decision.root_completion_evidence is None
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
    assert decision.root_completion_evidence is None


def test_rollout_decision_is_evidence_not_simulated_authority(
    gate, valid_packet
):
    packet = deepcopy(valid_packet)
    packet["rollout_evidence"]["authority"] = "simulated"

    decision = _evaluate(gate, packet)

    assert not decision.passed
    assert "rollout_non_authoritative" in decision.reason_codes
    assert decision.root_completion_evidence is None


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
    assert decision.root_completion_evidence is None
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
    assert decision.root_completion_evidence is None
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
    assert decision.root_completion_evidence is None
    assert "rollout_readiness_not_pre_default" in decision.reason_codes


def test_failed_decision_cannot_be_forged_with_completion_evidence(
    gate, valid_packet
):
    evidence = _evaluate(gate, valid_packet).root_completion_evidence
    assert evidence is not None

    with pytest.raises(ProofTestReuseCurrentTreeGateError):
        ProofTestReuseCurrentTreeGateDecision(
            passed=False,
            reason_codes=("forged",),
            evaluated_at_ms=NOW_MS,
            root_completion_evidence=evidence,
        )

    with pytest.raises(ProofTestReuseCurrentTreeGateError):
        ProofTestReuseCurrentTreeGateDecision(
            passed=True,
            reason_codes=(),
            evaluated_at_ms=NOW_MS,
            final_gate_completion_evidence=None,
            root_completion_evidence=evidence,
        )

    with pytest.raises(ProofTestReuseCurrentTreeGateError):
        replace(evidence, goal_id="PTR-G050")


def test_persisted_bundle_deserializes_and_replays_gate(gate, valid_packet):
    decision = _evaluate(gate, valid_packet)
    assert decision.passed is True

    bundle = gate.persist_bundle(decision, evaluate_packet=valid_packet)
    payload = bundle.to_dict()
    restored = ProofTestReusePersistedGateBundle.from_dict(payload)

    assert restored.producing_task_id == "PTR-122"
    assert restored.git_tree_id == GIT_TREE
    assert restored.repository_forest_cid == FOREST
    assert restored.objective_completion_tree_id == COMPLETION_TREE
    assert restored.decision.passed is True
    assert (
        restored.decision.final_gate_completion_evidence.acceptance_criterion
        == FINAL_GATE_ACCEPTANCE_CRITERION
    )
    assert (
        restored.decision.root_completion_evidence.acceptance_criterion
        == ROOT_ACCEPTANCE_CRITERION
    )

    replayed = verify_persisted_current_tree_gate_bundle(
        restored,
        rollout_policy=gate.rollout_policy,
        clock=lambda: NOW_SECONDS,
        required_task_ids=TASKS,
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
    )
    assert replayed.passed is True
    assert (
        replayed.final_gate_completion_evidence.evidence_id
        == decision.final_gate_completion_evidence.evidence_id
    )
    assert (
        replayed.root_completion_evidence.evidence_id
        == decision.root_completion_evidence.evidence_id
    )


def test_persisted_bundle_rejects_tampered_premise_cid(gate, valid_packet):
    decision = _evaluate(gate, valid_packet)
    bundle = gate.persist_bundle(decision, evaluate_packet=valid_packet)
    payload = bundle.to_dict()
    payload["retained_premise_bytes"] = {
        "baguqeera" + "0" * 50: '{"tampered":true}',
    }
    # Construction itself fails closed when retained bytes do not match CID.
    with pytest.raises(ProofTestReuseCurrentTreeGateError, match="premise"):
        ProofTestReusePersistedGateBundle.from_dict(payload)
