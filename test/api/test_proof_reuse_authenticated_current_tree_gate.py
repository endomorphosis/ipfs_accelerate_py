"""Authenticated 78-task current-tree gate coverage for PTR-169.

Seals ``AuthenticatedProofReuseCurrentTreeGateV5`` against the exact v9 board,
rejects historical 66/76/77-task and PTR-149 packets, requires G120/G130 child
premises, and allows only a pre-merge candidate receipt for PTR-169 itself.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
    AUTHENTICATED_CURRENT_TREE_REPAIR_EVIDENCE_REQUIREMENT,
    AUTHENTICATED_CURRENT_TREE_REPAIR_ID,
    AUTHENTICATED_CURRENT_TREE_REPAIR_PRODUCER_TASK_ID,
    AUTHENTICATED_CURRENT_TREE_REPAIR_TASK_IDS,
    AUTHENTICATED_PROOF_REUSE_CURRENT_TREE_GATE_V5_INTERFACE,
    FINAL_GATE_ACCEPTANCE_CRITERION,
    FINAL_GATE_GOAL_ID,
    FINAL_GATE_REVIEW_REVISION,
    FINAL_GATE_SATISFIED_REQUIREMENTS,
    FINAL_GATE_TASK_ID,
    HISTORICAL_66_TASK_POPULATION_COUNT,
    HISTORICAL_76_TASK_POPULATION_COUNT,
    HISTORICAL_77_TASK_POPULATION_COUNT,
    PRODUCTION_RUNTIME_ACTIVATION_ID,
    PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
    PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
    REQUIRED_CHILD_GOAL_IDS,
    REQUIRED_PTR_TASK_IDS,
    REQUIRED_SUPERVISOR_LANE_IDS,
    ROOT_ACCEPTANCE_CRITERION,
    ROOT_GOAL_ID,
    ROOT_SATISFIED_REQUIREMENTS,
    SEALED_PRODUCTION_TASK_COUNT,
    AuthenticatedProofReuseCurrentTreeGateV5,
    ProofTestReuseCurrentTreeGate,
    ProofTestReuseCurrentTreeGateError,
    build_authenticated_current_tree_repair_evidence,
    build_production_runtime_activation_evidence,
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

# Compact unit subset; production default is the full 78-task board.
TASKS = frozenset(
    {
        "PTR-001",
        "PTR-160",
        "PTR-168",
        "PTR-169",
        "PTR-170",
        "PTR-171",
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
        "PTR-G110",
        "PTR-G120",
        "PTR-G130",
    }
)
POPULATIONS = frozenset(
    {"mutation", "storage-security-concurrency", "cross-repository"}
)
ANALYZERS = frozenset(
    {"static-dependency", "runtime-dependency", "reuse-eligibility"}
)
LANES = frozenset({"ptr_lane_0", "ptr_lane_1", "ptr_lane_2"})

GIT_TREE = "tree:authenticated-current"
FOREST = "forest:authenticated-current"
COMPLETION_TREE = "completion-tree:authenticated-current"
G140_OBJECTIVE_REVISION = "objective:g140-current"
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


def _managed_merge_provenance(task_id: str) -> dict:
    return {
        "kind": "managed_merge",
        "merge_receipt_cid": f"merge:{task_id}",
        "merged_commit_id": f"commit:{task_id}",
        "merge_succeeded": True,
    }


def _pre_merge_candidate_provenance() -> dict:
    return {
        "kind": "pre_merge_candidate",
        "candidate_receipt_cid": "candidate:ptr-169",
        "candidate_commit_id": "commit:ptr-169-candidate",
        "candidate_tree_id": GIT_TREE,
        "pre_merge_candidate": True,
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
        policy_id="policy:ptr-auth",
        policy_revision="revision:v9",
        approved_stages=(
            ProofReuseRolloutStage.OFF,
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
        ),
    )
    return AuthenticatedProofReuseCurrentTreeGateV5(
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
        g110_objective_revision=G140_OBJECTIVE_REVISION,
        root_objective_revision=ROOT_OBJECTIVE_REVISION,
        rollout_policy=policy,
        required_task_ids=TASKS,
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
        required_supervisor_lane_ids=LANES,
        clock=lambda: NOW_SECONDS,
    )


def _authenticated_repair(gate) -> dict:
    return build_authenticated_current_tree_repair_evidence(
        repository_id=gate.repository_id,
        tree_id=gate.tree_id,
        commit_id=gate.commit_id,
        gitlink_state_cid=gate.gitlink_state_cid,
        repository_forest_cid=gate.repository_forest_cid,
        capability_cid=gate.capability_cid,
        verifying_key_cid=gate.verifying_key_cid,
        circuit_cid=gate.circuit_cid,
        policy_cid=gate.policy_cid,
        objective_completion_tree_id=gate.objective_completion_tree_id,
        observed_at_ms=FRESH_FROM,
        fresh_until_ms=FRESH_UNTIL,
        evidence_cid="repair:authenticated-current-tree",
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
        corpus_id="corpus:authenticated",
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
        "repair_evidence": _authenticated_repair(gate),
    }


def test_production_population_is_exact_78_task_authenticated_board() -> None:
    assert SEALED_PRODUCTION_TASK_COUNT == 78
    assert len(REQUIRED_PTR_TASK_IDS) == 78
    assert AUTHENTICATED_CURRENT_TREE_REPAIR_TASK_IDS <= REQUIRED_PTR_TASK_IDS
    assert PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS <= REQUIRED_PTR_TASK_IDS
    assert AUTHENTICATED_CURRENT_TREE_REPAIR_TASK_IDS == {
        "PTR-160",
        "PTR-161",
        "PTR-162",
        "PTR-163",
        "PTR-164",
        "PTR-165",
        "PTR-166",
        "PTR-167",
        "PTR-168",
        "PTR-169",
        "PTR-170",
        "PTR-171",
    }
    assert FINAL_GATE_TASK_ID == "PTR-169"
    assert FINAL_GATE_GOAL_ID == "PTR-G140"
    assert FINAL_GATE_ACCEPTANCE_CRITERION == (
        "ptr/authenticated-current-tree-gate-v5@1"
    )
    assert FINAL_GATE_REVIEW_REVISION == (
        "authenticated-receipt-current-tree-repair-v9"
    )
    assert FINAL_GATE_GOAL_ID not in REQUIRED_CHILD_GOAL_IDS
    assert {"PTR-G110", "PTR-G120", "PTR-G130"} <= REQUIRED_CHILD_GOAL_IDS
    assert AuthenticatedProofReuseCurrentTreeGateV5.interface == (
        AUTHENTICATED_PROOF_REUSE_CURRENT_TREE_GATE_V5_INTERFACE
    )
    assert REQUIRED_SUPERVISOR_LANE_IDS == {
        "ptr_lane_0",
        "ptr_lane_1",
        "ptr_lane_2",
    }


def test_authenticated_success_emits_g140_and_g000_evidence(
    gate, valid_packet
) -> None:
    decision = gate.evaluate(**valid_packet)
    assert decision.passed is True
    assert decision.pre_merge_candidate is False
    assert decision.reason_codes == ()
    assert decision.final_gate_completion_evidence is not None
    assert decision.root_completion_evidence is not None
    final = decision.final_gate_completion_evidence
    root = decision.root_completion_evidence
    assert final.goal_id == "PTR-G140"
    assert final.producing_task_id == "PTR-169"
    assert final.acceptance_criterion == FINAL_GATE_ACCEPTANCE_CRITERION
    assert final.satisfied_requirements == FINAL_GATE_SATISFIED_REQUIREMENTS
    assert final.to_dict()["task_count"] == 78
    assert final.to_dict()["review_revision"] == FINAL_GATE_REVIEW_REVISION
    assert root.goal_id == ROOT_GOAL_ID
    assert root.acceptance_criterion == ROOT_ACCEPTANCE_CRITERION
    assert root.satisfied_requirements == ROOT_SATISFIED_REQUIREMENTS
    assert root.producing_task_id == "PTR-169"


def test_ptr169_pre_merge_candidate_cannot_authorise_root(
    gate, valid_packet
) -> None:
    for item in valid_packet["task_evidence"]:
        if item["task_id"] == "PTR-169":
            item["task_provenance"] = _pre_merge_candidate_provenance()
            break
    else:
        pytest.fail("PTR-169 task evidence missing")

    decision = gate.evaluate(**valid_packet)
    assert decision.passed is True
    assert decision.pre_merge_candidate is True
    assert decision.candidate_receipt is not None
    assert decision.candidate_receipt["authority"] == "pre_merge_candidate"
    assert decision.candidate_receipt["task_count"] == 78
    assert decision.final_gate_completion_evidence is None
    assert decision.root_completion_evidence is None
    payload = decision.to_dict()
    assert payload["pre_merge_candidate"] is True
    assert payload["producing_task_id"] == "PTR-169"


def test_pre_merge_candidate_forbidden_for_other_tasks(
    gate, valid_packet
) -> None:
    for item in valid_packet["task_evidence"]:
        if item["task_id"] == "PTR-168":
            item["task_provenance"] = _pre_merge_candidate_provenance()
            break
    decision = gate.evaluate(**valid_packet)
    assert decision.passed is False
    assert any(
        code.startswith("pre_merge_candidate_not_allowed:PTR-168")
        for code in decision.reason_codes
    )


def test_historical_ptr149_and_stale_sealed_counts_fail_closed(
    gate, valid_packet
) -> None:
    historical = build_production_runtime_activation_evidence(
        repository_id=gate.repository_id,
        tree_id=gate.tree_id,
        commit_id=gate.commit_id,
        gitlink_state_cid=gate.gitlink_state_cid,
        repository_forest_cid=gate.repository_forest_cid,
        capability_cid=gate.capability_cid,
        verifying_key_cid=gate.verifying_key_cid,
        circuit_cid=gate.circuit_cid,
        policy_cid=gate.policy_cid,
        objective_completion_tree_id=gate.objective_completion_tree_id,
        observed_at_ms=FRESH_FROM,
        fresh_until_ms=FRESH_UNTIL,
        evidence_cid="repair:ptr149-stale",
    )
    assert historical["sealed_task_count"] == HISTORICAL_66_TASK_POPULATION_COUNT
    assert (
        historical["producer_task_id"]
        == PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID
    )
    valid_packet["repair_evidence"] = historical
    decision = gate.evaluate(**valid_packet)
    assert decision.passed is False
    assert "repair_evidence_historical_ptr149_inadmissible" in decision.reason_codes or (
        "repair_evidence_id_mismatch" in decision.reason_codes
    )

    for sealed, expected_code in (
        (HISTORICAL_66_TASK_POPULATION_COUNT, "repair_evidence_historical_66_task_population"),
        (HISTORICAL_76_TASK_POPULATION_COUNT, "repair_evidence_historical_76_task_population"),
        (HISTORICAL_77_TASK_POPULATION_COUNT, "repair_evidence_historical_77_task_population"),
    ):
        repair = _authenticated_repair(gate)
        repair["sealed_task_count"] = sealed
        valid_packet["repair_evidence"] = repair
        decision = gate.evaluate(**valid_packet)
        assert decision.passed is False
        assert (
            expected_code in decision.reason_codes
            or "repair_evidence_task_count_mismatch" in decision.reason_codes
        )


def test_production_gate_defaults_require_78_tasks() -> None:
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:prod",
        policy_revision="revision:1",
        approved_stages=(ProofReuseRolloutStage.OFF,),
    )
    gate = AuthenticatedProofReuseCurrentTreeGateV5(
        repository_id="repository:prod",
        tree_id="tree:prod",
        commit_id="commit:prod",
        gitlink_state_cid="gitlinks:prod",
        repository_forest_cid="forest:prod",
        objective_completion_tree_id="completion:prod",
        capability_cid="capability:prod",
        verifying_key_cid="key:prod",
        circuit_cid="circuit:prod",
        objective_revision="objective:prod",
        rollout_policy=policy,
        clock=lambda: NOW_SECONDS,
    )
    assert len(gate.required_task_ids) == 78
    assert gate.required_task_ids == REQUIRED_PTR_TASK_IDS
    assert "PTR-G120" in gate.required_child_goal_ids
    assert "PTR-G130" in gate.required_child_goal_ids
    assert FINAL_GATE_GOAL_ID not in gate.required_child_goal_ids


def test_g140_cannot_be_configured_as_child_premise() -> None:
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:bad",
        policy_revision="revision:1",
        approved_stages=(ProofReuseRolloutStage.OFF,),
    )
    with pytest.raises(ProofTestReuseCurrentTreeGateError, match="PTR-G140"):
        AuthenticatedProofReuseCurrentTreeGateV5(
            repository_id="repository:bad",
            tree_id="tree:bad",
            commit_id="commit:bad",
            gitlink_state_cid="gitlinks:bad",
            repository_forest_cid="forest:bad",
            objective_completion_tree_id="completion:bad",
            capability_cid="capability:bad",
            verifying_key_cid="key:bad",
            circuit_cid="circuit:bad",
            objective_revision="objective:bad",
            rollout_policy=policy,
            required_task_ids=TASKS,
            required_child_goal_ids=GOALS | {FINAL_GATE_GOAL_ID},
            required_adversarial_populations=POPULATIONS,
            required_analyzers=ANALYZERS,
            required_supervisor_lane_ids=LANES,
            clock=lambda: NOW_SECONDS,
        )


def test_adversarial_false_skips_fail_closed(gate, valid_packet) -> None:
    valid_packet["adversarial_evidence"][0]["false_skips"] = 1
    decision = gate.evaluate(**valid_packet)
    assert decision.passed is False
    assert any("false_skip_detected" in code for code in decision.reason_codes)


def test_missing_g120_or_g130_child_evidence_fails(gate, valid_packet) -> None:
    valid_packet["child_goal_evidence"] = [
        item
        for item in valid_packet["child_goal_evidence"]
        if item["goal_id"] not in {"PTR-G120", "PTR-G130"}
    ]
    decision = gate.evaluate(**valid_packet)
    assert decision.passed is False
    assert "missing_child_goal:PTR-G120" in decision.reason_codes
    assert "missing_child_goal:PTR-G130" in decision.reason_codes


def test_authenticated_repair_builder_covers_exact_wave() -> None:
    record = build_authenticated_current_tree_repair_evidence(
        repository_id="repository:x",
        tree_id="tree:x",
        commit_id="commit:x",
        gitlink_state_cid="gitlinks:x",
        repository_forest_cid="forest:x",
        capability_cid="capability:x",
        verifying_key_cid="key:x",
        circuit_cid="circuit:x",
        policy_cid="policy:x",
        observed_at_ms=FRESH_FROM,
        fresh_until_ms=FRESH_UNTIL,
        evidence_cid="repair:x",
    )
    assert record["repair_id"] == AUTHENTICATED_CURRENT_TREE_REPAIR_ID
    assert (
        record["producer_task_id"]
        == AUTHENTICATED_CURRENT_TREE_REPAIR_PRODUCER_TASK_ID
        == "PTR-169"
    )
    assert set(record["repair_task_ids"]) == AUTHENTICATED_CURRENT_TREE_REPAIR_TASK_IDS
    assert record["sealed_task_count"] == 78
    assert (
        record["requirement_id"]
        == AUTHENTICATED_CURRENT_TREE_REPAIR_EVIDENCE_REQUIREMENT
    )
    assert record["trusted_signed_receipts"] is True
    assert record["locally_verified_real_proofs"] is True
    assert record["genuine_three_repository_e2e"] is True
    assert record["forced_replay_agrees"] is True
    assert record["zero_false_skips"] is True
    assert record["benchmark_meets_threshold"] is True
    assert record["optional_capability_gaps_truthful"] is True
    assert record["review_revision"] == FINAL_GATE_REVIEW_REVISION


def test_base_gate_alias_still_constructible_for_subset() -> None:
    """Unit subsets remain usable for non-production hermetic fixtures."""

    policy = ProofReuseRolloutPolicy(
        policy_id="policy:subset",
        policy_revision="revision:1",
        approved_stages=(ProofReuseRolloutStage.OFF,),
    )
    gate = ProofTestReuseCurrentTreeGate(
        repository_id="repository:subset",
        tree_id="tree:subset",
        commit_id="commit:subset",
        gitlink_state_cid="gitlinks:subset",
        repository_forest_cid="forest:subset",
        objective_completion_tree_id="completion:subset",
        capability_cid="capability:subset",
        verifying_key_cid="key:subset",
        circuit_cid="circuit:subset",
        objective_revision="objective:subset",
        rollout_policy=policy,
        required_task_ids=TASKS,
        required_child_goal_ids=GOALS,
        required_adversarial_populations=POPULATIONS,
        required_analyzers=ANALYZERS,
        required_supervisor_lane_ids=LANES,
        clock=lambda: NOW_SECONDS,
    )
    assert isinstance(gate, ProofTestReuseCurrentTreeGate)
    assert FINAL_GATE_TASK_ID == "PTR-169"
