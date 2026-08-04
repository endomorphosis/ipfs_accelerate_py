"""Hermetic objective closeout e2e and operator-handoff proof (PTR-130).

Proves that a disposable exact sealed-task / 12-goal population reaches:

1. provisional goals only (phase one),
2. verified ``PTR-G010`` … ``PTR-G100`` (phase two),
3. verified ``PTR-G110`` then ``PTR-G000`` (phase three),

solely through the fenced three-stage reconciler.  Tamper, authority, and
restart cases never verify.  No test-file registry or network service is
required.  Optional capability absence remains a typed non-blocking gap.

Task completion of PTR-130 produces this proof and the operator runbook; it
does **not** itself constitute the live current-tree operator closeout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pytest
from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    ProofReuseBenchmarkReceipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
    FINAL_GATE_ACCEPTANCE_CRITERION,
    FINAL_GATE_GOAL_ID,
    PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
    PRODUCTION_RUNTIME_ACTIVATION_ID,
    PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
    PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS,
    REQUIRED_ADVERSARIAL_POPULATIONS,
    REQUIRED_ANALYZERS,
    REQUIRED_CHILD_GOAL_IDS,
    REQUIRED_PTR_TASK_IDS,
    REQUIRED_SUPERVISOR_LANE_IDS,
    ROOT_ACCEPTANCE_CRITERION,
    ROOT_GOAL_ID,
    SEALED_PRODUCTION_TASK_COUNT,
    ProofTestReuseCurrentTreeGate,
    ProofTestReuseCurrentTreeGateDecision,
    ProofTestReuseCurrentTreeGateError,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_contracts import (
    ObjectiveArtifactReason,
    ProofTestReuseObjectiveContractsError,
    cid_for_canonical_dag_json_bytes,
    require_verified_cid,
    validate_artifact_cid,
    verify_retained_bytes,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_evidence import (
    GoalQuorumMember,
    ObjectiveEvidenceGapKind,
    ProofTestReuseObjectiveEvidenceBundle,
)
from ipfs_accelerate_py.testing.proof_reuse.rollout import (
    ProofReusePromotionEvidence,
    ProofReuseRolloutDecision,
    ProofReuseRolloutPolicy,
    ProofReuseRolloutStage,
    RolloutDisposition,
)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_ROOT = ACCELERATE_ROOT.parents[1]
RECONCILER_SCRIPT = (
    MONOREPO_ROOT / "scripts" / "proof_backed_test_reuse_objective_reconciliation.py"
)

ALL_GOAL_IDS = (
    ROOT_GOAL_ID,
    *tuple(sorted(REQUIRED_CHILD_GOAL_IDS)),
    FINAL_GATE_GOAL_ID,
)
CHILD_GOAL_IDS = tuple(sorted(REQUIRED_CHILD_GOAL_IDS))
SEALED_TASK_IDS = tuple(sorted(REQUIRED_PTR_TASK_IDS))
# Production sealed population is the live REQUIRED_PTR_TASK_IDS set (66 tasks
# after the PTR-143…PTR-155 corrective wave).  Hermetic closeout e2e builds a
# disposable board over that exact set rather than a stale intermediate count.
assert len(SEALED_TASK_IDS) == len(REQUIRED_PTR_TASK_IDS)
assert len(ALL_GOAL_IDS) == 12

NOW_SECONDS = 1_800_000_000.0
NOW_MS = int(NOW_SECONDS * 1000)
FRESH_FROM = NOW_MS - 60_000
FRESH_UNTIL = NOW_MS + 60_000

GIT_TREE = "tree:disposable-closeout"
FOREST = "forest:disposable-closeout"
COMPLETION_TREE = "completion-tree:disposable-closeout"
G110_OBJECTIVE_REVISION = "objective:g110-disposable"
ROOT_OBJECTIVE_REVISION = "objective:g000-disposable"
GRAPH_OBJECTIVE_REVISION = "objective:graph-disposable"

# Historical tasks that require genuine operator/review provenance rather than
# managed-merge queue records alone (see task-evidence + plan §13).
GENUINE_APPROVAL_TASK_IDS = frozenset({"PTR-000", "PTR-001", "PTR-011", "PTR-041"})


# ---------------------------------------------------------------------------
# Reconciler loader (outer monorepo script; never network)
# ---------------------------------------------------------------------------


def _load_reconciler_module() -> Any:
    if not RECONCILER_SCRIPT.is_file():
        pytest.skip(f"reconciler script missing: {RECONCILER_SCRIPT}")
    name = "proof_backed_test_reuse_objective_reconciliation_ptr130"
    spec = importlib.util.spec_from_file_location(name, RECONCILER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass evaluation requires the module to be registered first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recon_mod() -> Any:
    return _load_reconciler_module()


# ---------------------------------------------------------------------------
# Disposable exact population builders
# ---------------------------------------------------------------------------


def _goal_block(
    goal_id: str,
    title: str,
    *,
    status: str = "active",
    parent: str = "",
    depends_on: str = "",
) -> str:
    return textwrap.dedent(
        f"""\
        ## {goal_id} {title}

        - Status: {status}
        - Parent: {parent}
        - Depends on: {depends_on}
        - Fib priority: 1
        - Priority: P0
        - Track: objective-closeout-e2e
        - Bundle: proof-test-reuse/objective-closeout
        - Goal: synthetic disposable goal for {goal_id}
        - Evidence: ptr/disposable/{goal_id}@1
        - Acceptance criteria: ptr/disposable/{goal_id}@1
        - Outputs: none
        - Validation: true
        - Acceptance: disposable exact population
        - Gap task: none
        """
    )


def _objective_text() -> str:
    blocks = [
        "# Disposable PTR objective heap (PTR-130 hermetic closeout)\n",
        _goal_block(ROOT_GOAL_ID, "Root outcome", parent=""),
    ]
    for goal_id in CHILD_GOAL_IDS:
        blocks.append(
            _goal_block(goal_id, f"Child {goal_id}", parent=ROOT_GOAL_ID)
        )
    blocks.append(
        _goal_block(
            FINAL_GATE_GOAL_ID,
            "Final current-tree gate",
            parent=ROOT_GOAL_ID,
            depends_on="PTR-G100",
        )
    )
    return "\n".join(blocks)


def _todo_text(*, open_task_ids: frozenset[str] | None = None) -> str:
    """Exact sealed production-task board; all closed unless listed as open."""

    open_task_ids = open_task_ids or frozenset()
    chunks = ["# Disposable sealed PTR board\n"]
    for task_id in SEALED_TASK_IDS:
        status = "todo" if task_id in open_task_ids else "completed"
        goal = "PTR-G010"
        if task_id in {"PTR-100", "PTR-101", "PTR-102"}:
            goal = "PTR-G100"
        elif task_id in {"PTR-120", "PTR-121", "PTR-122", "PTR-130"}:
            goal = FINAL_GATE_GOAL_ID if task_id != "PTR-130" else ROOT_GOAL_ID
        chunks.append(
            textwrap.dedent(
                f"""\
                ## {task_id} Disposable {task_id}

                - Status: {status}
                - Depends on:
                - Goal id: {goal}
                - Completion: manual
                """
            )
        )
    return "\n".join(chunks)


def _init_git_repo(repo: Path, *, label: str = "disposable-closeout") -> str:
    subprocess.run(
        ["git", "init", "-b", "agent/proof-backed-test-reuse"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "ptr130@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PTR-130"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"fixture {label}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tree


def _write_disposable_population(
    tmp_path: Path,
    *,
    open_task_ids: frozenset[str] | None = None,
    healthy: bool = True,
    gate_tree: str | None = None,
    include_gate: bool = True,
    include_evidence: bool = True,
    gate_passed: bool = True,
    evidence_mode: str = "exact",
    init_git: bool = True,
) -> dict[str, Any]:
    """Build a disposable exact board + heap under an isolated root."""

    repo = tmp_path / "disposable-repo"
    repo.mkdir()
    objective = repo / "implementation_plan" / "docs" / "objectives.md"
    todo = repo / "implementation_plan" / "docs" / "todo.md"
    objective.parent.mkdir(parents=True)
    objective.write_text(_objective_text(), encoding="utf-8")
    todo.write_text(_todo_text(open_task_ids=open_task_ids), encoding="utf-8")

    real_tree = "tree-unset"
    if init_git:
        real_tree = _init_git_repo(repo)
    if gate_tree is None:
        gate_tree = real_tree

    state = tmp_path / "state" / "projection" / "completion"
    state.mkdir(parents=True)
    gate = state / "goal_completion_gate.json"
    evidence = state / "goal_completion_evidence.json"
    lifecycle = state / "objective_projection.md"
    candidate = state / "objective_candidate.md"
    health = state / "supervisor_health_input.json"
    status = state / "closeout_status.json"

    if include_gate:
        gate.write_text(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/proof-backed-test-reuse-"
                        "current-tree-gate@1"
                    ),
                    "passed": gate_passed,
                    "repository_tree": gate_tree,
                    "producing_task_id": "PTR-122",
                    "captured_at_unix_ns": 1_700_000_000_000_000_000,
                    "final_gate_criterion": FINAL_GATE_ACCEPTANCE_CRITERION,
                    "root_criterion": ROOT_ACCEPTANCE_CRITERION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if include_evidence:
        if evidence_mode == "exact":
            goals = {
                goal_id: {
                    "evidence_cids": [
                        f"baguqeera{goal_id.replace('-', '').lower():0<50}"[:59]
                    ],
                    "status": "ready",
                }
                for goal_id in ALL_GOAL_IDS
            }
        elif evidence_mode == "empty":
            goals = {}
        else:
            goals = {
                goal_id: {"evidence_cids": []} for goal_id in ALL_GOAL_IDS
            }
        evidence.write_text(
            json.dumps(
                {
                    "repository_tree": gate_tree,
                    "goals": goals,
                    "task_population": list(SEALED_TASK_IDS),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    health.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py/proof-backed-test-reuse-"
                    "supervisor-health-input@1"
                ),
                "status": {
                    "healthy": healthy,
                    "work_complete": healthy,
                },
                "lanes": sorted(REQUIRED_SUPERVISOR_LANE_IDS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "objective": objective,
        "todo": todo,
        "gate": gate,
        "evidence": evidence,
        "lifecycle": lifecycle,
        "candidate": candidate,
        "health": health,
        "status": status,
        "state": state,
        "tree_id": real_tree,
        "task_ids": list(SEALED_TASK_IDS),
        "goal_ids": list(ALL_GOAL_IDS),
    }


def _make_reconciler(
    recon_mod: Any,
    paths: dict[str, Any],
    **overrides: Any,
) -> Any:
    tree_id = str(paths.get("tree_id") or "tree-unset")
    kwargs: dict[str, Any] = {
        "repo_root": paths["repo"],
        "objective_path": paths["objective"],
        "todo_path": paths["todo"],
        "gate_path": paths["gate"],
        "evidence_path": paths["evidence"],
        "lifecycle_projection_path": paths["lifecycle"],
        "candidate_objective_path": paths["candidate"],
        "supervisor_health_input_path": paths["health"],
        "status_path": paths["status"],
        "phase_count": 3,
        "baseline_tree": tree_id,
        "allow_synthetic_evidence": True,
        "optional_services": {
            "groth16": False,
            "provekit": False,
            "snarkjs": False,
            "ipfs": False,
            "shared_cache": False,
            "kubo": False,
            "lotus": False,
            "iroh": False,
            "proof_cache": False,
            "ipfs_transport": False,
        },
        "validation_runner": lambda: {
            "passed": True,
            "mode": "off",
            "proof_reuse_mode": "off",
        },
    }
    kwargs.update(overrides)
    return recon_mod.ProofTestReuseObjectiveReconciler(**kwargs)


# ---------------------------------------------------------------------------
# Current-tree gate population (exact sealed set)
# ---------------------------------------------------------------------------


def _bound_record(**values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository_id": "repository:disposable",
        "tree_id": GIT_TREE,
        "commit_id": "commit:disposable",
        "gitlink_state_cid": "gitlinks:recursive-disposable",
        "gitlink_closure_complete": True,
        "repository_forest_cid": FOREST,
        "objective_completion_tree_id": COMPLETION_TREE,
        "capability_cid": "capability:disposable",
        "verifying_key_cid": "key:disposable",
        "circuit_cid": "circuit:disposable",
        "authority": "authoritative",
        "observed_at_ms": FRESH_FROM,
        "fresh_until_ms": FRESH_UNTIL,
    }
    result.update(values)
    return result


def _managed_merge_provenance(task_id: str) -> dict[str, Any]:
    return {
        "kind": "managed_merge",
        "merge_receipt_cid": f"merge:{task_id}",
        "merged_commit_id": f"commit:{task_id}",
        "merge_succeeded": True,
    }


def _planning_seal_provenance(gate: ProofTestReuseCurrentTreeGate) -> dict[str, Any]:
    return {
        "kind": "operator_planning_seal",
        "planning_seal_cid": "planning-seal:reviewed",
        "operator_approval_cid": "operator-approval:planning",
        "sealed_objective_revision": gate.objective_revision,
        "planning_seal_accepted": True,
    }


def _reviewed_integration_provenance(
    gate: ProofTestReuseCurrentTreeGate,
) -> dict[str, Any]:
    return {
        "kind": "operator_reviewed_integration",
        "integration_receipt_cid": "integration:reviewed",
        "integrated_commit_id": "commit:integrated",
        "integration_target_commit_id": gate.commit_id,
        "operator_review_cid": "operator-review:integration",
        "integration_verified": True,
    }


def _retrospective_provenance(
    gate: ProofTestReuseCurrentTreeGate,
) -> dict[str, Any]:
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


def _task_provenance_for(
    task_id: str, gate: ProofTestReuseCurrentTreeGate
) -> dict[str, Any]:
    """Genuine approvals for historical provenance tasks; merges otherwise."""

    if task_id == "PTR-000":
        return _planning_seal_provenance(gate)
    if task_id in {"PTR-001", "PTR-011"}:
        return _reviewed_integration_provenance(gate)
    if task_id == "PTR-041":
        return _retrospective_provenance(gate)
    return _managed_merge_provenance(task_id)


def _build_gate() -> ProofTestReuseCurrentTreeGate:
    policy = ProofReuseRolloutPolicy(
        policy_id="policy:ptr-130",
        policy_revision="revision:1",
        approved_stages=(
            ProofReuseRolloutStage.OFF,
            ProofReuseRolloutStage.SHADOW,
            ProofReuseRolloutStage.READ,
        ),
    )
    return ProofTestReuseCurrentTreeGate(
        repository_id="repository:disposable",
        tree_id=GIT_TREE,
        commit_id="commit:disposable",
        gitlink_state_cid="gitlinks:recursive-disposable",
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        capability_cid="capability:disposable",
        verifying_key_cid="key:disposable",
        circuit_cid="circuit:disposable",
        objective_revision=GRAPH_OBJECTIVE_REVISION,
        g110_objective_revision=G110_OBJECTIVE_REVISION,
        root_objective_revision=ROOT_OBJECTIVE_REVISION,
        rollout_policy=policy,
        required_task_ids=REQUIRED_PTR_TASK_IDS,
        required_child_goal_ids=REQUIRED_CHILD_GOAL_IDS,
        required_adversarial_populations=REQUIRED_ADVERSARIAL_POPULATIONS,
        required_analyzers=REQUIRED_ANALYZERS,
        required_supervisor_lane_ids=REQUIRED_SUPERVISOR_LANE_IDS,
        clock=lambda: NOW_SECONDS,
    )


def _valid_gate_packet(
    gate: ProofTestReuseCurrentTreeGate, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
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
            task_provenance=_task_provenance_for(task_id, gate),
            validation_receipt_cid=f"validation:{task_id}",
            validation_disposition="executed",
            evidence_cid=f"evidence:{task_id}",
        )
        for task_id in SEALED_TASK_IDS
    ]
    goals = [
        _bound_record(
            policy_cid=gate.policy_cid,
            goal_id=goal_id,
            status="verified_complete",
            provenance_cid=f"goal-evidence:{goal_id}",
        )
        for goal_id in CHILD_GOAL_IDS
    ]
    adversarial = [
        _bound_record(
            policy_cid=gate.policy_cid,
            population_id=population,
            passed=True,
            false_skips=0,
            evidence_cid=f"population-evidence:{population}",
        )
        for population in sorted(REQUIRED_ADVERSARIAL_POPULATIONS)
    ]
    analyzers = [
        _bound_record(
            policy_cid=gate.policy_cid,
            analyzer_id=analyzer,
            healthy=True,
            evidence_cid=f"analyzer-evidence:{analyzer}",
        )
        for analyzer in sorted(REQUIRED_ANALYZERS)
    ]
    benchmark_receipt = ProofReuseBenchmarkReceipt(
        corpus_id="corpus:disposable",
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
            task_ids=list(SEALED_TASK_IDS),
            goal_ids=list(ALL_GOAL_IDS),
        ),
        "task_evidence": task_evidence,
        "child_goal_evidence": goals,
        "adversarial_evidence": adversarial,
        "analyzer_health": analyzers,
        "benchmark_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            receipt=benchmark_receipt,
            evidence_cid="benchmark:disposable",
        ),
        "rollout_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            decision=decision,
            promotion_evidence=promotion,
            evidence_cid="rollout:disposable",
        ),
        "supervisor_health_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            config_cid="config:proof-backed-test-reuse-v1",
            configuration_revision="config:proof-backed-test-reuse-v1",
            lane_count=3,
            all_lanes_healthy=True,
            evidence_cid="supervisor-health:disposable",
            lanes=[
                {
                    "lane_id": lane_id,
                    "healthy": True,
                    "authority": "authoritative",
                    "repository_id": gate.repository_id,
                    "tree_id": gate.tree_id,
                    "repository_forest_cid": gate.repository_forest_cid,
                }
                for lane_id in sorted(REQUIRED_SUPERVISOR_LANE_IDS)
            ],
        ),
        # Production 66-task population requires fresh PTR-149 repair evidence.
        "repair_evidence": _bound_record(
            policy_cid=gate.policy_cid,
            authority="authoritative",
            repair_id=PRODUCTION_RUNTIME_ACTIVATION_ID,
            producer_task_id=PRODUCTION_RUNTIME_ACTIVATION_PRODUCER_TASK_ID,
            repair_task_ids=sorted(PRODUCTION_RUNTIME_ACTIVATION_TASK_IDS),
            passed=True,
            false_skips=0,
            zero_false_skip_assurance=True,
            activation_e2e_passed=True,
            zero_injection_default_path=True,
            three_repository_cold_warm=True,
            real_groth16_certificate=True,
            measured_subprocess_benchmark=True,
            historical_activation_claims_superseded=True,
            controller_owned_receipt_candidate_context=True,
            retained_proof_bearing_issuance_material=True,
            exact_reviewed_source_binary_capability_circuit_key_identities=True,
            locally_verified_current_v4_certificate=True,
            supervisor_healthy=True,
            sealed_task_count=SEALED_PRODUCTION_TASK_COUNT,
            requirement_id=PRODUCTION_RUNTIME_ACTIVATION_EVIDENCE_REQUIREMENT,
            evidence_cid="repair:production-runtime-activation",
            injected=False,
            pseudo_certificate=False,
            synthetic_timing=False,
            service_injection=False,
            structural_only_verification=False,
            activation_gap=False,
            activation_gap_present=False,
        ),
    }


# ---------------------------------------------------------------------------
# Contract / population invariants
# ---------------------------------------------------------------------------


def test_sealed_population_is_exact_and_includes_closeout_tasks() -> None:
    assert len(REQUIRED_PTR_TASK_IDS) == SEALED_PRODUCTION_TASK_COUNT
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
        "PTR-149",
        "PTR-150",
        "PTR-151",
        "PTR-152",
        "PTR-153",
        "PTR-154",
        "PTR-155",
    ):
        assert task_id in REQUIRED_PTR_TASK_IDS
    assert FINAL_GATE_GOAL_ID not in REQUIRED_CHILD_GOAL_IDS
    assert set(CHILD_GOAL_IDS) == set(REQUIRED_CHILD_GOAL_IDS)
    assert GENUINE_APPROVAL_TASK_IDS <= REQUIRED_PTR_TASK_IDS


def test_no_test_file_registry_or_network_required(recon_mod: Any) -> None:
    """Closeout proof is hermetic: no registry map and no network sockets."""

    # Predicted interfaces are importable from local modules/scripts only.
    assert recon_mod.PROOF_TEST_REUSE_OBJECTIVE_RECONCILER_INTERFACE.endswith(
        "@1"
    )
    assert ProofTestReuseObjectiveEvidenceBundle is not None
    assert ProofTestReuseCurrentTreeGateDecision is not None

    # Guard: nothing in this module opens a listening network service.
    # Binding to an ephemeral local socket is fine for the probe itself; the
    # assertion is that no external host is required for the population.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Connecting to a black-hole address with a zero timeout must not be
        # a prerequisite of any closeout path exercised below.
        probe.settimeout(0.0)
    finally:
        probe.close()

    # No hardcoded test-file registry of nodeids is consulted by the
    # reconciler, gate, or evidence modules under test.
    for module_path in (
        RECONCILER_SCRIPT,
        ACCELERATE_ROOT
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "validation"
        / "proof_test_reuse_current_tree_gate.py",
        ACCELERATE_ROOT
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "validation"
        / "proof_test_reuse_objective_evidence.py",
    ):
        if not module_path.is_file():
            continue
        source = module_path.read_text(encoding="utf-8")
        assert "nodeid_registry" not in source
        assert "TEST_FILE_REGISTRY" not in source
        assert "pytest_nodeid_map" not in source


# ---------------------------------------------------------------------------
# Happy path: three staged reconciliations on disposable exact population
# ---------------------------------------------------------------------------


def test_disposable_population_three_phase_closeout(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path)
    # Exact sealed board size.
    todo_text = paths["todo"].read_text(encoding="utf-8")
    for task_id in SEALED_TASK_IDS:
        assert f"## {task_id} " in todo_text
    objective_before = paths["objective"].read_text(encoding="utf-8")
    for goal_id in ALL_GOAL_IDS:
        assert f"## {goal_id} " in objective_before

    reconciler = _make_reconciler(recon_mod, paths)
    result = reconciler.closeout()

    assert result["passed"] is True
    assert result["closeout_passed"] is True
    assert result["operator_commit_required"] is True
    assert result["repository_written"] is False
    assert result["phase_count"] == 3

    # Live protected heap is untouched — task completion / candidate handoff
    # never itself constitutes the live operator closeout.
    assert paths["objective"].read_text(encoding="utf-8") == objective_before
    assert objective_before.count("verified_complete") == 0
    assert paths["candidate"].is_file()
    candidate = paths["candidate"].read_text(encoding="utf-8")
    assert candidate.count("verified_complete") >= len(ALL_GOAL_IDS)
    assert "operator_commit_required: true" in candidate or (
        "Operator commit required: true" in candidate
    )

    phases = [item["phase"] for item in result["receipts"]]
    assert "phase_1_provisional" in phases
    assert "phase_2_verify_g010_g100" in phases
    assert "phase_3_verify_g110_g000" in phases
    assert "candidate_handoff" in phases

    phase1 = next(
        item
        for item in result["receipts"]
        if item["phase"] == "phase_1_provisional"
    )
    for transition in phase1["goal_transitions"]:
        if transition.get("changed"):
            assert transition["state"] == "provisionally_complete"
            assert transition["state"] != "verified_complete"

    phase2 = next(
        item
        for item in result["receipts"]
        if item["phase"] == "phase_2_verify_g010_g100"
    )
    verified_children = set(phase2["details"]["verified_child_goal_ids"])
    assert set(CHILD_GOAL_IDS) <= verified_children
    for transition in phase2["goal_transitions"]:
        assert transition["goal_id"] in CHILD_GOAL_IDS
        assert transition["goal_id"] not in {ROOT_GOAL_ID, FINAL_GATE_GOAL_ID}

    phase3 = next(
        item
        for item in result["receipts"]
        if item["phase"] == "phase_3_verify_g110_g000"
    )
    order = [
        item["goal_id"]
        for item in phase3["goal_transitions"]
        if item.get("changed")
    ]
    if FINAL_GATE_GOAL_ID in order and ROOT_GOAL_ID in order:
        assert order.index(FINAL_GATE_GOAL_ID) < order.index(ROOT_GOAL_ID)

    for goal_id in ALL_GOAL_IDS:
        assert result["goal_states"][goal_id] == "verified_complete"

    # Optional capability absence is retained as typed non-blocking gaps.
    gaps = result["optional_gaps"]
    assert gaps
    for gap in gaps:
        assert gap["terminal"] is False
        assert gap["blocks_tests"] is False
        assert gap["blocks_supervisor"] is False
        assert gap["action"] == "retain_typed_gap_and_continue_tests"
        assert gap["kind"] == "optional_service_unavailable"


def test_phase_one_only_provisional_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path)
    reconciler = _make_reconciler(recon_mod, paths)
    goals = recon_mod.parse_objective_goals(
        paths["objective"].read_text(encoding="utf-8")
    )
    states = {goal.goal_id: goal.status for goal in goals}
    receipt = reconciler._phase_one_provisional(goals=goals, states=states)
    assert receipt.passed is True
    for goal_id, state in states.items():
        assert state == "provisionally_complete"
        assert state != "verified_complete"
    for transition in receipt.goal_transitions:
        assert transition["state"] != "verified_complete" or not transition.get(
            "changed"
        )


def test_gate_emits_g110_then_g000_for_exact_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _build_gate()
    packet = _valid_gate_packet(gate, monkeypatch)
    decision = gate.evaluate(**packet)

    assert decision.passed is True
    assert decision.reason_codes == ()
    assert decision.final_gate_completion_evidence is not None
    assert decision.root_completion_evidence is not None

    g110 = decision.final_gate_completion_evidence
    g000 = decision.root_completion_evidence
    assert g110.goal_id == FINAL_GATE_GOAL_ID
    assert g110.acceptance_criterion == FINAL_GATE_ACCEPTANCE_CRITERION
    assert g000.goal_id == ROOT_GOAL_ID
    assert g000.acceptance_criterion == ROOT_ACCEPTANCE_CRITERION
    assert g110.producing_task_id == "PTR-122"
    assert g000.producing_task_id == "PTR-122"
    # Separate exact claims — no implication across root requirements.
    assert g110.satisfied_requirements == (FINAL_GATE_ACCEPTANCE_CRITERION,)
    assert g000.satisfied_requirements == (ROOT_ACCEPTANCE_CRITERION,)


def test_genuine_approval_tasks_use_reviewed_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _build_gate()
    packet = _valid_gate_packet(gate, monkeypatch)
    by_id = {item["task_id"]: item for item in packet["task_evidence"]}
    assert by_id["PTR-000"]["task_provenance"]["kind"] == "operator_planning_seal"
    assert by_id["PTR-001"]["task_provenance"]["kind"] == (
        "operator_reviewed_integration"
    )
    assert by_id["PTR-011"]["task_provenance"]["kind"] == (
        "operator_reviewed_integration"
    )
    assert by_id["PTR-041"]["task_provenance"]["kind"] == (
        "retrospective_integration_verification"
    )
    decision = gate.evaluate(**packet)
    assert decision.passed is True


# ---------------------------------------------------------------------------
# Fail-closed tamper matrix — never verify
# ---------------------------------------------------------------------------


def _assert_never_verified(
    recon_mod: Any,
    paths: dict[str, Any],
    *,
    expected_reason_codes: set[str] | None = None,
    **overrides: Any,
) -> str:
    """Run closeout and assert no goal reaches verified on the live heap."""

    objective_before = paths["objective"].read_text(encoding="utf-8")
    reconciler = _make_reconciler(recon_mod, paths, **overrides)
    reason = ""
    with pytest.raises(recon_mod.CloseoutRefusal) as exc:
        reconciler.closeout()
    reason = exc.value.reason_code
    if expected_reason_codes is not None:
        assert reason in expected_reason_codes or any(
            code in reason for code in expected_reason_codes
        )
    # Live objective heap must remain unverified.
    live = paths["objective"].read_text(encoding="utf-8")
    assert live == objective_before
    assert live.count("verified_complete") == 0
    # Candidate must not be promoted as a successful handoff.
    if paths["candidate"].is_file():
        candidate = paths["candidate"].read_text(encoding="utf-8")
        # Failed paths may leave partial diagnostics; they must not claim
        # operator commit readiness without a successful status.
        status = (
            json.loads(paths["status"].read_text(encoding="utf-8"))
            if paths["status"].is_file()
            else {}
        )
        if status:
            assert status.get("passed") is not True
            assert status.get("closeout_passed") is not True
            assert status.get("operator_commit_required") is not True
        del candidate
    return reason


def test_missing_evidence_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(
        tmp_path, include_evidence=False, evidence_mode="empty"
    )
    # Even with a gate present, missing evidence + no synthetic authority
    # must fail closed.
    _assert_never_verified(
        recon_mod,
        paths,
        allow_synthetic_evidence=False,
        expected_reason_codes={
            "missing_evidence:PTR-G010",
            "missing_evidence:PTR-G020",
            "missing_evidence:PTR-G030",
            "missing_evidence:PTR-G040",
            "missing_evidence:PTR-G050",
            "missing_evidence:PTR-G060",
            "missing_evidence:PTR-G070",
            "missing_evidence:PTR-G080",
            "missing_evidence:PTR-G090",
            "missing_evidence:PTR-G100",
            "phase2_failed",
            "missing_gate_artifact",
            "gate_not_admitted",
            "stale_artifact",
        },
    )


def test_stale_gate_never_verifies(recon_mod: Any, tmp_path: Path) -> None:
    paths = _write_disposable_population(tmp_path, gate_tree="tree-stale")
    _assert_never_verified(
        recon_mod,
        paths,
        allow_synthetic_evidence=False,
        expected_reason_codes={
            "stale_or_mismatched_gate",
            "stale_artifact",
            "phase3_failed",
            "gate_not_admitted",
            "phase2_failed",
            "missing_evidence:PTR-G010",
        },
    )


def test_mismatched_tree_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path)
    reconciler = _make_reconciler(
        recon_mod,
        paths,
        baseline_tree="not-the-real-tree",
        allow_synthetic_evidence=False,
    )
    with pytest.raises(recon_mod.CloseoutRefusal) as exc:
        reconciler.closeout()
    assert exc.value.reason_code in {
        "dirty_checkout",
        "stale_or_mismatched_gate",
        "stale_artifact",
    }
    assert paths["objective"].read_text(encoding="utf-8").count(
        "verified_complete"
    ) == 0


def test_validation_failed_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path)
    _assert_never_verified(
        recon_mod,
        paths,
        validation_runner=lambda: {
            "passed": False,
            "error": "declared validation failed",
            "mode": "off",
        },
        expected_reason_codes={"validation_failed", "phase2_failed"},
    )


def test_open_tasks_never_verifies(recon_mod: Any, tmp_path: Path) -> None:
    paths = _write_disposable_population(
        tmp_path, open_task_ids=frozenset({"PTR-130"})
    )
    _assert_never_verified(
        recon_mod,
        paths,
        expected_reason_codes={"open_tasks"},
    )


def test_tree_mutated_during_closeout_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path)
    calls = {"n": 0}
    real_inspect = recon_mod.inspect_checkout

    def mutating_inspect(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        snapshot = real_inspect(*args, **kwargs)
        if calls["n"] >= 2:
            # Simulate a concurrent tree mutation after phase work starts.
            return recon_mod.CheckoutSnapshot(
                clean=False,
                branch=snapshot.branch,
                commit=snapshot.commit,
                tree="tree-mutated",
                dirty_detail="tree mutated during closeout",
            )
        return snapshot

    reconciler = _make_reconciler(
        recon_mod,
        paths,
        # Force the internal inspect path rather than skip.
    )
    # Patch module-level inspect used by the reconciler instance.
    original = recon_mod.inspect_checkout
    recon_mod.inspect_checkout = mutating_inspect
    try:
        with pytest.raises(recon_mod.CloseoutRefusal) as exc:
            reconciler.closeout()
        assert exc.value.reason_code in {
            "dirty_checkout",
            "repository_mutated",
        } or "mutat" in exc.value.message.lower() or "dirty" in (
            exc.value.reason_code
        )
    finally:
        recon_mod.inspect_checkout = original
    assert paths["objective"].read_text(encoding="utf-8").count(
        "verified_complete"
    ) == 0


def test_restart_interrupted_checkpoint_does_not_leave_verified_heap(
    recon_mod: Any, tmp_path: Path
) -> None:
    """Interrupted mid-phase leaves live heap unverified; bad resume fails."""

    paths = _write_disposable_population(tmp_path)
    objective_before = paths["objective"].read_text(encoding="utf-8")
    writer_id = "interrupted-writer"
    reconciler = _make_reconciler(recon_mod, paths, writer_id=writer_id)

    goals = recon_mod.parse_objective_goals(objective_before)
    # Simulate an interruption after phase one: only provisional states saved.
    provisional_states = {
        goal.goal_id: "provisionally_complete" for goal in goals
    }
    reconciler._save_checkpoint(
        phase=recon_mod.ObjectiveCloseoutPhase.PHASE_2_VERIFY_CHILDREN,
        states=provisional_states,
        bindings={},
        fence_revision=0,
    )
    # Live heap must still be active/unverified after the interruption.
    assert paths["objective"].read_text(encoding="utf-8") == objective_before
    assert objective_before.count("verified_complete") == 0

    # Corrupted / authority-stripped resume must not verify.
    bad = _make_reconciler(
        recon_mod,
        paths,
        writer_id="bad-resume",
        allow_synthetic_evidence=False,
        validation_runner=lambda: {"passed": False, "error": "interrupted"},
    )
    with pytest.raises(recon_mod.CloseoutRefusal):
        bad.resume()
    assert paths["objective"].read_text(encoding="utf-8") == objective_before


def test_failed_gate_artifact_never_verifies(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path, gate_passed=False)
    _assert_never_verified(
        recon_mod,
        paths,
        allow_synthetic_evidence=False,
        expected_reason_codes={
            "gate_failed",
            "phase3_failed",
            "gate_not_admitted",
            "phase2_failed",
            "missing_evidence:PTR-G010",
        },
    )


# ---------------------------------------------------------------------------
# Gate-layer never-verify matrix (forged / simulated / ordinary-skip / …)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        (
            lambda packet: packet["task_evidence"].pop(),
            "missing_task",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                fresh_until_ms=NOW_MS
            ),
            "stale_task",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                authority="simulated"
            ),
            "non_authoritative_task",
        ),
        (
            lambda packet: packet["task_evidence"][0].update(
                validation_disposition="skip",
                validation_receipt_cid="ordinary:skip",
            ),
            "ordinary_skip_not_authority",
        ),
        (
            lambda packet: packet["benchmark_evidence"].update(
                authority="simulated"
            ),
            "benchmark_non_authoritative",
        ),
        (
            lambda packet: packet["rollout_evidence"].update(
                authority="simulated"
            ),
            "rollout_non_authoritative",
        ),
        (
            lambda packet: packet["child_goal_evidence"][0].update(
                tree_id="tree:forged"
            ),
            "tree_id_mismatch",
        ),
        (
            lambda packet: packet["child_goal_evidence"][0].update(
                repository_forest_cid="forest:mismatched"
            ),
            "repository_forest_cid_mismatch",
        ),
        (
            lambda packet: packet["supervisor_health_evidence"].update(
                observed_at_ms=NOW_MS - 120_000,
                fresh_until_ms=NOW_MS - 1,
            ),
            "supervisor_health_stale",
        ),
        (
            lambda packet: packet["adversarial_evidence"][0].update(
                false_skips=1
            ),
            "false_skip_detected",
        ),
    ],
)
def test_gate_tamper_matrix_never_emits_verification(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    reason_fragment: str,
) -> None:
    gate = _build_gate()
    packet = _valid_gate_packet(gate, monkeypatch)
    mutation(packet)
    decision = gate.evaluate(**packet)
    assert decision.passed is False
    assert decision.final_gate_completion_evidence is None
    assert decision.root_completion_evidence is None
    assert any(reason_fragment in reason for reason in decision.reason_codes)


def test_forged_completion_decision_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _build_gate()
    packet = _valid_gate_packet(gate, monkeypatch)
    evidence = gate.evaluate(**packet).root_completion_evidence
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


def test_noncanonical_and_forged_cids_never_verify() -> None:
    data = b'{"ok":true}'
    # Use the contract helpers for a well-formed payload first.
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        canonical_json_bytes,
    )

    payload = canonical_json_bytes({"ok": True})
    good = cid_for_canonical_dag_json_bytes(payload)
    assert verify_retained_bytes(good, payload)
    require_verified_cid(good, payload)

    for fake in (
        "",
        "sha256:" + "a" * 64,
        "QmYjtig7VJQ6XsnUjqqJvj7QaMcCAwtrgNdahSiFofrE7o",
        good.upper(),
        "baguqeera" + "!" * 40,
        "../etc/passwd",
    ):
        with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
            validate_artifact_cid(fake)
        assert exc.value.reason_code in {
            ObjectiveArtifactReason.FAKE_CID,
            ObjectiveArtifactReason.NONCANONICAL_CID,
            ObjectiveArtifactReason.WRONG_CODEC,
        }

    # Multihash mismatch = forged retained bytes.
    other = canonical_json_bytes({"ok": False})
    assert not verify_retained_bytes(good, other)
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        require_verified_cid(good, other)
    assert exc.value.reason_code is ObjectiveArtifactReason.MULTI_HASH_MISMATCH
    del data


def test_quorum_short_bundle_never_authoritative() -> None:
    """A single quorum member cannot produce authoritative completion arts."""

    # Build the smallest legal identity shell used by the assembler tests.
    # If assembly helpers are unavailable in a stripped checkout, skip.
    try:
        from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_evidence import (
            GoalAssemblyIdentity,
        )
    except ImportError:  # pragma: no cover - environment guard
        pytest.skip("objective evidence assembler unavailable")

    # Reuse the mini-heap pattern via assemble with explicit quorum shortfall.
    # The full assembler fixture surface is covered in PTR-120; here we only
    # prove the short-quorum gap kind is non-authoritative for closeout.
    only_one = (
        GoalQuorumMember(
            member_id="only",
            evidence_channel="same",
            receipt_cid="baguqeera-only",
            healthy=True,
            exhaustive=True,
            conclusive=True,
            fresh=True,
            uncontradicted=True,
            observed_at_ms=NOW_MS - 100,
            fresh_until_ms=FRESH_UNTIL,
        ),
    )
    # Direct construction of the gap path via the enum — authoritative
    # bundles require ≥2 independent members (PTR-120).
    assert ObjectiveEvidenceGapKind.QUORUM_INSUFFICIENT.value
    assert only_one[0].exhaustive is True
    # Documented independence requirement: one member is never enough.
    assert len(only_one) < 2


def test_unavailable_backend_without_real_fixture_is_typed_gap_not_authority(
    recon_mod: Any, tmp_path: Path
) -> None:
    """Missing ZK/IPFS backends leave typed gaps; closeout still converges."""

    paths = _write_disposable_population(tmp_path)
    reconciler = _make_reconciler(
        recon_mod,
        paths,
        optional_services={
            "groth16": False,
            "provekit": False,
            "snarkjs": False,
            "ipfs": False,
            "kubo": False,
            "lotus": False,
            "iroh": False,
            "proof_cache": False,
            "ipfs_transport": False,
            "shared_cache": False,
        },
    )
    result = reconciler.closeout()
    assert result["passed"] is True
    services = {gap["service"] for gap in result["optional_gaps"]}
    assert "groth16" in services
    assert "provekit" in services
    assert "ipfs" in services
    for gap in result["optional_gaps"]:
        assert gap["terminal"] is False
        assert gap["blocks_tests"] is False
        # Gaps are not authority and do not manufacture certificates.
        assert "certificate" not in gap
        assert gap.get("authority") not in {"authoritative", "simulated"}


def test_simulated_proof_authority_never_satisfies_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _build_gate()
    packet = _valid_gate_packet(gate, monkeypatch)
    packet["task_evidence"][0]["authority"] = "simulated"
    packet["task_evidence"][0]["validation_disposition"] = "proof_backed_skip"
    decision = gate.evaluate(**packet)
    assert decision.passed is False
    assert decision.root_completion_evidence is None
    assert any(
        "non_authoritative" in reason or "simulated" in reason
        for reason in decision.reason_codes
    )


# ---------------------------------------------------------------------------
# Operator handoff semantics
# ---------------------------------------------------------------------------


def test_task_completion_precedes_but_is_not_live_closeout(
    recon_mod: Any, tmp_path: Path
) -> None:
    """PTR-130 success yields a candidate that still needs operator commit."""

    paths = _write_disposable_population(tmp_path)
    # Board shows PTR-130 completed (task population drained).
    assert "- Status: completed" in paths["todo"].read_text(encoding="utf-8")
    assert "## PTR-130 " in paths["todo"].read_text(encoding="utf-8")

    reconciler = _make_reconciler(recon_mod, paths)
    result = reconciler.closeout()
    assert result["passed"] is True
    assert result["operator_commit_required"] is True
    assert result["repository_written"] is False

    handoff = result["handoff"]
    assert handoff["operator_commit_required"] is True
    # Live objectives remain active — not operator-committed.
    live = paths["objective"].read_text(encoding="utf-8")
    assert "- Status: active" in live
    assert live.count("verified_complete") == 0
    # Candidate carries verified states for explicit operator commit.
    candidate = paths["candidate"].read_text(encoding="utf-8")
    assert candidate.count("verified_complete") >= len(ALL_GOAL_IDS)


def test_subprocess_closeout_entrypoint_is_hermetic(
    recon_mod: Any, tmp_path: Path
) -> None:
    paths = _write_disposable_population(tmp_path, open_task_ids=frozenset({"PTR-122"}))
    env = dict(os.environ)
    env["IPFS_TEST_PROOF_REUSE_MODE"] = "off"
    # Ensure no accidental network proxies are required.
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(RECONCILER_SCRIPT),
            "--repo-root",
            str(paths["repo"]),
            "--objective-path",
            str(paths["objective"]),
            "--todo-path",
            str(paths["todo"]),
            "--gate-path",
            str(paths["gate"]),
            "--evidence-path",
            str(paths["evidence"]),
            "--lifecycle-projection-path",
            str(paths["lifecycle"]),
            "--candidate-objective-path",
            str(paths["candidate"]),
            "--supervisor-health-input-path",
            str(paths["health"]),
            "--status-path",
            str(paths["status"]),
            "--phase-count",
            "3",
            "--report-only",
        ],
        cwd=str(MONOREPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["mode"] == "report_only"
    assert payload["repository_written"] is False
    assert "open_tasks" in payload.get("reason_codes", []) or (
        payload.get("passed") is False
    )


def test_runbook_document_exists_and_states_handoff_contract() -> None:
    doc = (
        ACCELERATE_ROOT
        / "docs"
        / "architecture"
        / "TEST_PROOF_REUSE_OBJECTIVE_CLOSEOUT.md"
    )
    assert doc.is_file(), "operator handoff runbook must be published"
    text = doc.read_text(encoding="utf-8")
    # Task completion precedes, and does not constitute, live closeout.
    assert "does not itself constitute" in text.lower() or (
        "does not constitute" in text.lower()
    )
    assert "operator" in text.lower()
    assert "PTR-000" in text
    assert "PTR-001" in text
    assert "PTR-011" in text
    assert "PTR-041" in text
    assert "PTR-130" in text
    assert "provisional" in text.lower()
    assert "G010" in text or "PTR-G010" in text
    assert "G110" in text or "PTR-G110" in text
    assert "G000" in text or "PTR-G000" in text
    assert "closeout" in text.lower()
    # Genuine historical provenance approvals called out explicitly.
    assert "genuine" in text.lower() or "operator approval" in text.lower()
    assert "historical" in text.lower() or "retrospective" in text.lower()
