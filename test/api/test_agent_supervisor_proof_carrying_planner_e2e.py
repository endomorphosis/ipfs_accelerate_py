from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.formal_plan_validator import (
    FormalPlanValidator,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof_carrying_planner import (
    EvidenceRole,
    ProofCarryingPlanner,
    ProofCarryingPlannerConfig,
    ProverLane,
    WorkflowAdapters,
    WorkflowNodeKind,
    WorkflowNodeStatus,
    WorkflowPersistenceError,
    WorkflowStatus,
    replay_proof_carrying_workflow,
)


def _task(
    name: str,
    *,
    depends_on: tuple[str, ...] = (),
    shared_resources: tuple[str, ...] = (),
    path: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": name,
        "task_cid": f"task:{name.lower()}",
        "goal_id": "G12.S5",
        "actor_id": f"agent:{name.lower()}",
        "depends_on": list(depends_on),
        "changed_paths": [path or f"src/{name.lower()}.py"],
        "changed_ast_scopes": [f"symbol:{name.lower()}"],
        "shared_resources": list(shared_resources),
        "acceptance_criteria": [f"{name} behavior is verified"],
        "validation_commands": [f"pytest test_{name.lower()}.py"],
    }


def _source(*tasks: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "fixture/proof-carrying-e2e@1",
        "repository_tree_id": "tree:proof-carrying-e2e",
        "objectives": [
            {
                "goal_id": "G12.S5",
                "goal_cid": "goal:g12-s5",
                "owner_actor_id": "supervisor",
                "acceptance_criteria": [
                    "all intended transitions have accepted evidence"
                ],
            }
        ],
        "tasks": list(tasks),
        "ast": [
            {
                "symbol_cid": f"symbol:{str(task['task_id']).lower()}",
                "tree_cid": "tree:proof-carrying-e2e",
                "task_cid": task["task_cid"],
            }
            for task in tasks
        ],
        "policies": [
            {
                "policy_cid": "policy:g12-s5",
                "minimum_code_assurance": "candidate",
                "fallback_check_ids": ["fallback:pytest"],
            }
        ],
    }


class _CoordinatedValidator(FormalPlanValidator):
    def __init__(
        self,
        validator_entered: threading.Event,
        proof_entered: threading.Event,
    ) -> None:
        super().__init__()
        self.validator_entered = validator_entered
        self.proof_entered = proof_entered
        self.overlapped = False

    def validate(self, *args, **kwargs):
        self.validator_entered.set()
        self.overlapped = self.proof_entered.wait(timeout=3)
        return super().validate(*args, **kwargs)


def test_complete_restartable_workflow_preserves_every_assurance_boundary(
    tmp_path: Path,
) -> None:
    # A and B are independent. B and D share an exclusive external resource.
    # C is dependency ordered behind A's accepted merge.
    source = _source(
        _task("A"),
        _task("B", shared_resources=("database:test",)),
        _task("C", depends_on=("A",)),
        _task("D", shared_resources=("database:test",)),
    )
    validator_entered = threading.Event()
    proof_entered = threading.Event()
    validator = _CoordinatedValidator(validator_entered, proof_entered)
    lock = threading.Lock()
    active_implementations: set[str] = set()
    implementation_peak = 0
    active_shared = 0
    shared_peak = 0
    merge_active = 0
    merge_peak = 0
    implementation_times: dict[str, tuple[float, float]] = {}
    merge_finished: dict[str, float] = {}
    calls: dict[str, int] = {}
    repair_packets: list[dict[str, object]] = []
    monitor_rounds: list[int] = []

    def prover(context):
        lane = context["lane"]
        with lock:
            calls[f"proof:{lane}"] = calls.get(f"proof:{lane}", 0) + 1
        if lane == ProverLane.HAMMER.value:
            proof_entered.set()
            assert validator_entered.wait(timeout=3)
            time.sleep(0.04)
            return {
                "status": "accepted",
                "accepted": True,
                "reconstructed": True,
                "assurance": "kernel_verified",  # cannot self-promote
            }
        if lane == ProverLane.LEAN.value:
            return {
                "status": "accepted",
                "accepted": True,
                "kernel_checked": True,
                "binding_verified": True,
                "kernel": "lean",
            }
        if lane == ProverLane.COQ.value:
            return {
                "status": "unavailable",
                "accepted": False,
                "reason": "Coq is absent in this matrix image",
            }
        if lane == ProverLane.LEANSTRAL_SHADOW.value:
            return {
                "status": "accepted",
                "accepted": True,
                "authoritative": True,  # adversarial provider claim
                "proposal": "by omega",
            }
        if lane == ProverLane.TEST_FALLBACK.value:
            return {
                "status": "passed",
                "accepted": True,
                "tests": ["pytest fallback"],
            }
        assert lane == ProverLane.ZKP.value
        assert context["kernel_evidence"]
        return {
            "status": "accepted",
            "accepted": True,
            "cryptographic": True,
            "simulated": False,
            "binding_verified": True,
            "backend": "fixture:real-zkp",
        }

    def dispatch(context):
        nonlocal implementation_peak, active_shared, shared_peak
        assert context["context_binding"]["bounded"] is True
        assert context["context_binding"]["used_bytes"] <= 256 * 1024
        task_id = context["task_id"]
        shared = task_id in {"task:b", "task:d"}
        started = time.monotonic()
        with lock:
            calls[f"codex:{task_id}"] = calls.get(f"codex:{task_id}", 0) + 1
            active_implementations.add(task_id)
            implementation_peak = max(
                implementation_peak, len(active_implementations)
            )
            if shared:
                active_shared += 1
                shared_peak = max(shared_peak, active_shared)
        time.sleep(0.06)
        with lock:
            active_implementations.remove(task_id)
            if shared:
                active_shared -= 1
            implementation_times[task_id] = (started, time.monotonic())
        declared = context["declared_scope"]
        return {
            "status": "accepted",
            "accepted": True,
            "changed_paths": declared["paths"],
            "changed_ast_scope_ids": declared["ast_scope_ids"],
            "validation_passed": True,
            "commit": f"commit:{task_id}",
        }

    def merge(context):
        nonlocal merge_active, merge_peak
        with lock:
            merge_active += 1
            merge_peak = max(merge_peak, merge_active)
        time.sleep(0.015)
        with lock:
            merge_active -= 1
            if (
                context.get("task_id")
                and context.get("kind") != "counterexample_repair_merge"
            ):
                merge_finished[str(context["task_id"])] = time.monotonic()
        return {
            "status": "merged",
            "accepted": True,
            "merge_id": f"merge:{context.get('task_id', 'repair')}",
        }

    def monitor(context):
        repair_round = int(context["repair_round"])
        monitor_rounds.append(repair_round)
        if repair_round == 0:
            return {
                "status": "violated",
                "violated": True,
                "counterexample": {
                    "counterexample_id": "counterexample:seeded",
                    "task_id": "task:a",
                    "property": "lease_updates_are_visible",
                    "trace": ["write", "stale-read"],
                },
            }
        return {"status": "accepted", "accepted": True}

    def repair(context):
        repair_packets.append(dict(context))
        assert context["bounded_context"] is True
        assert context["repository_wide_analysis"] is False
        assert context["context_binding"]["bounded"] is True
        assert context["counterexample_id"] == "counterexample:seeded"
        return {
            "status": "accepted",
            "accepted": True,
            "changed_paths": context["focus_scope"]["paths"],
            "changed_ast_scope_ids": context["focus_scope"]["ast_scope_ids"],
            "validation_passed": True,
            "commit": "commit:repair",
        }

    adapters = WorkflowAdapters(
        codex_dispatch=dispatch,
        prover_lane=prover,
        merge=merge,
        monitor=monitor,
        repair_dispatch=repair,
    )
    result = ProofCarryingPlanner(
        source,
        artifact_path=tmp_path,
        adapters=adapters,
        validator=validator,
        config=ProofCarryingPlannerConfig(max_workers=5),
    ).run()

    assert result.status is WorkflowStatus.COMPLETED
    assert result.complete
    assert result.merged_task_ids == ("task:a", "task:b", "task:c", "task:d")
    assert result.repaired_counterexample_ids == ("counterexample:seeded",)
    assert monitor_rounds == [0, 1]
    assert len(repair_packets) == 1
    assert validator.overlapped, "plan validation and proof work must overlap"
    assert implementation_peak >= 2
    assert shared_peak == 1
    assert merge_peak == 1
    assert implementation_times["task:c"][0] >= merge_finished["task:a"]
    assert calls == {
        "proof:hammer": 1,
        "proof:lean": 1,
        "proof:coq": 1,
        "proof:leanstral_shadow": 1,
        "proof:test_fallback": 1,
        "proof:zkp": 1,
        "codex:task:a": 1,
        "codex:task:b": 1,
        "codex:task:c": 1,
        "codex:task:d": 1,
    }

    by_role = {}
    for evidence in result.evidence:
        by_role.setdefault(evidence.role, []).append(evidence)
    assert by_role[EvidenceRole.RECONSTRUCTION][0].assurance is (
        AssuranceLevel.CANDIDATE
    )
    assert by_role[EvidenceRole.RECONSTRUCTION][0].authoritative is False
    assert any(
        item.assurance is AssuranceLevel.KERNEL_VERIFIED
        and item.authoritative
        for item in by_role[EvidenceRole.KERNEL]
    )
    assert all(
        item.authoritative is False for item in by_role[EvidenceRole.SHADOW]
    )
    assert by_role[EvidenceRole.TEST][0].authoritative_for == ("regression",)
    assert by_role[EvidenceRole.ATTESTATION][0].assurance is AssuranceLevel.ATTESTED
    assert result.authoritative_assurance is AssuranceLevel.ATTESTED
    runtime = next(
        item for item in result.evidence
        if item.role is EvidenceRole.RUNTIME_OBSERVATION
        and item.verdict.value == "accepted"
    )
    assert runtime.assurance is AssuranceLevel.UNVERIFIED
    assert runtime.artifact["runtime_observation_only"] is True
    assert runtime.artifact["proved"] is False

    json_state = json.loads(result.json_path.read_text(encoding="utf-8"))
    connection = duckdb.connect(str(result.duckdb_path), read_only=True)
    try:
        row = connection.execute(
            "SELECT artifact_digest, snapshot_json FROM workflow_snapshot"
        ).fetchone()
        decisions = connection.execute(
            "SELECT decision_json FROM workflow_decisions ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    assert row is not None
    assert row[0] == result.artifact_digest == json_state["artifact_digest"]
    assert json.loads(row[1]) == json_state
    assert [json.loads(item[0]) for item in decisions] == json_state["decisions"]

    before_calls = dict(calls)
    before_decisions = [item.to_dict() for item in result.decisions]

    def forbidden(_context):
        raise AssertionError("accepted durable nodes must not run after restart")

    restarted = ProofCarryingPlanner.restart(
        tmp_path,
        adapters=WorkflowAdapters(
            codex_dispatch=forbidden,
            prover_lane=forbidden,
            merge=forbidden,
            monitor=forbidden,
            repair_dispatch=forbidden,
        ),
        validator=validator,
    ).run()
    replay = replay_proof_carrying_workflow(tmp_path)

    assert restarted.status is WorkflowStatus.COMPLETED
    assert calls == before_calls
    assert [item.to_dict() for item in restarted.decisions] == before_decisions
    assert restarted.artifact_digest == result.artifact_digest
    assert replay.status is WorkflowStatus.COMPLETED
    assert replay.artifact_digest == restarted.artifact_digest
    assert [item.to_dict() for item in replay.decisions] == before_decisions


def test_changed_scope_escape_blocks_merge_without_assurance_promotion(
    tmp_path: Path,
) -> None:
    source = _source(_task("A"))
    merge_calls: list[dict[str, object]] = []

    result = ProofCarryingPlanner(
        source,
        artifact_path=tmp_path,
        adapters=WorkflowAdapters(
            codex_dispatch=lambda _context: {
                "status": "accepted",
                "accepted": True,
                "changed_paths": ["src/not-declared.py"],
                "changed_ast_scope_ids": ["symbol:a"],
                "validation_passed": True,
            },
            merge=lambda context: merge_calls.append(dict(context)) or True,
        ),
        config=ProofCarryingPlannerConfig(enabled_lanes=()),
    ).run()

    assert result.status is WorkflowStatus.BLOCKED
    scope = next(
        node
        for node in result.nodes
        if node.kind is WorkflowNodeKind.CHANGED_SCOPE_VERIFICATION
    )
    merge = next(
        node for node in result.nodes if node.kind is WorkflowNodeKind.MERGE
    )
    assert scope.status is WorkflowNodeStatus.REJECTED
    assert "path outside declared scope" in scope.output["violations"][0]
    assert merge.status is WorkflowNodeStatus.BLOCKED
    assert merge_calls == []
    assert result.authoritative_assurance is AssuranceLevel.CANDIDATE


def test_simulated_zkp_and_shadow_claims_cannot_upgrade_kernel_assurance(
    tmp_path: Path,
) -> None:
    source = _source(_task("A"))

    def lane(context):
        if context["lane"] == ProverLane.ZKP.value:
            return {
                "status": "accepted",
                "accepted": True,
                "cryptographic": True,
                "simulated": True,
                "binding_verified": True,
            }
        if context["lane"] == ProverLane.LEANSTRAL_SHADOW.value:
            return {
                "status": "accepted",
                "accepted": True,
                "authoritative": True,
                "assurance": "attested",
            }
        if context["lane"] == ProverLane.LEAN.value:
            return {
                "status": "accepted",
                "accepted": True,
                "kernel_checked": True,
                "binding_verified": True,
            }
        return {"status": "unavailable", "accepted": False}

    result = ProofCarryingPlanner(
        source,
        artifact_path=tmp_path,
        adapters=WorkflowAdapters(
            prover_lane=lane,
            codex_dispatch=lambda context: {
                "status": "accepted",
                "accepted": True,
                "changed_paths": context["declared_scope"]["paths"],
                "changed_ast_scope_ids": context["declared_scope"][
                    "ast_scope_ids"
                ],
                "validation_passed": True,
            },
            merge=lambda _context: True,
        ),
        config=ProofCarryingPlannerConfig(
            required_plan_assurance=AssuranceLevel.KERNEL_VERIFIED,
            enabled_lanes=(
                ProverLane.LEAN,
                ProverLane.LEANSTRAL_SHADOW,
                ProverLane.ZKP,
            )
        ),
    ).run()

    assert result.complete
    zkp = next(
        item for item in result.evidence if item.role is EvidenceRole.ATTESTATION
    )
    shadow = next(
        item for item in result.evidence if item.role is EvidenceRole.SHADOW
    )
    assert not zkp.authoritative
    assert zkp.assurance is AssuranceLevel.UNVERIFIED
    assert not shadow.authoritative
    assert shadow.assurance is AssuranceLevel.CANDIDATE
    assert result.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED


def test_paired_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    source = _source(_task("A"))
    result = ProofCarryingPlanner(
        source,
        artifact_path=tmp_path,
        adapters=WorkflowAdapters(
            codex_dispatch=lambda context: {
                "status": "accepted",
                "accepted": True,
                "changed_paths": context["declared_scope"]["paths"],
                "changed_ast_scope_ids": context["declared_scope"][
                    "ast_scope_ids"
                ],
                "validation_passed": True,
            },
            merge=lambda _context: True,
        ),
        config=ProofCarryingPlannerConfig(enabled_lanes=()),
    ).run()
    assert result.complete

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    result.json_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        WorkflowPersistenceError,
        match="projections disagree",
    ):
        replay_proof_carrying_workflow(tmp_path)
