"""Independent current-tree checks for DOEP-050 incremental plan-impact analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ImpactClosureReceipt,
    ImpactCompleteness,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.dynamic_impact_frontier import (
    INCREMENTAL_PLAN_IMPACT_SCHEMA,
    AuthoritativeImpactEvent,
    DynamicImpactFrontierAnalyzer,
    DynamicImpactFrontierError,
    FrontierObservation,
    OrdinaryRefillDisposition,
    PlanDeltaSeedOperation,
    PlanImpactNode,
    analyze_incremental_plan_impact,
    compute_minimal_plan_impact_cone,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
FRONTIER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-050.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-050.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py",
    "test/api/doep/test_doep_050_implement_incremental_plan_impact_analysis.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-050.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-050.json",
)
TASK_CID = "sha256:023861ce3c6f450138983015d6be698ee8a088b710d70a449dcc380b35271313"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:doep-050",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:doep-050",
        index_id="index:doep-050",
        model_id="model:doep-050",
        config_id="config:doep-050",
        translator_id="translator:doep-050",
        toolchain_id="toolchain:doep-050",
        policy_id="policy:doep-050",
    )


def _nodes() -> tuple[PlanImpactNode, ...]:
    return (
        PlanImpactNode(
            node_id="task:A",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:A",),
            goal_id="goal:root",
        ),
        PlanImpactNode(
            node_id="task:B",
            kind="task",
            depends_on=("task:A",),
            lifecycle="unstarted",
            receipt_refs=("receipt:B",),
            goal_id="goal:root",
        ),
        PlanImpactNode(
            node_id="task:C",
            kind="task",
            depends_on=("task:B",),
            lifecycle="ready",
            receipt_refs=("receipt:C",),
            goal_id="goal:root",
        ),
        PlanImpactNode(
            node_id="task:D",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:D",),
            goal_id="goal:other",
        ),
        PlanImpactNode(
            node_id="task:E",
            kind="task",
            depends_on=("task:B",),
            lifecycle="claimed",
            receipt_refs=("receipt:E",),
            goal_id="goal:root",
        ),
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_minimal_impact_cone_preserves_unaffected_receipts_and_upstream_nodes() -> None:
    cone = compute_minimal_plan_impact_cone(_nodes(), ("task:B",))
    assert cone.anchor_ids == ("task:B",)
    assert cone.affected_ids == ("task:B", "task:C", "task:E")
    assert cone.unaffected_preserved_ids == ("task:A", "task:D")
    assert cone.preserved_receipt_refs == ("receipt:A", "receipt:D")
    assert "receipt:B" not in cone.preserved_receipt_refs
    assert cone.depth_by_id["task:B"] == 0
    assert cone.depth_by_id["task:C"] == 1
    assert cone.depth_by_id["task:E"] == 1

    with pytest.raises(DynamicImpactFrontierError, match="unknown seed_node_ids"):
        compute_minimal_plan_impact_cone(_nodes(), ("task:missing",))


def test_ordinary_refill_is_model_free_and_history_safe_delta_seeds() -> None:
    event = AuthoritativeImpactEvent(
        event_id="event:doep-050",
        plan_epoch=1,
        plan_root="plan:doep-050",
        seed_node_ids=("task:B",),
        evidence_refs=("evidence:event",),
    )
    analysis = analyze_incremental_plan_impact(event, _nodes())
    assert analysis.schema == INCREMENTAL_PLAN_IMPACT_SCHEMA
    assert analysis.completeness is ImpactCompleteness.COMPLETE
    assert analysis.refill_decision.model_free is True
    assert (
        analysis.refill_decision.disposition
        is OrdinaryRefillDisposition.REFILL_AFFECTED_SUFFIX
    )
    assert analysis.refill_decision.candidate_task_ids == ("task:B", "task:C")
    operations = {seed.operation for seed in analysis.delta_seeds}
    assert PlanDeltaSeedOperation.AMEND_UNSTARTED_TASK in operations
    assert PlanDeltaSeedOperation.ATTACH_EVIDENCE in operations
    assert PlanDeltaSeedOperation.RECORD_UNCERTAINTY in operations
    for seed in analysis.delta_seeds:
        assert seed.completion_authoritative is False
        if seed.target_id == "task:E":
            assert seed.operation in {
                PlanDeltaSeedOperation.ATTACH_EVIDENCE,
                PlanDeltaSeedOperation.RECORD_UNCERTAINTY,
            }
    assert '"completion_authoritative": true' not in json.dumps(analysis.to_dict())

    empty = analyze_incremental_plan_impact(
        {
            "event_id": "event:empty-seed-only",
            "plan_epoch": 1,
            "plan_root": "plan:doep-050",
            "seed_node_ids": ("task:D",),
        },
        _nodes(),
    )
    # D has no dependents; refill candidates are empty because D is completed.
    assert empty.cone.affected_ids == ("task:D",)
    assert empty.refill_decision.disposition is OrdinaryRefillDisposition.NO_REFILL


def test_open_dynamic_frontier_fail_closed_blocks_complete_impact_and_refill() -> None:
    roots = _roots()
    closure = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:doep-050",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(),
        frontier_node_ids=(),
        frontier_edge_ids=(),
        evidence_refs=("evidence:closure",),
    )
    analysis = DynamicImpactFrontierAnalyzer().analyze_plan_impact(
        AuthoritativeImpactEvent(
            event_id="event:frontier",
            plan_epoch=1,
            plan_root="plan:doep-050",
            seed_node_ids=("task:B",),
            roots=roots,
            delta_id="delta:doep-050",
            evidence_refs=("evidence:event",),
        ),
        _nodes(),
        observations=(
            FrontierObservation(
                kind="reflection",
                route="src/plugin.py:load",
                affected_contract_ref="contract:plugin",
                evidence_refs=("evidence:site",),
                required=True,
            ),
        ),
        impact_closure=closure,
    )
    assert analysis.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert analysis.impact_completeness_possible is False
    assert (
        analysis.refill_decision.disposition
        is OrdinaryRefillDisposition.BLOCKED_OPEN_FRONTIER
    )
    assert analysis.delta_seeds == ()
    assert analysis.dynamic_frontier is not None
    assert analysis.dynamic_frontier.open_required_entry_ids
    assert analysis.impact_closure is not None
    assert analysis.impact_closure.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert analysis.impact_closure.frontier_node_ids


def test_routes_through_canonical_dynamic_impact_frontier_without_competing_subsystem() -> None:
    assert analyze_incremental_plan_impact.__module__.endswith(
        "analysis.dynamic_impact_frontier"
    )
    assert DynamicImpactFrontierAnalyzer.__module__.endswith(
        "analysis.dynamic_impact_frontier"
    )
    analysis = analyze_incremental_plan_impact(
        {
            "event_id": "event:roundtrip",
            "plan_epoch": 1,
            "plan_root": "plan:doep-050",
            "seed_node_ids": ["task:B"],
        },
        [node.to_dict() for node in _nodes()],
    )
    restored = type(analysis).from_dict(analysis.to_dict())
    assert restored.to_dict() == analysis.to_dict()
    assert restored.cone.affected_ids == ("task:B", "task:C", "task:E")


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-050"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "analyze_incremental_plan_impact"
    assert manifest["canonical_extension"]["carrier"] == "DynamicImpactFrontierAnalyzer"
    assert manifest["canonical_extension"]["authority"] == "model_free_advisory_impact_only"
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(FRONTIER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
