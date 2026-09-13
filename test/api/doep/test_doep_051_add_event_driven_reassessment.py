"""Independent current-tree checks for DOEP-051 event-driven reassessment."""

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
    FrontierObservation,
    OrdinaryRefillDisposition,
    PlanDeltaSeedOperation,
    PlanImpactNode,
    analyze_incremental_plan_impact,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter import (
    EVENT_DRIVEN_REASSESSMENT_BINDING,
    EVENT_DRIVEN_REASSESSMENT_CONSUMES,
    EVENT_DRIVEN_REASSESSMENT_INTERFACE,
    EVENT_DRIVEN_REASSESSMENT_SCHEMA,
    PRODUCTION_EVENT_ADAPTER_MANIFEST,
    PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE,
    EventDrivenReassessment,
    EventDrivenReassessmentDisposition,
    EventDrivenReassessmentError,
    ProductionRefillEventAdapter,
    reassess_events,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_event_log import (
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    STALE_PLAN_EPOCH_BINDING,
    PlanRevisionStoreStalePlanEpochError,
    assert_plan_epoch_current,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/entrypoints/refill_event_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-051.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-051.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/refill_event_adapter.py",
    "test/api/doep/test_doep_051_add_event_driven_reassessment.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-051.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-051.json",
)
TASK_CID = "sha256:55028f70add58ed9a510a44754b7a2790b963ceb6e77e1201e320571ea8df13a"
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
        repository_id="repository:doep-051",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:doep-051",
        index_id="index:doep-051",
        model_id="model:doep-051",
        config_id="config:doep-051",
        translator_id="translator:doep-051",
        toolchain_id="toolchain:doep-051",
        policy_id="policy:doep-051",
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


def _event(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "event:doep-051",
        "kind": "validation_rejected",
        "plan_epoch": 1,
        "plan_root": "plan:doep-051",
        "seed_node_ids": ("task:B",),
    }
    payload.update(overrides)
    return payload


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_refill_event_adapter_without_competing_subsystem() -> None:
    assert PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE == "ProductionRefillEventAdapter@1"
    assert EVENT_DRIVEN_REASSESSMENT_BINDING == "EventDrivenReassessment@1"
    assert EVENT_DRIVEN_REASSESSMENT_INTERFACE == EVENT_DRIVEN_REASSESSMENT_BINDING
    assert EVENT_DRIVEN_REASSESSMENT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/event-driven-reassessment@1"
    )
    assert EVENT_DRIVEN_REASSESSMENT_CONSUMES == (
        IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
        STALE_PLAN_EPOCH_BINDING,
        INCREMENTAL_PLAN_IMPACT_SCHEMA,
    )
    assert ProductionRefillEventAdapter.INTERFACE == (
        PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE
    )
    assert (
        ProductionRefillEventAdapter.EVENT_DRIVEN_REASSESSMENT_BINDING
        == EVENT_DRIVEN_REASSESSMENT_BINDING
    )
    assert reassess_events.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter"
    )
    assert ProductionRefillEventAdapter.reassess.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter"
    )
    assert analyze_incremental_plan_impact.__module__.endswith(
        "analysis.dynamic_impact_frontier"
    )
    assert assert_plan_epoch_current.__module__.endswith(
        "task_sources.plan_revision_store"
    )
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "class ProductionRefillEventAdapter" in source
    assert "def to_observation(" in source
    assert "def reassess(" in source
    assert "def reassess_events(" in source
    assert "analyze_incremental_plan_impact" in source
    assert "assert_plan_epoch_current" in source
    assert "class CompetingReassessment" not in source
    assert "class ReassessmentBus" not in source
    assert "class EventDrivenPlanner" not in source
    assert "class ReassessmentEngine" not in source
    assert "CREATE TABLE" not in source


def test_production_observation_surface_is_preserved() -> None:
    adapter = ProductionRefillEventAdapter()
    manifest = adapter.manifest()
    assert manifest["schema"] == PRODUCTION_EVENT_ADAPTER_MANIFEST
    assert manifest["authorizes_append"] is False
    assert manifest["authorizes_completion"] is False
    assert manifest["database_write"] is False
    assert manifest["binding"] == EVENT_DRIVEN_REASSESSMENT_BINDING
    observation = adapter.to_observation(
        plan_root_cid="plan:doep-051",
        revision=1,
        events=({"kind": "validation_rejected"}, {"kind": "scheduler_low_water"}),
        ready_tasks=0,
        open_goals=2,
    )
    assert observation.validation_rejected is True
    assert observation.open_goals == 2
    with pytest.raises(ValueError, match="unsupported production refill events"):
        adapter.to_observation(
            plan_root_cid="plan:doep-051",
            revision=1,
            events=({"kind": "competing_bus"},),
        )


def test_authoritative_event_reassesses_impact_cone_and_preserves_unaffected() -> None:
    result = reassess_events(
        (_event(),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=3,
        ready_tasks=0,
        open_goals=1,
    )
    assert isinstance(result, EventDrivenReassessment)
    assert result.disposition is EventDrivenReassessmentDisposition.REASSESSED
    assert result.applied_event_ids == ("event:doep-051",)
    assert result.replayed_event_ids == ()
    assert result.consumed_event_ids == ("event:doep-051",)
    assert result.seed_node_ids == ("task:B",)
    assert result.observation.validation_rejected is True
    assert result.observation.open_goals == 1
    assert result.impact is not None
    assert result.impact.schema == INCREMENTAL_PLAN_IMPACT_SCHEMA
    assert result.impact.completeness is ImpactCompleteness.COMPLETE
    assert result.impact.cone.affected_ids == ("task:B", "task:C", "task:E")
    assert result.impact.cone.unaffected_preserved_ids == ("task:A", "task:D")
    assert result.impact.cone.preserved_receipt_refs == ("receipt:A", "receipt:D")
    assert result.impact.refill_decision.model_free is True
    assert (
        result.impact.refill_decision.disposition
        is OrdinaryRefillDisposition.REFILL_AFFECTED_SUFFIX
    )
    operations = {seed.operation for seed in result.impact.delta_seeds}
    assert PlanDeltaSeedOperation.AMEND_UNSTARTED_TASK in operations
    payload = result.to_dict()
    assert payload["authorizes_append"] is False
    assert payload["authorizes_completion"] is False
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["model_free"] is True
    assert payload["binding"] == EVENT_DRIVEN_REASSESSMENT_BINDING
    assert payload["carrier"] == PRODUCTION_REFILL_EVENT_ADAPTER_INTERFACE
    assert '"completion_authoritative": true' not in json.dumps(payload)
    restored = EventDrivenReassessment.from_dict(payload)
    assert restored.to_dict() == payload


def test_idempotent_replay_does_not_reapply_or_write_a_database() -> None:
    adapter = ProductionRefillEventAdapter()
    first = adapter.reassess(
        (_event(), _event(event_id="event:doep-051-dup", kind="scheduler_low_water")),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
    )
    assert first.applied_event_ids == ("event:doep-051", "event:doep-051-dup")
    replay = adapter.reassess(
        (
            _event(),
            _event(event_id="event:doep-051", kind="validation_rejected"),
        ),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
        previously_consumed_event_ids=first.consumed_event_ids,
        worker_assertion=True,
    )
    assert replay.disposition is EventDrivenReassessmentDisposition.REPLAYED
    assert replay.applied_event_ids == ()
    assert replay.replayed_event_ids == ("event:doep-051",)
    assert replay.impact is None
    assert replay.observation.validation_rejected is False
    assert replay.to_dict()["database_write"] is False
    same_batch = adapter.reassess(
        (_event(), _event()),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
    )
    assert same_batch.applied_event_ids == ("event:doep-051",)
    assert same_batch.replayed_event_ids == ("event:doep-051",)


def test_stale_plan_epoch_fails_closed_even_with_worker_assertion() -> None:
    with pytest.raises(PlanRevisionStoreStalePlanEpochError, match="stale plan epoch"):
        reassess_events(
            (_event(plan_epoch=1),),
            _nodes(),
            live_plan_epoch=2,
            plan_root_cid="plan:doep-051",
            revision=1,
            worker_assertion=True,
        )
    with pytest.raises(EventDrivenReassessmentError, match="plan_root"):
        reassess_events(
            (_event(plan_root="plan:other"),),
            _nodes(),
            live_plan_epoch=1,
            plan_root_cid="plan:doep-051",
            revision=1,
        )
    with pytest.raises(EventDrivenReassessmentError, match="seed_node_ids"):
        reassess_events(
            (_event(seed_node_ids=()),),
            _nodes(),
            live_plan_epoch=1,
            plan_root_cid="plan:doep-051",
            revision=1,
        )


def test_open_dynamic_frontier_blocks_reassessment_refill() -> None:
    roots = _roots()
    closure = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:doep-051",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(),
        frontier_node_ids=(),
        frontier_edge_ids=(),
        evidence_refs=("evidence:closure",),
    )
    result = reassess_events(
        (
            AuthoritativeImpactEvent(
                event_id="event:frontier",
                plan_epoch=1,
                plan_root="plan:doep-051",
                seed_node_ids=("task:B",),
                roots=roots,
                delta_id="delta:doep-051",
                evidence_refs=("evidence:event",),
            ),
        ),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
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
    assert result.disposition is EventDrivenReassessmentDisposition.BLOCKED
    assert result.impact is not None
    assert result.impact.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert (
        result.impact.refill_decision.disposition
        is OrdinaryRefillDisposition.BLOCKED_OPEN_FRONTIER
    )
    assert result.impact.delta_seeds == ()
    assert result.to_dict()["authorizes_append"] is False


def test_empty_events_are_model_free_no_reassessment() -> None:
    result = ProductionRefillEventAdapter().reassess(
        (),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
    )
    assert result.disposition is EventDrivenReassessmentDisposition.NO_REASSESSMENT
    assert result.impact is None
    assert result.model_free is True
    assert result.to_dict()["completion_authoritative"] is False


def test_completed_seed_without_dependents_does_not_refill() -> None:
    result = reassess_events(
        (_event(event_id="event:empty", seed_node_ids=("task:D",), kind=""),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
    )
    assert result.disposition is EventDrivenReassessmentDisposition.NO_REASSESSMENT
    assert result.impact is not None
    assert result.impact.cone.affected_ids == ("task:D",)
    assert (
        result.impact.refill_decision.disposition is OrdinaryRefillDisposition.NO_REFILL
    )


def test_routes_through_canonical_impact_analysis_without_competing_subsystem() -> None:
    analysis = analyze_incremental_plan_impact(
        {
            "event_id": "event:direct",
            "plan_epoch": 1,
            "plan_root": "plan:doep-051",
            "seed_node_ids": ["task:B"],
        },
        [node.to_dict() for node in _nodes()],
    )
    adapted = reassess_events(
        (_event(),),
        [node.to_dict() for node in _nodes()],
        live_plan_epoch=1,
        plan_root_cid="plan:doep-051",
        revision=1,
    )
    assert adapted.impact is not None
    assert adapted.impact.cone.affected_ids == analysis.cone.affected_ids
    assert DynamicImpactFrontierAnalyzer.__module__.endswith(
        "analysis.dynamic_impact_frontier"
    )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-051"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "reassess_events"
    assert manifest["canonical_extension"]["carrier"] == "ProductionRefillEventAdapter"
    assert manifest["canonical_extension"]["binding"] == EVENT_DRIVEN_REASSESSMENT_BINDING
    assert manifest["canonical_extension"]["authority"] == (
        "model_free_advisory_reassessment_only"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(ADAPTER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
