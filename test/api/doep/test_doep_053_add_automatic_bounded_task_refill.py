"""Independent current-tree checks for DOEP-053 automatic bounded task refill."""

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
    PLAN_IMPACT_REFILL_DECISION_SCHEMA,
    AuthoritativeImpactEvent,
    FrontierObservation,
    OrdinaryRefillDisposition,
    PlanImpactNode,
    analyze_incremental_plan_impact,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller import (
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    AUTOMATIC_BOUNDED_TASK_REFILL_CONSUMES,
    AUTOMATIC_BOUNDED_TASK_REFILL_INTERFACE,
    AUTOMATIC_BOUNDED_TASK_REFILL_SCHEMA,
    PLAN_DELTA_SCHEMA,
    AutomaticBoundedRefill,
    AutomaticBoundedRefillError,
    CompletionAuthorityDecision,
    RefillController,
    RefillDisposition,
    RefillObservation,
    RefillPolicy,
    ResidualEvidence,
    ResidualGap,
    automatic_bounded_refill,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter import (
    EVENT_DRIVEN_REASSESSMENT_BINDING,
    EventDrivenReassessmentDisposition,
    ProductionRefillEventAdapter,
    reassess_events,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    STALE_PLAN_EPOCH_BINDING,
    PlanRevisionStoreStalePlanEpochError,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-053.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-053.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/refill_controller.py",
    "test/api/doep/test_doep_053_add_automatic_bounded_task_refill.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-053.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-053.json",
)
TASK_CID = "sha256:362b1ec567e529cad26c02a970b11ecadff13decf5d985f7e4c80fc0a427f518"
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


def _gap(scope: str = "scope") -> ResidualGap:
    return ResidualGap(
        "goal",
        "evidence",
        scope,
        ("goal", "root"),
        0,
        {
            "priority": "P0",
            "track": "x",
            "parallel_lane": "lane",
            "resource_class": "cpu",
        },
    )


def _controller(gaps=(), completion=None, policy=None, append_ok: bool = True):
    appended: list = []

    def evaluate(_observation, *, force_final_scan):
        assert force_final_scan
        return ResidualEvidence(
            "tree:doep-053",
            tuple(gaps),
            completion or CompletionAuthorityDecision(False),
        )

    def append(work, cas):
        appended.append((work, cas))
        return append_ok

    return RefillController(evaluate, append, policy=policy), appended


def _roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:doep-053",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:doep-053",
        index_id="index:doep-053",
        model_id="model:doep-053",
        config_id="config:doep-053",
        translator_id="translator:doep-053",
        toolchain_id="toolchain:doep-053",
        policy_id="policy:doep-053",
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
        "event_id": "event:doep-053",
        "kind": "validation_rejected",
        "plan_epoch": 1,
        "plan_root": "plan:doep-053",
        "seed_node_ids": ("task:B",),
    }
    payload.update(overrides)
    return payload


def _plan_delta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PLAN_DELTA_SCHEMA,
        "schema_version": "external-work-plan-delta/v1",
        "base_plan_revision": "DOEP-PLAN-V5",
        "triggering_event_id": "event:doep-053",
        "impacted_task_ids": ("task:B", "task:C", "task:E"),
        "preserved_task_ids": ("task:A", "task:D"),
        "preserved_receipt_ids": ("receipt:A", "receipt:D"),
        "refill_task_ids": ("task:B", "task:C"),
        "history_preserving": True,
        "model_free_refill": True,
        "completion_authoritative": False,
    }
    payload.update(overrides)
    return payload


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_refill_controller_without_competing_subsystem() -> None:
    assert AUTOMATIC_BOUNDED_TASK_REFILL_INTERFACE == "AutomaticBoundedTaskRefill@1"
    assert AUTOMATIC_BOUNDED_TASK_REFILL_BINDING == AUTOMATIC_BOUNDED_TASK_REFILL_INTERFACE
    assert AUTOMATIC_BOUNDED_TASK_REFILL_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/automatic-bounded-task-refill@1"
    )
    assert AUTOMATIC_BOUNDED_TASK_REFILL_CONSUMES == (
        EVENT_DRIVEN_REASSESSMENT_BINDING,
        INCREMENTAL_PLAN_IMPACT_SCHEMA,
        PLAN_IMPACT_REFILL_DECISION_SCHEMA,
        STALE_PLAN_EPOCH_BINDING,
        PLAN_DELTA_SCHEMA,
    )
    assert RefillController.AUTOMATIC_REFILL_INTERFACE == (
        AUTOMATIC_BOUNDED_TASK_REFILL_INTERFACE
    )
    assert (
        RefillController.AUTOMATIC_BOUNDED_TASK_REFILL_BINDING
        == AUTOMATIC_BOUNDED_TASK_REFILL_BINDING
    )
    assert automatic_bounded_refill.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller"
    )
    assert RefillController.automatic_refill.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller"
    )
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    assert "class RefillController" in source
    assert "def decide(" in source
    assert "def automatic_refill(" in source
    assert "def automatic_bounded_refill(" in source
    assert "not a second refill owner" in source
    assert "never writes DuckDB" in source
    assert "cannot skip the live plan-epoch" in source
    assert "class AutomaticRefillEngine" not in source
    assert "class CompetingRefill" not in source
    assert "class AutomaticRefillBus" not in source
    assert "CREATE TABLE" not in source


def test_automatic_refill_appends_only_affected_suffix_and_preserves_unaffected() -> None:
    controller, appended = _controller()
    reassessment = reassess_events(
        (_event(),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-053",
        revision=3,
        ready_tasks=0,
        open_goals=1,
    )
    assert reassessment.disposition is EventDrivenReassessmentDisposition.REASSESSED
    result = automatic_bounded_refill(
        reassessment.observation,
        controller=controller,
        reassessment=reassessment,
        plan_delta=_plan_delta(),
        tree_id="tree:doep-053",
    )
    assert isinstance(result, AutomaticBoundedRefill)
    assert result.disposition is RefillDisposition.REFILLED
    assert result.model_free is True
    assert result.live_plan_epoch == 1
    assert result.epoch == 1
    assert result.candidate_task_ids == ("task:B", "task:C")
    assert result.appended_task_ids == ("task:B", "task:C")
    assert result.appended_count == 2
    assert "task:E" not in result.appended_task_ids
    assert result.preserved_task_ids == ("task:A", "task:D")
    assert result.preserved_receipt_refs == ("receipt:A", "receipt:D")
    assert result.cas is not None
    assert result.cas.expected_revision == 3
    assert result.cas.epoch == 1
    assert len(appended) == 1
    work, cas = appended[0]
    assert cas.expected_revision == 3
    assert [gap.scope_cid for gap in work] == ["task:B", "task:C"]
    payload = result.to_dict()
    assert payload["authorizes_append"] is False
    assert payload["authorizes_completion"] is False
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["empty_queue_is_completion"] is False
    assert payload["append_effect"] == "revision_cas_callback_only"
    assert payload["binding"] == AUTOMATIC_BOUNDED_TASK_REFILL_BINDING
    assert payload["carrier"] == "RefillController"
    assert payload["model_free"] is True
    assert '"completion_authoritative": true' not in json.dumps(payload)

    via_method = controller.automatic_refill(
        RefillObservation("plan:doep-053", 4, open_goals=1),
        impact=reassessment.impact,
        live_plan_epoch=1,
        worker_assertion=True,
        tree_id="tree:doep-053",
    )
    assert via_method.disposition is RefillDisposition.NO_REFILL
    assert via_method.reason == "no_novel_actionable_residual"
    assert len(appended) == 1


def test_plan_delta_refill_ids_must_be_subset_and_model_free() -> None:
    controller, appended = _controller()
    with pytest.raises(AutomaticBoundedRefillError, match="subset of the impacted"):
        automatic_bounded_refill(
            RefillObservation("plan:doep-053", 1),
            controller=controller,
            plan_delta=_plan_delta(refill_task_ids=("task:missing",)),
            live_plan_epoch=1,
        )
    with pytest.raises(AutomaticBoundedRefillError, match="model-free"):
        automatic_bounded_refill(
            RefillObservation("plan:doep-053", 1),
            controller=controller,
            plan_delta=_plan_delta(model_free_refill=False),
            live_plan_epoch=1,
        )
    with pytest.raises(AutomaticBoundedRefillError, match="completion authority"):
        automatic_bounded_refill(
            RefillObservation("plan:doep-053", 1),
            controller=controller,
            plan_delta=_plan_delta(completion_authoritative=True),
            live_plan_epoch=1,
        )
    empty = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 1, ready_tasks=4),
        controller=controller,
        plan_delta=_plan_delta(refill_task_ids=()),
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert empty.disposition is RefillDisposition.NO_REFILL
    assert not appended


def test_open_frontier_and_replay_do_not_append() -> None:
    controller, appended = _controller()
    roots = _roots()
    closure = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:doep-053",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(),
        frontier_node_ids=(),
        frontier_edge_ids=(),
        evidence_refs=("evidence:closure",),
    )
    blocked = reassess_events(
        (
            AuthoritativeImpactEvent(
                event_id="event:frontier",
                plan_epoch=1,
                plan_root="plan:doep-053",
                seed_node_ids=("task:B",),
                roots=roots,
                delta_id="delta:doep-053",
                evidence_refs=("evidence:event",),
            ),
        ),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-053",
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
    assert blocked.disposition is EventDrivenReassessmentDisposition.BLOCKED
    result = automatic_bounded_refill(
        blocked.observation,
        controller=controller,
        reassessment=blocked,
        tree_id="tree:doep-053",
    )
    assert result.disposition is RefillDisposition.BLOCKED
    assert result.reason == "reassessment_blocked"
    assert not appended

    adapter = ProductionRefillEventAdapter()
    first = adapter.reassess(
        (_event(),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-053",
        revision=1,
    )
    replay = adapter.reassess(
        (_event(),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-053",
        revision=1,
        previously_consumed_event_ids=first.consumed_event_ids,
        worker_assertion=True,
    )
    replayed = automatic_bounded_refill(
        replay.observation,
        controller=controller,
        reassessment=replay,
        live_plan_epoch=1,
        worker_assertion=True,
        tree_id="tree:doep-053",
    )
    assert replayed.disposition is RefillDisposition.NO_REFILL
    assert replayed.reason == "replayed_event"
    assert not appended


def test_stale_plan_epoch_fails_closed_even_with_worker_assertion() -> None:
    controller, appended = _controller()
    impact = analyze_incremental_plan_impact(
        {
            "event_id": "event:stale",
            "plan_epoch": 1,
            "plan_root": "plan:doep-053",
            "seed_node_ids": ["task:B"],
        },
        _nodes(),
    )
    with pytest.raises(PlanRevisionStoreStalePlanEpochError, match="stale plan epoch"):
        automatic_bounded_refill(
            RefillObservation("plan:doep-053", 1, open_goals=1),
            controller=controller,
            impact=impact,
            live_plan_epoch=2,
            worker_assertion=True,
        )
    with pytest.raises(AutomaticBoundedRefillError, match="plan_root"):
        automatic_bounded_refill(
            RefillObservation("plan:other", 1),
            controller=controller,
            impact=impact,
            live_plan_epoch=1,
        )
    with pytest.raises(AutomaticBoundedRefillError, match="requires reassessment"):
        automatic_bounded_refill(
            RefillObservation("plan:doep-053", 1),
            controller=controller,
            live_plan_epoch=1,
        )
    assert not appended


def test_epoch_and_per_wave_bounds_are_enforced() -> None:
    controller, appended = _controller(policy=RefillPolicy(max_new_work_per_epoch=1))
    impact = analyze_incremental_plan_impact(
        {
            "event_id": "event:bounded",
            "plan_epoch": 1,
            "plan_root": "plan:doep-053",
            "seed_node_ids": ["task:B"],
        },
        _nodes(),
    )
    assert impact.refill_decision.candidate_task_ids == ("task:B", "task:C")
    first = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 1, open_goals=1),
        controller=controller,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert first.disposition is RefillDisposition.REFILLED
    assert first.appended_count == 1
    assert first.appended_task_ids == ("task:B",)
    assert first.to_dict()["wave"] == 1
    second = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 2, open_goals=1),
        controller=controller,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert second.disposition is RefillDisposition.REFILLED
    assert second.appended_task_ids == ("task:C",)
    assert second.epoch == 2
    capped, _ = _controller(policy=RefillPolicy(max_epochs=1, max_new_work_per_epoch=1))
    automatic_bounded_refill(
        RefillObservation("plan:doep-053", 1, open_goals=1),
        controller=capped,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    exhausted = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 2, open_goals=1),
        controller=capped,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert exhausted.disposition is RefillDisposition.BLOCKED
    assert exhausted.reason == "epoch_budget_exhausted"


def test_empty_queue_is_not_completion_and_claimed_tasks_are_not_refilled() -> None:
    controller, appended = _controller()
    empty = reassess_events(
        (_event(event_id="event:empty", seed_node_ids=("task:D",), kind=""),),
        _nodes(),
        live_plan_epoch=1,
        plan_root_cid="plan:doep-053",
        revision=1,
        ready_tasks=0,
        active_tasks=0,
        open_goals=0,
    )
    result = automatic_bounded_refill(
        empty.observation,
        controller=controller,
        reassessment=empty,
        tree_id="tree:doep-053",
    )
    assert empty.observation.ready_tasks == 0
    assert result.disposition is RefillDisposition.NO_REFILL
    assert result.to_dict()["empty_queue_is_completion"] is False
    assert result.to_dict()["completion_authoritative"] is False
    assert not appended

    impact = analyze_incremental_plan_impact(
        {
            "event_id": "event:claimed",
            "plan_epoch": 1,
            "plan_root": "plan:doep-053",
            "seed_node_ids": ["task:B"],
        },
        _nodes(),
    )
    assert "task:E" not in impact.refill_decision.candidate_task_ids
    assert (
        impact.refill_decision.disposition
        is OrdinaryRefillDisposition.REFILL_AFFECTED_SUFFIX
    )
    refilled = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 1, ready_tasks=0, open_goals=1),
        controller=controller,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert refilled.disposition is RefillDisposition.REFILLED
    assert "task:E" not in refilled.appended_task_ids
    assert "task:A" not in refilled.appended_task_ids


def test_existing_residual_decide_path_is_preserved() -> None:
    controller, appended = _controller((_gap(),))
    result = controller.decide(RefillObservation("plan:doep-053", 4, open_goals=1))
    assert result.disposition is RefillDisposition.REFILLED
    assert result.appended_count == 1
    assert result.cas is not None
    assert result.cas.expected_revision == 4
    assert len(appended) == 1
    healthy, untouched = _controller()
    skipped = healthy.decide(RefillObservation("plan:doep-053", 1, ready_tasks=2))
    assert skipped.disposition is RefillDisposition.NO_REFILL
    assert not untouched


def test_cas_conflict_and_stale_completion_do_not_advance_epoch() -> None:
    conflicting, appended = _controller(append_ok=False)
    impact = analyze_incremental_plan_impact(
        {
            "event_id": "event:cas",
            "plan_epoch": 1,
            "plan_root": "plan:doep-053",
            "seed_node_ids": ["task:B"],
        },
        _nodes(),
    )
    conflict = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 9, open_goals=1),
        controller=conflicting,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert conflict.disposition is RefillDisposition.CAS_CONFLICT
    assert conflict.epoch == 0
    assert len(appended) == 1
    stale, untouched = _controller()
    reopened = automatic_bounded_refill(
        RefillObservation("plan:doep-053", 1, stale_evidence=True, open_goals=1),
        controller=stale,
        impact=impact,
        live_plan_epoch=1,
        tree_id="tree:doep-053",
    )
    assert reopened.disposition is RefillDisposition.REOPEN_CONVERGENCE
    assert not untouched


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-053"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "automatic_bounded_refill"
    assert manifest["canonical_extension"]["carrier"] == "RefillController"
    assert manifest["canonical_extension"]["binding"] == AUTOMATIC_BOUNDED_TASK_REFILL_BINDING
    assert manifest["canonical_extension"]["authority"] == (
        "model_free_automatic_affected_suffix_cas_append_only"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(CONTROLLER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
