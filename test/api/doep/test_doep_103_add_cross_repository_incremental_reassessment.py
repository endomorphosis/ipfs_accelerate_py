"""Independent current-tree checks for DOEP-103 cross-repository incremental reassessment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.dynamic_impact_frontier import (
    INCREMENTAL_PLAN_IMPACT_SCHEMA,
    OrdinaryRefillDisposition,
    PlanImpactNode,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_event_adapter import (
    EVENT_DRIVEN_REASSESSMENT_BINDING,
    EventDrivenReassessmentDisposition,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import (
    ALLOWED_SIBLING_CAPABILITIES,
    CANONICAL_EVENT_INTERFACE,
    CANONICAL_EVENT_SCHEMA_ID,
    CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING,
    CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_CAPABILITIES,
    CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_CONSUMES,
    CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_INTERFACE,
    CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_SCHEMA,
    CROSS_REPOSITORY_OWNERS,
    CROSS_SUPERVISOR_RECEIPT_BINDING,
    CROSS_SUPERVISOR_RECEIPT_CAPABILITY,
    CROSS_SUPERVISOR_RECEIPT_SCHEMA,
    DATABASE_EVENT_LOG_INTERFACE,
    DEFAULT_LOCAL_REPOSITORY,
    FORBIDDEN_SIBLING_CAPABILITIES,
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
    PLAN_DELTA_SCHEMA,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    STALE_PLAN_EPOCH_BINDING,
    SUPERVISOR_FABRIC_INTERFACE,
    CrossRepositoryIncrementalReassessment,
    CrossRepositoryIncrementalReassessmentDisposition,
    CrossRepositoryIncrementalReassessmentError,
    CrossSupervisorReceiptError,
    SiblingSupervisorCapabilityRegistryError,
    SiblingSupervisorEventAdmission,
    SupervisorFabric,
    SupervisorFabricError,
    admit_cross_supervisor_receipt,
    issue_fence,
    reassess_cross_repository,
    register_sibling_supervisor_capability,
    validate_sibling_supervisor_event,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionStoreStalePlanEpochError,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
FABRIC_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-103.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-103.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py",
    "test/api/doep/test_doep_103_add_cross_repository_incremental_reassessment.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-103.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-103.json",
)
TASK_CID = "sha256:352be477ff961381e8ce80cfdc8ea259cc78362b2d566a20bd26683ccdc7134d"
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
EVIDENCE_DIGEST = "sha256:" + hashlib.sha256(b"doep-103-independent-evidence").hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _nodes() -> tuple[PlanImpactNode, ...]:
    return (
        PlanImpactNode(
            node_id="task:datasets-schema",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:datasets-schema",),
            goal_id="goal:semantic",
        ),
        PlanImpactNode(
            node_id="task:accelerate-adapter",
            kind="task",
            depends_on=("task:datasets-schema",),
            lifecycle="unstarted",
            receipt_refs=("receipt:accelerate-adapter",),
            goal_id="goal:local",
        ),
        PlanImpactNode(
            node_id="task:accelerate-follow",
            kind="task",
            depends_on=("task:accelerate-adapter",),
            lifecycle="ready",
            receipt_refs=("receipt:accelerate-follow",),
            goal_id="goal:local",
        ),
        PlanImpactNode(
            node_id="task:kit-bytes",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:kit-bytes",),
            goal_id="goal:durable",
        ),
        PlanImpactNode(
            node_id="task:accelerate-other",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:accelerate-other",),
            goal_id="goal:other",
        ),
    )


def _node_repositories() -> dict[str, str]:
    return {
        "task:datasets-schema": "ipfs_datasets_py",
        "task:accelerate-adapter": "ipfs_accelerate_py",
        "task:accelerate-follow": "ipfs_accelerate_py",
        "task:kit-bytes": "ipfs_kit_py",
        "task:accelerate-other": "ipfs_accelerate_py",
    }


def _canonical_event(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "validation_rejected",
        "plan_epoch": 1,
        "plan_root": "plan:doep-103",
        "seed_node_ids": ["task:datasets-schema"],
        "outcome": "rejected",
        "attempt": 1,
    }
    values: dict[str, Any] = {
        "schema": CANONICAL_EVENT_SCHEMA_ID,
        "event_id": "event:sibling-103",
        "event_type": "task.validation.completed",
        "stream_id": "task:DOEP-103",
        "causal_parent_ids": ["event:parent-103"],
        "correlation_id": "correlation:attempt-103",
        "causation_id": "causation:reassessment-103",
        "payload": payload,
    }
    if "payload" in overrides:
        merged = dict(payload)
        merged.update(overrides.pop("payload"))
        values["payload"] = merged
    values.update(overrides)
    return values


def _receipt_envelope(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema": CROSS_SUPERVISOR_RECEIPT_SCHEMA,
        "receipt_id": "receipt:sibling-103",
        "task_id": "task:datasets-schema",
        "request_id": "request:DOEP-101",
        "carrier_event_id": "event:sibling-103",
        "outcome": "admitted",
        "evidence_digest": EVIDENCE_DIGEST,
        "payload": {
            "kind": "validation_rejected",
            "plan_epoch": 1,
            "plan_root": "plan:doep-103",
            "seed_node_ids": ["task:datasets-schema"],
        },
    }
    if "payload" in overrides:
        merged = dict(values["payload"])
        merged.update(overrides.pop("payload"))
        values["payload"] = merged
    values.update(overrides)
    return values


def _record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:datasets",
        "local_repository": "ipfs_accelerate_py",
        "sibling_repository": "ipfs_datasets_py",
        "capability": "event-exchange",
        "epoch": 2,
        "effect": "event_exchange",
        "live_plan_epoch": 1,
        "plan_root_cid": "plan:doep-103",
        "revision": 3,
        "nodes": _nodes(),
        "node_repositories": _node_repositories(),
        "event": _canonical_event(),
    }
    values.update(overrides)
    return values


def _capability_record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:datasets",
        "capability": "event-exchange",
        "epoch": 2,
        "effect": "event_exchange",
    }
    values.update(overrides)
    return values


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_supervisor_fabric_without_competing_subsystem() -> None:
    assert SUPERVISOR_FABRIC_INTERFACE == "SupervisorFabric@1"
    assert SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING == (
        "SiblingSupervisorEventValidation@1"
    )
    assert SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING == (
        "SiblingSupervisorCapabilityRegistry@1"
    )
    assert CROSS_SUPERVISOR_RECEIPT_BINDING == "CrossSupervisorReceipt@1"
    assert CANONICAL_EVENT_INTERFACE == "CanonicalEvent@1"
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert EVENT_DRIVEN_REASSESSMENT_BINDING == "EventDrivenReassessment@1"
    assert INCREMENTAL_PLAN_IMPACT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/incremental-plan-impact@1"
    )
    assert PLAN_DELTA_SCHEMA == "ipfs_datasets_py/logic/external-work-plan-delta@1"
    assert STALE_PLAN_EPOCH_BINDING == "StalePlanEpochHandling@1"
    assert IDEMPOTENT_EVENT_CONSUMPTION_BINDING == "IdempotentEventConsumption@1"
    assert CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING == (
        "CrossRepositoryIncrementalReassessment@1"
    )
    assert (
        CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_INTERFACE
        == CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING
    )
    assert CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/cross-repository-incremental-reassessment@1"
    )
    assert CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_CONSUMES == (
        SUPERVISOR_FABRIC_INTERFACE,
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
        CROSS_SUPERVISOR_RECEIPT_BINDING,
        EVENT_DRIVEN_REASSESSMENT_BINDING,
        INCREMENTAL_PLAN_IMPACT_SCHEMA,
        PLAN_DELTA_SCHEMA,
        STALE_PLAN_EPOCH_BINDING,
        IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
    )
    assert CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_CAPABILITIES == {
        "event-exchange",
        "receipt-exchange",
    }
    assert CROSS_REPOSITORY_OWNERS == {
        "ipfs_accelerate_py",
        "ipfs_datasets_py",
        "ipfs_kit_py",
    }
    assert DEFAULT_LOCAL_REPOSITORY == "ipfs_accelerate_py"
    assert SupervisorFabric.INTERFACE == SUPERVISOR_FABRIC_INTERFACE
    assert (
        SupervisorFabric.CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING
        == CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING
    )
    assert reassess_cross_repository.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert SupervisorFabric.reassess_cross_repository.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert "event-exchange" in ALLOWED_SIBLING_CAPABILITIES
    assert "receipt-exchange" in ALLOWED_SIBLING_CAPABILITIES
    assert "database-write" in FORBIDDEN_SIBLING_CAPABILITIES
    assert "terminalize-task" in FORBIDDEN_SIBLING_CAPABILITIES
    source = FABRIC_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "class SupervisorFabric" in source
    assert "def issue_fence(" in source
    assert "def validate_sibling_supervisor_event(" in source
    assert "def register_sibling_supervisor_capability(" in source
    assert "def admit_cross_supervisor_receipt(" in source
    assert "def reassess_cross_repository(" in source
    assert "class CrossRepositoryIncrementalReassessment" in source
    assert "reassess_events" in source
    assert "assert_plan_epoch_current" in source
    assert "not a competing subsystem" in lowered
    assert "not a second" in lowered
    assert "not a second reassessment" in lowered
    assert "class CrossRepositoryReassessmentBus" not in source
    assert "class CompetingReassessmentEngine" not in source
    assert "class SiblingReassessmentLog" not in source
    assert "class CrossRepositoryPlanner" not in source
    assert "CREATE TABLE" not in source


def test_issue_fence_still_requires_capability_and_rejects_stale_epoch() -> None:
    fence = issue_fence(
        {"supervisor_id": "S1", "capability": "event-exchange", "epoch": 2}
    )
    assert fence["fenced"] is True
    assert fence["epoch"] == 2
    with pytest.raises(SupervisorFabricError, match="stale"):
        issue_fence(
            {"supervisor_id": "S1", "capability": "event-exchange", "stale_epoch": True}
        )
    with pytest.raises(SupervisorFabricError, match="capability"):
        issue_fence({"supervisor_id": "S1", "capability": ""})


def test_sibling_event_reassesses_only_local_suffix_and_preserves_foreign_nodes() -> None:
    admitted = reassess_cross_repository(_record())
    assert isinstance(admitted, CrossRepositoryIncrementalReassessment)
    payload = admitted.to_dict()
    assert payload["admitted"] is True
    assert payload["fenced"] is True
    assert payload["database_write"] is False
    assert payload["direct_state_write"] is False
    assert payload["terminalize_task"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["worker_completion_insufficient"] is True
    assert payload["authorizes_append"] is False
    assert payload["authorizes_completion"] is False
    assert payload["model_free"] is True
    assert payload["no_competing_subsystem_created"] is True
    assert payload["binding"] == CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING
    assert payload["carrier"] == SUPERVISOR_FABRIC_INTERFACE
    assert payload["consumes"]["event_driven_reassessment"] == (
        EVENT_DRIVEN_REASSESSMENT_BINDING
    )
    assert payload["consumes"]["incremental_plan_impact"] == INCREMENTAL_PLAN_IMPACT_SCHEMA
    assert payload["consumes"]["plan_delta"] == PLAN_DELTA_SCHEMA
    assert payload["disposition"] == "reassessed"
    assert payload["local_repository"] == "ipfs_accelerate_py"
    assert payload["sibling_repository"] == "ipfs_datasets_py"
    assert payload["triggering_event_id"] == "event:sibling-103"
    assert payload["seed_node_ids"] == ["task:datasets-schema"]
    assert payload["applied_event_ids"] == ["event:sibling-103"]
    assert payload["local_impacted_ids"] == [
        "task:accelerate-adapter",
        "task:accelerate-follow",
    ]
    assert payload["foreign_affected_ids"] == ["task:datasets-schema"]
    assert payload["preserved_task_ids"] == ["task:accelerate-other"]
    assert "receipt:accelerate-other" in payload["preserved_receipt_ids"]
    assert "receipt:datasets-schema" not in payload["preserved_receipt_ids"]
    assert payload["refill_task_ids"] == [
        "task:accelerate-adapter",
        "task:accelerate-follow",
    ]
    assert payload["plan_delta"]["schema"] == PLAN_DELTA_SCHEMA
    assert payload["plan_delta"]["history_preserving"] is True
    assert payload["plan_delta"]["model_free_refill"] is True
    assert payload["plan_delta"]["completion_authoritative"] is False
    assert payload["plan_delta"]["impacted_task_ids"] == payload["local_impacted_ids"]
    assert "task:datasets-schema" in payload["plan_delta"]["preserved_task_ids"]
    assert set(payload["plan_delta"]["refill_task_ids"]).issubset(
        set(payload["plan_delta"]["impacted_task_ids"])
    )
    assert payload["reassessment"] is not None
    assert payload["reassessment"]["binding"] == EVENT_DRIVEN_REASSESSMENT_BINDING
    assert payload["reassessment"]["disposition"] == (
        EventDrivenReassessmentDisposition.REASSESSED.value
    )
    assert payload["reassessment"]["impact"]["schema"] == INCREMENTAL_PLAN_IMPACT_SCHEMA
    assert (
        payload["reassessment"]["impact"]["refill_decision"]["disposition"]
        == OrdinaryRefillDisposition.REFILL_AFFECTED_SUFFIX.value
    )
    assert payload["event_admission"]["event_id"] == "event:sibling-103"
    assert payload["event_admission"]["database_write"] is False
    round_trip = CrossRepositoryIncrementalReassessment.from_dict(payload)
    assert round_trip.to_dict() == payload
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:datasets",),
    )
    again = fabric.reassess_cross_repository(
        {
            "sibling_supervisor_id": "supervisor:datasets",
            "sibling_repository": "ipfs_datasets_py",
            "live_plan_epoch": 1,
            "plan_root_cid": "plan:doep-103",
            "revision": 3,
            "nodes": _nodes(),
            "node_repositories": _node_repositories(),
            "event": _canonical_event(),
        }
    )
    assert again.triggering_event_id == "event:sibling-103"
    assert again.local_impacted_ids == (
        "task:accelerate-adapter",
        "task:accelerate-follow",
    )
    assert again.to_dict()["database_write"] is False


def test_kit_receipt_carrier_reassesses_local_dependents_without_database_write() -> None:
    nodes = (
        PlanImpactNode(
            node_id="task:kit-wal",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:kit-wal",),
            goal_id="goal:durable",
        ),
        PlanImpactNode(
            node_id="task:accelerate-cas",
            kind="task",
            depends_on=("task:kit-wal",),
            lifecycle="unstarted",
            receipt_refs=("receipt:accelerate-cas",),
            goal_id="goal:local",
        ),
        PlanImpactNode(
            node_id="task:accelerate-stable",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:accelerate-stable",),
            goal_id="goal:other",
        ),
    )
    admitted = reassess_cross_repository(
        _record(
            sibling_supervisor_id="supervisor:kit",
            sibling_repository="ipfs_kit_py",
            capability="receipt-exchange",
            event=None,
            receipt=_receipt_envelope(
                task_id="task:kit-wal",
                payload={
                    "kind": "stale_evidence",
                    "seed_node_ids": ["task:kit-wal"],
                },
            ),
            nodes=nodes,
            node_repositories={
                "task:kit-wal": "ipfs_kit_py",
                "task:accelerate-cas": "ipfs_accelerate_py",
                "task:accelerate-stable": "ipfs_accelerate_py",
            },
        )
    )
    assert admitted.capability == CROSS_SUPERVISOR_RECEIPT_CAPABILITY
    assert admitted.sibling_repository == "ipfs_kit_py"
    assert admitted.disposition is (
        CrossRepositoryIncrementalReassessmentDisposition.REASSESSED
    )
    assert admitted.local_impacted_ids == ("task:accelerate-cas",)
    assert admitted.foreign_affected_ids == ("task:kit-wal",)
    assert admitted.preserved_task_ids == ("task:accelerate-stable",)
    assert admitted.receipt_admission is not None
    assert admitted.receipt_admission["database_write"] is False
    assert admitted.to_dict()["terminalize_task"] is False


def test_unknown_receipt_outcome_is_typed_and_does_not_reassess() -> None:
    admitted = reassess_cross_repository(
        _record(
            capability="receipt-exchange",
            event=None,
            receipt=_receipt_envelope(outcome="unknown"),
        )
    )
    assert admitted.disposition is (
        CrossRepositoryIncrementalReassessmentDisposition.UNKNOWN
    )
    assert admitted.applied_event_ids == ()
    assert admitted.local_impacted_ids == ()
    assert admitted.refill_task_ids == ()
    assert admitted.reassessment is None
    assert admitted.to_dict()["completion_authoritative"] is False
    assert admitted.to_dict()["authorizes_append"] is False


def test_idempotent_replay_does_not_reapply_or_write_a_database() -> None:
    first = reassess_cross_repository(_record())
    replay = reassess_cross_repository(
        _record(previously_consumed_event_ids=first.applied_event_ids)
    )
    assert replay.disposition is (
        CrossRepositoryIncrementalReassessmentDisposition.REPLAYED
    )
    assert replay.applied_event_ids == ()
    assert replay.replayed_event_ids == ("event:sibling-103",)
    assert replay.local_impacted_ids == ()
    assert replay.to_dict()["database_write"] is False
    identical = reassess_cross_repository(_record())
    assert identical.to_dict() == first.to_dict()
    assert identical.logical_once_key == first.logical_once_key


def test_foreign_only_suffix_does_not_locally_refill() -> None:
    nodes = (
        PlanImpactNode(
            node_id="task:datasets-schema",
            kind="task",
            depends_on=(),
            lifecycle="unstarted",
            receipt_refs=("receipt:datasets-schema",),
            goal_id="goal:semantic",
        ),
        PlanImpactNode(
            node_id="task:accelerate-other",
            kind="task",
            depends_on=(),
            lifecycle="completed",
            receipt_refs=("receipt:accelerate-other",),
            goal_id="goal:other",
        ),
    )
    admitted = reassess_cross_repository(
        _record(
            nodes=nodes,
            node_repositories={
                "task:datasets-schema": "ipfs_datasets_py",
                "task:accelerate-other": "ipfs_accelerate_py",
            },
        )
    )
    assert admitted.disposition is (
        CrossRepositoryIncrementalReassessmentDisposition.NO_REASSESSMENT
    )
    assert admitted.local_impacted_ids == ()
    assert admitted.foreign_affected_ids == ("task:datasets-schema",)
    assert admitted.refill_task_ids == ()
    assert admitted.to_dict()["authorizes_append"] is False


def test_self_supervisor_same_repository_and_unknown_peer_fail_closed() -> None:
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as self_exc:
        reassess_cross_repository(
            _record(
                local_supervisor_id="supervisor:local",
                sibling_supervisor_id="supervisor:local",
            )
        )
    assert self_exc.value.code == "not_a_sibling"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as same_repo:
        reassess_cross_repository(
            _record(
                local_repository="ipfs_accelerate_py",
                sibling_repository="ipfs_accelerate_py",
            )
        )
    assert same_repo.value.code == "not_cross_repository"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as unknown:
        reassess_cross_repository(_record(known_sibling_ids=("supervisor:other",)))
    assert unknown.value.code == "unknown_sibling"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as unknown_repo:
        reassess_cross_repository(_record(sibling_repository="lift_coding"))
    assert unknown_repo.value.code == "unknown_repository"


def test_unknown_and_forbidden_capabilities_fail_closed() -> None:
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as unknown:
        reassess_cross_repository(_record(capability="task-request"))
    assert unknown.value.code == "unknown_capability"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as forbidden:
        reassess_cross_repository(_record(capability="database-write"))
    assert forbidden.value.code == "forbidden_capability"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as terminal:
        reassess_cross_repository(_record(capability="terminalize-task"))
    assert terminal.value.code == "forbidden_capability"


def test_stale_fence_and_plan_epoch_fail_closed_even_with_worker_assertion() -> None:
    with pytest.raises(SupervisorFabricError, match="stale"):
        reassess_cross_repository(_record(stale_epoch=True))
    with pytest.raises(SupervisorFabricError, match="capability"):
        reassess_cross_repository(_record(capability=""))
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as stale:
        reassess_cross_repository(_record(epoch=1, current_epoch=2))
    assert stale.value.code == "stale_fence_epoch"
    with pytest.raises(PlanRevisionStoreStalePlanEpochError, match="stale plan epoch"):
        reassess_cross_repository(
            _record(
                live_plan_epoch=2,
                event=_canonical_event(payload={"plan_epoch": 1}),
                worker_assertion=True,
            )
        )
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=3,
        known_sibling_ids=("supervisor:datasets",),
    )
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as fabric_stale:
        fabric.reassess_cross_repository(
            {
                "sibling_supervisor_id": "supervisor:datasets",
                "sibling_repository": "ipfs_datasets_py",
                "epoch": 2,
                "live_plan_epoch": 1,
                "plan_root_cid": "plan:doep-103",
                "nodes": _nodes(),
                "node_repositories": _node_repositories(),
                "event": _canonical_event(),
            }
        )
    assert fabric_stale.value.code == "stale_fence_epoch"


def test_direct_state_writes_and_forbidden_effects_are_rejected() -> None:
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as duckdb:
        reassess_cross_repository(_record(duckdb_path="/tmp/control.duckdb"))
    assert duckdb.value.code == "direct_state_write"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as sql:
        reassess_cross_repository(_record(sql="UPDATE tasks SET status='done'"))
    assert sql.value.code == "direct_state_write"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as consume:
        reassess_cross_repository(_record(consume=True))
    assert consume.value.code == "direct_state_write"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as effect:
        reassess_cross_repository(_record(effect="authoritative_state"))
    assert effect.value.code == "forbidden_effect"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as completion:
        reassess_cross_repository(_record(completion_authoritative=True))
    assert completion.value.code == "completion_not_authoritative"


def test_missing_carrier_plan_root_and_seeds_fail_closed() -> None:
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as carrier:
        reassess_cross_repository(_record(event=None, receipt=None))
    assert carrier.value.code == "carrier_required"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as receipt_cap:
        reassess_cross_repository(_record(capability="receipt-exchange", receipt=None))
    assert receipt_cap.value.code == "carrier_required"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError, match="plan_root"):
        reassess_cross_repository(_record(plan_root_cid=""))
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as seeds:
        reassess_cross_repository(
            _record(event=_canonical_event(payload={"seed_node_ids": []}))
        )
    assert seeds.value.code == "seed_node_ids"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as unknown_seed:
        reassess_cross_repository(
            _record(
                event=_canonical_event(
                    payload={"seed_node_ids": ["task:missing"]}
                )
            )
        )
    assert unknown_seed.value.code == "unknown_seed"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as root_mismatch:
        reassess_cross_repository(
            _record(event=_canonical_event(payload={"plan_root": "plan:other"}))
        )
    assert root_mismatch.value.code == "plan_root_mismatch"
    with pytest.raises(CrossRepositoryIncrementalReassessmentError) as mismatch:
        reassess_cross_repository(
            _record(
                receipt=_receipt_envelope(carrier_event_id="event:other"),
            )
        )
    assert mismatch.value.code == "carrier_mismatch"


def test_worker_assertion_is_not_admission_authority() -> None:
    admitted = reassess_cross_repository(_record(worker_assertion=True))
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(CrossRepositoryIncrementalReassessmentError):
        reassess_cross_repository(
            _record(worker_assertion=True, duckdb_path="events.duckdb")
        )
    with pytest.raises(CrossRepositoryIncrementalReassessmentError):
        reassess_cross_repository(
            _record(worker_assertion=True, capability="database-write")
        )
    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        reassess_cross_repository(
            _record(
                worker_assertion=True,
                live_plan_epoch=2,
                event=_canonical_event(payload={"plan_epoch": 1}),
            )
        )


def test_capability_registry_gates_reassessment_when_present() -> None:
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:datasets",),
        sibling_capabilities=(_capability_record(),),
    )
    admitted = fabric.reassess_cross_repository(
        {
            "sibling_supervisor_id": "supervisor:datasets",
            "sibling_repository": "ipfs_datasets_py",
            "live_plan_epoch": 1,
            "plan_root_cid": "plan:doep-103",
            "nodes": _nodes(),
            "node_repositories": _node_repositories(),
            "event": _canonical_event(),
        }
    )
    assert admitted.capability == "event-exchange"
    receipts_only = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:datasets",),
        sibling_capabilities=(
            _capability_record(capability="receipt-exchange"),
        ),
    )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as gated:
        receipts_only.reassess_cross_repository(
            {
                "sibling_supervisor_id": "supervisor:datasets",
                "sibling_repository": "ipfs_datasets_py",
                "live_plan_epoch": 1,
                "plan_root_cid": "plan:doep-103",
                "nodes": _nodes(),
                "node_repositories": _node_repositories(),
                "event": _canonical_event(),
            }
        )
    assert gated.value.code == "unknown_capability"


def test_event_validation_and_receipt_admission_remain_unchanged() -> None:
    admitted_event = validate_sibling_supervisor_event(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:datasets",
            "capability": "event-exchange",
            "epoch": 2,
            "event": _canonical_event(),
        }
    )
    assert isinstance(admitted_event, SiblingSupervisorEventAdmission)
    assert admitted_event.to_dict()["database_write"] is False
    registered = register_sibling_supervisor_capability(_capability_record())
    assert registered.capability == "event-exchange"
    receipt = admit_cross_supervisor_receipt(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:datasets",
            "capability": "receipt-exchange",
            "epoch": 2,
            "receipt": _receipt_envelope(),
        }
    )
    assert receipt.receipt_id == "receipt:sibling-103"
    with pytest.raises(CrossSupervisorReceiptError) as sql:
        admit_cross_supervisor_receipt(
            {
                "local_supervisor_id": "supervisor:local",
                "sibling_supervisor_id": "supervisor:datasets",
                "capability": "receipt-exchange",
                "epoch": 2,
                "receipt": _receipt_envelope(),
                "sql": "SELECT 1",
            }
        )
    assert sql.value.code == "direct_state_write"


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-103"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["plan_epoch"] == 1
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["title"] == "Add cross-repository incremental reassessment"
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "SupervisorFabric"
    assert (
        manifest["canonical_extension"]["binding"]
        == CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_BINDING
    )
    assert (
        manifest["canonical_extension"]["entrypoint"] == "reassess_cross_repository"
    )
    assert manifest["canonical_extension"]["consumes"] == list(
        CROSS_REPOSITORY_INCREMENTAL_REASSESSMENT_CONSUMES
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(FABRIC_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert (
        receipt["required_evidence"]["verifier_admission"]
        == "pending_independent_fenced_supervisor"
    )
    assert receipt["title"] == "Add cross-repository incremental reassessment"
