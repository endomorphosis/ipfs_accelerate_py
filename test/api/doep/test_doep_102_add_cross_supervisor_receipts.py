"""Independent current-tree checks for DOEP-102 cross-supervisor receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import (
    ALLOWED_SIBLING_CAPABILITIES,
    CANONICAL_EVENT_INTERFACE,
    CANONICAL_EVENT_SCHEMA_ID,
    CROSS_SUPERVISOR_RECEIPT_BINDING,
    CROSS_SUPERVISOR_RECEIPT_CAPABILITY,
    CROSS_SUPERVISOR_RECEIPT_CARRIER,
    CROSS_SUPERVISOR_RECEIPT_CONSUMES,
    CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS,
    CROSS_SUPERVISOR_RECEIPT_INTERFACE,
    CROSS_SUPERVISOR_RECEIPT_OUTCOMES,
    CROSS_SUPERVISOR_RECEIPT_SCHEMA,
    DATABASE_EVENT_LOG_INTERFACE,
    FORBIDDEN_SIBLING_CAPABILITIES,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    SUPERVISOR_FABRIC_INTERFACE,
    CrossSupervisorReceiptAdmission,
    CrossSupervisorReceiptError,
    SiblingSupervisorCapabilityAdmission,
    SiblingSupervisorCapabilityRegistryError,
    SiblingSupervisorEventAdmission,
    SupervisorFabric,
    SupervisorFabricError,
    admit_cross_supervisor_receipt,
    issue_fence,
    register_sibling_supervisor_capability,
    validate_sibling_supervisor_event,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
FABRIC_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-102.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-102.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py",
    "test/api/doep/test_doep_102_add_cross_supervisor_receipts.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-102.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-102.json",
)
TASK_CID = "sha256:864089a98ea5ce5297a1aa163c9ded9e3814fdab02863745d05a3ee39cff8457"
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
EVIDENCE_DIGEST = "sha256:" + hashlib.sha256(b"doep-102-independent-evidence").hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_event(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema": CANONICAL_EVENT_SCHEMA_ID,
        "event_id": "event:receipt-carrier-102",
        "event_type": "task.receipt.exchanged",
        "stream_id": "task:DOEP-102",
        "causal_parent_ids": ["event:parent-102"],
        "correlation_id": "correlation:attempt-102",
        "causation_id": "causation:receipt-102",
        "payload": {"outcome": "admitted", "attempt": 1},
    }
    values.update(overrides)
    return values


def _receipt_envelope(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema": CROSS_SUPERVISOR_RECEIPT_SCHEMA,
        "receipt_id": "receipt:sibling-102",
        "task_id": "DOEP-102",
        "request_id": "request:DOEP-101",
        "carrier_event_id": "event:receipt-carrier-102",
        "outcome": "admitted",
        "evidence_digest": EVIDENCE_DIGEST,
        "payload": {"changed_paths": ["supervisor_fabric.py"]},
    }
    values.update(overrides)
    return values


def _record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:peer",
        "capability": CROSS_SUPERVISOR_RECEIPT_CAPABILITY,
        "epoch": 2,
        "effect": "event_exchange",
        "receipt": _receipt_envelope(),
    }
    values.update(overrides)
    return values


def _capability_record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:peer",
        "capability": CROSS_SUPERVISOR_RECEIPT_CAPABILITY,
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
    assert CANONICAL_EVENT_INTERFACE == "CanonicalEvent@1"
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert CROSS_SUPERVISOR_RECEIPT_BINDING == "CrossSupervisorReceipt@1"
    assert CROSS_SUPERVISOR_RECEIPT_INTERFACE == CROSS_SUPERVISOR_RECEIPT_BINDING
    assert CROSS_SUPERVISOR_RECEIPT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/cross-supervisor-receipt@1"
    )
    assert CROSS_SUPERVISOR_RECEIPT_CAPABILITY == "receipt-exchange"
    assert CROSS_SUPERVISOR_RECEIPT_CARRIER == "event"
    assert CROSS_SUPERVISOR_RECEIPT_CONSUMES == (
        SUPERVISOR_FABRIC_INTERFACE,
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
    )
    assert CROSS_SUPERVISOR_RECEIPT_OUTCOMES == {"admitted", "rejected", "unknown"}
    assert "lease_id" in CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS
    assert "sql" in CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS
    assert "consume" in CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS
    assert SupervisorFabric.INTERFACE == SUPERVISOR_FABRIC_INTERFACE
    assert (
        SupervisorFabric.CROSS_SUPERVISOR_RECEIPT_BINDING
        == CROSS_SUPERVISOR_RECEIPT_BINDING
    )
    assert admit_cross_supervisor_receipt.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert SupervisorFabric.admit_cross_supervisor_receipt.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
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
    assert "class CrossSupervisorReceiptAdmission" in source
    assert "not a competing subsystem" in lowered
    assert "not a second" in lowered
    assert "not a second receipt log" in lowered
    assert "class SiblingReceiptBus" not in source
    assert "class SiblingReceiptLog" not in source
    assert "class SiblingReceiptDatabase" not in source
    assert "class CrossSupervisorReceiptBus" not in source
    assert "class CompetingReceiptStore" not in source
    assert "CREATE TABLE" not in source


def test_issue_fence_still_requires_capability_and_rejects_stale_epoch() -> None:
    fence = issue_fence(
        {"supervisor_id": "S1", "capability": "receipt-exchange", "epoch": 2}
    )
    assert fence["fenced"] is True
    assert fence["epoch"] == 2
    with pytest.raises(SupervisorFabricError, match="stale"):
        issue_fence(
            {"supervisor_id": "S1", "capability": "receipt-exchange", "stale_epoch": True}
        )
    with pytest.raises(SupervisorFabricError, match="capability"):
        issue_fence({"supervisor_id": "S1", "capability": ""})


def test_admit_receipt_without_database_write() -> None:
    admitted = admit_cross_supervisor_receipt(_record())
    assert isinstance(admitted, CrossSupervisorReceiptAdmission)
    payload = admitted.to_dict()
    assert payload["admitted"] is True
    assert payload["fenced"] is True
    assert payload["database_write"] is False
    assert payload["direct_state_write"] is False
    assert payload["terminalize_task"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["worker_completion_insufficient"] is True
    assert payload["binding"] == CROSS_SUPERVISOR_RECEIPT_BINDING
    assert payload["carrier"] == SUPERVISOR_FABRIC_INTERFACE
    assert payload["consumes"]["supervisor_fabric"] == SUPERVISOR_FABRIC_INTERFACE
    assert (
        payload["consumes"]["sibling_event_validation"]
        == SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    assert (
        payload["consumes"]["sibling_capability_registry"]
        == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    assert payload["logical_once_key"] == "supervisor:peer:receipt:sibling-102"
    assert payload["receipt_digest"].startswith("sha256:")
    assert payload["evidence_digest"] == EVIDENCE_DIGEST
    assert payload["capability"] == "receipt-exchange"
    assert payload["outcome"] == "admitted"
    assert payload["request_id"] == "request:DOEP-101"
    assert payload["carrier_event_id"] == "event:receipt-carrier-102"
    round_trip = CrossSupervisorReceiptAdmission.from_dict(payload)
    assert round_trip.to_dict() == payload
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
    )
    again = fabric.admit_cross_supervisor_receipt(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "receipt": _receipt_envelope(),
        }
    )
    assert again.receipt_id == "receipt:sibling-102"
    assert again.to_dict()["database_write"] is False
    assert again.capability == CROSS_SUPERVISOR_RECEIPT_CAPABILITY


def test_self_supervisor_and_unknown_peer_fail_closed() -> None:
    with pytest.raises(CrossSupervisorReceiptError) as self_exc:
        admit_cross_supervisor_receipt(
            _record(
                local_supervisor_id="supervisor:local",
                sibling_supervisor_id="supervisor:local",
            )
        )
    assert self_exc.value.code == "not_a_sibling"
    with pytest.raises(CrossSupervisorReceiptError) as unknown:
        admit_cross_supervisor_receipt(
            _record(known_sibling_ids=("supervisor:other",))
        )
    assert unknown.value.code == "unknown_sibling"


def test_unknown_and_forbidden_capabilities_fail_closed() -> None:
    with pytest.raises(CrossSupervisorReceiptError) as unknown:
        admit_cross_supervisor_receipt(_record(capability="event-exchange"))
    assert unknown.value.code == "unknown_capability"
    with pytest.raises(CrossSupervisorReceiptError) as task_request:
        admit_cross_supervisor_receipt(_record(capability="task-request"))
    assert task_request.value.code == "unknown_capability"
    with pytest.raises(CrossSupervisorReceiptError) as forbidden:
        admit_cross_supervisor_receipt(_record(capability="database-write"))
    assert forbidden.value.code == "forbidden_capability"
    with pytest.raises(CrossSupervisorReceiptError) as terminal:
        admit_cross_supervisor_receipt(_record(capability="terminalize-task"))
    assert terminal.value.code == "forbidden_capability"


def test_stale_epoch_and_missing_capability_fail_closed() -> None:
    with pytest.raises(SupervisorFabricError, match="stale"):
        admit_cross_supervisor_receipt(_record(stale_epoch=True))
    with pytest.raises(SupervisorFabricError, match="capability"):
        admit_cross_supervisor_receipt(_record(capability=""))
    with pytest.raises(CrossSupervisorReceiptError) as stale:
        admit_cross_supervisor_receipt(_record(epoch=1, current_epoch=2))
    assert stale.value.code == "stale_fence_epoch"
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=3,
        known_sibling_ids=("supervisor:peer",),
    )
    with pytest.raises(CrossSupervisorReceiptError) as fabric_stale:
        fabric.admit_cross_supervisor_receipt(
            {
                "sibling_supervisor_id": "supervisor:peer",
                "epoch": 2,
                "receipt": _receipt_envelope(),
            }
        )
    assert fabric_stale.value.code == "stale_fence_epoch"


def test_direct_state_writes_and_forbidden_effects_are_rejected() -> None:
    with pytest.raises(CrossSupervisorReceiptError) as duckdb:
        admit_cross_supervisor_receipt(_record(duckdb_path="/tmp/control.duckdb"))
    assert duckdb.value.code == "direct_state_write"
    with pytest.raises(CrossSupervisorReceiptError) as sql:
        admit_cross_supervisor_receipt(
            _record(sql="UPDATE tasks SET status='done'")
        )
    assert sql.value.code == "direct_state_write"
    with pytest.raises(CrossSupervisorReceiptError) as consume:
        admit_cross_supervisor_receipt(_record(consume=True))
    assert consume.value.code == "direct_state_write"
    with pytest.raises(CrossSupervisorReceiptError) as effect:
        admit_cross_supervisor_receipt(_record(effect="authoritative_state"))
    assert effect.value.code == "forbidden_effect"
    with pytest.raises(CrossSupervisorReceiptError) as completion:
        admit_cross_supervisor_receipt(_record(completion_authoritative=True))
    assert completion.value.code == "completion_not_authoritative"


def test_receipt_envelope_rejects_malformed_and_operational_authority_fields() -> None:
    with pytest.raises(CrossSupervisorReceiptError) as lease_exc:
        admit_cross_supervisor_receipt(
            _record(receipt={**_receipt_envelope(), "lease_id": "lease:forbidden"})
        )
    assert lease_exc.value.code == "operational_authority_field"
    with pytest.raises(CrossSupervisorReceiptError) as extra:
        admit_cross_supervisor_receipt(
            _record(receipt={**_receipt_envelope(), "foreign_bus": True})
        )
    assert extra.value.code == "receipt_invalid"
    with pytest.raises(CrossSupervisorReceiptError) as schema_exc:
        admit_cross_supervisor_receipt(
            _record(receipt=_receipt_envelope(schema="receipt-bus/v0"))
        )
    assert schema_exc.value.code == "receipt_schema"
    with pytest.raises(CrossSupervisorReceiptError) as outcome_exc:
        admit_cross_supervisor_receipt(
            _record(receipt=_receipt_envelope(outcome="complete"))
        )
    assert outcome_exc.value.code == "unknown_outcome"
    with pytest.raises(CrossSupervisorReceiptError) as digest_exc:
        admit_cross_supervisor_receipt(
            _record(receipt=_receipt_envelope(evidence_digest="not-a-digest"))
        )
    assert digest_exc.value.code == "receipt_invalid"
    with pytest.raises(CrossSupervisorReceiptError):
        admit_cross_supervisor_receipt(_record(receipt="not-an-object"))
    with pytest.raises(CrossSupervisorReceiptError) as sql_field:
        admit_cross_supervisor_receipt(
            _record(receipt={**_receipt_envelope(), "sql": "SELECT 1"})
        )
    assert sql_field.value.code == "operational_authority_field"


def test_worker_assertion_is_not_admission_authority() -> None:
    admitted = admit_cross_supervisor_receipt(_record(worker_assertion=True))
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(CrossSupervisorReceiptError):
        admit_cross_supervisor_receipt(
            _record(
                worker_assertion=True,
                receipt={**_receipt_envelope(), "lease_id": "lease:worker"},
            )
        )
    with pytest.raises(CrossSupervisorReceiptError):
        admit_cross_supervisor_receipt(
            _record(worker_assertion=True, duckdb_path="events.duckdb")
        )
    with pytest.raises(CrossSupervisorReceiptError):
        admit_cross_supervisor_receipt(
            _record(worker_assertion=True, capability="database-write")
        )


def test_identical_receipts_are_idempotent_and_unknown_outcome_is_typed() -> None:
    first = admit_cross_supervisor_receipt(_record())
    second = admit_cross_supervisor_receipt(_record())
    assert first.to_dict() == second.to_dict()
    assert first.logical_once_key == second.logical_once_key
    assert first.receipt_digest == second.receipt_digest
    changed = admit_cross_supervisor_receipt(
        _record(receipt=_receipt_envelope(receipt_id="receipt:sibling-102-b"))
    )
    assert changed.logical_once_key != first.logical_once_key
    unknown = admit_cross_supervisor_receipt(
        _record(receipt=_receipt_envelope(outcome="unknown"))
    )
    assert unknown.outcome == "unknown"
    assert unknown.to_dict()["completion_authoritative"] is False
    rejected = admit_cross_supervisor_receipt(
        _record(receipt=_receipt_envelope(outcome="rejected"))
    )
    assert rejected.outcome == "rejected"
    empty_request = admit_cross_supervisor_receipt(
        _record(receipt=_receipt_envelope(request_id=""))
    )
    assert empty_request.request_id == ""


def test_capability_registry_gates_receipt_exchange_when_present() -> None:
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
        sibling_capabilities=(_capability_record(),),
    )
    admitted = fabric.admit_cross_supervisor_receipt(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "receipt": _receipt_envelope(),
        }
    )
    assert admitted.capability == "receipt-exchange"
    events_only = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
        sibling_capabilities=(
            _capability_record(capability="event-exchange"),
        ),
    )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as gated:
        events_only.admit_cross_supervisor_receipt(
            {
                "sibling_supervisor_id": "supervisor:peer",
                "receipt": _receipt_envelope(),
            }
        )
    assert gated.value.code == "unknown_capability"


def test_event_validation_and_capability_registry_remain_unchanged() -> None:
    admitted_event = validate_sibling_supervisor_event(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "epoch": 2,
            "event": _canonical_event(),
        }
    )
    assert isinstance(admitted_event, SiblingSupervisorEventAdmission)
    assert admitted_event.to_dict()["database_write"] is False
    registered = register_sibling_supervisor_capability(_capability_record())
    assert isinstance(registered, SiblingSupervisorCapabilityAdmission)
    assert registered.capability == "receipt-exchange"
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
        sibling_capabilities=(
            _capability_record(capability="event-exchange"),
            _capability_record(),
        ),
    )
    exchanged = fabric.validate_sibling_event(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "event": _canonical_event(),
        }
    )
    assert exchanged.event_id == "event:receipt-carrier-102"
    receipt = fabric.admit_cross_supervisor_receipt(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "receipt": _receipt_envelope(),
        }
    )
    assert receipt.receipt_id == "receipt:sibling-102"


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-102"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["plan_epoch"] == 1
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["title"] == "Add cross-supervisor receipts"
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "SupervisorFabric"
    assert (
        manifest["canonical_extension"]["binding"] == CROSS_SUPERVISOR_RECEIPT_BINDING
    )
    assert (
        manifest["canonical_extension"]["entrypoint"]
        == "admit_cross_supervisor_receipt"
    )
    assert manifest["canonical_extension"]["consumes"] == list(
        CROSS_SUPERVISOR_RECEIPT_CONSUMES
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
    assert receipt["title"] == "Add cross-supervisor receipts"
