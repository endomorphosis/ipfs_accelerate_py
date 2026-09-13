"""Independent current-tree checks for DOEP-035 sibling-event validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import (
    CANONICAL_EVENT_FORBIDDEN_FIELDS,
    CANONICAL_EVENT_INTERFACE,
    CANONICAL_EVENT_SCHEMA_ID,
    DATABASE_EVENT_LOG_INTERFACE,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA,
    SUPERVISOR_FABRIC_INTERFACE,
    SiblingSupervisorEventAdmission,
    SiblingSupervisorEventValidationError,
    SupervisorFabric,
    SupervisorFabricError,
    issue_fence,
    validate_sibling_supervisor_event,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
FABRIC_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-035.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-035.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py",
    "test/api/doep/test_doep_035_add_sibling_supervisor_event_validation.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-035.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-035.json",
)
TASK_CID = "sha256:aa38ccd566fe60eaf4860318a6862129fa63001dd9b58ac2abdf684b5af63ded"
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


def _canonical_event(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema": CANONICAL_EVENT_SCHEMA_ID,
        "event_id": "event:sibling-001",
        "event_type": "task.validation.completed",
        "stream_id": "task:DOEP-035",
        "causal_parent_ids": ["event:parent-001"],
        "correlation_id": "correlation:attempt-001",
        "causation_id": "causation:validation-001",
        "payload": {"outcome": "passed", "attempt": 1},
    }
    values.update(overrides)
    return values


def _record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:peer",
        "capability": "event-exchange",
        "epoch": 2,
        "effect": "event_exchange",
        "event": _canonical_event(),
    }
    values.update(overrides)
    return values


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_supervisor_fabric_without_competing_subsystem() -> None:
    assert SUPERVISOR_FABRIC_INTERFACE == "SupervisorFabric@1"
    assert CANONICAL_EVENT_INTERFACE == "CanonicalEvent@1"
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING == (
        "SiblingSupervisorEventValidation@1"
    )
    assert (
        SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE
        == SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    assert SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/sibling-supervisor-event-validation@1"
    )
    assert SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES == (
        SUPERVISOR_FABRIC_INTERFACE,
        CANONICAL_EVENT_INTERFACE,
        DATABASE_EVENT_LOG_INTERFACE,
    )
    assert SupervisorFabric.INTERFACE == SUPERVISOR_FABRIC_INTERFACE
    assert (
        SupervisorFabric.SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
        == SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    assert validate_sibling_supervisor_event.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert SupervisorFabric.validate_sibling_event.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    source = FABRIC_PATH.read_text(encoding="utf-8")
    assert "class SupervisorFabric" in source
    assert "def issue_fence(" in source
    assert "def validate_sibling_supervisor_event(" in source
    assert "class SiblingEventBus" not in source
    assert "class SiblingEventLog" not in source
    assert "class CompetingSiblingEvent" not in source
    assert "CREATE TABLE" not in source


def test_issue_fence_still_requires_capability_and_rejects_stale_epoch() -> None:
    fence = issue_fence({"supervisor_id": "S1", "capability": "dispatch", "epoch": 2})
    assert fence["fenced"] is True
    assert fence["epoch"] == 2
    with pytest.raises(SupervisorFabricError, match="stale"):
        issue_fence({"supervisor_id": "S1", "capability": "dispatch", "stale_epoch": True})
    with pytest.raises(SupervisorFabricError, match="capability"):
        issue_fence({"supervisor_id": "S1", "capability": ""})


def test_valid_sibling_event_is_admitted_without_database_write() -> None:
    admitted = validate_sibling_supervisor_event(_record())
    assert isinstance(admitted, SiblingSupervisorEventAdmission)
    payload = admitted.to_dict()
    assert payload["admitted"] is True
    assert payload["fenced"] is True
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["worker_completion_insufficient"] is True
    assert payload["binding"] == SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    assert payload["carrier"] == SUPERVISOR_FABRIC_INTERFACE
    assert payload["consumes"]["canonical_event"] == CANONICAL_EVENT_INTERFACE
    assert payload["consumes"]["database_event_log"] == DATABASE_EVENT_LOG_INTERFACE
    assert payload["logical_once_key"] == "supervisor:peer:event:sibling-001"
    assert payload["event_digest"].startswith("sha256:")
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
    )
    again = fabric.validate_sibling_event(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "event": _canonical_event(),
        }
    )
    assert again.event_id == admitted.event_id
    assert again.to_dict()["database_write"] is False


def test_self_supervisor_and_unknown_peer_fail_closed() -> None:
    with pytest.raises(SiblingSupervisorEventValidationError) as self_exc:
        validate_sibling_supervisor_event(
            _record(
                local_supervisor_id="supervisor:local",
                sibling_supervisor_id="supervisor:local",
            )
        )
    assert self_exc.value.code == "not_a_sibling"
    with pytest.raises(SiblingSupervisorEventValidationError) as unknown:
        validate_sibling_supervisor_event(
            _record(known_sibling_ids=("supervisor:other",))
        )
    assert unknown.value.code == "unknown_sibling"


def test_stale_epoch_and_missing_capability_fail_closed() -> None:
    with pytest.raises(SupervisorFabricError, match="stale"):
        validate_sibling_supervisor_event(_record(stale_epoch=True))
    with pytest.raises(SupervisorFabricError, match="capability"):
        validate_sibling_supervisor_event(_record(capability=""))


def test_canonical_event_rejects_malformed_and_operational_authority_fields() -> None:
    with pytest.raises(SiblingSupervisorEventValidationError) as lease_exc:
        validate_sibling_supervisor_event(
            _record(event={**_canonical_event(), "lease_id": "lease:forbidden"})
        )
    assert lease_exc.value.code == "operational_authority_field"
    with pytest.raises(SiblingSupervisorEventValidationError) as extra:
        validate_sibling_supervisor_event(
            _record(event={**_canonical_event(), "unknown": True})
        )
    assert extra.value.code == "canonical_event_invalid"
    with pytest.raises(SiblingSupervisorEventValidationError):
        validate_sibling_supervisor_event(
            _record(
                event=_canonical_event(
                    causal_parent_ids=["event:parent-001", "event:parent-001"]
                )
            )
        )
    with pytest.raises(SiblingSupervisorEventValidationError):
        validate_sibling_supervisor_event(
            _record(event=_canonical_event(causal_parent_ids=["event:sibling-001"]))
        )
    with pytest.raises(SiblingSupervisorEventValidationError) as schema_exc:
        validate_sibling_supervisor_event(
            _record(event=_canonical_event(schema="event-bus/v0"))
        )
    assert schema_exc.value.code == "canonical_event_schema"
    assert {"policy_id", "lease_id", "fencing_epoch"} <= CANONICAL_EVENT_FORBIDDEN_FIELDS


def test_direct_state_writes_and_log_mutations_are_rejected() -> None:
    with pytest.raises(SiblingSupervisorEventValidationError) as duckdb:
        validate_sibling_supervisor_event(_record(duckdb_path="/tmp/control.duckdb"))
    assert duckdb.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorEventValidationError) as sql:
        validate_sibling_supervisor_event(_record(sql="UPDATE tasks SET status='done'"))
    assert sql.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorEventValidationError) as consume:
        validate_sibling_supervisor_event(_record(consume=True))
    assert consume.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorEventValidationError) as effect:
        validate_sibling_supervisor_event(_record(effect="authoritative_state"))
    assert effect.value.code == "forbidden_effect"
    with pytest.raises(SiblingSupervisorEventValidationError) as completion:
        validate_sibling_supervisor_event(_record(completion_authoritative=True))
    assert completion.value.code == "completion_not_authoritative"


def test_worker_assertion_is_not_admission_authority() -> None:
    admitted = validate_sibling_supervisor_event(_record(worker_assertion=True))
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(SiblingSupervisorEventValidationError):
        validate_sibling_supervisor_event(
            _record(
                worker_assertion=True,
                event={**_canonical_event(), "lease_id": "lease:worker"},
            )
        )
    with pytest.raises(SiblingSupervisorEventValidationError):
        validate_sibling_supervisor_event(
            _record(worker_assertion=True, duckdb_path="events.duckdb")
        )


def test_identical_envelopes_are_idempotent_and_do_not_consume() -> None:
    first = validate_sibling_supervisor_event(_record())
    second = validate_sibling_supervisor_event(_record())
    assert first.to_dict() == second.to_dict()
    assert first.logical_once_key == second.logical_once_key
    assert first.event_digest == second.event_digest
    changed = validate_sibling_supervisor_event(
        _record(event=_canonical_event(event_id="event:sibling-002"))
    )
    assert changed.logical_once_key != first.logical_once_key


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-035"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["plan_epoch"] == 1
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "SupervisorFabric"
    assert (
        manifest["canonical_extension"]["binding"]
        == SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    assert (
        manifest["canonical_extension"]["entrypoint"]
        == "validate_sibling_supervisor_event"
    )
    assert manifest["canonical_extension"]["consumes"] == list(
        SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES
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
    assert receipt["title"] == "Add sibling-supervisor event validation"
