"""Independent current-tree checks for DOEP-100 sibling capability registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import (
    ALLOWED_SIBLING_CAPABILITIES,
    CANONICAL_EVENT_SCHEMA_ID,
    DATABASE_EVENT_LOG_INTERFACE,
    FORBIDDEN_SIBLING_CAPABILITIES,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    SUPERVISOR_FABRIC_INTERFACE,
    SiblingSupervisorCapabilityAdmission,
    SiblingSupervisorCapabilityRegistryError,
    SupervisorFabric,
    SupervisorFabricError,
    admit_sibling_supervisor_capability,
    issue_fence,
    list_sibling_supervisor_capabilities,
    lookup_sibling_supervisor_capability,
    register_sibling_supervisor_capability,
    revoke_sibling_supervisor_capability,
    sibling_supervisor_has_capability,
    validate_sibling_supervisor_event,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
FABRIC_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-100.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-100.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_fabric.py",
    "test/api/doep/test_doep_100_add_sibling_supervisor_capability_registry.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-100.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-100.json",
)
TASK_CID = "sha256:efc5b4e9a126d3e50ec396062ab45f4b4728683e93767c995e6152d886c92599"
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
        "event_id": "event:sibling-100",
        "event_type": "capability.advertised",
        "stream_id": "task:DOEP-100",
        "causal_parent_ids": ["event:parent-100"],
        "correlation_id": "correlation:attempt-100",
        "causation_id": "causation:capability-100",
        "payload": {"outcome": "advertised", "attempt": 1},
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
    }
    values.update(overrides)
    return values


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_supervisor_fabric_without_competing_subsystem() -> None:
    assert SUPERVISOR_FABRIC_INTERFACE == "SupervisorFabric@1"
    assert DATABASE_EVENT_LOG_INTERFACE == "DatabaseEventLog@1"
    assert SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING == (
        "SiblingSupervisorCapabilityRegistry@1"
    )
    assert (
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE
        == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    assert SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/sibling-supervisor-capability-registry@1"
    )
    assert SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES == (
        SUPERVISOR_FABRIC_INTERFACE,
        DATABASE_EVENT_LOG_INTERFACE,
    )
    assert SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING == (
        "SiblingSupervisorEventValidation@1"
    )
    assert SupervisorFabric.INTERFACE == SUPERVISOR_FABRIC_INTERFACE
    assert (
        SupervisorFabric.SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
        == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    assert register_sibling_supervisor_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert admit_sibling_supervisor_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert SupervisorFabric.register_sibling_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    source = FABRIC_PATH.read_text(encoding="utf-8")
    assert "class SupervisorFabric" in source
    assert "def issue_fence(" in source
    assert "def validate_sibling_supervisor_event(" in source
    assert "def admit_sibling_supervisor_capability(" in source
    assert "def register_sibling_supervisor_capability(" in source
    assert "class SiblingCapabilityDatabase" not in source
    assert "class SiblingCapabilityService" not in source
    assert "class CompetingSiblingCapabilityRegistry" not in source
    assert "class SiblingEventBus" not in source
    assert "class SiblingEventLog" not in source
    assert "CREATE TABLE" not in source
    assert "import duckdb" not in source


def test_issue_fence_still_requires_capability_and_rejects_stale_epoch() -> None:
    fence = issue_fence({"supervisor_id": "S1", "capability": "dispatch", "epoch": 2})
    assert fence["fenced"] is True
    assert fence["epoch"] == 2
    with pytest.raises(SupervisorFabricError, match="stale"):
        issue_fence({"supervisor_id": "S1", "capability": "dispatch", "stale_epoch": True})
    with pytest.raises(SupervisorFabricError, match="capability"):
        issue_fence({"supervisor_id": "S1", "capability": ""})


def test_valid_sibling_capability_is_registered_without_database_write() -> None:
    admitted = register_sibling_supervisor_capability(_record())
    assert isinstance(admitted, SiblingSupervisorCapabilityAdmission)
    payload = admitted.to_dict()
    assert payload["admitted"] is True
    assert payload["fenced"] is True
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["worker_completion_insufficient"] is True
    assert payload["binding"] == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    assert payload["carrier"] == SUPERVISOR_FABRIC_INTERFACE
    assert payload["consumes"]["database_event_log"] == DATABASE_EVENT_LOG_INTERFACE
    assert payload["capabilities"] == ["event-exchange"]
    assert payload["capability_digest"].startswith("sha256:")
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
    )
    again = fabric.register_sibling_capability(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capabilities": ["event-exchange", "task-request"],
        }
    )
    assert again.capabilities == ("event-exchange", "task-request")
    assert again.to_dict()["database_write"] is False
    assert fabric.sibling_has_capability("supervisor:peer", "task-request") is True
    snapshot = fabric.capability_registry_snapshot()
    assert snapshot["supervisor:peer"]["capabilities"] == [
        "event-exchange",
        "task-request",
    ]


def test_self_supervisor_and_unknown_peer_fail_closed() -> None:
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as self_exc:
        register_sibling_supervisor_capability(
            _record(
                local_supervisor_id="supervisor:local",
                sibling_supervisor_id="supervisor:local",
            )
        )
    assert self_exc.value.code == "not_a_sibling"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as unknown:
        register_sibling_supervisor_capability(
            _record(known_sibling_ids=("supervisor:other",))
        )
    assert unknown.value.code == "unknown_sibling"


def test_stale_epoch_and_missing_capability_fail_closed() -> None:
    with pytest.raises(SupervisorFabricError, match="stale"):
        register_sibling_supervisor_capability(_record(stale_epoch=True))
    with pytest.raises(SupervisorFabricError, match="capability"):
        register_sibling_supervisor_capability(_record(capability=""))


def test_forbidden_and_unknown_capabilities_fail_closed() -> None:
    assert "event-exchange" in ALLOWED_SIBLING_CAPABILITIES
    assert "task-request" in ALLOWED_SIBLING_CAPABILITIES
    assert "receipt-exchange" in ALLOWED_SIBLING_CAPABILITIES
    assert "incremental-reassessment" in ALLOWED_SIBLING_CAPABILITIES
    assert "database-write" in FORBIDDEN_SIBLING_CAPABILITIES
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as forbidden:
        register_sibling_supervisor_capability(_record(capability="database-write"))
    assert forbidden.value.code == "forbidden_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as terminal:
        register_sibling_supervisor_capability(
            _record(capability="task-terminalization")
        )
    assert terminal.value.code == "forbidden_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as unknown:
        register_sibling_supervisor_capability(_record(capability="model-completion"))
    assert unknown.value.code == "unknown_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as owner:
        register_sibling_supervisor_capability(_record(capability="quack-state-owner"))
    assert owner.value.code == "forbidden_capability"


def test_direct_state_writes_and_log_mutations_are_rejected() -> None:
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as duckdb:
        register_sibling_supervisor_capability(_record(duckdb_path="/tmp/control.duckdb"))
    assert duckdb.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as sql:
        register_sibling_supervisor_capability(
            _record(sql="UPDATE tasks SET status='done'")
        )
    assert sql.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as consume:
        register_sibling_supervisor_capability(_record(consume=True))
    assert consume.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as effect:
        register_sibling_supervisor_capability(_record(effect="authoritative_state"))
    assert effect.value.code == "forbidden_effect"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as completion:
        register_sibling_supervisor_capability(_record(completion_authoritative=True))
    assert completion.value.code == "completion_not_authoritative"


def test_worker_assertion_is_not_admission_authority() -> None:
    admitted = register_sibling_supervisor_capability(_record(worker_assertion=True))
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(SiblingSupervisorCapabilityRegistryError):
        register_sibling_supervisor_capability(
            _record(worker_assertion=True, capability="database-write")
        )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError):
        register_sibling_supervisor_capability(
            _record(worker_assertion=True, duckdb_path="events.duckdb")
        )


def test_identical_advertisements_are_idempotent_and_conflicts_fail_closed() -> None:
    registry: dict[str, SiblingSupervisorCapabilityAdmission] = {}
    first = register_sibling_supervisor_capability(_record(), registry)
    second = register_sibling_supervisor_capability(_record(), registry)
    assert first.to_dict() == second.to_dict()
    assert first.logical_once_key == second.logical_once_key
    assert first.capability_digest == second.capability_digest
    assert list(registry) == ["supervisor:peer"]
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as conflict:
        register_sibling_supervisor_capability(
            _record(capabilities=["event-exchange", "task-request"]),
            registry,
        )
    assert conflict.value.code == "capability_conflict"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as stale:
        register_sibling_supervisor_capability(_record(epoch=1), registry)
    assert stale.value.code == "stale_fence_epoch"
    replaced = register_sibling_supervisor_capability(
        _record(epoch=3, capabilities=["receipt-exchange", "event-exchange"]),
        registry,
    )
    assert replaced.epoch == 3
    assert replaced.capabilities == ("event-exchange", "receipt-exchange")
    listed = list_sibling_supervisor_capabilities(registry)
    assert [item.sibling_supervisor_id for item in listed] == ["supervisor:peer"]
    found = lookup_sibling_supervisor_capability(
        registry, "supervisor:peer", "receipt-exchange"
    )
    assert found.epoch == 3
    assert sibling_supervisor_has_capability(registry, "supervisor:peer", "task-request") is False
    revoked = revoke_sibling_supervisor_capability(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:peer",
        },
        registry,
    )
    assert revoked.sibling_supervisor_id == "supervisor:peer"
    assert registry == {}
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as missing:
        lookup_sibling_supervisor_capability(registry, "supervisor:peer")
    assert missing.value.code == "unknown_sibling"


def test_registered_capabilities_gate_sibling_event_admission() -> None:
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
    )
    unregistered = fabric.validate_sibling_event(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "event": _canonical_event(),
        }
    )
    assert unregistered.event_id == "event:sibling-100"
    fabric.register_sibling_capability(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capabilities": ["event-exchange"],
        }
    )
    admitted = fabric.validate_sibling_event(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "event": _canonical_event(),
        }
    )
    assert admitted.to_dict()["database_write"] is False
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as missing:
        fabric.validate_sibling_event(
            {
                "sibling_supervisor_id": "supervisor:peer",
                "capability": "task-request",
                "event": _canonical_event(event_id="event:sibling-101"),
            }
        )
    assert missing.value.code == "capability_not_advertised"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as stale:
        fabric.validate_sibling_event(
            {
                "sibling_supervisor_id": "supervisor:peer",
                "capability": "event-exchange",
                "epoch": 1,
                "event": _canonical_event(event_id="event:sibling-102"),
            }
        )
    assert stale.value.code == "stale_fence_epoch"
    module_event = validate_sibling_supervisor_event(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "epoch": 2,
            "event": _canonical_event(),
        }
    )
    assert module_event.event_id == "event:sibling-100"


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-100"
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
        == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    assert (
        manifest["canonical_extension"]["entrypoint"]
        == "register_sibling_supervisor_capability"
    )
    assert manifest["canonical_extension"]["consumes"] == list(
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES
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
    assert receipt["title"] == "Add sibling supervisor capability registry"
