"""Independent current-tree checks for DOEP-100 sibling capability registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import (
    ADMITTED_SIBLING_CAPABILITIES,
    CANONICAL_EVENT_INTERFACE,
    CANONICAL_EVENT_SCHEMA_ID,
    DATABASE_EVENT_LOG_INTERFACE,
    FORBIDDEN_SIBLING_CAPABILITIES,
    SIBLING_CAPABILITY_DEFAULT_EFFECT,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    SUPERVISOR_FABRIC_INTERFACE,
    SiblingSupervisorCapabilityRecord,
    SiblingSupervisorCapabilityRegistryError,
    SiblingSupervisorCapabilityRegistrySnapshot,
    SupervisorFabric,
    SupervisorFabricError,
    admit_sibling_supervisor_capability_registry,
    issue_fence,
    lookup_sibling_supervisor_capability,
    register_sibling_supervisor_capability,
    sibling_supervisor_capability_catalog,
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


def _capability_record(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "local_supervisor_id": "supervisor:local",
        "sibling_supervisor_id": "supervisor:peer",
        "capability": "event-exchange",
        "epoch": 2,
        "effect": "event_exchange",
    }
    values.update(overrides)
    return values


def _canonical_event(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema": CANONICAL_EVENT_SCHEMA_ID,
        "event_id": "event:sibling-100",
        "event_type": "task.validation.completed",
        "stream_id": "task:DOEP-100",
        "causal_parent_ids": ["event:parent-100"],
        "correlation_id": "correlation:attempt-100",
        "causation_id": "causation:validation-100",
        "payload": {"outcome": "passed", "attempt": 1},
    }
    values.update(overrides)
    return values


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_registry_extends_supervisor_fabric_without_competing_subsystem() -> None:
    assert SUPERVISOR_FABRIC_INTERFACE == "SupervisorFabric@1"
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
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
        CANONICAL_EVENT_INTERFACE,
        DATABASE_EVENT_LOG_INTERFACE,
    )
    assert SupervisorFabric.INTERFACE == SUPERVISOR_FABRIC_INTERFACE
    assert (
        SupervisorFabric.SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
        == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    assert register_sibling_supervisor_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert lookup_sibling_supervisor_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert admit_sibling_supervisor_capability_registry.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    assert SupervisorFabric.register_sibling_capability.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric"
    )
    source = FABRIC_PATH.read_text(encoding="utf-8")
    assert "class SupervisorFabric" in source
    assert "def issue_fence(" in source
    assert "def register_sibling_supervisor_capability(" in source
    assert "def lookup_sibling_supervisor_capability(" in source
    assert "def admit_sibling_supervisor_capability_registry(" in source
    assert "not a competing subsystem" in source
    assert "class SiblingCapabilityService" not in source
    assert "class SiblingCapabilityDatabase" not in source
    assert "class CompetingSiblingCapability" not in source
    assert "CREATE TABLE" not in source


def test_closed_catalog_is_not_runtime_extensible() -> None:
    catalog = sibling_supervisor_capability_catalog()
    assert tuple(item["capability"] for item in catalog) == ADMITTED_SIBLING_CAPABILITIES
    assert ADMITTED_SIBLING_CAPABILITIES == (
        "event-exchange",
        "incremental-reassessment",
        "receipt-exchange",
        "task-request",
    )
    assert SIBLING_CAPABILITY_DEFAULT_EFFECT == "event_exchange"
    for item in catalog:
        assert item["effect"] == "event_exchange"
        assert item["database_write"] is False
        assert item["completion_authoritative"] is False
        assert item["worker_assertion_is_authority"] is False
        assert item["worker_completion_insufficient"] is True
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as unknown:
        register_sibling_supervisor_capability(_capability_record(capability="dispatch"))
    assert unknown.value.code == "unknown_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as closed:
        register_sibling_supervisor_capability(
            _capability_record(catalog_override={"dispatch": True})
        )
    assert closed.value.code == "catalog_closed"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as granted:
        admit_sibling_supervisor_capability_registry(
            {
                "local_supervisor_id": "supervisor:local",
                "epoch": 2,
                "self_granted": True,
                "records": [_capability_record()],
            }
        )
    assert granted.value.code == "catalog_closed"


def test_register_and_lookup_are_idempotent_and_non_authoritative() -> None:
    first = register_sibling_supervisor_capability(_capability_record())
    second = register_sibling_supervisor_capability(_capability_record())
    assert isinstance(first, SiblingSupervisorCapabilityRecord)
    assert first.to_dict() == second.to_dict()
    payload = first.to_dict()
    assert payload["admitted"] is True
    assert payload["fenced"] is True
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["worker_completion_insufficient"] is True
    assert payload["binding"] == SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    assert payload["carrier"] == SUPERVISOR_FABRIC_INTERFACE
    assert payload["registry_key"] == "supervisor:peer:event-exchange"
    snapshot = admit_sibling_supervisor_capability_registry(
        {
            "local_supervisor_id": "supervisor:local",
            "epoch": 2,
            "siblings": {
                "supervisor:peer": ["event-exchange", "receipt-exchange"],
                "supervisor:other": [{"capability": "task-request"}],
            },
        }
    )
    assert isinstance(snapshot, SiblingSupervisorCapabilityRegistrySnapshot)
    assert snapshot.advertised_for("supervisor:peer") == (
        "event-exchange",
        "receipt-exchange",
    )
    found = lookup_sibling_supervisor_capability(
        snapshot, "supervisor:peer", "receipt-exchange"
    )
    assert found.capability == "receipt-exchange"
    again = lookup_sibling_supervisor_capability(
        snapshot.to_dict(), "supervisor:peer", "event-exchange"
    )
    assert again.sibling_supervisor_id == "supervisor:peer"
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
        sibling_capabilities=(_capability_record(capability="event-exchange"),),
    )
    registered = fabric.register_sibling_capability(
        {"sibling_supervisor_id": "supervisor:peer", "capability": "event-exchange"}
    )
    assert registered.to_dict()["database_write"] is False
    looked = fabric.lookup_sibling_capability("supervisor:peer", "event-exchange")
    assert looked.registry_key == registered.registry_key
    assert fabric.sibling_capability_snapshot().registry_digest.startswith("sha256:")


def test_unknown_sibling_self_and_unregistered_capability_fail_closed() -> None:
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as self_exc:
        register_sibling_supervisor_capability(
            _capability_record(
                local_supervisor_id="supervisor:local",
                sibling_supervisor_id="supervisor:local",
            )
        )
    assert self_exc.value.code == "not_a_sibling"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as unknown:
        register_sibling_supervisor_capability(
            _capability_record(known_sibling_ids=("supervisor:other",))
        )
    assert unknown.value.code == "unknown_sibling"
    snapshot = admit_sibling_supervisor_capability_registry(
        {
            "local_supervisor_id": "supervisor:local",
            "epoch": 2,
            "records": [_capability_record()],
        }
    )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as missing:
        lookup_sibling_supervisor_capability(
            snapshot, "supervisor:peer", "task-request"
        )
    assert missing.value.code == "unregistered_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as peer:
        lookup_sibling_supervisor_capability(
            snapshot, "supervisor:other", "event-exchange"
        )
    assert peer.value.code == "unregistered_capability"


def test_stale_epoch_and_missing_capability_fail_closed() -> None:
    fence = issue_fence({"supervisor_id": "S1", "capability": "dispatch", "epoch": 2})
    assert fence["fenced"] is True
    with pytest.raises(SupervisorFabricError, match="stale"):
        register_sibling_supervisor_capability(_capability_record(stale_epoch=True))
    with pytest.raises(SupervisorFabricError, match="capability"):
        register_sibling_supervisor_capability(_capability_record(capability=""))
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as epoch:
        admit_sibling_supervisor_capability_registry(
            {
                "local_supervisor_id": "supervisor:local",
                "epoch": 2,
                "stale_epoch": True,
                "records": [_capability_record()],
            }
        )
    assert epoch.value.code == "stale_fence_epoch"


def test_forbidden_capabilities_and_direct_state_writes_are_rejected() -> None:
    assert "database-write" in FORBIDDEN_SIBLING_CAPABILITIES
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as forbidden:
        register_sibling_supervisor_capability(
            _capability_record(capability="database-write")
        )
    assert forbidden.value.code == "forbidden_capability"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as duckdb:
        register_sibling_supervisor_capability(
            _capability_record(duckdb_path="/tmp/control.duckdb")
        )
    assert duckdb.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as sql:
        admit_sibling_supervisor_capability_registry(
            {
                "local_supervisor_id": "supervisor:local",
                "epoch": 2,
                "sql": "UPDATE tasks SET status='done'",
                "records": [_capability_record()],
            }
        )
    assert sql.value.code == "direct_state_write"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as effect:
        register_sibling_supervisor_capability(
            _capability_record(effect="authoritative_state")
        )
    assert effect.value.code == "forbidden_effect"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as completion:
        register_sibling_supervisor_capability(
            _capability_record(completion_authoritative=True)
        )
    assert completion.value.code == "completion_not_authoritative"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as advertised:
        register_sibling_supervisor_capability(_capability_record(advertised=False))
    assert advertised.value.code == "capability_not_advertised"


def test_worker_assertion_cannot_grant_or_widen_capability() -> None:
    admitted = register_sibling_supervisor_capability(
        _capability_record(worker_assertion=True)
    )
    assert admitted.to_dict()["worker_assertion_is_authority"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(SiblingSupervisorCapabilityRegistryError):
        register_sibling_supervisor_capability(
            _capability_record(worker_assertion=True, capability="write-database")
        )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError):
        register_sibling_supervisor_capability(
            _capability_record(worker_assertion=True, duckdb_path="events.duckdb")
        )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as conflict:
        admit_sibling_supervisor_capability_registry(
            {
                "local_supervisor_id": "supervisor:local",
                "epoch": 2,
                "worker_assertion": True,
                "records": [
                    _capability_record(effect="event_exchange"),
                    _capability_record(effect="read_only"),
                ],
            }
        )
    assert conflict.value.code == "capability_conflict"


def test_registry_gates_sibling_event_admission_without_database_write() -> None:
    snapshot = admit_sibling_supervisor_capability_registry(
        {
            "local_supervisor_id": "supervisor:local",
            "epoch": 2,
            "records": [_capability_record()],
        }
    )
    admitted = validate_sibling_supervisor_event(
        {
            "local_supervisor_id": "supervisor:local",
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "epoch": 2,
            "effect": "event_exchange",
            "capability_registry": snapshot.to_dict(),
            "event": _canonical_event(),
        }
    )
    assert admitted.to_dict()["database_write"] is False
    assert admitted.to_dict()["completion_authoritative"] is False
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as unregistered:
        validate_sibling_supervisor_event(
            {
                "local_supervisor_id": "supervisor:local",
                "sibling_supervisor_id": "supervisor:peer",
                "capability": "task-request",
                "epoch": 2,
                "capability_registry": snapshot,
                "event": _canonical_event(),
            }
        )
    assert unregistered.value.code == "unregistered_capability"
    fabric = SupervisorFabric(
        supervisor_id="supervisor:local",
        epoch=2,
        known_sibling_ids=("supervisor:peer",),
        sibling_capabilities=(_capability_record(),),
    )
    again = fabric.validate_sibling_event(
        {
            "sibling_supervisor_id": "supervisor:peer",
            "capability": "event-exchange",
            "event": _canonical_event(),
        }
    )
    assert again.event_id == "event:sibling-100"
    with pytest.raises(SiblingSupervisorCapabilityRegistryError):
        fabric.validate_sibling_event(
            {
                "sibling_supervisor_id": "supervisor:peer",
                "capability": "incremental-reassessment",
                "event": _canonical_event(),
            }
        )


def test_digest_mismatch_and_fabric_conflict_fail_closed() -> None:
    snapshot = admit_sibling_supervisor_capability_registry(
        {
            "local_supervisor_id": "supervisor:local",
            "epoch": 2,
            "records": [_capability_record()],
        }
    )
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as digest:
        admit_sibling_supervisor_capability_registry(
            {
                "local_supervisor_id": "supervisor:local",
                "epoch": 2,
                "records": [_capability_record()],
                "registry_digest": "sha256:" + ("0" * 64),
            }
        )
    assert digest.value.code == "registry_digest_mismatch"
    rebuilt = SiblingSupervisorCapabilityRegistrySnapshot(
        local_supervisor_id=snapshot.local_supervisor_id,
        epoch=snapshot.epoch,
        records=snapshot.records,
        registry_digest=snapshot.registry_digest,
    )
    assert rebuilt.to_dict()["registry_digest"] == snapshot.registry_digest
    fabric = SupervisorFabric(supervisor_id="supervisor:local", epoch=2)
    fabric.register_sibling_capability(_capability_record())
    with pytest.raises(SiblingSupervisorCapabilityRegistryError) as conflict:
        fabric.register_sibling_capability(_capability_record(effect="read_only"))
    assert conflict.value.code == "capability_conflict"


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
