"""DCR-050 strict composition adapter tests (no Doctor is built)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    CapabilityEvidenceReceipt,
    CapabilityReceipt,
    CapabilityStatus,
    NetworkMode,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.control.default_doctor_factory import (
    DcrDoctorCompositionAdapter,
    DcrDoctorCompositionError,
    inspect_dcr_doctor_composition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    REQUIRED_IR_LOGIC_IDENTITIES,
    REQUIRED_IR_LOGIC_STAGES,
    IrLogicRequiredGateDisposition,
    IrLogicRequiredGateResult,
)


def _record(kind: str, binding: dict[str, str]) -> dict[str, Any]:
    binding = dict(sorted(binding.items()))
    self_test = {
        "kind": kind,
        "binding_cid": content_identity({"kind": kind, "binding": binding}),
        "outcome": "passed",
        "mode": "read_only",
        "model_call_count": 0,
        "provider_call_count": 0,
        "network_call_count": 0,
    }
    return {
        "kind": kind,
        "identity": content_identity({"kind": kind, "binding": binding, "self_test": self_test}),
        "binding": binding,
        "self_test": self_test,
    }


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "dcr050.fixture",
            "kind": "replace_exact_bytes",
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "owner_root": "fixture",
            "write_scope": ["fixture.py"],
            "before_predicates": ["before"],
            "after_predicates": ["after"],
            "applicability_proofs": ["proof"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "before_bytes"},
            "validation_commands": [["pytest", "fixture.py"]],
        }
    )
    return OperatorRegistry(
        [descriptor], reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _capability() -> tuple[CapabilityReceipt, tuple[CapabilityEvidenceReceipt, ...]]:
    digest = "module:sha256:" + "a" * 64
    receipt = CapabilityReceipt(
        capability_id="ipfs_datasets_py.logic",
        status=CapabilityStatus.AVAILABLE,
        origin="/fixtures/logic.py",
        distribution="ipfs-datasets-py",
        expected_version="1.0",
        distribution_version="1.0",
        content_digest=digest,
        symbols=("prove",),
        initialized=True,
        reconstructed=True,
        self_test_passed=True,
        network_mode=NetworkMode.OFFLINE,
    )
    evidence = tuple(
        CapabilityEvidenceReceipt(
            evidence_id=receipt.capability_id,
            evidence_kind=kind,
            subject_id=receipt.capability_id,
            subject_digest=digest,
            subject_version="1.0",
            transcript_digest="transcript:sha256:" + kind.ljust(64, "0"),
            passed=True,
        )
        for kind in ("initialization", "reconstruction", "self_test")
    )
    return receipt, evidence


def _binding() -> dict[str, Any]:
    forest_id = "sha256:forest"
    checkout = _record(
        "checkout_forest", {"checkout_id": "sha256:checkout", "forest_id": forest_id}
    )
    graph = _record(
        "graph_findings",
        {"forest_id": forest_id, "graph_cid": "sha256:graph", "findings_cid": "sha256:findings"},
    )
    proof = _record(
        "proof_cache",
        {"cache_id": "sha256:cache", "forest_id": forest_id, "graph_cid": "sha256:graph"},
    )
    store = _record("receipt_store", {"forest_id": forest_id, "store_id": "sha256:store"})
    reader = _record(
        "source_reader",
        {"forest_id": forest_id, "reader_id": "sha256:reader", "source_digest": "sha256:source"},
    )
    transaction = _record(
        "transaction_controller",
        {
            "controller_id": "sha256:transaction",
            "forest_id": forest_id,
            "receipt_store_id": "sha256:store",
        },
    )
    identities = {name: f"sha256:{name}" for name in REQUIRED_IR_LOGIC_IDENTITIES}
    identities["dcr034"] = proof["identity"]
    gate = IrLogicRequiredGateResult(
        disposition=IrLogicRequiredGateDisposition.PASSING,
        reason_codes=(),
        required_identity_cids=identities,
        receipt_ids=tuple(f"ir-logic-stage:sha256:{stage}" for stage in REQUIRED_IR_LOGIC_STAGES),
    )
    capability, evidence = _capability()
    registry = _registry()
    return {
        "checkout_forest": checkout,
        "graph_findings": graph,
        "logic_capability_receipt": capability,
        "logic_evidence_receipts": evidence,
        "dcr035_gate": gate,
        "operator_registry": registry,
        "reviewed_registry_cid": registry.report()["registry_cid"],
        "proof_cache": proof,
        "receipt_store": store,
        "source_reader": reader,
        "transaction_controller": transaction,
    }


def test_complete_binding_is_projected_but_remains_live_integration_pending() -> None:
    adapter = DcrDoctorCompositionAdapter()
    result = adapter.inspect(_binding())

    assert result.disposition == "integration_pending"
    assert result.binding_complete is True
    report = result.to_dict()
    assert report["execution_authorized"] is False
    assert report["model_call_count"] == report["provider_call_count"] == 0
    assert adapter.factory.last_binding is None
    assert "dcr040_registry" in report["identities"]


@pytest.mark.parametrize(
    "field",
    [
        "checkout_forest",
        "graph_findings",
        "logic_capability_receipt",
        "logic_evidence_receipts",
        "dcr035_gate",
        "operator_registry",
        "reviewed_registry_cid",
        "proof_cache",
        "receipt_store",
        "source_reader",
        "transaction_controller",
    ],
)
def test_every_mandatory_binding_missing_or_forged_defers(field: str) -> None:
    binding = _binding()
    binding.pop(field)
    result = inspect_dcr_doctor_composition(binding)
    assert result.disposition == "deferred"
    assert result.binding_complete is False
    assert result.to_dict()["execution_authorized"] is False


def test_stale_identity_callable_and_empty_source_are_rejected() -> None:
    stale = _binding()
    stale["proof_cache"] = {**stale["proof_cache"], "identity": "sha256:forged"}
    assert inspect_dcr_doctor_composition(stale).disposition == "deferred"

    callable_binding = _binding()
    callable_binding["source_reader"] = lambda: None
    assert inspect_dcr_doctor_composition(callable_binding).disposition == "deferred"

    empty_source = _binding()
    source = empty_source["source_reader"]
    source["binding"] = {**source["binding"], "source_digest": ""}
    assert inspect_dcr_doctor_composition(empty_source).disposition == "deferred"

    wrong_capability = _binding()
    wrong_capability["logic_capability_receipt"] = replace(
        wrong_capability["logic_capability_receipt"], self_test_passed=False
    )
    assert inspect_dcr_doctor_composition(wrong_capability).disposition == "deferred"


def test_adapter_refuses_arbitrary_factory_callable() -> None:
    with pytest.raises(DcrDoctorCompositionError):
        DcrDoctorCompositionAdapter(factory=lambda: None)  # type: ignore[arg-type]
