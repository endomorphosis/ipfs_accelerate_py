"""Focused DCR-042 protocol operator previews; no repository mutation occurs."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.protocol_repairs import (
    JsonSchemaBinding,
    ProtocolRepairRequest,
    ProtocolRepairStatus,
    preview_protocol_repair,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    IrLogicRequiredStageReceipt,
    evaluate_required_ir_logic_gate,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_reconstruction import (
    KernelReconstructionDisposition,
    KernelReconstructionResult,
    KernelReconstructionRoots,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_obligations import (
    McpGraphContractObligation,
    McpObligationBackend,
    McpObligationDisposition,
    McpObligationFamily,
    McpObligationFragment,
)


def _schema(properties: dict[str, str]) -> JsonSchemaBinding:
    schema = {
        "type": "object",
        "required": sorted(properties),
        "properties": properties,
        "additional_properties": False,
    }
    return JsonSchemaBinding(schema, content_identity(schema))


def _descriptor() -> tuple[OperatorDescriptor, OperatorRegistry]:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "protocol.replace",
            "kind": "replace_exact_bytes",
            "owner_root": "ipfs-accelerate",
            "write_scope": ["fixture.py"],
            "before_predicates": ["protocol_response_valid"],
            "after_predicates": ["protocol_response_valid"],
            "applicability_proofs": ["dcr031"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["python", "-m", "py_compile", "fixture.py"]],
        }
    )
    return descriptor, OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _request() -> ProtocolRepairRequest:
    source = b"return BAD_RESPONSE\n"
    obligation = McpGraphContractObligation(
        obligation_id="obligation:dcr031",
        family=McpObligationFamily.JSONRPC_BASELINE,
        fragment=McpObligationFragment.JSONRPC,
        backend=McpObligationBackend.LOGIC_IR_CANDIDATE,
        disposition=McpObligationDisposition.OPEN,
        graph_cid="graph:dcr021",
        candidate_cid="candidate:dcr030",
        input_cids=("candidate:dcr030", "graph:dcr021"),
    )
    roots = KernelReconstructionRoots(
        RepairAuthorityRoots(
            "repo",
            "forest:current",
            "tree:current",
            "policy:current",
            "plan:current",
            "packet:current",
        ),
        obligation.graph_cid,
        "live:current",
    )
    reconstruction = KernelReconstructionResult(
        KernelReconstructionDisposition.RECONSTRUCTED,
        (),
        request_cid="request:dcr033",
        proof_cid="proof:current",
        certificate_cid="certificate:current",
        roots=roots,
    )
    identities = {
        "dcr030": "dcr030:current",
        "dcr031": content_identity(obligation.to_dict()),
        "dcr032": "dcr032:current",
        "dcr033": reconstruction.to_dict()["result_cid"],
        "dcr034": "dcr034:current",
    }
    gate = evaluate_required_ir_logic_gate(
        [
            IrLogicRequiredStageReceipt(stage, identities, ("surface:" + stage,))
            for stage in ("diagnose", "plan", "admit", "apply", "complete")
        ],
        required_identity_cids=identities,
    )
    descriptor, registry = _descriptor()
    return ProtocolRepairRequest(
        source,
        "sha256:0",
        "ipfs-accelerate",
        "fixture.py",
        7,
        19,
        "sha256:0",
        b"GOOD_RESPONSE",
        descriptor,
        registry,
        registry.report()["registry_cid"],
        descriptor.descriptor_id,
        obligation,
        reconstruction,
        gate,
        200,
        {"jsonrpc": "2.0", "id": 7, "method": "fixture.op", "params": {}},
        {"jsonrpc": "2.0", "id": 7, "result": {"value": "ok"}},
        _schema({"jsonrpc": "string", "id": "integer", "method": "string", "params": "object"}),
        _schema({"value": "string"}),
        ("mcp-idl",),
        ("mcp-idl", "cid-envelope"),
        True,
        "http_jsonrpc",
    )


def _bound_request() -> ProtocolRepairRequest:
    request = _request()
    span = request.source_bytes[request.span_start : request.span_end]
    return replace(
        request,
        source_digest="sha256:" + hashlib.sha256(request.source_bytes).hexdigest(),
        span_digest="sha256:" + hashlib.sha256(span).hexdigest(),
    )


def test_preview_is_reversible_but_swissknife_integration_stays_pending() -> None:
    preview = preview_protocol_repair(_bound_request())
    assert preview.status is ProtocolRepairStatus.PREVIEWED
    assert preview.after_bytes == b"return GOOD_RESPONSE\n"
    assert preview.forward_cid and preview.inverse_cid
    assert preview.to_dict()["activation_status"] == "swissknife_integration_pending"
    assert preview.to_dict()["execution_authorized"] is False


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda value: replace(value, http_status=500), "http_status_not_success"),
        (lambda value: replace(value, response={**value.response, "id": 8}), "jsonrpc_id_invalid"),
        (
            lambda value: replace(value, response={**value.response, "jsonrpc": "1.0"}),
            "jsonrpc_version_invalid",
        ),
        (
            lambda value: replace(value, response={**value.response, "error": {"code": 1}}),
            "jsonrpc_result_error_exclusivity_invalid",
        ),
        (
            lambda value: replace(value, response={"jsonrpc": "2.0", "id": 7}),
            "jsonrpc_result_error_exclusivity_invalid",
        ),
        (lambda value: replace(value, supported_profiles=()), "unsupported_profile"),
        (lambda value: replace(value, policy_available=False), "policy_outage"),
        (lambda value: replace(value, transport="mcp_p2p"), "transport_mismatch"),
    ],
)
def test_protocol_negative_vectors_abstain_or_reject(change, reason) -> None:
    preview = preview_protocol_repair(change(_bound_request()))
    assert preview.status in {ProtocolRepairStatus.ABSTAINED, ProtocolRepairStatus.REJECTED}
    assert preview.reason_codes == (reason,)
    assert not preview.forward_cid


def test_bad_schema_cid_descriptor_or_reconstruction_receipt_rejects() -> None:
    request = _bound_request()
    object.__setattr__(request.request_schema, "schema_cid", "bafy-forged")
    schema = preview_protocol_repair(request)
    descriptor = preview_protocol_repair(
        replace(_bound_request(), reviewed_descriptor_cid="bafy-forged")
    )
    stale = preview_protocol_repair(
        replace(
            _bound_request(),
            reconstruction=KernelReconstructionResult(
                KernelReconstructionDisposition.INVALID, ("bad",)
            ),
        )
    )
    assert schema.status is ProtocolRepairStatus.REJECTED
    assert descriptor.status is ProtocolRepairStatus.REJECTED
    assert stale.status is ProtocolRepairStatus.REJECTED


def test_self_authored_unreviewed_descriptor_is_rejected() -> None:
    request = _bound_request()
    descriptor, _registry = _descriptor()
    forged_registry = OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )
    # A different reviewed registry need not admit the request's descriptor.
    object.__setattr__(forged_registry, "_descriptors", ())
    preview = preview_protocol_repair(replace(request, registry=forged_registry))
    assert preview.status is ProtocolRepairStatus.REJECTED
    assert preview.reason_codes == ("reviewed_descriptor_manifest_invalid",)
