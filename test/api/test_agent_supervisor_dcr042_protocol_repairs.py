"""DCR-042: fail-closed JSON-RPC, schema, CID, and profile repair operators.

Acceptance:
* Negative vectors cover HTTP errors, wrong IDs/version, bad schemas/CIDs/
  receipts, unsupported profiles, policy outage, and transport mismatch.
* Never convert initialize/HTTP/RPC/policy errors to success, trust server
  verified flags, or downgrade an explicitly required profile.
* Operators remain proposal-only and never grant write/proof/semantic authority.
"""

from __future__ import annotations

import base64

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.protocol_repairs import (
    JSONRPC_VERSION,
    MCP_PROFILE_A,
    MCP_PROFILE_B,
    MCP_PROFILE_D,
    MCP_PROFILE_E,
    MCP_PROFILES_A_F,
    PROTOCOL_REPAIR_EVIDENCE,
    PROTOCOL_REPAIR_OPERATORS_INTERFACE,
    CanonicalCidOperator,
    CidReceipt,
    JsonRpcEnvelope,
    JsonRpcValidationOperator,
    OperatorRole,
    PolicyAvailability,
    ProfileNegotiation,
    ProfileNegotiationOperator,
    ProtocolRepairRequest,
    ProtocolVerdict,
    ReasonCode,
    RepairDisposition,
    SchemaBinding,
    SchemaBindingOperator,
    TransportKind,
    build_protocol_repair_operators,
    local_cid_for_bytes,
    materialize_protocol_operator_vectors,
    reviewed_schema_binding_for,
    validate_cid_receipt,
    validate_jsonrpc_envelope,
    validate_profile_negotiation,
    validate_schema_binding,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_protocol_repair_interface_and_registry_family() -> None:
    assert PROTOCOL_REPAIR_OPERATORS_INTERFACE == "ProtocolRepairOperators@1"
    assert PROTOCOL_REPAIR_EVIDENCE == "dcr/protocol-repair@1"
    reg = build_default_operator_registry()
    for kind, family_role in (
        (OperatorKind.REPAIR_JSONRPC_SCHEMA, OperatorRole.JSONRPC_VALIDATION),
        (OperatorKind.REPAIR_REQUEST_ADAPTER, OperatorRole.SCHEMA_BINDING),
        (OperatorKind.REPAIR_ERROR_ENVELOPE, OperatorRole.CANONICAL_CID),
        (OperatorKind.REPAIR_PROFILE_BINDING, OperatorRole.PROFILE_NEGOTIATION),
    ):
        descriptor = reg.require_known(kind)
        assert descriptor.family is OperatorFamily.PROTOCOL
        assert descriptor.proposal_only is True
        assert descriptor.grants_write_authority is False

    operators = build_protocol_repair_operators()
    for operator in (
        operators.jsonrpc_validation,
        operators.schema_binding,
        operators.canonical_cid,
        operators.profile_negotiation,
    ):
        assert operator.operator_id.startswith("dcr-operator:")
        assert operator.descriptor.family is OperatorFamily.PROTOCOL


def test_jsonrpc_validation_accepts_valid_envelope() -> None:
    envelope = JsonRpcEnvelope(
        request_id=42,
        response_id=42,
        jsonrpc=JSONRPC_VERSION,
        method="tools/list",
        http_status=200,
        result={"tools": []},
        has_result=True,
    )
    verdict, reasons = validate_jsonrpc_envelope(envelope)
    assert verdict is ProtocolVerdict.PASS
    assert ReasonCode.OK.value in reasons

    receipt = JsonRpcValidationOperator().apply(
        ProtocolRepairRequest(role=OperatorRole.JSONRPC_VALIDATION, envelope=envelope)
    )
    assert receipt.disposition is RepairDisposition.ACCEPTED
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"http_status": 500}, ReasonCode.HTTP_ERROR.value),
        ({"http_status": 503}, ReasonCode.HTTP_ERROR.value),
        ({"jsonrpc": "1.0"}, ReasonCode.WRONG_JSONRPC_VERSION.value),
        ({"response_id": 99}, ReasonCode.WRONG_ID.value),
        ({"response_id": None}, ReasonCode.WRONG_ID.value),
        (
            {
                "result": {"ok": True},
                "error": {"code": -32000, "message": "fail"},
                "has_result": True,
                "has_error": True,
            },
            ReasonCode.BOTH_RESULT_AND_ERROR.value,
        ),
        (
            {"result": None, "error": None, "has_result": False, "has_error": False},
            ReasonCode.MISSING_RESULT_AND_ERROR.value,
        ),
    ],
)
def test_jsonrpc_validation_negative_vectors(kwargs: dict, code: str) -> None:
    base = {
        "request_id": 1,
        "response_id": 1,
        "jsonrpc": JSONRPC_VERSION,
        "method": "initialize",
        "http_status": 200,
        "result": {"ok": True},
        "has_result": True,
    }
    base.update(kwargs)
    envelope = JsonRpcEnvelope(**base)
    verdict, reasons = validate_jsonrpc_envelope(envelope)
    assert verdict is ProtocolVerdict.REJECT
    assert code in reasons
    receipt = JsonRpcValidationOperator().apply(
        ProtocolRepairRequest(role=OperatorRole.JSONRPC_VALIDATION, envelope=envelope)
    )
    assert receipt.disposition is RepairDisposition.REJECTED
    assert code in receipt.reason_codes


def test_schema_binding_operator_preview_and_inverse() -> None:
    reviewed = reviewed_schema_binding_for("tools/call")
    operator = SchemaBindingOperator()
    preview = operator.apply(
        ProtocolRepairRequest(
            role=OperatorRole.SCHEMA_BINDING,
            reviewed_schema_binding=reviewed,
            schema_binding=None,
        )
    )
    assert preview.disposition is RepairDisposition.PREVIEW_READY
    assert preview.preview_schema_binding is not None
    assert preview.preview_schema_binding.content_id == reviewed.content_id
    assert operator.inverse(preview) is None

    aligned = operator.apply(
        ProtocolRepairRequest(
            role=OperatorRole.SCHEMA_BINDING,
            reviewed_schema_binding=reviewed,
            schema_binding=reviewed,
        )
    )
    assert aligned.disposition is RepairDisposition.ALREADY_ALIGNED
    assert operator.inverse(aligned) is not None
    assert operator.inverse(aligned).content_id == reviewed.content_id


def test_schema_binding_rejects_bad_and_unknown_methods() -> None:
    with pytest.raises(Exception):
        SchemaBinding(
            method="tools/invented",
            request_schema_ref="schema:tools/invented@1",
            response_schema_ref="schema:tools/invented/response@1",
        )

    reviewed = reviewed_schema_binding_for("tools/list")
    drifted = SchemaBinding(
        method="tools/list",
        request_schema_ref="schema:mcp/tools-list@1-forged",
        response_schema_ref=reviewed.response_schema_ref,
    )
    verdict, reasons = validate_schema_binding(drifted)
    assert verdict is ProtocolVerdict.REJECT
    assert ReasonCode.BAD_SCHEMA.value in reasons

    abstain = SchemaBindingOperator().apply(
        ProtocolRepairRequest(role=OperatorRole.SCHEMA_BINDING)
    )
    assert abstain.disposition is RepairDisposition.ABSTAIN
    assert ReasonCode.INVENTED_SCHEMA.value in abstain.reason_codes


def test_canonical_cid_never_trusts_server_verified_flag() -> None:
    payload = b'{"receipt":true}'
    good_cid = local_cid_for_bytes(payload)
    good = CidReceipt(
        claimed_cid=good_cid,
        found=True,
        server_verified=False,
        bytes_base64=_b64(payload),
    )
    verdict, reasons = validate_cid_receipt(good)
    assert verdict is ProtocolVerdict.PASS
    assert good.local_verified() is True

    forged = CidReceipt(
        claimed_cid="sha256:" + ("cd" * 32),
        found=True,
        server_verified=True,
        bytes_base64=_b64(payload),
    )
    verdict, reasons = validate_cid_receipt(forged)
    assert verdict is ProtocolVerdict.REJECT
    assert ReasonCode.SERVER_VERIFIED_UNTRUSTED.value in reasons
    assert ReasonCode.LOCAL_CID_MISMATCH.value in reasons
    assert forged.local_verified() is False

    multiformat = CidReceipt(
        claimed_cid="bafkreicecnx2gvntm6fbcrvnc336qze6st5u7qq7457igegamd3bzkx7ri",
        found=True,
        server_verified=True,
        bytes_base64=_b64(payload),
    )
    verdict, reasons = validate_cid_receipt(multiformat)
    assert verdict is ProtocolVerdict.REJECT
    assert ReasonCode.SERVER_VERIFIED_UNTRUSTED.value in reasons

    receipt = CanonicalCidOperator().apply(
        ProtocolRepairRequest(role=OperatorRole.CANONICAL_CID, cid_receipt=forged)
    )
    assert receipt.disposition is RepairDisposition.REJECTED
    assert receipt.local_verified is False


def test_profile_negotiation_subset_and_required_profiles() -> None:
    ok = ProfileNegotiation(
        requested_profiles=MCP_PROFILES_A_F,
        offered_profiles=(MCP_PROFILE_A, MCP_PROFILE_B, MCP_PROFILE_E),
        required_profiles=(MCP_PROFILE_A, MCP_PROFILE_B),
        policy_decision="allow",
    )
    verdict, reasons, negotiated = validate_profile_negotiation(ok)
    assert verdict is ProtocolVerdict.PASS
    assert negotiated == (MCP_PROFILE_A, MCP_PROFILE_B, MCP_PROFILE_E)

    missing_required = ProfileNegotiation(
        requested_profiles=(MCP_PROFILE_B,),
        offered_profiles=(MCP_PROFILE_B,),
        required_profiles=(MCP_PROFILE_D,),
        policy_decision="allow",
    )
    verdict, reasons, _ = validate_profile_negotiation(missing_required)
    assert verdict is ProtocolVerdict.REJECT
    assert ReasonCode.UNSUPPORTED_PROFILE.value in reasons
    assert ReasonCode.PROFILE_DOWNGRADE.value in reasons


def test_profile_negotiation_policy_outage_and_transport_mismatch() -> None:
    outage = ProfileNegotiation(
        requested_profiles=(MCP_PROFILE_D,),
        offered_profiles=(MCP_PROFILE_D,),
        required_profiles=(MCP_PROFILE_D,),
        policy_availability=PolicyAvailability.OUTAGE,
        policy_decision="allow",
    )
    verdict, reasons, _ = validate_profile_negotiation(outage)
    assert verdict is ProtocolVerdict.DENY
    assert ReasonCode.POLICY_OUTAGE.value in reasons
    assert ReasonCode.POLICY_ALLOW_FROM_OUTAGE.value in reasons
    denied = ProfileNegotiationOperator().apply(
        ProtocolRepairRequest(
            role=OperatorRole.PROFILE_NEGOTIATION, profile_negotiation=outage
        )
    )
    assert denied.disposition is RepairDisposition.DENIED

    mismatch = ProfileNegotiation(
        requested_profiles=(MCP_PROFILE_E,),
        offered_profiles=(MCP_PROFILE_E,),
        required_profiles=(MCP_PROFILE_E,),
        client_transport=TransportKind.HTTP,
        server_transport=TransportKind.LIBP2P,
        policy_decision="allow",
    )
    verdict, reasons, _ = validate_profile_negotiation(mismatch)
    assert verdict is ProtocolVerdict.REJECT
    assert ReasonCode.TRANSPORT_MISMATCH.value in reasons


def test_materialize_protocol_operator_vectors_cover_acceptance() -> None:
    vectors = materialize_protocol_operator_vectors()
    assert vectors["interface"] == PROTOCOL_REPAIR_OPERATORS_INTERFACE
    assert vectors["evidence_id"] == PROTOCOL_REPAIR_EVIDENCE
    assert vectors["server_verified_trusted"] is False
    assert vectors["policy_outage_denies"] is True
    assert vectors["vector_digest"].startswith("sha256:")

    names = {case["name"] for case in vectors["cases"]}
    for required in (
        "http_error",
        "wrong_version",
        "wrong_id",
        "bad_schema",
        "bad_cid_server_verified",
        "unsupported_profile",
        "policy_outage",
        "transport_mismatch",
    ):
        assert required in names

    by_name = {case["name"]: case for case in vectors["cases"]}
    assert by_name["http_error"]["disposition"] == RepairDisposition.REJECTED.value
    assert by_name["wrong_id"]["disposition"] == RepairDisposition.REJECTED.value
    assert by_name["wrong_version"]["disposition"] == RepairDisposition.REJECTED.value
    assert by_name["policy_outage"]["disposition"] == RepairDisposition.DENIED.value
    assert by_name["policy_outage"]["verdict"] == ProtocolVerdict.DENY.value
    assert by_name["transport_mismatch"]["disposition"] == RepairDisposition.REJECTED.value
    assert by_name["bad_cid_server_verified"]["local_verified"] is False
    assert by_name["cid_ok"]["local_verified"] is True
    assert by_name["profile_ok"]["disposition"] == RepairDisposition.ACCEPTED.value

    # Content-addressed and deterministic.
    again = materialize_protocol_operator_vectors()
    assert again["vector_digest"] == vectors["vector_digest"]
    assert again["content_id"] == vectors["content_id"]


def test_receipts_remain_proposal_only() -> None:
    operators = build_protocol_repair_operators()
    receipt = operators.jsonrpc_validation.apply(
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id="a",
                response_id="a",
                jsonrpc=JSONRPC_VERSION,
                method="ping",
                result={},
                has_result=True,
            ),
        )
    )
    payload = receipt.to_dict()
    assert payload["proposal_only"] is True
    assert payload["grants_write_authority"] is False
    assert payload["grants_proof_authority"] is False
    assert payload["semantic_authority"] is False
    assert payload["evidence_id"] == PROTOCOL_REPAIR_EVIDENCE
