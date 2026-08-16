"""DCR-042 protocol-repair previews: deterministic, reversible, non-writing.

This is an operator library only.  It neither locates SwissKnife code nor
applies a patch.  Every otherwise valid preview remains
``swissknife_integration_pending`` until an admitted DCR-070/072 transaction
exists outside this module.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ...proof.formal_verification_contracts import content_identity
from ...proof.ir_logic_application import IrLogicRequiredGateDisposition, IrLogicRequiredGateResult
from ...proof.kernel_reconstruction import (
    KernelReconstructionDisposition,
    KernelReconstructionResult,
)
from ...proof.mcp_contract_obligations import McpGraphContractObligation, McpObligationDisposition
from .registry import OPERATOR_REGISTRY_SCHEMA, OperatorDescriptor, OperatorRegistry


PROTOCOL_REPAIR_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-repair-preview@1"
)
_PROFILES: Final[frozenset[str]] = frozenset(
    {"mcp-idl", "cid-envelope", "ucan-delegation", "policy-evaluation", "mcp-p2p", "event-dag"}
)


class ProtocolRepairStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PREVIEWED = "previewed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class ProtocolRepairError(ValueError):
    """A protocol operator input is malformed or unadmitted."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _closed_profiles(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolRepairError(field + " must be a sequence")
    result = tuple(sorted(set(value)))
    if any(not isinstance(item, str) or item not in _PROFILES for item in result):
        raise ProtocolRepairError(field + " has an unsupported profile")
    return result


@dataclass(frozen=True)
class JsonSchemaBinding:
    """Small closed object-schema binding, adequate for the protocol envelope."""

    schema: Mapping[str, Any]
    schema_cid: str

    def __post_init__(self) -> None:
        schema = dict(self.schema)
        if set(schema) != {"type", "required", "properties", "additional_properties"}:
            raise ProtocolRepairError("schema must use the closed object shape")
        required = schema.get("required")
        properties = schema.get("properties")
        if (
            schema.get("type") != "object"
            or schema.get("additional_properties") is not False
            or not isinstance(required, Sequence)
            or isinstance(required, (str, bytes))
            or not isinstance(properties, Mapping)
            or set(required) != set(properties)
            or not all(isinstance(item, str) and item for item in required)
            or not all(
                value in {"string", "integer", "object", "array", "boolean"}
                for value in properties.values()
            )
        ):
            raise ProtocolRepairError("schema is not an admitted closed object schema")
        canonical = {
            "type": "object",
            "required": sorted(required),
            "properties": dict(sorted(properties.items())),
            "additional_properties": False,
        }
        if self.schema_cid != content_identity(canonical):
            raise ProtocolRepairError("schema CID does not match canonical schema")
        object.__setattr__(self, "schema", canonical)

    def validates(self, value: Any) -> bool:
        if not isinstance(value, Mapping) or set(value) != set(self.schema["required"]):
            return False
        kinds = {
            "string": str,
            "integer": int,
            "object": Mapping,
            "array": Sequence,
            "boolean": bool,
        }
        for key, kind in self.schema["properties"].items():
            item = value.get(key)
            if kind == "integer" and isinstance(item, bool):
                return False
            if kind == "array" and isinstance(item, (str, bytes)):
                return False
            if not isinstance(item, kinds[kind]):
                return False
        return True

    @property
    def identity_valid(self) -> bool:
        try:
            JsonSchemaBinding(self.schema, self.schema_cid)
        except ProtocolRepairError:
            return False
        return True


@dataclass(frozen=True)
class ProtocolRepairRequest:
    source_bytes: bytes
    source_digest: str
    owner_root: str
    relative_path: str
    span_start: int
    span_end: int
    span_digest: str
    replacement_bytes: bytes
    descriptor: OperatorDescriptor
    registry: OperatorRegistry
    reviewed_registry_cid: str
    reviewed_descriptor_cid: str
    obligation: McpGraphContractObligation
    reconstruction: KernelReconstructionResult
    logic_gate: IrLogicRequiredGateResult
    http_status: int
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    request_schema: JsonSchemaBinding
    result_schema: JsonSchemaBinding
    requested_profiles: tuple[str, ...]
    supported_profiles: tuple[str, ...]
    policy_available: bool
    transport: str


@dataclass(frozen=True)
class ProtocolRepairPreview:
    status: ProtocolRepairStatus
    reason_codes: tuple[str, ...]
    request_cid: str = ""
    forward_cid: str = ""
    inverse_cid: str = ""
    before_digest: str = ""
    after_digest: str = ""
    after_bytes: bytes = b""

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": PROTOCOL_REPAIR_PREVIEW_SCHEMA,
            "authoritative": False,
            "execution_authorized": False,
            "completion_authorized": False,
            "activation_status": "swissknife_integration_pending",
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "request_cid": self.request_cid,
            "forward_cid": self.forward_cid,
            "inverse_cid": self.inverse_cid,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "after_base64": base64.b64encode(self.after_bytes).decode("ascii"),
            "model_call_count": 0,
        }
        return {**body, "preview_cid": content_identity(body)}


def _validate_authority(request: ProtocolRepairRequest) -> str | None:
    if (
        not isinstance(request.descriptor, OperatorDescriptor)
        or not isinstance(request.registry, OperatorRegistry)
        or request.reviewed_descriptor_cid != request.descriptor.descriptor_id
    ):
        return "reviewed_descriptor_cid_invalid"
    report = request.registry.report()
    if (
        report.get("schema") != OPERATOR_REGISTRY_SCHEMA
        or request.reviewed_registry_cid != report.get("registry_cid")
        or report.get("reviewed_manifest", {}).get(request.descriptor.operator_id)
        != request.descriptor.descriptor_id
        or request.descriptor not in request.registry.enumerate()
    ):
        return "reviewed_descriptor_manifest_invalid"
    if (
        request.owner_root != request.descriptor.owner_root
        or request.relative_path not in request.descriptor.write_scope
    ):
        return "source_owner_or_path_not_admitted"
    if request.source_digest != _sha256(request.source_bytes):
        return "source_digest_invalid"
    if not (0 <= request.span_start < request.span_end <= len(request.source_bytes)):
        return "source_span_invalid"
    if request.span_digest != _sha256(request.source_bytes[request.span_start : request.span_end]):
        return "source_span_digest_invalid"
    if (
        not request.replacement_bytes
        or request.replacement_bytes == request.source_bytes[request.span_start : request.span_end]
    ):
        return "replacement_not_exact_nonempty_inverse"
    obligation = request.obligation
    if (
        not isinstance(obligation, McpGraphContractObligation)
        or obligation.disposition is not McpObligationDisposition.OPEN
    ):
        return "dcr031_obligation_not_current"
    reconstruction = request.reconstruction
    if (
        not isinstance(reconstruction, KernelReconstructionResult)
        or reconstruction.disposition
        not in {
            KernelReconstructionDisposition.RECONSTRUCTED,
            KernelReconstructionDisposition.REFUTED,
        }
        or reconstruction.roots is None
        or reconstruction.roots.graph_cid != obligation.graph_cid
    ):
        return "dcr033_reconstruction_or_counterexample_not_current"
    gate = request.logic_gate
    if (
        not isinstance(gate, IrLogicRequiredGateResult)
        or gate.disposition is not IrLogicRequiredGateDisposition.PASSING
    ):
        return "dcr035_logic_gate_not_passing"
    identities = gate.required_identity_cids
    if (
        identities.get("dcr031") != content_identity(obligation.to_dict())
        or identities.get("dcr033") != reconstruction.to_dict()["result_cid"]
    ):
        return "dcr035_identity_binding_invalid"
    return None


def _validate_protocol(request: ProtocolRepairRequest) -> str | None:
    if type(request.http_status) is not int or request.http_status != 200:
        return "http_status_not_success"
    if request.transport != "http_jsonrpc":
        return "transport_mismatch"
    if type(request.policy_available) is not bool or not request.policy_available:
        return "policy_outage"
    try:
        requested = _closed_profiles(request.requested_profiles, "requested_profiles")
        supported = _closed_profiles(request.supported_profiles, "supported_profiles")
    except ProtocolRepairError:
        return "unsupported_profile"
    if not set(requested).issubset(supported):
        return "unsupported_profile"
    payload, response = request.request, request.response
    if not isinstance(request.request_schema, JsonSchemaBinding) or not isinstance(
        request.result_schema, JsonSchemaBinding
    ):
        return "request_or_result_schema_invalid"
    if not request.request_schema.identity_valid or not request.result_schema.identity_valid:
        return "request_or_result_schema_invalid"
    if not request.request_schema.validates(payload) or not isinstance(response, Mapping):
        return "request_or_result_schema_invalid"
    if payload.get("jsonrpc") != "2.0" or response.get("jsonrpc") != "2.0":
        return "jsonrpc_version_invalid"
    if (
        payload.get("id") is None
        or isinstance(payload.get("id"), bool)
        or response.get("id") != payload.get("id")
    ):
        return "jsonrpc_id_invalid"
    has_result, has_error = "result" in response, "error" in response
    if has_result == has_error:
        return "jsonrpc_result_error_exclusivity_invalid"
    if has_error:
        return "jsonrpc_error_response"
    if not request.result_schema.validates(response["result"]):
        return "request_or_result_schema_invalid"
    return None


def preview_protocol_repair(request: ProtocolRepairRequest) -> ProtocolRepairPreview:
    """Build a metadata/byte preview and exact inverse; never apply either."""

    if not isinstance(request, ProtocolRepairRequest):
        return ProtocolRepairPreview(ProtocolRepairStatus.REJECTED, ("typed_request_required",))
    try:
        reason = _validate_authority(request) or _validate_protocol(request)
        request_cid = content_identity(
            {
                "source_digest": request.source_digest,
                "owner_root": request.owner_root,
                "relative_path": request.relative_path,
                "span": [request.span_start, request.span_end, request.span_digest],
                "replacement_sha256": _sha256(request.replacement_bytes),
                "descriptor_cid": request.reviewed_descriptor_cid,
                "registry_cid": request.reviewed_registry_cid,
                "obligation_cid": content_identity(request.obligation.to_dict()),
                "reconstruction_cid": request.reconstruction.to_dict().get("result_cid", ""),
                "gate": request.logic_gate.to_dict(),
                "request_schema": request.request_schema.schema_cid,
                "result_schema": request.result_schema.schema_cid,
                "requested_profiles": list(request.requested_profiles),
                "supported_profiles": list(request.supported_profiles),
            }
        )
    except (AttributeError, TypeError, ValueError):
        return ProtocolRepairPreview(
            ProtocolRepairStatus.REJECTED, ("typed_request_binding_invalid",)
        )
    if reason:
        status = (
            ProtocolRepairStatus.ABSTAINED
            if reason in {"jsonrpc_error_response", "policy_outage"}
            else ProtocolRepairStatus.REJECTED
        )
        return ProtocolRepairPreview(status, (reason,), request_cid=request_cid)
    before = request.source_bytes
    after = before[: request.span_start] + request.replacement_bytes + before[request.span_end :]
    forward = content_identity(
        {"before": _sha256(before), "after": _sha256(after), "request_cid": request_cid}
    )
    inverse = content_identity(
        {"before": _sha256(after), "after": _sha256(before), "request_cid": request_cid}
    )
    return ProtocolRepairPreview(
        ProtocolRepairStatus.PREVIEWED,
        ("swissknife_integration_pending",),
        request_cid,
        forward,
        inverse,
        _sha256(before),
        _sha256(after),
        after,
    )


__all__ = [
    "JsonSchemaBinding",
    "PROTOCOL_REPAIR_PREVIEW_SCHEMA",
    "ProtocolRepairError",
    "ProtocolRepairPreview",
    "ProtocolRepairRequest",
    "ProtocolRepairStatus",
    "preview_protocol_repair",
]
