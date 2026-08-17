"""MCP++ Profile A/B/F wire codec for semantic-state harness records.

Identity is exclusively ``canonicalize_artifact`` plus ``cid_for_bytes``.
This module does not implement envelopes, CID codecs, or datasets identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import canonicalize_artifact
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HARNESS_CONTRACTS_SCHEMA,
    HarnessError,
    SemanticStateRootManifest,
    _closed,
    _text,
    _unique_sorted_cids,
    validate_opaque_cid,
)

INTERFACE_NAME = "semantic-state-harness"
INTERFACE_NAMESPACE = "ipfs-accelerate.agent-supervisor"
INTERFACE_VERSION = "1.0.0"
WIRE_BOUNDARY = "mcp-plus-plus-profiles-a-b-f"


def cid_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the Kubo-compatible CIDv1 of a canonical artifact payload."""

    if not isinstance(payload, Mapping):
        raise HarnessError("payload must be an object")
    return cid_for_bytes(canonicalize_artifact(dict(payload)))


def semantic_state_interface_descriptor() -> dict[str, Any]:
    """Return the closed Profile A interface descriptor."""

    return {
        "name": INTERFACE_NAME,
        "namespace": INTERFACE_NAMESPACE,
        "version": INTERFACE_VERSION,
        "methods": [
            {
                "name": "open_semantic_state",
                "input": "root_cid+get_block",
                "output": "SemanticStateView",
            },
            {
                "name": "encode_root_manifest",
                "input": "SemanticStateRootManifest",
                "output": "ExecutionEnvelope",
            },
            {
                "name": "decode_root_manifest",
                "input": "ExecutionEnvelope",
                "output": "SemanticStateRootManifest",
            },
        ],
        "errors": [
            {"name": "HarnessError", "retryable": False},
            {"name": "UnavailableResult", "retryable": True},
        ],
        "requires": [
            "mcp++/profile-a/interface-description",
            "mcp++/profile-b/cid-native-artifacts",
            "mcp++/profile-f/event-dag-ordering",
        ],
        "compatibility": {
            "board_namespace": BOARD_NAMESPACE,
            "harness_contracts": HARNESS_CONTRACTS_SCHEMA,
            "wire_boundary": WIRE_BOUNDARY,
        },
        "application_schema_cid": None,
    }


def _descriptor_for_cid() -> dict[str, Any]:
    descriptor = semantic_state_interface_descriptor()
    body = dict(descriptor)
    body.pop("application_schema_cid", None)
    return body


def interface_descriptor_cid() -> str:
    return cid_for_payload(_descriptor_for_cid())


@dataclass(frozen=True)
class SemanticStateWireCodec:
    """Encode and strictly decode Profile B envelopes and Profile F events."""

    def interface_descriptor(self) -> dict[str, Any]:
        descriptor = semantic_state_interface_descriptor()
        descriptor["application_schema_cid"] = interface_descriptor_cid()
        return descriptor

    def encode_execution_envelope(
        self, payload: Mapping[str, Any], *, parents: list[str] | None = None
    ) -> dict[str, Any]:
        payload_cid = cid_for_payload(payload)
        envelope = {
            "interface_cid": interface_descriptor_cid(),
            "input_cid": payload_cid,
            "payload_cid": payload_cid,
            "payload": dict(payload),
            "parents": list(parents or []),
        }
        return envelope

    def decode_execution_envelope(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        required = frozenset(
            {"interface_cid", "input_cid", "payload_cid", "payload", "parents"}
        )
        data = _closed(envelope, required, "ExecutionEnvelope")
        validate_opaque_cid(data["interface_cid"], "interface_cid")
        validate_opaque_cid(data["input_cid"], "input_cid")
        validate_opaque_cid(data["payload_cid"], "payload_cid")
        if data["interface_cid"] != interface_descriptor_cid():
            raise HarnessError("interface_cid does not match the sealed descriptor")
        if not isinstance(data["payload"], Mapping):
            raise HarnessError("payload must be an object")
        recomputed = cid_for_payload(data["payload"])
        if recomputed != data["payload_cid"] or recomputed != data["input_cid"]:
            raise HarnessError("payload CID does not match canonical bytes")
        _unique_sorted_cids(data["parents"], "parents")
        return dict(data["payload"])

    def encode_execution_receipt(self, result: Mapping[str, Any]) -> dict[str, Any]:
        output_cid = cid_for_payload(result)
        receipt = {
            "output_cid": output_cid,
            "result": dict(result),
        }
        receipt["receipt_cid"] = cid_for_payload(
            {"output_cid": output_cid, "result": dict(result)}
        )
        return receipt

    def decode_execution_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        data = _closed(
            receipt, frozenset({"output_cid", "receipt_cid", "result"}), "ExecutionReceipt"
        )
        validate_opaque_cid(data["output_cid"], "output_cid")
        validate_opaque_cid(data["receipt_cid"], "receipt_cid")
        if not isinstance(data["result"], Mapping):
            raise HarnessError("result must be an object")
        if cid_for_payload(data["result"]) != data["output_cid"]:
            raise HarnessError("receipt output_cid does not match result bytes")
        expected_receipt = cid_for_payload(
            {"output_cid": data["output_cid"], "result": dict(data["result"])}
        )
        if expected_receipt != data["receipt_cid"]:
            raise HarnessError("receipt_cid does not match content-addressed result")
        return dict(data["result"])

    def encode_dag_event(
        self,
        payload: Mapping[str, Any],
        *,
        parent_event_cids: list[str] | None = None,
        timestamp: str = "0",
    ) -> dict[str, Any]:
        payload_cid = cid_for_payload(payload)
        parents = list(_unique_sorted_cids(list(parent_event_cids or []), "parent_event_cids"))
        body = {
            "timestamp": _text(timestamp, "timestamp"),
            "parents": parents,
            "payload_cid": payload_cid,
        }
        event = dict(body)
        event["event_cid"] = cid_for_payload(body)
        return event

    def decode_dag_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        data = _closed(
            event,
            frozenset({"event_cid", "timestamp", "parents", "payload_cid"}),
            "DAGEvent",
        )
        validate_opaque_cid(data["event_cid"], "event_cid")
        validate_opaque_cid(data["payload_cid"], "payload_cid")
        _text(data["timestamp"], "timestamp")
        parents = _unique_sorted_cids(data["parents"], "parents")
        expected = cid_for_payload(
            {
                "timestamp": data["timestamp"],
                "parents": list(parents),
                "payload_cid": data["payload_cid"],
            }
        )
        if expected != data["event_cid"]:
            raise HarnessError("event_cid does not match parents and payload")
        return {
            "event_cid": data["event_cid"],
            "timestamp": data["timestamp"],
            "parents": list(parents),
            "payload_cid": data["payload_cid"],
        }

    def encode_root_manifest(self, manifest: SemanticStateRootManifest) -> dict[str, Any]:
        return self.encode_execution_envelope(manifest.to_dict())

    def decode_root_manifest(
        self, envelope: Mapping[str, Any]
    ) -> SemanticStateRootManifest:
        return SemanticStateRootManifest.from_dict(self.decode_execution_envelope(envelope))
