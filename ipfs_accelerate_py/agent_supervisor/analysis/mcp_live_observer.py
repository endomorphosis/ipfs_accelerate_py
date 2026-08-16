"""DCR-023 fail-closed, zero-LLM local MCP observation receipts.

This module never creates a network client.  Its transport is deliberately
injected, so production integration must provide a reviewed local transport
after the DCR-021 graph, DCR-022 service identity, and read-only request
template have all become current.  Until then it emits a typed defer receipt
without touching the transport.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .mcp_contract_graph import MCP_CONTRACT_GRAPH_INTERFACE, MCP_CONTRACT_GRAPH_SCHEMA
from .runtime_service_identity import RuntimeServiceIdentity, ServiceIdentityStatus

MCP_LIVE_OBSERVER_INTERFACE: Final[str] = "McpLiveObserver@1"
MCP_LIVE_TRANSCRIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcp-live-observation-transcript@1"
)
MCP_LIVE_OBSERVATION_EPOCH_INTERFACE: Final[str] = "McpLiveObservationEpoch@1"
MCP_LIVE_OBSERVATION_EPOCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcp-live-observation-epoch@1"
)
_SEMANTIC_KEY_FIELDS: Final[tuple[str, ...]] = (
    "package",
    "operation",
    "direction",
    "schema",
    "profile",
    "transport",
)


class ObservationStatus(StrEnum):
    OBSERVED = "observed"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    TRANSPORT_FAILURE = "transport_failure"
    REMOTE_FAILURE = "remote_failure"
    UNKNOWN_RESPONSE = "unknown_response"


class TemplateStatus(StrEnum):
    VALID = "valid"
    INTEGRATION_PENDING = "integration_pending"


class ObservationFailureCode(StrEnum):
    INTEGRATION_PENDING = "integration_pending"
    GRAPH_INVALID = "graph_invalid"
    GRAPH_BLOCKED = "graph_blocked"
    IDENTITY_INVALID = "identity_invalid"
    ENDPOINT_REJECTED = "endpoint_rejected"
    TEMPLATE_INVALID = "template_invalid"
    TRANSPORT_FAILURE = "transport_failure"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_RESPONSE = "malformed_response"
    REMOTE_ERROR = "remote_error"
    UNKNOWN_RESPONSE = "unknown_response"


class LocalMcpByteTransport(Protocol):
    """A reviewed local transport boundary; implementations receive bytes only."""

    def exchange(self, *, endpoint: str, request: bytes) -> bytes: ...


@dataclass(frozen=True)
class McpObservationTemplate:
    """A reviewed literal request.  No user payload or dynamic substitution exists."""

    operation: str
    request_bytes: bytes
    status: TemplateStatus = TemplateStatus.INTEGRATION_PENDING
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("observation template operation is required")
        if not isinstance(self.request_bytes, bytes) or not self.request_bytes:
            raise ValueError("observation template must contain non-empty raw request bytes")

    @property
    def template_cid(self) -> str:
        return content_identity(
            {
                "operation": self.operation,
                "request_base64": base64.b64encode(self.request_bytes).decode("ascii"),
                "read_only": self.read_only,
            }
        )


@dataclass(frozen=True)
class McpObservationTranscript:
    """Non-authoritative raw-byte observation; it cannot establish completion."""

    status: ObservationStatus
    failure: ObservationFailureCode | None
    service_role: str
    transport: str
    operation: str
    endpoint: str
    request_bytes: bytes
    response_bytes: bytes
    graph_cid: str
    runtime_receipt_id: str
    process_witness_cid: str
    template_cid: str

    @property
    def request_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.request_bytes).hexdigest()

    @property
    def response_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.response_bytes).hexdigest()

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_receipt=False))

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": MCP_LIVE_TRANSCRIPT_SCHEMA,
            "interface": MCP_LIVE_OBSERVER_INTERFACE,
            "authoritative": False,
            "completion_authoritative": False,
            "status": self.status.value,
            "failure": self.failure.value if self.failure else None,
            "service_role": self.service_role,
            "transport": self.transport,
            "operation": self.operation,
            "endpoint": self.endpoint,
            "request_base64": base64.b64encode(self.request_bytes).decode("ascii"),
            "response_base64": base64.b64encode(self.response_bytes).decode("ascii"),
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "graph_cid": self.graph_cid,
            "runtime_receipt_id": self.runtime_receipt_id,
            "process_witness_cid": self.process_witness_cid,
            "template_cid": self.template_cid,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class RequiredMcpObservation:
    """One exact service role and operation that a current epoch must observe."""

    service_role: str
    edge_id: str
    package: str
    operation: str
    direction: str
    schema: str
    profile: str
    transport: str

    def __post_init__(self) -> None:
        if not self.service_role.strip() or not self.edge_id.strip():
            raise ValueError("required observation needs a service role and graph edge")
        if not all(getattr(self, field).strip() for field in _SEMANTIC_KEY_FIELDS):
            raise ValueError("required observation semantic key is incomplete")

    @property
    def semantic_key(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _SEMANTIC_KEY_FIELDS}

    @property
    def requirement_id(self) -> str:
        return content_identity(
            {
                "service_role": self.service_role,
                "edge_id": self.edge_id,
                "semantic_key": self.semantic_key,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "service_role": self.service_role,
            "edge_id": self.edge_id,
            "semantic_key": self.semantic_key,
        }


@dataclass(frozen=True)
class McpObservationEpoch:
    """Typed DCR-023 aggregate consumed directly by DCR-024.

    It remains non-authoritative: a valid epoch means only that its exact,
    reviewed read-only observations are current for the supplied roots.
    """

    graph_cid: str
    semantic_roots: Mapping[str, str]
    snapshot_roots: Mapping[str, str]
    required_observations: tuple[RequiredMcpObservation, ...]
    receipts: tuple[McpObservationTranscript, ...]
    checks: tuple[Mapping[str, Any], ...]
    valid: bool

    @property
    def epoch_cid(self) -> str:
        return content_identity(self.to_dict(include_cid=False))

    def to_dict(self, *, include_cid: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": MCP_LIVE_OBSERVATION_EPOCH_SCHEMA,
            "interface": MCP_LIVE_OBSERVATION_EPOCH_INTERFACE,
            "authoritative": False,
            "completion_authoritative": False,
            "valid": self.valid,
            "graph_cid": self.graph_cid,
            "semantic_roots": dict(sorted(self.semantic_roots.items())),
            "snapshot_roots": dict(sorted(self.snapshot_roots.items())),
            "required_observations": [
                item.to_dict()
                for item in sorted(self.required_observations, key=lambda item: item.requirement_id)
            ],
            "receipts": [
                item.to_dict() for item in sorted(self.receipts, key=lambda item: item.receipt_id)
            ],
            "checks": [dict(item) for item in self.checks],
        }
        if include_cid:
            payload["epoch_cid"] = self.epoch_cid
        return payload


def _roots(value: Mapping[str, str], field: str) -> dict[str, str]:
    normalized = {str(key).strip(): str(item).strip() for key, item in value.items()}
    if not normalized or any(not key or not item for key, item in normalized.items()):
        raise ValueError(f"{field} must have non-empty exact roots")
    return dict(sorted(normalized.items()))


def _epoch_check(
    *,
    requirement: RequiredMcpObservation | None,
    status: str,
    receipt: McpObservationTranscript | None = None,
    edge_id: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mismatch_class": "liveness",
        "status": status,
        "edge_id": requirement.edge_id if requirement else edge_id,
    }
    if requirement:
        result["service_role"] = requirement.service_role
        result["semantic_key"] = requirement.semantic_key
        result["requirement_id"] = requirement.requirement_id
    if receipt:
        result["receipt_id"] = receipt.receipt_id
        result["request_digest"] = receipt.request_digest
        result["response_digest"] = receipt.response_digest
    return result


def build_mcp_observation_epoch(
    *,
    graph_cid: str,
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
    required_observations: Sequence[RequiredMcpObservation],
    receipts: Sequence[McpObservationTranscript],
) -> McpObservationEpoch:
    """Bind exact required roles/operations to individual DCR-023 receipts."""
    if not isinstance(graph_cid, str) or not graph_cid.strip():
        raise ValueError("observation epoch requires graph_cid")
    semantic = _roots(semantic_roots, "semantic_roots")
    snapshot = _roots(snapshot_roots, "snapshot_roots")
    required = tuple(sorted(required_observations, key=lambda item: item.requirement_id))
    if not required or len({item.requirement_id for item in required}) != len(required):
        raise ValueError("required observations must be non-empty and exact")
    observed = tuple(sorted(receipts, key=lambda item: item.receipt_id))
    checks: list[dict[str, Any]] = []
    matched_receipt_ids: set[str] = set()
    for requirement in required:
        matches = [
            receipt
            for receipt in observed
            if receipt.operation == requirement.operation
            and receipt.service_role == requirement.service_role
            and receipt.transport == requirement.transport
            and receipt.endpoint
            and receipt.graph_cid == graph_cid
            and receipt.runtime_receipt_id
            and receipt.process_witness_cid
            and receipt.status is ObservationStatus.OBSERVED
            and requirement.transport == "mcp"
        ]
        if len(matches) == 1:
            matched_receipt_ids.add(matches[0].receipt_id)
            checks.append(
                _epoch_check(requirement=requirement, status="passed", receipt=matches[0])
            )
        elif not matches:
            checks.append(_epoch_check(requirement=requirement, status="missing"))
        else:
            checks.append(_epoch_check(requirement=requirement, status="ambiguous"))
    for receipt in observed:
        if receipt.receipt_id not in matched_receipt_ids:
            checks.append(
                _epoch_check(
                    requirement=None,
                    status="failed",
                    receipt=receipt,
                    edge_id="dcr023:unexpected-receipt:" + receipt.receipt_id,
                )
            )
    checks.sort(key=canonical_json_bytes)
    return McpObservationEpoch(
        graph_cid=graph_cid,
        semantic_roots=semantic,
        snapshot_roots=snapshot,
        required_observations=required,
        receipts=observed,
        checks=tuple(checks),
        valid=bool(checks) and all(item["status"] == "passed" for item in checks),
    )


def is_current_mcp_observation_epoch(
    value: McpObservationEpoch | Mapping[str, Any] | None,
    *,
    graph_cid: str,
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
) -> bool:
    """Verify an epoch's canonical binding before DCR-024 consumes checks."""
    payload = value.to_dict() if isinstance(value, McpObservationEpoch) else value
    if not isinstance(payload, Mapping):
        return False
    epoch_cid = payload.get("epoch_cid")
    body = {key: item for key, item in payload.items() if key != "epoch_cid"}
    if (
        payload.get("schema") != MCP_LIVE_OBSERVATION_EPOCH_SCHEMA
        or payload.get("interface") != MCP_LIVE_OBSERVATION_EPOCH_INTERFACE
        or payload.get("authoritative") is not False
        or payload.get("completion_authoritative") is not False
        or not isinstance(epoch_cid, str)
        or epoch_cid != content_identity(body)
        or payload.get("graph_cid") != graph_cid
        or payload.get("semantic_roots") != _roots(semantic_roots, "semantic_roots")
        or payload.get("snapshot_roots") != _roots(snapshot_roots, "snapshot_roots")
        or payload.get("valid") is not True
    ):
        return False
    checks = payload.get("checks")
    required = payload.get("required_observations")
    receipts = payload.get("receipts")
    if not all(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in (checks, required, receipts)
    ):
        return False
    if not bool(required) or not bool(checks) or not bool(receipts):
        return False
    if not all(
        isinstance(item, Mapping)
        and isinstance(item.get("service_role"), str)
        and isinstance(item.get("edge_id"), str)
        and isinstance(item.get("semantic_key"), Mapping)
        and all(
            isinstance(item["semantic_key"].get(field), str) and item["semantic_key"][field]
            for field in _SEMANTIC_KEY_FIELDS
        )
        for item in required
    ):
        return False
    requirements_by_id: dict[str, Mapping[str, Any]] = {}
    for requirement in required:
        requirement_body = {
            "service_role": requirement["service_role"],
            "edge_id": requirement["edge_id"],
            "semantic_key": dict(requirement["semantic_key"]),
        }
        requirement_id = requirement.get("requirement_id")
        if (
            not isinstance(requirement_id, str)
            or requirement_id != content_identity(requirement_body)
            or requirement_id in requirements_by_id
        ):
            return False
        requirements_by_id[requirement_id] = requirement
    receipts_by_id: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            return False
        receipt_body = {key: item for key, item in receipt.items() if key != "receipt_id"}
        if not isinstance(receipt.get("receipt_id"), str) or receipt[
            "receipt_id"
        ] != content_identity(receipt_body):
            return False
        try:
            request = base64.b64decode(receipt.get("request_base64", ""), validate=True)
            response = base64.b64decode(receipt.get("response_base64", ""), validate=True)
        except (TypeError, ValueError):
            return False
        if (
            receipt.get("schema") != MCP_LIVE_TRANSCRIPT_SCHEMA
            or receipt.get("interface") != MCP_LIVE_OBSERVER_INTERFACE
            or receipt.get("authoritative") is not False
            or receipt.get("completion_authoritative") is not False
            or receipt.get("graph_cid") != graph_cid
            or receipt.get("status") != ObservationStatus.OBSERVED.value
            or receipt.get("failure") is not None
            or not isinstance(receipt.get("service_role"), str)
            or not receipt["service_role"]
            or receipt.get("transport") != "mcp"
            or not isinstance(receipt.get("operation"), str)
            or not receipt["operation"]
            or not _loopback(str(receipt.get("endpoint") or ""))
            or not isinstance(receipt.get("runtime_receipt_id"), str)
            or not receipt["runtime_receipt_id"]
            or not isinstance(receipt.get("process_witness_cid"), str)
            or not receipt["process_witness_cid"]
            or receipt.get("request_digest") != "sha256:" + hashlib.sha256(request).hexdigest()
            or receipt.get("response_digest") != "sha256:" + hashlib.sha256(response).hexdigest()
        ):
            return False
        receipt_id = str(receipt["receipt_id"])
        if receipt_id in receipts_by_id:
            return False
        receipts_by_id[receipt_id] = receipt
    closed_statuses = {"passed", "expected_only", "missing", "ambiguous", "unobserved", "failed"}
    if len(checks) != len(requirements_by_id):
        return False
    checked_requirements: set[str] = set()
    checked_receipts: set[str] = set()
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") not in closed_statuses:
            return False
        # A current valid epoch is an exact passing observation set.  Any
        # missing/ambiguous/failed check is preserved in a non-current epoch.
        if check.get("status") != "passed":
            return False
        requirement_id = check.get("requirement_id")
        if not isinstance(requirement_id, str) or requirement_id in checked_requirements:
            return False
        requirement = requirements_by_id.get(requirement_id)
        if requirement is None:
            return False
        receipt_id = check.get("receipt_id")
        if not isinstance(receipt_id, str) or receipt_id in checked_receipts:
            return False
        receipt = receipts_by_id.get(receipt_id)
        if receipt is None:
            return False
        semantic_key = requirement["semantic_key"]
        if (
            check.get("mismatch_class") != "liveness"
            or check.get("edge_id") != requirement["edge_id"]
            or check.get("service_role") != requirement["service_role"]
            or check.get("semantic_key") != semantic_key
            or check.get("request_digest") != receipt.get("request_digest")
            or check.get("response_digest") != receipt.get("response_digest")
            or receipt.get("service_role") != requirement["service_role"]
            or receipt.get("operation") != semantic_key["operation"]
            or receipt.get("transport") != semantic_key["transport"]
        ):
            return False
        checked_requirements.add(requirement_id)
        checked_receipts.add(receipt_id)
    return (
        set(requirements_by_id) == checked_requirements and set(receipts_by_id) == checked_receipts
    )


def _loopback(endpoint: str) -> bool:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed.username is None
        and parsed.password is None
        and port is not None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _graph_body(graph: Mapping[str, Any]) -> Mapping[str, Any] | None:
    canonical = graph.get("canonical_bytes")
    graph_cid = graph.get("graph_cid")
    if not isinstance(canonical, str) or not isinstance(graph_cid, str):
        return None
    try:
        decoded = json.loads(canonical)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    if canonical_json_bytes(decoded).decode("utf-8") != canonical:
        return None
    if content_identity(decoded) != graph_cid:
        return None
    if (
        decoded.get("schema") != MCP_CONTRACT_GRAPH_SCHEMA
        or decoded.get("interface") != MCP_CONTRACT_GRAPH_INTERFACE
        or decoded.get("authoritative") is not False
    ):
        return None
    if not isinstance(decoded.get("nodes"), Sequence) or isinstance(decoded["nodes"], (str, bytes)):
        return None
    if not isinstance(decoded.get("blockers"), Sequence) or isinstance(
        decoded["blockers"], (str, bytes)
    ):
        return None
    return decoded


def _template_is_literal_read_only(template: McpObservationTemplate) -> bool:
    if not template.read_only:
        return False
    try:
        request = json.loads(template.request_bytes)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(request, Mapping) or request.get("method") != template.operation:
        return False
    # Dynamic parameters are user data.  A literal empty object is the sole
    # supported parameter shape until a reviewed operation-specific schema is
    # integrated.
    return "params" not in request or request.get("params") == {}


def _operation_is_in_graph(graph: Mapping[str, Any], operation: str) -> bool:
    return any(
        isinstance(node, Mapping)
        and node.get("kind") == "expected_descriptor"
        and node.get("label") == operation
        for node in graph["nodes"]
    )


def _transcript(
    *,
    status: ObservationStatus,
    failure: ObservationFailureCode | None,
    template: McpObservationTemplate,
    identity: RuntimeServiceIdentity | None,
    graph_cid: str = "",
    response: bytes = b"",
) -> McpObservationTranscript:
    observation = identity.observation if identity else None
    return McpObservationTranscript(
        status=status,
        failure=failure,
        service_role=observation.role if observation else "",
        transport=observation.transport if observation else "",
        operation=template.operation,
        endpoint=observation.endpoint if observation else "",
        request_bytes=template.request_bytes,
        response_bytes=response,
        graph_cid=graph_cid,
        runtime_receipt_id=identity.receipt_id if identity else "",
        process_witness_cid=identity.process_witness_cid if identity else "",
        template_cid=template.template_cid,
    )


def observe_mcp_template(
    *,
    graph: Mapping[str, Any],
    runtime_identity: RuntimeServiceIdentity | None,
    template: McpObservationTemplate,
    transport: LocalMcpByteTransport,
) -> McpObservationTranscript:
    """Observe one approved byte template without filesystem, model, or network code.

    The injected transport is not called until every static and runtime gate
    succeeds.  In particular, an empty response remains an explicit typed
    failure rather than a successful observation.
    """
    if template.status is not TemplateStatus.VALID:
        return _transcript(
            status=(
                ObservationStatus.DEFERRED
                if template.status is TemplateStatus.INTEGRATION_PENDING
                else ObservationStatus.REJECTED
            ),
            failure=(
                ObservationFailureCode.INTEGRATION_PENDING
                if template.status is TemplateStatus.INTEGRATION_PENDING
                else ObservationFailureCode.TEMPLATE_INVALID
            ),
            template=template,
            identity=runtime_identity,
        )
    if (
        runtime_identity is None
        or runtime_identity.status is ServiceIdentityStatus.INTEGRATION_PENDING
    ):
        return _transcript(
            status=ObservationStatus.DEFERRED,
            failure=ObservationFailureCode.INTEGRATION_PENDING,
            template=template,
            identity=runtime_identity,
        )
    if runtime_identity.status is not ServiceIdentityStatus.VALID:
        return _transcript(
            status=ObservationStatus.REJECTED,
            failure=ObservationFailureCode.IDENTITY_INVALID,
            template=template,
            identity=runtime_identity,
        )
    if runtime_identity.observation.transport != "mcp":
        return _transcript(
            status=ObservationStatus.REJECTED,
            failure=ObservationFailureCode.IDENTITY_INVALID,
            template=template,
            identity=runtime_identity,
        )
    if not _loopback(runtime_identity.observation.endpoint):
        return _transcript(
            status=ObservationStatus.REJECTED,
            failure=ObservationFailureCode.ENDPOINT_REJECTED,
            template=template,
            identity=runtime_identity,
        )
    body = _graph_body(graph)
    if body is None:
        return _transcript(
            status=ObservationStatus.REJECTED,
            failure=ObservationFailureCode.GRAPH_INVALID,
            template=template,
            identity=runtime_identity,
        )
    graph_cid = str(graph["graph_cid"])
    if body["blockers"]:
        return _transcript(
            status=ObservationStatus.DEFERRED,
            failure=ObservationFailureCode.GRAPH_BLOCKED,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
        )
    if not _operation_is_in_graph(body, template.operation) or not _template_is_literal_read_only(
        template
    ):
        return _transcript(
            status=ObservationStatus.REJECTED,
            failure=ObservationFailureCode.TEMPLATE_INVALID,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
        )
    try:
        response = transport.exchange(
            endpoint=runtime_identity.observation.endpoint,
            request=template.request_bytes,
        )
    except Exception:  # Transport implementations are untrusted integration code.
        return _transcript(
            status=ObservationStatus.TRANSPORT_FAILURE,
            failure=ObservationFailureCode.TRANSPORT_FAILURE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
        )
    if not isinstance(response, bytes):
        return _transcript(
            status=ObservationStatus.TRANSPORT_FAILURE,
            failure=ObservationFailureCode.TRANSPORT_FAILURE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
        )
    if not response:
        return _transcript(
            status=ObservationStatus.UNKNOWN_RESPONSE,
            failure=ObservationFailureCode.EMPTY_RESPONSE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
            response=response,
        )
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return _transcript(
            status=ObservationStatus.UNKNOWN_RESPONSE,
            failure=ObservationFailureCode.MALFORMED_RESPONSE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
            response=response,
        )
    if not isinstance(payload, Mapping):
        return _transcript(
            status=ObservationStatus.UNKNOWN_RESPONSE,
            failure=ObservationFailureCode.UNKNOWN_RESPONSE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
            response=response,
        )
    if "error" in payload:
        return _transcript(
            status=ObservationStatus.REMOTE_FAILURE,
            failure=ObservationFailureCode.REMOTE_ERROR,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
            response=response,
        )
    if "result" not in payload:
        return _transcript(
            status=ObservationStatus.UNKNOWN_RESPONSE,
            failure=ObservationFailureCode.UNKNOWN_RESPONSE,
            template=template,
            identity=runtime_identity,
            graph_cid=graph_cid,
            response=response,
        )
    return _transcript(
        status=ObservationStatus.OBSERVED,
        failure=None,
        template=template,
        identity=runtime_identity,
        graph_cid=graph_cid,
        response=response,
    )


__all__ = [
    "LocalMcpByteTransport",
    "MCP_LIVE_OBSERVATION_EPOCH_INTERFACE",
    "MCP_LIVE_OBSERVATION_EPOCH_SCHEMA",
    "MCP_LIVE_OBSERVER_INTERFACE",
    "MCP_LIVE_TRANSCRIPT_SCHEMA",
    "McpObservationEpoch",
    "McpObservationTemplate",
    "McpObservationTranscript",
    "ObservationFailureCode",
    "ObservationStatus",
    "RequiredMcpObservation",
    "TemplateStatus",
    "build_mcp_observation_epoch",
    "is_current_mcp_observation_epoch",
    "observe_mcp_template",
]
