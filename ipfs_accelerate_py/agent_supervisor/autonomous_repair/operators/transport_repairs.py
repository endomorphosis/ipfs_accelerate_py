"""DCR-044 static transport repair previews; never target or run SwissKnife.

This is an operator-library boundary only.  It accepts current reviewed
metadata and source bytes supplied by a caller, computes reversible previews,
and always leaves activation pending outside this module.
"""

from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final
from urllib.parse import urlsplit

from ...analysis.mcp_contract_graph import (
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
)
from ...analysis.mcp_live_observer import (
    McpObservationEpoch,
    is_current_mcp_observation_epoch,
)
from ...proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .registry import OperatorDescriptor, OperatorRegistry

TRANSPORT_REPAIR_PREVIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/transport-repair-preview@1"
)
TRANSPORT_REPAIR_ACTIVATION: Final = "integration_pending_dcr035_dcr040_dcr070_dcr072"
_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "normalize_endpoint_profile",
        "gate_lifecycle_readiness",
        "route_browser_mediation",
    }
)
_PROFILES: Final[frozenset[str]] = frozenset({"mcp-http-jsonrpc", "browser-mediated"})


class TransportRepairStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PREVIEWED = "previewed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class TransportRepairError(ValueError):
    """A closed static transport preview input was not admissible."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _line_starts(value: bytes) -> list[int]:
    return [0, *(index + 1 for index, item in enumerate(value) if item == 10)]


def _span(value: bytes, node: ast.AST) -> tuple[int, int]:
    if not all(
        hasattr(node, field)
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        raise TransportRepairError("anchor has no exact AST source span")
    starts = _line_starts(value)
    try:
        return (
            starts[node.lineno - 1] + node.col_offset,  # type: ignore[attr-defined]
            starts[node.end_lineno - 1] + node.end_col_offset,  # type: ignore[attr-defined]
        )
    except (AttributeError, IndexError) as exc:
        raise TransportRepairError("anchor span is outside source bytes") from exc


def transport_ast_span_identity(source: bytes, node: ast.AST) -> dict[str, Any]:
    start, end = _span(source, node)
    return {
        "node_type": type(node).__name__,
        "start": start,
        "end": end,
        "sha256": _sha256(source[start:end]),
    }


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _literal(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(item in value for item in "\r\n\x00"):
        raise TransportRepairError(f"{field} must be non-empty literal text")
    return value.strip()


def _loopback_endpoint(value: Any) -> str:
    endpoint = _literal(value, "endpoint")
    if any(item in endpoint for item in ("@", "?", "#", "\\")):
        raise TransportRepairError("endpoint contains forbidden userinfo/query/fragment syntax")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise TransportRepairError("endpoint must be an http loopback URL without userinfo")
    if parsed.query or parsed.fragment or parsed.path not in {"/mcp", "/jsonrpc"}:
        raise TransportRepairError("endpoint path is not a closed MCP loopback path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise TransportRepairError("endpoint port is invalid") from exc
    if port is None or not 1 <= port <= 65535:
        raise TransportRepairError("endpoint must declare one bounded loopback port")
    host = parsed.hostname.lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise TransportRepairError("endpoint host is not loopback")
        except ValueError as exc:
            raise TransportRepairError("endpoint host is not a closed loopback literal") from exc
    return endpoint


def _profile(value: Any) -> str:
    profile = _literal(value, "profile")
    if profile not in _PROFILES:
        raise TransportRepairError("profile is not reviewed")
    return profile


def _closed_postcondition(value: Any) -> dict[str, Any]:
    expected = {
        "mediator": "governed_mediator",
        "requires_mediator": True,
        "lifecycle_errors_fail": True,
        "raw_proxy_exposed": False,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise TransportRepairError("behavioral postcondition must prove governed mediation")
    return expected


def _graph_edge(graph: Any, edge_id: str) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(graph, Mapping):
        raise TransportRepairError("DCR-021 graph must be a mapping")
    body = dict(graph)
    graph_cid = body.pop("graph_cid", "")
    encoded = body.pop("canonical_bytes", "")
    if (
        body.get("schema") != MCP_CONTRACT_GRAPH_SCHEMA
        or body.get("interface") != MCP_CONTRACT_GRAPH_INTERFACE
        or body.get("authoritative") is not False
        or graph_cid != content_identity(body)
        or encoded != canonical_json_bytes(body).decode("utf-8")
    ):
        raise TransportRepairError("DCR-021 graph identity is stale or invalid")
    edges = body.get("edges")
    if not isinstance(edges, list):
        raise TransportRepairError("DCR-021 graph edges are invalid")
    matches = [item for item in edges if isinstance(item, Mapping) and item.get("id") == edge_id]
    if len(matches) != 1 or matches[0].get("relation") != "binds_mediator_route":
        raise TransportRepairError("current DCR-021 mediator edge is not uniquely resolved")
    return str(graph_cid), matches[0]


@dataclass(frozen=True)
class TransportRepairPreview:
    status: TransportRepairStatus
    reason_codes: tuple[str, ...]
    request_cid: str = ""
    forward_cid: str = ""
    inverse_cid: str = ""
    before_digest: str = ""
    after_digest: str = ""
    after_bytes: bytes = b""
    behavioral_postcondition_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": TRANSPORT_REPAIR_PREVIEW_SCHEMA,
            "authoritative": False,
            "execution_authorized": False,
            "completion_authorized": False,
            "activation_status": TRANSPORT_REPAIR_ACTIVATION,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "request_cid": self.request_cid,
            "forward_cid": self.forward_cid,
            "inverse_cid": self.inverse_cid,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "after_base64": base64.b64encode(self.after_bytes).decode("ascii"),
            "behavioral_postcondition_cid": self.behavioral_postcondition_cid,
            "model_call_count": 0,
            "provider_call_count": 0,
        }
        return {**body, "preview_cid": content_identity(body)}


def _rejected(reason: str, request_cid: str = "") -> TransportRepairPreview:
    return TransportRepairPreview(TransportRepairStatus.REJECTED, (reason,), request_cid)


def build_transport_repair_preview(
    request: Mapping[str, Any], *, registry: OperatorRegistry
) -> TransportRepairPreview:
    """Return a reversible static preview only; this function never writes or runs."""

    try:
        required = {
            "action", "operator_id", "descriptor_id", "manifest_cid", "owner_root",
            "relative_path", "source_bytes", "source_digest", "anchor", "target_api",
            "endpoint", "profile", "graph", "graph_edge_id", "semantic_roots",
            "snapshot_roots", "observation_epoch", "observation_epoch_cid",
            "behavioral_postcondition",
        }
        if not isinstance(request, Mapping) or set(request) != required:
            raise TransportRepairError("transport preview request fields are closed")
        source = request["source_bytes"]
        if not isinstance(source, bytes) or not source:
            raise TransportRepairError("exact non-empty source bytes are required")
        if request["source_digest"] != _sha256(source):
            raise TransportRepairError("source bytes or digest are stale")
        action = _literal(request["action"], "action")
        if action not in _ACTIONS:
            raise TransportRepairError("transport action is not closed")
        endpoint, profile = _loopback_endpoint(request["endpoint"]), _profile(request["profile"])
        postcondition = _closed_postcondition(request["behavioral_postcondition"])
        registry_report = registry.report()
        if request["manifest_cid"] != registry_report["registry_cid"]:
            raise TransportRepairError("reviewed DCR-040 manifest is stale")
        descriptors = {item.operator_id: item for item in registry.enumerate()}
        descriptor = descriptors.get(_literal(request["operator_id"], "operator_id"))
        if (
            not isinstance(descriptor, OperatorDescriptor)
            or request["descriptor_id"] != descriptor.descriptor_id
            or request["owner_root"] != descriptor.owner_root
            or request["relative_path"] not in descriptor.write_scope
        ):
            raise TransportRepairError("reviewed DCR-040 descriptor or write scope is invalid")
        graph_cid, _edge = _graph_edge(
            request["graph"], _literal(request["graph_edge_id"], "graph_edge_id")
        )
        epoch = request["observation_epoch"]
        if (
            not isinstance(epoch, McpObservationEpoch)
            or request["observation_epoch_cid"] != epoch.epoch_cid
            or not isinstance(request["semantic_roots"], Mapping)
            or not isinstance(request["snapshot_roots"], Mapping)
            or not is_current_mcp_observation_epoch(
                epoch,
                graph_cid=graph_cid,
                semantic_roots=request["semantic_roots"],
                snapshot_roots=request["snapshot_roots"],
            )
        ):
            raise TransportRepairError("DCR-023 observation epoch is stale or invalid")
        target_api = _literal(request["target_api"], "target_api")
        tree = ast.parse(source, filename=str(request["relative_path"]))
        anchors = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _dotted_name(node.func) == target_api
        ]
        if len(anchors) != 1:
            return _rejected("dynamic_or_multiple_transport_anchors")
        anchor = anchors[0]
        if dict(request["anchor"]) != transport_ast_span_identity(source, anchor):
            raise TransportRepairError("AST anchor is stale or not exact")
        kwargs = {item.arg: item.value for item in anchor.keywords if item.arg}
        if set(kwargs) != {"endpoint", "profile", "mediator", "lifecycle"} or any(
            not isinstance(item, ast.Constant) or not isinstance(item.value, str)
            for item in kwargs.values()
        ):
            return _rejected("dynamic_or_raw_proxy_transport_shape")
        if (
            kwargs["mediator"].value != "governed_mediator"
            or kwargs["lifecycle"].value != "strict_failure"
        ):
            return _rejected("absent_mediation_or_lifecycle_error_gate")
        values = {key: item.value for key, item in kwargs.items()}
        if action == "normalize_endpoint_profile":
            values.update(endpoint=endpoint, profile=profile)
        elif action == "gate_lifecycle_readiness":
            values["lifecycle"] = "strict_failure"
        else:
            values["mediator"] = "governed_mediator"
        replacement = (
            f"{target_api}(endpoint={values['endpoint']!r}, profile={values['profile']!r}, "
            f"mediator={values['mediator']!r}, lifecycle={values['lifecycle']!r})"
        ).encode()
        start, end = _span(source, anchor)
        after = source[:start] + replacement + source[end:]
        if after == source:
            return _rejected("preview_requires_effectful_normalization")
        before_lines = source.decode("utf-8").splitlines(keepends=True)
        after_lines = after.decode("utf-8").splitlines(keepends=True)
        forward = "".join(
            difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after")
        )
        inverse = "".join(
            difflib.unified_diff(after_lines, before_lines, fromfile="after", tofile="before")
        )
        request_body = {
            key: value
            for key, value in request.items()
            if key not in {"source_bytes", "graph", "observation_epoch"}
        }
        request_cid = content_identity(
            {
                **request_body,
                "source_base64": base64.b64encode(source).decode("ascii"),
                "graph_cid": graph_cid,
                "observation_epoch_cid": epoch.epoch_cid,
            }
        )
        return TransportRepairPreview(
            TransportRepairStatus.PREVIEWED,
            ("activation_pending",),
            request_cid=request_cid,
            forward_cid=content_identity({"forward": forward}),
            inverse_cid=content_identity({"inverse": inverse}),
            before_digest=_sha256(source),
            after_digest=_sha256(after),
            after_bytes=after,
            behavioral_postcondition_cid=content_identity(postcondition),
        )
    except (TransportRepairError, SyntaxError, TypeError, ValueError) as exc:
        return _rejected(str(exc))


__all__ = [
    "TRANSPORT_REPAIR_ACTIVATION",
    "TRANSPORT_REPAIR_PREVIEW_SCHEMA",
    "TransportRepairError",
    "TransportRepairPreview",
    "TransportRepairStatus",
    "build_transport_repair_preview",
    "transport_ast_span_identity",
]
