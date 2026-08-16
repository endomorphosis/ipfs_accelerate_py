"""DCR-045 non-writing UI/ORB/IDL/mobile projection repair previews.

Projection evidence is consumer context, never server truth.  This module only
constructs exact byte previews/inverses for reviewed callers; no write,
execution, network, provider, or model route exists here.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ...analysis.deterministic_desktop_expectations import (
    DESKTOP_EXPECTATIONS_INTERFACE,
    DESKTOP_EXPECTATIONS_SCHEMA,
)
from ...analysis.deterministic_repair_forest import DCR_FOREST_PORTABLE_SCHEMA, DCR_FOREST_SCHEMA
from ...analysis.mcp_contract_graph import MCP_CONTRACT_GRAPH_INTERFACE, MCP_CONTRACT_GRAPH_SCHEMA
from ...proof.formal_verification_contracts import canonical_json_bytes, content_identity
from ...proof.ir_integration import DatasetsLogicIrDisposition, DatasetsLogicIrResult
from .registry import OPERATOR_REGISTRY_SCHEMA, OperatorDescriptor, OperatorRegistry


UI_PROJECTION_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ui-projection-repair-preview@1"
)
_SURFACES: Final[frozenset[str]] = frozenset({"ui", "orb_idl", "mobile_descriptor"})
_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "operation",
    "request_schema",
    "result_schema",
    "effect",
    "security",
    "transport",
)


class UiProjectionPreviewStatus(str, Enum):  # noqa: UP042 - Python 3.8
    PREVIEWED = "previewed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class UiProjectionRepairError(ValueError):
    """A UI projection request is malformed or lacks current authority context."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _projection(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
        raise UiProjectionRepairError(field + " must carry the exact projection identity")
    normalized = {key: value[key] for key in _IDENTITY_FIELDS}
    if any(not isinstance(item, str) or not item for item in normalized.values()):
        raise UiProjectionRepairError(field + " has an empty or dynamic identity field")
    return normalized


def _forest_current(value: Any) -> str:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != DCR_FOREST_SCHEMA
        or value.get("authoritative") is not True
    ):
        raise UiProjectionRepairError("current typed DCR-011 forest required")
    portable = value.get("portable")
    identity = value.get("portable_identity")
    if (
        not isinstance(portable, Mapping)
        or portable.get("schema") != DCR_FOREST_PORTABLE_SCHEMA
        or identity != content_identity(dict(portable))
    ):
        raise UiProjectionRepairError("DCR-011 portable forest identity is stale")
    return str(identity)


def _desktop_current(value: Any) -> str:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != DESKTOP_EXPECTATIONS_SCHEMA
        or value.get("interface") != DESKTOP_EXPECTATIONS_INTERFACE
    ):
        raise UiProjectionRepairError("current typed DCR-014 desktop expectations required")
    body = dict(value)
    identity = body.pop("identity", "")
    if (
        identity != "sha256:" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        or body.get("authoritative") is not False
        or body.get("scan_mode") != "static_source_only"
    ):
        raise UiProjectionRepairError("DCR-014 desktop identity is stale")
    if not isinstance(body.get("effective_expectations"), Sequence) or body.get("blockers"):
        raise UiProjectionRepairError("DCR-014 desktop expectations are blocked")
    return str(identity)


def _graph_current(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise UiProjectionRepairError("current DCR-021 graph required")
    body = dict(value)
    identity, encoded = body.pop("graph_cid", ""), body.pop("canonical_bytes", "")
    if (
        body.get("schema") != MCP_CONTRACT_GRAPH_SCHEMA
        or body.get("interface") != MCP_CONTRACT_GRAPH_INTERFACE
        or body.get("authoritative") is not False
        or encoded != canonical_json_bytes(body).decode("utf-8")
        or identity != content_identity(body)
        or body.get("blockers")
    ):
        raise UiProjectionRepairError("DCR-021 graph is stale or blocked")
    return str(identity)


@dataclass(frozen=True)
class UiProjectionRepairRequest:
    surface_kind: str
    source_bytes: bytes
    source_digest: str
    owner_root: str
    relative_path: str
    span_start: int
    span_end: int
    span_digest: str
    replacement_bytes: bytes
    anchor_kind: str
    projection: Mapping[str, str]
    reverse_projection: Mapping[str, str]
    projection_cid: str
    reverse_projection_cid: str
    forest: Mapping[str, Any]
    desktop_expectations: Mapping[str, Any]
    graph: Mapping[str, Any]
    logic_candidate: DatasetsLogicIrResult
    descriptor: OperatorDescriptor
    registry: OperatorRegistry
    reviewed_descriptor_cid: str
    pinned_registry_cid: str = ""
    model_call_count: int = 0
    provider_call_count: int = 0


@dataclass(frozen=True)
class UiProjectionRepairPreview:
    status: UiProjectionPreviewStatus
    reason_codes: tuple[str, ...]
    request_cid: str = ""
    forward_cid: str = ""
    inverse_cid: str = ""
    after_bytes: bytes = b""

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": UI_PROJECTION_PREVIEW_SCHEMA,
            "authoritative": False,
            "server_truth_created": False,
            "implementation_authorized": False,
            "completion_authorized": False,
            "activation_status": "integration_pending_dcr035_dcr040_dcr070_dcr072",
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "request_cid": self.request_cid,
            "forward_cid": self.forward_cid,
            "inverse_cid": self.inverse_cid,
            "after_base64": base64.b64encode(self.after_bytes).decode("ascii"),
            "model_call_count": 0,
            "provider_call_count": 0,
        }
        return {**body, "preview_cid": content_identity(body)}


def _validate(value: UiProjectionRepairRequest) -> tuple[str | None, tuple[str, str, str] | None]:
    if value.surface_kind not in _SURFACES or value.anchor_kind != "literal_exact_span":
        return "dynamic_or_ambiguous_anchor", None
    if (
        type(value.model_call_count) is not int
        or type(value.provider_call_count) is not int
        or value.model_call_count
        or value.provider_call_count
    ):
        return "model_or_provider_route_forbidden", None
    if not isinstance(value.source_bytes, bytes) or value.source_digest != _sha256(
        value.source_bytes
    ):
        return "source_digest_invalid", None
    if not (
        0 <= value.span_start < value.span_end <= len(value.source_bytes)
    ) or value.span_digest != _sha256(value.source_bytes[value.span_start : value.span_end]):
        return "source_span_invalid", None
    if (
        not value.replacement_bytes
        or value.replacement_bytes == value.source_bytes[value.span_start : value.span_end]
    ):
        return "inverse_or_replacement_invalid", None
    try:
        projection, reverse = (
            _projection(value.projection, "projection"),
            _projection(value.reverse_projection, "reverse_projection"),
        )
        if (
            projection != reverse
            or value.projection_cid != content_identity(projection)
            or value.reverse_projection_cid != content_identity(reverse)
        ):
            return "reverse_projection_or_round_trip_identity_invalid", None
        forest_cid, desktop_cid, graph_cid = (
            _forest_current(value.forest),
            _desktop_current(value.desktop_expectations),
            _graph_current(value.graph),
        )
    except UiProjectionRepairError as exc:
        return str(exc), None
    matches = [
        item
        for item in value.desktop_expectations["effective_expectations"]
        if item.get("operation") == projection["operation"]
    ]
    if len(matches) != 1 or matches[0].get("authority_class") in {
        "generated",
        "conformance_test",
        "inferred_prose",
        "archive",
    }:
        return "generated_fixture_or_descriptor_only_authority", None
    if any(
        matches[0].get(field, "") != projection[key]
        for field, key in (
            ("request", "request_schema"),
            ("result", "result_schema"),
            ("transport", "transport"),
        )
    ):
        return "projection_identity_weakened", None
    if (
        not isinstance(value.logic_candidate, DatasetsLogicIrResult)
        or value.logic_candidate.disposition is not DatasetsLogicIrDisposition.NORMALIZED
        or value.logic_candidate.model_call_count != 0
        or value.logic_candidate.mutation_authorized
        or graph_cid not in value.logic_candidate.input_cids
        or forest_cid not in value.logic_candidate.input_cids
    ):
        return "dcr030_logic_candidate_not_current", None
    if (
        not isinstance(value.descriptor, OperatorDescriptor)
        or not isinstance(value.registry, OperatorRegistry)
        or value.reviewed_descriptor_cid != value.descriptor.descriptor_id
        or value.owner_root != value.descriptor.owner_root
        or value.relative_path not in value.descriptor.write_scope
    ):
        return "reviewed_descriptor_manifest_or_owner_invalid", None
    report = value.registry.report()
    if (
        report.get("schema") != OPERATOR_REGISTRY_SCHEMA
        or not isinstance(value.pinned_registry_cid, str)
        or not value.pinned_registry_cid
        or value.pinned_registry_cid != report.get("registry_cid")
        or report.get("reviewed_manifest", {}).get(value.descriptor.operator_id)
        != value.descriptor.descriptor_id
        or value.descriptor not in value.registry.enumerate()
    ):
        return "reviewed_descriptor_manifest_or_owner_invalid", None
    return None, (forest_cid, desktop_cid, graph_cid)


def preview_ui_projection_repair(request: UiProjectionRepairRequest) -> UiProjectionRepairPreview:
    """Create an exact UI projection byte preview plus inverse binding only."""
    if not isinstance(request, UiProjectionRepairRequest):
        return UiProjectionRepairPreview(
            UiProjectionPreviewStatus.REJECTED, ("typed_request_required",)
        )
    reason, roots = _validate(request)
    if reason:
        status = (
            UiProjectionPreviewStatus.ABSTAINED
            if reason
            in {"dynamic_or_ambiguous_anchor", "generated_fixture_or_descriptor_only_authority"}
            else UiProjectionPreviewStatus.REJECTED
        )
        return UiProjectionRepairPreview(status, (reason,))
    assert roots is not None
    forest_cid, desktop_cid, graph_cid = roots
    request_cid = content_identity(
        {
            "surface_kind": request.surface_kind,
            "source_digest": request.source_digest,
            "owner_root": request.owner_root,
            "relative_path": request.relative_path,
            "span": [request.span_start, request.span_end, request.span_digest],
            "replacement_sha256": _sha256(request.replacement_bytes),
            "projection_cid": request.projection_cid,
            "descriptor_cid": request.reviewed_descriptor_cid,
            "pinned_registry_cid": request.pinned_registry_cid,
            "forest_cid": forest_cid,
            "desktop_cid": desktop_cid,
            "graph_cid": graph_cid,
            "logic_candidate_cid": content_identity(request.logic_candidate.to_dict()),
        }
    )
    after = (
        request.source_bytes[: request.span_start]
        + request.replacement_bytes
        + request.source_bytes[request.span_end :]
    )
    forward = content_identity(
        {
            "request_cid": request_cid,
            "before": _sha256(request.source_bytes),
            "after": _sha256(after),
        }
    )
    inverse = content_identity(
        {
            "request_cid": request_cid,
            "before": _sha256(after),
            "after": _sha256(request.source_bytes),
        }
    )
    return UiProjectionRepairPreview(
        UiProjectionPreviewStatus.PREVIEWED,
        ("integration_pending_dcr035_dcr040_dcr070_dcr072",),
        request_cid,
        forward,
        inverse,
        after,
    )


__all__ = [
    "UI_PROJECTION_PREVIEW_SCHEMA",
    "UiProjectionPreviewStatus",
    "UiProjectionRepairError",
    "UiProjectionRepairPreview",
    "UiProjectionRepairRequest",
    "preview_ui_projection_repair",
]
