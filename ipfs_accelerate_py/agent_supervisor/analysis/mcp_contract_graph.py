"""DCR-021 deterministic cross-repository MCP contract graph.

The graph consumes static DCR-013/014/020 projections as mappings.  It never
imports a provider or desktop application, and expected declarations remain
separate from observed provider registrations throughout the projection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity

MCP_CONTRACT_GRAPH_INTERFACE: Final = "McpContractGraph@1"
MCP_CONTRACT_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-graph@1"
)


class McpContractGraphError(ValueError):
    """A graph input is malformed or lacks its required static provenance."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise McpContractGraphError(f"{name} must be a mapping")
    return value


def _records(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise McpContractGraphError(f"{name} must be a sequence of records")
    return [_mapping(item, name) for item in value]


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _node_id(kind: str, *parts: str) -> str:
    return content_identity({"kind": kind, "parts": list(parts)})


def _span(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("source_span")
    return value if isinstance(value, Mapping) else {}


def _node(
    kind: str, label: str, *, authority: str, state: str, source: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    source_value = dict(source or {})
    return {
        "id": _node_id(kind, label, authority, state, str(source_value.get("sha256", ""))),
        "kind": kind,
        "label": label,
        "authority_class": authority,
        "state": state,
        "source": source_value,
    }


def _edge(source: str, target: str, relation: str, authority: str) -> dict[str, str]:
    return {
        "id": _node_id("edge", source, target, relation, authority),
        "source": source,
        "target": target,
        "relation": relation,
        "authority_class": authority,
    }


def _provider_rows(provider_surfaces: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _records(provider_surfaces.get("rows", ()), "provider rows")


def _identity_records(identities: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for item in _records(identities, "identities"):
        semantic = item.get("semantic_key")
        if not isinstance(semantic, Mapping) or not _text(item.get("semantic_cid")):
            raise McpContractGraphError("identity must carry a semantic key and recomputed CID")
        result.append(item)
    return result


def build_mcp_contract_graph(
    *,
    provider_surfaces: Mapping[str, Any],
    desktop_expectations: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project DCR inventories into a canonical, fail-closed graph.

    Mandatory desktop consumers need exactly one operation, descriptor,
    ORB/IDL declaration, resolved provider row, and DCR-020 runtime identity.
    Missing or multiple candidates remain blockers; no expected node can mint a
    provider handler/effect node.
    """

    provider = _mapping(provider_surfaces, "provider_surfaces")
    desktop = _mapping(desktop_expectations, "desktop_expectations")
    rows = _provider_rows(provider)
    evidence = _records(desktop.get("evidence", ()), "desktop evidence")
    expectations = _records(desktop.get("effective_expectations", ()), "effective expectations")
    consumers = _records(desktop.get("consumers", ()), "desktop consumers")
    identity_rows = _identity_records(identities)
    blockers: list[dict[str, Any]] = []
    for item in _records(desktop.get("blockers", ()), "desktop blockers"):
        upstream_kind = _text(item.get("kind"))
        blockers.append(
            {
                "kind": "authority_conflict"
                if upstream_kind == "contradictory_desktop_expectation"
                else "upstream_desktop_blocker",
                "detail": item,
            }
        )
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    expectation_by_op: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    idl_by_op: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    evidence_by_consumer: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    provider_by_op: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identity_by_op: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in expectations:
        if _text(record.get("operation")):
            expectation_by_op[_text(record["operation"])].append(record)
    for record in evidence:
        operation = _text(record.get("operation"))
        if operation and _text(record.get("declaration_kind")) == "orb_idl":
            idl_by_op[operation].append(record)
        span = _span(record)
        key = (_text(span.get("root")), _text(span.get("path")))
        if operation and all(key):
            evidence_by_consumer[key].append(record)
    for row in rows:
        operation = _text(row.get("operation"))
        if operation:
            provider_by_op[operation].append(row)
    for identity in identity_rows:
        semantic = _mapping(identity["semantic_key"], "semantic key")
        operation = _text(semantic.get("operation"))
        if operation:
            identity_by_op[operation].append(identity)

    for consumer in consumers:
        consumer_root, consumer_path = _text(consumer.get("root")), _text(consumer.get("path"))
        consumer_id = _node_id("consumer", consumer_root, consumer_path, _text(consumer.get("sha256")))
        consumer_node = {
            "id": consumer_id,
            "kind": "desktop_consumer",
            "label": f"{consumer_root}:{consumer_path}",
            "authority_class": "registration",
            "state": "observed_static_consumer",
            "source": dict(consumer),
        }
        nodes.append(consumer_node)
        operations = sorted({_text(item.get("operation")) for item in evidence_by_consumer[(consumer_root, consumer_path)] if _text(item.get("operation"))})
        if len(operations) != 1:
            blockers.append(
                {
                    "kind": "mandatory_consumer_not_exactly_one_operation",
                    "consumer": consumer_node["label"],
                    "operations": operations,
                }
            )
            continue
        operation = operations[0]
        descriptor_matches = expectation_by_op[operation]
        idl_matches = idl_by_op[operation]
        observed_rows = [row for row in provider_by_op[operation] if _text(row.get("status")) == "resolved"]
        identity_matches = identity_by_op[operation]
        for kind, matches in (
            ("descriptor", descriptor_matches),
            ("orb_idl", idl_matches),
            ("observed_provider", observed_rows),
            ("runtime_identity", identity_matches),
        ):
            if len(matches) != 1:
                blockers.append(
                    {
                        "kind": "mandatory_consumer_unresolved" if not matches else "mandatory_consumer_ambiguous",
                        "consumer": consumer_node["label"],
                        "operation": operation,
                        "required": kind,
                        "candidates": len(matches),
                    }
                )
        if not all(len(matches) == 1 for matches in (descriptor_matches, idl_matches, observed_rows, identity_matches)):
            continue
        descriptor, idl, observed, identity = (
            descriptor_matches[0], idl_matches[0], observed_rows[0], identity_matches[0]
        )
        descriptor_node = _node(
            "expected_descriptor", operation,
            authority=_text(descriptor.get("authority_class")) or "unknown",
            state="expected_declaration", source=_span(descriptor),
        )
        idl_node = _node(
            "orb_idl", operation,
            authority=_text(idl.get("authority_class")) or "unknown",
            state="expected_declaration", source=_span(idl),
        )
        method_node = _node(
            "method_schema", operation + ":" + _text(descriptor.get("request")),
            authority=_text(descriptor.get("authority_class")) or "unknown",
            state="expected_declaration", source=_span(descriptor),
        )
        route_node = _node(
            "mediator_route", operation + ":" + _text(observed.get("dispatcher")),
            authority="observed_provider", state="observed_implementation",
            source={"sha256": _text(observed.get("source_digest"))},
        )
        dispatcher_node = _node(
            "dispatcher", _text(observed.get("dispatcher")), authority="observed_provider",
            state="observed_implementation", source={"sha256": _text(observed.get("source_digest"))},
        )
        handler_node = _node(
            "handler", _text(observed.get("handler")), authority="observed_provider",
            state="observed_implementation", source={"sha256": _text(observed.get("source_digest"))},
        )
        effect_node = _node(
            "effect", _text(observed.get("effect")), authority="observed_provider",
            state="observed_implementation", source={"sha256": _text(observed.get("source_digest"))},
        )
        receipt_node = _node(
            "receipt_runtime_identity", _text(identity.get("semantic_cid")), authority="identity",
            state="recomputed_identity", source={"declaration_cid": _text(identity.get("declaration_cid"))},
        )
        nodes.extend((descriptor_node, idl_node, method_node, route_node, dispatcher_node, handler_node, effect_node, receipt_node))
        ui_records = [item for item in evidence_by_consumer[(consumer_root, consumer_path)] if _text(item.get("ui_action"))]
        ui_node = consumer_node
        if ui_records:
            ui = ui_records[0]
            ui_node = _node(
                "ui_action", _text(ui.get("ui_action")), authority=_text(ui.get("authority_class")) or "unknown",
                state="expected_ui_action", source=_span(ui),
            )
            nodes.append(ui_node)
            edges.append(_edge(consumer_node["id"], ui_node["id"], "declares_action", "registration"))
        chain = (
            (ui_node, descriptor_node, "expects_descriptor"),
            (descriptor_node, idl_node, "binds_orb_idl"),
            (idl_node, method_node, "defines_method_schema"),
            (method_node, route_node, "binds_mediator_route"),
            (route_node, dispatcher_node, "routes_to_observed_dispatcher"),
            (dispatcher_node, handler_node, "dispatches_to_handler"),
            (handler_node, effect_node, "performs_effect"),
            (effect_node, receipt_node, "emits_receipt_runtime_identity"),
        )
        for source, target, relation in chain:
            edges.append(_edge(source["id"], target["id"], relation, source["authority_class"]))
    for row in rows:
        if _text(row.get("status")) != "resolved":
            blockers.append(
                {
                    "kind": "provider_surface_unresolved_or_ambiguous",
                    "operation": _text(row.get("operation")),
                    "status": _text(row.get("status")),
                    "reason": _text(row.get("reason")),
                }
            )
    body = {
        "schema": MCP_CONTRACT_GRAPH_SCHEMA,
        "interface": MCP_CONTRACT_GRAPH_INTERFACE,
        "authoritative": False,
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(edges, key=lambda item: item["id"]),
        "blockers": sorted(blockers, key=lambda item: canonical_json_bytes(item)),
    }
    return {
        **body,
        "graph_cid": content_identity(body),
        "canonical_bytes": canonical_json_bytes(body).decode("utf-8"),
    }


__all__ = [
    "MCP_CONTRACT_GRAPH_INTERFACE",
    "MCP_CONTRACT_GRAPH_SCHEMA",
    "McpContractGraphError",
    "build_mcp_contract_graph",
]
