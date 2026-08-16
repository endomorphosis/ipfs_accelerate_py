"""DCR-021 deterministic cross-repository MCP graph tests."""

from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    build_mcp_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _span(root: str, path: str, digest: str) -> dict[str, object]:
    return {"root": root, "path": path, "sha256": digest, "line": 1, "start_column": 1, "end_column": 20}


def _desktop() -> dict[str, object]:
    return {
        "consumers": [{"root": "swissknife", "path": "src/desktop/client.ts", "sha256": "sha256:client"}],
        "evidence": [
            {"operation": "desktop.open", "ui_action": "open-window", "authority_class": "registration", "declaration_kind": "ui_ir", "source_span": _span("swissknife", "src/desktop/client.ts", "sha256:client")},
            {"operation": "desktop.open", "ui_action": "", "authority_class": "reviewed_declaration", "declaration_kind": "orb_idl", "source_span": _span("mcp-plus-plus", "idl/desktop.idl", "sha256:idl")},
        ],
        "effective_expectations": [
            {"operation": "desktop.open", "request": "OpenRequest", "authority_class": "reviewed_declaration", "source_span": _span("mcp-plus-plus", "descriptors/desktop.json", "sha256:descriptor")}
        ],
        "blockers": [],
    }


def _provider(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"rows": rows if rows is not None else [{
        "operation": "desktop.open", "status": "resolved", "dispatcher": "DesktopDispatcher",
        "handler": "open_desktop", "effect": "desktop.open", "source_digest": "sha256:provider", "reason": "",
    }]}


def _identities() -> list[dict[str, object]]:
    return [{
        "semantic_cid": "bafy-runtime", "declaration_cid": "bafy-declaration",
        "semantic_key": {"operation": "desktop.open", "runtime_instance": "desktop-main"},
    }]


def test_complete_chain_is_typed_and_canonical_bytes_reconstruct_its_cid() -> None:
    graph = build_mcp_contract_graph(
        provider_surfaces=_provider(), desktop_expectations=_desktop(), identities=_identities()
    )

    assert graph["blockers"] == []
    assert {node["kind"] for node in graph["nodes"]} >= {
        "ui_action", "expected_descriptor", "orb_idl", "method_schema", "mediator_route",
        "dispatcher", "handler", "effect", "receipt_runtime_identity",
    }
    assert all(node["state"] != "observed_implementation" for node in graph["nodes"] if node["kind"] == "expected_descriptor")
    assert content_identity(json.loads(graph["canonical_bytes"])) == graph["graph_cid"]


def test_missing_or_ambiguous_observed_provider_is_a_blocker_not_an_expected_handler() -> None:
    graph = build_mcp_contract_graph(
        provider_surfaces=_provider([]), desktop_expectations=_desktop(), identities=_identities()
    )
    assert any(item["kind"] == "mandatory_consumer_unresolved" for item in graph["blockers"])
    assert not any(node["kind"] == "handler" for node in graph["nodes"])

    ambiguous = build_mcp_contract_graph(
        provider_surfaces=_provider(_provider()["rows"] * 2), desktop_expectations=_desktop(), identities=_identities()
    )
    assert any(item["kind"] == "mandatory_consumer_ambiguous" for item in ambiguous["blockers"])

    conflicting_desktop = _desktop()
    conflicting_desktop["blockers"] = [{"kind": "contradictory_desktop_expectation", "operation": "desktop.open"}]
    conflict = build_mcp_contract_graph(
        provider_surfaces=_provider(), desktop_expectations=conflicting_desktop, identities=_identities()
    )
    assert any(item["kind"] == "authority_conflict" for item in conflict["blockers"])
