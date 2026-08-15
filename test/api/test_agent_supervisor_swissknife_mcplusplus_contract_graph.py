"""DCR-090: SwissKnife ↔ MCP++ hermetic contract graph fixture tests.

Complements DCR-021's live graph module with a structural interop fixture that
cannot self-green as live conformance.
"""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    HERMETIC_CONFORMANCE_INTERFACE,
    build_contract_graph_fixture,
    materialize_hermetic_conformance,
    validate_hermetic_conformance,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "Mcp-Plus-Plus").is_dir() and (candidate / "swissknife").is_dir():
            return candidate
    return here.parents[4]


def test_swissknife_mcp_contract_graph_fixture_shape() -> None:
    graph = build_contract_graph_fixture()
    node_ids = {node["id"] for node in graph["nodes"]}
    assert "ui_action" in node_ids
    assert "orb_idl" in node_ids
    assert "mcp_method" in node_ids
    assert "handler" in node_ids
    edge_kinds = {edge["kind"] for edge in graph["edges"]}
    assert "binds" in edge_kinds
    assert "declares" in edge_kinds
    assert "routes" in edge_kinds
    roots = {node["root"] for node in graph["nodes"]}
    assert "swissknife" in roots
    assert "Mcp-Plus-Plus" in roots
    assert "external/ipfs_accelerate" in roots


def test_graph_fixture_stable_cid() -> None:
    a = build_contract_graph_fixture(snapshot_id="snap:dcr090")
    b = build_contract_graph_fixture(snapshot_id="snap:dcr090")
    assert a["graph_cid"] == b["graph_cid"]
    c = build_contract_graph_fixture(snapshot_id="snap:other")
    assert c["graph_cid"] != a["graph_cid"]


def test_hermetic_report_includes_profile_matrix() -> None:
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        real_server_available=False,
    )
    assert report.INTERFACE == HERMETIC_CONFORMANCE_INTERFACE
    matrix = report.profile_matrix
    assert matrix.get("live_required_for_green") is True
    assert "initialize" in matrix.get("families", [])
    assert "tools/list" in matrix.get("families", [])
    assert "tools/call" in matrix.get("families", [])
    assert "logic" in matrix.get("families", [])


def test_materialized_payload_embeds_contract_graph(tmp_path: Path) -> None:
    dest = tmp_path / "hermetic-conformance.json"
    payload = materialize_hermetic_conformance(
        repo_root=_repo_root(),
        destination=dest,
    )
    graph = payload["contract_graph"]
    assert graph["live_conformance"] is False
    assert len(graph["nodes"]) >= 4
    assert graph["graph_cid"].startswith("sha256:")
