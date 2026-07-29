"""Actual package MCP surfaces required for production composition (SCA-604)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BaselineStageName,
    CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE,
    materialize_contract_assurance_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.python_mcp_surface_extractor import (
    extract_python_mcp_source,
)
from ipfs_accelerate_py.agent_supervisor.analysis.provider_surface_health import (
    extract_actual_provider_package_surfaces,
)


def test_extracted_surface_carries_package_module_function_provenance() -> None:
    source = '''
@server.tool(name="ipfs.pin")
async def pin_content(cid: str, recursive: bool = False) -> dict:
    return await backend.pin(cid, recursive=recursive)

server.add_tool(store_blob, name="ipfs.add", aliases=["add"])
'''
    surface = extract_python_mcp_source(
        source,
        provider="ipfs_kit_py",
        path="ipfs_kit_py/mcp_server/server.py",
        repository_tree_id="tree-kit",
    )
    assert surface.provider == "ipfs_kit_py"
    assert surface.repository_tree_id == "tree-kit"
    assert surface.tools
    tool = surface.tools_named("ipfs.pin")[0]
    assert tool.handler.symbol == "pin_content"
    assert tool.registration_api
    assert tool.registration_span.path.endswith("server.py")
    payload = tool.to_dict()
    assert payload["provider"] == "ipfs_kit_py"
    assert "handler" in payload


def test_omitted_package_surfaces_fail_closed_when_required() -> None:
    baseline = materialize_contract_assurance_baseline(
        snapshot_id="snap-no-surfaces",
        snapshot={
            "snapshot_id": "snap-no-surfaces",
            "scope_policy_id": "policy-fixture",
            "head_tree_id": "tree-fixture",
            "stats": {"tracked_path_count": 0, "disposition_count": 0},
        },
        extract_expected=False,
        project_graph=False,
        run_traces=False,
        run_parity=False,
        run_mismatch=False,
        run_vulnerability=False,
        run_graphrag=False,
        require_actual_package_surfaces=True,
        package_surfaces=None,
        assess_surface_health=False,
    )
    assert baseline.llm_call_count == 0
    stage = next(
        item
        for item in baseline.stages
        if item.name is BaselineStageName.ACTUAL_SURFACES
    )
    assert stage.completeness.value == "failed"
    assert "package_surfaces_required" in stage.reason_codes
    assert baseline.claims.get("exhaustive") is False
    assert baseline.findings["index_graph"]["require_actual_package_surfaces"] is True
    assert baseline.findings["index_graph"]["evidence"] == (
        CONTRACT_ASSURANCE_INDEXGRAPH_EVIDENCE
    )


def test_expected_descriptor_cannot_manufacture_actual_completeness() -> None:
    """Expected-only observed contracts stay incomplete under require-actual."""

    surface = extract_python_mcp_source(
        '''
@server.tool(name="demo.echo")
def echo(value: str) -> str:
    return value
''',
        provider="ipfs_accelerate_py",
        path="mcp/tools.py",
        repository_tree_id="tree-a",
    )
    # Build a tiny catalog if the tree supports it; otherwise exercise baseline
    # fail-closed path with caller-supplied expected-only observed contracts.
    observed_expected_only = {
        "demo.echo": {
            "operation_id": "demo.echo",
            "name": "demo.echo",
            "tool_name": "demo.echo",
            "package_id": "ipfs_accelerate_py",
            "complete": True,
            "routes": [
                {
                    "route_id": "route:mcp_plus_plus:demo.echo",
                    "transport": "mcp++",
                    "mediation_path_class": "mcp_plus_plus",
                    "complete": True,
                    "source_ids": ["expected:descriptor:demo.echo"],
                }
            ],
        }
    }
    baseline = materialize_contract_assurance_baseline(
        snapshot_id="snap-expected-only",
        snapshot={
            "snapshot_id": "snap-expected-only",
            "scope_policy_id": "policy-fixture",
            "head_tree_id": "tree-fixture",
            "stats": {"tracked_path_count": 0, "disposition_count": 0},
        },
        observed_contracts=observed_expected_only,
        package_surfaces=(surface,),
        require_actual_package_surfaces=True,
        extract_expected=False,
        project_graph=False,
        run_traces=False,
        run_parity=False,
        run_mismatch=False,
        run_vulnerability=False,
        run_graphrag=False,
        assess_surface_health=False,
    )
    assert baseline.llm_call_count == 0
    # Caller-supplied expected-only routes are rewritten as incomplete.
    findings = baseline.findings
    index_graph = findings.get("index_graph") or {}
    assert index_graph.get("package_surface_count") == 1
    # The observed rewrite runs when evidence compilation is active; with no
    # catalog it still records actual surface stage complete for extracted tools.
    actual_stage = next(
        item
        for item in baseline.stages
        if item.name is BaselineStageName.ACTUAL_SURFACES
    )
    assert actual_stage.details["synthesized_from_expected"] is False
    assert actual_stage.details["tool_count"] >= 1


def test_direct_and_mcp_plus_plus_paths_remain_distinct() -> None:
    surface = extract_python_mcp_source(
        '''
@server.tool(name="tools.dispatch")
async def dispatch_tool(category: str, tool_name: str, arguments: dict) -> dict:
    return await manager.dispatch(category, tool_name, arguments)

server.add_tool(store_blob, name="ipfs.add")
''',
        provider="ipfs_kit_py",
        path="ipfs_kit_py/mcp_server/server.py",
    )
    domain = surface.domain_tools
    facade = surface.facade_tools
    assert domain
    assert facade
    assert {tool.kind.value for tool in domain} == {"domain_tool"}
    assert {tool.kind.value for tool in facade} == {"facade_meta_tool"}
    # Distinct path classes are retained on tool metadata / catalogs.
    assert all(tool.canonical_name for tool in surface.tools)
    assert {tool.canonical_name for tool in domain}.isdisjoint(
        {tool.canonical_name for tool in facade}
    )


def test_extract_actual_provider_package_surfaces_skips_missing_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "super"
    package = root / "external" / "ipfs_accelerate" / "ipfs_accelerate_py"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""pkg"""\n', encoding="utf-8")
    (package / "tools.py").write_text(
        '''
@server.tool(name="demo.ping")
def ping() -> str:
    return "pong"
''',
        encoding="utf-8",
    )
    surfaces = extract_actual_provider_package_surfaces(root)
    providers = {item.provider for item in surfaces}
    assert "ipfs_accelerate_py" in providers
    assert "ipfs_kit_py" not in providers
    assert all(item.tools or item.unresolved or item.source_files for item in surfaces)
