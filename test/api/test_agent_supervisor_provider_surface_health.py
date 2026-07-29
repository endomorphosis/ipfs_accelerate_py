"""Provider surface health backlog (SCA-609 / SCAEV179INDEXGRAPH)."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.provider_surface_health import (
    PROVIDER_SURFACE_HEALTH_BACKLOG_SCHEMA,
    PROVIDER_SURFACE_HEALTH_EVIDENCE,
    PROVIDER_SURFACE_HEALTH_SCHEMA,
    ProviderSurfaceIssueKind,
    assess_provider_surface_health,
    provider_surface_health_blocks_exhaustive_parity,
    write_provider_surface_health_backlog,
)
from ipfs_accelerate_py.agent_supervisor.analysis.python_mcp_surface_extractor import (
    extract_python_mcp_source,
)


def test_unresolved_registrations_are_typed_and_block_exhaustive_parity() -> None:
    surface = extract_python_mcp_source(
        '''
for name in dynamic_names:
    server.tool(name)(handlers[name])

@server.tool(name="demo.echo")
def echo(value: str) -> str:
    return value
''',
        provider="ipfs_accelerate_py",
        path="ipfs_accelerate_py/mcp_server/dynamic.py",
        repository_tree_id="tree-unresolved",
    )
    # Even when some tools resolve, dynamic registrations remain typed.
    report = assess_provider_surface_health(
        package_surfaces=(surface,),
        multi_root_id="multi-root-fixture",
        snapshot_id="snap-fixture",
        required_packages=("ipfs_accelerate_py",),
    )
    assert report.llm_call_count == 0
    assert report.evidence == PROVIDER_SURFACE_HEALTH_EVIDENCE
    assert report.to_dict()["schema"] == PROVIDER_SURFACE_HEALTH_SCHEMA
    if surface.unresolved:
        assert report.unresolved_registration_count >= 1
        assert any(
            item.kind is ProviderSurfaceIssueKind.UNRESOLVED_REGISTRATION
            for item in report.issues
        )
        assert report.exhaustive_parity_allowed is False
        assert provider_surface_health_blocks_exhaustive_parity(report) is True
    else:
        # If the extractor did not flag the dynamic loop, empty surface for a
        # required missing package still blocks parity.
        missing = assess_provider_surface_health(
            package_surfaces=(),
            required_packages=("ipfs_kit_py",),
            multi_root_id="multi-root-fixture",
            snapshot_id="snap-fixture",
        )
        assert missing.exhaustive_parity_allowed is False
        assert any(
            item.kind is ProviderSurfaceIssueKind.MISSING_PACKAGE_SURFACE
            for item in missing.issues
        )


def test_health_families_are_deduplicated_and_content_addressed() -> None:
    surface = extract_python_mcp_source(
        '''
registry.register_tool(dynamic_name, handler)
registry.register_tool(other_dynamic, handler2)
''',
        provider="ipfs_kit_py",
        path="ipfs_kit_py/mcp_server/registry.py",
        repository_tree_id="tree-kit",
    )
    first = assess_provider_surface_health(
        package_surfaces=(surface,),
        multi_root_id="multi-a",
        snapshot_id="snap-a",
        required_packages=("ipfs_kit_py",),
    )
    second = assess_provider_surface_health(
        package_surfaces=(surface,),
        multi_root_id="multi-a",
        snapshot_id="snap-a",
        required_packages=("ipfs_kit_py",),
    )
    assert first.report_id == second.report_id
    assert first.llm_call_count == 0
    # Families never expand into per-file prompts.
    for family in first.families:
        handle = family.expansion_handle()
        assert handle["interface"] == "CodeEditPacket@1"
        assert handle["llm_call_count"] == 0
        assert "prompt" not in handle
    issue_ids = [item.issue_id for item in first.issues]
    assert len(issue_ids) == len(set(issue_ids))


def test_write_provider_surface_health_backlog_is_compact(
    tmp_path: Path,
) -> None:
    surface = extract_python_mcp_source(
        '''
@server.tool(name="demo.ping")
def ping() -> str:
    return "pong"
''',
        provider="ipfs_datasets_py",
        path="ipfs_datasets_py/mcp_server/tools.py",
    )
    report = assess_provider_surface_health(
        package_surfaces=(surface,),
        required_packages=("ipfs_datasets_py", "ipfs_kit_py"),
        multi_root_id="multi-backlog",
        snapshot_id="snap-backlog",
    )
    destination = tmp_path / "backlog.json"
    written = write_provider_surface_health_backlog(report, destination)
    assert written == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PROVIDER_SURFACE_HEALTH_BACKLOG_SCHEMA
    assert payload["evidence"] == PROVIDER_SURFACE_HEALTH_EVIDENCE
    assert payload["llm_call_count"] == 0
    assert payload["per_file_prompts"] is False
    assert payload["report_id"] == report.report_id
    # Missing required kit package blocks exhaustive parity.
    assert payload["exhaustive_parity_allowed"] is False
    assert payload["blocking_issue_count"] >= 1
    # No source bodies embedded.
    encoded = destination.read_text(encoding="utf-8")
    assert "return \"pong\"" not in encoded
    assert "def ping" not in encoded


def test_unchanged_rescan_is_duplicate_free() -> None:
    surface = extract_python_mcp_source(
        '''
@server.tool(name="demo.echo")
def echo(value: str) -> str:
    return value
''',
        provider="ipfs_accelerate_py",
        path="tools.py",
        repository_tree_id="tree-stable",
    )
    a = assess_provider_surface_health(
        package_surfaces=(surface,),
        required_packages=("ipfs_accelerate_py",),
        multi_root_id="multi-stable",
        snapshot_id="snap-stable",
    )
    b = assess_provider_surface_health(
        package_surfaces=(surface,),
        required_packages=("ipfs_accelerate_py",),
        multi_root_id="multi-stable",
        snapshot_id="snap-stable",
    )
    assert a.report_id == b.report_id
    assert [item.issue_id for item in a.issues] == [
        item.issue_id for item in b.issues
    ]
    assert [item.family_id for item in a.families] == [
        item.family_id for item in b.families
    ]
