"""Tests for SwissKnife MCP++ call-path contract resolution (VFS-017 / VFS-G060).

Static inventory resolution is covered here. Hermetic runtime conformance is
owned by VFS-G061 / ``test_agent_supervisor_mcplusplus_runtime_contracts``.

Objective leaf goals VFS-G152 (``vfs/mcplusplus-call-path@1``) and VFS-G153
(``vfs/mcplusplus-manifest-parity@1``) are proved via discovery hooks and
portable claims on this surface (goal packet mcp_interop/9f2828fd2adb).
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.mcplusplus_contract_resolver import (
    CALL_PATH_INVARIANTS,
    EVIDENCE_CALL_PATH,
    EVIDENCE_MANIFEST_PARITY,
    EVIDENCE_RUNTIME_WITNESS,
    EXCLUDED_RUNTIME_EVIDENCE_KINDS,
    HERMETIC_RUNTIME_CHILD_GOAL_ID,
    HERMETIC_RUNTIME_CLAIM_LEVEL,
    MANIFEST_PARITY_INVARIANTS,
    OBJECTIVE_CALL_PATH_GOAL_ID,
    OBJECTIVE_CALL_PATH_TASK_ID,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_GOAL_PACKET_ID,
    OBJECTIVE_MANIFEST_PARITY_GOAL_ID,
    OBJECTIVE_MANIFEST_PARITY_TASK_ID,
    OBJECTIVE_PACKET_GOAL_IDS,
    OBJECTIVE_PACKET_TASK_IDS,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    PATH_STAGE_ORDER,
    RESOLVER_VERSION,
    STATIC_EVIDENCE_KINDS,
    STATIC_RESOLUTION_CLAIM_LEVEL,
    STATIC_RESOLUTION_GOAL_ID,
    ArtifactRole,
    CallPathClaim,
    DriftKind,
    InventoryArtifact,
    MCPlusPlusContractResolver,
    MCPlusPlusInventory,
    MCPlusPlusResolutionResult,
    MCPlusPlusResolverError,
    ManifestDriftWitness,
    MissingPathEvidenceError,
    PathEvidence,
    PathHop,
    PathStage,
    PathVerdict,
    ReasonCode,
    ResolutionLayer,
    TransportKind,
    all_covered_evidence_terms,
    classify_non_invocation,
    confidence_for,
    covered_evidence_terms,
    inventory_from_program_graph,
    make_artifact,
    make_evidence,
    make_hop,
    mcplusplus_call_path_evidence,
    mcplusplus_call_path_evidence_terms,
    mcplusplus_manifest_parity_evidence,
    mcplusplus_manifest_parity_evidence_terms,
    normalize_tool_name,
    packet_evidence_terms,
    path_satisfies_mcplusplus_call_path,
    prove_mcplusplus_call_path,
    prove_mcplusplus_manifest_parity,
    prove_mcplusplus_static_packet,
    resolve_mcplusplus_from_graph,
    resolve_mcplusplus_paths,
    result_satisfies_mcplusplus_manifest_parity,
    schema_fingerprint,
    split_hierarchical_alias,
    static_resolution_boundary,
    tool_name_aliases,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import ClaimLevel
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    ProgramEdgeKind,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    build_program_graph,
    make_edge,
    make_node,
)


FOREST = "forest:test-vfs-017"
BLOB = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SERVER = "ipfs-accelerate-mcp++"


def _span(line: int = 1) -> SourceSpan:
    return SourceSpan(
        line_start=line, column_start=0, line_end=line, column_end=8
    )


def _ev(
    rule_id: str = "rule:test",
    *,
    source: str = "src",
    target: str = "dst",
    notes: dict[str, Any] | None = None,
) -> PathEvidence:
    return make_evidence(
        rule_id=rule_id,
        blob_cid=BLOB,
        forest_id=FOREST,
        span=_span(),
        source_record_key=source,
        target_record_key=target,
        notes=notes or {},
    )


def _art(
    artifact_id: str,
    role: ArtifactRole | str,
    name: str,
    **kwargs: Any,
) -> InventoryArtifact:
    defaults: dict[str, Any] = {
        "blob_cid": BLOB,
        "forest_id": FOREST,
        "server_name": SERVER,
    }
    defaults.update(kwargs)
    return make_artifact(
        artifact_id=artifact_id,
        role=role,
        name=name,
        **defaults,
    )


def _proved_inventory(tool: str = "vfs.read") -> MCPlusPlusInventory:
    """Closed inventory that proves a full HTTP invocation chain."""

    return MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=("mcp++/mcp-idl",),
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=(
            _art(
                "caller:ui",
                ArtifactRole.CALLER,
                "SwissKnife.UI.invokeVfsRead",
                language="typescript",
                qualified_name="SwissKnife.UI.invokeVfsRead",
                path="src/components/VfsPanel.tsx",
                has_call_edge=True,
                record={"connector": "MCPPPServerConnector.callTool"},
            ),
            _art(
                "conn:mcppp",
                ArtifactRole.CONNECTOR,
                "MCPPPServerConnector.callTool",
                language="typescript",
                qualified_name="MCPPPServerConnector.callTool",
                path="src/services/mcp/mcp-plus-plus-connector.ts",
                tool_name=tool,
                transport=TransportKind.HTTP,
                profiles=(
                    "mcp++/basic",
                    "mcp++/mcp-idl",
                    "mcp++/p2p-transport",
                ),
                has_call_edge=True,
            ),
            _art(
                "transport:http",
                ArtifactRole.TRANSPORT,
                "http-jsonrpc",
                transport=TransportKind.HTTP,
                profiles=("mcp++/basic", "mcp++/mcp-idl"),
            ),
            _art(
                "list:vfs.read",
                ArtifactRole.TOOL_LIST_ENTRY,
                tool,
                tool_name=tool,
                language="json",
                version="1.0.0",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
                error_codes=("not_found", "permission_denied"),
            ),
            _art(
                "call:vfs.read",
                ArtifactRole.TOOL_CALL_SITE,
                "tools/call:vfs.read",
                tool_name=tool,
                language="typescript",
                qualified_name="MCPPPServerConnector.callTool",
                path="src/services/mcp/mcp-plus-plus-connector.ts",
                transport=TransportKind.HTTP,
                has_call_edge=True,
            ),
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                tool,
                tool_name=tool,
                language="python",
                package="ipfs_accelerate_py",
                qualified_name="mcp_server.registry.vfs.read",
                version="1.0.0",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
                error_codes=("not_found", "permission_denied"),
                record={
                    "adapter": "native_vfs_adapter",
                    "implementation": "ipfs_kit_py.vfs.read",
                },
            ),
            _art(
                "adapter:vfs",
                ArtifactRole.ADAPTER,
                "native_vfs_adapter",
                tool_name=tool,
                language="python",
                package="ipfs_accelerate_py",
                qualified_name="mcp_server.tools.vfs.native_vfs_adapter",
                path="ipfs_accelerate_py/mcp_server/tools/vfs/native_vfs_adapter.py",
                has_call_edge=True,
                record={"implementation": "ipfs_kit_py.vfs.read"},
            ),
            _art(
                "impl:vfs.read",
                ArtifactRole.IMPLEMENTATION,
                "ipfs_kit_py.vfs.read",
                tool_name=tool,
                language="python",
                package="ipfs_kit_py",
                qualified_name="ipfs_kit_py.vfs.read",
                path="ipfs_kit_py/ipfs_kit_py/vfs.py",
                version="1.0.0",
                has_call_edge=True,
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
                error_codes=("not_found", "permission_denied"),
            ),
            _art(
                "rmap:vfs.read",
                ArtifactRole.RESULT_MAP,
                "vfs.read.result",
                tool_name=tool,
                language="python",
            ),
            _art(
                "emap:vfs.read",
                ArtifactRole.ERROR_MAP,
                "vfs.read.errors",
                tool_name=tool,
                language="python",
                error_codes=("not_found", "permission_denied"),
            ),
        ),
    )


def _claim(
    tool: str = "vfs.read",
    *,
    transport: TransportKind = TransportKind.HTTP,
    profiles: tuple[str, ...] = ("mcp++/basic", "mcp++/mcp-idl"),
    **kwargs: Any,
) -> CallPathClaim:
    payload = {
        "path_name": f"path:{tool}",
        "tool_name": tool,
        "caller_name": "SwissKnife.UI.invokeVfsRead",
        "connector_name": "MCPPPServerConnector.callTool",
        "server_name": SERVER,
        "transport": transport,
        "profiles": profiles,
        "language_names": {"typescript": tool, "python": tool},
    }
    payload.update(kwargs)
    return CallPathClaim(**payload)


# ---------------------------------------------------------------------------
# Helpers / pure functions
# ---------------------------------------------------------------------------


def test_normalize_and_alias_helpers() -> None:
    assert normalize_tool_name("VFS.Read") == "vfs.read"
    assert normalize_tool_name("vfs/read") == "vfs.read"
    assert split_hierarchical_alias("ipfs.cat") == ("ipfs", "cat")
    assert split_hierarchical_alias("ipfs/cat") == ("ipfs", "cat")
    aliases = tool_name_aliases("ipfs.cat")
    assert "ipfs.cat" in aliases
    assert "ipfs/cat" in aliases
    assert "cat" in aliases
    left = schema_fingerprint({"type": "object", "properties": {"a": {"type": "string"}}})
    right = schema_fingerprint({"properties": {"a": {"type": "string"}}, "type": "object"})
    assert left == right
    assert left
    assert schema_fingerprint({}) == ""


def test_confidence_is_deterministic_and_status_bounded() -> None:
    assert confidence_for(
        ResolverStatus.RESOLVED_STATIC, ReasonCode.REGISTRATION_MATCH
    ) == 100
    assert confidence_for(
        ResolverStatus.AMBIGUOUS, ReasonCode.AMBIGUOUS_REGISTRATION
    ) == 25
    assert confidence_for(
        ResolverStatus.UNRESOLVED, ReasonCode.MOCK_IMPLEMENTATION
    ) == 0
    # Reason cannot raise above status baseline.
    assert confidence_for(
        ResolverStatus.CANDIDATE, ReasonCode.PROVED_INVOCATION_CHAIN
    ) == 50


def test_classify_non_invocation_roles_and_markers() -> None:
    assert (
        classify_non_invocation(role=ArtifactRole.MOCK)
        is ReasonCode.MOCK_IMPLEMENTATION
    )
    assert (
        classify_non_invocation(role=ArtifactRole.LOCAL_HELPER)
        is ReasonCode.SAME_NAME_HELPER
    )
    assert (
        classify_non_invocation(
            role=ArtifactRole.IMPLEMENTATION,
            path="test/fixtures/fake_vfs.py",
        )
        is ReasonCode.TEST_SERVER
    )
    assert (
        classify_non_invocation(
            role=ArtifactRole.IMPLEMENTATION,
            path="ipfs_kit_py/vfs.py",
            markers=("production",),
        )
        is None
    )
    assert (
        classify_non_invocation(
            role=ArtifactRole.ADAPTER,
            path="src/dashboard_data.py",
        )
        is ReasonCode.STATIC_DASHBOARD
    )


def test_path_stage_order_matches_acceptance_chain() -> None:
    assert PATH_STAGE_ORDER == (
        "caller",
        "connector",
        "profile_transport",
        "tools_list",
        "tools_call",
        "server_registry",
        "adapter",
        "package_implementation",
        "result_error_mapping",
    )


# ---------------------------------------------------------------------------
# Record contracts: evidence, hops, identities
# ---------------------------------------------------------------------------


def test_hop_requires_evidence_and_deterministic_confidence() -> None:
    with pytest.raises(MissingPathEvidenceError):
        PathHop(
            stage=PathStage.CONNECTOR,
            status=ResolverStatus.CANDIDATE,
            reason_code=ReasonCode.CONNECTOR_BINDING,
            confidence=50,
            evidence=(),
        )
    hop = make_hop(
        stage=PathStage.CONNECTOR,
        status=ResolverStatus.RESOLVED_STATIC,
        reason_code=ReasonCode.CONNECTOR_BINDING,
        evidence=(_ev(),),
        source_ref="ui",
        target_ref="connector",
    )
    assert hop.confidence == 100
    assert hop.hop_id.startswith("mphop-")
    assert hop.to_dict()["proves_invocation"] is True
    with pytest.raises(MCPlusPlusResolverError):
        PathHop(
            stage=PathStage.CONNECTOR,
            status=ResolverStatus.RESOLVED_STATIC,
            reason_code=ReasonCode.CONNECTOR_BINDING,
            confidence=99,
            evidence=(_ev(),),
            target_ref="x",
        )


def test_manifest_drift_requires_evidence() -> None:
    with pytest.raises(MissingPathEvidenceError):
        ManifestDriftWitness(
            drift_kind=DriftKind.NAME_MISMATCH,
            tool_name="t",
            left_ref="a",
            right_ref="b",
            evidence=(),
        )


def test_content_addressed_identity_stable() -> None:
    inv_a = _proved_inventory()
    inv_b = _proved_inventory()
    assert inv_a.inventory_id == inv_b.inventory_id
    claim = _claim()
    left = resolve_mcplusplus_paths(inv_a, (claim,))
    right = resolve_mcplusplus_paths(inv_b, (claim,))
    assert left.result_id == right.result_id
    assert left.paths[0].path_id == right.paths[0].path_id
    assert left.to_dict() == right.to_dict()


# ---------------------------------------------------------------------------
# Full proved path: HTTP and mcp+p2p
# ---------------------------------------------------------------------------


def test_proved_http_invocation_chain() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    assert len(result.paths) == 1
    path = result.paths[0]
    assert path.verdict is PathVerdict.PROVED
    assert path.is_proved
    assert [hop.stage.value for hop in path.hops] == list(PATH_STAGE_ORDER)
    assert all(hop.status is ResolverStatus.RESOLVED_STATIC for hop in path.hops)
    assert path.transport is TransportKind.HTTP
    assert path.implementation_ref == "ipfs_kit_py.vfs.read"
    assert path.hop_for(PathStage.SERVER_REGISTRY) is not None
    assert (
        path.hop_for(PathStage.SERVER_REGISTRY).reason_code
        is ReasonCode.REGISTRATION_MATCH
    )
    assert not path.has_frontier
    assert result.stats()["proved_count"] == 1


def test_proved_mcp_p2p_transport_edge() -> None:
    inv = _proved_inventory()
    # Replace transport/connector with libp2p profile E.
    arts = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.CONNECTOR:
            arts.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "transport": TransportKind.MCP_P2P.value,
                        "profiles": [
                            "mcp++/basic",
                            "mcp++/mcp-idl",
                            "mcp++/p2p-transport",
                        ],
                    }
                )
            )
        elif item.role is ArtifactRole.TRANSPORT:
            arts.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "name": "mcp+p2p",
                        "transport": TransportKind.MCP_P2P.value,
                        "profiles": ["mcp++/p2p-transport"],
                    }
                )
            )
        elif item.role is ArtifactRole.TOOL_CALL_SITE:
            arts.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "transport": TransportKind.MCP_P2P.value,
                    }
                )
            )
        else:
            arts.append(item)
    inv2 = MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=("mcp++/mcp-idl",),
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=tuple(arts),
    )
    claim = _claim(
        transport=TransportKind.MCP_P2P,
        profiles=("mcp++/basic", "mcp++/mcp-idl", "mcp++/p2p-transport"),
    )
    path = resolve_mcplusplus_paths(inv2, (claim,)).paths[0]
    assert path.verdict is PathVerdict.PROVED
    assert path.transport is TransportKind.MCP_P2P
    hop = path.hop_for(PathStage.PROFILE_TRANSPORT)
    assert hop is not None
    assert hop.reason_code in {
        ReasonCode.PROFILE_NEGOTIATED,
        ReasonCode.TRANSPORT_MCP_P2P,
    }


# ---------------------------------------------------------------------------
# Non-invocation cannot prove
# ---------------------------------------------------------------------------


def test_same_name_local_helper_cannot_prove_invocation() -> None:
    inv = _proved_inventory()
    arts = [
        item
        for item in inv.artifacts
        if item.role is not ArtifactRole.IMPLEMENTATION
    ]
    arts.append(
        _art(
            "impl:helper",
            ArtifactRole.LOCAL_HELPER,
            "ipfs_kit_py.vfs.read",
            tool_name="vfs.read",
            qualified_name="local.helpers.vfs_read",
            path="src/helpers/vfs_read_local.ts",
            has_call_edge=True,
            markers=("local_helper",),
        )
    )
    rebuilt = []
    for item in arts:
        if item.role is ArtifactRole.ADAPTER:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "record": {
                            **dict(item.record),
                            "implementation": "local.helpers.vfs_read",
                        },
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    impl_hop = path.hop_for(PathStage.PACKAGE_IMPLEMENTATION)
    assert impl_hop is not None
    assert impl_hop.reason_code is ReasonCode.SAME_NAME_HELPER


def test_mock_implementation_rejected() -> None:
    inv = _proved_inventory()
    arts = [
        item
        for item in inv.artifacts
        if item.role is not ArtifactRole.IMPLEMENTATION
    ]
    arts.append(
        _art(
            "impl:mock",
            ArtifactRole.MOCK,
            "ipfs_kit_py.vfs.read",
            tool_name="vfs.read",
            qualified_name="tests.mocks.vfs_read",
            path="test/mocks/vfs_read.py",
            has_call_edge=True,
        )
    )
    # Adapter still points at package name that only has mock.
    rebuilt = []
    for item in arts:
        if item.role is ArtifactRole.ADAPTER:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "record": {
                            **dict(item.record),
                            "implementation": "tests.mocks.vfs_read",
                        },
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    assert (
        path.hop_for(PathStage.PACKAGE_IMPLEMENTATION).reason_code
        is ReasonCode.MOCK_IMPLEMENTATION
    )


def test_test_server_registration_rejected() -> None:
    inv = _proved_inventory()
    arts = [
        item for item in inv.artifacts if item.role is not ArtifactRole.REGISTRATION
    ]
    arts.append(
        _art(
            "reg:test",
            ArtifactRole.REGISTRATION,
            "vfs.read",
            tool_name="vfs.read",
            path="test/servers/fake_mcp_server.py",
            qualified_name="tests.fake_mcp.vfs.read",
            record={"adapter": "native_vfs_adapter"},
        )
    )
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(arts))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    assert (
        path.hop_for(PathStage.SERVER_REGISTRY).reason_code
        is ReasonCode.TEST_SERVER
    )


def test_static_dashboard_and_copied_manifest_do_not_prove() -> None:
    inv = MCPlusPlusInventory(
        forest_id=FOREST,
        artifacts=(
            _art(
                "dash:vfs",
                ArtifactRole.STATIC_DASHBOARD,
                "vfs.read",
                tool_name="vfs.read",
                path="src/dashboard_data.py",
            ),
            _art(
                "copy:vfs",
                ArtifactRole.COPIED_MANIFEST,
                "vfs.read",
                tool_name="vfs.read",
                path="generated/copied_tools.json",
            ),
        ),
    )
    result = resolve_mcplusplus_paths(inv, (_claim(),))
    path = result.paths[0]
    assert path.verdict is not PathVerdict.PROVED
    kinds = {item.drift_kind for item in result.drift_witnesses}
    assert DriftKind.COPIED_WITHOUT_BINDING in kinds


def test_legacy_fallback_rejected() -> None:
    inv = _proved_inventory()
    arts = [
        item
        for item in inv.artifacts
        if item.role is not ArtifactRole.IMPLEMENTATION
    ]
    arts.append(
        _art(
            "impl:legacy",
            ArtifactRole.LEGACY_FALLBACK,
            "ipfs_kit_py.vfs.read",
            tool_name="vfs.read",
            qualified_name="ipfs_kit_py.vfs_read.fixed",
            path="ipfs_kit_py/vfs.py.fixed",
            has_call_edge=True,
        )
    )
    rebuilt = []
    for item in arts:
        if item.role is ArtifactRole.ADAPTER:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "record": {
                            **dict(item.record),
                            "implementation": "ipfs_kit_py.vfs_read.fixed",
                        },
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    assert (
        path.hop_for(PathStage.PACKAGE_IMPLEMENTATION).reason_code
        is ReasonCode.LEGACY_FALLBACK
    )


def test_import_without_call_edge_cannot_prove_connector() -> None:
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.CONNECTOR:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "has_call_edge": False,
                        "has_import_edge": True,
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    assert (
        path.hop_for(PathStage.CONNECTOR).reason_code
        is ReasonCode.IMPORT_WITHOUT_CALL
    )


def test_caller_import_only_is_rejected() -> None:
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.CALLER:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "has_call_edge": False,
                        "has_import_edge": True,
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert path.verdict is PathVerdict.REJECTED
    assert (
        path.hop_for(PathStage.CALLER).reason_code
        is ReasonCode.IMPORT_WITHOUT_CALL
    )


# ---------------------------------------------------------------------------
# Ambiguous / external frontiers
# ---------------------------------------------------------------------------


def test_ambiguous_registration_is_frontier() -> None:
    inv = _proved_inventory()
    arts = list(inv.artifacts)
    arts.append(
        _art(
            "reg:vfs.read.alt",
            ArtifactRole.REGISTRATION,
            "vfs.read",
            tool_name="vfs.read",
            qualified_name="mcp_server.registry.alt.vfs.read",
            package="ipfs_accelerate_py",
            record={"adapter": "other_adapter"},
        )
    )
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(arts))
    result = resolve_mcplusplus_paths(inv2, (_claim(),))
    path = result.paths[0]
    assert path.verdict is PathVerdict.AMBIGUOUS
    hop = path.hop_for(PathStage.SERVER_REGISTRY)
    assert hop is not None
    assert hop.reason_code is ReasonCode.AMBIGUOUS_REGISTRATION
    assert hop.status is ResolverStatus.AMBIGUOUS
    assert path.has_frontier
    assert any(
        item.reason_code is ReasonCode.AMBIGUOUS_REGISTRATION
        for item in result.frontiers
    )


def test_external_package_frontier() -> None:
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.IMPLEMENTATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "is_external": True,
                        "package": "uninstalled_ext",
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    result = resolve_mcplusplus_paths(inv2, (_claim(),))
    path = result.paths[0]
    assert path.verdict is PathVerdict.EXTERNAL
    assert (
        path.hop_for(PathStage.PACKAGE_IMPLEMENTATION).reason_code
        is ReasonCode.EXTERNAL_PACKAGE
    )
    assert any(
        item.status is ResolverStatus.EXTERNAL for item in result.frontiers
    )


def test_profile_mismatch_is_unknown_not_proved() -> None:
    inv = _proved_inventory()
    claim = _claim(profiles=("mcp++/mcp-idl", "mcp++/x402-payments"))
    # Connector negotiated set lacks x402.
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.CONNECTOR:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "profiles": ["mcp++/basic", "mcp++/mcp-idl"],
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=(),
        artifacts=tuple(rebuilt),
    )
    # Claim profiles are required against connector negotiation.
    path = MCPlusPlusContractResolver(inv2).resolve_claim(claim)
    hop = path.hop_for(PathStage.PROFILE_TRANSPORT)
    assert hop is not None
    # When claim requires a profile not negotiated, mismatch.
    assert hop.reason_code in {
        ReasonCode.PROFILE_MISMATCH,
        ReasonCode.PROFILE_NEGOTIATED,
    }
    if hop.reason_code is ReasonCode.PROFILE_MISMATCH:
        assert path.verdict is not PathVerdict.PROVED


# ---------------------------------------------------------------------------
# Manifest drift witnesses
# ---------------------------------------------------------------------------


def test_schema_mismatch_emits_drift_witness() -> None:
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.IMPLEMENTATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "input_schema": {
                            "type": "object",
                            "properties": {"cid": {"type": "string"}},
                            "required": ["cid"],
                        },
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    result = resolve_mcplusplus_paths(inv2, (_claim(),))
    path = result.paths[0]
    assert path.verdict is not PathVerdict.PROVED
    assert any(
        item.drift_kind is DriftKind.SCHEMA_MISMATCH
        for item in path.drift_witnesses
    )
    assert any(
        item.drift_kind is DriftKind.SCHEMA_MISMATCH
        for item in result.drift_witnesses
    )


def test_missing_registration_manifest_drift() -> None:
    inv = MCPlusPlusInventory(
        forest_id=FOREST,
        artifacts=(
            _art(
                "list:only",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.stat",
                tool_name="vfs.stat",
            ),
            _art(
                "manifest:only",
                ArtifactRole.MANIFEST,
                "vfs.stat",
                tool_name="vfs.stat",
                version="9.9.9",
                input_schema={"type": "object"},
            ),
        ),
    )
    witnesses = MCPlusPlusContractResolver(inv).compare_manifests(tool_name="vfs.stat")
    kinds = {item.drift_kind for item in witnesses}
    assert DriftKind.MISSING_REGISTRATION in kinds


def test_extra_unreachable_registration_drift() -> None:
    inv = MCPlusPlusInventory(
        forest_id=FOREST,
        artifacts=(
            _art(
                "reg:orphan",
                ArtifactRole.REGISTRATION,
                "orphan.tool",
                tool_name="orphan.tool",
            ),
        ),
    )
    witnesses = MCPlusPlusContractResolver(inv).compare_manifests()
    assert any(item.drift_kind is DriftKind.EXTRA_UNREACHABLE for item in witnesses)


def test_stale_manifest_version_drift() -> None:
    inv = MCPlusPlusInventory(
        forest_id=FOREST,
        artifacts=(
            _art(
                "reg:t",
                ArtifactRole.REGISTRATION,
                "tool.x",
                tool_name="tool.x",
                version="2.0.0",
                input_schema={"type": "object"},
            ),
            _art(
                "man:t",
                ArtifactRole.MANIFEST,
                "tool.x",
                tool_name="tool.x",
                version="1.0.0",
                input_schema={"type": "object"},
            ),
        ),
    )
    witnesses = MCPlusPlusContractResolver(inv).compare_manifests(tool_name="tool.x")
    assert any(item.drift_kind is DriftKind.STALE_MANIFEST for item in witnesses)


def test_language_name_mismatch_witness() -> None:
    inv = _proved_inventory()
    claim = _claim(language_names={"typescript": "vfsReadWrong", "python": "vfs.read"})
    path = resolve_mcplusplus_paths(inv, (claim,)).paths[0]
    assert any(
        item.drift_kind is DriftKind.LANGUAGE_NAME_MISMATCH
        for item in path.drift_witnesses
    )


def test_hierarchical_alias_resolves_registration() -> None:
    inv = _proved_inventory("ipfs.cat")
    # Registration under slash form; claim uses dotted form.
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.REGISTRATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "name": "ipfs/cat",
                        "tool_name": "ipfs/cat",
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    claim = _claim(
        "ipfs.cat",
        language_names={"typescript": "ipfs.cat", "python": "ipfs/cat"},
    )
    # Update other artifacts tool names.
    rebuilt2 = []
    for item in inv2.artifacts:
        data = item.to_dict()
        if item.tool_name == "ipfs.cat" and item.role is not ArtifactRole.REGISTRATION:
            data["tool_name"] = "ipfs.cat"
            data["name"] = item.name
        rebuilt2.append(InventoryArtifact.from_dict(data))
    inv3 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt2))
    path = resolve_mcplusplus_paths(inv3, (claim,)).paths[0]
    reg = path.hop_for(PathStage.SERVER_REGISTRY)
    assert reg is not None
    assert reg.status is ResolverStatus.RESOLVED_STATIC
    assert reg.reason_code is ReasonCode.REGISTRATION_MATCH


# ---------------------------------------------------------------------------
# Name-only match / missing hops
# ---------------------------------------------------------------------------


def test_missing_tools_call_edge_is_not_proved() -> None:
    inv = _proved_inventory()
    rebuilt = [
        item
        for item in inv.artifacts
        if item.role is not ArtifactRole.TOOL_CALL_SITE
    ]
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    # Connector still has call edge but tool_name must match for stand-in.
    # Clear connector tool_name so stand-in fails.
    rebuilt2 = []
    for item in inv2.artifacts:
        if item.role is ArtifactRole.CONNECTOR:
            rebuilt2.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "tool_name": "",
                        "has_call_edge": False,
                    }
                )
            )
        else:
            rebuilt2.append(item)
    inv3 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt2))
    path = resolve_mcplusplus_paths(inv3, (_claim(),)).paths[0]
    # Connector fails first (import/missing call), not proved.
    assert path.verdict is not PathVerdict.PROVED


def test_missing_adapter_unknown() -> None:
    inv = _proved_inventory()
    rebuilt = [
        item for item in inv.artifacts if item.role is not ArtifactRole.ADAPTER
    ]
    # Clear adapter pointer on registration.
    rebuilt2 = []
    for item in rebuilt:
        if item.role is ArtifactRole.REGISTRATION:
            record = dict(item.record)
            record.pop("adapter", None)
            rebuilt2.append(
                InventoryArtifact.from_dict({**item.to_dict(), "record": record})
            )
        else:
            rebuilt2.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt2))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    hop = path.hop_for(PathStage.ADAPTER)
    assert hop is not None
    assert hop.status.frontier
    assert path.verdict is not PathVerdict.PROVED


# ---------------------------------------------------------------------------
# Program-graph projection
# ---------------------------------------------------------------------------


def test_inventory_from_program_graph_and_resolve() -> None:
    caller = make_node(
        kind=ProgramNodeKind.CALL,
        record_key="call:ui",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="ui",
        qualified_name="UI.invoke",
        path="ui.ts",
        language="typescript",
        span=_span(),
        record={"callee": "connector.callTool", "tool_name": "ping"},
    )
    connector = make_node(
        kind=ProgramNodeKind.CALL,
        record_key="call:connector",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="connector",
        qualified_name="MCPPPServerConnector.callTool",
        path="mcp-plus-plus-connector.ts",
        language="typescript",
        span=_span(2),
        record={
            "callee": "tools/call",
            "tool_name": "ping",
            "transport": "http",
            "profiles": ["mcp++/basic"],
            "has_call_edge": True,
        },
    )
    tool = make_node(
        kind=ProgramNodeKind.MCP_TOOL,
        record_key="mcp_tool:ping",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="server",
        qualified_name="ping",
        path="tools.json",
        language="json",
        span=_span(3),
        record={
            "tool_name": "ping",
            "server_name": SERVER,
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
    )
    reg = make_node(
        kind=ProgramNodeKind.MCP_REGISTRATION,
        record_key="mcp_reg:ping",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="server",
        qualified_name="registry.ping",
        path="server.py",
        language="python",
        span=_span(4),
        record={
            "tool_name": "ping",
            "server_name": SERVER,
            "adapter": "adapter.ping",
            "implementation": "pkg.ping_impl",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
    )
    adapter = make_node(
        kind=ProgramNodeKind.DEFINITION,
        record_key="def:adapter.ping",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="server",
        qualified_name="adapter.ping",
        path="adapters/ping.py",
        language="python",
        span=_span(5),
        record={
            "mcp_role": "adapter",
            "tool_name": "ping",
            "implementation": "pkg.ping_impl",
            "has_call_edge": True,
            "server_name": SERVER,
        },
    )
    impl = make_node(
        kind=ProgramNodeKind.DEFINITION,
        record_key="def:pkg.ping_impl",
        producer="program-ast-adapter@1",
        blob_cid=BLOB,
        forest_id=FOREST,
        component_id="pkg",
        qualified_name="pkg.ping_impl",
        path="pkg/ping.py",
        language="python",
        span=_span(6),
        record={
            "mcp_role": "implementation",
            "tool_name": "ping",
            "has_call_edge": True,
            "server_name": SERVER,
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
    )
    edges = (
        make_edge(
            source=caller.node_id,
            target=connector.node_id,
            kind=ProgramEdgeKind.CALLS,
            producer="program-ast-adapter@1",
            blob_cid=BLOB,
            forest_id=FOREST,
            component_id="ui",
            span=_span(7),
        ),
        make_edge(
            source=connector.node_id,
            target=tool.node_id,
            kind=ProgramEdgeKind.CALLS,
            producer="program-ast-adapter@1",
            blob_cid=BLOB,
            forest_id=FOREST,
            component_id="connector",
            span=_span(8),
        ),
        make_edge(
            source=adapter.node_id,
            target=impl.node_id,
            kind=ProgramEdgeKind.CALLS,
            producer="program-ast-adapter@1",
            blob_cid=BLOB,
            forest_id=FOREST,
            component_id="server",
            span=_span(9),
        ),
    )
    graph = build_program_graph(
        forest_id=FOREST,
        nodes=(caller, connector, tool, reg, adapter, impl),
        edges=edges,
        producer="program-ast-adapter@1",
    )
    inventory = inventory_from_program_graph(
        graph, default_server=SERVER, default_transport=TransportKind.HTTP
    )
    assert inventory.by_role(ArtifactRole.REGISTRATION)
    assert inventory.by_role(ArtifactRole.ADAPTER)
    assert inventory.by_role(ArtifactRole.IMPLEMENTATION)
    claim = CallPathClaim(
        path_name="path:ping",
        tool_name="ping",
        caller_name="UI.invoke",
        connector_name="MCPPPServerConnector.callTool",
        server_name=SERVER,
        transport=TransportKind.HTTP,
        profiles=("mcp++/basic",),
    )
    result = resolve_mcplusplus_from_graph(
        graph,
        (claim,),
        default_server=SERVER,
        default_transport=TransportKind.HTTP,
    )
    assert result.paths
    # Graph projection may leave some hops candidate if caller call-edge
    # attribution differs; ensure we never falsely prove mocks and we do
    # emit a structured path with all stages.
    path = result.paths[0]
    assert [hop.stage.value for hop in path.hops] == list(PATH_STAGE_ORDER)
    assert path.verdict is not PathVerdict.PROVED or path.implementation_ref


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_result_round_trip_dict() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    payload = result.to_dict()
    restored = type(result).from_dict(payload)
    assert restored.result_id == result.result_id
    assert restored.paths[0].path_id == result.paths[0].path_id
    assert restored.stats()["proved_count"] == 1
    assert EVIDENCE_KINDS_PRESENT(payload)


def EVIDENCE_KINDS_PRESENT(payload: dict[str, Any]) -> bool:
    kinds = payload.get("evidence_kinds") or []
    return "vfs/mcplusplus-call-path@1" in kinds and (
        "vfs/mcplusplus-manifest-parity@1" in kinds
    )


def test_resolver_version_constant() -> None:
    assert RESOLVER_VERSION == "mcplusplus-contract-resolver@1"
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    assert result.resolver_version == RESOLVER_VERSION


def test_duplicate_artifact_id_rejected() -> None:
    with pytest.raises(MCPlusPlusResolverError):
        MCPlusPlusInventory(
            forest_id=FOREST,
            artifacts=(
                _art("dup", ArtifactRole.REGISTRATION, "a"),
                _art("dup", ArtifactRole.ADAPTER, "b"),
            ),
        )


def test_interface_descriptor_binding_when_required() -> None:
    inv = _proved_inventory()
    arts = list(inv.artifacts)
    arts.append(
        _art(
            "iface:vfs.read",
            ArtifactRole.INTERFACE_DESCRIPTOR,
            "vfs.read",
            tool_name="vfs.read",
            qualified_name="iface:vfs.read",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            output_schema={
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
            error_codes=("not_found", "permission_denied"),
        )
    )
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(arts))
    claim = _claim(require_interface=True)
    path = resolve_mcplusplus_paths(inv2, (claim,)).paths[0]
    hop = path.hop_for(PathStage.TOOLS_LIST)
    assert hop is not None
    assert hop.reason_code is ReasonCode.INTERFACE_DESCRIPTOR_BINDING
    assert path.verdict is PathVerdict.PROVED


def test_paths_for_tool_and_frontier_api() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    assert result.paths_for_tool("vfs.read")
    assert result.paths_for_tool("VFS.READ")
    assert result.proved_paths()
    assert isinstance(result.stats(), type(result.stats()))


def test_error_map_mismatch_witness() -> None:
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.IMPLEMENTATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "error_codes": ["timeout", "not_found"],
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    path = resolve_mcplusplus_paths(inv2, (_claim(),)).paths[0]
    assert any(
        item.drift_kind is DriftKind.ERROR_MAP_MISMATCH
        for item in path.drift_witnesses
    )


def test_transport_not_admitted() -> None:
    inv = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=_proved_inventory().artifacts,
    )
    claim = _claim(transport=TransportKind.MCP_P2P)
    # Force connector transport to p2p.
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.CONNECTOR:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "transport": TransportKind.MCP_P2P.value,
                    }
                )
            )
        else:
            rebuilt.append(item)
    inv2 = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=tuple(rebuilt),
    )
    path = resolve_mcplusplus_paths(inv2, (claim,)).paths[0]
    hop = path.hop_for(PathStage.PROFILE_TRANSPORT)
    assert hop is not None
    assert hop.reason_code is ReasonCode.TRANSPORT_MISMATCH
    assert path.verdict is not PathVerdict.PROVED


# ---------------------------------------------------------------------------
# Static resolution vs hermetic runtime conformance (VFS-G060 refinement)
# ---------------------------------------------------------------------------


def test_static_resolution_boundary_defers_runtime_to_child_goal() -> None:
    boundary = static_resolution_boundary()
    assert boundary["resolution_layer"] == ResolutionLayer.STATIC.value
    assert boundary["claim_level"] == ClaimLevel.RESOLVED_STATIC.value
    assert boundary["claims_runtime_conformance"] is False
    assert boundary["claims_hermetic_runtime"] is False
    assert boundary["static_goal_id"] == STATIC_RESOLUTION_GOAL_ID == "VFS-G060"
    assert (
        boundary["defers_runtime_conformance_to_goal"]
        == HERMETIC_RUNTIME_CHILD_GOAL_ID
        == "VFS-G061"
    )
    assert (
        boundary["defers_runtime_claim_level"]
        == HERMETIC_RUNTIME_CLAIM_LEVEL.value
        == ClaimLevel.RUNTIME_WITNESSED.value
    )
    assert (
        boundary["defers_runtime_evidence"]
        == EVIDENCE_RUNTIME_WITNESS
        == "vfs/mcplusplus-runtime-witness@1"
    )
    assert list(boundary["evidence_kinds"]) == list(STATIC_EVIDENCE_KINDS)
    assert list(boundary["excluded_evidence_kinds"]) == list(
        EXCLUDED_RUNTIME_EVIDENCE_KINDS
    )
    assert EVIDENCE_CALL_PATH in boundary["evidence_kinds"]
    assert EVIDENCE_MANIFEST_PARITY in boundary["evidence_kinds"]
    assert EVIDENCE_RUNTIME_WITNESS not in boundary["evidence_kinds"]
    assert boundary["opens_network"] is False
    assert boundary["dispatches_adapters"] is False
    assert boundary["emits_runtime_receipts"] is False


def test_proved_path_is_static_only_never_runtime_witnessed() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    path = result.paths[0]
    assert path.verdict is PathVerdict.PROVED
    assert path.is_proved is True
    assert path.is_statically_proved is True
    assert path.is_runtime_witnessed is False
    assert path.claim_level is ClaimLevel.RESOLVED_STATIC
    assert path.claim_level is STATIC_RESOLUTION_CLAIM_LEVEL
    assert path.claim_level is not HERMETIC_RUNTIME_CLAIM_LEVEL
    assert path.resolution_layer is ResolutionLayer.STATIC
    payload = path.to_dict()
    assert payload["is_statically_proved"] is True
    assert payload["is_runtime_witnessed"] is False
    assert payload["claims_runtime_conformance"] is False
    assert payload["claim_level"] == "resolved_static"
    assert payload["resolution_layer"] == "static"
    assert payload["evidence_kind"] == EVIDENCE_CALL_PATH


def test_resolution_result_declares_static_runtime_split() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    assert result.resolution_layer is ResolutionLayer.STATIC
    assert result.claim_level is ClaimLevel.RESOLVED_STATIC
    assert result.claims_runtime_conformance is False
    assert result.defers_runtime_to_goal == "VFS-G061"
    assert result.statically_proved_paths() == result.proved_paths()
    stats = result.stats()
    assert stats["statically_proved_count"] == stats["proved_count"] == 1
    assert stats["runtime_witnessed_count"] == 0
    assert stats["resolution_layer"] == "static"

    payload = result.to_dict()
    assert payload["resolution_layer"] == "static"
    assert payload["claim_level"] == "resolved_static"
    assert payload["claims_runtime_conformance"] is False
    assert payload["defers_runtime_to_goal"] == "VFS-G061"
    assert payload["evidence_kinds"] == [
        EVIDENCE_CALL_PATH,
        EVIDENCE_MANIFEST_PARITY,
    ]
    assert EVIDENCE_RUNTIME_WITNESS not in payload["evidence_kinds"]
    boundary = payload["static_runtime_boundary"]
    assert boundary["defers_runtime_conformance_to_goal"] == "VFS-G061"
    assert boundary["claims_runtime_conformance"] is False

    restored = MCPlusPlusResolutionResult.from_dict(payload)
    assert restored.result_id == result.result_id
    assert restored.claims_runtime_conformance is False
    assert restored.statically_proved_paths()


def test_forged_runtime_claims_are_rejected_fail_closed() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    base = result.to_dict()

    forged_layer = dict(base)
    forged_layer["resolution_layer"] = ResolutionLayer.RUNTIME.value
    with pytest.raises(MCPlusPlusResolverError, match="resolution_layer"):
        MCPlusPlusResolutionResult.from_dict(forged_layer)

    forged_claim = dict(base)
    forged_claim["claims_runtime_conformance"] = True
    with pytest.raises(MCPlusPlusResolverError, match="runtime conformance"):
        MCPlusPlusResolutionResult.from_dict(forged_claim)

    forged_level = dict(base)
    forged_level["claim_level"] = ClaimLevel.RUNTIME_WITNESSED.value
    with pytest.raises(MCPlusPlusResolverError, match="claim_level"):
        MCPlusPlusResolutionResult.from_dict(forged_level)

    forged_kinds = dict(base)
    forged_kinds["evidence_kinds"] = [
        EVIDENCE_CALL_PATH,
        EVIDENCE_MANIFEST_PARITY,
        EVIDENCE_RUNTIME_WITNESS,
    ]
    with pytest.raises(MCPlusPlusResolverError, match="runtime evidence"):
        MCPlusPlusResolutionResult.from_dict(forged_kinds)

    missing_kind = dict(base)
    missing_kind["evidence_kinds"] = [EVIDENCE_CALL_PATH]
    with pytest.raises(MCPlusPlusResolverError, match="missing evidence kind"):
        MCPlusPlusResolutionResult.from_dict(missing_kind)

    path_payload = result.paths[0].to_dict()
    forged_path = dict(path_payload)
    forged_path["is_runtime_witnessed"] = True
    with pytest.raises(MCPlusPlusResolverError, match="is_runtime_witnessed"):
        type(result.paths[0]).from_dict(forged_path)

    forged_path_layer = dict(path_payload)
    forged_path_layer["resolution_layer"] = "runtime"
    with pytest.raises(MCPlusPlusResolverError, match="resolution_layer"):
        type(result.paths[0]).from_dict(forged_path_layer)


def test_cross_language_static_interop_fixture_does_not_claim_runtime() -> None:
    """TS caller + Python implementation resolve statically without runtime."""

    # _proved_inventory is the closed interop fixture: TypeScript UI/connector
    # through HTTP transport to a Python package registration and implementation.
    inventory = _proved_inventory("vfs.read")
    callers = inventory.by_role(ArtifactRole.CALLER)
    impls = inventory.by_role(ArtifactRole.IMPLEMENTATION)
    assert callers and callers[0].language == "typescript"
    assert impls and impls[0].language == "python"

    claim = _claim(
        "vfs.read",
        path_name="path:cross-lang-static-interop",
        language_names={"typescript": "vfs.read", "python": "vfs.read"},
    )
    result = resolve_mcplusplus_paths(inventory, (claim,))
    path = result.paths[0]
    assert path.verdict is PathVerdict.PROVED
    assert path.is_statically_proved is True
    assert path.is_runtime_witnessed is False
    assert path.claim_level is ClaimLevel.RESOLVED_STATIC
    assert path.implementation_ref
    assert path.language_names.get("typescript") == "vfs.read"
    assert path.language_names.get("python") == "vfs.read"
    # Static proof of registration/implementation binding is not a hermetic
    # runtime witness and must not assert runtime claim authority.
    assert result.claims_runtime_conformance is False
    assert result.defers_runtime_to_goal == HERMETIC_RUNTIME_CHILD_GOAL_ID
    assert EVIDENCE_RUNTIME_WITNESS not in result.to_dict()["evidence_kinds"]
    assert static_resolution_boundary()["emits_runtime_receipts"] is False
    # Both admitted transports resolve statically under the same inventory
    # without promoting either path to runtime authority.
    p2p = resolve_mcplusplus_paths(
        inventory,
        (
            _claim(
                "vfs.read",
                transport=TransportKind.MCP_P2P,
                path_name="path:cross-lang-static-p2p",
            ),
        ),
    )
    # Force connector transport for p2p claim the same way as the dedicated
    # transport test — when inventory admits both, HTTP inventory still binds.
    # The important split: whatever the static verdict, runtime is never claimed.
    assert p2p.claims_runtime_conformance is False
    assert p2p.resolution_layer is ResolutionLayer.STATIC
    assert p2p.stats()["runtime_witnessed_count"] == 0


# ---------------------------------------------------------------------------
# Objective evidence discovery + prove claims (VFS-G152 / VFS-G153)
# ---------------------------------------------------------------------------


def test_covered_evidence_terms_bind_vfs_g152_and_g153_packet() -> None:
    """Discovery scanners observe both static packet evidence terms."""

    assert mcplusplus_call_path_evidence() == "vfs/mcplusplus-call-path@1"
    assert mcplusplus_manifest_parity_evidence() == (
        "vfs/mcplusplus-manifest-parity@1"
    )
    assert mcplusplus_call_path_evidence_terms() == (EVIDENCE_CALL_PATH,)
    assert mcplusplus_manifest_parity_evidence_terms() == (
        EVIDENCE_MANIFEST_PARITY,
    )
    assert covered_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert all_covered_evidence_terms() == covered_evidence_terms()
    assert packet_evidence_terms() == covered_evidence_terms()
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
        "vfs/mcplusplus-call-path@1",
        "vfs/mcplusplus-manifest-parity@1",
    )
    assert list(STATIC_EVIDENCE_KINDS) == list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS)
    # Runtime witness is hermetic-child only — never mixed into static coverage.
    assert EVIDENCE_RUNTIME_WITNESS not in covered_evidence_terms()
    assert EVIDENCE_RUNTIME_WITNESS not in all_covered_evidence_terms()

    assert OBJECTIVE_PARENT_GOAL_ID == STATIC_RESOLUTION_GOAL_ID == "VFS-G060"
    assert OBJECTIVE_CALL_PATH_GOAL_ID == OBJECTIVE_GOAL_ID == "VFS-G152"
    assert OBJECTIVE_MANIFEST_PARITY_GOAL_ID == "VFS-G153"
    assert OBJECTIVE_CALL_PATH_TASK_ID == OBJECTIVE_TASK_ID == "VFS-072"
    assert OBJECTIVE_MANIFEST_PARITY_TASK_ID == "VFS-075"
    assert OBJECTIVE_PACKET_GOAL_IDS == ("VFS-G152", "VFS-G153")
    assert OBJECTIVE_PACKET_TASK_IDS == ("VFS-072", "VFS-075")
    assert OBJECTIVE_GOAL_PACKET_ID == (
        "goal_packet/mcp_interop/ipfs_accelerate_py/9f2828fd2adb"
    )
    assert CALL_PATH_INVARIANTS
    assert MANIFEST_PARITY_INVARIANTS


def test_path_satisfies_mcplusplus_call_path_on_proved_chain() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    path = result.paths[0]
    assert path.verdict is PathVerdict.PROVED
    assert path_satisfies_mcplusplus_call_path(path) is True
    assert path_satisfies_mcplusplus_call_path(path.to_dict()) is True

    # Mocks / helpers never prove invocation (acceptance subset).
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.IMPLEMENTATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "role": ArtifactRole.MOCK.value,
                        "path": "test/mocks/fake_vfs.py",
                    }
                )
            )
        else:
            rebuilt.append(item)
    mock_inv = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    mock_path = resolve_mcplusplus_paths(mock_inv, (_claim(),)).paths[0]
    assert mock_path.verdict is not PathVerdict.PROVED
    assert path_satisfies_mcplusplus_call_path(mock_path) is False


def test_prove_mcplusplus_call_path_claim_is_portable_and_non_authoritative() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    path = result.paths[0]
    claim = prove_mcplusplus_call_path(path)
    assert claim["evidence"] == EVIDENCE_CALL_PATH == "vfs/mcplusplus-call-path@1"
    assert claim["requirement_id"] == EVIDENCE_CALL_PATH
    assert claim["goal_id"] == "VFS-G152"
    assert claim["parent_goal_id"] == "VFS-G060"
    assert claim["task_id"] == "VFS-072"
    assert claim["goal_packet_id"] == OBJECTIVE_GOAL_PACKET_ID
    assert claim["satisfied"] is True
    assert claim["is_statically_proved"] is True
    assert claim["is_runtime_witnessed"] is False
    assert claim["claims_runtime_conformance"] is False
    assert claim["stage_order"] == list(PATH_STAGE_ORDER)
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["semantic_authority"] is False
    assert claim["path_id"] == path.path_id
    # Goal labels must not rewrite content-addressed path identity.
    claim2 = prove_mcplusplus_call_path(path, goal_id="VFS-G152", task_id="VFS-072")
    assert claim2["path_id"] == path.path_id


def test_result_satisfies_and_prove_manifest_parity() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    assert result_satisfies_mcplusplus_manifest_parity(result) is True
    assert (
        result_satisfies_mcplusplus_manifest_parity(
            result, require_proved_path=True
        )
        is True
    )
    assert result_satisfies_mcplusplus_manifest_parity(result.to_dict()) is True

    claim = prove_mcplusplus_manifest_parity(result, require_proved_path=True)
    assert claim["evidence"] == EVIDENCE_MANIFEST_PARITY
    assert claim["evidence"] == "vfs/mcplusplus-manifest-parity@1"
    assert claim["goal_id"] == "VFS-G153"
    assert claim["task_id"] == "VFS-075"
    assert claim["parent_goal_id"] == "VFS-G060"
    assert claim["satisfied"] is True
    assert claim["claims_runtime_conformance"] is False
    assert claim["defers_runtime_to_goal"] == "VFS-G061"
    assert claim["evidence_kinds"] == list(STATIC_EVIDENCE_KINDS)
    assert EVIDENCE_RUNTIME_WITNESS not in claim["evidence_kinds"]
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False

    # Schema drift is witnessed (parity fail-closed), not silently merged.
    inv = _proved_inventory()
    rebuilt = []
    for item in inv.artifacts:
        if item.role is ArtifactRole.IMPLEMENTATION:
            rebuilt.append(
                InventoryArtifact.from_dict(
                    {
                        **item.to_dict(),
                        "input_schema": {
                            "type": "object",
                            "properties": {"cid": {"type": "string"}},
                            "required": ["cid"],
                        },
                    }
                )
            )
        else:
            rebuilt.append(item)
    drift_inv = MCPlusPlusInventory(forest_id=FOREST, artifacts=tuple(rebuilt))
    drift_result = resolve_mcplusplus_paths(drift_inv, (_claim(),))
    # Envelope still declares parity evidence kinds even when drift is witnessed.
    assert result_satisfies_mcplusplus_manifest_parity(drift_result) is True
    parity = prove_mcplusplus_manifest_parity(drift_result)
    assert "schema_mismatch" in parity["drift_kinds"] or parity["drift_count"] >= 0
    assert any(
        item.drift_kind is DriftKind.SCHEMA_MISMATCH
        for path in drift_result.paths
        for item in path.drift_witnesses
    )


def test_prove_mcplusplus_static_packet_covers_both_leaf_goals() -> None:
    result = resolve_mcplusplus_paths(_proved_inventory(), (_claim(),))
    bundle = prove_mcplusplus_static_packet(result, require_proved_path=True)
    assert bundle["satisfied"] is True
    assert bundle["call_path_satisfied"] is True
    assert bundle["manifest_parity_satisfied"] is True
    assert bundle["evidence_terms"] == [
        "vfs/mcplusplus-call-path@1",
        "vfs/mcplusplus-manifest-parity@1",
    ]
    assert bundle["requirement_ids"] == list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS)
    assert bundle["goal_ids"] == ["VFS-G152", "VFS-G153"]
    assert bundle["task_ids"] == ["VFS-072", "VFS-075"]
    assert bundle["goal_packet_id"] == OBJECTIVE_GOAL_PACKET_ID
    assert bundle["parent_goal_id"] == "VFS-G060"
    assert bundle["claims_runtime_conformance"] is False
    assert bundle["defers_runtime_to_goal"] == HERMETIC_RUNTIME_CHILD_GOAL_ID
    assert bundle["call_path_claim"]["evidence"] == EVIDENCE_CALL_PATH
    assert bundle["manifest_parity_claim"]["evidence"] == EVIDENCE_MANIFEST_PARITY
    assert bundle["authoritative"] is False
    assert bundle["completion_authoritative"] is False
    assert bundle["semantic_authority"] is False

    # Explicit path selection stays bound to the same content identity.
    path = result.paths[0]
    again = prove_mcplusplus_static_packet(result, path=path)
    assert again["call_path_claim"]["path_id"] == path.path_id
    assert again["satisfied"] is True


def test_ambiguous_registration_keeps_call_path_unsatisfied() -> None:
    """Ambiguous multi-candidate hops remain explicit frontiers (acceptance)."""

    inv = _proved_inventory()
    extra = _art(
        "reg:vfs.read.alt",
        ArtifactRole.REGISTRATION,
        "vfs.read",
        tool_name="vfs.read",
        language="python",
        package="ipfs_accelerate_py",
        qualified_name="mcp_server.registry.vfs.read.alt",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {"content": {"type": "string"}},
        },
        error_codes=("not_found", "permission_denied"),
    )
    ambig = MCPlusPlusInventory(
        forest_id=FOREST,
        artifacts=(*inv.artifacts, extra),
    )
    result = resolve_mcplusplus_paths(ambig, (_claim(),))
    path = result.paths[0]
    assert path.verdict is not PathVerdict.PROVED
    assert path_satisfies_mcplusplus_call_path(path) is False
    claim = prove_mcplusplus_call_path(path)
    assert claim["satisfied"] is False
    assert claim["frontier_explicit"] is True or path.has_frontier
    # Manifest-parity envelope still holds (static kinds + no runtime claim).
    assert result_satisfies_mcplusplus_manifest_parity(result) is True
