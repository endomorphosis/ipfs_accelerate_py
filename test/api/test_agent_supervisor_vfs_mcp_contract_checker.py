"""Tests for end-to-end VFS manifest/SDK/MCP/MCP++ parity checking (VFS-028)."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.mcplusplus_contract_resolver import (
    ArtifactRole,
    CallPathClaim,
    DriftKind,
    MCPlusPlusContractResolver,
    MCPlusPlusInventory,
    PathVerdict,
    TransportKind,
    make_artifact,
    make_evidence,
    schema_fingerprint,
)
from ipfs_accelerate_py.agent_supervisor.mcplusplus_runtime_witness import (
    BackendAvailability,
    CallObservation,
    CallRequest,
    CapabilityNegotiationRecord,
    CleanupStatus,
    ImplementationKind,
    RuntimeWitness,
    RuntimeWitnessReceipt,
    ToolDiscoveryRecord,
    ValidationVerdict,
    WitnessOutcome,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import ClaimLevel
from ipfs_accelerate_py.agent_supervisor.program_graph import SourceSpan
from ipfs_accelerate_py.agent_supervisor.vfs_contract_pack import (
    canonical_vfs_contract_pack,
)
from ipfs_accelerate_py.agent_supervisor.vfs_mcp_contract_checker import (
    CHECKER_AUTHORIZES_REPAIR,
    CHECKER_IS_COMPLETION_EVIDENCE,
    CHECKER_VERSION,
    EVIDENCE_VFS_MCP_PARITY,
    GOAL_ID,
    REQUIRED_PROVED_STAGES,
    ParityFindingKind,
    ParitySurface,
    ReportVerdict,
    ToolParityVerdict,
    VfsMcpCheckerError,
    VfsMcpContractChecker,
    VfsMcpParityReport,
    call_path_is_proved,
    check_vfs_mcp_parity,
    compare_tool_surfaces,
    discover_tool_names,
    finding_kinds,
    make_parity_witness,
    make_surface_view,
    parity_surfaces,
    report_content_identity,
    text_names_agree,
)


FOREST = "forest:test-vfs-028"
BLOB = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SERVER = "ipfs-accelerate-mcp++"
SCHEMA_PATH = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}
SCHEMA_CONTENT = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
}
SCHEMA_PATH_ALT = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
    "required": ["path"],
}


def _span(line: int = 1) -> SourceSpan:
    return SourceSpan(
        line_start=line, column_start=0, line_end=line, column_end=8
    )


def _art(
    artifact_id: str,
    role: ArtifactRole | str,
    name: str,
    **kwargs: Any,
):
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


def _proved_chain_artifacts(tool: str = "vfs.read") -> tuple:
    """Full SwissKnife → registration → implementation inventory slice."""

    return (
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
            profiles=("mcp++/basic", "mcp++/mcp-idl", "mcp++/p2p-transport"),
            has_call_edge=True,
        ),
        _art(
            "transport:http",
            ArtifactRole.TRANSPORT,
            "http-jsonrpc",
            tool_name=tool,
            transport=TransportKind.HTTP,
            profiles=("mcp++/basic", "mcp++/mcp-idl"),
        ),
        _art(
            f"list:{tool}",
            ArtifactRole.TOOL_LIST_ENTRY,
            tool,
            tool_name=tool,
            language="json",
            version="1.0.0",
            input_schema=SCHEMA_PATH,
            output_schema=SCHEMA_CONTENT,
            error_codes=("not_found", "permission_denied"),
        ),
        _art(
            f"call:{tool}",
            ArtifactRole.TOOL_CALL_SITE,
            f"tools/call:{tool}",
            tool_name=tool,
            language="typescript",
            qualified_name="MCPPPServerConnector.callTool",
            path="src/services/mcp/mcp-plus-plus-connector.ts",
            transport=TransportKind.HTTP,
            has_call_edge=True,
        ),
        _art(
            f"reg:{tool}",
            ArtifactRole.REGISTRATION,
            tool,
            tool_name=tool,
            language="python",
            package="ipfs_accelerate_py",
            qualified_name=f"mcp_server.registry.{tool}",
            version="1.0.0",
            input_schema=SCHEMA_PATH,
            output_schema=SCHEMA_CONTENT,
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
            f"impl:{tool}",
            ArtifactRole.IMPLEMENTATION,
            "ipfs_kit_py.vfs.read",
            tool_name=tool,
            language="python",
            package="ipfs_kit_py",
            qualified_name="ipfs_kit_py.vfs.read",
            path="ipfs_kit_py/ipfs_kit_py/vfs.py",
            version="1.0.0",
            has_call_edge=True,
            input_schema=SCHEMA_PATH,
            output_schema=SCHEMA_CONTENT,
            error_codes=("not_found", "permission_denied"),
        ),
        _art(
            f"rmap:{tool}",
            ArtifactRole.RESULT_MAP,
            f"{tool}.result",
            tool_name=tool,
            language="python",
        ),
        _art(
            f"emap:{tool}",
            ArtifactRole.ERROR_MAP,
            f"{tool}.errors",
            tool_name=tool,
            language="python",
            error_codes=("not_found", "permission_denied"),
        ),
        _art(
            f"manifest:{tool}",
            ArtifactRole.MANIFEST,
            tool,
            tool_name=tool,
            language="json",
            version="1.0.0",
            input_schema=SCHEMA_PATH,
            output_schema=SCHEMA_CONTENT,
            markers=("generated",),
            record={"artifact_kind": "json_manifest"},
        ),
        _art(
            f"sdk:{tool}",
            ArtifactRole.MANIFEST,
            tool,
            tool_name=tool,
            language="typescript",
            version="1.0.0",
            input_schema=SCHEMA_PATH,
            output_schema=SCHEMA_CONTENT,
            markers=("generated", "sdk"),
            path="sdk/generated/vfs.ts",
            record={"artifact_kind": "typescript_sdk"},
        ),
    )


def _proved_inventory(tool: str = "vfs.read") -> MCPlusPlusInventory:
    return MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=("mcp++/mcp-idl",),
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=_proved_chain_artifacts(tool),
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


def _text_only_inventory(tool: str = "vfs.read") -> MCPlusPlusInventory:
    """Same names on registration/list/manifest without call-path edges."""

    return MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                f"list:{tool}",
                ArtifactRole.TOOL_LIST_ENTRY,
                tool,
                tool_name=tool,
                language="json",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
            ),
            _art(
                f"reg:{tool}",
                ArtifactRole.REGISTRATION,
                tool,
                tool_name=tool,
                language="python",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
                package="ipfs_accelerate_py",
            ),
            _art(
                f"manifest:{tool}",
                ArtifactRole.MANIFEST,
                tool,
                tool_name=tool,
                language="json",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
                markers=("generated",),
            ),
            _art(
                f"impl:{tool}",
                ArtifactRole.IMPLEMENTATION,
                "ipfs_kit_py.vfs.read",
                tool_name=tool,
                language="python",
                package="ipfs_kit_py",
                qualified_name="ipfs_kit_py.vfs.read",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Vocabulary / identity
# ---------------------------------------------------------------------------


def test_closed_vocabularies_and_authority_flags() -> None:
    kinds = finding_kinds()
    surfaces = parity_surfaces()
    assert ParityFindingKind.MISSING_RESOLVED_CALL_PATH.value in kinds
    assert ParityFindingKind.STALE_GENERATED_ARTIFACT.value in kinds
    assert ParityFindingKind.DIRECT_LOCAL_BYPASS.value in kinds
    assert ParityFindingKind.MOCK_FALLBACK_DISPATCH.value in kinds
    assert ParitySurface.PYTHON_SIGNATURE.value in surfaces
    assert ParitySurface.TYPESCRIPT_SDK.value in surfaces
    assert ParitySurface.SWISSKNIFE_CONNECTOR.value in surfaces
    assert "connector" in REQUIRED_PROVED_STAGES
    assert "package_implementation" in REQUIRED_PROVED_STAGES
    assert CHECKER_IS_COMPLETION_EVIDENCE is False
    assert CHECKER_AUTHORIZES_REPAIR is False
    assert GOAL_ID == "VFS-028"
    assert CHECKER_VERSION.startswith("vfs-mcp-contract-checker@")


def test_finding_requires_witness_and_proved_parity_requires_path() -> None:
    with pytest.raises(VfsMcpCheckerError, match="witness"):
        from ipfs_accelerate_py.agent_supervisor.vfs_mcp_contract_checker import (
            ParityFinding,
            ParitySeverity,
        )

        ParityFinding(
            kind=ParityFindingKind.NAME_MISMATCH,
            tool_name="vfs.read",
            severity=ParitySeverity.ERROR,
            summary="no witness",
            witnesses=(),
        )

    from ipfs_accelerate_py.agent_supervisor.vfs_mcp_contract_checker import (
        ToolParityResult,
    )

    with pytest.raises(VfsMcpCheckerError, match="resolved call path"):
        ToolParityResult(
            tool_name="vfs.read",
            verdict=ToolParityVerdict.PROVED_PARITY,
            surfaces={},
            findings=(),
            proved_call_path=False,
            text_names_agree=True,
        )


def test_surface_view_and_witness_helpers_are_content_addressed() -> None:
    left = make_surface_view(
        surface=ParitySurface.REGISTRATION,
        present=True,
        tool_name="vfs.read",
        version="1.0.0",
        input_schema_fingerprint=schema_fingerprint(SCHEMA_PATH),
    )
    right = make_surface_view(
        surface=ParitySurface.TOOLS_LIST,
        present=True,
        tool_name="vfs.read",
        version="1.0.0",
        input_schema_fingerprint=schema_fingerprint(SCHEMA_PATH),
    )
    assert left.view_id.startswith("vfsurf-")
    assert left.to_dict()["surface"] == "registration"
    wit = make_parity_witness(
        kind=ParityFindingKind.SCHEMA_MISMATCH,
        tool_name="vfs.read",
        left_surface=ParitySurface.REGISTRATION,
        right_surface=ParitySurface.TOOLS_LIST,
        left_value=left.input_schema_fingerprint,
        right_value=right.input_schema_fingerprint,
    )
    assert wit.witness_id.startswith("vfswit-")
    roundtrip = type(wit).from_dict(wit.to_dict())
    assert roundtrip.witness_id == wit.witness_id


# ---------------------------------------------------------------------------
# Happy path: proved end-to-end parity
# ---------------------------------------------------------------------------


def test_proved_call_path_with_matching_surfaces_is_proved_parity() -> None:
    inventory = _proved_inventory()
    report = check_vfs_mcp_parity(
        inventory,
        claims=(_claim(),),
        tool_names=("vfs.read",),
        use_canonical_contract_pack=True,
    )
    assert isinstance(report, VfsMcpParityReport)
    assert report.goal_id == "VFS-028"
    assert EVIDENCE_VFS_MCP_PARITY in report.evidence_kinds
    assert report.is_completion_evidence is False
    assert report.authorizes_repair is False

    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.proved_call_path is True
    assert tool.text_names_agree is True
    assert tool.verdict is ToolParityVerdict.PROVED_PARITY
    assert report.verdict is ReportVerdict.ALL_PROVED

    # All compared surfaces present.
    for surface in (
        ParitySurface.PYTHON_SIGNATURE,
        ParitySurface.REGISTRATION,
        ParitySurface.TOOLS_LIST,
        ParitySurface.JSON_MANIFEST,
        ParitySurface.TYPESCRIPT_SDK,
        ParitySurface.SWISSKNIFE_CONNECTOR,
        ParitySurface.TRANSPORT_PROFILE,
        ParitySurface.RESULT_ERROR_MAP,
        ParitySurface.IMPLEMENTATION_TARGET,
    ):
        view = tool.surfaces[surface.value]
        assert view.present, surface.value

    # Round-trip identity stability.
    again = VfsMcpParityReport.from_dict(report.to_dict())
    assert again.report_id == report.report_id
    assert report_content_identity(report) == report.report_id


def test_resolver_resolution_receipt_is_consumed() -> None:
    inventory = _proved_inventory()
    resolution = MCPlusPlusContractResolver(inventory).resolve((_claim(),))
    assert any(call_path_is_proved(p) for p in resolution.paths)

    checker = VfsMcpContractChecker(
        inventory, contract_pack=canonical_vfs_contract_pack()
    )
    report = checker.check(
        tool_names=("vfs.read",),
        resolution=resolution,
    )
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.PROVED_PARITY
    assert tool.path_ids


# ---------------------------------------------------------------------------
# Same text without resolved call path is insufficient
# ---------------------------------------------------------------------------


def test_same_text_without_resolved_call_path_is_insufficient() -> None:
    inventory = _text_only_inventory()
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.text_names_agree is True
    assert tool.proved_call_path is False
    assert tool.verdict is ToolParityVerdict.INSUFFICIENT_PATH
    assert report.verdict is ReportVerdict.HAS_INSUFFICIENT_PATH

    missing = [
        f
        for f in tool.findings
        if f.kind is ParityFindingKind.MISSING_RESOLVED_CALL_PATH
    ]
    assert missing, "must report missing_resolved_call_path"
    assert missing[0].witnesses
    note = missing[0].witnesses[0].notes
    assert note.get("rule") == "same_text_without_resolved_call_path_insufficient"


def test_text_names_agree_helper() -> None:
    views = {
        ParitySurface.REGISTRATION.value: make_surface_view(
            surface=ParitySurface.REGISTRATION,
            present=True,
            tool_name="vfs.read",
        ),
        ParitySurface.TOOLS_LIST.value: make_surface_view(
            surface=ParitySurface.TOOLS_LIST,
            present=True,
            tool_name="vfs/read",
        ),
        ParitySurface.JSON_MANIFEST.value: make_surface_view(
            surface=ParitySurface.JSON_MANIFEST,
            present=True,
            tool_name="VFS.Read",
        ),
    }
    assert text_names_agree(views) is True
    views[ParitySurface.TYPESCRIPT_SDK.value] = make_surface_view(
        surface=ParitySurface.TYPESCRIPT_SDK,
        present=True,
        tool_name="vfs.write",
    )
    assert text_names_agree(views) is False


# ---------------------------------------------------------------------------
# Drift: stale generated, missing registration, extra unreachable, schema/errors
# ---------------------------------------------------------------------------


def test_stale_generated_manifest_and_sdk_version() -> None:
    artifacts = list(_proved_chain_artifacts())
    # Stale generated manifest/SDK versions.
    refreshed = []
    for item in artifacts:
        if item.artifact_id.startswith("manifest:") or item.artifact_id.startswith(
            "sdk:"
        ):
            refreshed.append(
                _art(
                    item.artifact_id,
                    item.role,
                    item.name,
                    tool_name=item.tool_name,
                    language=item.language,
                    version="0.9.0",
                    input_schema=dict(item.input_schema),
                    output_schema=dict(item.output_schema),
                    markers=item.markers,
                    path=item.path,
                    record=dict(item.record),
                )
            )
        else:
            refreshed.append(item)
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=tuple(refreshed),
    )
    report = check_vfs_mcp_parity(
        inventory,
        claims=(_claim(),),
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    stale = report.findings_of(ParityFindingKind.STALE_GENERATED_ARTIFACT)
    assert stale
    assert any(f.tool_name == "vfs.read" for f in stale)
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.WITNESSED_DRIFT


def test_missing_registration_finding() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "list:vfs.stat",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.stat",
                tool_name="vfs.stat",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
            ),
            _art(
                "manifest:vfs.stat",
                ArtifactRole.MANIFEST,
                "vfs.stat",
                tool_name="vfs.stat",
                version="1.0.0",
                markers=("generated",),
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.stat",),
        use_canonical_contract_pack=False,
    )
    missing = report.findings_of(ParityFindingKind.MISSING_REGISTRATION)
    assert missing
    assert missing[0].witnesses[0].left_value


def test_extra_unreachable_registration() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg:vfs.ghost",
                ArtifactRole.REGISTRATION,
                "vfs.ghost",
                tool_name="vfs.ghost",
                language="python",
                version="1.0.0",
                qualified_name="mcp_server.registry.vfs.ghost",
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.ghost",),
        use_canonical_contract_pack=False,
    )
    extra = report.findings_of(ParityFindingKind.EXTRA_UNREACHABLE_TOOL)
    assert extra
    tool = report.tool_result("vfs.ghost")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.WITNESSED_DRIFT


def test_schema_and_error_map_mismatch() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "list:vfs.read",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
                error_codes=("not_found",),
            ),
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                version="1.0.0",
                input_schema=SCHEMA_PATH_ALT,
                output_schema=SCHEMA_CONTENT,
                error_codes=("not_found", "permission_denied", "io_failure"),
            ),
            _art(
                "emap:vfs.read",
                ArtifactRole.ERROR_MAP,
                "vfs.read.errors",
                tool_name="vfs.read",
                error_codes=("not_found",),
            ),
            _art(
                "rmap:vfs.read",
                ArtifactRole.RESULT_MAP,
                "vfs.read.result",
                tool_name="vfs.read",
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    schema = report.findings_of(ParityFindingKind.SCHEMA_MISMATCH)
    errors = report.findings_of(ParityFindingKind.ERROR_MAP_MISMATCH)
    assert schema, "input schema drift must be witnessed"
    assert errors, "error code drift must be witnessed"
    for finding in list(schema) + list(errors):
        assert finding.witnesses
        assert finding.witnesses[0].left_value != finding.witnesses[0].right_value


def test_wrong_alias_on_sdk() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                version="1.0.0",
            ),
            _art(
                "list:vfs.read",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
            _art(
                "sdk:bad",
                ArtifactRole.MANIFEST,
                "vfsReadCompat",
                tool_name="vfsReadCompat",
                language="typescript",
                alias_of="vfs.write",
                markers=("sdk", "generated"),
                record={"artifact_kind": "typescript_sdk"},
            ),
        ),
    )
    # Force comparison under the registration tool name; alias mismatch still
    # surfaces when the SDK entry is bound via alias set for vfs.read? The SDK
    # tool name differs, so check under the SDK tool name as well.
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfsReadCompat",),
        use_canonical_contract_pack=False,
    )
    # Registration missing for vfsReadCompat + wrong alias relative to absent reg.
    # Seed a registration under the SDK tool name with alias mismatch path.
    inventory2 = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                version="1.0.0",
            ),
            _art(
                "sdk:bad",
                ArtifactRole.MANIFEST,
                "vfs.read",
                tool_name="vfs.read",
                language="typescript",
                alias_of="totally.wrong",
                markers=("sdk", "generated"),
                record={"artifact_kind": "typescript_sdk"},
            ),
        ),
    )
    report2 = check_vfs_mcp_parity(
        inventory2,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    aliases = report2.findings_of(ParityFindingKind.WRONG_ALIAS)
    assert aliases
    assert aliases[0].witnesses[0].left_value == "totally.wrong"
    del report  # keep first scenario from raising unused lint noise in some runners


# ---------------------------------------------------------------------------
# Direct local bypass / mock fallback / ambiguous path
# ---------------------------------------------------------------------------


def test_direct_local_bypass_rejected() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "list:vfs.read",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
            ),
            _art(
                "helper:local",
                ArtifactRole.LOCAL_HELPER,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                qualified_name="local.vfs_read_helper",
                path="src/helpers/vfs_read.ts",
                markers=("local_bypass",),
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    bypass = report.findings_of(ParityFindingKind.DIRECT_LOCAL_BYPASS)
    assert bypass
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.REJECTED


def test_mock_fallback_dispatch_rejected() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "list:vfs.read",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
            _art(
                "reg:vfs.read",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
            ),
            _art(
                "mock:vfs.read",
                ArtifactRole.MOCK,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                qualified_name="tests.mocks.vfs_read",
                path="test/mocks/vfs_read.py",
                markers=("mock",),
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    mocks = report.findings_of(ParityFindingKind.MOCK_FALLBACK_DISPATCH)
    assert mocks
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.REJECTED


def test_ambiguous_registration_path_finding() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=("mcp++/mcp-idl",),
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=_proved_chain_artifacts("vfs.read")
        + (
            _art(
                "reg:vfs.read:dup",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                package="ipfs_accelerate_py",
                qualified_name="mcp_server.registry.vfs.read.alt",
                version="1.0.0",
                input_schema=SCHEMA_PATH,
                output_schema=SCHEMA_CONTENT,
                error_codes=("not_found", "permission_denied"),
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        claims=(_claim(),),
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    tool = report.tool_result("vfs.read")
    assert tool is not None
    # Resolver marks ambiguous registration; checker must surface it.
    assert (
        tool.verdict is ToolParityVerdict.AMBIGUOUS
        or any(f.kind is ParityFindingKind.AMBIGUOUS_PATH for f in tool.findings)
        or PathVerdict.AMBIGUOUS.value in tool.path_verdicts
    )


# ---------------------------------------------------------------------------
# Transport / capability / degradation / runtime mock authority
# ---------------------------------------------------------------------------


def test_transport_and_profile_mismatch() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=(
            _art(
                "conn",
                ArtifactRole.CONNECTOR,
                "MCPPPServerConnector.callTool",
                tool_name="vfs.read",
                language="typescript",
                transport=TransportKind.HTTP,
                profiles=("mcp++/basic",),
                has_call_edge=True,
            ),
            _art(
                "transport",
                ArtifactRole.TRANSPORT,
                "mcp-p2p",
                tool_name="vfs.read",
                transport=TransportKind.MCP_P2P,
                profiles=("mcp++/p2p-transport",),
            ),
            _art(
                "reg",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
            ),
            _art(
                "list",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    assert report.findings_of(ParityFindingKind.TRANSPORT_MISMATCH)
    assert report.findings_of(ParityFindingKind.PROFILE_MISMATCH)


def test_silent_degradation_claim_is_reported() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                record={
                    "degradation_claims": ("silent_success", "placeholder_success"),
                    "capabilities": ("profile:mcp++/basic",),
                },
            ),
            _art(
                "list",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
            _art(
                "conn",
                ArtifactRole.CONNECTOR,
                "MCPPPServerConnector.callTool",
                tool_name="vfs.read",
                profiles=("mcp++/basic",),
                has_call_edge=True,
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    deg = report.findings_of(ParityFindingKind.DEGRADATION_CLAIM_MISMATCH)
    assert deg


def _mock_runtime_receipt(tool: str = "vfs.read") -> RuntimeWitnessReceipt:
    discovery = ToolDiscoveryRecord(
        tool_names=(tool,),
        adapter_ids=(f"adapter:{tool}:mock",),
        production_tools=(),
        mock_tools=(tool,),
        fixture_tools=(),
        manifest_cid="baguqeeramockmanifest",
        server_name=SERVER,
    )
    negotiation = CapabilityNegotiationRecord(
        requested_profiles=("mcp++/basic",),
        admitted_profiles=("mcp++/basic",),
        active_profile="mcp++/basic",
        requested_transport="http",
        admitted_transports=("http",),
        active_transport="http",
        negotiated=True,
        reason="ok",
    )
    request = CallRequest(
        tool_name=tool,
        arguments={"path": "/x"},
        requested_profiles=("mcp++/basic",),
        transport="http",
        call_id="call-1",
    )
    observation = CallObservation(
        outcome=WitnessOutcome.PASSED,
        tool_name=tool,
        adapter_id=f"adapter:{tool}:mock",
        implementation_kind=ImplementationKind.MOCK,
        implementation_target="tests.mocks.vfs_read",
        input_validation=ValidationVerdict.VALID,
        input_errors=(),
        output_validation=ValidationVerdict.VALID,
        output_errors=(),
        error_code="",
        error_schema_ok=True,
        result={"ok": True},
        phases_completed=("discovery", "dispatch"),
        duration_ms=1,
        timed_out=False,
        cancelled=False,
        cleanup_status=CleanupStatus.CLEAN,
        claim_level=ClaimLevel.RUNTIME_WITNESSED,
        grants_runtime_authority=False,
        reason="mock always ok",
    )
    witness = RuntimeWitness(
        fixture_id="fixture:mock",
        forest_id=FOREST,
        discovery=discovery,
        negotiation=negotiation,
        request=request,
        observation=observation,
        transport="http",
        timeout_ms=1000,
        network_enabled=False,
    )
    return RuntimeWitnessReceipt(
        fixture_id="fixture:mock",
        forest_id=FOREST,
        manifest_cid="baguqeeramockmanifest",
        witnesses=(witness,),
        network_enabled=False,
    )


def test_runtime_mock_cannot_grant_parity_authority() -> None:
    inventory = _text_only_inventory()
    receipt = _mock_runtime_receipt()
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        runtime_receipts=(receipt,),
        use_canonical_contract_pack=False,
    )
    runtime_findings = report.findings_of(ParityFindingKind.RUNTIME_MOCK_AUTHORITY)
    assert runtime_findings
    assert runtime_findings[0].confidence == 0
    tool = report.tool_result("vfs.read")
    assert tool is not None
    assert tool.surfaces[ParitySurface.RUNTIME_WITNESS.value].present
    assert tool.surfaces[ParitySurface.RUNTIME_WITNESS.value].is_mock_or_fallback


# ---------------------------------------------------------------------------
# Discover + compare helpers + contract pack binding
# ---------------------------------------------------------------------------


def test_discover_tool_names_and_compare_without_claims() -> None:
    inventory = _text_only_inventory("vfs.stat")
    names = discover_tool_names(inventory)
    assert "vfs.stat" in names
    report = check_vfs_mcp_parity(
        inventory,
        use_canonical_contract_pack=True,
    )
    assert report.contract_pack_id
    assert report.tool_result("vfs.stat") is not None


def test_implementation_target_mismatch() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                record={"implementation": "ipfs_kit_py.vfs.read"},
            ),
            _art(
                "impl",
                ArtifactRole.IMPLEMENTATION,
                "other.package.vfs.read",
                tool_name="vfs.read",
                language="python",
                qualified_name="other.package.vfs.read",
                record={"implementation": "other.package.vfs.read"},
            ),
            _art(
                "list",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    assert report.findings_of(ParityFindingKind.IMPLEMENTATION_TARGET_MISMATCH)


def test_name_mismatch_across_surfaces() -> None:
    views = {
        ParitySurface.REGISTRATION.value: make_surface_view(
            surface=ParitySurface.REGISTRATION,
            present=True,
            tool_name="vfs.read",
        ),
        ParitySurface.TOOLS_LIST.value: make_surface_view(
            surface=ParitySurface.TOOLS_LIST,
            present=True,
            tool_name="vfs.write",
        ),
        ParitySurface.IMPLEMENTATION_TARGET.value: make_surface_view(
            surface=ParitySurface.IMPLEMENTATION_TARGET,
            present=True,
            tool_name="vfs.read",
        ),
    }
    result = compare_tool_surfaces("vfs.read", views)
    assert any(f.kind is ParityFindingKind.NAME_MISMATCH for f in result.findings)


def test_mcp_p2p_proved_path_parity() -> None:
    tool = "vfs.stat"
    arts = list(_proved_chain_artifacts(tool))
    # Switch transport artifacts to mcp+p2p.
    switched = []
    for item in arts:
        if item.role is ArtifactRole.TRANSPORT:
            switched.append(
                _art(
                    item.artifact_id,
                    item.role,
                    "mcp-p2p",
                    tool_name=tool,
                    transport=TransportKind.MCP_P2P,
                    profiles=("mcp++/p2p-transport", "mcp++/basic"),
                )
            )
        elif item.role is ArtifactRole.CONNECTOR:
            switched.append(
                _art(
                    item.artifact_id,
                    item.role,
                    item.name,
                    language=item.language,
                    qualified_name=item.qualified_name,
                    path=item.path,
                    tool_name=tool,
                    transport=TransportKind.MCP_P2P,
                    profiles=item.profiles,
                    has_call_edge=True,
                )
            )
        elif item.role is ArtifactRole.TOOL_CALL_SITE:
            switched.append(
                _art(
                    item.artifact_id,
                    item.role,
                    item.name,
                    tool_name=tool,
                    language=item.language,
                    qualified_name=item.qualified_name,
                    path=item.path,
                    transport=TransportKind.MCP_P2P,
                    has_call_edge=True,
                )
            )
        else:
            switched.append(item)
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        required_profiles=("mcp++/basic",),
        admitted_transports=(TransportKind.HTTP, TransportKind.MCP_P2P),
        artifacts=tuple(switched),
    )
    report = check_vfs_mcp_parity(
        inventory,
        claims=(
            _claim(
                tool,
                transport=TransportKind.MCP_P2P,
                profiles=("mcp++/basic", "mcp++/p2p-transport"),
            ),
        ),
        tool_names=(tool,),
        use_canonical_contract_pack=False,
    )
    tool_result = report.tool_result(tool)
    assert tool_result is not None
    assert tool_result.proved_call_path is True
    assert tool_result.verdict is ToolParityVerdict.PROVED_PARITY


def test_report_findings_of_and_empty_inventory() -> None:
    inventory = MCPlusPlusInventory(forest_id=FOREST, artifacts=())
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=(),
        use_canonical_contract_pack=False,
    )
    assert report.verdict is ReportVerdict.EMPTY
    assert report.findings_of(ParityFindingKind.MISSING_REGISTRATION) == ()


def test_legacy_fallback_role_is_mock_fallback() -> None:
    inventory = MCPlusPlusInventory(
        forest_id=FOREST,
        admitted_transports=(TransportKind.HTTP,),
        artifacts=(
            _art(
                "reg",
                ArtifactRole.REGISTRATION,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
            ),
            _art(
                "list",
                ArtifactRole.TOOL_LIST_ENTRY,
                "vfs.read",
                tool_name="vfs.read",
            ),
            _art(
                "legacy",
                ArtifactRole.LEGACY_FALLBACK,
                "vfs.read",
                tool_name="vfs.read",
                language="python",
                path="legacy/vfs_fallback.py",
                markers=("fallback",),
            ),
        ),
    )
    report = check_vfs_mcp_parity(
        inventory,
        tool_names=("vfs.read",),
        use_canonical_contract_pack=False,
    )
    assert report.findings_of(ParityFindingKind.MOCK_FALLBACK_DISPATCH)
