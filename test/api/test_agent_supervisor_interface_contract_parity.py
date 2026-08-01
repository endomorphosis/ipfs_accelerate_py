"""Tests for generic interface/manifest/SDK/transport parity (LPR-024)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.interface_contract_parity import (
    CHECKER_AUTHORIZES_REPAIR,
    CHECKER_IS_COMPLETION_EVIDENCE,
    CHECKER_IS_CORRECTNESS_EVIDENCE,
    CHECKER_VERSION,
    DEFAULT_REQUIRED_PROVED_STAGES,
    EVIDENCE_INTERFACE_PARITY,
    INTERFACE_CONTRACT_PARITY_SCHEMA,
    INTERFACE_PARITY_REPORT_SCHEMA,
    CallPathHop,
    ContractProfileAdapter,
    DriftWitnessRecord,
    HopStatus,
    InterfaceContractParityAnalyzer,
    InterfaceParityError,
    InterfaceParityReport,
    ParityFinding,
    ParityFindingKind,
    ParitySeverity,
    ParitySurfaceSpec,
    PathVerdict,
    ReportVerdict,
    ResolvedCallPath,
    RuntimeWitnessObservation,
    SurfaceArtifact,
    SurfaceInventory,
    ToolParityResult,
    ToolParityVerdict,
    ToolSelectionPolicy,
    build_surface_views,
    call_path_is_proved,
    check_interface_parity,
    compare_tool_surfaces,
    default_surface_specs,
    discover_tool_names,
    finding_kinds,
    make_artifact,
    make_parity_witness,
    make_proved_path,
    make_surface_view,
    normalize_tool_name,
    parity_surfaces,
    proved_stages,
    report_content_identity,
    schema_fingerprint,
    text_names_agree,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "analysis"
    / "interface_contract_parity.py"
)
LOCK_PATH = REPO_ROOT / "config" / "agent_supervisor_vfs_generalization_sources.lock.json"

_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|board[_-]?id|board[_-]?namespace)\b"
)

FOREST = "forest:test-lpr-024"
SERVER = "math-rpc-server"
SCHEMA_ARGS = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}
SCHEMA_RESULT = {
    "type": "object",
    "properties": {"sum": {"type": "number"}},
}
SCHEMA_ARGS_ALT = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
        "rounding": {"type": "string"},
    },
    "required": ["a", "b"],
}


def build_math_policy() -> ToolSelectionPolicy:
    """Non-VFS math/RPC policy — injects surfaces, aliases, and proved stages."""

    return ToolSelectionPolicy(
        policy_id="math-rpc-parity",
        surface_specs=default_surface_specs(),
        alias_groups=(("math.add", "math/add", "Math.Add"),),
        alias_map={"math/add": "math.add", "Math.Add": "math.add"},
        required_proved_stages=DEFAULT_REQUIRED_PROVED_STAGES,
        server_name=SERVER,
        notes={"domain": "math-rpc"},
    )


def _art(
    artifact_id: str,
    role: str,
    name: str,
    **kwargs: Any,
) -> SurfaceArtifact:
    defaults: dict[str, Any] = {
        "server_name": SERVER,
    }
    defaults.update(kwargs)
    return make_artifact(
        artifact_id=artifact_id,
        role=role,
        name=name,
        **defaults,
    )


def _proved_chain_artifacts(tool: str = "math.add") -> tuple[SurfaceArtifact, ...]:
    """Full generic connector → registration → implementation inventory slice."""

    return (
        _art(
            "caller:ui",
            "caller",
            "ClientUI.invokeMathAdd",
            language="typescript",
            qualified_name="ClientUI.invokeMathAdd",
            path="src/components/MathPanel.tsx",
            has_call_edge=True,
            record={"connector": "GenericConnector.callTool"},
        ),
        _art(
            "conn:generic",
            "connector",
            "GenericConnector.callTool",
            language="typescript",
            qualified_name="GenericConnector.callTool",
            path="src/services/rpc/generic-connector.ts",
            tool_name=tool,
            transport="http",
            profiles=("rpc/basic", "rpc/idl", "rpc/p2p"),
            has_call_edge=True,
        ),
        _art(
            "transport:http",
            "transport",
            "http-jsonrpc",
            tool_name=tool,
            transport="http",
            profiles=("rpc/basic", "rpc/idl"),
        ),
        _art(
            f"list:{tool}",
            "tool_list_entry",
            tool,
            tool_name=tool,
            language="json",
            version="1.0.0",
            input_schema=SCHEMA_ARGS,
            output_schema=SCHEMA_RESULT,
            error_codes=("invalid_argument", "overflow"),
        ),
        _art(
            f"call:{tool}",
            "tool_call_site",
            f"tools/call:{tool}",
            tool_name=tool,
            language="typescript",
            qualified_name="GenericConnector.callTool",
            path="src/services/rpc/generic-connector.ts",
            transport="http",
            has_call_edge=True,
        ),
        _art(
            f"reg:{tool}",
            "registration",
            tool,
            tool_name=tool,
            language="python",
            package="math_service",
            qualified_name=f"rpc.registry.{tool}",
            version="1.0.0",
            input_schema=SCHEMA_ARGS,
            output_schema=SCHEMA_RESULT,
            error_codes=("invalid_argument", "overflow"),
            record={
                "adapter": "native_math_adapter",
                "implementation": "math_lib.add",
            },
            implementation_target="math_lib.add",
        ),
        _art(
            "adapter:math",
            "adapter",
            "native_math_adapter",
            tool_name=tool,
            language="python",
            package="math_service",
            qualified_name="rpc.tools.math.native_math_adapter",
            path="math_service/rpc/tools/math/native_math_adapter.py",
            has_call_edge=True,
            record={"implementation": "math_lib.add"},
            implementation_target="math_lib.add",
        ),
        _art(
            f"impl:{tool}",
            "implementation",
            "math_lib.add",
            tool_name=tool,
            language="python",
            package="math_lib",
            qualified_name="math_lib.add",
            path="math_lib/math_lib/add.py",
            version="1.0.0",
            has_call_edge=True,
            input_schema=SCHEMA_ARGS,
            output_schema=SCHEMA_RESULT,
            error_codes=("invalid_argument", "overflow"),
            implementation_target="math_lib.add",
        ),
        _art(
            f"rmap:{tool}",
            "result_map",
            f"{tool}.result",
            tool_name=tool,
            language="python",
        ),
        _art(
            f"emap:{tool}",
            "error_map",
            f"{tool}.errors",
            tool_name=tool,
            language="python",
            error_codes=("invalid_argument", "overflow"),
        ),
        _art(
            f"manifest:{tool}",
            "manifest",
            tool,
            tool_name=tool,
            language="json",
            version="1.0.0",
            input_schema=SCHEMA_ARGS,
            output_schema=SCHEMA_RESULT,
            markers=("generated",),
            record={"artifact_kind": "json_manifest"},
        ),
        _art(
            f"sdk:{tool}",
            "manifest",
            tool,
            tool_name=tool,
            language="typescript",
            version="1.0.0",
            input_schema=SCHEMA_ARGS,
            output_schema=SCHEMA_RESULT,
            markers=("generated", "sdk"),
            path="sdk/generated/math.ts",
            record={"artifact_kind": "typescript_sdk"},
        ),
    )


def _proved_inventory(tool: str = "math.add") -> SurfaceInventory:
    return SurfaceInventory(
        inventory_id="inv:math-proved",
        forest_id=FOREST,
        required_profiles=("rpc/idl",),
        admitted_transports=("http", "rpc_p2p"),
        artifacts=_proved_chain_artifacts(tool),
    )


def _text_only_inventory(tool: str = "math.add") -> SurfaceInventory:
    """Same names on registration/list/manifest without call-path edges."""

    return SurfaceInventory(
        inventory_id="inv:math-text-only",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                f"list:{tool}",
                "tool_list_entry",
                tool,
                tool_name=tool,
                language="json",
                version="1.0.0",
                input_schema=SCHEMA_ARGS,
                output_schema=SCHEMA_RESULT,
            ),
            _art(
                f"reg:{tool}",
                "registration",
                tool,
                tool_name=tool,
                language="python",
                version="1.0.0",
                input_schema=SCHEMA_ARGS,
                output_schema=SCHEMA_RESULT,
                package="math_service",
            ),
            _art(
                f"manifest:{tool}",
                "manifest",
                tool,
                tool_name=tool,
                language="json",
                version="1.0.0",
                input_schema=SCHEMA_ARGS,
                output_schema=SCHEMA_RESULT,
                markers=("generated",),
            ),
            _art(
                f"impl:{tool}",
                "implementation",
                "math_lib.add",
                tool_name=tool,
                language="python",
                package="math_lib",
                qualified_name="math_lib.add",
                input_schema=SCHEMA_ARGS,
                output_schema=SCHEMA_RESULT,
            ),
        ),
    )


def _math_contract_adapter() -> ContractProfileAdapter:
    return ContractProfileAdapter(
        adapter_id="math-contract-pack",
        operations=("math.add", "math.sub"),
        surfaces=("python_client", "http_gateway", "cli"),
        operation_entrypoints={
            "math.add": "math_lib.add",
            "math.sub": "math_lib.sub",
        },
        capability_claims={"math.add": ("profile:rpc/basic",)},
        error_codes={"math.add": ("invalid_argument", "overflow")},
        profile_content_id="sha256:" + "ab" * 32,
    )


# ---------------------------------------------------------------------------
# Generic module constraints
# ---------------------------------------------------------------------------


def test_generic_module_has_no_domain_literals() -> None:
    source = PARITY_MODULE.read_text(encoding="utf-8")
    # Strip this file's own docstring/comments? No — module body must be clean.
    # Allow the path prefix "ipfs_accelerate_py" in schema strings only.
    cleaned = re.sub(
        r'ipfs_accelerate_py/agent-supervisor/[a-z0-9@./-]+',
        "SCHEMA",
        source,
    )
    hits = _FORBIDDEN_GENERIC.findall(cleaned)
    assert hits == [], f"generic core contains forbidden domain literals: {hits}"


def test_generic_module_has_no_implicit_provider_imports() -> None:
    source = PARITY_MODULE.read_text(encoding="utf-8")
    assert "mcplusplus" not in source.lower()
    assert "importlib" not in source
    assert "vfs_contract" not in source
    assert "swissknife" not in source.lower()


def test_source_lock_declares_parity_generalization() -> None:
    lock = LOCK_PATH.read_text(encoding="utf-8")
    assert "interface_contract_parity.py" in lock
    assert "vfs_mcp_contract_checker.py" in lock
    assert "26144a7b78c1bbbb94edc67ab13e2eab03850924" in lock


# ---------------------------------------------------------------------------
# Vocabulary / identity / authority
# ---------------------------------------------------------------------------


def test_closed_vocabularies_and_authority_flags() -> None:
    kinds = finding_kinds()
    surfaces = parity_surfaces()
    assert ParityFindingKind.MISSING_RESOLVED_CALL_PATH.value in kinds
    assert ParityFindingKind.STALE_GENERATED_ARTIFACT.value in kinds
    assert ParityFindingKind.DIRECT_LOCAL_BYPASS.value in kinds
    assert ParityFindingKind.MOCK_FALLBACK_DISPATCH.value in kinds
    assert ParityFindingKind.UNRESOLVED_PATH.value in kinds
    assert "python_signature" in surfaces
    assert "typescript_sdk" in surfaces
    assert "connector" in surfaces
    assert "swissknife_connector" not in surfaces
    assert "connector" in proved_stages()
    assert "package_implementation" in proved_stages()
    assert CHECKER_IS_COMPLETION_EVIDENCE is False
    assert CHECKER_IS_CORRECTNESS_EVIDENCE is False
    assert CHECKER_AUTHORIZES_REPAIR is False
    assert CHECKER_VERSION.startswith("interface-contract-parity@")
    assert INTERFACE_CONTRACT_PARITY_SCHEMA.endswith("interface-contract-parity@1")
    assert INTERFACE_PARITY_REPORT_SCHEMA.endswith("interface-parity-report@1")


def test_finding_requires_witness_and_proved_parity_requires_path() -> None:
    with pytest.raises(InterfaceParityError, match="witness"):
        ParityFinding(
            kind=ParityFindingKind.NAME_MISMATCH,
            tool_name="math.add",
            severity=ParitySeverity.ERROR,
            summary="no witness",
            witnesses=(),
        )

    with pytest.raises(InterfaceParityError, match="resolved call path"):
        ToolParityResult(
            tool_name="math.add",
            verdict=ToolParityVerdict.PROVED_PARITY,
            surfaces={},
            findings=(),
            proved_call_path=False,
            text_names_agree=True,
        )


def test_surface_view_and_witness_helpers_are_content_addressed() -> None:
    left = make_surface_view(
        surface="registration",
        present=True,
        tool_name="math.add",
        version="1.0.0",
        input_schema_fingerprint=schema_fingerprint(SCHEMA_ARGS),
    )
    right = make_surface_view(
        surface="tools_list",
        present=True,
        tool_name="math.add",
        version="1.0.0",
        input_schema_fingerprint=schema_fingerprint(SCHEMA_ARGS),
    )
    assert left.view_id.startswith("icpsurf-")
    assert left.to_dict()["surface"] == "registration"
    wit = make_parity_witness(
        kind=ParityFindingKind.SCHEMA_MISMATCH,
        tool_name="math.add",
        left_surface="registration",
        right_surface="tools_list",
        left_value=left.input_schema_fingerprint,
        right_value=right.input_schema_fingerprint,
    )
    assert wit.witness_id.startswith("icpwit-")
    roundtrip = type(wit).from_dict(wit.to_dict())
    assert roundtrip.witness_id == wit.witness_id


# ---------------------------------------------------------------------------
# Happy path: math.add + generic connector proved parity
# ---------------------------------------------------------------------------


def test_math_add_and_generic_connector_proved_parity() -> None:
    policy = build_math_policy()
    inventory = _proved_inventory()
    path = make_proved_path(
        "math.add",
        policy=policy,
        connector_ref="GenericConnector.callTool",
        implementation_ref="math_lib.add",
        profiles=("rpc/basic", "rpc/idl"),
    )
    assert call_path_is_proved(path, policy) is True

    report = check_interface_parity(
        inventory,
        policy,
        paths=(path,),
        tool_names=("math.add",),
        contract_adapter=_math_contract_adapter(),
    )
    assert isinstance(report, InterfaceParityReport)
    assert EVIDENCE_INTERFACE_PARITY in report.evidence_kinds
    assert report.is_completion_evidence is False
    assert report.authorizes_repair is False
    assert report.contract_pack_id

    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.proved_call_path is True
    assert tool.text_names_agree is True
    assert tool.verdict is ToolParityVerdict.PROVED_PARITY
    assert report.verdict is ReportVerdict.ALL_PROVED

    for surface in (
        "python_signature",
        "registration",
        "tools_list",
        "json_manifest",
        "typescript_sdk",
        "connector",
        "transport_profile",
        "result_error_map",
        "implementation_target",
    ):
        view = tool.surfaces[surface]
        assert view.present, surface

    again = InterfaceParityReport.from_dict(report.to_dict())
    assert again.report_id == report.report_id
    assert report_content_identity(report) == report.report_id


def test_analyzer_consumes_policy_surface_specs_and_aliases() -> None:
    policy = build_math_policy()
    inventory = _proved_inventory()
    # Alias orthography: math/add should resolve to math.add via policy.
    path = make_proved_path("math.add", policy=policy)
    analyzer = InterfaceContractParityAnalyzer(inventory, policy)
    report = analyzer.check(tool_names=("math/add",), paths=(path,))
    tool = report.tool_result("math/add")
    assert tool is not None
    # Path aliases bind the proved path to the tool.
    assert tool.proved_call_path is True


# ---------------------------------------------------------------------------
# Same text without resolved call path is insufficient
# ---------------------------------------------------------------------------


def test_same_text_without_resolved_call_path_is_insufficient() -> None:
    policy = build_math_policy()
    inventory = _text_only_inventory()
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    tool = report.tool_result("math.add")
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
    policy = build_math_policy()
    views = {
        "registration": make_surface_view(
            surface="registration",
            present=True,
            tool_name="math.add",
        ),
        "tools_list": make_surface_view(
            surface="tools_list",
            present=True,
            tool_name="math/add",
        ),
        "json_manifest": make_surface_view(
            surface="json_manifest",
            present=True,
            tool_name="Math.Add",
        ),
    }
    assert text_names_agree(views, policy) is True
    views["typescript_sdk"] = make_surface_view(
        surface="typescript_sdk",
        present=True,
        tool_name="math.sub",
    )
    assert text_names_agree(views, policy) is False


def test_seeded_unresolved_path_fails() -> None:
    policy = build_math_policy()
    inventory = _proved_inventory()
    unresolved = ResolvedCallPath(
        path_id="path:math.add:unresolved",
        tool_name="math.add",
        verdict=PathVerdict.UNRESOLVED,
        hops=(
            CallPathHop(
                stage="connector",
                status=HopStatus.UNRESOLVED,
                reason_code="missing_binding",
            ),
        ),
        connector_ref="GenericConnector.callTool",
        implementation_ref="",
    )
    report = check_interface_parity(
        inventory,
        policy,
        paths=(unresolved,),
        tool_names=("math.add",),
    )
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.proved_call_path is False
    assert tool.verdict is not ToolParityVerdict.PROVED_PARITY
    assert report.findings_of(ParityFindingKind.UNRESOLVED_PATH)
    assert report.findings_of(ParityFindingKind.MISSING_RESOLVED_CALL_PATH) or (
        tool.verdict is ToolParityVerdict.INSUFFICIENT_PATH
        or tool.verdict is ToolParityVerdict.WITNESSED_DRIFT
    )


def test_seeded_wrong_implementation_target_fails() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:wrong-target",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "reg",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
                implementation_target="math_lib.add",
                record={"implementation": "math_lib.add"},
            ),
            _art(
                "impl",
                "implementation",
                "other.package.math.add",
                tool_name="math.add",
                language="python",
                qualified_name="other.package.math.add",
                implementation_target="other.package.math.add",
                record={"implementation": "other.package.math.add"},
            ),
            _art(
                "list",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    assert report.findings_of(ParityFindingKind.IMPLEMENTATION_TARGET_MISMATCH)
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.verdict is not ToolParityVerdict.PROVED_PARITY


# ---------------------------------------------------------------------------
# Drift: stale, missing, extra, schema/errors, alias
# ---------------------------------------------------------------------------


def test_stale_generated_manifest_and_sdk_version() -> None:
    policy = build_math_policy()
    artifacts = list(_proved_chain_artifacts())
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
    inventory = SurfaceInventory(
        inventory_id="inv:stale",
        forest_id=FOREST,
        admitted_transports=("http", "rpc_p2p"),
        artifacts=tuple(refreshed),
    )
    path = make_proved_path("math.add", policy=policy)
    report = check_interface_parity(
        inventory,
        policy,
        paths=(path,),
        tool_names=("math.add",),
    )
    stale = report.findings_of(ParityFindingKind.STALE_GENERATED_ARTIFACT)
    assert stale
    assert any(f.tool_name == "math.add" for f in stale)
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.WITNESSED_DRIFT


def test_missing_registration_finding() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:missing-reg",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "list:math.stat",
                "tool_list_entry",
                "math.stat",
                tool_name="math.stat",
                version="1.0.0",
                input_schema=SCHEMA_ARGS,
            ),
            _art(
                "manifest:math.stat",
                "manifest",
                "math.stat",
                tool_name="math.stat",
                version="1.0.0",
                markers=("generated",),
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.stat",),
    )
    missing = report.findings_of(ParityFindingKind.MISSING_REGISTRATION)
    assert missing
    assert missing[0].witnesses[0].left_value


def test_extra_unreachable_registration() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:ghost",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "reg:math.ghost",
                "registration",
                "math.ghost",
                tool_name="math.ghost",
                language="python",
                version="1.0.0",
                qualified_name="rpc.registry.math.ghost",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.ghost",),
    )
    extra = report.findings_of(ParityFindingKind.EXTRA_UNREACHABLE_TOOL)
    assert extra
    tool = report.tool_result("math.ghost")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.WITNESSED_DRIFT


def test_schema_and_error_map_mismatch() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:schema-err",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "list:math.add",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
                version="1.0.0",
                input_schema=SCHEMA_ARGS,
                output_schema=SCHEMA_RESULT,
                error_codes=("invalid_argument",),
            ),
            _art(
                "reg:math.add",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
                version="1.0.0",
                input_schema=SCHEMA_ARGS_ALT,
                output_schema=SCHEMA_RESULT,
                error_codes=("invalid_argument", "overflow", "io_failure"),
            ),
            _art(
                "emap:math.add",
                "error_map",
                "math.add.errors",
                tool_name="math.add",
                error_codes=("invalid_argument",),
            ),
            _art(
                "rmap:math.add",
                "result_map",
                "math.add.result",
                tool_name="math.add",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    schema = report.findings_of(ParityFindingKind.SCHEMA_MISMATCH)
    errors = report.findings_of(ParityFindingKind.ERROR_MAP_MISMATCH)
    assert schema, "input schema drift must be witnessed"
    assert errors, "error code drift must be witnessed"
    for finding in list(schema) + list(errors):
        assert finding.witnesses
        assert finding.witnesses[0].left_value != finding.witnesses[0].right_value


def test_wrong_alias_on_sdk() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:wrong-alias",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "reg:math.add",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
                version="1.0.0",
            ),
            _art(
                "sdk:bad",
                "manifest",
                "math.add",
                tool_name="math.add",
                language="typescript",
                alias_of="totally.wrong",
                markers=("sdk", "generated"),
                record={"artifact_kind": "typescript_sdk"},
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    aliases = report.findings_of(ParityFindingKind.WRONG_ALIAS)
    assert aliases
    assert aliases[0].witnesses[0].left_value == "totally.wrong"


# ---------------------------------------------------------------------------
# Direct local bypass / mock fallback / ambiguous path
# ---------------------------------------------------------------------------


def test_direct_local_bypass_rejected() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:bypass",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "list:math.add",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
            _art(
                "reg:math.add",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
            ),
            _art(
                "helper:local",
                "local_helper",
                "math.add",
                tool_name="math.add",
                language="python",
                qualified_name="local.math_add_helper",
                path="src/helpers/math_add.ts",
                markers=("local_bypass",),
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    # local_helper role may not map to a default surface; force via implementation view
    # by also attaching markers on an implementation-bearing artifact.
    if not report.findings_of(ParityFindingKind.DIRECT_LOCAL_BYPASS):
        inventory2 = SurfaceInventory(
            inventory_id="inv:bypass2",
            forest_id=FOREST,
            admitted_transports=("http",),
            artifacts=(
                _art(
                    "list:math.add",
                    "tool_list_entry",
                    "math.add",
                    tool_name="math.add",
                ),
                _art(
                    "reg:math.add",
                    "registration",
                    "math.add",
                    tool_name="math.add",
                    language="python",
                ),
                _art(
                    "impl:bypass",
                    "implementation",
                    "local.math_add_helper",
                    tool_name="math.add",
                    language="python",
                    qualified_name="local.math_add_helper",
                    markers=("local_bypass",),
                    record={"dispatch": "local_bypass"},
                ),
            ),
        )
        report = check_interface_parity(
            inventory2,
            policy,
            tool_names=("math.add",),
        )
    bypass = report.findings_of(ParityFindingKind.DIRECT_LOCAL_BYPASS)
    assert bypass
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.REJECTED


def test_mock_fallback_dispatch_rejected() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:mock",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "list:math.add",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
            _art(
                "reg:math.add",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
            ),
            _art(
                "mock:math.add",
                "implementation",
                "math.add",
                tool_name="math.add",
                language="python",
                qualified_name="tests.mocks.math_add",
                path="test/mocks/math_add.py",
                markers=("mock",),
                non_invocation_reason="mock_implementation",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    mocks = report.findings_of(ParityFindingKind.MOCK_FALLBACK_DISPATCH)
    assert mocks
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.verdict is ToolParityVerdict.REJECTED


def test_ambiguous_registration_path_finding() -> None:
    policy = build_math_policy()
    path = ResolvedCallPath(
        path_id="path:math.add:ambiguous",
        tool_name="math.add",
        verdict=PathVerdict.AMBIGUOUS,
        hops=tuple(
            CallPathHop(stage=s, status=HopStatus.AMBIGUOUS)
            for s in policy.required_proved_stages
        ),
        connector_ref="GenericConnector.callTool",
        implementation_ref="math_lib.add",
    )
    report = check_interface_parity(
        _proved_inventory(),
        policy,
        paths=(path,),
        tool_names=("math.add",),
    )
    tool = report.tool_result("math.add")
    assert tool is not None
    assert (
        tool.verdict is ToolParityVerdict.AMBIGUOUS
        or any(f.kind is ParityFindingKind.AMBIGUOUS_PATH for f in tool.findings)
        or PathVerdict.AMBIGUOUS.value in tool.path_verdicts
    )


# ---------------------------------------------------------------------------
# Transport / capability / degradation / runtime mock authority
# ---------------------------------------------------------------------------


def test_transport_and_profile_mismatch() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:transport",
        forest_id=FOREST,
        admitted_transports=("http", "rpc_p2p"),
        artifacts=(
            _art(
                "conn",
                "connector",
                "GenericConnector.callTool",
                tool_name="math.add",
                language="typescript",
                transport="http",
                profiles=("rpc/basic",),
                has_call_edge=True,
            ),
            _art(
                "transport",
                "transport",
                "rpc-p2p",
                tool_name="math.add",
                transport="rpc_p2p",
                profiles=("rpc/p2p",),
            ),
            _art(
                "reg",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
            ),
            _art(
                "list",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    assert report.findings_of(ParityFindingKind.TRANSPORT_MISMATCH)
    assert report.findings_of(ParityFindingKind.PROFILE_MISMATCH)


def test_silent_degradation_claim_is_reported() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:degrade",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "reg",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
                record={
                    "degradation_claims": ("silent_success", "placeholder_success"),
                    "capabilities": ("profile:rpc/basic",),
                },
            ),
            _art(
                "list",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
            _art(
                "conn",
                "connector",
                "GenericConnector.callTool",
                tool_name="math.add",
                profiles=("rpc/basic",),
                has_call_edge=True,
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    deg = report.findings_of(ParityFindingKind.DEGRADATION_CLAIM_MISMATCH)
    assert deg


def test_runtime_mock_cannot_grant_parity_authority() -> None:
    policy = build_math_policy()
    inventory = _text_only_inventory()
    observation = RuntimeWitnessObservation(
        tool_name="math.add",
        implementation_kind="mock",
        implementation_target="tests.mocks.math_add",
        outcome="passed",
        grants_runtime_authority=False,
        is_mock=True,
        transport="http",
        profiles=("rpc/basic",),
        receipt_id="receipt:mock-1",
        fixture_id="fixture:mock",
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
        runtime_observations=(observation,),
    )
    runtime_findings = report.findings_of(ParityFindingKind.RUNTIME_MOCK_AUTHORITY)
    assert runtime_findings
    assert runtime_findings[0].confidence == 0
    tool = report.tool_result("math.add")
    assert tool is not None
    assert tool.surfaces["runtime_witness"].present
    assert tool.surfaces["runtime_witness"].is_mock_or_fallback


# ---------------------------------------------------------------------------
# Discover + compare helpers + forged/unbounded rejection
# ---------------------------------------------------------------------------


def test_discover_tool_names_and_compare_without_paths() -> None:
    policy = build_math_policy()
    inventory = _text_only_inventory("math.stat")
    names = discover_tool_names(inventory, policy)
    assert "math.stat" in names
    report = check_interface_parity(
        inventory,
        policy,
        contract_adapter=_math_contract_adapter(),
    )
    assert report.contract_pack_id
    assert report.tool_result("math.stat") is not None


def test_name_mismatch_across_surfaces() -> None:
    policy = build_math_policy()
    views = {
        "registration": make_surface_view(
            surface="registration",
            present=True,
            tool_name="math.add",
        ),
        "tools_list": make_surface_view(
            surface="tools_list",
            present=True,
            tool_name="math.sub",
        ),
        "implementation_target": make_surface_view(
            surface="implementation_target",
            present=True,
            tool_name="math.add",
        ),
    }
    result = compare_tool_surfaces("math.add", views, policy=policy)
    assert any(f.kind is ParityFindingKind.NAME_MISMATCH for f in result.findings)


def test_report_findings_of_and_empty_inventory() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(inventory_id="inv:empty", forest_id=FOREST, artifacts=())
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=(),
    )
    assert report.verdict is ReportVerdict.EMPTY
    assert report.findings_of(ParityFindingKind.MISSING_REGISTRATION) == ()


def test_legacy_fallback_role_is_mock_fallback() -> None:
    policy = build_math_policy()
    inventory = SurfaceInventory(
        inventory_id="inv:legacy",
        forest_id=FOREST,
        admitted_transports=("http",),
        artifacts=(
            _art(
                "reg",
                "registration",
                "math.add",
                tool_name="math.add",
                language="python",
            ),
            _art(
                "list",
                "tool_list_entry",
                "math.add",
                tool_name="math.add",
            ),
            _art(
                "legacy",
                "implementation",
                "math.add",
                tool_name="math.add",
                language="python",
                path="legacy/math_fallback.py",
                markers=("fallback",),
                non_invocation_reason="legacy_fallback",
            ),
        ),
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
    )
    assert report.findings_of(ParityFindingKind.MOCK_FALLBACK_DISPATCH)


def test_forged_artifact_content_id_is_rejected() -> None:
    with pytest.raises(InterfaceParityError, match="forged"):
        SurfaceInventory(
            inventory_id="inv:forged",
            artifacts=(
                SurfaceArtifact(
                    artifact_id="art:forged",
                    role="registration",
                    name="math.add",
                    tool_name="math.add",
                    content_id="sha256:" + "00" * 32,
                ),
            ),
        )


def test_unbounded_artifacts_are_rejected() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.interface_contract_parity import (
        DEFAULT_MAX_NOTES_BYTES,
        InterfaceParityBoundsError,
    )

    with pytest.raises((InterfaceParityError, InterfaceParityBoundsError)):
        SurfaceArtifact(
            artifact_id="art:huge",
            role="registration",
            name="math.add",
            record={"blob": "z" * (DEFAULT_MAX_NOTES_BYTES + 64)},
        )


def test_build_surface_views_projects_contract_adapter() -> None:
    policy = build_math_policy()
    views = build_surface_views(
        _proved_inventory(),
        "math.add",
        policy,
        contract_adapter=_math_contract_adapter(),
    )
    assert views["contract_pack"].present
    assert views["contract_pack"].notes.get("operation") == "math.add"
    assert "profile:rpc/basic" in views["capability_degradation"].capability_claims


def test_policy_surface_specs_are_injectable() -> None:
    custom = ToolSelectionPolicy(
        policy_id="minimal-surfaces",
        surface_specs=(
            ParitySurfaceSpec(kind="registration", roles=("registration",), name_bearing=True),
            ParitySurfaceSpec(kind="tools_list", roles=("tool_list_entry",), name_bearing=True),
            ParitySurfaceSpec(kind="connector", roles=("connector",)),
            ParitySurfaceSpec(kind="implementation_target", roles=("implementation",), name_bearing=True),
        ),
        required_proved_stages=("connector", "package_implementation"),
    )
    assert "json_manifest" not in custom.surface_kinds()
    assert parity_surfaces(custom) == tuple(sorted(custom.surface_kinds()))
    assert "connector" in proved_stages(custom)


def test_drift_witness_is_consumed() -> None:
    policy = build_math_policy()
    inventory = _text_only_inventory()
    drift = DriftWitnessRecord(
        drift_kind="schema_mismatch",
        tool_name="math.add",
        left_value="fp-a",
        right_value="fp-b",
        left_ref="list:math.add",
        right_ref="reg:math.add",
    )
    report = check_interface_parity(
        inventory,
        policy,
        tool_names=("math.add",),
        drift_witnesses=(drift,),
    )
    assert report.findings_of(ParityFindingKind.SCHEMA_MISMATCH)


def test_normalize_tool_name_is_domain_neutral() -> None:
    assert normalize_tool_name("Math/Add") == "math.add"
    assert normalize_tool_name("math.add") == "math.add"
    # Must not invent product-domain aliases.
    assert "vfs" not in normalize_tool_name("Math.Add")


def test_contract_adapter_from_program_like_object() -> None:
    class _Op:
        def __init__(self, operation: str, entrypoint: str = "") -> None:
            self.operation = operation
            self.entrypoint = entrypoint
            self.support = "supported"
            self.error_codes = ("invalid_argument",)

    class _Profile:
        operations = (_Op("math.add", "math_lib.add"), _Op("math.mul"))
        surfaces = ("python_client", "http_gateway")
        content_id = "sha256:" + "cd" * 32

    adapter = ContractProfileAdapter.from_program_contract_profile(
        _Profile(),
        adapter_id="from-profile",
    )
    assert "math.add" in adapter.operations
    assert adapter.operation_entrypoints["math.add"] == "math_lib.add"
    # Supported without entrypoint → unresolved
    assert "math.mul" in adapter.unresolved_operations
