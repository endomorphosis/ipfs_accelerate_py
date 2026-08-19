"""SCA-041: SwissKnife expected MCP++ contract extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    ContractInvalidationKind,
    ReviewState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.swissknife_contract_extractor import (
    CANONICAL_SERVER_PACKAGES,
    SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE,
    InvocationEdgeKind,
    SourceRole,
    SwissKnifeContractExtractor,
    SwissKnifeContractExtractorError,
    SwissKnifeSource,
    extract_swissknife_contracts,
)


def _interface(
    constant: str,
    *,
    name: str,
    namespace: str,
    version: str = "1.0.0",
    method: str = "status",
    stream: bool = False,
) -> str:
    pattern = ", interaction_pattern: 'stream'" if stream else ""
    return f"""
export const {constant}: MCPPPInterfaceDescriptor = {{
  name: {name!r},
  namespace: {namespace!r},
  version: {version!r},
  interface_cid: 'bafy-{constant.lower()}',
  methods: [{{
    name: {method!r},
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error']{pattern},
    resource_cost_hints: {{ retries_default: 2 }},
  }}],
  errors: [{{ name: 'Unavailable', code: 503 }}],
  requires: ['mcp++/cid-envelope', 'mcp++/deontic-policy', 'mcp++/p2p-transport'],
  compatibility: {{ compatible_with: ['bafy-peer'], supersedes: [] }},
}};
"""


def _canonical_source() -> str:
    return "\n".join(
        (
            _interface(
                "IPFS_KIT_INTERFACE",
                name="ipfs-kit",
                namespace="com.ipfs.kit",
                method="ipfs.add",
            ),
            _interface(
                "IPFS_ACCELERATE_INTERFACE",
                name="ipfs-accelerate",
                namespace="com.ipfs.accelerate",
                method="accelerate.inference",
                stream=True,
            ),
            _interface(
                "IPFS_DATASETS_INTERFACE",
                name="ipfs-datasets",
                namespace="com.ipfs.datasets",
                method="datasets.search",
            ),
        )
    )


def _extract(
    sources: dict[str, str],
    *,
    source_version: str = "git:fixture",
):
    return SwissKnifeContractExtractor().extract(
        sources,
        repository_tree_id="git-tree:fixture",
        source_version=source_version,
    )


def test_interface_name_and_canonical_package_coverage() -> None:
    result = _extract({"src/services/mcp/mcp-plus-plus.ts": _canonical_source()})

    assert SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE == "SwissKnifeContractExtractor@1"
    assert result.canonical_packages_present == CANONICAL_SERVER_PACKAGES
    assert result.missing_canonical_packages == ()
    assert result.require_canonical_packages() is result
    by_package = {descriptor.package_id: descriptor for descriptor in result.descriptors}
    assert set(by_package) == set(CANONICAL_SERVER_PACKAGES)
    assert by_package["ipfs_accelerate_py"].methods[0].streaming is True
    assert by_package["ipfs_accelerate_py"].methods[0].interaction_pattern == "stream"
    assert by_package["ipfs_kit_py"].methods[0].error_schemas == ("bafy-error",)
    assert by_package["ipfs_kit_py"].errors[0]["code"] == 503
    assert "mcp++/deontic-policy" in by_package["ipfs_kit_py"].policy_requirements
    assert "mcp++/p2p-transport" in by_package["ipfs_kit_py"].transport_expectations


def test_descriptor_wrapper_resolves_local_constants_and_spreads() -> None:
    source = """
const EXTRA_REFS = { receipt: 'contracts/receipt.schema.json' } as const;
export const PACKAGE_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-kit',
  namespace: 'com.ipfs.kit',
  version: '1.2.0',
  interface_cid: 'bafy-kit',
  methods: [],
  errors: [],
  requires: [],
  compatibility: { compatible_with: [IMPORTED_INTERFACE.interface_cid], supersedes: [] },
};
export const PACKAGE_DESCRIPTOR = {
  descriptor_id: 'ipfs-kit@1.2.0',
  interface: PACKAGE_INTERFACE,
  schema_refs: { ...EXTRA_REFS, dynamic: runtimeSchema() },
};
"""
    result = _extract(
        {"src/services/mcp/ipfs-kit-interop-descriptor.ts": source}
    )

    assert len(result.descriptors) == 1
    descriptor = result.descriptors[0]
    assert descriptor.descriptor_id == "ipfs-kit@1.2.0"
    assert descriptor.schema_refs["receipt"] == "contracts/receipt.schema.json"
    fields = {item.field_path for item in result.unresolved_values}
    assert "PACKAGE_INTERFACE.compatibility.compatible_with[0]" in fields
    assert "PACKAGE_DESCRIPTOR.schema_refs.dynamic" in fields


def test_dynamic_values_are_unresolved_with_exact_source_span() -> None:
    source = """
export const IPFS_ACCELERATE_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-accelerate',
  namespace: 'com.ipfs.accelerate',
  version: process.env.MCP_VERSION,
  interface_cid: `bafy-${runtimeCid}`,
  methods: [],
  errors: [],
  requires: [],
  compatibility: { compatible_with: [], supersedes: [] },
};
"""
    result = _extract({"src/services/mcp/mcp-plus-plus.ts": source})
    dynamic = next(
        item
        for item in result.unresolved_values
        if item.field_path.endswith(".version")
    )

    assert dynamic.expression == "process.env.MCP_VERSION"
    assert dynamic.source_span.start_line == 5
    assert (
        source[dynamic.source_span.start_offset : dynamic.source_span.end_offset]
        == dynamic.expression
    )
    # A dynamic required descriptor version is not guessed into a descriptor.
    assert result.descriptors == ()


def test_connector_retains_jsonrpc_direct_and_dynamic_invocation_edges() -> None:
    source = """
export class Connector {
  async list() { return this.jsonRpc('tools/list', {}); }
  async call(name: string, args: object) {
    return this.jsonRpc('tools/call', { name, arguments: args });
  }
  async direct() { return globalThis.fetch('/api/v0/ipfs/add'); }
  async computed(path: string) { return globalThis.fetch(`${this.base}${path}`); }
  async facade() { return this.callTool('tools_dispatch', { category: 'ipfs', tool: 'add' }); }
}
"""
    result = _extract(
        {"src/services/mcp/mcp-plus-plus-connector.ts": source}
    )

    kinds = {edge.kind for edge in result.invocation_edges}
    assert InvocationEdgeKind.TOOLS_LIST in kinds
    assert InvocationEdgeKind.TOOLS_CALL in kinds
    assert InvocationEdgeKind.COMPATIBILITY_ROUTE in kinds
    assert InvocationEdgeKind.HIERARCHICAL_DISPATCH in kinds
    direct = next(
        edge for edge in result.invocation_edges if edge.target == "/api/v0/ipfs/add"
    )
    assert direct.bypass_candidate is True
    assert direct.compatibility is True
    computed = next(
        edge
        for edge in result.invocation_edges
        if edge.kind is InvocationEdgeKind.DIRECT_FETCH and edge.target is None
    )
    assert computed.unresolved_id
    assert any(
        item.unresolved_id == computed.unresolved_id
        for item in result.unresolved_values
    )


def test_capability_registry_retains_direct_and_compatibility_bindings() -> None:
    source = """
export const swissknifeMCPCapabilityRegistry = [{
  server_package: 'ipfs_kit_py',
  transport: 'mcp-server',
  capability_descriptor: {
    command_intents: [
      { intent: 'storage.add', tool_name: 'ipfs_add',
        upstream_function: '/api/v0/ipfs/add',
        payload_contracts: ['content_ref'] },
      { intent: 'storage.dispatch', tool_name: 'tools_dispatch',
        tool_category: 'ipfs_tools', upstream_function: 'add',
        payload_contracts: ['mediation_receipt'] },
    ],
  },
}];
"""
    result = _extract(
        {"src/services/apps/swissknife-mcp-capability-registry.ts": source}
    )
    direct = next(edge for edge in result.invocation_edges if edge.target == "/api/v0/ipfs/add")
    dispatch = next(edge for edge in result.invocation_edges if edge.target == "add")

    assert direct.kind is InvocationEdgeKind.COMPATIBILITY_ROUTE
    assert direct.transport == "http"
    assert dispatch.kind is InvocationEdgeKind.COMPATIBILITY_ROUTE
    assert dispatch.metadata["tool_category"] == "ipfs_tools"
    retained_fields = {item.field_path for item in result.expectations}
    assert "[0].transport" in retained_fields
    assert any(path.endswith(".payload_contracts") for path in retained_fields)


def test_versions_defaults_errors_streaming_policy_and_transport_are_preserved() -> None:
    source = _interface(
        "IPFS_ACCELERATE_INTERFACE",
        name="ipfs-accelerate",
        namespace="com.ipfs.accelerate",
        version="3.4.5",
        method="accelerate.inference",
        stream=True,
    ) + """
export function createDelegation(expirationHours: number = 24, strict = true) {
  return { expirationHours, strict };
}
"""
    result = _extract({"src/services/mcp/mcp-plus-plus.ts": source})
    descriptor = result.descriptors[0]
    defaults = {
        (item.metadata.get("parameter"), item.value)
        for item in result.expectations
        if item.metadata.get("symbol") == "createDelegation"
    }

    assert descriptor.version == "3.4.5"
    assert descriptor.schema_version == "3.4.5"
    assert descriptor.streaming is True
    assert descriptor.errors[0] == {"code": 503, "name": "Unavailable"}
    assert defaults == {("expirationHours", 24), ("strict", True)}


def test_json_schema_defaults_error_states_and_versions() -> None:
    schema = """
{
  "$id": "swissknife/contracts/interaction-envelope@2",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Interaction envelope",
  "properties": {
    "timeout_ms": {"type": "integer", "default": 5000},
    "outcome": {"type": "string", "enum": ["ok", "denied", "timed_out", "partial"]}
  }
}
"""
    result = _extract({"contracts/interaction_envelope.schema.json": schema})
    record = result.schemas[0]

    assert record.schema_id == "swissknife/contracts/interaction-envelope@2"
    assert record.schema_version.endswith("/draft/2020-12/schema")
    assert record.defaults["properties.timeout_ms"] == 5000
    assert {"denied", "timed_out", "partial"} <= set(record.error_values)
    default = next(
        item for item in result.expectations if item.metadata.get("default")
    )
    assert default.value == 5000


def test_conflicting_descriptor_and_contract_test_remain_contradicted() -> None:
    descriptor = _interface(
        "IPFS_ACCELERATE_INTERFACE",
        name="ipfs-accelerate",
        namespace="com.ipfs.accelerate",
        version="1.0.0",
    )
    test_source = """
import { IPFS_ACCELERATE_INTERFACE } from '../../src/services/mcp/mcp-plus-plus.js';
it('binds the reviewed version', () => {
  expect(IPFS_ACCELERATE_INTERFACE.version).toBe('2.0.0');
});
"""
    result = SwissKnifeContractExtractor().extract(
        (
            SwissKnifeSource(
                "src/services/mcp/mcp-plus-plus.ts",
                descriptor,
                source_version="git:abc",
            ),
            SwissKnifeSource(
                "test/mcp-plus-plus/descriptor.test.ts",
                test_source,
                source_version="git:abc",
            ),
        ),
        repository_tree_id="tree:abc",
    )

    contradictions = result.catalog.contradictions_for(
        "mcp-interface:ipfs_accelerate_py:version"
    )
    assert contradictions
    assert all(item.resolved is False for item in contradictions)
    contract = next(
        item
        for item in result.catalog.contracts
        if item.subject == "mcp-interface:ipfs_accelerate_py:version"
    )
    assert contract.review_state is ReviewState.CONTRADICTED
    assert contract.contradiction_ids


def test_matching_descriptor_and_contract_test_do_not_conflict() -> None:
    descriptor = _interface(
        "IPFS_ACCELERATE_INTERFACE",
        name="ipfs-accelerate",
        namespace="com.ipfs.accelerate",
        version="1.0.0",
    )
    test_source = """
expect(IPFS_ACCELERATE_INTERFACE.version).toEqual('1.0.0');
"""
    result = _extract(
        {
            "src/services/mcp/mcp-plus-plus.ts": descriptor,
            "test/mcp-plus-plus/descriptor.test.ts": test_source,
        }
    )
    assert (
        result.catalog.contradictions_for(
            "mcp-interface:ipfs_accelerate_py:version"
        )
        == ()
    )


def test_catalog_sources_bind_explicit_versions_tree_and_authority() -> None:
    result = _extract({"src/services/mcp/mcp-plus-plus.ts": _canonical_source()})
    source = next(
        item
        for item in result.catalog.sources
        if item.subject == "mcp-interface:ipfs_kit_py:version"
    )
    invalidators = {item.kind: item.value for item in source.invalidators}

    assert source.source_version == "git:fixture"
    assert source.schema_version == "1.0.0"
    assert source.review_state is ReviewState.REVIEWED
    assert invalidators[ContractInvalidationKind.SOURCE_VERSION] == "git:fixture"
    assert invalidators[ContractInvalidationKind.SCHEMA_VERSION] == "1.0.0"
    assert invalidators[ContractInvalidationKind.REPOSITORY_TREE] == "git-tree:fixture"


def test_extraction_is_content_addressed_and_input_order_independent() -> None:
    sources = {
        "src/services/mcp/mcp-plus-plus.ts": _canonical_source(),
        "src/services/mcp/mcp-plus-plus-connector.ts": (
            "export async function call() { return fetch('/mcp/tools/list'); }"
        ),
    }
    first = _extract(sources)
    second = _extract(dict(reversed(tuple(sources.items()))))

    assert first.extraction_id.startswith("b")
    assert first.extraction_id == second.extraction_id
    assert first.catalog.catalog_id == second.catalog.catalog_id
    assert first.to_dict() == second.to_dict()


def test_source_role_override_keeps_conformance_authority() -> None:
    source = _interface(
        "IPFS_KIT_INTERFACE",
        name="ipfs-kit",
        namespace="com.ipfs.kit",
    )
    result = SwissKnifeContractExtractor().extract(
        (
            SwissKnifeSource(
                "fixtures/descriptor.ts",
                source,
                source_version="fixture:1",
                role=SourceRole.CONTRACT_TEST,
            ),
        )
    )
    assert result.descriptors[0].source_role is SourceRole.CONTRACT_TEST
    assert all(
        item.kind.value == "conformance_test"
        for item in result.catalog.sources
    )


def test_repository_extraction_uses_explicit_scoped_paths(tmp_path: Path) -> None:
    mcp_dir = tmp_path / "src" / "services" / "mcp"
    mcp_dir.mkdir(parents=True)
    source_path = mcp_dir / "mcp-plus-plus.ts"
    source_path.write_text(_canonical_source(), encoding="utf-8")
    (tmp_path / "ignored.ts").write_text("throw new Error('ignored')", encoding="utf-8")

    result = SwissKnifeContractExtractor().extract_repository(
        tmp_path,
        include_paths=("src/services/mcp/mcp-plus-plus.ts",),
        source_version="git:tmp",
    )
    assert set(result.source_versions) == {"src/services/mcp/mcp-plus-plus.ts"}
    assert result.canonical_packages_present == CANONICAL_SERVER_PACKAGES


def test_input_limits_duplicate_paths_and_traversal_fail_closed() -> None:
    with pytest.raises(SwissKnifeContractExtractorError, match="max_files"):
        SwissKnifeContractExtractor(max_files=0)
    with pytest.raises(SwissKnifeContractExtractorError, match="traverse"):
        SwissKnifeSource("../descriptor.ts", "export const x = 1")
    with pytest.raises(SwissKnifeContractExtractorError, match="duplicate"):
        SwissKnifeContractExtractor().extract(
            (
                SwissKnifeSource("a.ts", "export const a = 1"),
                SwissKnifeSource("a.ts", "export const b = 2"),
            )
        )
    with pytest.raises(SwissKnifeContractExtractorError, match="file byte"):
        SwissKnifeContractExtractor(max_file_bytes=4, max_total_bytes=8).extract(
            {"a.ts": "12345"}
        )
    partial = _extract(
        {
            "src/services/mcp/mcp-plus-plus.ts": _interface(
                "IPFS_KIT_INTERFACE",
                name="ipfs-kit",
                namespace="com.ipfs.kit",
            )
        }
    )
    with pytest.raises(SwissKnifeContractExtractorError, match="missing canonical"):
        partial.require_canonical_packages()


def test_invalid_schema_fails_with_path_and_location() -> None:
    with pytest.raises(
        SwissKnifeContractExtractorError,
        match=r"contracts/bad\.json: invalid JSON at line 1, column",
    ):
        _extract({"contracts/bad.json": '{"type": }'})


def test_convenience_function_returns_catalog_ready_extraction() -> None:
    result = extract_swissknife_contracts(
        {"src/services/mcp/mcp-plus-plus.ts": _canonical_source()},
        repository_tree_id="tree:convenience",
        source_version="git:convenience",
    )
    assert result.catalog.contracts
    assert result.catalog.sources
    assert result.canonical_packages_present == CANONICAL_SERVER_PACKAGES
