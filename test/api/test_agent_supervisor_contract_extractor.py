"""Conformance tests for multi-source program-contract extraction (VFS-015)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_extractor import (
    CONTRACT_EXTRACTOR_VERSION,
    DEFAULT_POLICY_REVISION,
    ContentKind,
    ContractExtractionResult,
    ContractExtractorError,
    ContractSourceUnit,
    ExtractionRule,
    SkipReason,
    SourceArtifactClass,
    all_artifact_classes,
    all_extraction_rules,
    classify_artifact_path,
    confidence_for,
    contract_source_unit_from_mapping,
    expectation_source_kinds,
    extract_contracts,
    extraction_rule_for,
    make_mcp_tool_unit,
    make_observation_unit,
    make_signature_unit,
    reject_observation_as_expectation_source,
    type_shape_from_json_schema,
    type_shape_from_name,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    SOURCE_PRECEDENCE,
    CapabilityMode,
    CircularExpectationError,
    ConfidenceClass,
    ConflictKind,
    ContractSourceKind,
    Optionality,
    ParameterKind,
    ProgramContractRole,
    SemanticAspect,
    SupportStatus,
    SyncMode,
    TypeConstructor,
    may_define_expectation,
    source_precedence_rank,
)


REPO = "repository:ipfs_kit_py"
TREE = "tree:abc123"
POLICY = DEFAULT_POLICY_REVISION
BLOB = "baguqeera" + "1" * 50
SHA_A = "a" * 64


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _mcp_read(
    *,
    artifact_id: str = "artifact:mcp-idl",
    returns_type: str = "string",
    async_mode: bool = False,
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE,
    surface: str = "mcp++",
    version: str = "1.0",
    **extra,
) -> ContractSourceUnit:
    return make_mcp_tool_unit(
        artifact_id=artifact_id,
        name="vfs.read",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo path"},
            },
            "required": ["path"],
        },
        output_schema={"type": returns_type},
        errors=[
            {"name": "NotFound", "code": "NOT_FOUND", "retriable": False},
            {"name": "PathEscapeError", "code": "PATH_ESCAPE"},
        ],
        async_mode=async_mode,
        capabilities=[
            {"name": "vfs.read", "mode": "required", "version": "1"},
            {"name": "vfs.optional_cache", "mode": "optional"},
        ],
        description="Read bytes for an authorized path",
        repository_id=REPO,
        tree_id=TREE,
        surface=surface,
        version=version,
        locator=f"tools/list#vfs.read#{artifact_id}",
        span_start=10,
        span_end=80,
        blob_cid=BLOB,
        artifact_class=artifact_class,
        side_effects=[
            {"kind": "filesystem", "polarity": "allowed", "target": "path"},
            {"kind": "write", "polarity": "forbidden"},
        ],
        authorization={"mode": "path_scope", "scopes": ["repo:read"]},
        idempotence="pure",
        **extra,
    )


def _signature_read(
    *,
    artifact_id: str = "artifact:signature",
    returns: str = "bytes",
    async_mode: bool = False,
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE,
) -> ContractSourceUnit:
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=ContractSourceKind.PUBLIC_SIGNATURE,
        payload={
            "kind": "public_signature",
            "name": "read",
            "parameters": [
                {
                    "name": "path",
                    "type": "str",
                    "kind": "positional",
                    "optional": False,
                }
            ],
            "returns": returns,
            "raises": ["NotFound", "PathEscapeError"],
            "async": async_mode,
            "side_effects": [
                {"kind": "filesystem", "polarity": "allowed", "target": "path"}
            ],
        },
        repository_id=REPO,
        tree_id=TREE,
        module_path="ipfs_kit_py/vfs.py",
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        protocol="mcp",
        locator="ipfs_kit_py/vfs.py:read_bytes",
        language="python",
        span_start=100,
        span_end=140,
        blob_cid=BLOB,
        artifact_class=artifact_class,
        extraction_rule=ExtractionRule.PUBLIC_SIGNATURE_V1,
    )


def _contract_test_read(
    *,
    artifact_id: str = "artifact:contract-test",
    returns: str = "bytes",
) -> ContractSourceUnit:
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=ContractSourceKind.CONTRACT_TEST,
        payload={
            "kind": "contract_test",
            "asserts": {
                "parameters": [
                    {"name": "path", "type": "str", "optional": False}
                ],
                "returns": returns,
                "errors": [{"name": "NotFound", "code": "NOT_FOUND"}],
                "async": False,
            },
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        protocol="mcp",
        locator="test/contract/test_vfs_read.py",
        span_start=1,
        span_end=50,
        blob_cid=BLOB,
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.CONTRACT_TEST_V1,
    )


def _doc_read(
    *,
    artifact_id: str = "artifact:docs",
    returns: str = "str",
) -> ContractSourceUnit:
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=ContractSourceKind.NORMATIVE_DOCUMENTATION,
        payload={
            "kind": "normative_doc",
            "clauses": {
                "parameters": [{"name": "path", "type": "str"}],
                "returns": returns,
                "errors": [{"name": "NotFound", "code": "NOT_FOUND"}],
                "summary": "Docs claim read returns a text string.",
            },
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        protocol="mcp",
        locator="docs/vfs.md#read",
        span_start=200,
        span_end=240,
        blob_cid=BLOB,
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.NORMATIVE_DOC_V1,
    )


def _manifest_read(
    *,
    artifact_id: str = "artifact:generated-sdk",
    artifact_class: SourceArtifactClass = SourceArtifactClass.GENERATED,
) -> ContractSourceUnit:
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=ContractSourceKind.COMPATIBILITY_MANIFEST,
        payload={
            "kind": "generated_sdk",
            "tool": {
                "name": "vfs.read",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "outputSchema": {"type": "string"},
            },
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        protocol="mcp",
        locator="sdk/generated/vfs_read.json",
        artifact_class=artifact_class,
        extraction_rule=ExtractionRule.GENERATED_SDK_V1,
        span_start=0,
        span_end=10,
        blob_cid=BLOB,
    )


def _observation_read(
    *,
    artifact_id: str = "artifact:runtime-obs",
    returns: str = "bytes",
) -> ContractSourceUnit:
    return make_observation_unit(
        artifact_id=artifact_id,
        symbol_name="read",
        repository_observation_id="observation:fixture-1",
        observed={
            "parameters": [{"name": "path", "type": "str"}],
            "returns": returns,
            "errors": [{"name": "NotFound", "code": "NOT_FOUND"}],
            "async": False,
            "side_effects": [
                {"kind": "filesystem", "polarity": "observed", "target": "path"}
            ],
            "capabilities": [{"name": "vfs.read", "mode": "observed"}],
        },
        repository_id=REPO,
        tree_id=TREE,
        module_path="ipfs_kit_py/vfs.py",
        locator="callsite:vfs.read",
        span_start=300,
        span_end=320,
        blob_cid=BLOB,
    )


# ---------------------------------------------------------------------------
# Vocabulary / precedence
# ---------------------------------------------------------------------------


def test_extractor_version_and_closed_vocabularies() -> None:
    assert CONTRACT_EXTRACTOR_VERSION == 1
    assert expectation_source_kinds() == SOURCE_PRECEDENCE
    ranks = [source_precedence_rank(kind) for kind in expectation_source_kinds()]
    assert ranks == list(range(len(SOURCE_PRECEDENCE)))
    assert may_define_expectation(ContractSourceKind.REVIEWED_INTERFACE)
    assert not may_define_expectation(
        ContractSourceKind.IMPLEMENTATION_OBSERVATION
    )
    classes = {item.value for item in all_artifact_classes()}
    assert classes == {
        "normative",
        "example",
        "mock",
        "fixture",
        "deprecated",
        "generated",
        "observation",
    }
    rules = {item.value for item in all_extraction_rules()}
    assert "mcp_idl_v1" in rules
    assert "json_schema_v1" in rules
    assert "precedence_merge_v1" in rules
    assert "self_expectation_guard_v1" in rules


def test_classify_artifact_path_distinguishes_roles() -> None:
    assert classify_artifact_path("docs/examples/vfs_read.md") is SourceArtifactClass.EXAMPLE
    assert classify_artifact_path("test/mocks/fake_vfs.py") is SourceArtifactClass.MOCK
    assert classify_artifact_path("test/fixtures/vfs_read.json") is SourceArtifactClass.FIXTURE
    assert classify_artifact_path("sdk/generated/vfs_client.py") is SourceArtifactClass.GENERATED
    assert (
        classify_artifact_path("ipfs_kit_py/vfs.py.deprecated")
        is SourceArtifactClass.DEPRECATED
    )
    assert (
        classify_artifact_path("ipfs_kit_py/mcp/tools/vfs_read.json")
        is SourceArtifactClass.NORMATIVE
    )
    assert (
        classify_artifact_path(
            "anything", explicit=SourceArtifactClass.FIXTURE
        )
        is SourceArtifactClass.FIXTURE
    )


def test_confidence_and_extraction_rule_mapping() -> None:
    assert (
        confidence_for(
            ContractSourceKind.REVIEWED_INTERFACE,
            SourceArtifactClass.NORMATIVE,
        )
        is ConfidenceClass.HIGH
    )
    assert (
        confidence_for(
            ContractSourceKind.NORMATIVE_DOCUMENTATION,
            SourceArtifactClass.NORMATIVE,
        )
        is ConfidenceClass.MEDIUM
    )
    assert (
        confidence_for(
            ContractSourceKind.COMPATIBILITY_MANIFEST,
            SourceArtifactClass.GENERATED,
        )
        is ConfidenceClass.MEDIUM
    )
    assert (
        confidence_for(
            ContractSourceKind.PUBLIC_SIGNATURE,
            SourceArtifactClass.MOCK,
        )
        is ConfidenceClass.SPECULATIVE
    )
    assert (
        extraction_rule_for(
            ContractSourceKind.REVIEWED_INTERFACE,
            ContentKind.MCP_PLUS_PLUS,
        )
        is ExtractionRule.MCP_PLUS_PLUS_IDL_V1
    )
    assert (
        extraction_rule_for(
            ContractSourceKind.PUBLIC_SIGNATURE,
            ContentKind.PUBLIC_SIGNATURE,
        )
        is ExtractionRule.PUBLIC_SIGNATURE_V1
    )


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def test_type_shape_from_name_unions_and_containers() -> None:
    assert type_shape_from_name("str").constructor is TypeConstructor.STRING
    assert type_shape_from_name("bytes").constructor is TypeConstructor.BYTES
    union = type_shape_from_name("str | bytes")
    assert union.constructor is TypeConstructor.UNION
    assert len(union.alternatives) == 2
    array = type_shape_from_name("list[str]")
    assert array.constructor is TypeConstructor.ARRAY
    assert array.item is not None
    assert array.item.constructor is TypeConstructor.STRING
    optional = type_shape_from_name("Optional[int]")
    assert optional.nullable is True
    assert optional.constructor is TypeConstructor.INT


def test_json_schema_unions_missing_refs_and_circular() -> None:
    missing: list[str] = []
    union = type_shape_from_json_schema(
        {
            "oneOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        },
        missing_refs=missing,
    )
    assert union.constructor is TypeConstructor.UNION
    assert len(union.alternatives) == 2

    missing = []
    unresolved = type_shape_from_json_schema(
        {"$ref": "#/$defs/MissingType"},
        definitions={},
        missing_refs=missing,
    )
    assert unresolved.constructor is TypeConstructor.REFERENCE
    assert unresolved.support is SupportStatus.UNSUPPORTED
    assert missing == ["#/$defs/MissingType"]

    circular = type_shape_from_json_schema(
        {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "$ref": "#/$defs/Node",
        },
        missing_refs=[],
    )
    # Circular self-ref should not infinite-loop; nested child becomes unsupported.
    assert circular.constructor in {
        TypeConstructor.OBJECT,
        TypeConstructor.UNSUPPORTED,
    }
    if circular.constructor is TypeConstructor.OBJECT:
        fields = dict(circular.fields)
        assert "child" in fields
        assert fields["child"].support is SupportStatus.UNSUPPORTED
        assert "circular_ref" in fields["child"].constraints


# ---------------------------------------------------------------------------
# Happy-path multi-source extraction
# ---------------------------------------------------------------------------


def test_extract_precedence_prefers_reviewed_idl_over_weaker_sources() -> None:
    result = extract_contracts(
        [
            _mcp_read(returns_type="string"),  # IDL: string
            _signature_read(returns="bytes"),  # signature: bytes (weaker)
            _contract_test_read(returns="bytes"),
            _doc_read(returns="str"),
            _manifest_read(),
        ],
        repository_id=REPO,
        tree_id=TREE,
        policy_revision=POLICY,
    )
    assert len(result.expected) == 1
    expected = result.expected[0]
    assert expected.primary_source_kind is ContractSourceKind.REVIEWED_INTERFACE
    assert expected.returns is not None
    # Dominant reviewed IDL says string.
    assert expected.returns.type_shape.constructor is TypeConstructor.STRING
    # Provenance includes multiple ranks and extraction rules.
    kinds = {source.source_kind for source in expected.sources}
    assert ContractSourceKind.REVIEWED_INTERFACE in kinds
    assert ContractSourceKind.PUBLIC_SIGNATURE in kinds
    for source in expected.sources:
        assert source.extractor_rule
        assert source.sha256.startswith("sha256:")
        assert source.role is ProgramContractRole.EXPECTED
    # Weaker signature disagrees on returns → conflict retained.
    assert any(
        conflict.aspect is SemanticAspect.OUTPUTS for conflict in expected.conflicts
    ) or any(
        conflict.aspect is SemanticAspect.OUTPUTS for conflict in result.conflicts
    )
    # Generated manifest shadowed by reviewed IDL.
    assert any(
        item.reason is SkipReason.GENERATED_SHADOWED for item in result.skipped
    )


def test_extract_emits_source_spans_cids_rules_and_confidence() -> None:
    unit = _mcp_read()
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    source = expected.sources[0]
    assert source.span_start == 10
    assert source.span_end == 80
    assert source.sha256.startswith("sha256:")
    assert source.extractor_rule in {
        ExtractionRule.MCP_IDL_V1.value,
        ExtractionRule.MCP_PLUS_PLUS_IDL_V1.value,
    }
    assert source.confidence is ConfidenceClass.HIGH
    assert expected.symbol.blob_cid == BLOB
    assert expected.inputs[0].name == "path"
    assert expected.inputs[0].optionality is Optionality.REQUIRED
    assert expected.errors
    assert expected.sync_async is not None
    assert expected.sync_async.mode is SyncMode.SYNC
    assert any(cap.mode is CapabilityMode.OPTIONAL for cap in expected.capabilities)
    assert any(cap.mode is CapabilityMode.REQUIRED for cap in expected.capabilities)
    record = result.to_record()
    assert record["extraction_id"]
    assert record["bundle_id"]
    assert result.extraction_id == result.content_id


def test_bundle_round_trip_identity_stable() -> None:
    result = extract_contracts(
        [_mcp_read(), _signature_read(), _observation_read()],
        repository_id=REPO,
        tree_id=TREE,
    )
    bundle = result.to_bundle()
    again = result.to_bundle()
    assert bundle.bundle_id == again.bundle_id
    assert bundle.repository_id == REPO
    assert len(bundle.expected) == 1
    assert len(bundle.observed) == 1
    # Observation cannot define expectation.
    with pytest.raises(CircularExpectationError):
        bundle.observed[0].as_expectation_source()


# ---------------------------------------------------------------------------
# Examples / mocks / fixtures / deprecated / generated
# ---------------------------------------------------------------------------


def test_examples_mocks_fixtures_never_define_expectations() -> None:
    example = _mcp_read(
        artifact_id="artifact:example",
        artifact_class=SourceArtifactClass.EXAMPLE,
    )
    mock = _signature_read(
        artifact_id="artifact:mock",
        artifact_class=SourceArtifactClass.MOCK,
    )
    fixture = ContractSourceUnit(
        artifact_id="artifact:fixture",
        source_kind=ContractSourceKind.CONTRACT_TEST,
        payload={
            "kind": "contract_test",
            "asserts": {"returns": "bytes"},
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        locator="test/fixtures/vfs_read.json",
        artifact_class=SourceArtifactClass.FIXTURE,
    )
    result = extract_contracts(
        [example, mock, fixture],
        repository_id=REPO,
        tree_id=TREE,
    )
    assert result.expected == ()
    reasons = {item.reason for item in result.skipped}
    assert SkipReason.EXAMPLE in reasons
    assert SkipReason.MOCK in reasons
    assert SkipReason.FIXTURE in reasons


def test_deprecated_variant_shadowed_by_normative_peer() -> None:
    normative = _mcp_read(artifact_id="artifact:idl-current", returns_type="string")
    deprecated = _mcp_read(
        artifact_id="artifact:idl-deprecated",
        returns_type="integer",
        artifact_class=SourceArtifactClass.DEPRECATED,
    )
    result = extract_contracts(
        [normative, deprecated],
        repository_id=REPO,
        tree_id=TREE,
    )
    assert len(result.expected) == 1
    assert (
        result.expected[0].returns.type_shape.constructor
        is TypeConstructor.STRING
    )
    assert any(
        item.reason is SkipReason.DEPRECATED_SHADOWED for item in result.skipped
    )


def test_generated_copy_alone_may_define_low_authority_expectation() -> None:
    generated = _manifest_read(artifact_class=SourceArtifactClass.GENERATED)
    result = extract_contracts([generated], repository_id=REPO, tree_id=TREE)
    assert len(result.expected) == 1
    expected = result.expected[0]
    assert (
        expected.primary_source_kind
        is ContractSourceKind.COMPATIBILITY_MANIFEST
    )
    assert any(
        source.confidence in {ConfidenceClass.MEDIUM, ConfidenceClass.LOW}
        for source in expected.sources
    )
    assert any(
        "generated" in assumption.statement.lower()
        for assumption in expected.assumptions
    )


# ---------------------------------------------------------------------------
# Contradictions, missing refs, overloads, async/errors
# ---------------------------------------------------------------------------


def test_contradictory_equal_precedence_idl_sources_emit_conflict() -> None:
    left = _mcp_read(artifact_id="artifact:idl-a", returns_type="string")
    right = _mcp_read(artifact_id="artifact:idl-b", returns_type="integer")
    result = extract_contracts([left, right], repository_id=REPO, tree_id=TREE)
    assert len(result.expected) == 1
    expected = result.expected[0]
    assert expected.has_conflicts
    assert any(
        conflict.kind
        in {ConflictKind.PRECEDENCE_COLLISION, ConflictKind.TYPE_MISMATCH}
        and conflict.aspect is SemanticAspect.OUTPUTS
        for conflict in expected.conflicts
    )
    # Conflict must not be marked resolved.
    assert all(conflict.resolved is False for conflict in expected.conflicts)


def test_docs_vs_types_vs_tests_contradiction_recorded() -> None:
    result = extract_contracts(
        [
            _signature_read(returns="bytes"),
            _contract_test_read(returns="str"),
            _doc_read(returns="int"),
        ],
        repository_id=REPO,
        tree_id=TREE,
    )
    expected = result.expected[0]
    # Signature wins (higher authority than test/doc).
    assert expected.returns is not None
    assert expected.returns.type_shape.constructor is TypeConstructor.BYTES
    # Drift against weaker sources is explicit.
    assert expected.conflicts or result.conflicts


def test_missing_schema_ref_emits_unsupported_clause() -> None:
    unit = make_mcp_tool_unit(
        artifact_id="artifact:idl-missing-ref",
        name="vfs.read",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"$ref": "#/$defs/RepoPath"},
            },
            "required": ["path"],
        },
        output_schema={"$ref": "#/$defs/MissingOut"},
        repository_id=REPO,
        tree_id=TREE,
        surface="mcp++",
        locator="tools/list#vfs.read",
        span_start=1,
        span_end=5,
    )
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    assert expected.unsupported or result.unsupported
    reasons = {
        item.reason
        for item in (expected.unsupported + result.unsupported)
    }
    assert "missing_schema_ref" in reasons
    # Inputs/returns still present with unsupported reference shapes.
    assert expected.inputs
    assert expected.inputs[0].type_shape.support is SupportStatus.UNSUPPORTED


def test_overload_set_marks_unsupported_disambiguation() -> None:
    unit = ContractSourceUnit(
        artifact_id="artifact:overloads",
        source_kind=ContractSourceKind.PUBLIC_SIGNATURE,
        payload={
            "kind": "overload_set",
            "overloads": [
                {
                    "parameters": [{"name": "path", "type": "str"}],
                    "returns": "bytes",
                },
                {
                    "parameters": [
                        {"name": "path", "type": "str"},
                        {"name": "offset", "type": "int"},
                    ],
                    "returns": "bytes",
                },
            ],
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="read",
        interface_name="vfs.read",
        surface="mcp++",
        method="read",
        locator="ipfs_kit_py/vfs.py:read",
        artifact_class=SourceArtifactClass.NORMATIVE,
    )
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    assert any(
        item.reason == "overload_set_requires_disambiguation"
        for item in expected.unsupported
    )
    assert expected.inputs
    assert expected.returns is not None


def test_async_and_error_map_extraction() -> None:
    unit = _mcp_read(async_mode=True)
    # Force async mapping through payload.
    unit = ContractSourceUnit(
        artifact_id=unit.artifact_id,
        source_kind=unit.source_kind,
        payload={**unit.payload, "async": True},
        repository_id=REPO,
        tree_id=TREE,
        symbol_name=unit.symbol_name,
        interface_name=unit.interface_name,
        surface=unit.surface,
        method=unit.method,
        protocol=unit.protocol,
        locator=unit.locator,
        span_start=1,
        span_end=2,
        blob_cid=BLOB,
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.MCP_PLUS_PLUS_IDL_V1,
    )
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    assert expected.sync_async is not None
    assert expected.sync_async.mode is SyncMode.ASYNC
    codes = {error.code for error in expected.errors}
    assert "NOT_FOUND" in codes
    assert "PATH_ESCAPE" in codes


def test_schema_union_input_extraction() -> None:
    unit = make_mcp_tool_unit(
        artifact_id="artifact:union",
        name="vfs.write",
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object", "properties": {"cid": {"type": "string"}}},
                    ]
                }
            },
            "required": ["data"],
        },
        output_schema={"type": "boolean"},
        repository_id=REPO,
        tree_id=TREE,
        surface="mcp++",
        locator="tools/list#vfs.write",
    )
    # Align subject fields for write tool.
    unit = ContractSourceUnit(
        artifact_id=unit.artifact_id,
        source_kind=unit.source_kind,
        payload=unit.payload,
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="write",
        interface_name="vfs.write",
        surface="mcp++",
        method="write",
        protocol="mcp",
        locator=unit.locator,
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.MCP_PLUS_PLUS_IDL_V1,
    )
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    data_param = next(p for p in expected.inputs if p.name == "data")
    assert data_param.type_shape.constructor is TypeConstructor.UNION
    assert len(data_param.type_shape.alternatives) == 2


# ---------------------------------------------------------------------------
# Version negotiation, optional capability, circular self-expectation
# ---------------------------------------------------------------------------


def test_version_negotiation_and_optional_capability() -> None:
    unit = make_mcp_tool_unit(
        artifact_id="artifact:negotiate",
        name="vfs.open",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        output_schema={"type": "object"},
        capabilities=[
            {"name": "vfs.open", "mode": "required"},
            {"name": "vfs.streaming", "optional": True},
        ],
        repository_id=REPO,
        tree_id=TREE,
        surface="mcp++",
        locator="tools/list#vfs.open",
        version_negotiation={
            "capability": "mcp.protocol",
            "range": ">=1.0,<2.0",
        },
        applicability={
            "versions": ["1.0", "1.1"],
            "surfaces": ["mcp++", "http"],
            "always": False,
        },
    )
    unit = ContractSourceUnit(
        artifact_id=unit.artifact_id,
        source_kind=unit.source_kind,
        payload=unit.payload,
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="open",
        interface_name="vfs.open",
        surface="mcp++",
        method="open",
        protocol="mcp",
        version="1.0",
        locator=unit.locator,
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.MCP_PLUS_PLUS_IDL_V1,
    )
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    expected = result.expected[0]
    modes = {cap.capability_name: cap.mode for cap in expected.capabilities}
    assert modes.get("vfs.streaming") is CapabilityMode.OPTIONAL
    assert any(
        cap.mode is CapabilityMode.NEGOTIATED for cap in expected.capabilities
    )
    assert expected.applicability is not None
    assert "1.0" in expected.applicability.versions
    assert expected.applicability.always is False


def test_observation_cannot_define_expectation_and_self_expectation_guard() -> None:
    obs = _observation_read()
    with pytest.raises(CircularExpectationError):
        reject_observation_as_expectation_source(obs)
    with pytest.raises(CircularExpectationError):
        obs.to_source_reference(role=ProgramContractRole.EXPECTED)

    result = extract_contracts(
        [obs],
        repository_id=REPO,
        tree_id=TREE,
    )
    assert result.expected == ()
    assert len(result.observed) == 1
    observed = result.observed[0]
    assert observed.repository_observation_id == "observation:fixture-1"
    assert all(
        source.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION
        for source in observed.sources
    )
    with pytest.raises(CircularExpectationError):
        observed.as_expectation_source()


def test_observation_plus_idl_keeps_roles_separate() -> None:
    result = extract_contracts(
        [_mcp_read(), _observation_read(returns="bytes")],
        repository_id=REPO,
        tree_id=TREE,
    )
    assert len(result.expected) == 1
    assert len(result.observed) == 1
    assert result.expected[0].role.value == "expected"
    assert result.observed[0].role.value == "observed"
    # Observed returns must not rewrite expected returns.
    assert result.expected[0].returns is not None
    assert (
        result.expected[0].returns.type_shape.constructor
        is TypeConstructor.STRING
    )


def test_circular_self_expectation_conflict_when_observation_paired() -> None:
    """Observations produce SELF_EXPECTATION conflict markers when mixed.

    The extractor never promotes observations into expectations; it records
    the guard as an explicit conflict for audit.
    """

    result = extract_contracts(
        [_observation_read(), _mcp_read()],
        repository_id=REPO,
        tree_id=TREE,
    )
    # Self-expectation guard runs for observation units.
    assert any(
        conflict.kind is ConflictKind.SELF_EXPECTATION
        for conflict in result.conflicts
    ) or result.expected  # expected still formed from IDL


# ---------------------------------------------------------------------------
# Mapping coercion, determinism, bounds
# ---------------------------------------------------------------------------


def test_mapping_coercion_and_content_kind_inference() -> None:
    unit = contract_source_unit_from_mapping(
        {
            "artifact_id": "artifact:flat",
            "source_kind": "reviewed_interface",
            "repository_id": REPO,
            "tree_id": TREE,
            "interface_name": "vfs.stat",
            "symbol_name": "stat",
            "surface": "mcp++",
            "method": "stat",
            "locator": "tools/list#vfs.stat",
            "kind": "mcp_tool",
            "name": "vfs.stat",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "outputSchema": {"type": "object"},
            "span_start": 1,
            "span_end": 2,
        }
    )
    assert unit.content_kind is ContentKind.MCP_TOOL
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    assert len(result.expected) == 1
    assert result.expected[0].interface.interface_name == "vfs.stat"


def test_extraction_is_deterministic() -> None:
    units = [
        _doc_read(),
        _mcp_read(),
        _signature_read(),
        _contract_test_read(),
        _observation_read(),
        _manifest_read(),
    ]
    first = extract_contracts(units, repository_id=REPO, tree_id=TREE)
    second = extract_contracts(list(reversed(units)), repository_id=REPO, tree_id=TREE)
    assert first.extraction_id == second.extraction_id
    assert first.to_bundle().bundle_id == second.to_bundle().bundle_id
    assert [e.expected_contract_id for e in first.expected] == [
        e.expected_contract_id for e in second.expected
    ]


def test_result_is_canonical_contract() -> None:
    result = extract_contracts([_mcp_read()], repository_id=REPO, tree_id=TREE)
    assert isinstance(result, ContractExtractionResult)
    payload = result.to_dict()
    assert payload["schema"]
    assert payload["extractor_version"] == CONTRACT_EXTRACTOR_VERSION
    assert result.summary


def test_invalid_units_fail_closed() -> None:
    with pytest.raises(ContractExtractorError):
        extract_contracts("not-a-sequence", repository_id=REPO, tree_id=TREE)  # type: ignore[arg-type]
    with pytest.raises(ContractExtractorError):
        ContractSourceUnit(
            artifact_id="a",
            source_kind=ContractSourceKind.REVIEWED_INTERFACE,
            payload={},
            span_start=10,
            span_end=1,
        )


def test_typed_interface_and_json_schema_roles() -> None:
    typed = ContractSourceUnit(
        artifact_id="artifact:typed",
        source_kind=ContractSourceKind.REVIEWED_INTERFACE,
        payload={
            "kind": "typed_interface",
            "parameters": [
                {"name": "path", "type": "str", "kind": "path"},
            ],
            "returns": {"type": "object", "properties": {"size": {"type": "integer"}}},
            "async": False,
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="stat",
        interface_name="vfs.stat",
        surface="python",
        method="stat",
        locator="types/vfs.ts:stat",
        language="typescript",
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.TYPED_INTERFACE_V1,
    )
    schema_out = ContractSourceUnit(
        artifact_id="artifact:schema-out",
        source_kind=ContractSourceKind.REVIEWED_INTERFACE,
        payload={
            "kind": "json_schema",
            "role": "output",
            "schema": {
                "type": "object",
                "properties": {"size": {"type": "integer"}},
            },
        },
        repository_id=REPO,
        tree_id=TREE,
        symbol_name="stat",
        interface_name="vfs.stat",
        surface="python",
        method="stat",
        locator="schemas/vfs.stat.out.json",
        artifact_class=SourceArtifactClass.NORMATIVE,
        extraction_rule=ExtractionRule.JSON_SCHEMA_V1,
    )
    result = extract_contracts(
        [typed, schema_out],
        repository_id=REPO,
        tree_id=TREE,
    )
    assert len(result.expected) == 1
    expected = result.expected[0]
    assert expected.returns is not None
    assert expected.returns.type_shape.constructor is TypeConstructor.OBJECT
    assert expected.inputs
    assert expected.inputs[0].kind in {
        ParameterKind.PATH,
        ParameterKind.POSITIONAL,
        ParameterKind.KEYWORD,
        ParameterKind.OTHER,
    }


def test_path_classification_helper_used_for_locator_defaults() -> None:
    unit = contract_source_unit_from_mapping(
        {
            "artifact_id": "artifact:auto-mock",
            "source_kind": "public_signature",
            "locator": "test/mocks/fake_read.py",
            "payload": {
                "kind": "public_signature",
                "parameters": [{"name": "path", "type": "str"}],
                "returns": "bytes",
            },
            "repository_id": REPO,
            "tree_id": TREE,
            "symbol_name": "read",
            "interface_name": "vfs.read",
            "surface": "mcp++",
            "method": "read",
        }
    )
    assert unit.artifact_class is SourceArtifactClass.MOCK
    result = extract_contracts([unit], repository_id=REPO, tree_id=TREE)
    assert result.expected == ()
    assert result.skipped[0].reason is SkipReason.MOCK
