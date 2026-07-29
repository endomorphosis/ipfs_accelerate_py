"""Mixed Python and contract-evidence adapter conformance tests."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis_ast_index import AnalysisASTIndex
from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    ASTBlobRecord,
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import (
    ProgramASTAdapterResult,
    SourceDocument,
    adapt_json_source,
    adapt_markdown_source,
    adapt_program_source,
    adapt_python_source,
    build_program_ast_blob_record,
    build_program_evidence_index,
)


def _facts(result: ProgramASTAdapterResult, kind: str):
    return result.facts_of_kind(kind)


def test_python_uses_canonical_ast_and_extracts_typed_behavior() -> None:
    source = """\
from contextlib import asynccontextmanager as acm
import transport.client as client

@acm
async def session(value: list[str], /, *, limit: int = 2) -> str:
    async with client.open(value) as stream:
        try:
            await stream.send(limit)
        except (OSError, ValueError) as error:
            raise RuntimeError("failed") from error
"""
    result = adapt_python_source(source, path="src/service.py")

    assert result.status == "success"
    assert isinstance(result.ast_record, ASTBlobRecord)
    assert result.ast_record == build_python_ast_blob_record(source)
    assert result.ast_record.qualified_symbols == ("session",)
    assert "from contextlib import asynccontextmanager as acm" in result.ast_record.imports
    assert "session->client.open" in result.ast_record.calls
    assert "session->stream.send" in result.ast_record.calls

    definition = _facts(result, "async_function_definition")[0]
    assert definition.name == "session"
    assert definition.details["async"] is True
    assert "value: list[str]" in definition.details["signature"]
    assert "limit: int=2" in definition.details["signature"]
    assert definition.target.endswith("-> str")
    assert {(item.name, item.target) for item in _facts(result, "annotation")} >= {
        ("value", "list[str]"),
        ("limit", "int"),
        ("return", "str"),
    }
    assert _facts(result, "decorator")[0].target == "acm"
    assert _facts(result, "async_context_manager")[0].target == "client.open"
    assert _facts(result, "await")[0].target == "stream.send"
    assert _facts(result, "exception_handler")[0].target == "(OSError, ValueError)"
    assert _facts(result, "raise")[0].details["cause"] == "error"
    assert all(item.span.line_start > 0 for item in result.facts)


def test_python_import_alias_calls_are_candidates_not_resolved_edges() -> None:
    source = """\
from package.transport import Client as WireClient
import package.registry as registry

def run():
    wire = WireClient()
    return registry.dispatch(wire)
"""
    result = adapt_program_source(source, path="adapter.py")

    imports = {(item.name, item.target) for item in _facts(result, "import")}
    assert imports == {
        ("WireClient", "package.transport.Client"),
        ("registry", "package.registry"),
    }
    calls = {item.name: item for item in _facts(result, "call")}
    assert calls["WireClient"].ambiguous is True
    assert calls["WireClient"].details["import_candidate"] == "package.transport.Client"
    assert calls["registry.dispatch"].ambiguous is True
    assert (
        calls["registry.dispatch"].details["import_candidate"]
        == "package.registry.dispatch"
    )
    assert all(item.relationship == "calls_candidate" for item in calls.values())


def test_python_monkey_patches_remain_explicitly_ambiguous() -> None:
    source = """\
import service

service.Handler.run = replacement
setattr(service.Handler, "close", close_replacement)

class Local:
    def set_value(self, value):
        self.value = value
"""
    result = adapt_python_source(source, path="patches.py")
    patches = _facts(result, "monkey_patch")

    assert {item.details["mechanism"] for item in patches} == {
        "attribute_assignment",
        "setattr",
    }
    assert all(item.ambiguous for item in patches)
    # Ordinary instance mutation is retained by the canonical record as state,
    # but is not mislabeled as a module-level monkey patch.
    assert not any(item.name == "self.value" for item in patches)
    assert any("self.value:assign:value" in item for item in result.ast_record.state_transitions)


def test_python_malformed_input_is_content_bound_and_explicit() -> None:
    result = adapt_python_source(
        "async def broken(:\n    pass\n",
        path="broken.py",
        blob_identity="git:broken",
    )

    assert result.status == "malformed"
    assert result.blob_identity == "git:broken"
    assert result.ast_record is not None
    assert result.ast_record.parse_error.startswith("SyntaxError at line 1")
    assert result.diagnostics[0].code == "python_syntax_error"
    assert result.diagnostics[0].span.line_start == 1
    assert result.facts == ()


def test_python_exact_record_reuse_and_changed_blob_invalidation() -> None:
    source = "def stable(value: int) -> int:\n    return value\n"
    canonical = build_python_ast_blob_record(source, blob_identity="git:abc")

    warm = adapt_python_source(
        source,
        path="renamed.py",
        blob_identity="git:abc",
        previous=canonical,
    )
    assert warm.reused is True
    assert warm.ast_record is canonical

    replay = adapt_python_source(
        source,
        path="other.py",
        blob_identity="git:abc",
        previous=warm,
    )
    assert replay.reused is True
    assert replay.ast_record is canonical
    assert replay.facts == warm.facts

    changed = adapt_python_source(
        source.replace("return value", "return value + 1"),
        path="renamed.py",
        blob_identity="git:def",
        previous=warm,
    )
    assert changed.reused is False
    assert changed.ast_record is not canonical
    assert changed.ast_record.record_id != canonical.record_id


def test_json_schema_refs_duplicate_keys_and_spans_are_retained() -> None:
    source = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:example:request",
  "$defs": {
    "Identifier": {"type": "string"}
  },
  "properties": {
    "id": {"$ref": "#/$defs/Identifier"},
    "remote": {"$ref": "https://example.invalid/common.json#/$defs/Remote"}
  },
  "required": ["id"],
  "required": ["id", "remote"]
}
"""
    result = adapt_json_source(source, path="schemas/request.schema.json")

    assert result.status == "partial"
    assert result.language == "json-schema"
    assert isinstance(result.ast_record, ASTBlobRecord)
    assert result.ast_record.calls == ()
    assert {item.target for item in _facts(result, "schema_reference")} == {
        "#/$defs/Identifier",
        "https://example.invalid/common.json#/$defs/Remote",
    }
    local, external = sorted(
        _facts(result, "schema_reference"), key=lambda item: item.target
    )
    assert local.ambiguous is False
    assert external.ambiguous is True
    assert {item.name for item in _facts(result, "schema_property")} == {
        "id",
        "remote",
    }
    assert {item.name for item in _facts(result, "schema_required")} == {
        "id",
        "remote",
    }
    duplicate = next(
        item for item in result.diagnostics if item.code == "duplicate_json_key"
    )
    assert duplicate.details["key"] == "required"
    assert duplicate.span.line_start == 12
    assert "$ref #/$defs/Identifier" in result.ast_record.imports
    assert result.ast_record.symbol_lines


def test_json_malformed_input_returns_canonical_failed_record() -> None:
    result = adapt_program_source(
        '{"tools": [{"name": "stat",}]}',
        path="mcp.json",
    )

    assert result.status == "malformed"
    assert result.language == "json"
    assert result.ast_record is not None
    assert result.ast_record.parse_error.startswith("json_syntax_error:")
    assert result.diagnostics[0].code == "json_syntax_error"
    assert result.diagnostics[0].span.line_start == 1


def test_mcp_and_generated_manifest_evidence_is_typed() -> None:
    source = """{
  "generatedBy": "mcp-codegen 2",
  "mcpServers": {
    "local": {"command": "server"}
  },
  "tools": [
    {
      "name": "vfs.stat",
      "description": "Stat one path",
      "inputSchema": {"type": "object"}
    }
  ],
  "prompts": [{"name": "repair"}],
  "resources": [{"uri": "ipfs://{cid}"}]
}
"""
    result = adapt_json_source(
        source, path="src/generated/mcp.manifest.json"
    )

    assert result.status == "success"
    assert result.language == "mcp-manifest"
    assert result.generated is True
    assert _facts(result, "generated_manifest")[0].generated is True
    assert _facts(result, "mcp_server")[0].name == "local"
    assert _facts(result, "mcp_tool")[0].name == "vfs.stat"
    assert _facts(result, "mcp_tool")[0].details["has_input_schema"] is True
    assert _facts(result, "mcp_prompt")[0].name == "repair"
    assert _facts(result, "mcp_resource")[0].name == "ipfs://{cid}"
    assert not result.ast_record.calls
    assert any(value.startswith("mcp_tool:") for value in result.ast_record.interfaces)


def test_generated_path_classification_does_not_change_canonical_identity() -> None:
    source = '{"tools":[{"name":"vfs.read","inputSchema":{"type":"object"}}]}'
    ordinary = adapt_json_source(source, path="manifests/tools.json")
    generated = adapt_json_source(
        source,
        path="generated/tools.json",
        previous=ordinary,
    )

    assert ordinary.generated is False
    assert generated.generated is True
    assert _facts(generated, "generated_manifest")
    assert generated.reused is True
    assert generated.ast_record is ordinary.ast_record


def test_markdown_normative_text_is_distinct_from_fenced_examples() -> None:
    source = """\
# VFS Contract

Clients MUST call `vfs.stat` before `vfs.read`.

```python
# This example MAY call `unsafe_read` directly.
unsafe_read(path)
```

## Errors

Implementations SHOULD raise `FileNotFoundError`.
"""
    result = adapt_markdown_source(source, path="docs/vfs.md")

    assert result.status == "success"
    assert [item.name for item in _facts(result, "heading")] == [
        "VFS Contract",
        "Errors",
    ]
    normative_refs = {
        item.name
        for item in _facts(result, "code_reference")
        if item.normative
    }
    example_refs = {
        item.name
        for item in _facts(result, "code_reference")
        if item.details["example"]
    }
    assert normative_refs == {"vfs.stat", "vfs.read", "FileNotFoundError"}
    assert example_refs == {"unsafe_read"}
    assert all(
        item.normative is False for item in _facts(result, "code_fence")
    )
    assert not any("unsafe_read" in item for item in result.ast_record.interfaces)
    assert result.ast_record.calls == ()


def test_unclosed_markdown_fence_is_partial_not_normative() -> None:
    result = adapt_markdown_source(
        "# Example\n\n```python\nvalue MUST call `not_a_contract`\n",
        path="docs/example.md",
    )

    assert result.status == "partial"
    assert result.diagnostics[0].code == "unclosed_markdown_fence"
    assert _facts(result, "normative_statement") == ()
    reference = _facts(result, "code_reference")[0]
    assert reference.normative is False
    assert reference.details["example"] is True


def test_mixed_batch_reuses_canonical_records_in_the_same_index() -> None:
    documents = (
        SourceDocument("src/service.py", "def stat(path: str):\n    return path\n"),
        SourceDocument(
            "schemas/stat.schema.json",
            '{"$schema":"x","properties":{"path":{"type":"string"}}}',
        ),
        SourceDocument("docs/stat.md", "# Stat\nCallers MUST use `stat`.\n"),
        SourceDocument("config/settings.yaml", "enabled: true\n"),
    )
    first = build_program_evidence_index(documents)

    assert isinstance(first.analysis_index, AnalysisASTIndex)
    assert [item.path for item in first.analysis_index.path_records] == [
        "docs/stat.md",
        "schemas/stat.schema.json",
        "src/service.py",
    ]
    assert len(first.results) == 4
    unsupported = first.unsupported_results
    assert len(unsupported) == 1
    assert unsupported[0].path == "config/settings.yaml"
    assert unsupported[0].diagnostics[0].code == "unsupported_language"
    assert unsupported[0].ast_record is None

    second = build_program_evidence_index(documents, previous=first)
    assert second.analysis_index.stats.reused_blob_count == 3
    assert all(
        result.reused
        for result in second.results
        if result.status != "unsupported"
    )
    first_records = {
        item.path: item.ast_record for item in first.analysis_index.path_records
    }
    assert all(
        item.ast_record is first_records[item.path]
        for item in second.analysis_index.path_records
    )


def test_explicit_unsupported_and_record_only_api() -> None:
    result = adapt_program_source(
        "service: enabled\n",
        path="config/service.yaml",
    )

    assert result.status == "unsupported"
    assert result.supported is False
    assert result.record is None
    assert result.language == "unknown"
    assert result.diagnostics[0].details["path"] == "config/service.yaml"
    assert (
        build_program_ast_blob_record(
            "service: enabled\n", path="config/service.yaml"
        )
        is None
    )
