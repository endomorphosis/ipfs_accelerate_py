from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ASTBlobRecord
from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import (
    JAVASCRIPT_ADAPTER_VERSION,
    ProgramASTAdapterResult,
    adapt_ecmascript_source,
    adapt_program_source,
    detect_program_language,
)


def _facts(result: ProgramASTAdapterResult, kind: str):
    return result.facts_of_kind(kind)


def test_detects_javascript_typescript_and_jsx_variants() -> None:
    for path, expected in {
        "service.js": "javascript",
        "service.mjs": "javascript",
        "service.cjs": "javascript",
        "view.jsx": "jsx",
        "service.ts": "typescript",
        "service.mts": "typescript",
        "service.cts": "typescript",
        "view.tsx": "tsx",
    }.items():
        assert detect_program_language(path=path) == expected

    for hint, expected in {
        "js": "javascript",
        "javascript": "javascript",
        "jsx": "jsx",
        "ts": "typescript",
        "typescript": "typescript",
        "tsx": "tsx",
    }.items():
        assert detect_program_language(path="unknown", language=hint) == expected


def test_adapts_swissknife_style_service_connector_evidence() -> None:
    source = """
import { Server as MCPServer, type Request } from "@modelcontextprotocol/sdk/server";
import type { Service } from "./service";

@service("mcp")
export class Connector implements Service {
  async run(request: Request): Promise<void> {
    const client = new MCPServer({ name: "swissknife-mcp" });
    await client?.registerTool("tools/search", async (query: string) => {
      return await import("./worker");
    });
  }
}
"""
    result = adapt_program_source(
        path="src/connectors/swissknife.ts",
        source=source,
        blob_identity="cid-swissknife",
    )

    assert result.status == "success"
    assert result.language == "typescript"
    assert result.parser == JAVASCRIPT_ADAPTER_VERSION
    assert result.ast_record is not None
    assert result.ast_record.blob_identity == "cid-swissknife"
    assert result.ast_record.source_sha256 == result.source_sha256
    assert {"Connector", "Connector.run", "Connector.run.client"} <= set(
        result.ast_record.qualified_symbols
    )

    imports = _facts(result, "import")
    assert any(
        fact.details.get("source") == "@modelcontextprotocol/sdk/server"
        and fact.details.get("imported") == "Server"
        and fact.details.get("local") == "MCPServer"
        for fact in imports
    )
    assert any(
        fact.details.get("source") == "./service"
        and fact.details.get("imported") == "Service"
        and fact.details.get("type_only") is True
        for fact in imports
    )

    assert any(
        fact.name == "run"
        and fact.owner == "Connector"
        and fact.details.get("async") is True
        for fact in _facts(result, "method_definition")
    )
    assert any(
        fact.name == "service" and fact.target == "Connector"
        for fact in _facts(result, "decorator")
    )
    assert any(
        fact.name == "MCPServer"
        and fact.details.get("resolved_name") == "Server"
        for fact in _facts(result, "new_expression")
    )
    assert any(
        fact.name == "client.registerTool"
        and fact.details.get("optional") is True
        and fact.details.get("awaited") is True
        for fact in _facts(result, "call")
    )
    assert any(
        fact.name == "client.registerTool"
        and fact.details.get("callback_kind") == "async_arrow"
        for fact in _facts(result, "callback")
    )
    assert any(
        fact.name == "client.registerTool"
        and fact.details.get("registration") == "tools/search"
        for fact in _facts(result, "registration")
    )
    assert any(
        fact.details.get("source") == "./worker"
        and fact.details.get("awaited") is True
        for fact in _facts(result, "dynamic_import")
    )
    assert any(
        fact.details.get("value") == "tools/search"
        for fact in _facts(result, "string_literal")
    )
    assert any(
        fact.name == "request"
        and fact.target == "Request"
        for fact in _facts(result, "type_annotation")
    )

    for fact in result.facts:
        assert fact.span.line_start >= 1
        assert fact.span.line_end >= fact.span.line_start
        assert fact.span.column_start >= 0
        assert fact.span.column_end >= 0


def test_barrel_reexports_and_import_aliases_are_preserved_without_fake_definitions() -> None:
    source = """
import DefaultConnector, {
  Wire as ServiceWire,
  type Spec,
} from "./wire";
export * from "./alpha";
export * as connectors from "./connectors";
export { Service as ConnectorService, type Options } from "./service";
export type { Protocol as MCPProtocol } from "./protocol";
"""
    result = adapt_ecmascript_source(
        path="src/index.ts",
        source=source,
        blob_identity="cid-barrel",
    )

    assert result.status == "success"
    imports = {
        (
            fact.details.get("source"),
            fact.details.get("imported"),
            fact.details.get("local"),
            fact.details.get("type_only"),
        )
        for fact in _facts(result, "import")
    }
    assert ("./wire", "default", "DefaultConnector", False) in imports
    assert ("./wire", "Wire", "ServiceWire", False) in imports
    assert ("./wire", "Spec", "Spec", True) in imports

    reexports = {
        (
            fact.details.get("source"),
            fact.details.get("imported"),
            fact.details.get("exported"),
            fact.details.get("type_only"),
        )
        for fact in _facts(result, "re_export")
    }
    assert ("./alpha", "*", "*", False) in reexports
    assert ("./connectors", "*", "connectors", False) in reexports
    assert ("./service", "Service", "ConnectorService", False) in reexports
    assert ("./service", "Options", "Options", True) in reexports
    assert ("./protocol", "Protocol", "MCPProtocol", True) in reexports

    definitions = {
        fact.name
        for fact in result.facts
        if fact.kind.endswith("_definition") or fact.kind == "method_definition"
    }
    assert definitions == set()


def test_tsx_elements_are_emitted_without_treating_typescript_generics_as_jsx() -> None:
    tsx_source = """
interface Props { endpoint: string }
export const View = (props: Props) => (
  <ToolPanel endpoint={props.endpoint} mode="mcp/server" />
);
"""
    tsx_result = adapt_ecmascript_source(
        path="src/View.tsx",
        source=tsx_source,
        blob_identity="cid-tsx",
    )

    assert tsx_result.status == "success"
    assert any(fact.name == "Props" for fact in _facts(tsx_result, "interface_definition"))
    assert any(
        fact.name == "View" and fact.details.get("async") is False
        for fact in _facts(tsx_result, "arrow_function_definition")
    )
    assert any(fact.name == "ToolPanel" for fact in _facts(tsx_result, "jsx_element"))
    assert any(
        fact.details.get("value") == "mcp/server"
        for fact in _facts(tsx_result, "string_literal")
    )

    ts_result = adapt_ecmascript_source(
        path="src/generic.ts",
        source="const result = generic<Type>(input);",
        blob_identity="cid-ts",
    )
    assert _facts(ts_result, "jsx_element") == ()


def test_callback_registration_optional_async_call_and_new_expression_in_javascript() -> None:
    source = """
const client = new ServiceClient();
server.setRequestHandler("tools/call", async (request) => {
  return await dispatcher?.run(request);
});
"""
    result = adapt_ecmascript_source(
        path="src/connector.js",
        source=source,
        blob_identity="cid-js-callback",
    )

    assert result.status == "success"
    assert any(fact.name == "ServiceClient" for fact in _facts(result, "new_expression"))
    assert any(
        fact.name == "server.setRequestHandler"
        and fact.details.get("callback_kind") == "async_arrow"
        for fact in _facts(result, "callback")
    )
    assert any(
        fact.details.get("registration") == "tools/call"
        for fact in _facts(result, "registration")
    )
    assert any(
        fact.name == "dispatcher.run"
        and fact.details.get("optional") is True
        and fact.details.get("awaited") is True
        for fact in _facts(result, "call")
    )


def test_name_collisions_and_unsupported_nodes_remain_explicit() -> None:
    source = """
function duplicate() {}
function duplicate() {}
with (legacy) { debugger; }
"""
    result = adapt_ecmascript_source(
        path="src/legacy.js",
        source=source,
        blob_identity="cid-collision",
    )

    assert result.status == "success"
    duplicate_definitions = [
        fact for fact in _facts(result, "function_definition") if fact.name == "duplicate"
    ]
    assert len(duplicate_definitions) == 2
    assert all(fact.ambiguous for fact in duplicate_definitions)
    assert any(
        diagnostic.code == "ecmascript_name_collision"
        for diagnostic in result.diagnostics
    )
    assert {fact.name for fact in _facts(result, "unsupported_node")} == {
        "debugger",
        "with",
    }
    assert any(
        symbol.startswith("duplicate#")
        for symbol in result.ast_record.qualified_symbols  # type: ignore[union-attr]
    )


def test_malformed_source_preserves_partial_evidence_and_parser_diagnostics() -> None:
    source = """
export class Broken {
  async run() {
    const value =
    return client?.call("mcp/tools");
  }
}}
"""
    result = adapt_ecmascript_source(
        path="src/broken.ts",
        source=source,
        blob_identity="cid-broken",
    )

    assert result.status == "malformed"
    assert result.parse_error
    assert result.ast_record is not None
    assert result.ast_record.parse_error == result.parse_error
    assert result.parser == JAVASCRIPT_ADAPTER_VERSION
    assert result.diagnostics
    assert all(
        diagnostic.details.get("parser") == JAVASCRIPT_ADAPTER_VERSION
        for diagnostic in result.diagnostics
    )
    assert any(fact.name == "Broken" for fact in _facts(result, "class_definition"))
    assert any(
        fact.name == "run" and fact.owner == "Broken"
        for fact in _facts(result, "method_definition")
    )
    assert any(fact.name == "client.call" for fact in _facts(result, "call"))


def test_exact_reuse_and_changed_blob_or_source_invalidation() -> None:
    source = "export const connector = new MCPConnector();"
    first = adapt_ecmascript_source(
        path="src/connector.ts",
        source=source,
        blob_identity="cid-a",
    )

    exact = adapt_ecmascript_source(
        path="src/renamed.ts",
        source=source,
        blob_identity="cid-a",
        previous=first,
    )
    assert exact.reused is True
    assert exact.ast_record is first.ast_record

    assert isinstance(first.ast_record, ASTBlobRecord)
    record_reuse = adapt_ecmascript_source(
        path="src/connector.ts",
        source=source,
        blob_identity="cid-a",
        previous=first.ast_record,
    )
    assert record_reuse.reused is True
    assert record_reuse.ast_record is first.ast_record

    changed_blob = adapt_ecmascript_source(
        path="src/connector.ts",
        source=source,
        blob_identity="cid-b",
        previous=first,
    )
    assert changed_blob.reused is False
    assert changed_blob.ast_record is not first.ast_record
    assert changed_blob.ast_record is not None
    assert changed_blob.ast_record.blob_identity == "cid-b"

    changed_source = adapt_ecmascript_source(
        path="src/connector.ts",
        source=source + "\nconnector.start();",
        blob_identity="cid-a",
        previous=first,
    )
    assert changed_source.reused is False
    assert changed_source.source_sha256 != first.source_sha256
    assert changed_source.ast_record is not None
    assert changed_source.ast_record.source_sha256 == changed_source.source_sha256


def test_comments_and_literals_do_not_create_phantom_syntax_evidence() -> None:
    source = r'''
/*
export class Phantom {}
import { fake } from "./fake";
*/
const quoted = "function Quoted() {}";
const template = `
export { ghost } from "./ghost";
<GhostPanel />
`;
const protocol = "mcp/server";
export function real() {
  return protocol;
}
'''
    result = adapt_ecmascript_source(
        path="src/real.tsx",
        source=source,
        blob_identity="cid-real",
    )

    assert result.status == "success"
    assert any(
        fact.kind == "function_definition" and fact.name == "real"
        for fact in result.facts
    )
    assert not any(
        fact.kind in {"class_definition", "function_definition"}
        and fact.name in {"Phantom", "Quoted"}
        for fact in result.facts
    )
    assert not any(
        fact.kind in {"import", "export", "reexport"}
        and (
            fact.details.get("source") in {"./fake", "./ghost"}
            or fact.name in {"fake", "ghost"}
        )
        for fact in result.facts
    )
    assert not any(
        fact.kind == "jsx_element" and fact.name == "GhostPanel"
        for fact in result.facts
    )
    assert any(
        fact.kind == "string_literal"
        and fact.details.get("value") == "mcp/server"
        for fact in result.facts
    )
