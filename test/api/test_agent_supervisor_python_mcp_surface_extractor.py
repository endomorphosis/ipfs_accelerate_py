"""SCA-042: cold Python MCP surface extraction tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.python_mcp_surface_extractor import (
    PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
    LiveDiscoveryCapability,
    PythonMcpSurfaceError,
    PythonMcpSurfaceExtractor,
    ResolutionState,
    ToolSurfaceKind,
    UnresolvedReason,
    bind_live_tools_list,
    extract_python_mcp_source,
)


_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.analysis."
    "python_mcp_surface_extractor"
)


def test_cold_import_and_extraction_do_not_import_provider_packages(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "danger_provider"
    provider.mkdir()
    (provider / "__init__.py").write_text(
        "raise AssertionError('provider package was imported')\n",
        encoding="utf-8",
    )
    (provider / "server.py").write_text(
        """from unavailable_mcp_dependency import FastMCP
from danger_provider import implementation

mcp = FastMCP("danger")

@mcp.tool()
def echo(value: str) -> str:
    return implementation.echo(value)
""",
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[2]
    code = f"""
import sys
from {_MODULE} import PythonMcpSurfaceExtractor
surface = PythonMcpSurfaceExtractor().extract_package(
    {str(provider)!r}, provider="danger_provider", repository_tree_id="tree-fixture"
)
assert [tool.canonical_name for tool in surface.tools] == ["echo"]
assert "danger_provider" not in sys.modules
assert "unavailable_mcp_dependency" not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_facade_meta_tools_and_domain_tools_remain_distinguishable() -> None:
    source = '''
def require_capability(ctx):
    return True

def store_blob(content: bytes, pin: bool = True) -> dict:
    require_capability("write")
    return backend.add(content, pin=pin)

server.add_tool(store_blob, name="ipfs.add", aliases=["add"])

@server.tool(name="tools_list_tools")
async def list_category_tools(category: str) -> list:
    return manager.list_tools(category)

@server.tool(name="tools.dispatch")
async def dispatch_tool(category: str, tool_name: str, arguments: dict) -> dict:
    require_capability("dispatch")
    return await manager.dispatch(category, tool_name, arguments)
'''
    surface = extract_python_mcp_source(
        source,
        provider="ipfs_kit_py",
        path="ipfs_kit_py/mcp_server/server.py",
        repository_tree_id="tree-kit",
    )

    assert {tool.canonical_name for tool in surface.domain_tools} == {"ipfs.add"}
    assert {tool.canonical_name for tool in surface.facade_tools} == {
        "tools.dispatch",
        "tools.list_tools",
    }
    list_facade = surface.tools_named("tools.list_tools")[0]
    assert list_facade.declared_name == "tools_list_tools"
    domain = surface.tools_named("add")[0]
    assert domain.canonical_name == "ipfs.add"
    assert domain.aliases == ("add",)
    assert domain.kind is ToolSurfaceKind.DOMAIN
    assert surface.tools_named("tools.dispatch")[0].kind is ToolSurfaceKind.FACADE_META


def test_handler_reachability_schema_policy_and_source_spans_are_retained() -> None:
    source = '''def authorize_write(subject: str) -> None:
    pass

async def pin_content(cid: str, recursive: bool = False) -> dict:
    authorize_write(cid)
    return await ipfs.pin.add(cid, recursive=recursive)

registry.register_tool(
    "ipfs.pin",
    pin_content,
    "Pin content",
    {"type": "object", "required": ["cid"]},
)
'''
    surface = extract_python_mcp_source(
        source,
        provider="ipfs_accelerate_py",
        path="mcp/tools.py",
    )

    assert len(surface.tools) == 1
    tool = surface.tools[0]
    assert tool.handler.state is ResolutionState.RESOLVED
    assert tool.handler.symbol == "pin_content"
    assert tool.handler.span is not None
    assert tool.handler.span.start_line == 4
    assert tool.registration_span.start_line == 8
    assert "ipfs.pin.add" in tool.handler.calls
    assert tool.handler.policy_gates == ("authorize_write",)
    assert tool.input_schema["required"] == ["cid"]
    assert tool.schema_state is ResolutionState.RESOLVED
    assert tool.to_dict()["registration_span"]["source_sha256"].startswith("sha256:")
    assert surface.catalog_registration_records()[0]["path"] == "mcp/tools.py"


def test_protocol_handlers_are_distinct_from_registered_tools() -> None:
    source = '''
@server.list_tools()
async def enumerate_tools():
    return registry.schemas()

@server.call_tool()
async def invoke(name: str, arguments: dict):
    return await registry.call(name, arguments)

@server.tool()
async def search(query: str):
    return await index.search(query)
'''
    surface = extract_python_mcp_source(
        source, provider="ipfs_datasets_py", path="mcp_server/server.py"
    )
    kinds = {tool.canonical_name: tool.kind for tool in surface.tools}
    assert kinds == {
        "call_tool": ToolSurfaceKind.INVOCATION_HANDLER,
        "list_tools": ToolSurfaceKind.DISCOVERY_HANDLER,
        "search": ToolSurfaceKind.DOMAIN,
    }

    direct_router = extract_python_mcp_source(
        '''class Server:
    async def route(self, method: str, params: dict):
        if method == "tools/list":
            return {"tools": registry.schemas()}
        if method == "tools/call":
            return await registry.call(params["name"], params["arguments"])
''',
        provider="ipfs_kit_py",
        path="mcp_server/native_server.py",
    )
    assert {
        tool.canonical_name: tool.kind for tool in direct_router.tools
    } == {
        "tools.call": ToolSurfaceKind.INVOCATION_HANDLER,
        "tools.list": ToolSurfaceKind.DISCOVERY_HANDLER,
    }


def test_dynamic_registration_is_unresolved_not_absent() -> None:
    source = '''
from importlib import import_module

def register_discovered(server, mapping):
    for external_name, descriptor in mapping.items():
        module = import_module(descriptor["module"])
        handler = getattr(module, descriptor["function"])
        server.register_tool(external_name, handler, schema=descriptor["schema"])
'''
    surface = extract_python_mcp_source(
        source,
        provider="ipfs_datasets_py",
        path="mcp_server/tools/tool_registration.py",
    )

    assert surface.tools == ()
    reasons = {item.reason for item in surface.unresolved}
    assert UnresolvedReason.DYNAMIC_NAME in reasons
    assert UnresolvedReason.DYNAMIC_DISCOVERY in reasons
    unresolved = next(
        item
        for item in surface.unresolved
        if item.reason is UnresolvedReason.DYNAMIC_NAME
    )
    assert "register_tool" in unresolved.expression
    assert unresolved.span.start_line > 0
    assert surface.to_dict()["unresolved"]

    dynamic_decorator = extract_python_mcp_source(
        """TOOL_NAME = make_name()

@server.tool(name=TOOL_NAME)
def implementation(value: str):
    return value
""",
        provider="ipfs_datasets_py",
        path="mcp_server/dynamic_decorator.py",
    )
    assert dynamic_decorator.tools == ()
    assert dynamic_decorator.unresolved[0].reason is UnresolvedReason.DYNAMIC_NAME


def test_live_tools_list_fixtures_require_and_preserve_capability_binding() -> None:
    capability = LiveDiscoveryCapability(
        capability_id="cap:fixture:tools-list",
        provider="ipfs_kit_py",
        transport="stdio",
        endpoint_identity="fixture:kit-daemon",
        repository_tree_id="tree-kit",
    )
    fixture = {
        "jsonrpc": "2.0",
        "result": {
            "tools": [
                {"name": "ipfs.cat", "inputSchema": {"type": "object"}},
                {"name": "tools.dispatch", "inputSchema": {"type": "object"}},
            ]
        },
    }
    evidence = bind_live_tools_list(fixture, capability=capability)

    assert evidence.capability_id == capability.capability_id
    assert evidence.repository_tree_id == "tree-kit"
    assert evidence.fixture_sha256.startswith("sha256:")
    assert evidence.to_dict()["authority"] == "observation"
    assert [tool["name"] for tool in evidence.tools] == [
        "ipfs.cat",
        "tools.dispatch",
    ]

    with pytest.raises(PythonMcpSurfaceError, match="outside the granted"):
        bind_live_tools_list(
            fixture,
            capability=capability,
            transport="http",
        )
    with pytest.raises(PythonMcpSurfaceError, match="LiveDiscoveryCapability"):
        bind_live_tools_list(fixture, capability=object())  # type: ignore[arg-type]


def test_package_extraction_is_deterministic_bounded_and_source_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.py").write_text(
        '@mcp.tool(name="beta")\ndef handler_b(x: int = 1): return impl.b(x)\n',
        encoding="utf-8",
    )
    (tmp_path / "a.py").write_text(
        '@mcp.tool()\ndef alpha(value: str): return impl.a(value)\n',
        encoding="utf-8",
    )
    extractor = PythonMcpSurfaceExtractor()
    first = extractor.extract_package(
        tmp_path, provider="fixture", repository_tree_id="tree-1"
    )
    second = extractor.extract_package(
        tmp_path,
        provider="fixture",
        repository_tree_id="tree-1",
        paths=["b.py", "a.py"],
    )

    assert first.surface_id == second.surface_id
    assert first.to_dict() == second.to_dict()
    assert [tool.canonical_name for tool in first.tools] == ["alpha", "beta"]
    assert first.to_dict()["interface"] == PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE
    assert "source" not in first.to_dict()["source_files"][0]

    with pytest.raises(PythonMcpSurfaceError, match="max_files"):
        PythonMcpSurfaceExtractor(max_files=1).extract_package(
            tmp_path, provider="fixture"
        )


def test_parse_failure_is_typed_unresolved_evidence() -> None:
    surface = extract_python_mcp_source(
        "@mcp.tool(\ndef broken(:\n",
        provider="fixture",
        path="broken.py",
    )
    assert surface.tools == ()
    assert len(surface.unresolved) == 1
    assert surface.unresolved[0].reason is UnresolvedReason.PARSE_ERROR
