"""Tests for the local Leanstral route exposed through MCP++."""

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.mcp_server.server import StandaloneMCP, register_tools
from ipfs_accelerate_py.mcp_server.tools.shared_tools.native_shared_tools import (
    generate_text,
)
from ipfs_accelerate_py.mcplusplus_module.trio.server import TrioMCPServer


@pytest.mark.anyio
async def test_generate_text_routes_leanstral_to_llama_cpp(monkeypatch):
    captured = {}

    def _generate(prompt, *, model_name=None, provider=None, **kwargs):
        captured.update(
            prompt=prompt,
            model_name=model_name,
            provider=provider,
            kwargs=kwargs,
        )
        return "OK"

    monkeypatch.setattr(llm_router, "generate_text", _generate)

    result = await generate_text(
        "Reply with exactly OK",
        model="Leanstral",
        max_tokens=8,
        temperature=0.0,
    )

    assert result["status"] == "success"
    assert result["generated_text"] == "OK"
    assert result["provider"] == "llama_cpp"
    assert captured == {
        "prompt": "Reply with exactly OK",
        "model_name": None,
        "provider": "llama_cpp",
        "kwargs": {"max_tokens": 8, "temperature": 0.0},
    }


def test_register_tools_exposes_generate_text_for_mcplusplus():
    mcp = StandaloneMCP(name="test-mcplusplus")

    register_tools(mcp, include_p2p_taskqueue_tools=False)

    assert "generate_text" in mcp.tools
    assert callable(mcp.tools["generate_text"]["function"])


@pytest.mark.anyio
async def test_mcplusplus_jsonrpc_calls_registered_generate_text(monkeypatch):
    monkeypatch.setattr(
        llm_router,
        "generate_text",
        lambda prompt, **_kwargs: f"Leanstral: {prompt}",
    )
    server = TrioMCPServer()
    server.mcp = StandaloneMCP(name="test-mcplusplus")
    register_tools(server.mcp, include_p2p_taskqueue_tools=False)

    response = await server._handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "generate_text",
                "arguments": {
                    "prompt": "OK",
                    "model": "Leanstral",
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
            },
        }
    )

    assert response["id"] == 7
    assert response["result"]["provider"] == "llama_cpp"
    assert response["result"]["generated_text"] == "Leanstral: OK"
