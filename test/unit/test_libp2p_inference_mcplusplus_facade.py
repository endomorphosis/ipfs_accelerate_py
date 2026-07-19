"""Regression tests for the MCP++ inference p2p facade."""

from __future__ import annotations

import asyncio
from pathlib import Path


def test_inference_facade_no_longer_contains_raw_stream_handlers() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "libp2p_inference.py"
    )
    text = module_path.read_text(encoding="utf-8")

    assert "set_stream_handler" not in text
    assert "new_stream(" not in text
    assert "stream.read(" not in text
    assert "/ipfs-accelerate/inference" not in text


def test_inference_facade_routes_remote_inference_through_mcplusplus(monkeypatch) -> None:
    from ipfs_accelerate_py.libp2p_inference import LibP2PInferenceNode
    from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools

    calls = {}

    async def fake_list_peers(**_kwargs):
        return {"ok": True, "peers": []}

    async def fake_call_tool(**kwargs):
        calls.update(kwargs)
        return {"ok": True, "result": {"text": "ok"}}

    monkeypatch.setattr(native_p2p_tools, "list_peers", fake_list_peers)
    monkeypatch.setattr(native_p2p_tools, "p2p_taskqueue_call_tool", fake_call_tool)

    async def run_case():
        node = LibP2PInferenceNode(
            {
                "peer_id": "local-peer",
                "discovery_interval": 9999.0,
                "remote_peers": [
                    {
                        "peer_id": "remote-peer",
                        "multiaddr": "/ip4/127.0.0.1/tcp/4001/p2p/remote-peer",
                        "capabilities": ["text-generation"],
                        "models": ["gpt2"],
                    }
                ],
            }
        )
        await node.start()
        try:
            return await node.submit_inference_request(
                task="text-generation",
                model="gpt2",
                inputs="Hello",
                parameters={"max_length": 16},
            )
        finally:
            await node.stop()

    response = asyncio.run(run_case())

    assert response.success is True
    assert response.result == {"text": "ok"}
    assert response.peer_id == "remote-peer"
    assert calls["tool_name"] == "inference_run"
    assert calls["remote_peer_id"] == "remote-peer"
    assert calls["remote_multiaddr"] == "/ip4/127.0.0.1/tcp/4001/p2p/remote-peer"
    assert calls["args"]["model"] == "gpt2"
    assert calls["args"]["input_data"] == "Hello"
    assert calls["args"]["task"] == "text-generation"
    assert calls["args"]["max_length"] == 16
