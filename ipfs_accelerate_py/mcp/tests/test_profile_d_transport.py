"""Profile D HTTP and MCP+p2p transport parity for ipfs_accelerate_py."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from starlette.requests import Request

from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import (
    decode_jsonrpc_frame,
    encode_jsonrpc_frame,
)
from ipfs_accelerate_py.p2p_tasks.mcp_p2p import handle_mcp_p2p_stream


class _FakeStream:
    def __init__(self, inbound: bytes) -> None:
        self._inbound = bytearray(inbound)
        self.written = bytearray()
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        if not self._inbound:
            return b""
        actual = len(self._inbound) if size < 0 else min(size, len(self._inbound))
        chunk = bytes(self._inbound[:actual])
        del self._inbound[:actual]
        return chunk

    async def write(self, payload: bytes) -> None:
        self.written.extend(payload)

    async def close(self) -> None:
        self.closed = True


def _responses(raw: bytes) -> list[dict]:
    offset = 0
    responses: list[dict] = []
    while offset < len(raw):
        response, consumed = decode_jsonrpc_frame(raw[offset:])
        responses.append(response)
        offset += consumed
    return responses


def _policy_request(request_id: int = 2) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "mcp++/policy/evaluate",
        "params": {
            "actor": "did:key:accelerate",
            "action": "tools.call",
            "policy": {
                "clauses": [
                    {
                        "clause_type": "permission",
                        "actor": "did:key:accelerate",
                        "action": "tools.call",
                    }
                ]
            },
            "request_zkp_certificate": True,
        },
    }


def test_profile_d_p2p_endpoint_uses_canonical_datasets_evaluator() -> None:
    stream = _FakeStream(
        encode_jsonrpc_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + encode_jsonrpc_frame(_policy_request())
    )

    asyncio.run(handle_mcp_p2p_stream(stream, local_peer_id="accelerate-test"))
    responses = _responses(bytes(stream.written))

    assert stream.closed is True
    assert responses[0]["result"]["capabilities"]["mcpPlusPlusProfiles"] == [
        "mcp++/idl",
        "mcp++/cid-envelope",
        "mcp++/ucan",
        "mcp++/deontic-policy",
        "mcp++/p2p-transport",
        "mcp++/risk-scheduling",
    ]
    assert responses[1]["result"]["decision"] == "allow"
    assert responses[1]["result"]["zkp_certificate"]["status"] == "statement_ready"


def test_profile_d_http_rest_endpoint_uses_canonical_datasets_evaluator() -> None:
    from ipfs_accelerate_py.mcp_server.fastapi_service import create_fastapi_app

    async def call(endpoint):
        body = json.dumps(_policy_request()["params"]).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return await endpoint(
            Request({"type": "http", "method": "POST", "path": "/mcp/policy/evaluate", "headers": []}, receive)
        )

    async def _empty_asgi(*_args):
        return None

    with patch(
        "ipfs_accelerate_py.mcp_server.fastapi_service.create_server",
        return_value=SimpleNamespace(app=_empty_asgi),
    ):
        app = create_fastapi_app()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/mcp/policy/evaluate")
    response = asyncio.run(call(endpoint))

    assert response["decision"] == "allow"
    assert response["formal_logic"]
    assert response["zkp_certificate"]["zero_knowledge"] is False
