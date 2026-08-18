"""Tests for live served-model discovery exposed to MCP clients."""

import json

import pytest

from ipfs_accelerate_py.mcp_server.server import StandaloneMCP, register_tools
from ipfs_accelerate_py.mcplusplus_module.trio.server import TrioMCPServer
from ipfs_accelerate_py.model_manager import ModelManager


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(
            {
                "data": [
                    {
                        "id": "example/Leanstral",
                        "owned_by": "llamacpp",
                        "meta": {"n_ctx": 8192},
                    }
                ]
            }
        ).encode()


def test_model_manager_discovers_openai_compatible_model(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response())
    manager = ModelManager(storage_path=str(tmp_path / "models.json"), use_database=False)

    models = manager.list_served_models("http://127.0.0.1:8080/v1")

    assert models[0]["id"] == "leanstral_local"
    assert models[0]["logical_model_id"] == "leanstral_local"
    assert models[0]["transport_model_id"] == "example/Leanstral"
    assert models[0]["transport"] == "llamacpp"
    assert models[0]["provider"] == "llamacpp"
    assert models[0]["status"] == "available"
    assert (
        manager.get_served_model("leanstral_local", "http://127.0.0.1:8080/v1")["metadata"]["n_ctx"]
        == 8192
    )


def test_leanstral_alias_does_not_select_unrelated_sole_model(monkeypatch, tmp_path):
    class _OtherResponse(_Response):
        def read(self):
            return json.dumps(
                {
                    "data": [
                        {
                            "id": "example/OtherModel",
                            "owned_by": "llamacpp",
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _OtherResponse(),
    )
    manager = ModelManager(
        storage_path=str(tmp_path / "models.json"),
        use_database=False,
    )

    assert (
        manager.get_served_model(
            "leanstral",
            "http://127.0.0.1:8080/v1",
        )
        is None
    )


def test_leanstral_logical_lookup_fails_closed_when_transport_is_ambiguous(
    monkeypatch,
    tmp_path,
):
    class _DuplicateResponse(_Response):
        def read(self):
            return json.dumps(
                {
                    "data": [
                        {"id": "example/Leanstral-A", "owned_by": "llamacpp"},
                        {"id": "example/Leanstral-B", "owned_by": "llamacpp"},
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _DuplicateResponse(),
    )
    manager = ModelManager(
        storage_path=str(tmp_path / "models.json"),
        use_database=False,
    )

    models = manager.list_served_models("http://127.0.0.1:8080/v1")

    assert len(models) == 2
    assert {model["transport_model_id"] for model in models} == {
        "example/Leanstral-A",
        "example/Leanstral-B",
    }
    assert (
        manager.get_served_model(
            "leanstral_local",
            "http://127.0.0.1:8080/v1",
        )
        is None
    )


@pytest.mark.anyio
async def test_mcp_client_can_list_served_models(monkeypatch):
    monkeypatch.setattr(
        ModelManager,
        "list_served_models",
        lambda self, endpoint_url=None, timeout=2.0: [
            {
                "id": "example/Leanstral",
                "status": "available",
                "endpoint": endpoint_url,
            }
        ],
    )
    mcp = StandaloneMCP("served-model-test")
    register_tools(mcp)
    server = TrioMCPServer()
    server.mcp = mcp

    response = await server._handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "model_list_served",
                "arguments": {"endpoint_url": "http://127.0.0.1:8080/v1"},
            },
        }
    )

    assert response["result"]["count"] == 1
    assert response["result"]["models"][0]["id"] == "example/Leanstral"
