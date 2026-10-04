"""Live-contract tests for the shared, supervisor-owned Leanstral service.

The default suite is hermetic: it exercises the real model-manager parsing and
MCP JSON-RPC dispatch paths while replacing only the outbound HTTP transport.
Set ``HSSL_RUN_LEANSTRAL_LIVE=1`` to additionally probe the pinned localhost
service.  The opt-in probe is bounded and only calls the model-list endpoint; it
never submits a benchmark input or an inference request.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pytest
import trio

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.mcp_server.server import StandaloneMCP, register_tools
from ipfs_accelerate_py.mcplusplus_module.trio.server import TrioMCPServer
from ipfs_accelerate_py.model_manager import ModelManager


PINNED_ENDPOINT = "http://127.0.0.1:8080/v1"
PINNED_ROUTING_PROVIDER = "leanstral_local"
PINNED_TRANSPORT_PROVIDER = "llamacpp"
PINNED_MODEL = "Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4"
PINNED_SERVICE = "leanstral-119b-shared"
PINNED_SERVER_BUILD = "llama.cpp"

# Stable evidence marker for HSSL-G112.  Keeping the exact identity together
# makes drift in any model-manager or MCP assertion reviewable as one contract.
HSSLEV1126C73 = {
    "endpoint": PINNED_ENDPOINT,
    "routing_provider": PINNED_ROUTING_PROVIDER,
    "transport_provider": PINNED_TRANSPORT_PROVIDER,
    "model": PINNED_MODEL,
    "service": PINNED_SERVICE,
    "server_build": PINNED_SERVER_BUILD,
}

_LIVE_ENABLED = os.getenv("HSSL_RUN_LEANSTRAL_LIVE", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _models_payload() -> dict[str, Any]:
    """Return a safe OpenAI-compatible model-list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": PINNED_MODEL,
                "object": "model",
                "owned_by": PINNED_TRANSPORT_PROVIDER,
                "capabilities": ["text-generation", "lean-proof-draft"],
                "meta": {
                    "service_id": PINNED_SERVICE,
                    "server_build": PINNED_SERVER_BUILD,
                    "shared_with": ["leanstral", "symai"],
                },
            }
        ],
    }


class _JSONResponse:
    """Small urllib response double that retains real response semantics."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload, sort_keys=True).encode("utf-8")

    def __enter__(self) -> "_JSONResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _request_url(request: urllib.request.Request | str) -> str:
    return request.full_url if isinstance(request, urllib.request.Request) else request


def _identity_token(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


@pytest.fixture
def manager(tmp_path) -> ModelManager:
    """Create an isolated manager with no database or IPFS writes."""
    return ModelManager(
        storage_path=str(tmp_path / "models.json"),
        use_database=False,
        enable_ipfs=False,
    )


@pytest.fixture
def served_model_transport(
    monkeypatch,
) -> list[tuple[urllib.request.Request | str, float]]:
    """Install a deterministic transport for the exact pinned model endpoint."""
    calls: list[tuple[urllib.request.Request | str, float]] = []

    def _urlopen(
        request: urllib.request.Request | str,
        timeout: float = 0,
        **_kwargs: Any,
    ) -> _JSONResponse:
        calls.append((request, timeout))
        assert _request_url(request) == f"{PINNED_ENDPOINT}/models"
        return _JSONResponse(_models_payload())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return calls


def _assert_pinned_identity(
    model: dict[str, Any],
    *,
    require_service_metadata: bool = True,
) -> None:
    assert model["id"] == PINNED_ROUTING_PROVIDER
    assert model["model_id"] == PINNED_ROUTING_PROVIDER
    assert model["logical_model_id"] == PINNED_ROUTING_PROVIDER
    assert model["transport_model_id"] == PINNED_MODEL
    assert model["provider"] == PINNED_TRANSPORT_PROVIDER
    assert model["transport"] == PINNED_TRANSPORT_PROVIDER
    assert model["provider"] != PINNED_ROUTING_PROVIDER
    assert _identity_token(model["provider"]) == _identity_token(PINNED_SERVER_BUILD)
    assert model["endpoint"] == PINNED_ENDPOINT
    assert model["status"] == "available"
    assert model["served"] is True
    metadata = model["metadata"]
    assert isinstance(metadata, dict)
    if require_service_metadata or "service_id" in metadata:
        assert model["metadata"]["service_id"] == PINNED_SERVICE
    if require_service_metadata or "server_build" in metadata:
        assert model["metadata"]["server_build"] == PINNED_SERVER_BUILD

    serialized = json.dumps(model, sort_keys=True).lower()
    for secret_name in ("api_key", "authorization", "credential", "secret"):
        assert secret_name not in serialized


def test_contract_distinguishes_logical_route_from_llamacpp_transport() -> None:
    """The logical service pin and observed transport are separate identities."""
    assert HSSLEV1126C73 == {
        "endpoint": PINNED_ENDPOINT,
        "routing_provider": PINNED_ROUTING_PROVIDER,
        "transport_provider": PINNED_TRANSPORT_PROVIDER,
        "model": PINNED_MODEL,
        "service": PINNED_SERVICE,
        "server_build": PINNED_SERVER_BUILD,
    }
    assert PINNED_ROUTING_PROVIDER != PINNED_TRANSPORT_PROVIDER
    assert _identity_token(PINNED_TRANSPORT_PROVIDER) == _identity_token(PINNED_SERVER_BUILD)


def test_model_manager_reports_exact_shared_leanstral_identity(
    manager: ModelManager,
    served_model_transport: list[tuple[urllib.request.Request | str, float]],
) -> None:
    """The manager preserves endpoint, model, transport, and service metadata."""
    models = manager.list_served_models(endpoint_url=PINNED_ENDPOINT, timeout=0.4)

    assert len(models) == 1
    _assert_pinned_identity(models[0])
    request, timeout = served_model_transport[0]
    assert isinstance(request, urllib.request.Request)
    assert request.get_method() == "GET"
    assert request.get_header("Accept") == "application/json"
    assert timeout == 0.4


def test_model_manager_exact_lookup_and_alias_retain_served_identity(
    manager: ModelManager,
    served_model_transport: list[tuple[urllib.request.Request | str, float]],
) -> None:
    """Transport and logical lookups resolve to the same explicit identities."""
    exact = manager.get_served_model(
        PINNED_MODEL,
        endpoint_url=PINNED_ENDPOINT,
        timeout=0.35,
    )
    routed = manager.get_served_model(
        PINNED_ROUTING_PROVIDER,
        endpoint_url=PINNED_ENDPOINT,
        timeout=0.35,
    )

    assert exact is not None
    assert routed is not None
    _assert_pinned_identity(exact)
    _assert_pinned_identity(routed)
    assert "requested_alias" not in exact
    assert "requested_alias" not in routed
    assert routed["provider"] == PINNED_TRANSPORT_PROVIDER
    assert len(served_model_transport) == 2


def test_model_manager_uses_configured_endpoint_and_fails_closed_when_unreachable(
    manager: ModelManager,
    monkeypatch,
) -> None:
    """Discovery must remain bounded and must not invent a fallback identity."""
    calls: list[tuple[str, float]] = []

    def _unreachable(
        request: urllib.request.Request | str,
        timeout: float = 0,
        **_kwargs: Any,
    ) -> _JSONResponse:
        calls.append((_request_url(request), timeout))
        raise urllib.error.URLError("service unavailable")

    monkeypatch.setenv("IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS", PINNED_ENDPOINT)
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable)

    assert manager._served_model_endpoints() == [PINNED_ENDPOINT]
    assert manager.list_served_models(timeout=0.2) == []
    assert manager.get_served_model("leanstral", timeout=0.2) is None
    assert calls == [
        (f"{PINNED_ENDPOINT}/models", 0.2),
        (f"{PINNED_ENDPOINT}/models", 0.2),
    ]


def test_mcp_client_lists_and_gets_the_same_pinned_leanstral_identity(
    manager: ModelManager,
    served_model_transport: list[tuple[urllib.request.Request | str, float]],
    monkeypatch,
) -> None:
    """A real MCP JSON-RPC dispatch must expose the manager's exact identity."""
    monkeypatch.setattr(
        model_manager_module,
        "get_default_model_manager",
        lambda: manager,
    )

    mcp = StandaloneMCP("hssl-leanstral-live-contract")
    register_tools(mcp)
    assert {"model_list_served", "model_get_served"} <= set(mcp.tools)

    server = TrioMCPServer(name="hssl-leanstral-live-contract")
    server.mcp = mcp

    async def _call_tools() -> tuple[dict[str, Any], dict[str, Any]]:
        list_response = await server._handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": "leanstral-list",
                "method": "tools/call",
                "params": {
                    "name": "model_list_served",
                    "arguments": {
                        "endpoint_url": PINNED_ENDPOINT,
                        "timeout": 0.3,
                    },
                },
            }
        )
        get_response = await server._handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": "leanstral-get",
                "method": "tools/call",
                "params": {
                    "name": "model_get_served",
                    "arguments": {
                        "model_id": PINNED_ROUTING_PROVIDER,
                        "endpoint_url": PINNED_ENDPOINT,
                        "timeout": 0.3,
                    },
                },
            }
        )
        return list_response, get_response

    # Run explicitly on Trio.  The repository's root pytest configuration uses
    # asyncio auto mode, which otherwise claims native-Trio async tests before
    # pytest-anyio can select the correct backend in a combined suite.
    list_response, get_response = trio.run(_call_tools)

    assert list_response["id"] == "leanstral-list"
    assert "error" not in list_response, list_response
    assert list_response["result"]["status"] == "success"
    assert list_response["result"]["count"] == 1
    _assert_pinned_identity(list_response["result"]["models"][0])

    assert get_response["id"] == "leanstral-get"
    assert "error" not in get_response, get_response
    assert get_response["result"]["status"] == "success"
    _assert_pinned_identity(get_response["result"]["model"])
    assert "requested_alias" not in get_response["result"]["model"]
    assert [timeout for _request, timeout in served_model_transport] == [0.3, 0.3]


@pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason="set HSSL_RUN_LEANSTRAL_LIVE=1 to probe the supervisor-owned service",
)
def test_supervisor_owned_leanstral_model_list_live(tmp_path) -> None:
    """Opt-in, bounded proof that the actual shared endpoint advertises the pin."""
    endpoint = os.getenv("HSSL_LEANSTRAL_ENDPOINT", PINNED_ENDPOINT).rstrip("/")
    assert endpoint == PINNED_ENDPOINT

    manager = ModelManager(
        storage_path=str(tmp_path / "live-models.json"),
        use_database=False,
        enable_ipfs=False,
    )
    model = manager.get_served_model(
        PINNED_ROUTING_PROVIDER,
        endpoint_url=endpoint,
        timeout=2.0,
    )

    assert model is not None
    _assert_pinned_identity(model, require_service_metadata=False)
    assert "requested_alias" not in model
    # llama.cpp does not advertise the supervisor's logical service alias in
    # /v1/models.  That alias and exact service/build pair are lock-bound above;
    # this live read proves that the pinned model is served by the truthful
    # llama.cpp transport without issuing an inference request.
